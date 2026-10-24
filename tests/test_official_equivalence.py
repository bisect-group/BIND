"""The patched BIND model must equal the official ConditionModel in fp64: predictions, loss and gradients.

Both models read the same real ESM-2 hidden states (computed once on CPU in fp32, then cast to fp64), so the test
checks the model maths and not fp16 cache storage. Layer 0 is the rescaled token embedding in fp64 for both models.
"""

import re
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

import loading
from data import BondType
from emulator_bench import modeling
from emulator_bench.common import HEAD_ORDER, normalize_sequence
from emulator_bench.dataset import collate_bind_samples
from emulator_bench.feature_pipeline import ESM_TOKEN_SCALE, PROTEIN_MODEL_NAME, build_token_lut, graph_cache_item, token_ids
from model_graph import ConditionModel

KI_SPLIT = (
    Path(__file__).resolve().parents[3] / "data/processed/datasets/kinetic_dataset/ki/random_splits_grouped_sequence/train.parquet"
)
LAYERS = [0, 10, 20, 30]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED]
# (rows, micro_batch_size): the first two batches stay one chunk, the third is split into several chunks.
BATCHES = [(6, 32), (5, 32), (9, 4)]
PRED_RTOL, LOSS_RTOL, GRAD_RTOL, GRAD_FLOOR = 1e-10, 1e-10, 1e-9, 1e-6

pytestmark = pytest.mark.skipif(not KI_SPLIT.exists(), reason="needs the Ki train split")


class _Fp64Torch:
    """torch with the explicit float32 dtype of the model code mapped to float64."""

    float32 = torch.float64

    def __getattr__(self, name):
        return getattr(torch, name)


def official_key(key):
    """Our state_dict key -> official key, or None for the official keys we do not have."""
    key = re.sub(r"^pool\.", "pool.pool.", key)
    return re.sub(r"^heads\.(\w+)\.", r"\1_head.", key)


def our_key(key):
    key = re.sub(r"^pool\.pool\.", "pool.", key)
    return re.sub(r"^(\w+)_head\.", r"heads.\1.", key)


def official_graph(smiles):
    graph = loading.get_data(smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True)
    x, a, e = loading.convert(*graph, bonds=BONDS)
    return Data(x=torch.Tensor(x), edge_index=dense_to_sparse(torch.Tensor(a))[0], edge_features=torch.Tensor(e))


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(PROTEIN_MODEL_NAME, local_files_only=True)
        esm = AutoModel.from_pretrained(PROTEIN_MODEL_NAME, local_files_only=True).eval()
    except OSError:
        pytest.skip("ESM-2 650M is not cached")

    frame = pd.read_parquet(KI_SPLIT)
    frame = frame[(frame["sequence"].str.len() <= 90) & frame["smiles"].str.len().between(8, 60)]
    frame = frame.drop_duplicates("sequence").iloc[: sum(rows for rows, _ in BATCHES)]
    table = esm.embeddings.word_embeddings.weight.detach().double()
    lut = build_token_lut(tokenizer)

    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(8)
    mp = pytest.MonkeyPatch()
    try:
        batches, start = [], 0
        for rows, micro in BATCHES:
            part = frame.iloc[start : start + rows]
            start += rows
            sequences = [normalize_sequence(s) for s in part["sequence"]]
            encoded = tokenizer(sequences, padding="longest", return_tensors="pt")
            with torch.no_grad():
                states = [h.double() for h in esm(**encoded, output_hidden_states=True).hidden_states]
            states[0] = table[encoded["input_ids"]] * ESM_TOKEN_SCALE
            targets = torch.tensor(part["log10_value"].to_numpy(), dtype=torch.float32).double()
            samples = [
                {
                    "index": i,
                    "graph": graph_cache_item(smiles),
                    "tokens": token_ids(sequence, lut),
                    "target": float(target),
                    "metadata": {"sequence": sequence},
                }
                for i, (sequence, smiles, target) in enumerate(zip(sequences, part["smiles"], targets))
            ]
            chunks = collate_bind_samples(samples, micro)
            for chunk in chunks:
                rows_idx = chunk.indices
                chunk.flat = torch.stack(
                    [torch.cat([states[layer][r, : len(sequences[r]) + 2] for r in rows_idx]) for layer in LAYERS[1:]]
                )
                chunk.targets = chunk.targets.double()
            official = {
                "graph": Batch.from_data_list([official_graph(s) for s in part["smiles"]]),
                "states": states,
                "mask": encoded["attention_mask"],
                "targets": targets.unsqueeze(1),
            }
            batches.append((official, chunks))
        # The explicit fp32 casts of both models run in fp64, so the maths is compared without rounding.
        mp.setattr(torch.Tensor, "float", lambda self: self.double())
        mp.setattr(modeling, "torch", _Fp64Torch())

        torch.manual_seed(0)
        official_model = ConditionModel()
        ours = modeling.build_model(table)
        ours.esm_table = table
        mapped = {our_key(k): v for k, v in official_model.state_dict().items() if not k.startswith(("classifier_head.", "temperature"))}
        ours.load_state_dict(mapped, strict=True)
        assert {official_key(k) for k in ours.state_dict()} <= set(official_model.state_dict())
        yield official_model, ours, batches
    finally:
        mp.undo()
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


def _loss(outputs, targets):
    """Huber loss (delta 2, as the trainer) summed over the four heads, so every head gets a gradient."""
    huber = nn.HuberLoss(delta=2.0)
    return sum(huber(outputs[head], targets) for head in HEAD_ORDER)


def run_official(model, batch, seed):
    model.zero_grad()
    torch.manual_seed(seed)
    ki, ic50, kd, ec50, _ = model(batch["graph"], batch["states"], batch["mask"])
    outputs = dict(zip(HEAD_ORDER, (ki, ic50, kd, ec50)))
    loss = _loss(outputs, batch["targets"])
    loss.backward()
    grads = {our_key(n): p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return {h: o.detach() for h, o in outputs.items()}, loss.detach(), grads


def run_ours(model, chunks, seed):
    """Micro-batches as the trainer: each chunk loss is weighted by its share of the batch rows."""
    model.zero_grad()
    torch.manual_seed(seed)
    total_rows = sum(len(chunk.indices) for chunk in chunks)
    preds = {h: torch.zeros(total_rows, 1) for h in HEAD_ORDER}
    total = torch.zeros(())
    for chunk in chunks:
        outputs = model(chunk)
        loss = _loss(outputs, chunk.targets)
        (loss * len(chunk.indices) / total_rows).backward()
        total += (loss * len(chunk.indices) / total_rows).detach()
        for h in HEAD_ORDER:
            preds[h][chunk.indices] = outputs[h].detach()
    return preds, total, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


def _compare(official_model, ours, batches, seed):
    worst = {"pred": 0.0, "loss": 0.0, "grad": 0.0}
    worst_grad = ""
    for i, (batch, chunks) in enumerate(batches):
        # Dropout masks only match when the chunk has the shape of the full batch.
        if len(chunks) > 1:
            for model in (official_model, ours):
                for module in model.modules():
                    if isinstance(module, nn.MultiheadAttention):
                        module.dropout = 0.0
        p_o, l_o, g_o = run_official(official_model, batch, seed + i)
        p_f, l_f, g_f = run_ours(ours, chunks, seed + i)
        assert g_o.keys() == g_f.keys()
        worst["pred"] = max(worst["pred"], *(_rel(p_f[h], p_o[h]) for h in HEAD_ORDER))
        worst["loss"] = max(worst["loss"], _rel(l_f, l_o))
        norm = torch.cat([g.flatten() for g in g_o.values()]).norm()
        for n in g_o:
            # Tensors with a gradient near zero get a scale floor.
            err = float((g_f[n] - g_o[n]).norm() / g_o[n].norm().clamp(min=GRAD_FLOOR * norm))
            if err > worst["grad"]:
                worst["grad"], worst_grad = err, n
    return worst, worst_grad


@pytest.mark.parametrize("train_mode", [False, True], ids=["eval", "train_dropout"])
def test_patched_model_matches_official(setup, train_mode, capsys):
    official_model, ours, batches = setup
    official_model.train(train_mode)
    ours.train(train_mode)
    dropouts = [m.dropout for m in official_model.modules() if isinstance(m, nn.MultiheadAttention)]
    try:
        worst, worst_grad = _compare(official_model, ours, batches, 100)
    finally:
        for model in (official_model, ours):
            for module, p in zip((m for m in model.modules() if isinstance(m, nn.MultiheadAttention)), dropouts):
                module.dropout = p
    with capsys.disabled():
        print(f"\n[{'train' if train_mode else 'eval'}] max relative diffs: {worst} (worst grad: {worst_grad})")
    assert worst["pred"] < PRED_RTOL
    assert worst["loss"] < LOSS_RTOL
    assert worst["grad"] < GRAD_RTOL, worst_grad


def test_split_batch_has_several_chunks(setup):
    assert [len(chunks) for _, chunks in setup[2]] == [1, 1, 3]
