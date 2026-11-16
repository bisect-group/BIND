"""fp16 embedding cache and the packed collate against the fp32 / original BIND inputs (CPU, toy data).

Tolerances:
- fp16 has an 11-bit significand, so one rounding has a relative error <= 2**-11 per element
  (plus 2**-25 absolute for subnormal values).
- The model output with fp16-cached states is compared with fp32 states by max absolute difference
  (OUTPUT_ATOL, set from the measured error with a safety factor).
- Collate comparisons are exact: the same values go into the same positions.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

import loading
from data import BondType
from emulator_bench.dataset import collate_bind_samples
from emulator_bench.feature_pipeline import (
    ESM_TOKEN_SCALE,
    graph_cache_item,
    protein_cache_item,
    token_ids,
)
from emulator_bench.modeling import build_model

WIDTH, LAYERS = 1280, 3
FP16_REL = 2.0**-11
OUTPUT_ATOL = 5e-3
SEQUENCES = ["MKV", "ACDEFGHIK", "MKVLA", "ACDW"]
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
BONDS = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
    BondType.NOT_CONNECTED,
]
ID_BY_RESIDUE = {c: 4 + i for i, c in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO")}
PAD_ID = 1


def hidden_states(rng, length):
    """fp32 states of scale ~1 with a few large channels, as in transformer hidden states."""
    states = rng.standard_normal((LAYERS, length, WIDTH)).astype(np.float32)
    states[:, :, :4] *= 50
    return states


def make_samples(rng):
    lut = np.zeros(256, dtype=np.uint8)
    for residue, token_id in ID_BY_RESIDUE.items():
        lut[ord(residue)] = token_id
    return [
        {
            "index": i,
            "graph": graph_cache_item(smiles),
            "tokens": token_ids(sequence, lut),
            "target": float(i) / 2,
            "metadata": {"sequence": sequence},
        }
        for i, (sequence, smiles) in enumerate(zip(SEQUENCES, SMILES))
    ]


def test_fp16_cache_round_trip_is_exact_for_fp16_and_bounded_vs_fp32(tmp_path):
    states = hidden_states(np.random.default_rng(0), 11)
    item = protein_cache_item(list(states))
    path = tmp_path / "protein.npy"
    np.save(path, item)
    loaded = np.load(path, mmap_mode="r")
    assert loaded.dtype == np.float16 and loaded.shape == states.shape
    np.testing.assert_array_equal(loaded, item)
    error = np.abs(loaded.astype(np.float32) - states)
    # Values below the fp16 normal range (6.1e-5) have an absolute error of at most half the subnormal step, 2**-25.
    assert (error <= FP16_REL * np.abs(states) + 2.0**-25).all()
    # Saving the already-rounded values again changes nothing.
    np.testing.assert_array_equal(protein_cache_item(list(loaded.astype(np.float32))), item)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(1)
    samples = make_samples(rng)
    states = [hidden_states(rng, len(s["tokens"])) for s in samples]
    table = rng.standard_normal((33, WIDTH)).astype(np.float32)
    return samples, states, table


def pack(states):
    return torch.from_numpy(np.concatenate(states, axis=1))


def test_model_output_with_fp16_cache_is_within_bound_of_fp32(toy):
    samples, states, table = toy
    torch.manual_seed(0)
    model = build_model(table, cross_attention_dropout=0.0).eval()
    (chunk,) = collate_bind_samples(samples, 8)
    cached = [protein_cache_item(list(s)) for s in states]
    exact, rounded = pack(states), pack(cached)
    with torch.no_grad():
        chunk.flat = exact
        reference = model(chunk)
        chunk.flat = rounded
        with_fp16 = model(chunk)
        chunk.flat = rounded.float()
        with_rounded_fp32 = model(chunk)
    worst = 0.0
    for head, out in reference.items():
        # The model casts fp16 states to fp32, so the fp16 tensor equals its fp32 copy exactly.
        torch.testing.assert_close(with_fp16[head], with_rounded_fp32[head], rtol=0, atol=0)
        worst = max(worst, float((with_fp16[head] - out).abs().max()))
    print(f"\nmax |fp16-cache output - fp32 output|: {worst:.3e} (bound {OUTPUT_ATOL})")
    assert worst < OUTPUT_ATOL
    assert worst > 0


def official_graph(smiles):
    graph = loading.get_data(
        smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True
    )
    x, a, e = loading.convert(*graph, bonds=BONDS)
    return Data(
        x=torch.Tensor(x),
        edge_index=dense_to_sparse(torch.Tensor(a))[0],
        edge_features=torch.Tensor(e),
    )


def test_collate_matches_the_original_padded_batch(toy):
    samples, states, _ = toy
    (chunk,) = collate_bind_samples(samples, 8)
    chunk.flat = pack([protein_cache_item(list(s)) for s in states])

    # Original batch: padded token ids, attention mask, padded states and a PyG batch of float graphs.
    lengths = [len(s["tokens"]) for s in samples]
    max_len = max(lengths)
    ids = torch.full((len(samples), max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros(len(samples), max_len, dtype=torch.long)
    padded = torch.zeros(LAYERS, len(samples), max_len, WIDTH)
    for row, (sample, state) in enumerate(zip(samples, states)):
        ids[row, : lengths[row]] = torch.from_numpy(sample["tokens"]).long()
        mask[row, : lengths[row]] = 1
        padded[:, row, : lengths[row]] = torch.from_numpy(
            state.astype(np.float16).astype(np.float32)
        )
    original = Batch.from_data_list([official_graph(s) for s in SMILES])

    assert (
        chunk.indices == [0, 1, 2, 3] and chunk.sequences == SEQUENCES and chunk.max_len == max_len
    )
    padding_mask = torch.arange(max_len)[None, :] >= chunk.lengths[:, None]
    assert torch.equal(~padding_mask, mask.bool())
    grid = torch.full((len(samples) * max_len,), PAD_ID, dtype=torch.long)
    grid[chunk.token_index] = chunk.tokens.long()
    assert torch.equal(grid.view(len(samples), max_len), ids)
    for layer in range(LAYERS):
        grid_states = torch.zeros(len(samples) * max_len, WIDTH)
        grid_states[chunk.token_index] = chunk.flat[layer].float()
        assert torch.equal(grid_states.view(len(samples), max_len, WIDTH), padded[layer])

    graphs = chunk.graphs
    assert torch.equal(graphs.x.float(), original.x)
    assert torch.equal(graphs.edge_index, original.edge_index)
    assert torch.equal(graphs.edge_features.float(), original.edge_features)
    assert torch.equal(graphs.batch, original.batch)
    assert torch.equal(chunk.targets, torch.tensor([0.0, 0.5, 1.0, 1.5]).unsqueeze(1))


def test_layer_zero_is_the_rescaled_token_embedding(toy):
    samples, _, table = toy
    model = build_model(table)
    (chunk,) = collate_bind_samples(samples, 8)
    expected = torch.from_numpy(table)[chunk.tokens.long()] * ESM_TOKEN_SCALE
    assert torch.equal(model.esm_table[chunk.tokens.long()] * ESM_TOKEN_SCALE, expected)
