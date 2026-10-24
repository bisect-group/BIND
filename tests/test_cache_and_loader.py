import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from functools import partial

from emulator_bench.common import protein_cache_path
from emulator_bench.dataset import CachedBindDataset, GatherLoader, collate_bind_samples
from emulator_bench.feature_pipeline import (
    ProteinEmbeddingStore,
    build_token_lut,
    graph_cache_item,
    load_esm_embedding,
    protein_cache_item,
    save_esm_embedding,
    token_ids,
)
from emulator_bench.modeling import build_model

MODEL, LAYERS, CAP, WIDTH = "toy/esm", [0, 10, 20, 30], 2048, 1280
SEQUENCES = ["MKV", "ACDEFGHIK", "MKVLA", "AC"]
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]


class ToyTokenizer:
    unk_token_id = 3

    def get_vocab(self):
        return {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, **{c: 4 + i for i, c in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO")}}


def test_protein_cache_item_is_fp16_and_rejects_overflow():
    layers = [np.random.rand(5, WIDTH).astype(np.float32) for _ in range(3)]
    item = protein_cache_item(layers)
    assert item.dtype == np.float16 and item.shape == (3, 5, WIDTH)
    np.testing.assert_array_equal(item[1], layers[1].astype(np.float16))
    with pytest.raises(ValueError):
        protein_cache_item([np.full((2, WIDTH), 1e6, dtype=np.float32)])


def test_token_ids_and_esm_embedding_round_trip(tmp_path):
    lut = build_token_lut(ToyTokenizer())
    np.testing.assert_array_equal(token_ids("MKV", lut), [0, 20, 15, 7, 2])
    assert token_ids("J", lut).tolist() == [0, 3, 2]
    table = np.random.rand(33, WIDTH).astype(np.float32)
    save_esm_embedding(tmp_path, MODEL, LAYERS, CAP, table, lut)
    loaded_table, loaded_lut = load_esm_embedding(tmp_path, MODEL, LAYERS, CAP)
    np.testing.assert_array_equal(loaded_table, table)
    np.testing.assert_array_equal(loaded_lut, lut)
    with pytest.raises(FileNotFoundError):
        load_esm_embedding(tmp_path / "missing", MODEL, LAYERS, CAP)


@pytest.fixture
def toy_cache(tmp_path):
    rng = np.random.default_rng(0)
    for sequence in SEQUENCES:
        path = protein_cache_path(tmp_path, sequence, MODEL, LAYERS, CAP)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rng.random((3, len(sequence) + 2, WIDTH)).astype(np.float16))
    return tmp_path


def make_loader(cache, micro_batch_size, batch_size):
    import pandas as pd

    class Ligands:
        def get(self, smiles):
            return graph_cache_item(smiles)

    frame = pd.DataFrame({"sequence": SEQUENCES, "smiles": SMILES, "log10_value": [0.0, 1.0, 2.0, 3.0]})
    dataset = CachedBindDataset(frame, Ligands(), build_token_lut(ToyTokenizer()))
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_bind_samples, micro_batch_size=micro_batch_size))
    store = ProteinEmbeddingStore(cache, MODEL, LAYERS, CAP)
    return GatherLoader(loader, store, threads=2, pin=False), store


@pytest.mark.parametrize("micro_batch_size", [4, 2])
def test_packed_residues_match_the_stored_arrays(toy_cache, micro_batch_size):
    loader, store = make_loader(toy_cache, micro_batch_size, batch_size=4)
    (chunks,) = list(loader)
    assert len(chunks) == 4 // micro_batch_size
    seen = []
    for chunk in chunks:
        lengths = chunk.lengths.tolist()
        assert chunk.flat.shape == (3, sum(lengths), WIDTH) and chunk.flat.dtype == torch.float16
        assert chunk.max_len == max(lengths)
        padded = torch.zeros(len(lengths) * chunk.max_len, WIDTH, dtype=torch.float16)
        padded[chunk.token_index] = chunk.flat[1]
        position = 0
        for row, (sequence, length) in enumerate(zip(chunk.sequences, lengths)):
            assert length == len(sequence) + 2
            np.testing.assert_array_equal(chunk.flat[:, position : position + length].numpy(), store.get(sequence))
            np.testing.assert_array_equal(padded.view(len(lengths), chunk.max_len, WIDTH)[row, :length].numpy(), store.get(sequence)[1])
            assert chunk.tokens[position] == 0 and chunk.tokens[position + length - 1] == 2
            position += length
        seen += chunk.indices
    assert sorted(seen) == [0, 1, 2, 3]
    if micro_batch_size < 4:
        assert [len(s) for c in chunks for s in c.sequences] == sorted([len(s) for s in SEQUENCES], reverse=True)


def test_model_rows_do_not_depend_on_chunk_mates(toy_cache):
    torch.manual_seed(0)
    table = np.random.rand(33, 1280).astype(np.float32)
    model = build_model(table, cross_attention_dropout=0.0).eval()
    loader, _ = make_loader(toy_cache, 4, batch_size=4)
    (together,) = list(loader)
    alone_loader, _ = make_loader(toy_cache, 1, batch_size=4)
    (separate,) = list(alone_loader)
    with torch.no_grad():
        joint = model(together[0])["ic50"].squeeze(1)
        singles = {chunk.indices[0]: model(chunk)["ic50"].item() for chunk in separate}
    for row, index in enumerate(together[0].indices):
        assert joint[row].item() == pytest.approx(singles[index], abs=1e-4)
    model(together[0])["ic50"].sum().backward()


def test_cache_config_blocks_mixed_settings(tmp_path):
    from argparse import Namespace

    from emulator_bench.cache_embeddings import _check_cache_config

    args = Namespace(
        embeddings_dir=tmp_path, protein_model_name=MODEL, hidden_state_indices=LAYERS, protein_length_cap=CAP, overwrite=False, num_shards=1
    )
    fingerprint = _check_cache_config(args, "fp32", "rev1")
    assert _check_cache_config(args, "fp32", "rev1") == fingerprint
    with pytest.raises(RuntimeError):
        _check_cache_config(args, "bf16-mixed", "rev1")
    with pytest.raises(RuntimeError):
        _check_cache_config(args, "fp32", "rev2")
    entry = protein_cache_path(tmp_path, "MKV", MODEL, LAYERS, CAP)
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"old")
    args.overwrite = True
    assert _check_cache_config(args, "bf16-mixed", "rev1") != fingerprint
    assert not entry.exists()


def test_entry_validation_checks_the_header_shape(tmp_path):
    from emulator_bench.cache_embeddings import _entry_is_valid

    path = tmp_path / "a.npy"
    np.save(path, np.zeros((3, 5, 8), dtype=np.float16))
    assert _entry_is_valid(path, "MKV", 3, 8)
    assert not _entry_is_valid(path, "MKVL", 3, 8)
    assert not _entry_is_valid(path, "MKV", 2, 8)
    path.write_bytes(b"")
    assert not _entry_is_valid(path, "MKV", 3, 8)
    assert not _entry_is_valid(tmp_path / "missing.npy", "MKV", 3, 8)


def test_chunks_respect_row_and_attention_cell_limits(monkeypatch):
    from emulator_bench import dataset

    monkeypatch.setattr(dataset, "_collate_chunk", lambda chunk: chunk)
    monkeypatch.setattr(dataset, "MAX_ATTENTION_CELLS", 500)

    def sample(residues, atoms):
        return {"tokens": np.zeros(residues, dtype=np.uint8), "graph": {"x": np.zeros((atoms, 15))}}

    rows = [sample(10, 2), sample(40, 2), sample(20, 2), sample(40, 5), sample(10, 2)]
    chunks = dataset.collate_bind_samples(rows, micro_batch_size=3)
    assert [[len(s["tokens"]) for s in c] for c in chunks] == [[40, 40], [20, 10, 10]]


def test_collate_applies_the_cell_cap_to_batches_within_the_row_limit(monkeypatch):
    from emulator_bench import dataset

    monkeypatch.setattr(dataset, "_collate_chunk", lambda chunk: chunk)
    monkeypatch.setattr(dataset, "MAX_ATTENTION_CELLS", 500)

    def sample(residues, atoms):
        return {"tokens": np.zeros(residues, dtype=np.uint8), "graph": {"x": np.zeros((atoms, 15))}}

    rows = [sample(10, 2), sample(40, 2), sample(20, 2)]
    assert dataset.collate_bind_samples(rows, micro_batch_size=3) == [rows]
    assert dataset.collate_bind_samples(rows, micro_batch_size=8) == [rows]
    big = rows + [sample(40, 5)]
    chunks = dataset.collate_bind_samples(big, micro_batch_size=8)
    assert len(chunks) > 1
    assert all(len(c) * len(c[0]["tokens"]) * max(len(s["graph"]["x"]) for s in c) <= 500 for c in chunks)
    assert sorted(id(s) for c in chunks for s in c) == sorted(id(s) for s in big)


def test_gather_loader_keeps_order_with_more_batches_than_threads(toy_cache):
    loader, _ = make_loader(toy_cache, 1, batch_size=1)
    loader.in_flight = 3
    expected = [chunk.indices for chunks in loader.loader for chunk in chunks]
    got = [chunk.indices for chunks in loader for chunk in chunks]
    assert got == expected == [[0], [1], [2], [3]]


def test_gather_loader_surfaces_worker_errors_and_shuts_down_the_pool(toy_cache, monkeypatch):
    from emulator_bench import dataset

    pools = []

    class RecordingPool(dataset.futures.ThreadPoolExecutor):
        def shutdown(self, *args, **kwargs):
            pools.append(kwargs)
            super().shutdown(*args, **kwargs)

    monkeypatch.setattr(dataset.futures, "ThreadPoolExecutor", RecordingPool)
    loader, store = make_loader(toy_cache, 1, batch_size=1)
    real_get = store.get

    def failing_get(sequence):
        if sequence == SEQUENCES[2]:
            raise RuntimeError("bad entry")
        return real_get(sequence)

    store.get = failing_get
    seen = []
    with pytest.raises(RuntimeError, match="bad entry"):
        for chunks in loader:
            seen.append(chunks[0].indices)
    assert seen == [[0], [1]]
    assert pools and pools[0]["wait"] is True


def test_chunked_training_gradients_match_the_unchunked_batch(toy_cache):
    from emulator_bench.train_regression_tvt import train_one_epoch

    class Recorder(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {})
            self.grads = None

        def step(self, closure=None):
            self.grads = [p.grad.clone() for group in self.param_groups for p in group["params"] if p.grad is not None]

    torch.manual_seed(0)
    model = build_model(np.random.rand(33, WIDTH).astype(np.float32), cross_attention_dropout=0.0)
    optimizer = Recorder(model.parameters())

    def grads(micro_batch_size):
        loader, _ = make_loader(toy_cache, micro_batch_size, batch_size=4)
        train_one_epoch(
            model,
            list(loader),
            optimizer,
            torch.nn.HuberLoss(delta=0.5),
            torch.device("cpu"),
            torch.amp.GradScaler("cuda", enabled=False),
            "ic50",
            1,
            1,
        )
        return optimizer.grads

    for whole, chunked in zip(grads(4), grads(3)):
        torch.testing.assert_close(chunked, whole, rtol=1e-3, atol=1e-6)


def test_parallel_ligand_cache_matches_the_serial_graphs(tmp_path):
    from argparse import Namespace

    from emulator_bench.cache_embeddings import AsyncNpzWriter, cache_ligands
    from emulator_bench.common import ligand_cache_path
    from emulator_bench.feature_pipeline import graph_cache_item, load_npz

    smiles = ["CCO", "c1ccccc1O", "CC(=O)O", "[Na+].[Cl-]", "CCN(CC)CC"]
    args = Namespace(embeddings_dir=tmp_path, overwrite=False, num_shards=1, shard_index=0, ligand_workers=3)
    stats = cache_ligands(args, smiles)
    assert stats["ligands_written"] + stats["ligands_failed"] == len(smiles)
    for value in smiles:
        path = ligand_cache_path(tmp_path, value)
        if not path.exists():
            continue
        stored, expected = load_npz(path), graph_cache_item(value)
        assert stored.keys() == expected.keys()
        assert all(np.array_equal(stored[key], expected[key]) for key in expected)
    assert not list(tmp_path.rglob("*.tmp"))
    assert cache_ligands(args, smiles)["ligands_written"] == 0

    layers = [np.random.rand(7, 4).astype(np.float32) for _ in range(3)]
    writer = AsyncNpzWriter(max_workers=2)
    writer.submit(tmp_path / "p.npy", layers)
    writer.close()
    assert np.array_equal(np.load(tmp_path / "p.npy"), protein_cache_item(layers))
