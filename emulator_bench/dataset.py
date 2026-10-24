import concurrent.futures as futures
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch, Data

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import LigandGraphStore, ProteinEmbeddingStore, token_ids
from emulator_bench.modeling import GraphPlan, build_graph_plan


# The attention weights take about 1 KB per cell with 4 cross-attention layers (fp16 autocast).
MAX_ATTENTION_CELLS = 8_000_000


@dataclass
class BindBatch:
    """One micro-batch. Residues of all rows are packed without padding.

    tokens, token_index and flat run over the packed residues: token_index is the position of each residue in
    the padded (rows, max_len) grid. flat holds the cached layers (layers, residues, features) in fp16 and is
    filled by GatherLoader."""

    graphs: Batch
    tokens: torch.Tensor
    token_index: torch.Tensor
    lengths: torch.Tensor
    plan: GraphPlan
    targets: torch.Tensor
    max_len: int
    sequences: List[str]
    metadata: List[Dict]
    indices: List[int]
    flat: Optional[torch.Tensor] = None

    def pin_memory(self):
        self.graphs = self.graphs.pin_memory()
        self.tokens, self.token_index, self.lengths, self.targets = (
            t.pin_memory() for t in (self.tokens, self.token_index, self.lengths, self.targets)
        )
        self.plan = self.plan.apply(lambda t: t.pin_memory())
        return self

    def to(self, device: torch.device):
        self.graphs = self.graphs.to(device, non_blocking=True)
        self.tokens, self.token_index, self.lengths, self.targets, self.flat = (
            t.to(device, non_blocking=True) for t in (self.tokens, self.token_index, self.lengths, self.targets, self.flat)
        )
        self.plan = self.plan.apply(lambda t: t.to(device, non_blocking=True))
        return self


class ReplacementSampler(Sampler):
    """Draws len(dataset) rows per epoch with replacement from the global RNG, as the original sampler does."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(random.choices(range(self.size), k=self.size))


class CachedBindDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        ligand_store: LigandGraphStore,
        token_lut: np.ndarray,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: str = "log10_value",
    ):
        self.frame = frame.reset_index(drop=True)
        self.token_lut = token_lut
        self.ligand_store = ligand_store
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        smiles = str(row[self.smiles_col]).strip()
        sequence = normalize_sequence(str(row[self.sequence_col]))
        ligand = self.ligand_store.get(smiles)

        metadata = {}
        for key, value in row.items():
            metadata[key] = _normalize_metadata_value(value)
        metadata["sequence"] = sequence
        metadata["smiles"] = smiles

        return {
            "index": idx,
            "graph": ligand,
            "tokens": token_ids(sequence, self.token_lut),
            "target": float(row[self.target_col]),
            "metadata": metadata,
        }


def _normalize_metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def collate_bind_samples(samples: List[Dict], micro_batch_size: int) -> List[BindBatch]:
    """Split one drawn batch into micro-batches.

    A chunk holds at most `micro_batch_size` rows and MAX_ATTENTION_CELLS (rows x residues x ligand atoms).
    A batch that fits in one chunk keeps the original row order.
    Otherwise rows are sorted by protein length first, so chunk contents stay similar in length.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    cells = len(samples) * max(len(sample["tokens"]) for sample in samples) * max(len(sample["graph"]["x"]) for sample in samples)
    if len(samples) <= micro_batch_size and cells <= MAX_ATTENTION_CELLS:
        return [_collate_chunk(samples)]
    chunks, current, atoms = [], [], 0
    for sample in sorted(samples, key=lambda sample: -len(sample["tokens"])):
        grown = max(atoms, len(sample["graph"]["x"]))
        if current and (len(current) == micro_batch_size or (len(current) + 1) * len(current[0]["tokens"]) * grown > MAX_ATTENTION_CELLS):
            chunks.append(current)
            current, grown = [], len(sample["graph"]["x"])
        current.append(sample)
        atoms = grown
    chunks.append(current)
    return [_collate_chunk(chunk) for chunk in chunks]


def _collate_chunk(samples: List[Dict]) -> BindBatch:
    lengths = np.asarray([len(sample["tokens"]) for sample in samples], dtype=np.int64)
    max_len = int(lengths.max())
    starts = np.cumsum(lengths) - lengths
    token_index = np.repeat(np.arange(len(samples)) * max_len, lengths) + np.arange(lengths.sum()) - np.repeat(starts, lengths)
    graphs = [
        Data(
            x=torch.from_numpy(sample["graph"]["x"]),
            edge_index=torch.from_numpy(sample["graph"]["edge_index"]),
            edge_features=torch.from_numpy(sample["graph"]["edge_features"]),
        )
        for sample in samples
    ]
    return BindBatch(
        graphs=Batch.from_data_list(graphs),
        tokens=torch.from_numpy(np.concatenate([sample["tokens"] for sample in samples])),
        token_index=torch.from_numpy(token_index),
        lengths=torch.from_numpy(lengths),
        plan=build_graph_plan([graph.num_nodes for graph in graphs]),
        targets=torch.tensor([float(sample["target"]) for sample in samples], dtype=torch.float32).unsqueeze(1),
        max_len=max_len,
        sequences=[sample["metadata"]["sequence"] for sample in samples],
        metadata=[sample["metadata"] for sample in samples],
        indices=[sample["index"] for sample in samples],
    )


def _fill_flat(store: ProteinEmbeddingStore, chunk: BindBatch, pin: bool) -> BindBatch:
    arrays = [store.get(sequence) for sequence in chunk.sequences]
    assert [array.shape[1] for array in arrays] == chunk.lengths.tolist(), "Cached array length differs from the token count"
    flat = torch.empty((arrays[0].shape[0], int(chunk.lengths.sum()), arrays[0].shape[2]), dtype=torch.float16, pin_memory=pin)
    target = flat.numpy()
    position = 0
    for array in arrays:
        target[:, position : position + array.shape[1]] = array
        position += array.shape[1]
    chunk.flat = flat
    return chunk


class GatherLoader:
    """Wraps a DataLoader of micro-batch lists. Threads copy the cached residue states of each micro-batch
    into pinned memory (the copy releases the GIL) while the workers prepare the small tensors.
    At most `in_flight` batches are filled or waiting at one time, because each holds up to 0.5 GB of pinned memory."""

    def __init__(self, loader: DataLoader, store: ProteinEmbeddingStore, threads: int = 6, pin: bool = True, in_flight: int = 2):
        self.loader, self.store, self.pin = loader, store, pin
        self.threads, self.in_flight = max(1, int(threads)), max(1, int(in_flight))

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        pool = futures.ThreadPoolExecutor(self.threads)
        pending = deque()
        batches = iter(self.loader)

        def submit() -> None:
            chunks = next(batches, None)
            if chunks is not None:
                pending.append([pool.submit(_fill_flat, self.store, chunk, self.pin) for chunk in chunks])

        try:
            for _ in range(self.in_flight):
                submit()
            while pending:
                chunk_futures = pending.popleft()
                submit()
                yield [future.result() for future in chunk_futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def shutdown(self) -> None:
        iterator = getattr(self.loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator = None
