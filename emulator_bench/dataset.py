import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch, Data

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import LigandGraphStore, ProteinEmbeddingStore


@dataclass
class BindBatch:
    graphs: Batch
    protein_layers: List[torch.Tensor]
    attention_mask: torch.Tensor
    targets: torch.Tensor
    metadata: List[Dict]
    indices: List[int]

    def pin_memory(self):
        self.graphs = self.graphs.pin_memory()
        self.protein_layers = [layer.pin_memory() for layer in self.protein_layers]
        self.attention_mask = self.attention_mask.pin_memory()
        self.targets = self.targets.pin_memory()
        return self

    def to(self, device: torch.device):
        self.graphs = self.graphs.to(device, non_blocking=True)
        self.attention_mask = self.attention_mask.to(device, non_blocking=True)
        self.targets = self.targets.to(device, non_blocking=True)
        self.protein_layers = [layer.to(device, non_blocking=True) for layer in self.protein_layers]
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
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandGraphStore,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: str = "log10_value",
    ):
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
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
        protein = self.protein_store.get(sequence)
        ligand = self.ligand_store.get(smiles)

        metadata = {}
        for key, value in row.items():
            metadata[key] = _normalize_metadata_value(value)
        metadata["sequence"] = sequence
        metadata["smiles"] = smiles

        return {
            "index": idx,
            "graph": ligand,
            "protein": protein,
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
    """Split one drawn batch into micro-batches, each padded to its own longest protein.

    A micro-batch size of at least the batch size keeps one chunk in the original row order.
    Otherwise rows are sorted by protein length first, so chunk contents stay similar in length.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if micro_batch_size < len(samples):
        samples = sorted(samples, key=lambda sample: -sample["protein"].shape[1])
    return [_collate_chunk(samples[start : start + micro_batch_size]) for start in range(0, len(samples), micro_batch_size)]


def _collate_chunk(samples: List[Dict]) -> BindBatch:
    proteins = [sample["protein"] for sample in samples]
    lengths = [protein.shape[1] for protein in proteins]
    max_len = max(lengths)
    batch_size = len(samples)
    padded = np.zeros((proteins[0].shape[0], batch_size, max_len, proteins[0].shape[2]), dtype=proteins[0].dtype)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    graphs = []
    for batch_index, (sample, length) in enumerate(zip(samples, lengths)):
        padded[:, batch_index, :length] = sample["protein"]
        graph_item = sample["graph"]
        graphs.append(
            Data(
                x=torch.from_numpy(graph_item["x"]),
                edge_index=torch.from_numpy(graph_item["edge_index"]),
                edge_features=torch.from_numpy(graph_item["edge_features"]),
            )
        )
        attention_mask[batch_index, :length] = True

    return BindBatch(
        graphs=Batch.from_data_list(graphs),
        protein_layers=list(torch.from_numpy(padded).unbind(0)),
        attention_mask=attention_mask,
        targets=torch.tensor([float(sample["target"]) for sample in samples], dtype=torch.float32).unsqueeze(1),
        metadata=[sample["metadata"] for sample in samples],
        indices=[sample["index"] for sample in samples],
    )
