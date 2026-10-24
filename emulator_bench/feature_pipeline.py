import os
import zipfile
import zlib
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import loading
from data import BondType
from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_dir, protein_cache_path


PROTEIN_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
# ESM-2 rescales the embeddings of an input without <mask> tokens by 1 - 0.15 * 0.8 (token dropout).
ESM_TOKEN_SCALE = 1 - 0.15 * 0.8
ESM_EMBEDDING_FILE = "esm_embedding.npz"
GRAPH_BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED]


def _autocast_context(device: torch.device, dtype=None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def resolve_amp_dtype(device: torch.device, amp: bool = False):
    """fp32 unless amp is set; with amp pick bf16 on Ampere+ GPUs and fp16 on older CUDA."""
    if not amp or device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed"
    return torch.float16, "fp16-mixed"


def load_esm2_model(device: torch.device, model_name: str = PROTEIN_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    return model, tokenizer


def esm_sequence_limit(model) -> int:
    max_positions = int(getattr(model.config, "max_position_embeddings", 1026))
    return max(1, max_positions - 2)


def esm_sequence_limit_from_pretrained(model_name: str = PROTEIN_MODEL_NAME) -> int:
    config = AutoConfig.from_pretrained(model_name)
    max_positions = int(getattr(config, "max_position_embeddings", 1026))
    return max(1, max_positions - 2)


def build_esm_batches(
    sequences: Sequence[str],
    max_residues: int = 4096,
    max_batch: int = 4,
) -> List[List[str]]:
    ordered = sorted([normalize_sequence(seq) for seq in sequences], key=len, reverse=True)
    batches: List[List[str]] = []
    batch: List[str] = []
    batch_residues = 0

    for sequence in ordered:
        seq_len = len(sequence)
        if batch and (len(batch) >= max_batch or batch_residues + seq_len > max_residues):
            batches.append(batch)
            batch = []
            batch_residues = 0
        batch.append(sequence)
        batch_residues += seq_len

    if batch:
        batches.append(batch)
    return batches


def _extract_layers(hidden_states, layer_indices: Sequence[int], sequence_length: int, batch_index: int) -> List[np.ndarray]:
    # Keep the <cls> and <eos> states, as the original does: sequence_length + 2 rows.
    extracted = []
    for layer_index in layer_indices:
        extracted.append(hidden_states[layer_index][batch_index, : sequence_length + 2].detach().cpu().numpy())
    return extracted


def build_token_lut(tokenizer) -> np.ndarray:
    """Byte value -> ESM-2 token id, so a sequence maps to ids without the tokenizer."""
    lut = np.full(256, tokenizer.unk_token_id, dtype=np.uint8)
    for token, token_id in tokenizer.get_vocab().items():
        if len(token) == 1:
            lut[ord(token)] = token_id
    return lut


def token_ids(sequence: str, lut: np.ndarray) -> np.ndarray:
    """<cls>, residues, <eos> as uint8 ids."""
    residues = lut[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    return np.concatenate([[0], residues, [2]]).astype(np.uint8)


def _esm_forward(
    model,
    tokenizer,
    sequences: Sequence[str],
    layer_indices: Sequence[int],
    device: torch.device,
    autocast_dtype=None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run ESM-2 and return the requested layers. Layer 0 must equal the rescaled token embedding."""
    encoded = tokenizer(list(sequences), padding="longest", truncation=False, return_tensors="pt")
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    with torch.no_grad(), _autocast_context(device, autocast_dtype):
        outputs = model(**encoded, output_hidden_states=True)
    valid = encoded["attention_mask"].bool()
    rebuilt = model.embeddings.word_embeddings.weight[encoded["input_ids"]] * ESM_TOKEN_SCALE
    if not torch.equal(rebuilt[valid], outputs.hidden_states[0][valid]):
        raise RuntimeError("ESM-2 layer 0 differs from the rescaled token embedding.")
    lut = build_token_lut(tokenizer)
    for batch_index, sequence in enumerate(sequences):
        if not np.array_equal(token_ids(normalize_sequence(sequence), lut), encoded["input_ids"][batch_index, : len(sequence) + 2].cpu().numpy()):
            raise RuntimeError("Token ids from the lookup table differ from the tokenizer.")
    embedded = {}
    for batch_index, sequence in enumerate(sequences):
        normalized = normalize_sequence(sequence)
        seq_len = len(normalized)
        item = {"length": np.asarray([seq_len], dtype=np.int32)}
        for hidden_state, layer_index in zip(
            _extract_layers(outputs.hidden_states, layer_indices=layer_indices, sequence_length=seq_len, batch_index=batch_index),
            layer_indices,
        ):
            item[f"layer_{int(layer_index)}"] = np.asarray(hidden_state, dtype=np.float32)
        embedded[normalized] = item
    return embedded


def graph_cache_item(smiles: str) -> Dict[str, np.ndarray]:
    graph = loading.get_data(
        str(smiles).strip(),
        apply_paths=False,
        parse_cis_trans=False,
        unknown_atom_is_dummy=True,
    )
    x, adjacency, edge_features = loading.convert(*graph, bonds=GRAPH_BOND_TYPES)
    x = np.asarray(x, dtype=np.uint8)
    adjacency = np.asarray(adjacency)
    edge_index = np.vstack(np.nonzero(adjacency)).astype(np.int64, copy=False)
    if edge_index.shape[1] == 0:
        raise ValueError(f"Graph has no edges for SMILES: {smiles}")
    edge_features = np.asarray(edge_features, dtype=np.uint8)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": np.asarray([x.shape[0]], dtype=np.int32),
    }


def protein_cache_item(hidden_states: Sequence[np.ndarray]) -> np.ndarray:
    """One fp16 array per protein, shape (cached layers, residues + 2, features), so it can be memory-mapped."""
    with np.errstate(over="ignore"):
        stacked = np.stack(hidden_states).astype(np.float16)
    if not np.isfinite(stacked).all():
        raise ValueError("Hidden states overflow float16.")
    return stacked


def save_esm_embedding(embeddings_dir: Path, model_name: str, layer_indices: Sequence[int], protein_length_cap: int, table: np.ndarray, lut: np.ndarray) -> None:
    path = protein_cache_dir(embeddings_dir, model_name, layer_indices, protein_length_cap) / ESM_EMBEDDING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp_path, table=table.astype(np.float32), lut=lut)
    tmp_path.replace(path)


def load_esm_embedding(embeddings_dir: Path, model_name: str, layer_indices: Sequence[int], protein_length_cap: int):
    """Token embedding table (layer 0 source) and the byte-to-token-id table of the cache."""
    path = protein_cache_dir(embeddings_dir, model_name, layer_indices, protein_length_cap) / ESM_EMBEDDING_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run cache_embeddings.py first.")
    with np.load(path) as data:
        return data["table"], data["lut"]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(
            f"Corrupted cache file: {path}. Rebuild with `cache_embeddings.py --overwrite`."
        ) from exc


class ProteinEmbeddingStore:
    """Memory-maps one .npy per protein (layers 10, 20, 30 in fp16). The OS page cache is shared by all runs."""

    def __init__(self, embeddings_dir: Path, model_name: str, layer_indices: Sequence[int], protein_length_cap: int):
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        self.layer_indices = list(layer_indices)
        self.protein_length_cap = int(protein_length_cap)

    def get(self, sequence: str) -> np.ndarray:
        path = protein_cache_path(self.embeddings_dir, sequence, self.model_name, self.layer_indices, self.protein_length_cap)
        try:
            return np.load(path, mmap_mode="r")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing cached protein embedding: {path}") from None
        except (EOFError, OSError, ValueError) as exc:
            raise RuntimeError(f"Corrupted cache file: {path}. Rebuild with `cache_embeddings.py --overwrite`.") from exc


class LigandGraphStore:
    def __init__(
        self,
        embeddings_dir: Path,
        smiles_values: Optional[Sequence[str]] = None,
        preload: bool = True,
        max_items: int = 4096,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        if preload and smiles_values is not None:
            for smiles in sorted({str(value).strip() for value in smiles_values}):
                path = ligand_cache_path(self.embeddings_dir, smiles)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached ligand graph: {path}")
                self._cache[smiles] = load_npz(path)

    def get(self, smiles: str) -> Dict[str, np.ndarray]:
        smiles = str(smiles).strip()
        if smiles in self._cache:
            self._cache.move_to_end(smiles)
            return self._cache[smiles]
        path = ligand_cache_path(self.embeddings_dir, smiles)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ligand graph: {path}")
        item = load_npz(path)
        self._cache[smiles] = item
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item
