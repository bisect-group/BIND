from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from torch import nn

from emulator_bench.common import ligand_cache_path, protein_cache_path
from emulator_bench.train_regression_tvt import _filter_frame, _with_train_mean_fallback

MODEL, LAYERS, CAP = "toy/esm", [0, 10, 20, 30], 2048


@pytest.fixture
def args():
    return Namespace(
        sequence_col="sequence", smiles_col="smiles", target_col="log10_value", protein_model_name=MODEL,
        hidden_state_indices=LAYERS, protein_length_cap=CAP, long_sequence_strategy="direct",
    )


@pytest.fixture
def cache(tmp_path):
    for path in [protein_cache_path(tmp_path, "MKV", MODEL, LAYERS, CAP), ligand_cache_path(tmp_path, "CCO")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return tmp_path


def frame():
    return pd.DataFrame({"id": ["a", "b"], "sequence": ["MKV", "MKV"], "smiles": ["CCO", "[Na+]"], "log10_value": [1.0, 3.0]})


def test_val_row_without_ligand_is_dropped(cache, args):
    kept, no_ligand, info = _filter_frame(frame(), "val", cache, args, 1022)
    assert kept["id"].tolist() == ["a"]
    assert info["rows_no_ligand"] == 1 and info["rows_missing_cache"] == 1


def test_test_row_without_ligand_gets_train_mean(cache, args):
    kept, no_ligand, info = _filter_frame(frame(), "test", cache, args, 1022)
    assert no_ligand["id"].tolist() == ["b"]
    metadata = [{"id": "a", "sequence": "MKV", "smiles": "CCO", "prediction_source": "model"}]
    metadata, y_true, y_pred, metrics = _with_train_mean_fallback(
        metadata, np.array([1.0], dtype=np.float32), np.array([0.5], dtype=np.float32), no_ligand, args, 2.0,
        nn.HuberLoss(delta=2.0), 0.125,
    )
    assert [row["prediction_source"] for row in metadata] == ["model", "train_mean_fallback"]
    assert metadata[1]["id"] == "b" and metadata[1]["smiles"] == "[Na+]"
    np.testing.assert_allclose(y_true, [1.0, 3.0])
    np.testing.assert_allclose(y_pred, [0.5, 2.0])
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.25 + 1.0) / 2))
    assert metrics["loss"] == pytest.approx((0.125 + 0.5) / 2)


def test_fallback_accepts_2d_model_arrays(args):
    no_ligand = pd.DataFrame({"id": ["b", "c"], "sequence": ["MKV", "MKV"], "smiles": ["[Na+]", "[K+]"], "log10_value": [3.0, 1.0]})
    metadata = [{"id": "a", "sequence": "MKV", "smiles": "CCO", "prediction_source": "model"}]
    metadata, y_true, y_pred, metrics = _with_train_mean_fallback(
        metadata, np.array([[1.0]], dtype=np.float32), np.array([[0.5]], dtype=np.float32), no_ligand, args, 2.0,
        nn.HuberLoss(delta=2.0), 0.125,
    )
    assert len(metadata) == len(y_true) == len(y_pred) == 3
    assert y_true.ndim == y_pred.ndim == 1
    np.testing.assert_allclose(y_pred[1:], 2.0)
    assert all(np.isfinite(value) for value in metrics.values())


def test_missing_protein_is_not_a_ligand_fallback(cache, args):
    data = frame()
    data.loc[1, "sequence"] = "AAA"
    _, no_ligand, info = _filter_frame(data, "test", cache, args, 1022)
    assert len(no_ligand) == 0 and info["rows_missing_cache"] == 1
