import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; run by verify_equivalence.sh --gpu")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device visible")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
