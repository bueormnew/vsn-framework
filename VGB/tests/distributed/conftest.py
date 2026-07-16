"""Conftest for distributed tests — registers marks and provides skip logic."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config):
    """Register the 'distributed' marker."""
    config.addinivalue_line(
        "markers",
        "distributed: mark test as requiring multi-GPU distributed environment",
    )


# Module-level skip condition: skip all distributed tests when < 2 GPUs available
requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs for distributed tests",
)
