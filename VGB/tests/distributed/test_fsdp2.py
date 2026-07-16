"""Distributed test: FSDP2 training with torchrun (2 GPUs).

This module provides:
1. A training script function (`fsdp2_train_2_steps`) that can be launched
   with torchrun --nproc_per_node=2 for actual multi-GPU training.
2. A pytest contract test that verifies the module imports correctly and
   the training function is callable — without requiring actual GPUs.

Validates: Requirements 13.5
"""

from __future__ import annotations

import math
import os
from typing import Optional

import pytest
import torch
import torch.nn as nn

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vgb.runtime.fsdp import FSDP2Config, apply_fsdp2

from .conftest import requires_multi_gpu


def _make_small_model() -> VSNModel:
    """Create a minimal VSN model for distributed testing."""
    config = VSNConfig(
        X_enc=2,
        X_dec=2,
        Y=2,
        Z=2,
        d=16,
        ics=8,
        Y_H=2,
        Z_H=2,
        d_H=16,
        p_mode="identity",
        Y_dec=2,
        Z_dec=2,
        dgw=2,
        head_type="regression",
    )
    return VSNModel(config)


class DecoderStateLoss(nn.Module):
    """Simple loss that sums decoder states — always produces a scalar."""

    def forward(self, model_output, targets=None):
        decoder_states = model_output.states["decoder_states"]
        loss = sum(s.pow(2).mean() for s in decoder_states)
        return loss


def fsdp2_train_2_steps(
    precision: str = "fp32",
) -> list[float]:
    """Train 2 steps with FSDP2 sharding.

    This function is designed to be launched via:
        torchrun --nproc_per_node=2 -m pytest VGB/tests/distributed/test_fsdp2.py

    Or directly as a script function in a torchrun-launched process.

    Args:
        precision: AMP precision mode for FSDP2 ('bf16', 'fp16', 'fp32').

    Returns:
        List of loss values for each step.

    Raises:
        RuntimeError: If distributed is not initialized.
    """
    # Setup distributed (should already be done by torchrun)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Initialize process group if not already done
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # Create model on GPU
    model = _make_small_model().to(device)

    # Apply FSDP2
    fsdp_config = FSDP2Config(precision=precision, reshard_after_forward=True)
    model = apply_fsdp2(model, fsdp_config)

    # Optimizer MUST be created after sharding
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DecoderStateLoss()

    # Synthetic input: (batch=2, num_tokens=ics, d=16)
    batch_input = torch.randn(2, 8, 16, device=device)

    losses: list[float] = []
    model.train()

    for step in range(2):
        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = loss_fn(outputs)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        assert math.isfinite(loss_val), (
            f"Rank {rank}, step {step}: loss is not finite ({loss_val})"
        )

    # Cleanup
    torch.distributed.destroy_process_group()

    return losses


# ---------------------------------------------------------------------------
# Pytest contract tests (always run — verify API is callable)
# ---------------------------------------------------------------------------


class TestFSDP2Contract:
    """Contract tests verifying the FSDP2 distributed training API is correct."""

    def test_fsdp2_train_function_is_callable(self):
        """Verify fsdp2_train_2_steps function exists and is callable."""
        assert callable(fsdp2_train_2_steps)

    def test_fsdp2_config_is_importable(self):
        """Verify FSDP2Config can be instantiated."""
        config = FSDP2Config(precision="bf16")
        assert config.precision == "bf16"
        assert config.reshard_after_forward is True
        assert config.prefetch_num == 1

    def test_apply_fsdp2_is_importable(self):
        """Verify apply_fsdp2 function is importable and callable."""
        assert callable(apply_fsdp2)

    def test_model_creation_for_fsdp2(self):
        """Verify the model used in FSDP2 tests can be created."""
        model = _make_small_model()
        assert isinstance(model, nn.Module)
        # Verify model has encoder/decoder with vgb_blocks
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")


@pytest.mark.distributed
@requires_multi_gpu
class TestFSDP2Distributed:
    """Actual distributed tests — only run with multi-GPU hardware.

    To run these tests:
        torchrun --nproc_per_node=2 -m pytest VGB/tests/distributed/test_fsdp2.py -k "Distributed"
    """

    def test_fsdp2_train_2_steps_multi_gpu(self):
        """Train 2 steps with FSDP2 on 2 GPUs, verify loss is finite."""
        losses = fsdp2_train_2_steps(precision="fp32")
        assert len(losses) == 2
        for loss_val in losses:
            assert math.isfinite(loss_val)


# ---------------------------------------------------------------------------
# Script entry point for torchrun
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Direct execution via torchrun:
        torchrun --nproc_per_node=2 VGB/tests/distributed/test_fsdp2.py
    """
    losses = fsdp2_train_2_steps(precision="fp32")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {rank}] FSDP2 training complete. Losses: {losses}")
