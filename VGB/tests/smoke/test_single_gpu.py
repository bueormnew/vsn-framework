"""Smoke test: single GPU training for 5 steps without crash.

Creates a small VSN model, runs 5 training steps on CPU,
and verifies no crash occurs and loss is finite.

Validates: Requirements 13.4
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel


class DecoderStateLoss(nn.Module):
    """Simple loss that sums decoder states — always produces a scalar."""

    def forward(self, model_output, targets=None):
        # model_output is a ModelOutputs; extract decoder states
        decoder_states = model_output.states["decoder_states"]
        # Mean of all decoder states as a pseudo-loss
        loss = sum(s.pow(2).mean() for s in decoder_states)
        return loss


def _make_small_model():
    """Create a minimal VSN model for smoke testing."""
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


class TestSingleGPUTraining:
    """Smoke tests for training on a single device (CPU)."""

    def test_five_steps_no_crash(self):
        """Train 5 steps on CPU, verify no crash and loss is finite."""
        model = _make_small_model()
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = DecoderStateLoss()

        # Create synthetic input: (batch=2, num_tokens=8, d=16)
        # num_tokens must equal ics (8) for InputCache
        batch_input = torch.randn(2, 8, 16)

        losses = []
        for step in range(5):
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = loss_fn(outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            # Loss must be finite
            assert math.isfinite(loss_val), (
                f"Step {step}: loss is not finite ({loss_val})"
            )

        # All 5 steps completed without crash
        assert len(losses) == 5

    def test_backward_produces_gradients(self):
        """Verify backward pass produces non-zero gradients."""
        model = _make_small_model()
        model.train()

        batch_input = torch.randn(2, 8, 16)
        outputs = model(batch_input)

        # Get some parameter to check grad
        # Use decoder states as pseudo-loss
        decoder_states = outputs.states["decoder_states"]
        loss = sum(s.sum() for s in decoder_states)
        loss.backward()

        # At least one parameter should have non-zero grad
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Expected at least one parameter with non-zero gradient"
