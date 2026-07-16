"""Smoke test: AMP autocast does not crash on forward pass.

Creates a small VSN model, enables autocast context (fp16 on CPU),
runs forward, and verifies output is finite.

Validates: Requirements 13.4
"""

from __future__ import annotations

import math

import pytest
import torch

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vgb.runtime.amp import amp_autocast, train_autocast, eval_autocast


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


class TestAMPSmoke:
    """Smoke tests for AMP autocast with VSN model."""

    def test_forward_with_fp16_autocast_no_crash(self):
        """Run forward under fp16 autocast on CPU — no crash."""
        model = _make_small_model()
        model.eval()

        batch_input = torch.randn(2, 8, 16)

        with amp_autocast(device_type="cpu", precision="fp16"):
            outputs = model(batch_input)

        # Verify outputs are present and finite
        decoder_states = outputs.states["decoder_states"]
        assert len(decoder_states) > 0
        for state in decoder_states:
            assert torch.isfinite(state).all(), "Output contains non-finite values"

    def test_forward_with_fp32_autocast_no_crash(self):
        """Run forward under fp32 (no-op autocast) — no crash."""
        model = _make_small_model()
        model.eval()

        batch_input = torch.randn(2, 8, 16)

        with amp_autocast(device_type="cpu", precision="fp32"):
            outputs = model(batch_input)

        decoder_states = outputs.states["decoder_states"]
        assert len(decoder_states) > 0
        for state in decoder_states:
            assert torch.isfinite(state).all()

    def test_train_autocast_no_crash(self):
        """Run forward under train_autocast — no crash."""
        model = _make_small_model()
        model.train()

        batch_input = torch.randn(2, 8, 16)

        with train_autocast(device_type="cpu", precision="fp16"):
            outputs = model(batch_input)

        decoder_states = outputs.states["decoder_states"]
        assert len(decoder_states) > 0

    def test_eval_autocast_no_crash(self):
        """Run forward under eval_autocast (no_grad + autocast) — no crash."""
        model = _make_small_model()
        model.eval()

        batch_input = torch.randn(2, 8, 16)

        with eval_autocast(device_type="cpu", precision="fp16"):
            outputs = model(batch_input)

        decoder_states = outputs.states["decoder_states"]
        assert len(decoder_states) > 0
        # Verify no grad was tracked
        for state in decoder_states:
            assert not state.requires_grad
