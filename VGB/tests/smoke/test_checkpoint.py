"""Smoke test: save → load → forward produces same output.

Creates a model, runs forward, saves checkpoint, loads it back,
runs same forward, and verifies outputs match.

Validates: Requirements 13.4
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.io.save_load import save_model, load_model


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


class TestCheckpointSmoke:
    """Smoke tests for save/load round-trip."""

    def test_save_load_produces_same_output(self):
        """Save model, load it back, verify forward produces same output."""
        model = _make_small_model()
        model.eval()

        # Fixed input for reproducibility
        torch.manual_seed(42)
        batch_input = torch.randn(2, 8, 16)

        # Run forward on original model
        with torch.no_grad():
            outputs_before = model(batch_input)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            save_model(model, ckpt_path)

            # Load from checkpoint
            loaded_model = load_model(ckpt_path, device="cpu")
            loaded_model.eval()

            # Run forward on loaded model with same input
            with torch.no_grad():
                outputs_after = loaded_model(batch_input)

        # Compare decoder states
        states_before = outputs_before.states["decoder_states"]
        states_after = outputs_after.states["decoder_states"]

        assert len(states_before) == len(states_after)

        for i, (sb, sa) in enumerate(zip(states_before, states_after)):
            assert torch.allclose(sb, sa, atol=1e-6), (
                f"Decoder state {i} differs after load. "
                f"Max diff: {(sb - sa).abs().max().item()}"
            )

    def test_save_does_not_crash(self):
        """Simply saving a model should not crash."""
        model = _make_small_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_save.pt"
            save_model(model, ckpt_path)

            # File should exist and be non-empty
            assert ckpt_path.exists()
            assert ckpt_path.stat().st_size > 0

    def test_load_reconstructs_correct_config(self):
        """Loaded model should have the same config as original."""
        model = _make_small_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            save_model(model, ckpt_path)
            loaded = load_model(ckpt_path, device="cpu")

        assert loaded.config.X_enc == model.config.X_enc
        assert loaded.config.d == model.config.d
        assert loaded.config.Y == model.config.Y
        assert loaded.config.Z == model.config.Z
        assert loaded.config.ics == model.config.ics
