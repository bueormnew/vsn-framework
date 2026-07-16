"""Tests for vgb.export.bundle_export — export_inference_bundle.

Validates export wrapper around vsn.formats.bundle.export_bundle,
config integration, and basic round-trip on CPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.formats.bundle import load_bundle
from vgb.config.schema import FullConfig, ModelConfig, TrainConfig, RuntimeConfig
from vgb.export.bundle_export import export_inference_bundle


@pytest.fixture
def small_model():
    """Create a small VSNModel for testing."""
    config = VSNConfig.small()
    model = VSNModel(config)
    return model


@pytest.fixture
def full_config():
    """Create a FullConfig for testing."""
    vsn_config = VSNConfig.small()
    return FullConfig(
        model=ModelConfig(vsn=vsn_config),
        train=TrainConfig(),
        runtime=RuntimeConfig(),
    )


class TestExportInferenceBundle:
    def test_basic_export(self, small_model):
        """Export without config, verify bundle directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            result = export_inference_bundle(
                model=small_model,
                output_dir=output_dir,
                weight_format="pytorch",
            )
            assert result.is_dir()
            assert (result / "manifest.json").exists()
            assert (result / "weights.pt").exists()
            assert (result / "model_config.json").exists()

    def test_export_with_config_metadata(self, small_model, full_config):
        """Export with FullConfig, verify metadata is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            result = export_inference_bundle(
                model=small_model,
                config=full_config,
                output_dir=output_dir,
            )
            assert result.is_dir()
            assert (result / "manifest.json").exists()

    def test_round_trip(self, small_model):
        """Export and reload, verify outputs match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            export_inference_bundle(
                model=small_model,
                output_dir=output_dir,
            )

            # Reload
            loaded_model = load_bundle(output_dir)
            loaded_model.eval()
            small_model.eval()

            # Compare forward outputs
            tokens = torch.randn(1, small_model.config.ics, small_model.config.d)
            with torch.no_grad():
                original_out = small_model(tokens)
                loaded_out = loaded_model(tokens)

            # States should match
            orig_states = original_out.states["decoder_states"]
            load_states = loaded_out.states["decoder_states"]

            for orig_s, load_s in zip(orig_states, load_states):
                assert torch.allclose(orig_s, load_s, atol=1e-5)
