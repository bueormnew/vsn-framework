"""Tests for vgb.inference.predictor — Predictor class.

Validates model loading from bundle/checkpoint, input validation,
and inference execution on CPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.formats.bundle import export_bundle
from vsn.io.save_load import save_model
from vgb.inference.predictor import Predictor, PredictorError


@pytest.fixture
def small_model():
    """Create a small VSNModel for testing."""
    config = VSNConfig.small()
    model = VSNModel(config)
    return model


@pytest.fixture
def bundle_dir(small_model):
    """Export a small model to a temporary bundle directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "test_bundle"
        export_bundle(small_model, bundle_path)
        yield bundle_path


@pytest.fixture
def checkpoint_path(small_model):
    """Save a small model as a .pt checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.pt"
        save_model(small_model, ckpt_path)
        yield ckpt_path


class TestPredictorLoading:
    def test_load_from_bundle(self, bundle_dir):
        predictor = Predictor(bundle_dir, device="cpu", precision="fp32")
        assert predictor.model is not None
        assert predictor.config is not None

    def test_load_from_checkpoint(self, checkpoint_path):
        predictor = Predictor(checkpoint_path, device="cpu", precision="fp32")
        assert predictor.model is not None

    def test_invalid_path_raises_error(self):
        with pytest.raises(PredictorError):
            Predictor("/nonexistent/path", device="cpu")


class TestPredictorInference:
    def test_predict_valid_input(self, bundle_dir):
        predictor = Predictor(bundle_dir, device="cpu", precision="fp32")
        config = predictor.config

        # Create valid input: (batch=1, num_tokens=ics, d)
        tokens = torch.randn(1, config.ics, config.d)
        outputs = predictor.predict(tokens)

        # Should return ModelOutputs
        assert hasattr(outputs, "metadata")
        assert hasattr(outputs, "states")

    def test_predict_invalid_ndim(self, bundle_dir):
        predictor = Predictor(bundle_dir, device="cpu", precision="fp32")

        # 2D input should fail
        tokens = torch.randn(10, 64)
        with pytest.raises(PredictorError, match="3-dimensional"):
            predictor.predict(tokens)

    def test_predict_invalid_d(self, bundle_dir):
        predictor = Predictor(bundle_dir, device="cpu", precision="fp32")

        # Wrong embedding dimension
        tokens = torch.randn(1, 10, 999)
        with pytest.raises(PredictorError, match="dimension mismatch"):
            predictor.predict(tokens)
