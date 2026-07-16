"""Inference predictor for the VGB framework.

Provides a Predictor class that loads a model from a bundle or checkpoint
and runs inference under no_grad + autocast. Validates input shapes
before executing forward.

Validates: Requirements 12.1, 12.2, 12.3, 12.4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from vsn.contracts.outputs import ModelOutputs
from vsn.core.model import VSNModel
from vsn.formats.bundle import load_bundle
from vsn.io.save_load import load_model
from vgb.runtime.amp import infer_autocast

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    """Error raised during prediction operations.

    Indicates invalid input shapes, failed model loading,
    or inference failures.
    """

    pass


class Predictor:
    """Inference predictor that loads a model and runs forward passes.

    Loads a model from either a bundle directory or a .pt checkpoint,
    then provides a predict() method that runs under no_grad + autocast.

    Args:
        model_path_or_bundle: Path to either:
            - A bundle directory (containing manifest.json)
            - A .pt checkpoint file
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.).
        precision: Precision for inference ('bf16', 'fp16', 'fp32').

    Raises:
        PredictorError: If the model cannot be loaded from the path.

    Example:
        predictor = Predictor("./my_bundle", device="cuda", precision="bf16")
        outputs = predictor.predict(tokens)
    """

    def __init__(
        self,
        model_path_or_bundle: Union[str, Path],
        device: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        self.device = device
        self.precision = precision

        # Determine device type for autocast
        self.device_type = "cuda" if "cuda" in device else "cpu"

        # Load model
        self.model = self._load_model(model_path_or_bundle, device)
        self.model.eval()

        logger.info(
            "Predictor initialized: device=%s, precision=%s, path=%s",
            device,
            precision,
            model_path_or_bundle,
        )

    def _load_model(
        self, path: Union[str, Path], device: str
    ) -> VSNModel:
        """Load model from bundle or checkpoint.

        Args:
            path: Path to bundle directory or .pt checkpoint.
            device: Target device.

        Returns:
            Loaded VSNModel instance.

        Raises:
            PredictorError: If loading fails.
        """
        path = Path(path)

        try:
            if path.is_dir():
                # Bundle directory
                model = load_bundle(path, device=device)
            elif path.is_file() and path.suffix == ".pt":
                # Checkpoint file
                model = load_model(path, device=device)
            else:
                raise PredictorError(
                    f"Cannot determine model format for path: '{path}'. "
                    f"Expected a directory (bundle) or .pt file (checkpoint)."
                )
        except PredictorError:
            raise
        except Exception as e:
            raise PredictorError(
                f"Failed to load model from '{path}': {e}"
            ) from e

        return model

    def predict(self, tokens: Tensor) -> ModelOutputs:
        """Run inference on input tokens.

        Validates input shape, then runs forward under no_grad + autocast.

        Args:
            tokens: Input tensor of shape (batch, num_tokens, d) where d
                matches the model's embedding dimension.

        Returns:
            ModelOutputs with logits, embeddings, states, and metadata.

        Raises:
            PredictorError: If input shape is invalid.
        """
        self._validate_input(tokens)

        # Move input to model device
        tokens = tokens.to(self.device)

        # Run inference under no_grad + autocast
        with infer_autocast(
            device_type=self.device_type,
            precision=self.precision,
        ):
            outputs = self.model(tokens)

        return outputs

    def _validate_input(self, tokens: Tensor) -> None:
        """Validate input tensor shape before forward pass.

        Args:
            tokens: Input tensor to validate.

        Raises:
            PredictorError: If shape is invalid.
        """
        if tokens.ndim != 3:
            raise PredictorError(
                f"Input tokens must be 3-dimensional (batch, num_tokens, d), "
                f"got shape {tuple(tokens.shape)} ({tokens.ndim}D)."
            )

        expected_d = self.model.config.d
        actual_d = tokens.shape[-1]

        if actual_d != expected_d:
            raise PredictorError(
                f"Input embedding dimension mismatch: "
                f"expected d={expected_d}, got {actual_d}. "
                f"Tokens shape: {tuple(tokens.shape)}."
            )

    @property
    def config(self):
        """Access the underlying model configuration."""
        return self.model.config
