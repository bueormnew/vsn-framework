"""Tests for AMP helpers (vgb.runtime.amp).

Verifies autodetection, fallback logic, GradScaler creation,
and context managers work correctly on CPU.

Validates: Requirements 10.2
"""

import torch
import pytest

from vgb.runtime.amp import (
    amp_autocast,
    create_grad_scaler,
    eval_autocast,
    get_amp_dtype,
    infer_autocast,
    is_bf16_supported,
    train_autocast,
)


class TestIsBf16Supported:
    """Tests for is_bf16_supported()."""

    def test_returns_bool(self):
        """Should always return a boolean."""
        result = is_bf16_supported()
        assert isinstance(result, bool)

    def test_false_on_cpu_only(self):
        """On a machine without CUDA, bf16 is not supported."""
        if not torch.cuda.is_available():
            assert is_bf16_supported() is False


class TestGetAmpDtype:
    """Tests for get_amp_dtype()."""

    def test_fp32_returns_none(self):
        """fp32 precision means AMP is disabled."""
        assert get_amp_dtype("fp32") is None

    def test_fp16_returns_float16(self):
        """fp16 always returns torch.float16."""
        assert get_amp_dtype("fp16") is torch.float16

    def test_bf16_on_cpu_falls_back_to_fp16(self):
        """On CPU-only, bf16 should fallback to fp16."""
        if not torch.cuda.is_available():
            assert get_amp_dtype("bf16") is torch.float16

    def test_invalid_precision_raises(self):
        """Invalid precision string raises ValueError."""
        with pytest.raises(ValueError, match="precision must be one of"):
            get_amp_dtype("fp64")

    def test_invalid_precision_empty(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="precision must be one of"):
            get_amp_dtype("")


class TestCreateGradScaler:
    """Tests for create_grad_scaler()."""

    def test_fp16_creates_scaler(self):
        """fp16 precision should create a GradScaler."""
        scaler = create_grad_scaler("fp16")
        assert scaler is not None
        assert isinstance(scaler, torch.amp.GradScaler)

    def test_fp32_returns_none(self):
        """fp32 does not need a scaler."""
        assert create_grad_scaler("fp32") is None

    def test_bf16_on_cpu_creates_scaler(self):
        """On CPU (bf16 unsupported), fallback to fp16 triggers scaler."""
        if not torch.cuda.is_available():
            scaler = create_grad_scaler("bf16")
            assert scaler is not None
            assert isinstance(scaler, torch.amp.GradScaler)

    def test_bf16_on_cuda_no_scaler(self):
        """On CUDA with bf16 support, no scaler needed."""
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            assert create_grad_scaler("bf16") is None


class TestAmpAutocast:
    """Tests for amp_autocast context manager."""

    def test_fp32_is_noop(self):
        """fp32 autocast should not change computation dtype."""
        x = torch.randn(4, 4)
        with amp_autocast("cpu", "fp32"):
            y = x @ x.T
        assert y.dtype == torch.float32

    def test_fp16_enables_autocast_on_cpu(self):
        """fp16 autocast on CPU should produce float16 results."""
        x = torch.randn(4, 4)
        with amp_autocast("cpu", "fp16"):
            y = x @ x.T
        # On CPU autocast with fp16, matmul is cast to fp16
        assert y.dtype == torch.float16

    def test_bf16_fallback_on_cpu(self):
        """When bf16 is not supported, falls back to fp16 autocast."""
        if not torch.cuda.is_available():
            x = torch.randn(4, 4)
            with amp_autocast("cpu", "bf16"):
                y = x @ x.T
            # Falls back to fp16 on CPU-only machines
            assert y.dtype == torch.float16


class TestTrainAutocast:
    """Tests for train_autocast context manager."""

    def test_fp16_train_autocast(self):
        """train_autocast with fp16 should autocast operations."""
        x = torch.randn(4, 4)
        with train_autocast("cpu", "fp16"):
            y = x @ x.T
        assert y.dtype == torch.float16

    def test_fp32_train_no_cast(self):
        """train_autocast with fp32 is a no-op."""
        x = torch.randn(4, 4)
        with train_autocast("cpu", "fp32"):
            y = x @ x.T
        assert y.dtype == torch.float32


class TestEvalAutocast:
    """Tests for eval_autocast context manager."""

    def test_eval_disables_grad(self):
        """eval_autocast should disable gradient computation."""
        x = torch.randn(4, 4, requires_grad=True)
        with eval_autocast("cpu", "fp32"):
            y = x @ x.T
            assert not y.requires_grad

    def test_eval_fp16_autocast(self):
        """eval_autocast with fp16 should autocast and disable grad."""
        x = torch.randn(4, 4, requires_grad=True)
        with eval_autocast("cpu", "fp16"):
            y = x @ x.T
            assert y.dtype == torch.float16
            assert not y.requires_grad


class TestInferAutocast:
    """Tests for infer_autocast context manager."""

    def test_infer_uses_inference_mode(self):
        """infer_autocast should use inference_mode (no grad tracking)."""
        x = torch.randn(4, 4)
        with infer_autocast("cpu", "fp32"):
            y = x @ x.T
            assert not y.requires_grad

    def test_infer_fp16_autocast(self):
        """infer_autocast with fp16 should autocast and disable grad."""
        x = torch.randn(4, 4)
        with infer_autocast("cpu", "fp16"):
            y = x @ x.T
            assert y.dtype == torch.float16
            assert not y.requires_grad

    def test_infer_cannot_enable_grad(self):
        """Inside infer_autocast, tensors are in inference_mode."""
        x = torch.randn(4, 4)
        with infer_autocast("cpu", "fp32"):
            y = x * 2.0
            # In inference_mode, tensors don't track operations
            assert not y.requires_grad
