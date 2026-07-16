"""AMP (Automatic Mixed Precision) helpers for VGB runtime.

Provides autodetection of bf16 support, fallback to fp16 with GradScaler,
and context managers for autocast in train/eval/infer modes.

Validates: Requirements 10.2
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import torch

logger = logging.getLogger(__name__)


def is_bf16_supported() -> bool:
    """Check if the current GPU supports bfloat16.

    Returns True only when CUDA is available and the device
    has native bf16 support (compute capability >= 8.0).

    Returns:
        True if bf16 is supported on the current device, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


def get_amp_dtype(precision: str) -> Optional[torch.dtype]:
    """Return the torch dtype for autocast, or None for fp32 (no AMP).

    Implements autodetection: if 'bf16' is requested but not supported,
    falls back to fp16 with a warning.

    Args:
        precision: One of 'bf16', 'fp16', 'fp32'.

    Returns:
        torch.bfloat16, torch.float16, or None (fp32 = AMP disabled).

    Raises:
        ValueError: If precision is not one of the valid options.
    """
    valid = ("bf16", "fp16", "fp32")
    if precision not in valid:
        raise ValueError(
            f"precision must be one of {valid}, got {precision!r}"
        )

    if precision == "fp32":
        return None

    if precision == "bf16":
        if is_bf16_supported():
            return torch.bfloat16
        else:
            logger.warning(
                "bf16 requested but not supported on this device; "
                "falling back to fp16 with GradScaler."
            )
            return torch.float16

    # precision == "fp16"
    return torch.float16


def create_grad_scaler(precision: str) -> Optional[torch.amp.GradScaler]:
    """Create a GradScaler only when using fp16.

    bf16 does not need loss scaling (sufficient dynamic range).
    fp32 (AMP disabled) does not need it either.
    When bf16 is requested but unsupported, the fallback to fp16
    triggers scaler creation.

    Args:
        precision: One of 'bf16', 'fp16', 'fp32'.

    Returns:
        A GradScaler instance for fp16 mode, or None otherwise.
    """
    amp_dtype = get_amp_dtype(precision)

    if amp_dtype == torch.float16:
        return torch.amp.GradScaler()

    return None


@contextmanager
def amp_autocast(
    device_type: str = "cuda",
    precision: str = "bf16",
) -> Generator[None, None, None]:
    """Context manager for autocast. No-op if precision='fp32'.

    Wraps torch.amp.autocast with the appropriate dtype resolved
    from precision string. Handles the bf16 fallback transparently.

    Args:
        device_type: Device type for autocast ('cuda', 'cpu').
        precision: One of 'bf16', 'fp16', 'fp32'.

    Yields:
        None — use as a context manager around forward passes.

    Example:
        with amp_autocast("cuda", "bf16"):
            output = model(input)
    """
    amp_dtype = get_amp_dtype(precision)

    if amp_dtype is None:
        # fp32 mode — no autocast, just pass through
        yield
    else:
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
            yield


@contextmanager
def train_autocast(
    device_type: str = "cuda",
    precision: str = "bf16",
) -> Generator[None, None, None]:
    """Autocast context for training forward + loss computation.

    Identical to amp_autocast but semantically named for clarity
    in training loops.

    Args:
        device_type: Device type for autocast.
        precision: One of 'bf16', 'fp16', 'fp32'.

    Yields:
        None
    """
    with amp_autocast(device_type=device_type, precision=precision):
        yield


@contextmanager
def eval_autocast(
    device_type: str = "cuda",
    precision: str = "bf16",
) -> Generator[None, None, None]:
    """Autocast context for evaluation (no grad + autocast).

    Combines torch.no_grad() with autocast for efficient eval.

    Args:
        device_type: Device type for autocast.
        precision: One of 'bf16', 'fp16', 'fp32'.

    Yields:
        None
    """
    with torch.no_grad():
        with amp_autocast(device_type=device_type, precision=precision):
            yield


@contextmanager
def infer_autocast(
    device_type: str = "cuda",
    precision: str = "bf16",
) -> Generator[None, None, None]:
    """Autocast context for inference (no grad + autocast + inference_mode).

    Uses torch.inference_mode() for maximum efficiency during serving,
    combined with autocast for reduced precision.

    Args:
        device_type: Device type for autocast.
        precision: One of 'bf16', 'fp16', 'fp32'.

    Yields:
        None
    """
    with torch.inference_mode():
        with amp_autocast(device_type=device_type, precision=precision):
            yield
