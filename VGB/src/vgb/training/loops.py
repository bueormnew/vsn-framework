"""Training step loop for the VGB framework.

Provides a single train_step function that executes one full training step:
forward under autocast → loss.backward (scaled if fp16) → unscale →
clip_grad_norm → optimizer step → scaler.update.

Supports gradient accumulation: only step/update every `grad_accum_steps`.

Validates: Requirements 8.3, 8.6
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vgb.config.schema import TrainConfig
from vgb.runtime.amp import train_autocast

logger = logging.getLogger(__name__)


def train_step(
    model: nn.Module,
    batch: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    config: TrainConfig,
    *,
    step: int = 0,
    loss_fn: Optional[nn.Module] = None,
    targets: Optional[Tensor] = None,
) -> Dict[str, float]:
    """Execute one full training step with AMP and gradient accumulation.

    Performs: forward → loss → backward → (every grad_accum_steps:
    unscale → clip → step → zero_grad → scaler.update).

    Args:
        model: The model to train.
        batch: Input tensor (tokens) to feed the model.
        optimizer: The optimizer instance.
        scaler: GradScaler for fp16 AMP. None if bf16 or fp32.
        config: TrainConfig with precision, grad_clip_norm,
            grad_accum_steps settings.
        step: Current global training step (0-indexed). Used to
            determine whether to execute optimizer step based on
            gradient accumulation.
        loss_fn: Optional loss function. If None, model is expected
            to return outputs with a loss attribute or the outputs
            themselves are treated as the loss.
        targets: Optional targets tensor for loss computation.

    Returns:
        Dict with:
            - 'loss': scalar loss value (float)
            - 'grad_norm': gradient norm after clipping (float),
              or 0.0 if this step was an accumulation step without
              optimizer update.
    """
    # Determine device type for autocast
    device_type = "cpu"
    for p in model.parameters():
        if p.device.type == "cuda":
            device_type = "cuda"
        break

    # Forward pass under autocast
    with train_autocast(device_type=device_type, precision=config.precision):
        outputs = model(batch)

        # Compute loss
        if loss_fn is not None and targets is not None:
            # Extract logits from ModelOutputs if necessary
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = loss_fn(logits, targets)
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            # Fallback: use a simple cross-entropy with self-supervised target
            # This is mainly for smoke tests; real training should pass loss_fn
            loss = outputs.logits.sum() * 0.0  # Placeholder — no real loss
            logger.warning(
                "No loss_fn or targets provided; using zero loss placeholder."
            )
        else:
            # Model returned a scalar loss directly
            loss = outputs if isinstance(outputs, Tensor) else outputs.logits

    # Scale loss by accumulation steps for proper gradient averaging
    scaled_loss = loss / config.grad_accum_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    # Determine if this is an accumulation boundary
    is_accum_boundary = ((step + 1) % config.grad_accum_steps == 0)

    grad_norm = 0.0

    if is_accum_boundary:
        # Unscale gradients (required before clip_grad_norm with scaler)
        if scaler is not None:
            scaler.unscale_(optimizer)

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.grad_clip_norm,
        ).item()

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Zero gradients for next accumulation cycle
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": loss.detach().item(),
        "grad_norm": grad_norm,
    }
