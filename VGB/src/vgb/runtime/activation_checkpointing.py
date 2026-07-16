"""Activation checkpointing for VGB runtime.

Applies selective activation checkpointing using torch.utils.checkpoint
with use_reentrant=False on configurable encoder/decoder segments.

Validates: Requirements 10.3
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class CheckpointSegment(str, Enum):
    """Segments that can be checkpointed."""

    ENCODER = "encoder"
    DECODER = "decoder"
    BOTH = "both"


def _wrap_forward_with_checkpoint(module: nn.Module) -> None:
    """Wrap a module's forward method with activation checkpointing.

    Replaces module.forward with a version that uses
    torch.utils.checkpoint with use_reentrant=False.

    Args:
        module: The nn.Module whose forward will be wrapped.
    """
    original_forward = module.forward

    def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
        # torch.utils.checkpoint requires a function and its arguments
        # use_reentrant=False is required for compatibility with FSDP2
        # and is the recommended mode in PyTorch 2.x
        def run_fn(*inputs: Any) -> Any:
            return original_forward(*inputs, **kwargs)

        return checkpoint(run_fn, *args, use_reentrant=False)

    module.forward = checkpointed_forward  # type: ignore[assignment]


def _find_vgb_blocks(model: nn.Module, segment: str) -> list[nn.Module]:
    """Find VGB blocks in the specified segment of the model.

    Searches for submodules matching common VGB block patterns:
    - model.encoder.vgb_blocks[i]
    - model.decoder.vgb_blocks[i]

    Args:
        model: The full model to search.
        segment: One of 'encoder', 'decoder'.

    Returns:
        List of VGB block modules found in the segment.
    """
    blocks: list[nn.Module] = []

    # Look for the segment attribute
    segment_module = getattr(model, segment, None)
    if segment_module is None:
        logger.warning(
            "Segment '%s' not found on model %s. "
            "Activation checkpointing skipped for this segment.",
            segment,
            type(model).__name__,
        )
        return blocks

    # Look for vgb_blocks ModuleList
    vgb_blocks = getattr(segment_module, "vgb_blocks", None)
    if vgb_blocks is not None and isinstance(vgb_blocks, nn.ModuleList):
        blocks.extend(vgb_blocks)
    else:
        # Fallback: look for any child modules that have 'VGB' or 'vgb' in class name
        for child in segment_module.modules():
            class_name = type(child).__name__.lower()
            if "vgb" in class_name and child is not segment_module:
                blocks.append(child)

    return blocks


def apply_activation_checkpointing(
    model: nn.Module,
    segments: str | CheckpointSegment = CheckpointSegment.BOTH,
    block_filter: Optional[Callable[[nn.Module, int], bool]] = None,
) -> int:
    """Apply activation checkpointing selectively on encoder/decoder VGB blocks.

    Wraps the forward pass of VGB blocks with torch.utils.checkpoint
    (use_reentrant=False) to trade compute for memory.

    Args:
        model: The model to apply checkpointing to. Expected to have
            .encoder and/or .decoder attributes with .vgb_blocks.
        segments: Which segments to checkpoint:
            - 'encoder': Only encoder VGB blocks
            - 'decoder': Only decoder VGB blocks
            - 'both': Both encoder and decoder VGB blocks
        block_filter: Optional callable (module, index) -> bool.
            If provided, only blocks where filter returns True are
            checkpointed. Useful for checkpointing every N-th block.

    Returns:
        Number of blocks that were wrapped with checkpointing.

    Example:
        # Checkpoint all VGB blocks in both encoder and decoder
        n = apply_activation_checkpointing(model, segments='both')

        # Checkpoint only every other block in encoder
        n = apply_activation_checkpointing(
            model,
            segments='encoder',
            block_filter=lambda m, i: i % 2 == 0,
        )
    """
    segments_value = CheckpointSegment(segments)

    # Determine which segments to process
    if segments_value == CheckpointSegment.BOTH:
        segment_names = ["encoder", "decoder"]
    elif segments_value == CheckpointSegment.ENCODER:
        segment_names = ["encoder"]
    else:
        segment_names = ["decoder"]

    total_wrapped = 0

    for seg_name in segment_names:
        blocks = _find_vgb_blocks(model, seg_name)

        if not blocks:
            continue

        for idx, block in enumerate(blocks):
            # Apply filter if provided
            if block_filter is not None and not block_filter(block, idx):
                continue

            _wrap_forward_with_checkpoint(block)
            total_wrapped += 1

    logger.info(
        "Activation checkpointing applied to %d VGB blocks "
        "(segments=%s).",
        total_wrapped,
        segments_value.value,
    )

    return total_wrapped
