"""FSDP2 (Fully Sharded Data Parallel v2) integration for VGB runtime.

Applies PyTorch 2.x fully_shard bottom-up on VGB blocks with configurable
mixed precision policy and prefetch strategies.

IMPORTANT: The optimizer MUST be created AFTER sharding is applied.

Validates: Requirements 10.1, 10.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FSDP2Config:
    """Configuration for FSDP2 sharding.

    Attributes:
        precision: AMP precision mode ('bf16', 'fp16', 'fp32').
        reshard_after_forward: Whether to reshard parameters after forward.
            True saves memory, False saves communication in backward.
        prefetch_num: Number of blocks to prefetch during forward/backward.
            Higher values overlap communication with compute.
    """

    precision: str = "bf16"
    reshard_after_forward: bool = True
    prefetch_num: int = 1


def _check_distributed_available() -> None:
    """Verify that distributed training is properly initialized.

    Raises:
        RuntimeError: If torch.distributed is not available or not initialized.
    """
    if not torch.distributed.is_available():
        raise RuntimeError(
            "torch.distributed is not available. "
            "FSDP2 requires a PyTorch build with distributed support."
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "FSDP2 requires torchrun or manual init_process_group() "
            "before calling apply_fsdp2(). "
            "Use `torchrun --nproc_per_node=N script.py` to launch."
        )


def _build_mixed_precision_policy(
    precision: str,
) -> Any:
    """Build a MixedPrecisionPolicy based on the precision setting.

    Args:
        precision: One of 'bf16', 'fp16', 'fp32'.

    Returns:
        A MixedPrecisionPolicy instance or None for fp32.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if precision == "bf16":
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
    elif precision == "fp16":
        return MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            output_dtype=torch.float32,  # Keep output in fp32 for stability
        )
    else:
        # fp32: no mixed precision
        return None


def _find_shardable_blocks(model: nn.Module) -> list[nn.Module]:
    """Find VGB blocks suitable for FSDP2 sharding (bottom-up order).

    Searches for repetitive VGB blocks in encoder and decoder,
    which are the natural units for sharding.

    Args:
        model: The model to inspect.

    Returns:
        List of blocks to shard, ordered bottom-up (leaves first).
    """
    blocks: list[nn.Module] = []

    # Look in encoder
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        vgb_blocks = getattr(encoder, "vgb_blocks", None)
        if vgb_blocks is not None and isinstance(vgb_blocks, nn.ModuleList):
            blocks.extend(vgb_blocks)

    # Look in decoder
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        vgb_blocks = getattr(decoder, "vgb_blocks", None)
        if vgb_blocks is not None and isinstance(vgb_blocks, nn.ModuleList):
            blocks.extend(vgb_blocks)

    return blocks


def apply_fsdp2(
    model: nn.Module,
    config: Optional[FSDP2Config] = None,
) -> nn.Module:
    """Apply FSDP2 fully_shard bottom-up on VGB blocks.

    Shards individual VGB blocks first (bottom-up), then applies
    fully_shard to the top-level model. This enables per-block
    communication overlap and efficient memory usage.

    IMPORTANT: Create the optimizer AFTER calling this function.
    FSDP2 modifies parameter storage, so optimizer param groups
    must reference the post-sharding parameters.

    Args:
        model: The model to shard. Expected to have .encoder/.decoder
            with .vgb_blocks attributes.
        config: FSDP2 configuration. Defaults to FSDP2Config().

    Returns:
        The model with FSDP2 sharding applied (modified in-place,
        also returned for convenience).

    Raises:
        RuntimeError: If distributed is not initialized.

    Example:
        model = VSNModel(vsn_config).cuda()
        model = apply_fsdp2(model, FSDP2Config(precision='bf16'))
        # Create optimizer AFTER sharding
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    """
    if config is None:
        config = FSDP2Config()

    _check_distributed_available()

    from torch.distributed.fsdp import fully_shard

    # Build mixed precision policy
    mp_policy = _build_mixed_precision_policy(config.precision)

    # Build kwargs for fully_shard
    shard_kwargs: dict[str, Any] = {}
    if mp_policy is not None:
        shard_kwargs["mp_policy"] = mp_policy
    shard_kwargs["reshard_after_forward"] = config.reshard_after_forward

    # Step 1: Shard individual VGB blocks bottom-up
    blocks = _find_shardable_blocks(model)
    for block in blocks:
        fully_shard(block, **shard_kwargs)

    # Step 2: Shard the top-level model
    fully_shard(model, **shard_kwargs)

    logger.info(
        "FSDP2 applied: %d VGB blocks sharded (precision=%s, "
        "reshard_after_forward=%s).",
        len(blocks),
        config.precision,
        config.reshard_after_forward,
    )

    return model
