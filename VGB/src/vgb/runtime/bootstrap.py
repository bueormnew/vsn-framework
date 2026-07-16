"""Bootstrap module for VGB runtime initialization.

Handles seed setup, logging configuration, device detection,
and distributed process group initialization (torchrun/NCCL).

Validates: Requirements 11.5, 10.4
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import torch

from vgb.config.schema import RuntimeConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeContext:
    """Context produced by bootstrap, available throughout the run.

    Encapsulates all runtime state needed by training/inference
    code without requiring repeated environment inspection.
    """

    rank: int  # Global rank (0 for single-gpu)
    local_rank: int  # Local rank within node
    world_size: int  # Total number of processes
    device: torch.device  # Device assigned to this process
    is_main: bool  # True if rank == 0 (for logging/saving)
    distributed: bool  # True if running with torchrun (world_size > 1)


def _set_seeds(seed: int) -> None:
    """Set seeds for reproducibility across torch, random, and numpy.

    Args:
        seed: Base random seed.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # numpy is optional — set seed if available
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # Enable deterministic CUDA ops when possible
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detect_distributed_env() -> tuple[int, int, int]:
    """Detect torchrun environment from environment variables.

    Returns:
        Tuple of (rank, local_rank, world_size).
        Defaults to (0, 0, 1) for single-GPU mode.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _init_process_group(rank: int, world_size: int) -> None:
    """Initialize NCCL process group for distributed training.

    Only called when world_size > 1 (torchrun environment detected).

    Args:
        rank: Global rank of this process.
        world_size: Total number of processes.
    """
    if not torch.distributed.is_available():
        raise RuntimeError(
            "torch.distributed is not available but torchrun env detected "
            f"(WORLD_SIZE={world_size}). Install a torch build with "
            "distributed support."
        )

    # Use NCCL for GPU, gloo as fallback for CPU
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def _configure_logging(is_main: bool) -> None:
    """Configure logging: only main process logs at INFO, others at WARNING.

    Args:
        is_main: True if this is rank 0.
    """
    level = logging.INFO if is_main else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _select_device(local_rank: int) -> torch.device:
    """Select device based on CUDA availability and local rank.

    Args:
        local_rank: Local rank within the node.

    Returns:
        torch.device for this process (cuda:{local_rank} or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def bootstrap(config: RuntimeConfig) -> RuntimeContext:
    """Initialize the runtime: seeds, logging, device, distributed context.

    Steps:
        1. Set seeds for reproducibility (torch, random, numpy if available)
        2. Detect torchrun environment (RANK, LOCAL_RANK, WORLD_SIZE from env vars)
        3. If distributed: init_process_group (NCCL for GPU, gloo for CPU)
        4. Set local device (cuda:{local_rank} or cpu)
        5. Configure logging (only main process logs at INFO level)
        6. Return RuntimeContext

    Args:
        config: RuntimeConfig with seed and other runtime settings.

    Returns:
        RuntimeContext with rank, device, and distributed state.
    """
    # Step 1: Seeds
    _set_seeds(config.seed)

    # Step 2: Detect distributed environment
    rank, local_rank, world_size = _detect_distributed_env()
    distributed = world_size > 1

    # Step 3: Initialize process group if distributed
    if distributed:
        _init_process_group(rank, world_size)

    # Step 4: Select device
    device = _select_device(local_rank)

    # Step 5: Configure logging
    is_main = rank == 0
    _configure_logging(is_main)

    # Log bootstrap summary (only on main)
    if is_main:
        logger.info(
            "Bootstrap complete: rank=%d, world_size=%d, device=%s, "
            "distributed=%s, seed=%d",
            rank,
            world_size,
            device,
            distributed,
            config.seed,
        )

    # Step 6: Return context
    return RuntimeContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=is_main,
        distributed=distributed,
    )
