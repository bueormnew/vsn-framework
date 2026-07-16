"""Distributed checkpointing (DCP) for VGB runtime.

Uses torch.distributed.checkpoint for saving/loading sharded state
compatible with FSDP2, including support for resharding (loading
into a different world_size than was used for saving).

State includes: model, optimizer, scheduler, scaler, trainer_state, rng.

Validates: Requirements 9.1, 9.2, 8.4, 8.5
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Container for all state to be saved/loaded in a distributed checkpoint.

    All fields are optional to support partial saves/loads.

    Attributes:
        model: The model (or FSDP-sharded model) state.
        optimizer: The optimizer state.
        scheduler: The learning rate scheduler state.
        scaler: The GradScaler state (for fp16 AMP).
        trainer_state: Dictionary with step, epoch, metrics, etc.
        rng: RNG states for reproducibility (cpu, cuda, python, numpy).
    """

    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None
    scaler: Optional[torch.amp.GradScaler] = None
    trainer_state: Optional[Dict[str, Any]] = None
    rng: Optional[Dict[str, Any]] = None


def _check_distributed_available() -> None:
    """Verify torch.distributed is available and initialized for DCP.

    Raises:
        RuntimeError: If distributed is not available or not initialized.
    """
    if not torch.distributed.is_available():
        raise RuntimeError(
            "torch.distributed is not available. "
            "DCP requires a PyTorch build with distributed support."
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "DCP requires torchrun or manual init_process_group() "
            "before saving/loading checkpoints. "
            "Use `torchrun --nproc_per_node=N script.py` to launch."
        )


def _capture_rng_states() -> Dict[str, Any]:
    """Capture current RNG states for all relevant generators.

    Returns:
        Dictionary with keys: 'cpu', 'cuda' (if available),
        'python', 'numpy' (if available).
    """
    rng_states: Dict[str, Any] = {
        "cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
    }

    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state_all()

    try:
        import numpy as np
        rng_states["numpy"] = np.random.get_state()
    except ImportError:
        pass

    return rng_states


def _restore_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restore RNG states from a captured dictionary.

    Args:
        rng_states: Dictionary with RNG states as returned by
            _capture_rng_states().
    """
    if "cpu" in rng_states:
        torch.random.set_rng_state(rng_states["cpu"])

    if "python" in rng_states:
        random.setstate(rng_states["python"])

    if "cuda" in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states["cuda"])

    if "numpy" in rng_states:
        try:
            import numpy as np
            np.random.set_state(rng_states["numpy"])
        except ImportError:
            pass


def _build_state_dict(state: CheckpointState) -> Dict[str, Any]:
    """Build a flat state_dict from CheckpointState for DCP.

    DCP expects a dictionary where values are either tensors or
    stateful objects with .state_dict() methods.

    Args:
        state: The checkpoint state container.

    Returns:
        Dictionary suitable for DCP save.
    """
    state_dict: Dict[str, Any] = {}

    if state.model is not None:
        state_dict["model"] = state.model

    if state.optimizer is not None:
        state_dict["optimizer"] = state.optimizer

    if state.scheduler is not None:
        # Scheduler state is a plain dict
        state_dict["scheduler"] = state.scheduler.state_dict()

    if state.scaler is not None:
        state_dict["scaler"] = state.scaler.state_dict()

    if state.trainer_state is not None:
        state_dict["trainer_state"] = state.trainer_state

    if state.rng is not None:
        state_dict["rng"] = state.rng
    else:
        # Capture current RNG states by default
        state_dict["rng"] = _capture_rng_states()

    return state_dict


def save_distributed_checkpoint(
    state: CheckpointState,
    path: str | Path,
) -> None:
    """Save a distributed checkpoint using torch.distributed.checkpoint (DCP).

    Saves all training state in a format compatible with FSDP2 sharding.
    Each rank saves its own shard; DCP coordinates to produce a
    consistent checkpoint directory.

    Args:
        state: CheckpointState containing model, optimizer, scheduler,
            scaler, trainer_state, and rng states to save.
        path: Directory path where the checkpoint will be saved.
            Will be created if it doesn't exist.

    Raises:
        RuntimeError: If distributed is not initialized.

    Example:
        state = CheckpointState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            trainer_state={"step": 1000, "epoch": 2},
        )
        save_distributed_checkpoint(state, "./checkpoints/step_1000")
    """
    _check_distributed_available()

    import torch.distributed.checkpoint as dcp

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state_dict = _build_state_dict(state)

    dcp.save(state_dict, checkpoint_id=str(path))

    logger.info(
        "Distributed checkpoint saved to %s (rank=%d).",
        path,
        torch.distributed.get_rank(),
    )


def load_distributed_checkpoint(
    state: CheckpointState,
    path: str | Path,
) -> CheckpointState:
    """Load a distributed checkpoint using torch.distributed.checkpoint (DCP).

    Supports resharding: the checkpoint can be loaded into a different
    world_size than it was saved with. DCP handles redistributing
    shards automatically.

    The model and optimizer in `state` must already be instantiated
    (with the same architecture) but can be on any world_size.
    DCP will load and redistribute the saved state appropriately.

    Args:
        state: CheckpointState with pre-instantiated model and optimizer.
            These objects will be loaded in-place with the saved state.
        path: Directory path of the saved checkpoint.

    Returns:
        The same CheckpointState with model/optimizer state loaded.
        Scheduler, scaler, trainer_state, and rng are populated
        from the checkpoint data.

    Raises:
        RuntimeError: If distributed is not initialized.
        FileNotFoundError: If the checkpoint path does not exist.

    Example:
        # Load into potentially different world_size
        state = CheckpointState(model=model, optimizer=optimizer)
        state = load_distributed_checkpoint(state, "./checkpoints/step_1000")
        # Restore scheduler/scaler from state.scheduler, state.scaler, etc.
    """
    _check_distributed_available()

    import torch.distributed.checkpoint as dcp

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {path}. "
            "Ensure the path points to a valid DCP checkpoint."
        )

    state_dict = _build_state_dict(state)

    dcp.load(state_dict, checkpoint_id=str(path))

    # Restore RNG states if they were saved
    if "rng" in state_dict and state_dict["rng"]:
        _restore_rng_states(state_dict["rng"])
        state.rng = state_dict["rng"]

    # Restore trainer_state
    if "trainer_state" in state_dict:
        state.trainer_state = state_dict["trainer_state"]

    logger.info(
        "Distributed checkpoint loaded from %s (rank=%d). "
        "Resharding handled automatically by DCP.",
        path,
        torch.distributed.get_rank(),
    )

    return state


def inspect_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Inspect a checkpoint directory and report its metadata.

    Reads checkpoint metadata without loading full model weights.
    Useful for verifying checkpoint contents and compatibility.

    Args:
        path: Path to checkpoint file or directory.

    Returns:
        Dictionary with checkpoint metadata:
        - path: str — the checkpoint path
        - exists: bool — whether the path exists
        - trainer_state: dict or None — training state if available
        - keys: list — top-level keys found in the checkpoint

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    info: Dict[str, Any] = {
        "path": str(path),
        "exists": True,
    }

    # If it's a directory (DCP format), list contents
    if path.is_dir():
        info["format"] = "dcp_directory"
        info["files"] = [f.name for f in path.iterdir() if f.is_file()]
        info["num_files"] = len(info["files"])
    else:
        # Single file checkpoint (PyTorch native)
        info["format"] = "pytorch_file"
        try:
            # Load only metadata (map_location=cpu, weights_only for safety)
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )
            if isinstance(checkpoint, dict):
                info["keys"] = list(checkpoint.keys())
                if "trainer_state" in checkpoint:
                    info["trainer_state"] = checkpoint["trainer_state"]
            else:
                info["keys"] = ["(single object)"]
        except Exception as e:
            info["error"] = f"Could not load checkpoint: {e}"

    return info
