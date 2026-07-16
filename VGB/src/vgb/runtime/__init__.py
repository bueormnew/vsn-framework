"""Runtime utilities: bootstrap, AMP, FSDP2, activation checkpointing, DCP."""

from vgb.runtime.activation_checkpointing import (
    CheckpointSegment,
    apply_activation_checkpointing,
)
from vgb.runtime.amp import (
    amp_autocast,
    create_grad_scaler,
    eval_autocast,
    get_amp_dtype,
    infer_autocast,
    is_bf16_supported,
    train_autocast,
)
from vgb.runtime.bootstrap import RuntimeContext, bootstrap
from vgb.runtime.checkpointing import (
    CheckpointState,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from vgb.runtime.fsdp import (
    FSDP2Config,
    apply_fsdp2,
)

__all__ = [
    # Bootstrap
    "bootstrap",
    "RuntimeContext",
    # AMP
    "amp_autocast",
    "create_grad_scaler",
    "eval_autocast",
    "get_amp_dtype",
    "infer_autocast",
    "is_bf16_supported",
    "train_autocast",
    # Activation Checkpointing
    "apply_activation_checkpointing",
    "CheckpointSegment",
    # FSDP2
    "apply_fsdp2",
    "FSDP2Config",
    # DCP (Distributed Checkpointing)
    "save_distributed_checkpoint",
    "load_distributed_checkpoint",
    "CheckpointState",
]
