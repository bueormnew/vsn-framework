"""Bundle export wrapper for the VGB framework.

Provides export_inference_bundle as a high-level wrapper around
vsn.formats.bundle.export_bundle with VGB config integration.
Materializes full weights if FSDP2 is active (gathers all parameters first).

Validates: Requirements 9.3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vsn.formats.bundle import export_bundle
from vgb.config.schema import FullConfig

logger = logging.getLogger(__name__)


def _is_fsdp_active(model: nn.Module) -> bool:
    """Check if the model has FSDP2 sharding applied.

    Detects FSDP2 by checking for the flat_param attribute
    or the FSDPModule wrapper class.

    Args:
        model: The model to inspect.

    Returns:
        True if FSDP2 sharding is detected on the model.
    """
    # Check for FSDP2 markers
    try:
        from torch.distributed.fsdp import FlatParamHandle  # noqa: F401
        # FSDP2 uses fully_shard which attaches specific attributes
        if hasattr(model, "_fsdp_wrapped_module"):
            return True
    except ImportError:
        pass

    # Alternative detection: check for FSDP state
    try:
        from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
        state = _get_module_fsdp_state(model)
        if state is not None:
            return True
    except (ImportError, AttributeError):
        pass

    # FSDP2 (PyTorch 2.x fully_shard) detection
    if hasattr(model, "__fsdp_state"):
        return True

    return False


def _gather_full_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Gather full (unsharded) state_dict from an FSDP2 model.

    When FSDP2 is active, parameters are sharded across ranks.
    This function gathers all shards to produce a complete state_dict
    suitable for single-process inference.

    Args:
        model: The FSDP2-sharded model.

    Returns:
        Full (unsharded) state_dict dictionary.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        # Use FSDP full state dict context
        full_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
            return model.state_dict()
    except (ImportError, AttributeError, RuntimeError):
        # Fallback: FSDP2 (fully_shard API) may not use the FSDP wrapper class.
        # Try to get state_dict directly — it may already be full.
        logger.warning(
            "Could not use FSDP state_dict_type context; "
            "falling back to direct state_dict() call."
        )
        return model.state_dict()


def export_inference_bundle(
    model: nn.Module,
    config: Optional[FullConfig] = None,
    output_dir: str | Path = "bundle_output",
    weight_format: str = "pytorch",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export a model as an inference bundle.

    High-level wrapper around vsn.formats.bundle.export_bundle that:
    1. Detects and handles FSDP2 sharding (gathers all parameters)
    2. Integrates VGB FullConfig metadata
    3. Delegates actual export to the vsn core bundle exporter

    Args:
        model: The model to export. Can be FSDP2-sharded or plain.
        config: Optional FullConfig for adding VGB-specific metadata
            to the bundle manifest.
        output_dir: Directory where the bundle will be created.
        weight_format: Weight format — 'pytorch' or 'safetensors'.
        metadata: Optional additional metadata for the manifest.

    Returns:
        Path to the created bundle directory.

    Raises:
        ValueError: If the model does not have a .config attribute
            (required by export_bundle).
    """
    output_dir = Path(output_dir)

    # Build metadata from VGB config if provided
    bundle_metadata = metadata or {}
    if config is not None:
        bundle_metadata["vgb_config"] = {
            "precision": config.train.precision,
            "strategy": config.runtime.strategy,
        }

    # Handle FSDP2: materialize full weights
    if _is_fsdp_active(model):
        logger.info(
            "FSDP2 detected. Gathering full state_dict for export..."
        )
        full_state_dict = _gather_full_state_dict(model)

        # We need to create a non-sharded copy for export_bundle
        # Since export_bundle calls model.state_dict() internally,
        # we temporarily replace the state_dict
        original_state_dict_fn = model.state_dict

        def _patched_state_dict(*args, **kwargs):
            return full_state_dict

        model.state_dict = _patched_state_dict  # type: ignore[method-assign]

        try:
            bundle_path = export_bundle(
                model=model,
                output_dir=output_dir,
                weight_format=weight_format,
                metadata=bundle_metadata,
            )
        finally:
            # Restore original state_dict method
            model.state_dict = original_state_dict_fn  # type: ignore[method-assign]

        logger.info("Bundle exported (FSDP2 gathered): %s", bundle_path)
    else:
        # Standard export (no FSDP)
        bundle_path = export_bundle(
            model=model,
            output_dir=output_dir,
            weight_format=weight_format,
            metadata=bundle_metadata,
        )
        logger.info("Bundle exported: %s", bundle_path)

    return bundle_path
