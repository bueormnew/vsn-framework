"""Core matemático: VGB v1/v2, encoder, decoder, operadores P/Q/Ψ, modelo completo."""

from vsn.core.config import ConfigurationError, VSNConfig
from vsn.core.decoder import VSNDecoder
from vsn.core.encoder import VSNEncoder
from vsn.core.input_cache import InputCache
from vsn.core.invariants import (
    InvariantError,
    validate_h_sole_interface,
    validate_input_cache_precedes_encoder,
    validate_no_weight_sharing,
)
from vsn.core.latent import ProjectorP
from vsn.core.model import VSNModel
from vsn.core.positioning import PositioningOperator
from vsn.core.psi import PsiOperator
from vsn.core.rms_norm import RMSNorm
from vsn.core.transitions import TransitionQ
from vsn.core.vgb import VGBv1
from vsn.core.vgb_v2 import VGBv2
from vsn.core.vgb_v3 import VGBv3

__all__ = [
    "ConfigurationError",
    "InputCache",
    "InvariantError",
    "PositioningOperator",
    "ProjectorP",
    "PsiOperator",
    "RMSNorm",
    "TransitionQ",
    "VGBv1",
    "VGBv2",
    "VGBv3",
    "VSNConfig",
    "VSNDecoder",
    "VSNEncoder",
    "VSNModel",
    "validate_no_weight_sharing",
    "validate_input_cache_precedes_encoder",
    "validate_h_sole_interface",
]
