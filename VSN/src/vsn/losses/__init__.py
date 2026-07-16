"""Funciones de pérdida: cross-entropy, MSE, L1, combinador multi-tarea."""

from vsn.losses.combiner import MultiTaskLoss
from vsn.losses.cross_entropy import MaskedCrossEntropyLoss
from vsn.losses.l1 import MaskedL1Loss
from vsn.losses.mse import MaskedMSELoss

__all__ = [
    "MaskedCrossEntropyLoss",
    "MaskedMSELoss",
    "MaskedL1Loss",
    "MultiTaskLoss",
]
