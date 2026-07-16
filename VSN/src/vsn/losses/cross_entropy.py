"""Masked Cross-Entropy Loss — wrapper tipado sobre F.cross_entropy con soporte de máscara."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss con soporte de máscara opcional.

    Cuando se proporciona una máscara, se anulan las pérdidas en posiciones
    enmascaradas antes de aplicar la reducción.

    Args:
        reduction: Estrategia de reducción ('mean', 'sum', 'none').
        label_smoothing: Factor de label smoothing (0.0 por defecto).
    """

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction debe ser 'mean', 'sum' o 'none', got '{reduction}'"
            )
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calcula cross-entropy con máscara opcional.

        Args:
            input: Logits de shape (N, C) o (N, C, ...) donde C es el número de clases.
            target: Targets de shape (N,) o (N, ...) con índices de clase.
            mask: Máscara booleana de shape (N,) o (N, ...). True indica posiciones válidas.

        Returns:
            Pérdida escalar (si reduction='mean'|'sum') o tensor (si reduction='none').
        """
        # Calcular loss sin reducción para poder aplicar máscara
        unreduced = F.cross_entropy(
            input,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        if mask is not None:
            # Asegurar que la máscara tenga la misma forma que las pérdidas
            mask = mask.to(dtype=unreduced.dtype, device=unreduced.device)
            unreduced = unreduced * mask

        if self.reduction == "none":
            return unreduced
        elif self.reduction == "sum":
            return unreduced.sum()
        else:  # mean
            if mask is not None:
                # Media solo sobre posiciones válidas
                num_valid = mask.sum().clamp(min=1.0)
                return unreduced.sum() / num_valid
            return unreduced.mean()
