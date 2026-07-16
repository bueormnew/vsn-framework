"""Masked L1 Loss — L1Loss con reducción configurable y soporte de máscara."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedL1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss con soporte de máscara opcional.

    L_L1 = (1/n) Σ_i |out_i − target_i|

    Cuando se proporciona una máscara, se anulan las pérdidas en posiciones
    enmascaradas antes de aplicar la reducción.

    Args:
        reduction: Estrategia de reducción ('mean', 'sum', 'none').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction debe ser 'mean', 'sum' o 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calcula L1 loss con máscara opcional.

        Args:
            input: Predicciones de shape arbitraria.
            target: Targets de la misma shape que input.
            mask: Máscara booleana. True indica posiciones válidas.
                  Shape debe ser broadcastable con |input - target|.

        Returns:
            Pérdida escalar (si reduction='mean'|'sum') o tensor (si reduction='none').
        """
        # Calcular loss element-wise
        unreduced = F.l1_loss(input, target, reduction="none")

        if mask is not None:
            # Expandir máscara si es necesario para que sea broadcastable
            mask_float = mask.to(dtype=unreduced.dtype, device=unreduced.device)
            while mask_float.dim() < unreduced.dim():
                mask_float = mask_float.unsqueeze(-1)
            unreduced = unreduced * mask_float

        if self.reduction == "none":
            return unreduced
        elif self.reduction == "sum":
            return unreduced.sum()
        else:  # mean
            if mask is not None:
                mask_float = mask.to(dtype=unreduced.dtype, device=unreduced.device)
                while mask_float.dim() < unreduced.dim():
                    mask_float = mask_float.unsqueeze(-1)
                num_valid = mask_float.expand_as(unreduced).sum().clamp(min=1.0)
                return unreduced.sum() / num_valid
            return unreduced.mean()
