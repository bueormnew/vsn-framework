"""MultiTaskLoss — combinador de múltiples pérdidas con pesos configurables."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class MultiTaskLoss(nn.Module):
    """Combinador multi-tarea que pondera múltiples loss functions.

    Acepta un diccionario de funciones de pérdida y un diccionario de pesos.
    El forward() toma un diccionario de (predictions, targets, optional_mask) por tarea
    y retorna la suma ponderada Σ(w_i × loss_i) junto con los valores individuales.

    Args:
        losses: Dict[str, nn.Module] — funciones de pérdida por nombre de tarea.
        weights: Dict[str, float] — pesos por nombre de tarea.

    Raises:
        ValueError: Si las claves de losses y weights no coinciden.
    """

    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Dict[str, float],
    ) -> None:
        super().__init__()

        if set(losses.keys()) != set(weights.keys()):
            missing_in_weights = set(losses.keys()) - set(weights.keys())
            missing_in_losses = set(weights.keys()) - set(losses.keys())
            msg = "Las claves de losses y weights deben coincidir."
            if missing_in_weights:
                msg += f" Faltan en weights: {missing_in_weights}."
            if missing_in_losses:
                msg += f" Faltan en losses: {missing_in_losses}."
            raise ValueError(msg)

        self.losses = nn.ModuleDict(losses)
        self.weights: Dict[str, float] = weights

    def forward(
        self,
        inputs: Dict[str, Tuple[Tensor, Tensor, Optional[Tensor]]],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Calcula la pérdida combinada multi-tarea.

        Args:
            inputs: Dict[str, Tuple[predictions, targets, Optional[mask]]]
                Cada entrada mapea un nombre de tarea a una tupla de
                (predicciones, targets, máscara_opcional).

        Returns:
            Tuple de:
                - total_loss: Tensor escalar con la suma ponderada Σ(w_i × loss_i)
                - individual_losses: Dict[str, Tensor] con cada pérdida individual

        Raises:
            ValueError: Si las claves de inputs no coinciden con las tareas registradas.
        """
        if set(inputs.keys()) != set(self.losses.keys()):
            missing = set(self.losses.keys()) - set(inputs.keys())
            extra = set(inputs.keys()) - set(self.losses.keys())
            msg = "Las claves de inputs deben coincidir con las tareas registradas."
            if missing:
                msg += f" Faltan: {missing}."
            if extra:
                msg += f" Extras: {extra}."
            raise ValueError(msg)

        individual_losses: Dict[str, Tensor] = {}
        total_loss: Optional[Tensor] = None

        for task_name, loss_fn in self.losses.items():
            preds, targets, mask = inputs[task_name]

            if mask is not None:
                task_loss = loss_fn(preds, targets, mask)
            else:
                task_loss = loss_fn(preds, targets)

            individual_losses[task_name] = task_loss

            weighted = self.weights[task_name] * task_loss
            if total_loss is None:
                total_loss = weighted
            else:
                total_loss = total_loss + weighted

        # Si no hay tareas (caso borde), retornar tensor cero
        if total_loss is None:
            total_loss = torch.tensor(0.0)

        return total_loss, individual_losses
