"""Operador P: Proyección V_{X-1} → Plano Latente H.

Definición 9.1 de la especificación formal:
    H = P(V_{X-1})
    P: ℝ^{Y×Z×d} → ℝ^{Y_H×Z_H×d_H}

P es una proyección ENTRENABLE desde el último plano del encoder hacia
el plano latente H. Soporta tres modos dimensionales:
    - compress: reduce dimensiones (Y_H < Y, Z_H < Z, o d_H < d)
    - identity: mismas dimensiones (aún con capa lineal entrenable)
    - expand: aumenta dimensiones (Y_H > Y, Z_H > Z, o d_H > d)

Implementación:
    1. Flatten (B, Y, Z, d) → (B, Y*Z*d)
    2. nn.Linear(Y*Z*d, Y_H*Z_H*d_H)
    3. Reshape → (B, Y_H, Z_H, d_H)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


_VALID_MODES = ("compress", "identity", "expand")


class ProjectorP(nn.Module):
    """Proyección entrenable V_{X-1} → H.

    Args:
        Y: Alto del plano de entrada (encoder).
        Z: Ancho del plano de entrada (encoder).
        d: Dimensión de embedding de entrada.
        Y_H: Alto del plano latente H.
        Z_H: Ancho del plano latente H.
        d_H: Dimensión de embedding de H.
        mode: Uno de 'compress', 'identity', 'expand'.
    """

    def __init__(
        self,
        Y: int,
        Z: int,
        d: int,
        Y_H: int,
        Z_H: int,
        d_H: int,
        mode: str,
    ) -> None:
        super().__init__()

        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode debe ser uno de {_VALID_MODES}, recibido: '{mode}'"
            )

        # Validar coherencia del modo con las dimensiones
        in_size = Y * Z * d
        out_size = Y_H * Z_H * d_H

        if mode == "compress" and out_size >= in_size:
            raise ValueError(
                f"mode='compress' requiere Y_H*Z_H*d_H < Y*Z*d, "
                f"pero {out_size} >= {in_size}"
            )
        if mode == "expand" and out_size <= in_size:
            raise ValueError(
                f"mode='expand' requiere Y_H*Z_H*d_H > Y*Z*d, "
                f"pero {out_size} <= {in_size}"
            )
        if mode == "identity" and out_size != in_size:
            raise ValueError(
                f"mode='identity' requiere Y_H*Z_H*d_H == Y*Z*d, "
                f"pero {out_size} != {in_size}"
            )

        self.Y = Y
        self.Z = Z
        self.d = d
        self.Y_H = Y_H
        self.Z_H = Z_H
        self.d_H = d_H
        self.mode = mode

        self.in_features = in_size
        self.out_features = out_size

        # Capa lineal entrenable: la transformación principal
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)

        # Para mode='identity', inicializar cerca de la identidad
        # para estabilidad al inicio del entrenamiento
        if mode == "identity":
            nn.init.eye_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, V_last: Tensor) -> Tensor:
        """Proyecta V_{X-1} al plano latente H.

        Args:
            V_last: Tensor de shape (B, Y, Z, d) — último plano del encoder.

        Returns:
            H: Tensor de shape (B, Y_H, Z_H, d_H) — plano latente.
        """
        B = V_last.shape[0]

        # Paso 1: Flatten espacial+feature → vector
        flat = V_last.reshape(B, -1)  # (B, Y*Z*d)

        # Paso 2: Proyección lineal entrenable
        projected = self.linear(flat)  # (B, Y_H*Z_H*d_H)

        # Paso 3: Reshape a estructura volumétrica de H
        H = projected.reshape(B, self.Y_H, self.Z_H, self.d_H)

        return H

    def extra_repr(self) -> str:
        return (
            f"({self.Y}, {self.Z}, {self.d}) → ({self.Y_H}, {self.Z_H}, {self.d_H}), "
            f"mode='{self.mode}'"
        )
