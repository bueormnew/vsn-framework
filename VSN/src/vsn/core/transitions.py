"""Operador Q: Transición H → V^dec_0.

Definición 10.1 de la especificación formal:
    V^dec_0 = Q(H)
    Q: ℝ^{Y_H×Z_H×d_H} → ℝ^{Y_dec×Z_dec×d}

Q es una proyección ENTRENABLE desde el plano latente H hacia el primer
plano del decoder V^dec_0. Q es completamente independiente de P — parámetros
separados, sin compartir pesos.

Implementación:
    1. Flatten (B, Y_H, Z_H, d_H) → (B, Y_H*Z_H*d_H)
    2. nn.Linear(Y_H*Z_H*d_H, Y_dec*Z_dec*d)
    3. Reshape → (B, Y_dec, Z_dec, d)
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TransitionQ(nn.Module):
    """Proyección entrenable H → V^dec_0, independiente de P.

    Args:
        Y_H: Alto del plano latente H (entrada).
        Z_H: Ancho del plano latente H (entrada).
        d_H: Dimensión de embedding de H (entrada).
        Y_dec: Alto del primer plano del decoder (salida).
        Z_dec: Ancho del primer plano del decoder (salida).
        d: Dimensión de embedding del decoder (salida).

    Raises:
        ValueError: Si alguna dimensión es <= 0.
    """

    def __init__(
        self,
        Y_H: int,
        Z_H: int,
        d_H: int,
        Y_dec: int,
        Z_dec: int,
        d: int,
    ) -> None:
        super().__init__()

        # Validar que todas las dimensiones sean positivas
        dims = {"Y_H": Y_H, "Z_H": Z_H, "d_H": d_H, "Y_dec": Y_dec, "Z_dec": Z_dec, "d": d}
        for name, value in dims.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} debe ser un entero positivo, recibido: {value!r}"
                )

        self.Y_H = Y_H
        self.Z_H = Z_H
        self.d_H = d_H
        self.Y_dec = Y_dec
        self.Z_dec = Z_dec
        self.d = d

        self.in_features = Y_H * Z_H * d_H
        self.out_features = Y_dec * Z_dec * d

        # Capa lineal entrenable: la transformación principal
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)

    def forward(self, H: Tensor) -> Tensor:
        """Proyecta el plano latente H al primer plano del decoder V^dec_0.

        Args:
            H: Tensor de shape (B, Y_H, Z_H, d_H) — plano latente.

        Returns:
            V_dec_0: Tensor de shape (B, Y_dec, Z_dec, d) — primer plano del decoder.
        """
        B = H.shape[0]

        # Paso 1: Flatten espacial+feature → vector
        flat = H.reshape(B, -1)  # (B, Y_H*Z_H*d_H)

        # Paso 2: Proyección lineal entrenable
        projected = self.linear(flat)  # (B, Y_dec*Z_dec*d)

        # Paso 3: Reshape a estructura volumétrica del decoder
        V_dec_0 = projected.reshape(B, self.Y_dec, self.Z_dec, self.d)

        return V_dec_0

    def extra_repr(self) -> str:
        return (
            f"({self.Y_H}, {self.Z_H}, {self.d_H}) → "
            f"({self.Y_dec}, {self.Z_dec}, {self.d}), "
            f"in_features={self.in_features}, out_features={self.out_features}"
        )
