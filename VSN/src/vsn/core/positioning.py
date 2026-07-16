"""Operador de posicionamiento Φ — mapeo determinista tokens → volumen 3D.

Implementación de la Sección 6.1 de la especificación formal:

    j ∈ [0, num_tokens)  — índice lineal del token en el Input Cache

    x = ⌊ j / (Y·Z) ⌋   — plano (profundidad)
    i = j mod (Y·Z)      — índice intra-plano
    y = i mod Y           — fila dentro del plano
    z = ⌊ i / Y ⌋        — columna dentro del plano

    Φ(C)[x, y, z] := C[j]

Los tokens se distribuyen en raster order: dentro de cada plano,
primero se llena la dimensión Y (filas), luego Z (columnas).
Al completar un plano Y×Z, se avanza al siguiente plano en X.

Propiedades garantizadas:
    - Determinismo: misma entrada → misma salida (sin estado mutable).
    - Sin parámetros aprendibles: operador puramente geométrico.
    - Padding con ceros: si num_tokens < X*Y*Z, posiciones restantes = 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PositioningOperator(nn.Module):
    """Operador Φ: mapeo determinista de tokens a volumen (X, Y, Z, d).

    Distribuye una secuencia plana de tokens en una estructura volumétrica
    3D usando raster order (Y primero, luego Z, luego X).

    No tiene parámetros entrenables — es una operación puramente geométrica
    que garantiza determinismo: la misma secuencia de entrada siempre produce
    el mismo volumen de salida.

    Args:
        X: Número de planos (profundidad del volumen).
        Y: Alto del plano (filas).
        Z: Ancho del plano (columnas).
    """

    def __init__(self, X: int, Y: int, Z: int) -> None:
        super().__init__()

        if X <= 0:
            raise ValueError(f"X debe ser positivo, recibido: {X}")
        if Y <= 0:
            raise ValueError(f"Y debe ser positivo, recibido: {Y}")
        if Z <= 0:
            raise ValueError(f"Z debe ser positivo, recibido: {Z}")

        self.X = X
        self.Y = Y
        self.Z = Z
        self.capacity = X * Y * Z  # máximo de tokens que caben en el volumen

        # Pre-computar índices de mapeo para evitar cálculo repetido.
        # Para cada posición j en [0, capacity), se calcula (x, y, z).
        indices_j = torch.arange(self.capacity)
        plane_size = Y * Z

        x_indices = indices_j // plane_size          # x = ⌊ j / (Y·Z) ⌋
        intra_plane = indices_j % plane_size         # i = j mod (Y·Z)
        y_indices = intra_plane % Y                  # y = i mod Y
        z_indices = intra_plane // Y                 # z = ⌊ i / Y ⌋

        # Registrar como buffers (se mueven con .to(device) pero no se entrenan)
        self.register_buffer("x_indices", x_indices, persistent=False)
        self.register_buffer("y_indices", y_indices, persistent=False)
        self.register_buffer("z_indices", z_indices, persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """Posiciona tokens en el volumen 3D según Φ.

        Args:
            tokens: Tensor de shape (batch, num_tokens, d).
                    num_tokens debe ser ≤ X*Y*Z (capacity).

        Returns:
            Tensor de shape (batch, X, Y, Z, d) con los tokens posicionados.
            Posiciones sin token asignado contienen ceros.

        Raises:
            ValueError: Si num_tokens > capacity (X*Y*Z).
            ValueError: Si tokens no es 3D.
        """
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens debe ser 3D (batch, num_tokens, d), "
                f"recibido ndim={tokens.ndim}, shape={tokens.shape}"
            )

        batch, num_tokens, d = tokens.shape

        if num_tokens > self.capacity:
            raise ValueError(
                f"num_tokens ({num_tokens}) excede la capacidad del volumen "
                f"({self.capacity} = {self.X}×{self.Y}×{self.Z})"
            )

        # Crear volumen de salida inicializado en ceros
        volume = tokens.new_zeros(batch, self.X, self.Y, self.Z, d)

        if num_tokens == 0:
            return volume

        # Usar los índices pre-computados (solo los primeros num_tokens)
        x_idx = self.x_indices[:num_tokens]  # (num_tokens,)
        y_idx = self.y_indices[:num_tokens]  # (num_tokens,)
        z_idx = self.z_indices[:num_tokens]  # (num_tokens,)

        # Indexación avanzada para posicionar todos los tokens de una vez.
        # batch_idx expande para cubrir la dimensión batch.
        batch_idx = torch.arange(batch, device=tokens.device).unsqueeze(1)  # (batch, 1)

        # volume[b, x, y, z, :] = tokens[b, j, :] para cada j
        volume[batch_idx, x_idx, y_idx, z_idx, :] = tokens

        return volume

    def __repr__(self) -> str:
        return (
            f"PositioningOperator(X={self.X}, Y={self.Y}, Z={self.Z}, "
            f"capacity={self.capacity})"
        )
