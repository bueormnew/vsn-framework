"""VSN Encoder — Propagación volumétrica sobre el eje X.

Implementación de la Sección 8 de la especificación formal:

    V_x = 𝟙[x=0]·Φ(C) + 𝟙[x≥1]·F(V_(x−1); θ_(x−1)) + 𝟙[x≥2]·G(V_(x−2); θ_(x−2))

Pipeline completo:
    1. Tokens entran → Input Cache los acumula
    2. Cuando ICS se alcanza o flush es invocado → extraer tokens
    3. Operador de posicionamiento Φ mapea tokens a volumen (X, Y, Z, d)
    4. Propagar a través de bloques VGB plano por plano sobre X
    5. Retornar V_{X-1} (el último plano)

La propagación es EXCLUSIVAMENTE hacia adelante en el eje X:
    - F del plano x contribuye aditivamente al plano x+1
    - G del plano x contribuye aditivamente al plano x+2
    - La memoria M se propaga secuencialmente: M_0=zeros, cada VGB produce M_new

El forward acepta tokens ya embedidos (batch, num_tokens, d) — NO IDs raw.
La capa de embedding es externa al encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from vsn.core.input_cache import InputCache
from vsn.core.positioning import PositioningOperator
from vsn.core.vgb import VGBv1
from vsn.core.vgb_v2 import VGBv2
from vsn.core.vgb_v3 import VGBv3


class VSNEncoder(nn.Module):
    """Encoder volumétrico VSN con propagación sobre eje X.

    Integra Input Cache, operador de posicionamiento Φ, y una secuencia
    de bloques VGB v1 que propagan información exclusivamente hacia adelante.

    Args:
        X: Número de planos (profundidad del volumen / capas del encoder).
        Y: Alto del plano (filas).
        Z: Ancho del plano (columnas).
        d: Dimensión del embedding por voxel.
        ics: Tamaño del Input Cache (Input Cache Size).
    """

    def __init__(self, X: int, Y: int, Z: int, d: int, ics: int, vgb_version: str = "v1") -> None:
        super().__init__()

        if X <= 0:
            raise ValueError(f"X debe ser positivo, recibido: {X}")
        if Y <= 0:
            raise ValueError(f"Y debe ser positivo, recibido: {Y}")
        if Z <= 0:
            raise ValueError(f"Z debe ser positivo, recibido: {Z}")
        if d <= 0:
            raise ValueError(f"d debe ser positivo, recibido: {d}")
        if ics <= 0:
            raise ValueError(f"ics debe ser positivo, recibido: {ics}")

        self.X = X
        self.Y = Y
        self.Z = Z
        self.d = d
        self.ics = ics
        self.vgb_version = vgb_version

        # Operador de posicionamiento Φ: mapeo determinista tokens → volumen
        self.phi = PositioningOperator(X, Y, Z)

        # Bloques VGB: seleccionar versión según configuración
        if vgb_version == "v3":
            spatial_size = Y * Z
            self.vgb_blocks = nn.ModuleList([VGBv3(d, plane_idx=x, spatial_size=spatial_size) for x in range(X)])
        elif vgb_version == "v2":
            spatial_size = Y * Z
            self.vgb_blocks = nn.ModuleList([VGBv2(d, plane_idx=x, spatial_size=spatial_size) for x in range(X)])
        else:
            self.vgb_blocks = nn.ModuleList([VGBv1(d, plane_idx=x) for x in range(X)])

    def forward(self, tokens: Tensor) -> Tensor:
        """Procesa tokens a través del posicionamiento Φ y propaga sobre X.

        El encoder gestiona internamente el llenado del volumen:
        1. Recibe tokens ya embedidos
        2. Aplica Φ para posicionarlos en el volumen (X, Y, Z, d)
        3. Propaga plano a plano con contribuciones F y G aditivas
        4. Retorna el último plano V_{X-1}

        Args:
            tokens: Tensor de shape (batch, num_tokens, d).
                    Los tokens ya están embedidos (no son IDs).
                    num_tokens debe ser ≤ X*Y*Z.

        Returns:
            Tensor de shape (batch, Y, Z, d) — el plano V_{X-1}.
        """
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens debe ser 3D (batch, num_tokens, d), "
                f"recibido ndim={tokens.ndim}, shape={tokens.shape}"
            )

        batch, num_tokens, d = tokens.shape

        if d != self.d:
            raise ValueError(
                f"Dimensión de tokens ({d}) no coincide con d={self.d}"
            )

        # Paso 1: Posicionamiento Φ — mapear tokens al volumen (batch, X, Y, Z, d)
        volume = self.phi(tokens)  # (batch, X, Y, Z, d)

        # Paso 2: Inicializar memoria M como zeros
        M = tokens.new_zeros(batch, self.Y, self.Z, self.d)

        # Paso 3: Extraer V_0 del volumen posicionado (plano x=0)
        # V[x] almacena el estado del plano x después de procesar contribuciones
        # Usamos una lista para almacenar los planos procesados
        V = [volume[:, x, :, :, :] for x in range(self.X)]  # cada uno (batch, Y, Z, d)

        # Paso 4: Propagar plano a plano con VGB blocks
        # La fórmula formal:
        #   V_x = 𝟙[x=0]·Φ(C)[x] + 𝟙[x≥1]·F(V_(x−1); θ_(x−1)) + 𝟙[x≥2]·G(V_(x−2); θ_(x−2))
        #
        # Interpretación de implementación:
        #   - V_0 viene directamente de Φ (ya está en V[0])
        #   - Para x≥1: al plano V[x] se le SUMA F del VGB en plano x-1
        #   - Para x≥2: al plano V[x] se le SUMA G del VGB en plano x-2
        #
        # El VGB en plano x procesa V[x] y produce (F, G, r, M_new):
        #   - F contribuye aditivamente al plano x+1
        #   - G contribuye aditivamente al plano x+2
        #   - M_new se pasa al VGB del plano x+1

        # Almacenar F y G producidos por cada VGB para propagarlos
        F_outputs = [None] * self.X  # F[x] = F producido por VGB en plano x
        G_outputs = [None] * self.X  # G[x] = G producido por VGB en plano x

        last_r = None  # Residual del último VGB procesado

        for x in range(self.X):
            # Construir el input del plano x acumulando contribuciones
            plane_input = V[x]

            # Contribución F del plano x-1 (si x≥1)
            if x >= 1 and F_outputs[x - 1] is not None:
                plane_input = plane_input + F_outputs[x - 1]

            # Contribución G del plano x-2 (si x≥2)
            if x >= 2 and G_outputs[x - 2] is not None:
                plane_input = plane_input + G_outputs[x - 2]

            # Procesar el plano x con su VGB block
            F_x, G_x, r_x, M_new = self.vgb_blocks[x](plane_input, M)

            # Almacenar F y G para contribuciones futuras
            F_outputs[x] = F_x
            G_outputs[x] = G_x

            # Guardar el residual del último bloque procesado
            last_r = r_x

            # Propagar memoria al siguiente plano
            M = M_new

        # Retornar el residual del último VGB (plano X-1 procesado).
        # El residual r = V_{X-1} + MLP_out garantiza que:
        # - Todos los parámetros del encoder reciben gradientes
        # - La información de memoria contribuye a la salida
        # - Las conexiones F/G previas fluyen hasta el output
        return last_r

    def __repr__(self) -> str:
        return (
            f"VSNEncoder(X={self.X}, Y={self.Y}, Z={self.Z}, "
            f"d={self.d}, ics={self.ics}, "
            f"num_params={sum(p.numel() for p in self.parameters()):,})"
        )
