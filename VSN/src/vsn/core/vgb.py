"""VGB v1 — Voxel Gate Block, versión 1.

Implementación exacta de los 6 pasos formales (Sección 7 de la spec):

    Paso 1 — Normalización (RMSNorm):
        v̂ = (v / RMS(v)) ⊙ γ_x

    Paso 2 — Proyecciones lineales (memoria, características, compuerta):
        m = W_m · v̂ + b_m
        c = W_c · v̂ + b_c
        g = σ(W_g · v̂ + b_g)       g ∈ (0,1)^d

    Paso 3 — Actualización de memoria (VGB v1):
        M_new[y,z] = g ⊙ M_old[y,z] + (1 − g) ⊙ m

    Paso 4 — Expansión MLP (d → 4d → d):
        h = GELU(W_1 · c + b_1)     h ∈ ℝ^(4d)
        o = W_2 · h + b_2           o ∈ ℝ^d

    Paso 5 — Conexión residual:
        r = v + o

    Paso 6 — Conexiones de salida:
        F(v; θ_x) := r              (contribución al plano x+1)
        G(v; θ_x) := W_P2 · r + b_P2  (contribución al plano x+2)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vsn.core.rms_norm import RMSNorm


class VGBv1(nn.Module):
    """Voxel Gate Block v1 — bloque elemental con 6 pasos formales.

    Cada instancia tiene parámetros independientes (θ_x) asociados
    a su posición de plano ``plane_idx``.

    Args:
        d: Dimensión del embedding por voxel.
        plane_idx: Índice del plano al que pertenece este bloque.
    """

    def __init__(self, d: int, plane_idx: int) -> None:
        super().__init__()
        self.d = d
        self.plane_idx = plane_idx

        # Paso 1 — RMSNorm
        self.norm = RMSNorm(d)

        # Paso 2 — Proyecciones lineales (todas con bias según spec formal)
        self.W_m = nn.Linear(d, d, bias=True)  # proyección memoria
        self.W_c = nn.Linear(d, d, bias=True)  # proyección características
        self.W_g = nn.Linear(d, d, bias=True)  # proyección compuerta

        # Paso 4 — MLP d → 4d → d con GELU
        self.mlp_up = nn.Linear(d, 4 * d, bias=True)    # W_1: d → 4d
        self.mlp_down = nn.Linear(4 * d, d, bias=True)  # W_2: 4d → d
        self.activation = nn.GELU()

        # Paso 6 — Proyección G (F := r, no necesita parámetros)
        self.W_P2 = nn.Linear(d, d, bias=True)  # W_P2: contribución al plano x+2

    def forward(
        self, x: Tensor, M: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward del VGB v1 con los 6 pasos formales.

        Args:
            x: Tensor de shape (B, Y, Z, d) — input del plano actual.
            M: Tensor de shape (B, Y, Z, d) — estado de memoria previo.

        Returns:
            Tuple (F, G, r, M_new) donde:
                F: (B, Y, Z, d) — contribución al plano siguiente (x+1).
                G: (B, Y, Z, d) — contribución al plano subsiguiente (x+2).
                r: (B, Y, Z, d) — residual completo.
                M_new: (B, Y, Z, d) — memoria actualizada.
        """
        # Paso 1: RMSNorm
        x_norm = self.norm(x)

        # Paso 2: Proyecciones lineales
        m = self.W_m(x_norm)                    # m = W_m · x_norm + b_m
        c = self.W_c(x_norm)                    # c = W_c · x_norm + b_c
        g = torch.sigmoid(self.W_g(x_norm))     # g = σ(W_g · x_norm + b_g)

        # Paso 3: Actualización gated de memoria
        M_new = g * M + (1 - g) * m

        # Paso 4: MLP d → 4d → d (input = c solamente)
        h = self.activation(self.mlp_up(c))     # h = GELU(W_1 · c + b_1)
        o = self.mlp_down(h)                    # o = W_2 · h + b_2

        # Paso 5: Conexión residual
        r = x + o

        # Paso 6: Conexiones de salida
        F = r                                   # F := r (contribución plano x+1)
        G = self.W_P2(r)                        # G := W_P2 · r + b_P2 (plano x+2)

        return F, G, r, M_new
