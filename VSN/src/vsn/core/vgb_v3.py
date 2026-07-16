"""VGB v3 — Voxel Gate Block con Causal Spatial Mixing.

Evolución del VGB v2 que añade CAUSAL MASKING al spatial mixing.
La posición t solo puede ver posiciones 0..t (no futuras).
Esto permite generación autoregresiva coherente.

Cambio respecto a v2:
    - v2: spatial mixing es bidireccional (cada posición ve TODAS las demás)
    - v3: spatial mixing es CAUSAL (cada posición solo ve las anteriores)

Esto se logra aplicando una máscara triangular inferior al mixing,
de forma que el peso de posiciones futuras sea cero.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vsn.core.rms_norm import RMSNorm


class VGBv3(nn.Module):
    """Voxel Gate Block v3 — causal spatial mixing.

    Igual que VGB v2 pero el spatial mixing es causal:
    la posición t solo recibe información de posiciones 0..t.

    Args:
        d: Dimensión del embedding por voxel.
        plane_idx: Índice del plano.
        spatial_size: Número máximo de posiciones (Y * Z).
    """

    def __init__(self, d: int, plane_idx: int, spatial_size: int) -> None:
        super().__init__()
        self.d = d
        self.plane_idx = plane_idx
        self.spatial_size = spatial_size

        # Paso 1 — RMSNorm
        self.norm = RMSNorm(d)

        # Paso 2 — CAUSAL SPATIAL MIXING
        self.spatial_norm = RMSNorm(d)
        self.spatial_mix = nn.Linear(spatial_size, spatial_size, bias=True)
        # Registrar máscara causal (triangular inferior)
        causal_mask = torch.tril(torch.ones(spatial_size, spatial_size))
        self.register_buffer("causal_mask", causal_mask)

        # Paso 3 — Proyecciones lineales
        self.W_m = nn.Linear(d, d, bias=True)
        self.W_c = nn.Linear(d, d, bias=True)
        self.W_g = nn.Linear(d, d, bias=True)

        # Paso 5 — MLP d → 4d → d
        self.mlp_up = nn.Linear(d, 4 * d, bias=True)
        self.mlp_down = nn.Linear(4 * d, d, bias=True)
        self.activation = nn.GELU()

        # Paso 7 — Proyección G
        self.W_P2 = nn.Linear(d, d, bias=True)

    def forward(
        self, x: Tensor, M: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward del VGB v3 con causal spatial mixing.

        Args:
            x: (B, Y, Z, d) — input del plano actual.
            M: (B, Y, Z, d) — estado de memoria previo.

        Returns:
            Tuple (F, G, r, M_new).
        """
        B, Y, Z, d = x.shape
        N = Y * Z

        # Paso 1: RMSNorm
        x_norm = self.norm(x)

        # Paso 2: CAUSAL SPATIAL MIXING
        x_flat = x_norm.reshape(B, N, d)
        x_mix_in = self.spatial_norm(x_flat)
        # (B, N, d) → (B, d, N)
        x_t = x_mix_in.transpose(1, 2)

        # Pad si N < spatial_size
        if N < self.spatial_size:
            x_t = F.pad(x_t, (0, self.spatial_size - N))

        # Aplicar mixing con máscara causal
        # W tiene shape (spatial_size, spatial_size)
        # Multiplicar W por la máscara causal: W_causal = W * tril
        W = self.spatial_mix.weight * self.causal_mask
        b = self.spatial_mix.bias
        x_mixed = F.linear(x_t, W, b)

        # Truncar si fue paddeado
        if N < self.spatial_size:
            x_mixed = x_mixed[:, :, :N]

        # (B, d, N) → (B, N, d)
        x_mixed = x_mixed.transpose(1, 2)
        x_spatial = (x_flat + x_mixed).reshape(B, Y, Z, d)

        # Paso 3: Proyecciones lineales
        m = self.W_m(x_spatial)
        c = self.W_c(x_spatial)
        g = torch.sigmoid(self.W_g(x_spatial))

        # Paso 4: Memoria gated
        M_new = g * M + (1 - g) * m

        # Paso 5: MLP
        h = self.activation(self.mlp_up(c))
        o = self.mlp_down(h)

        # Paso 6: Residual
        r = x + o

        # Paso 7: Salidas
        F_out = r
        G_out = self.W_P2(r)

        return F_out, G_out, r, M_new
