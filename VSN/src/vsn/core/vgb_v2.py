"""VGB v2 — Voxel Gate Block con Spatial Mixing integrado.

Evolución del VGB v1 que añade comunicación entre posiciones (Y,Z) dentro
de cada plano. Preserva todos los invariantes de la arquitectura VSN:

    I1: Propagación exclusiva sobre eje X ✓
    I2: Todos los planos usan la misma arquitectura de bloque ✓
    I3: Parámetros independientes por plano ✓
    I4: Sin bloques especializados por profundidad ✓
    I10: El VGB evoluciona conservando los invariantes ✓

Cambio respecto a v1:
    - v1: las operaciones son per-voxel (cada [y,z] se transforma independientemente)
    - v2: se añade un paso de SPATIAL MIXING que permite a cada posición ver
      todas las demás posiciones del plano ANTES de las proyecciones lineales.
      Esto asegura que los datos se cruzan entre posiciones en CADA bloque.

Los 7 pasos del VGB v2:

    Paso 1 — Normalización (RMSNorm)
    Paso 2 — SPATIAL MIXING: flatten(Y×Z) → Linear(N,N) → reshape ← NUEVO
    Paso 3 — Proyecciones lineales (memoria, características, compuerta)
    Paso 4 — Actualización gated de memoria
    Paso 5 — Expansión MLP (d → 4d → d)
    Paso 6 — Conexión residual
    Paso 7 — Conexiones de salida (F, G)

El spatial mixing es una proyección lineal sobre la dimensión espacial (Y*Z),
aplicada independientemente a cada canal d. Esto es:
    - O(Y*Z * Y*Z * d) en cómputo — lineal en d, cuadrático en posiciones del plano
    - Equivalente a un "token mixing" tipo MLP-Mixer pero integrado en el bloque
    - Permite que CADA posición tenga información de TODAS las demás posiciones
    - Se aplica en TODOS los planos (no es una capa separada)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vsn.core.rms_norm import RMSNorm


class VGBv2(nn.Module):
    """Voxel Gate Block v2 — con spatial mixing integrado.

    Evolución del VGB v1 que añade comunicación entre posiciones del plano.
    Cada instancia tiene parámetros independientes (θ_x) asociados a su
    posición de plano.

    Args:
        d: Dimensión del embedding por voxel.
        plane_idx: Índice del plano al que pertenece este bloque.
        spatial_size: Número total de posiciones en el plano (Y * Z).
            Requerido para dimensionar la capa de spatial mixing.
    """

    def __init__(self, d: int, plane_idx: int, spatial_size: int) -> None:
        super().__init__()
        self.d = d
        self.plane_idx = plane_idx
        self.spatial_size = spatial_size  # Y * Z

        # Paso 1 — RMSNorm
        self.norm = RMSNorm(d)

        # Paso 2 — SPATIAL MIXING: comunicación entre posiciones del plano
        # Opera sobre la dimensión espacial (Y*Z) independientemente por canal d
        # Input: (B, Y*Z, d) → transpose → (B, d, Y*Z) → Linear(Y*Z, Y*Z) → back
        self.spatial_norm = RMSNorm(d)
        self.spatial_mix = nn.Linear(spatial_size, spatial_size, bias=True)

        # Paso 3 — Proyecciones lineales (todas con bias)
        self.W_m = nn.Linear(d, d, bias=True)  # proyección memoria
        self.W_c = nn.Linear(d, d, bias=True)  # proyección características
        self.W_g = nn.Linear(d, d, bias=True)  # proyección compuerta

        # Paso 5 — MLP d → 4d → d con GELU
        self.mlp_up = nn.Linear(d, 4 * d, bias=True)
        self.mlp_down = nn.Linear(4 * d, d, bias=True)
        self.activation = nn.GELU()

        # Paso 7 — Proyección G
        self.W_P2 = nn.Linear(d, d, bias=True)

    def forward(
        self, x: Tensor, M: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward del VGB v2 con spatial mixing.

        Args:
            x: Tensor de shape (B, Y, Z, d) — input del plano actual.
            M: Tensor de shape (B, Y, Z, d) — estado de memoria previo.

        Returns:
            Tuple (F, G, r, M_new) con las mismas shapes que v1.
        """
        B, Y, Z, d = x.shape
        N = Y * Z  # spatial_size actual (puede ser menor que self.spatial_size)

        # Paso 1: RMSNorm
        x_norm = self.norm(x)

        # Paso 2: SPATIAL MIXING — cruce de información entre posiciones
        # Flatten espacial: (B, Y, Z, d) → (B, N, d)
        x_flat = x_norm.reshape(B, N, d)
        # Normalizar antes del mixing
        x_mix_in = self.spatial_norm(x_flat)
        # Transponer: (B, N, d) → (B, d, N)
        x_t = x_mix_in.transpose(1, 2)  # (B, d, N)
        
        # Pad a spatial_size si N < spatial_size (generación con secuencia corta)
        if N < self.spatial_size:
            x_t = torch.nn.functional.pad(x_t, (0, self.spatial_size - N))
        
        # Linear(spatial_size, spatial_size)
        x_mixed = self.spatial_mix(x_t)
        
        # Truncar de vuelta a N si fue paddeado
        if N < self.spatial_size:
            x_mixed = x_mixed[:, :, :N]
        
        # Transponer de vuelta: (B, d, N) → (B, N, d)
        x_mixed = x_mixed.transpose(1, 2)
        
        # Residual del mixing + reshape back
        x_spatial = (x_flat + x_mixed).reshape(B, Y, Z, d)

        # Paso 3: Proyecciones lineales (ahora con info spatial mezclada)
        m = self.W_m(x_spatial)
        c = self.W_c(x_spatial)
        g = torch.sigmoid(self.W_g(x_spatial))

        # Paso 4: Actualización gated de memoria
        M_new = g * M + (1 - g) * m

        # Paso 5: MLP d → 4d → d
        h = self.activation(self.mlp_up(c))
        o = self.mlp_down(h)

        # Paso 6: Conexión residual (con input original, no el mixed)
        r = x + o

        # Paso 7: Conexiones de salida
        F_out = r
        G_out = self.W_P2(r)

        return F_out, G_out, r, M_new
