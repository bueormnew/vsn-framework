"""VSN Decoder — Generación volumétrica por ventanas DGW.

Implementación de la Sección 11 de la especificación formal:

    V^dec_x = 𝟙[x=0]·Q(H) + 𝟙[x≥1]·F(V^dec_(x−1); θ^dec_(x−1))
              + 𝟙[x≥2]·G(V^dec_(x−2); θ^dec_(x−2))

Y para ventanas (Sección 11.1):

    V^{dec,(k)}_0 = Ψ(V^{dec,(k−1)})   para k ≥ 1

Pipeline:
    1. V_dec_0 es recibido externamente (producido por Q(H))
    2. Se propaga a través de X_dec planos usando bloques VGB v1 con θ^dec
    3. Usa MISMA fórmula de propagación que el encoder (F + G aditivos)
    4. Memoria M se inicializa desde zeros (o desde Ψ para k≥1)
    5. Para multi-ventana:
       - Tras cada ventana, Ψ produce V_dec_0_next y M_next
       - El recent_output_summary se computa desde el estado final (mean espacial)

La propagación es EXCLUSIVAMENTE hacia adelante en el eje X.
Los parámetros θ^dec son independientes de θ^enc.

Requisitos implementados: 5.1, 5.2, 5.3, 5.4
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from vsn.core.psi import PsiOperator
from vsn.core.vgb import VGBv1
from vsn.core.vgb_v2 import VGBv2
from vsn.core.vgb_v3 import VGBv3


class VSNDecoder(nn.Module):
    """Decoder volumétrico VSN con generación por ventanas DGW.

    Integra una secuencia de bloques VGB v1 (con parámetros θ^dec
    independientes del encoder) y el operador Ψ para continuidad
    temporal entre ventanas.

    La propagación sigue la misma fórmula que el encoder:
        - F del plano x contribuye aditivamente al plano x+1
        - G del plano x contribuye aditivamente al plano x+2
        - Memoria M se propaga secuencialmente entre planos

    Para generación multi-ventana (DGW):
        - Ventana k=0: V_dec_0 viene de Q(H), M_init = zeros
        - Ventana k≥1: V_dec_0 = Ψ(estado_final_ventana_anterior)
        - Cada ventana produce X_dec planos de propagación

    Args:
        X_dec: Número de planos del decoder (profundidad).
        Y: Alto del plano (filas).
        Z: Ancho del plano (columnas).
        d: Dimensión del embedding por voxel.
        dgw: Tamaño de ventana DGW (Decoder Generation Window).
             Controla cuántos planos se generan por ventana.
             En esta implementación, cada ventana procesa los X_dec planos completos.
    """

    def __init__(self, X_dec: int, Y: int, Z: int, d: int, dgw: int, vgb_version: str = "v1") -> None:
        super().__init__()

        if X_dec <= 0:
            raise ValueError(f"X_dec debe ser positivo, recibido: {X_dec}")
        if Y <= 0:
            raise ValueError(f"Y debe ser positivo, recibido: {Y}")
        if Z <= 0:
            raise ValueError(f"Z debe ser positivo, recibido: {Z}")
        if d <= 0:
            raise ValueError(f"d debe ser positivo, recibido: {d}")
        if dgw <= 0:
            raise ValueError(f"dgw debe ser positivo, recibido: {dgw}")

        self.X_dec = X_dec
        self.Y = Y
        self.Z = Z
        self.d = d
        self.dgw = dgw
        self.vgb_version = vgb_version

        # Bloques VGB del decoder: seleccionar versión
        if vgb_version == "v3":
            spatial_size = Y * Z
            self.vgb_blocks = nn.ModuleList(
                [VGBv3(d, plane_idx=x, spatial_size=spatial_size) for x in range(X_dec)]
            )
        elif vgb_version == "v2":
            spatial_size = Y * Z
            self.vgb_blocks = nn.ModuleList(
                [VGBv2(d, plane_idx=x, spatial_size=spatial_size) for x in range(X_dec)]
            )
        else:
            self.vgb_blocks = nn.ModuleList(
                [VGBv1(d, plane_idx=x) for x in range(X_dec)]
            )

        # Operador Ψ: continuidad temporal entre ventanas
        self.psi = PsiOperator(Y, Z, d)

    def _propagate_single_window(
        self, V_dec_0: Tensor, M_init: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Propaga una ventana completa del decoder sobre X_dec planos.

        Implementa la fórmula de propagación:
            V^dec_x = 𝟙[x=0]·V_dec_0
                    + 𝟙[x≥1]·F(V^dec_(x−1); θ^dec_(x−1))
                    + 𝟙[x≥2]·G(V^dec_(x−2); θ^dec_(x−2))

        Args:
            V_dec_0: Plano inicial del decoder. Shape: (batch, Y, Z, d)
            M_init: Estado inicial de la memoria. Shape: (batch, Y, Z, d)

        Returns:
            Tuple de:
                last_r: Residual del último VGB (estado final de la ventana).
                    Shape: (batch, Y, Z, d)
                M_final: Estado final de la memoria tras propagar todos los planos.
                    Shape: (batch, Y, Z, d)
        """
        # Almacenar F y G producidos por cada VGB
        F_outputs: List[Tensor | None] = [None] * self.X_dec
        G_outputs: List[Tensor | None] = [None] * self.X_dec

        M = M_init
        last_r = V_dec_0  # Fallback si X_dec == 0 (no debería pasar)

        for x in range(self.X_dec):
            # Construir input del plano x
            if x == 0:
                plane_input = V_dec_0
            else:
                # Iniciar desde zeros — las contribuciones F y G son aditivas
                plane_input = V_dec_0.new_zeros(V_dec_0.shape)

                # Contribución F del plano x-1
                if F_outputs[x - 1] is not None:
                    plane_input = plane_input + F_outputs[x - 1]

                # Contribución G del plano x-2 (si x≥2)
                if x >= 2 and G_outputs[x - 2] is not None:
                    plane_input = plane_input + G_outputs[x - 2]

            # Procesar el plano x con su bloque VGB
            F_x, G_x, r_x, M_new = self.vgb_blocks[x](plane_input, M)

            # Almacenar F y G para contribuciones futuras
            F_outputs[x] = F_x
            G_outputs[x] = G_x

            # Guardar residual del último bloque
            last_r = r_x

            # Propagar memoria al siguiente plano
            M = M_new

        return last_r, M

    def forward(
        self, V_dec_0: Tensor, num_windows: int = 1
    ) -> List[Tensor]:
        """Genera num_windows ventanas de decodificación.

        Para la primera ventana (k=0):
            - V_dec_0 proviene de Q(H) (pasado como argumento)
            - M se inicializa como zeros

        Para ventanas k≥1:
            - Ψ produce V_dec_0_next y M_next a partir del estado final
              de la ventana anterior

        Args:
            V_dec_0: Plano inicial del decoder (de Q(H)).
                Shape: (batch, Y, Z, d)
            num_windows: Número de ventanas DGW a generar. Default 1.

        Returns:
            Lista de tensores, uno por ventana. Cada tensor es el
            residual final de esa ventana, shape (batch, Y, Z, d).
            Estos estados se pasan al Head para producir la salida.
        """
        if V_dec_0.ndim != 4:
            raise ValueError(
                f"V_dec_0 debe ser 4D (batch, Y, Z, d), "
                f"recibido ndim={V_dec_0.ndim}, shape={V_dec_0.shape}"
            )

        batch_size = V_dec_0.shape[0]

        # Memoria inicial: zeros para la primera ventana
        M_init = V_dec_0.new_zeros(batch_size, self.Y, self.Z, self.d)

        window_outputs: List[Tensor] = []
        current_V_dec_0 = V_dec_0
        current_M = M_init

        for k in range(num_windows):
            # Propagar ventana k a través de todos los X_dec planos
            last_r, M_final = self._propagate_single_window(
                current_V_dec_0, current_M
            )

            # Almacenar el estado final de esta ventana
            window_outputs.append(last_r)

            # Si hay más ventanas, invocar Ψ para producir estado siguiente
            if k < num_windows - 1:
                # Computar recent_output_summary: media espacial del estado final
                # (batch, Y, Z, d) → (batch, d) via mean sobre Y, Z
                recent_output_summary = last_r.mean(dim=(1, 2))

                # Ψ produce V_dec_0_next y M_next
                current_V_dec_0, current_M = self.psi(
                    decoder_volume_final=last_r,
                    memory_final=M_final,
                    recent_output_summary=recent_output_summary,
                )

        return window_outputs

    def __repr__(self) -> str:
        return (
            f"VSNDecoder(X_dec={self.X_dec}, Y={self.Y}, Z={self.Z}, "
            f"d={self.d}, dgw={self.dgw}, "
            f"num_params={sum(p.numel() for p in self.parameters()):,})"
        )
