"""Operador Ψ — Continuidad temporal entre ventanas del decoder.

Ψ es un operador entrenable que transporta el estado del decoder entre ventanas
DGW sin introducir atención global ni cross-window shortcuts. Acepta el estado
final de la ventana actual y produce el estado inicial de la siguiente.

Componentes:
- volume_summarizer: resumen entrenable del volumen final del decoder
- memory_transform: transformación entrenable del estado de memoria M
- output_summarizer: procesamiento del resumen de tokens/logits recientes
- gate: fusión gated de los tres resúmenes
- state_projector: proyección al espacio V^dec_0 para la siguiente ventana
- memory_projector: proyección al espacio M para la siguiente ventana

Requisitos implementados: 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class PsiOperator(nn.Module):
    """Operador de continuidad temporal entre ventanas del decoder.

    Acepta el estado final de la ventana actual (volumen del decoder,
    memoria M y resumen de salida reciente) y produce el estado inicial
    de la siguiente ventana (V_dec_0_next y M_next).

    Este operador:
    - Es diferenciable (todas las operaciones son estándar de PyTorch)
    - Es serializable via state_dict (es un nn.Module estándar)
    - Preserva semántica de propagación local (no introduce atención global)
    - Es reemplazable por versiones alternativas sin modificar la interfaz del decoder

    Args:
        Y: Alto del plano del decoder.
        Z: Ancho del plano del decoder.
        d: Dimensión de embedding.
    """

    def __init__(self, Y: int, Z: int, d: int) -> None:
        super().__init__()
        self.Y = Y
        self.Z = Z
        self.d = d

        flat_dim = Y * Z * d

        # Resumen entrenable del volumen final del decoder
        # (batch, Y*Z*d) → (batch, d)
        self.volume_summarizer = nn.Linear(flat_dim, d)

        # Transformación entrenable del estado de memoria M
        # (batch, Y*Z*d) → (batch, d)
        self.memory_transform = nn.Linear(flat_dim, d)

        # Procesamiento del resumen de salida reciente
        # (batch, d) → (batch, d)
        self.output_summarizer = nn.Linear(d, d)

        # Gate: fusión de los tres resúmenes con activación sigmoidal
        # (batch, 3*d) → (batch, d)
        self.gate = nn.Linear(3 * d, d)

        # Proyección al espacio V^dec_0 para la siguiente ventana
        # (batch, d) → (batch, Y*Z*d) → reshape a (batch, Y, Z, d)
        self.state_projector = nn.Linear(d, flat_dim)

        # Proyección al espacio M para la siguiente ventana
        # (batch, d) → (batch, Y*Z*d) → reshape a (batch, Y, Z, d)
        self.memory_projector = nn.Linear(d, flat_dim)

    def forward(
        self,
        decoder_volume_final: Tensor,
        memory_final: Tensor,
        recent_output_summary: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Produce el estado inicial de la siguiente ventana del decoder.

        Args:
            decoder_volume_final: Estado final del volumen del decoder.
                Shape: (batch, Y, Z, d)
            memory_final: Estado final de la memoria M del decoder.
                Shape: (batch, Y, Z, d)
            recent_output_summary: Resumen de los tokens/logits generados
                en la ventana actual. Shape: (batch, d)

        Returns:
            Tuple de:
                V_dec_0_next: Estado inicial del volumen para la siguiente ventana.
                    Shape: (batch, Y, Z, d)
                M_next: Estado inicial de la memoria para la siguiente ventana.
                    Shape: (batch, Y, Z, d)
        """
        batch_size = decoder_volume_final.shape[0]

        # Flatten volúmenes: (batch, Y, Z, d) → (batch, Y*Z*d)
        vol_flat = decoder_volume_final.reshape(batch_size, -1)
        mem_flat = memory_final.reshape(batch_size, -1)

        # Resúmenes: cada uno produce (batch, d)
        vol_summary = self.volume_summarizer(vol_flat)
        mem_summary = self.memory_transform(mem_flat)
        out_summary = self.output_summarizer(recent_output_summary)

        # Gate: concatenar resúmenes y aplicar sigmoide
        # (batch, 3*d) → sigmoid → (batch, d)
        combined = torch.cat([vol_summary, mem_summary, out_summary], dim=-1)
        gate_value = torch.sigmoid(self.gate(combined))

        # Fusión gated: combinación ponderada de resúmenes
        # El gate modula la mezcla de la información destilada
        fused = gate_value * vol_summary + (1 - gate_value) * mem_summary

        # Proyectar al espacio V^dec_0: (batch, d) → (batch, Y*Z*d) → (batch, Y, Z, d)
        v_dec_0_next = self.state_projector(fused)
        v_dec_0_next = v_dec_0_next.reshape(batch_size, self.Y, self.Z, self.d)

        # Proyectar al espacio M: (batch, d) → (batch, Y*Z*d) → (batch, Y, Z, d)
        m_next = self.memory_projector(fused)
        m_next = m_next.reshape(batch_size, self.Y, self.Z, self.d)

        return v_dec_0_next, m_next
