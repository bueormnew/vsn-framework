"""DenseHead — Output Head O para predicción per-voxel (densa).

Implementa la descomposición O:
1. Aggregation A: NINGUNA — opera directamente sobre el último window
2. Projection W_O: nn.Linear(d, output_dim, bias=True) — (batch, Y, Z, d) → (batch, Y, Z, output_dim)
3. Task function: retorna salida densa preservando estructura espacial

Registrado como @register_head("dense") para uso en VSNModel.
"""

from __future__ import annotations

from typing import Any, Dict, List

from torch import Tensor, nn

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import BaseHead, register_head


@register_head("dense")
class DenseHead(BaseHead):
    """Head denso: proyección per-voxel sin colapsar estructura espacial.

    A diferencia de otros heads, DenseHead NO aplica agregación.
    Opera directamente sobre el último window del decoder, aplicando
    una proyección lineal a cada posición (y, z) independientemente.

    Sigue la descomposición O:
        (sin aggregation) → projection W_O per-voxel → salida densa

    El head recibe decoder states como READ-ONLY y produce ModelOutputs
    con embeddings de shape (batch, Y, Z, output_dim), preservando la
    estructura espacial del volumen.

    Args:
        d: Dimensión de features del decoder (embedding dimension).
        output_dim: Dimensión de salida por voxel.
    """

    def __init__(
        self,
        d: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.d = d
        self.output_dim = output_dim
        # Projection W_O: (batch, Y, Z, d) → (batch, Y, Z, output_dim)
        # nn.Linear opera sobre la última dimensión, así que se aplica per-voxel
        self.W_O = nn.Linear(d, output_dim, bias=True)

    def forward(
        self,
        decoder_states: List[Tensor],
        batch: MultimodalBatch,
        metadata: Dict[str, Any],
    ) -> ModelOutputs:
        """Procesa estados del decoder y produce salida densa per-voxel.

        Pipeline:
            1. Selecciona el último window (sin agregación)
            2. Projection per-voxel: (batch, Y, Z, d) → (batch, Y, Z, output_dim)
            3. Return ModelOutputs con embeddings preservando estructura espacial

        Args:
            decoder_states: Lista de tensores del decoder (uno por ventana DGW).
                Cada tensor tiene shape (batch, Y, Z, d).
            batch: Batch multimodal con targets y metadata de tarea.
            metadata: Metadatos adicionales (e.g. mode, step).

        Returns:
            ModelOutputs con embeddings de shape (batch, Y, Z, output_dim) y
            metadata enriquecida con head_type.
        """
        # Step 1: Tomar el último window — sin colapsar dimensiones espaciales
        last_window = decoder_states[-1]  # (batch, Y, Z, d)

        # Step 2: Projection W_O per-voxel — Linear opera sobre última dim
        dense_output = self.W_O(last_window)  # (batch, Y, Z, output_dim)

        # Step 3: Enriquecer metadata
        output_metadata = dict(metadata)
        output_metadata["head_type"] = "dense"

        return ModelOutputs(embeddings=dense_output, metadata=output_metadata)
