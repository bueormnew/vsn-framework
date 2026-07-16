"""RegressionHead — Output Head O para regresión continua.

Implementa la descomposición O:
1. Aggregation A: configurable (default 'last_token') — (batch, Y, Z, d) → (batch, d)
2. Projection W_O: nn.Linear(d, output_dim, bias=True) — (batch, d) → (batch, output_dim)
3. Task function: retorna predicciones continuas (sin activación final)

Registrado como @register_head("regression") para uso en VSNModel.
"""

from __future__ import annotations

from typing import Any, Dict, List

from torch import Tensor, nn

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import BaseHead, build_aggregation, register_head


@register_head("regression")
class RegressionHead(BaseHead):
    """Head de regresión: pooling + proyección lineal continua.

    Sigue la descomposición O:
        aggregation A → projection W_O → predicciones

    El head recibe decoder states como READ-ONLY y produce ModelOutputs
    con embeddings de shape (batch, output_dim). No se aplica activación
    final ya que es responsabilidad del usuario o la función de pérdida.

    Args:
        d: Dimensión de features del decoder (embedding dimension).
        output_dim: Dimensión de la salida de regresión. Default: 1.
        aggregation: Nombre de la función de agregación a usar.
            Opciones: 'last_token', 'mean_pool', 'max_pool', 'cls_token'.
            Default: 'last_token'.
    """

    def __init__(
        self,
        d: int,
        output_dim: int = 1,
        aggregation: str = "last_token",
    ) -> None:
        super().__init__()
        self.d = d
        self.output_dim = output_dim
        self.aggregation_name = aggregation
        self.aggregation = build_aggregation(aggregation)
        # Projection W_O: (batch, d) → (batch, output_dim)
        self.W_O = nn.Linear(d, output_dim, bias=True)

    def forward(
        self,
        decoder_states: List[Tensor],
        batch: MultimodalBatch,
        metadata: Dict[str, Any],
    ) -> ModelOutputs:
        """Procesa estados del decoder y produce predicciones de regresión.

        Pipeline:
            1. Aggregation: List[(batch, Y, Z, d)] → (batch, d)
            2. Projection: (batch, d) → (batch, output_dim)
            3. Return ModelOutputs con embeddings = predictions

        Args:
            decoder_states: Lista de tensores del decoder (uno por ventana DGW).
                Cada tensor tiene shape (batch, Y, Z, d).
            batch: Batch multimodal con targets y metadata de tarea.
            metadata: Metadatos adicionales (e.g. mode, step).

        Returns:
            ModelOutputs con embeddings de shape (batch, output_dim) y metadata
            enriquecida con head_type y aggregation_mode.
        """
        # Step 1: Aggregation A — colapsa dimensiones espaciales
        aggregated = self.aggregation(decoder_states)  # (batch, d)

        # Step 2: Projection W_O — proyecta a espacio de salida continua
        predictions = self.W_O(aggregated)  # (batch, output_dim)

        # Step 3: Enriquecer metadata
        output_metadata = dict(metadata)
        output_metadata["head_type"] = "regression"
        output_metadata["aggregation_mode"] = self.aggregation_name

        return ModelOutputs(embeddings=predictions, metadata=output_metadata)
