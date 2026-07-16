"""ClassificationHead — Output Head O para clasificación.

Implementa la descomposición O:
1. Aggregation A: configurable (default 'last_token') — (batch, Y, Z, d) → (batch, d)
2. Projection W_O: nn.Linear(d, num_classes, bias=True) — (batch, d) → (batch, num_classes)
3. Task function: retorna logits (softmax es implícita vía loss)

Registrado como @register_head("classification") para uso en VSNModel.
"""

from __future__ import annotations

from typing import Any, Dict, List

from torch import Tensor, nn

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import BaseHead, build_aggregation, register_head


@register_head("classification")
class ClassificationHead(BaseHead):
    """Head de clasificación: pooling + proyección lineal a clases.

    Sigue la descomposición O:
        aggregation A → projection W_O → logits

    El head recibe decoder states como READ-ONLY y produce ModelOutputs
    con logits de shape (batch, num_classes). El softmax no se aplica aquí
    ya que es responsabilidad de la función de pérdida (CrossEntropy).

    Args:
        d: Dimensión de features del decoder (embedding dimension).
        num_classes: Número de clases de salida.
        aggregation: Nombre de la función de agregación a usar.
            Opciones: 'last_token', 'mean_pool', 'max_pool', 'cls_token'.
            Default: 'last_token'.
    """

    def __init__(
        self,
        d: int,
        num_classes: int,
        aggregation: str = "last_token",
    ) -> None:
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.aggregation_name = aggregation
        self.aggregation = build_aggregation(aggregation)
        # Projection W_O: (batch, d) → (batch, num_classes)
        self.W_O = nn.Linear(d, num_classes, bias=True)

    def forward(
        self,
        decoder_states: List[Tensor],
        batch: MultimodalBatch,
        metadata: Dict[str, Any],
    ) -> ModelOutputs:
        """Procesa estados del decoder y produce logits de clasificación.

        Pipeline:
            1. Aggregation: List[(batch, Y, Z, d)] → (batch, d)
            2. Projection: (batch, d) → (batch, num_classes)
            3. Return ModelOutputs con logits

        Args:
            decoder_states: Lista de tensores del decoder (uno por ventana DGW).
                Cada tensor tiene shape (batch, Y, Z, d).
            batch: Batch multimodal con targets y metadata de tarea.
            metadata: Metadatos adicionales (e.g. mode, step).

        Returns:
            ModelOutputs con logits de shape (batch, num_classes) y metadata
            enriquecida con head_type y aggregation_mode.
        """
        # Step 1: Aggregation A — colapsa dimensiones espaciales
        aggregated = self.aggregation(decoder_states)  # (batch, d)

        # Step 2: Projection W_O — proyecta a espacio de clases
        logits = self.W_O(aggregated)  # (batch, num_classes)

        # Step 3: Enriquecer metadata
        output_metadata = dict(metadata)
        output_metadata["head_type"] = "classification"
        output_metadata["aggregation_mode"] = self.aggregation_name

        return ModelOutputs(logits=logits, metadata=output_metadata)
