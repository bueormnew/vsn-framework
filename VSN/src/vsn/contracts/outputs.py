"""Contrato de salidas para la arquitectura VSN.

Define la estructura tipada ModelOutputs que encapsula todos los campos
de salida que cualquier head o modo puede producir. Todos los campos son
opcionales excepto aux_losses y metadata que tienen defaults vacíos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor


@dataclass
class ModelOutputs:
    """Estructura de salida unificada del modelo VSN.

    Encapsula todos los posibles campos de salida que cualquier Output_Head_O
    o modo de inferencia puede producir. Los campos opcionales permiten que
    distintos heads devuelvan solo los campos relevantes a su tarea.

    Atributos:
        logits: Tensor de logits de salida — shape típica (batch, seq_len, vocab_size)
            para TextHead o (batch, num_classes) para ClassificationHead.
            None si el head no produce logits (e.g. RegressionHead).
        embeddings: Tensor de embeddings producidos por el head — shape típica
            (batch, d) o (batch, seq_len, d). None si el head no los genera.
        aux_losses: Diccionario de pérdidas auxiliares computadas durante el
            forward (e.g. regularización, sparsity). Las claves identifican
            cada pérdida auxiliar. Default: dict vacío.
        states: Diccionario opcional de estados intermedios para inspección
            o reutilización (e.g. decoder final state, latent H). None si
            no se solicitan estados.
        metadata: Diccionario de metadatos asociados a la salida (e.g. head_type,
            task, aggregation_mode, timestamps). Default: dict vacío.
    """

    logits: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    aux_losses: Dict[str, Tensor] = field(default_factory=dict)
    states: Optional[Dict[str, Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
