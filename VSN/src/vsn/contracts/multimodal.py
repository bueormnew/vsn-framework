"""Contratos multimodales para la arquitectura VSN.

Define las estructuras de datos tipadas (ModalityTensor, MultimodalBatch) y sus
validadores que garantizan compatibilidad de shapes, dtypes y consistencia entre
modalidades antes de ejecutar cualquier compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------


class ContractError(Exception):
    """Error lanzado cuando un contrato multimodal es violado.

    Incluye contexto descriptivo sobre qué dimensión, dtype o campo es
    incompatible para facilitar el diagnóstico sin necesidad de debugger.
    """

    pass


# ---------------------------------------------------------------------------
# Dataclasses de contrato
# ---------------------------------------------------------------------------


@dataclass
class ModalityTensor:
    """Tensor tipado que representa una modalidad de entrada.

    Atributos:
        values: Tensor de valores — shape (batch, seq_len, d) o (batch, Y, Z, d).
        mask: Tensor booleano de máscara — shape (batch, seq_len) o (batch, Y, Z).
        lengths: Tensor de longitudes por ejemplo en el batch — shape (batch,), int64.
        positions: Tensor de posiciones — shape (batch, seq_len, pos_dims).
        metadata: Diccionario con información adicional (nombre de modalidad,
            dtype original, etc.).
    """

    values: Tensor
    mask: Tensor
    lengths: Tensor
    positions: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalBatch:
    """Batch multimodal que agrupa múltiples modalidades con targets opcionales.

    Atributos:
        modalities: Diccionario nombre→ModalityTensor para cada modalidad presente.
        task: Identificador de la tarea (e.g. 'text', 'classification').
        targets: Tensor de targets opcionales para la pérdida.
        target_mask: Máscara booleana sobre targets (posiciones válidas).
    """

    modalities: Dict[str, ModalityTensor]
    task: str
    targets: Optional[Tensor] = None
    target_mask: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# Validación de ModalityTensor
# ---------------------------------------------------------------------------


def validate_modality_tensor(mt: ModalityTensor) -> None:
    """Valida que un ModalityTensor tiene shapes, dtypes y campos compatibles.

    Verificaciones realizadas (fail-fast — se detiene en el primer error):
        1. Todos los campos requeridos son tensores de PyTorch.
        2. ``values`` tiene al menos 3 dimensiones (batch, ..., d).
        3. ``mask`` es booleano y su shape coincide con las dims espaciales de values.
        4. ``lengths`` es int64, 1-D y su primer dim coincide con batch de values.
        5. ``positions`` tiene batch_size consistente con values.
        6. ``metadata`` es un diccionario.

    Args:
        mt: ModalityTensor a validar.

    Raises:
        ContractError: Si alguna validación falla, con mensaje descriptivo.
    """

    # 1. Verificar que los campos son tensores
    _assert_is_tensor(mt.values, "values")
    _assert_is_tensor(mt.mask, "mask")
    _assert_is_tensor(mt.lengths, "lengths")
    _assert_is_tensor(mt.positions, "positions")

    # 2. values debe tener al menos 3 dimensiones: (batch, ..., d)
    if mt.values.ndim < 3:
        raise ContractError(
            f"ModalityTensor.values debe tener al menos 3 dimensiones "
            f"(batch, seq_len, d) o (batch, Y, Z, d), pero tiene ndim={mt.values.ndim}."
        )

    batch_size = mt.values.shape[0]

    # 3. mask — debe ser booleano y tener shape espacial compatible
    if mt.mask.dtype != torch.bool:
        raise ContractError(
            f"ModalityTensor.mask debe ser dtype=torch.bool, "
            f"pero tiene dtype={mt.mask.dtype}."
        )

    # mask shape debe ser (batch, seq_len) o (batch, Y, Z) — es decir,
    # las dimensiones de values sin la última (d).
    expected_mask_shape = mt.values.shape[:-1]
    if mt.mask.shape != expected_mask_shape:
        raise ContractError(
            f"ModalityTensor.mask shape incompatible: esperado {tuple(expected_mask_shape)} "
            f"(values.shape[:-1]), pero obtuvo {tuple(mt.mask.shape)}."
        )

    # 4. lengths — int64, 1-D, batch_size coincide
    if mt.lengths.dtype != torch.int64:
        raise ContractError(
            f"ModalityTensor.lengths debe ser dtype=torch.int64, "
            f"pero tiene dtype={mt.lengths.dtype}."
        )
    if mt.lengths.ndim != 1:
        raise ContractError(
            f"ModalityTensor.lengths debe ser 1-D (batch,), "
            f"pero tiene ndim={mt.lengths.ndim}."
        )
    if mt.lengths.shape[0] != batch_size:
        raise ContractError(
            f"ModalityTensor.lengths batch_size={mt.lengths.shape[0]} "
            f"no coincide con values batch_size={batch_size}."
        )

    # 5. positions — batch_size consistente
    if mt.positions.ndim < 2:
        raise ContractError(
            f"ModalityTensor.positions debe tener al menos 2 dimensiones "
            f"(batch, seq_len, pos_dims), pero tiene ndim={mt.positions.ndim}."
        )
    if mt.positions.shape[0] != batch_size:
        raise ContractError(
            f"ModalityTensor.positions batch_size={mt.positions.shape[0]} "
            f"no coincide con values batch_size={batch_size}."
        )

    # Para tensores de 3 dims (batch, seq_len, d), positions debe ser (batch, seq_len, pos_dims)
    # Para tensores de 4 dims (batch, Y, Z, d), positions puede ser (batch, Y*Z, pos_dims)
    # En ambos casos, la cantidad de posiciones debe cubrir los elementos espaciales
    if mt.values.ndim == 3:
        # (batch, seq_len, d) -> positions (batch, seq_len, pos_dims)
        seq_len = mt.values.shape[1]
        if mt.positions.ndim != 3:
            raise ContractError(
                f"ModalityTensor.positions debe ser 3-D (batch, seq_len, pos_dims) "
                f"cuando values es 3-D, pero tiene ndim={mt.positions.ndim}."
            )
        if mt.positions.shape[1] != seq_len:
            raise ContractError(
                f"ModalityTensor.positions seq_len={mt.positions.shape[1]} "
                f"no coincide con values seq_len={seq_len}."
            )

    # 6. metadata debe ser diccionario
    if not isinstance(mt.metadata, dict):
        raise ContractError(
            f"ModalityTensor.metadata debe ser un dict, "
            f"pero es de tipo {type(mt.metadata).__name__}."
        )


# ---------------------------------------------------------------------------
# Validación de MultimodalBatch
# ---------------------------------------------------------------------------


def validate_multimodal_batch(batch: MultimodalBatch) -> None:
    """Valida consistencia de un MultimodalBatch completo.

    Verificaciones realizadas (fail-fast):
        1. ``modalities`` no está vacío y es un diccionario.
        2. Cada ModalityTensor individual es válido (via validate_modality_tensor).
        3. Todos los ModalityTensors tienen el mismo batch_size.
        4. ``task`` es un string no vacío.
        5. Si ``targets`` está presente, su batch_size coincide con las modalidades.
        6. Si ``target_mask`` está presente, es booleano y su shape es compatible con targets.

    Args:
        batch: MultimodalBatch a validar.

    Raises:
        ContractError: Si alguna validación falla, con mensaje descriptivo.
    """

    # 1. modalities no vacío y es dict
    if not isinstance(batch.modalities, dict):
        raise ContractError(
            f"MultimodalBatch.modalities debe ser un dict, "
            f"pero es de tipo {type(batch.modalities).__name__}."
        )
    if len(batch.modalities) == 0:
        raise ContractError(
            "MultimodalBatch.modalities no puede estar vacío — "
            "se requiere al menos una modalidad."
        )

    # 2. Validar cada ModalityTensor individual
    for name, mt in batch.modalities.items():
        if not isinstance(mt, ModalityTensor):
            raise ContractError(
                f"MultimodalBatch.modalities['{name}'] debe ser ModalityTensor, "
                f"pero es de tipo {type(mt).__name__}."
            )
        try:
            validate_modality_tensor(mt)
        except ContractError as e:
            raise ContractError(
                f"MultimodalBatch.modalities['{name}'] inválido: {e}"
            ) from e

    # 3. Consistencia de batch_size entre modalidades
    batch_sizes: Dict[str, int] = {}
    for name, mt in batch.modalities.items():
        batch_sizes[name] = mt.values.shape[0]

    unique_sizes = set(batch_sizes.values())
    if len(unique_sizes) > 1:
        details = ", ".join(f"'{k}'={v}" for k, v in batch_sizes.items())
        raise ContractError(
            f"MultimodalBatch tiene batch_sizes inconsistentes entre modalidades: "
            f"{details}. Todas las modalidades deben tener el mismo batch_size."
        )

    reference_batch_size = next(iter(batch_sizes.values()))

    # 4. task es string no vacío
    if not isinstance(batch.task, str) or len(batch.task.strip()) == 0:
        raise ContractError(
            "MultimodalBatch.task debe ser un string no vacío."
        )

    # 5. targets — si presente, batch_size consistente
    if batch.targets is not None:
        _assert_is_tensor(batch.targets, "targets")
        if batch.targets.shape[0] != reference_batch_size:
            raise ContractError(
                f"MultimodalBatch.targets batch_size={batch.targets.shape[0]} "
                f"no coincide con el batch_size de las modalidades={reference_batch_size}."
            )

    # 6. target_mask — si presente, booleano y compatible con targets
    if batch.target_mask is not None:
        _assert_is_tensor(batch.target_mask, "target_mask")
        if batch.target_mask.dtype != torch.bool:
            raise ContractError(
                f"MultimodalBatch.target_mask debe ser dtype=torch.bool, "
                f"pero tiene dtype={batch.target_mask.dtype}."
            )
        if batch.targets is None:
            raise ContractError(
                "MultimodalBatch.target_mask está presente pero targets es None. "
                "target_mask requiere que targets esté definido."
            )
        if batch.target_mask.shape != batch.targets.shape:
            raise ContractError(
                f"MultimodalBatch.target_mask shape={tuple(batch.target_mask.shape)} "
                f"no coincide con targets shape={tuple(batch.targets.shape)}."
            )


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _assert_is_tensor(obj: Any, field_name: str) -> None:
    """Verifica que un objeto es un Tensor de PyTorch."""
    if not isinstance(obj, Tensor):
        raise ContractError(
            f"Se esperaba un torch.Tensor para '{field_name}', "
            f"pero se recibió {type(obj).__name__}."
        )
