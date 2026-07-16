"""Feature: vsn-library, Property 1: Validación de contratos acepta válidos y rechaza inválidos

**Validates: Requirements 1.3, 1.4**

Tests de propiedad que verifican que:
- ModalityTensors válidos (shapes, dtypes, masks compatibles) pasan validación.
- ModalityTensors inválidos (wrong dtypes, wrong shapes, dims incompatibles) son rechazados con ContractError.
- MultimodalBatch válidos (batch_sizes consistentes, modalidades no vacías) pasan validación.
- MultimodalBatch inválidos (batch_sizes inconsistentes, modalidades vacías) son rechazados con ContractError.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.contracts import (
    ContractError,
    ModalityTensor,
    MultimodalBatch,
    validate_modality_tensor,
    validate_multimodal_batch,
)

# ---------------------------------------------------------------------------
# Estrategias para generar datos válidos
# ---------------------------------------------------------------------------

# Dimensiones razonables para tests rápidos
dim_strategy = st.integers(min_value=1, max_value=8)
batch_strategy = st.integers(min_value=1, max_value=4)
pos_dims_strategy = st.integers(min_value=1, max_value=3)


@st.composite
def valid_modality_tensor_3d(draw: st.DrawFn) -> ModalityTensor:
    """Genera un ModalityTensor válido con values de 3 dimensiones (batch, seq_len, d)."""
    batch = draw(batch_strategy)
    seq_len = draw(dim_strategy)
    d = draw(dim_strategy)
    pos_dims = draw(pos_dims_strategy)

    values = torch.randn(batch, seq_len, d)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    lengths = torch.full((batch,), seq_len, dtype=torch.int64)
    positions = torch.randn(batch, seq_len, pos_dims)
    metadata = {"modality": "test", "dtype_original": "float32"}

    return ModalityTensor(
        values=values,
        mask=mask,
        lengths=lengths,
        positions=positions,
        metadata=metadata,
    )


@st.composite
def valid_modality_tensor_4d(draw: st.DrawFn) -> ModalityTensor:
    """Genera un ModalityTensor válido con values de 4 dimensiones (batch, Y, Z, d)."""
    batch = draw(batch_strategy)
    y = draw(dim_strategy)
    z = draw(dim_strategy)
    d = draw(dim_strategy)

    values = torch.randn(batch, y, z, d)
    mask = torch.ones(batch, y, z, dtype=torch.bool)
    lengths = torch.full((batch,), y * z, dtype=torch.int64)
    # Para 4D, positions no se valida estrictamente en seq_len,
    # solo se requiere batch_size consistente y ndim >= 2
    positions = torch.randn(batch, y * z, 2)
    metadata = {"modality": "volumetric"}

    return ModalityTensor(
        values=values,
        mask=mask,
        lengths=lengths,
        positions=positions,
        metadata=metadata,
    )


@st.composite
def valid_multimodal_batch(draw: st.DrawFn) -> MultimodalBatch:
    """Genera un MultimodalBatch válido con 1 a 3 modalidades consistentes."""
    num_modalities = draw(st.integers(min_value=1, max_value=3))
    batch = draw(batch_strategy)

    modalities = {}
    for i in range(num_modalities):
        name = f"modality_{i}"
        seq_len = draw(dim_strategy)
        d = draw(dim_strategy)
        pos_dims = draw(pos_dims_strategy)

        values = torch.randn(batch, seq_len, d)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        lengths = torch.full((batch,), seq_len, dtype=torch.int64)
        positions = torch.randn(batch, seq_len, pos_dims)
        metadata = {"modality": name}

        modalities[name] = ModalityTensor(
            values=values,
            mask=mask,
            lengths=lengths,
            positions=positions,
            metadata=metadata,
        )

    task = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("L",))))

    return MultimodalBatch(
        modalities=modalities,
        task=task,
        targets=None,
        target_mask=None,
    )


# ---------------------------------------------------------------------------
# Estrategias para generar datos inválidos
# ---------------------------------------------------------------------------


@st.composite
def modality_tensor_wrong_mask_dtype(draw: st.DrawFn) -> ModalityTensor:
    """ModalityTensor con mask de dtype incorrecto (no bool)."""
    batch = draw(batch_strategy)
    seq_len = draw(dim_strategy)
    d = draw(dim_strategy)

    values = torch.randn(batch, seq_len, d)
    # mask con dtype float en lugar de bool
    mask = torch.ones(batch, seq_len, dtype=torch.float32)
    lengths = torch.full((batch,), seq_len, dtype=torch.int64)
    positions = torch.randn(batch, seq_len, 2)

    return ModalityTensor(
        values=values, mask=mask, lengths=lengths, positions=positions, metadata={}
    )


@st.composite
def modality_tensor_wrong_mask_shape(draw: st.DrawFn) -> ModalityTensor:
    """ModalityTensor con mask de shape incompatible con values."""
    batch = draw(batch_strategy)
    seq_len = draw(dim_strategy)
    d = draw(dim_strategy)
    # Generar un offset distinto de 0 para que la shape sea diferente
    offset = draw(st.integers(min_value=1, max_value=4))

    values = torch.randn(batch, seq_len, d)
    # mask con seq_len diferente
    mask = torch.ones(batch, seq_len + offset, dtype=torch.bool)
    lengths = torch.full((batch,), seq_len, dtype=torch.int64)
    positions = torch.randn(batch, seq_len, 2)

    return ModalityTensor(
        values=values, mask=mask, lengths=lengths, positions=positions, metadata={}
    )


@st.composite
def modality_tensor_wrong_lengths_dtype(draw: st.DrawFn) -> ModalityTensor:
    """ModalityTensor con lengths de dtype incorrecto (no int64)."""
    batch = draw(batch_strategy)
    seq_len = draw(dim_strategy)
    d = draw(dim_strategy)

    values = torch.randn(batch, seq_len, d)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    # lengths con float32 en vez de int64
    lengths = torch.full((batch,), seq_len, dtype=torch.float32)
    positions = torch.randn(batch, seq_len, 2)

    return ModalityTensor(
        values=values, mask=mask, lengths=lengths, positions=positions, metadata={}
    )


@st.composite
def modality_tensor_values_too_few_dims(draw: st.DrawFn) -> ModalityTensor:
    """ModalityTensor con values de menos de 3 dimensiones."""
    batch = draw(batch_strategy)
    d = draw(dim_strategy)

    # Solo 2 dimensiones — inválido
    values = torch.randn(batch, d)
    mask = torch.ones(batch, dtype=torch.bool)
    lengths = torch.full((batch,), 1, dtype=torch.int64)
    positions = torch.randn(batch, 1)

    return ModalityTensor(
        values=values, mask=mask, lengths=lengths, positions=positions, metadata={}
    )


@st.composite
def modality_tensor_inconsistent_batch_size(draw: st.DrawFn) -> ModalityTensor:
    """ModalityTensor con lengths.batch_size distinto de values.batch_size."""
    batch = draw(batch_strategy)
    seq_len = draw(dim_strategy)
    d = draw(dim_strategy)
    # Offset para batch diferente
    offset = draw(st.integers(min_value=1, max_value=3))

    values = torch.randn(batch, seq_len, d)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    # lengths con batch_size diferente
    lengths = torch.full((batch + offset,), seq_len, dtype=torch.int64)
    positions = torch.randn(batch, seq_len, 2)

    return ModalityTensor(
        values=values, mask=mask, lengths=lengths, positions=positions, metadata={}
    )


@st.composite
def multimodal_batch_inconsistent_batch_sizes(draw: st.DrawFn) -> MultimodalBatch:
    """MultimodalBatch donde las modalidades tienen batch_sizes distintos."""
    d = draw(dim_strategy)
    seq_len = draw(dim_strategy)
    batch_a = draw(st.integers(min_value=1, max_value=4))
    # batch_b diferente de batch_a
    batch_b = draw(st.integers(min_value=1, max_value=4).filter(lambda x: x != batch_a))

    def make_mt(batch: int) -> ModalityTensor:
        return ModalityTensor(
            values=torch.randn(batch, seq_len, d),
            mask=torch.ones(batch, seq_len, dtype=torch.bool),
            lengths=torch.full((batch,), seq_len, dtype=torch.int64),
            positions=torch.randn(batch, seq_len, 2),
            metadata={},
        )

    modalities = {
        "mod_a": make_mt(batch_a),
        "mod_b": make_mt(batch_b),
    }

    return MultimodalBatch(modalities=modalities, task="test")


@st.composite
def multimodal_batch_empty_modalities(draw: st.DrawFn) -> MultimodalBatch:
    """MultimodalBatch con diccionario de modalidades vacío."""
    task = draw(st.text(min_size=1, max_size=5, alphabet=st.characters(categories=("L",))))
    return MultimodalBatch(modalities={}, task=task)


# ---------------------------------------------------------------------------
# Tests de propiedad: ModalityTensor válido pasa validación
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(mt=valid_modality_tensor_3d())
def test_valid_modality_tensor_3d_passes_validation(mt: ModalityTensor) -> None:
    """Valid ModalityTensors (3D) con shapes y dtypes correctos pasan validación."""
    # No debe lanzar excepción
    validate_modality_tensor(mt)


@settings(max_examples=100, deadline=None)
@given(mt=valid_modality_tensor_4d())
def test_valid_modality_tensor_4d_passes_validation(mt: ModalityTensor) -> None:
    """Valid ModalityTensors (4D) con shapes y dtypes correctos pasan validación."""
    validate_modality_tensor(mt)


# ---------------------------------------------------------------------------
# Tests de propiedad: ModalityTensor inválido es rechazado con ContractError
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(mt=modality_tensor_wrong_mask_dtype())
def test_invalid_modality_tensor_wrong_mask_dtype_rejected(mt: ModalityTensor) -> None:
    """ModalityTensor con mask dtype incorrecto es rechazado con ContractError."""
    try:
        validate_modality_tensor(mt)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado


@settings(max_examples=100, deadline=None)
@given(mt=modality_tensor_wrong_mask_shape())
def test_invalid_modality_tensor_wrong_mask_shape_rejected(mt: ModalityTensor) -> None:
    """ModalityTensor con mask shape incompatible es rechazado con ContractError."""
    try:
        validate_modality_tensor(mt)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado


@settings(max_examples=100, deadline=None)
@given(mt=modality_tensor_wrong_lengths_dtype())
def test_invalid_modality_tensor_wrong_lengths_dtype_rejected(mt: ModalityTensor) -> None:
    """ModalityTensor con lengths dtype incorrecto es rechazado con ContractError."""
    try:
        validate_modality_tensor(mt)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado


@settings(max_examples=100, deadline=None)
@given(mt=modality_tensor_values_too_few_dims())
def test_invalid_modality_tensor_too_few_dims_rejected(mt: ModalityTensor) -> None:
    """ModalityTensor con values de < 3 dimensiones es rechazado con ContractError."""
    try:
        validate_modality_tensor(mt)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado


@settings(max_examples=100, deadline=None)
@given(mt=modality_tensor_inconsistent_batch_size())
def test_invalid_modality_tensor_inconsistent_batch_rejected(mt: ModalityTensor) -> None:
    """ModalityTensor con batch_size inconsistente es rechazado con ContractError."""
    try:
        validate_modality_tensor(mt)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado


# ---------------------------------------------------------------------------
# Tests de propiedad: MultimodalBatch válido pasa validación
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(batch=valid_multimodal_batch())
def test_valid_multimodal_batch_passes_validation(batch: MultimodalBatch) -> None:
    """MultimodalBatch válido (batch_sizes consistentes, no vacío) pasa validación."""
    validate_multimodal_batch(batch)


# ---------------------------------------------------------------------------
# Tests de propiedad: MultimodalBatch inválido es rechazado con ContractError
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(batch=multimodal_batch_inconsistent_batch_sizes())
def test_invalid_multimodal_batch_inconsistent_batch_sizes_rejected(
    batch: MultimodalBatch,
) -> None:
    """MultimodalBatch con batch_sizes inconsistentes entre modalidades es rechazado."""
    try:
        validate_multimodal_batch(batch)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado


@settings(max_examples=100, deadline=None)
@given(batch=multimodal_batch_empty_modalities())
def test_invalid_multimodal_batch_empty_modalities_rejected(
    batch: MultimodalBatch,
) -> None:
    """MultimodalBatch con modalidades vacías es rechazado con ContractError."""
    try:
        validate_multimodal_batch(batch)
        assert False, "Se esperaba ContractError pero la validación pasó"
    except ContractError:
        pass  # Esperado
