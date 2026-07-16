"""Feature: vsn-library, Property 13: Head O no modifica estado del decoder

**Validates: Requirements 7.1, 7.2, 7.5**

Tests de propiedad que verifican:
- Para cualquier Output_Head_O (TextHead, ClassificationHead, RegressionHead, DenseHead)
  y para cualquier decoder_state tensor, después de ejecutar head.forward(), el tensor
  decoder_state SHALL permanecer idéntico bit a bit al valor previo a la invocación.
- El head SHALL seguir la descomposición: aggregation → projection → task function.
- La salida es un ModelOutputs válido con los campos apropiados.
"""

from __future__ import annotations

from typing import List

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from vsn.contracts.multimodal import ModalityTensor, MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import BaseHead
from vsn.heads.classification import ClassificationHead
from vsn.heads.dense import DenseHead
from vsn.heads.regression import RegressionHead
from vsn.heads.text import TextHead

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones y estados válidos
# ---------------------------------------------------------------------------

batch_strategy = st.integers(min_value=1, max_value=4)
spatial_strategy = st.integers(min_value=1, max_value=6)
d_strategy = st.integers(min_value=2, max_value=16)
num_windows_strategy = st.integers(min_value=1, max_value=3)
vocab_strategy = st.integers(min_value=2, max_value=50)
num_classes_strategy = st.integers(min_value=2, max_value=20)
output_dim_strategy = st.integers(min_value=1, max_value=16)


@st.composite
def head_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para los heads."""
    B = draw(batch_strategy)
    Y = draw(spatial_strategy)
    Z = draw(spatial_strategy)
    d = draw(d_strategy)
    num_windows = draw(num_windows_strategy)
    vocab_size = draw(vocab_strategy)
    num_classes = draw(num_classes_strategy)
    output_dim = draw(output_dim_strategy)
    return {
        "batch": B,
        "Y": Y,
        "Z": Z,
        "d": d,
        "num_windows": num_windows,
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "output_dim": output_dim,
    }


def _make_decoder_states(B: int, Y: int, Z: int, d: int, num_windows: int) -> List[Tensor]:
    """Genera lista de decoder states aleatorios."""
    return [torch.randn(B, Y, Z, d) for _ in range(num_windows)]


def _make_dummy_batch(B: int, d: int) -> MultimodalBatch:
    """Crea un MultimodalBatch dummy válido para pasar a los heads."""
    seq_len = 4
    values = torch.randn(B, seq_len, d)
    mask = torch.ones(B, seq_len, dtype=torch.bool)
    lengths = torch.full((B,), seq_len, dtype=torch.int64)
    positions = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).expand(B, seq_len, 1).float()

    mt = ModalityTensor(
        values=values,
        mask=mask,
        lengths=lengths,
        positions=positions,
        metadata={"modality": "text"},
    )
    return MultimodalBatch(
        modalities={"text": mt},
        task="text",
    )


def _clone_decoder_states(states: List[Tensor]) -> List[Tensor]:
    """Clona profundamente la lista de decoder states."""
    return [s.clone() for s in states]


# ---------------------------------------------------------------------------
# Propiedad 13: TextHead — no modifica estado del decoder
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=head_dims())
def test_text_head_does_not_modify_decoder_state(dims: dict) -> None:
    """Feature: vsn-library, Property 13: Head O no modifica estado del decoder (TextHead)

    **Validates: Requirements 7.1, 7.2, 7.5**

    Para cualquier TextHead y decoder_state, después de forward(), los tensores
    decoder_state SHALL permanecer idénticos bit a bit. La salida SHALL ser
    ModelOutputs con logits de shape (batch, vocab_size).
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]
    num_windows = dims["num_windows"]
    vocab_size = dims["vocab_size"]

    head = TextHead(d=d, vocab_size=vocab_size, aggregation="last_token")
    head.eval()

    decoder_states = _make_decoder_states(B, Y, Z, d, num_windows)
    cloned_states = _clone_decoder_states(decoder_states)
    batch = _make_dummy_batch(B, d)
    metadata: dict = {"mode": "eval"}

    with torch.no_grad():
        output = head(decoder_states, batch, metadata)

    # 1. Verificar que decoder_states no fue modificado
    assert len(decoder_states) == len(cloned_states), (
        "La cantidad de decoder states cambió después de forward()"
    )
    for i, (original, cloned) in enumerate(zip(decoder_states, cloned_states)):
        assert torch.equal(original, cloned), (
            f"TextHead modificó decoder_states[{i}]. "
            f"Max diff: {(original - cloned).abs().max().item()}"
        )

    # 2. Verificar que la salida es un ModelOutputs válido
    assert isinstance(output, ModelOutputs), (
        f"TextHead no retornó ModelOutputs, retornó {type(output).__name__}"
    )

    # 3. Verificar campos apropiados: logits para TextHead
    assert output.logits is not None, "TextHead debe producir logits"
    assert output.logits.shape == (B, vocab_size), (
        f"TextHead logits shape {output.logits.shape} != expected ({B}, {vocab_size})"
    )
    assert torch.isfinite(output.logits).all(), "TextHead logits contiene NaN o Inf"


# ---------------------------------------------------------------------------
# Propiedad 13: ClassificationHead — no modifica estado del decoder
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=head_dims())
def test_classification_head_does_not_modify_decoder_state(dims: dict) -> None:
    """Feature: vsn-library, Property 13: Head O no modifica estado del decoder (ClassificationHead)

    **Validates: Requirements 7.1, 7.2, 7.5**

    Para cualquier ClassificationHead y decoder_state, después de forward(), los tensores
    decoder_state SHALL permanecer idénticos bit a bit. La salida SHALL ser
    ModelOutputs con logits de shape (batch, num_classes).
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]
    num_windows = dims["num_windows"]
    num_classes = dims["num_classes"]

    head = ClassificationHead(d=d, num_classes=num_classes, aggregation="mean_pool")
    head.eval()

    decoder_states = _make_decoder_states(B, Y, Z, d, num_windows)
    cloned_states = _clone_decoder_states(decoder_states)
    batch = _make_dummy_batch(B, d)
    metadata: dict = {"mode": "eval"}

    with torch.no_grad():
        output = head(decoder_states, batch, metadata)

    # 1. Verificar que decoder_states no fue modificado
    assert len(decoder_states) == len(cloned_states), (
        "La cantidad de decoder states cambió después de forward()"
    )
    for i, (original, cloned) in enumerate(zip(decoder_states, cloned_states)):
        assert torch.equal(original, cloned), (
            f"ClassificationHead modificó decoder_states[{i}]. "
            f"Max diff: {(original - cloned).abs().max().item()}"
        )

    # 2. Verificar que la salida es un ModelOutputs válido
    assert isinstance(output, ModelOutputs), (
        f"ClassificationHead no retornó ModelOutputs, retornó {type(output).__name__}"
    )

    # 3. Verificar campos apropiados: logits para ClassificationHead
    assert output.logits is not None, "ClassificationHead debe producir logits"
    assert output.logits.shape == (B, num_classes), (
        f"ClassificationHead logits shape {output.logits.shape} != expected ({B}, {num_classes})"
    )
    assert torch.isfinite(output.logits).all(), "ClassificationHead logits contiene NaN o Inf"


# ---------------------------------------------------------------------------
# Propiedad 13: RegressionHead — no modifica estado del decoder
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=head_dims())
def test_regression_head_does_not_modify_decoder_state(dims: dict) -> None:
    """Feature: vsn-library, Property 13: Head O no modifica estado del decoder (RegressionHead)

    **Validates: Requirements 7.1, 7.2, 7.5**

    Para cualquier RegressionHead y decoder_state, después de forward(), los tensores
    decoder_state SHALL permanecer idénticos bit a bit. La salida SHALL ser
    ModelOutputs con embeddings de shape (batch, output_dim).
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]
    num_windows = dims["num_windows"]
    output_dim = dims["output_dim"]

    head = RegressionHead(d=d, output_dim=output_dim, aggregation="max_pool")
    head.eval()

    decoder_states = _make_decoder_states(B, Y, Z, d, num_windows)
    cloned_states = _clone_decoder_states(decoder_states)
    batch = _make_dummy_batch(B, d)
    metadata: dict = {"mode": "eval"}

    with torch.no_grad():
        output = head(decoder_states, batch, metadata)

    # 1. Verificar que decoder_states no fue modificado
    assert len(decoder_states) == len(cloned_states), (
        "La cantidad de decoder states cambió después de forward()"
    )
    for i, (original, cloned) in enumerate(zip(decoder_states, cloned_states)):
        assert torch.equal(original, cloned), (
            f"RegressionHead modificó decoder_states[{i}]. "
            f"Max diff: {(original - cloned).abs().max().item()}"
        )

    # 2. Verificar que la salida es un ModelOutputs válido
    assert isinstance(output, ModelOutputs), (
        f"RegressionHead no retornó ModelOutputs, retornó {type(output).__name__}"
    )

    # 3. Verificar campos apropiados: embeddings para RegressionHead
    assert output.embeddings is not None, "RegressionHead debe producir embeddings"
    assert output.embeddings.shape == (B, output_dim), (
        f"RegressionHead embeddings shape {output.embeddings.shape} != expected ({B}, {output_dim})"
    )
    assert torch.isfinite(output.embeddings).all(), "RegressionHead embeddings contiene NaN o Inf"


# ---------------------------------------------------------------------------
# Propiedad 13: DenseHead — no modifica estado del decoder
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=head_dims())
def test_dense_head_does_not_modify_decoder_state(dims: dict) -> None:
    """Feature: vsn-library, Property 13: Head O no modifica estado del decoder (DenseHead)

    **Validates: Requirements 7.1, 7.2, 7.5**

    Para cualquier DenseHead y decoder_state, después de forward(), los tensores
    decoder_state SHALL permanecer idénticos bit a bit. La salida SHALL ser
    ModelOutputs con embeddings de shape (batch, Y, Z, output_dim).
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]
    num_windows = dims["num_windows"]
    output_dim = dims["output_dim"]

    head = DenseHead(d=d, output_dim=output_dim)
    head.eval()

    decoder_states = _make_decoder_states(B, Y, Z, d, num_windows)
    cloned_states = _clone_decoder_states(decoder_states)
    batch = _make_dummy_batch(B, d)
    metadata: dict = {"mode": "eval"}

    with torch.no_grad():
        output = head(decoder_states, batch, metadata)

    # 1. Verificar que decoder_states no fue modificado
    assert len(decoder_states) == len(cloned_states), (
        "La cantidad de decoder states cambió después de forward()"
    )
    for i, (original, cloned) in enumerate(zip(decoder_states, cloned_states)):
        assert torch.equal(original, cloned), (
            f"DenseHead modificó decoder_states[{i}]. "
            f"Max diff: {(original - cloned).abs().max().item()}"
        )

    # 2. Verificar que la salida es un ModelOutputs válido
    assert isinstance(output, ModelOutputs), (
        f"DenseHead no retornó ModelOutputs, retornó {type(output).__name__}"
    )

    # 3. Verificar campos apropiados: embeddings para DenseHead (preserva estructura espacial)
    assert output.embeddings is not None, "DenseHead debe producir embeddings"
    assert output.embeddings.shape == (B, Y, Z, output_dim), (
        f"DenseHead embeddings shape {output.embeddings.shape} != expected ({B}, {Y}, {Z}, {output_dim})"
    )
    assert torch.isfinite(output.embeddings).all(), "DenseHead embeddings contiene NaN o Inf"
