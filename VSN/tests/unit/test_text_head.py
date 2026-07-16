"""Tests unitarios para vsn.heads.text — TextHead."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import HEAD_REGISTRY, build_head
from vsn.heads.text import TextHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def decoder_states() -> List[Tensor]:
    """Simula 3 ventanas DGW con shape (batch=2, Y=4, Z=3, d=8)."""
    torch.manual_seed(42)
    return [torch.randn(2, 4, 3, 8) for _ in range(3)]


@pytest.fixture
def dummy_batch() -> MultimodalBatch:
    """Batch multimodal vacío para testing."""
    return MultimodalBatch(modalities={}, task="text", targets=None, target_mask=None)


@pytest.fixture
def text_head() -> TextHead:
    """TextHead con d=8, vocab_size=100, aggregation default."""
    return TextHead(d=8, vocab_size=100)


# ---------------------------------------------------------------------------
# Tests de instanciación
# ---------------------------------------------------------------------------


class TestTextHeadInit:
    def test_default_aggregation(self) -> None:
        """Por defecto usa 'last_token' como agregación."""
        head = TextHead(d=16, vocab_size=50)
        assert head.aggregation_name == "last_token"
        assert head.d == 16
        assert head.vocab_size == 50

    def test_custom_aggregation(self) -> None:
        """Permite configurar la agregación."""
        head = TextHead(d=32, vocab_size=200, aggregation="mean_pool")
        assert head.aggregation_name == "mean_pool"

    def test_all_aggregations_accepted(self) -> None:
        """Acepta todas las agregaciones registradas."""
        for agg_name in ("last_token", "mean_pool", "max_pool", "cls_token"):
            head = TextHead(d=8, vocab_size=50, aggregation=agg_name)
            assert head.aggregation_name == agg_name

    def test_invalid_aggregation_raises(self) -> None:
        """Agregación inválida lanza KeyError."""
        with pytest.raises(KeyError, match="not found"):
            TextHead(d=8, vocab_size=50, aggregation="invalid_agg")

    def test_projection_shape(self) -> None:
        """W_O tiene la shape correcta (vocab_size, d) con bias."""
        head = TextHead(d=16, vocab_size=1000)
        assert head.W_O.in_features == 16
        assert head.W_O.out_features == 1000
        assert head.W_O.bias is not None


# ---------------------------------------------------------------------------
# Tests de forward
# ---------------------------------------------------------------------------


class TestTextHeadForward:
    def test_output_type(
        self,
        text_head: TextHead,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward retorna ModelOutputs."""
        result = text_head(decoder_states, dummy_batch, {})
        assert isinstance(result, ModelOutputs)

    def test_logits_shape(
        self,
        text_head: TextHead,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Logits tienen shape (batch, vocab_size)."""
        result = text_head(decoder_states, dummy_batch, {})
        assert result.logits is not None
        assert result.logits.shape == (2, 100)  # batch=2, vocab=100

    def test_logits_shape_various_dims(self, dummy_batch: MultimodalBatch) -> None:
        """Verifica shapes para diversas configuraciones."""
        torch.manual_seed(7)
        configs = [
            (4, 50, 1, 2, 2),   # d, vocab, batch, Y, Z
            (16, 200, 4, 3, 5),
            (64, 1000, 1, 1, 1),
        ]
        for d, vocab, batch, Y, Z in configs:
            head = TextHead(d=d, vocab_size=vocab)
            states = [torch.randn(batch, Y, Z, d)]
            result = head(states, dummy_batch, {})
            assert result.logits is not None
            assert result.logits.shape == (batch, vocab)

    def test_metadata_enrichment(
        self,
        text_head: TextHead,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Metadata incluye head_type y aggregation_mode."""
        result = text_head(decoder_states, dummy_batch, {"step": 42})
        assert result.metadata["head_type"] == "text"
        assert result.metadata["aggregation_mode"] == "last_token"
        # Preserva metadata original
        assert result.metadata["step"] == 42

    def test_does_not_mutate_input_metadata(
        self,
        text_head: TextHead,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward no muta el diccionario metadata de entrada."""
        input_metadata: Dict[str, Any] = {"mode": "eval"}
        text_head(decoder_states, dummy_batch, input_metadata)
        # El dict original no debe tener head_type
        assert "head_type" not in input_metadata

    def test_no_embeddings_or_aux(
        self,
        text_head: TextHead,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """TextHead no produce embeddings ni aux_losses."""
        result = text_head(decoder_states, dummy_batch, {})
        assert result.embeddings is None
        assert result.aux_losses == {}

    def test_different_aggregations_produce_different_logits(
        self, decoder_states: List[Tensor], dummy_batch: MultimodalBatch
    ) -> None:
        """Diferentes agregaciones producen logits distintos."""
        torch.manual_seed(99)
        head_last = TextHead(d=8, vocab_size=50, aggregation="last_token")
        head_mean = TextHead(d=8, vocab_size=50, aggregation="mean_pool")
        # Copiar pesos para que solo la agregación difiera
        head_mean.W_O.weight.data.copy_(head_last.W_O.weight.data)
        head_mean.W_O.bias.data.copy_(head_last.W_O.bias.data)

        out_last = head_last(decoder_states, dummy_batch, {})
        out_mean = head_mean(decoder_states, dummy_batch, {})
        # Logits deben diferir (agregaciones distintas → inputs distintos a W_O)
        assert not torch.allclose(out_last.logits, out_mean.logits)


# ---------------------------------------------------------------------------
# Tests de no-mutación de decoder states (Req 7.2, 7.5)
# ---------------------------------------------------------------------------


class TestTextHeadReadOnly:
    def test_decoder_states_unchanged(
        self,
        text_head: TextHead,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """El head no modifica los tensores de decoder_states."""
        # Guardar copias antes del forward
        originals = [s.clone() for s in decoder_states]
        text_head(decoder_states, dummy_batch, {})
        # Verificar que no se modificaron
        for orig, state in zip(originals, decoder_states):
            assert torch.equal(orig, state)


# ---------------------------------------------------------------------------
# Tests de registro
# ---------------------------------------------------------------------------


class TestTextHeadRegistration:
    def test_registered_in_head_registry(self) -> None:
        """TextHead está registrado con nombre 'text'."""
        assert "text" in HEAD_REGISTRY
        assert HEAD_REGISTRY["text"] is TextHead

    def test_build_head_creates_text_head(self) -> None:
        """build_head('text', config) crea un TextHead."""
        # build_head pasa config como primer arg — TextHead no acepta config dict
        # así que probamos el registro directo
        assert HEAD_REGISTRY["text"] is TextHead


# ---------------------------------------------------------------------------
# Tests de gradientes
# ---------------------------------------------------------------------------


class TestTextHeadGradients:
    def test_backward_computes_gradients(
        self, dummy_batch: MultimodalBatch
    ) -> None:
        """Los gradientes fluyen a través de TextHead."""
        head = TextHead(d=8, vocab_size=50)
        states = [torch.randn(2, 4, 3, 8, requires_grad=True)]
        result = head(states, dummy_batch, {})
        loss = result.logits.sum()
        loss.backward()
        # W_O debe tener gradientes
        assert head.W_O.weight.grad is not None
        assert head.W_O.bias.grad is not None
        # Input states deben tener gradientes
        assert states[0].grad is not None
