"""Tests unitarios para vsn.heads — ClassificationHead, RegressionHead, DenseHead."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import HEAD_REGISTRY
from vsn.heads.classification import ClassificationHead
from vsn.heads.dense import DenseHead
from vsn.heads.regression import RegressionHead


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
    return MultimodalBatch(modalities={}, task="classification", targets=None, target_mask=None)


# ===========================================================================
# ClassificationHead Tests
# ===========================================================================


class TestClassificationHeadInit:
    def test_default_aggregation(self) -> None:
        """Por defecto usa 'last_token' como agregación."""
        head = ClassificationHead(d=16, num_classes=10)
        assert head.aggregation_name == "last_token"
        assert head.d == 16
        assert head.num_classes == 10

    def test_custom_aggregation(self) -> None:
        """Permite configurar la agregación."""
        head = ClassificationHead(d=32, num_classes=5, aggregation="mean_pool")
        assert head.aggregation_name == "mean_pool"

    def test_all_aggregations_accepted(self) -> None:
        """Acepta todas las agregaciones registradas."""
        for agg_name in ("last_token", "mean_pool", "max_pool", "cls_token"):
            head = ClassificationHead(d=8, num_classes=10, aggregation=agg_name)
            assert head.aggregation_name == agg_name

    def test_invalid_aggregation_raises(self) -> None:
        """Agregación inválida lanza KeyError."""
        with pytest.raises(KeyError, match="not found"):
            ClassificationHead(d=8, num_classes=10, aggregation="invalid_agg")

    def test_projection_shape(self) -> None:
        """W_O tiene la shape correcta (num_classes, d) con bias."""
        head = ClassificationHead(d=16, num_classes=100)
        assert head.W_O.in_features == 16
        assert head.W_O.out_features == 100
        assert head.W_O.bias is not None


class TestClassificationHeadForward:
    def test_output_type(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward retorna ModelOutputs."""
        head = ClassificationHead(d=8, num_classes=10)
        result = head(decoder_states, dummy_batch, {})
        assert isinstance(result, ModelOutputs)

    def test_logits_shape(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Logits tienen shape (batch, num_classes)."""
        head = ClassificationHead(d=8, num_classes=10)
        result = head(decoder_states, dummy_batch, {})
        assert result.logits is not None
        assert result.logits.shape == (2, 10)  # batch=2, num_classes=10

    def test_logits_shape_various_dims(self, dummy_batch: MultimodalBatch) -> None:
        """Verifica shapes para diversas configuraciones."""
        torch.manual_seed(7)
        configs = [
            (4, 5, 1, 2, 2),    # d, num_classes, batch, Y, Z
            (16, 20, 4, 3, 5),
            (64, 100, 1, 1, 1),
        ]
        for d, num_classes, batch, Y, Z in configs:
            head = ClassificationHead(d=d, num_classes=num_classes)
            states = [torch.randn(batch, Y, Z, d)]
            result = head(states, dummy_batch, {})
            assert result.logits is not None
            assert result.logits.shape == (batch, num_classes)

    def test_metadata_enrichment(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Metadata incluye head_type y aggregation_mode."""
        head = ClassificationHead(d=8, num_classes=10)
        result = head(decoder_states, dummy_batch, {"step": 42})
        assert result.metadata["head_type"] == "classification"
        assert result.metadata["aggregation_mode"] == "last_token"
        assert result.metadata["step"] == 42

    def test_does_not_mutate_input_metadata(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward no muta el diccionario metadata de entrada."""
        head = ClassificationHead(d=8, num_classes=10)
        input_metadata: Dict[str, Any] = {"mode": "eval"}
        head(decoder_states, dummy_batch, input_metadata)
        assert "head_type" not in input_metadata

    def test_no_embeddings_field(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """ClassificationHead no produce embeddings."""
        head = ClassificationHead(d=8, num_classes=10)
        result = head(decoder_states, dummy_batch, {})
        assert result.embeddings is None
        assert result.aux_losses == {}


class TestClassificationHeadReadOnly:
    def test_decoder_states_unchanged(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """El head no modifica los tensores de decoder_states."""
        head = ClassificationHead(d=8, num_classes=10)
        originals = [s.clone() for s in decoder_states]
        head(decoder_states, dummy_batch, {})
        for orig, state in zip(originals, decoder_states):
            assert torch.equal(orig, state)


class TestClassificationHeadRegistration:
    def test_registered_in_head_registry(self) -> None:
        """ClassificationHead está registrado con nombre 'classification'."""
        assert "classification" in HEAD_REGISTRY
        assert HEAD_REGISTRY["classification"] is ClassificationHead


class TestClassificationHeadGradients:
    def test_backward_computes_gradients(self, dummy_batch: MultimodalBatch) -> None:
        """Los gradientes fluyen a través de ClassificationHead."""
        head = ClassificationHead(d=8, num_classes=10)
        states = [torch.randn(2, 4, 3, 8, requires_grad=True)]
        result = head(states, dummy_batch, {})
        loss = result.logits.sum()
        loss.backward()
        assert head.W_O.weight.grad is not None
        assert head.W_O.bias.grad is not None
        assert states[0].grad is not None


# ===========================================================================
# RegressionHead Tests
# ===========================================================================


class TestRegressionHeadInit:
    def test_default_params(self) -> None:
        """Por defecto output_dim=1 y aggregation='last_token'."""
        head = RegressionHead(d=16)
        assert head.aggregation_name == "last_token"
        assert head.d == 16
        assert head.output_dim == 1

    def test_custom_output_dim(self) -> None:
        """Permite configurar output_dim."""
        head = RegressionHead(d=32, output_dim=5)
        assert head.output_dim == 5

    def test_custom_aggregation(self) -> None:
        """Permite configurar la agregación."""
        head = RegressionHead(d=32, output_dim=3, aggregation="mean_pool")
        assert head.aggregation_name == "mean_pool"

    def test_all_aggregations_accepted(self) -> None:
        """Acepta todas las agregaciones registradas."""
        for agg_name in ("last_token", "mean_pool", "max_pool", "cls_token"):
            head = RegressionHead(d=8, output_dim=1, aggregation=agg_name)
            assert head.aggregation_name == agg_name

    def test_invalid_aggregation_raises(self) -> None:
        """Agregación inválida lanza KeyError."""
        with pytest.raises(KeyError, match="not found"):
            RegressionHead(d=8, output_dim=1, aggregation="invalid_agg")

    def test_projection_shape(self) -> None:
        """W_O tiene la shape correcta (output_dim, d) con bias."""
        head = RegressionHead(d=16, output_dim=3)
        assert head.W_O.in_features == 16
        assert head.W_O.out_features == 3
        assert head.W_O.bias is not None


class TestRegressionHeadForward:
    def test_output_type(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward retorna ModelOutputs."""
        head = RegressionHead(d=8, output_dim=1)
        result = head(decoder_states, dummy_batch, {})
        assert isinstance(result, ModelOutputs)

    def test_embeddings_shape_default(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Embeddings tienen shape (batch, output_dim) con default output_dim=1."""
        head = RegressionHead(d=8, output_dim=1)
        result = head(decoder_states, dummy_batch, {})
        assert result.embeddings is not None
        assert result.embeddings.shape == (2, 1)  # batch=2, output_dim=1

    def test_embeddings_shape_multi_dim(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Embeddings con output_dim > 1."""
        head = RegressionHead(d=8, output_dim=5)
        result = head(decoder_states, dummy_batch, {})
        assert result.embeddings is not None
        assert result.embeddings.shape == (2, 5)

    def test_embeddings_shape_various_dims(self, dummy_batch: MultimodalBatch) -> None:
        """Verifica shapes para diversas configuraciones."""
        torch.manual_seed(7)
        configs = [
            (4, 1, 1, 2, 2),    # d, output_dim, batch, Y, Z
            (16, 3, 4, 3, 5),
            (64, 10, 1, 1, 1),
        ]
        for d, output_dim, batch, Y, Z in configs:
            head = RegressionHead(d=d, output_dim=output_dim)
            states = [torch.randn(batch, Y, Z, d)]
            result = head(states, dummy_batch, {})
            assert result.embeddings is not None
            assert result.embeddings.shape == (batch, output_dim)

    def test_metadata_enrichment(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Metadata incluye head_type y aggregation_mode."""
        head = RegressionHead(d=8, output_dim=1)
        result = head(decoder_states, dummy_batch, {"step": 7})
        assert result.metadata["head_type"] == "regression"
        assert result.metadata["aggregation_mode"] == "last_token"
        assert result.metadata["step"] == 7

    def test_does_not_mutate_input_metadata(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward no muta el diccionario metadata de entrada."""
        head = RegressionHead(d=8, output_dim=1)
        input_metadata: Dict[str, Any] = {"mode": "train"}
        head(decoder_states, dummy_batch, input_metadata)
        assert "head_type" not in input_metadata

    def test_no_logits_field(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """RegressionHead no produce logits."""
        head = RegressionHead(d=8, output_dim=1)
        result = head(decoder_states, dummy_batch, {})
        assert result.logits is None
        assert result.aux_losses == {}


class TestRegressionHeadReadOnly:
    def test_decoder_states_unchanged(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """El head no modifica los tensores de decoder_states."""
        head = RegressionHead(d=8, output_dim=1)
        originals = [s.clone() for s in decoder_states]
        head(decoder_states, dummy_batch, {})
        for orig, state in zip(originals, decoder_states):
            assert torch.equal(orig, state)


class TestRegressionHeadRegistration:
    def test_registered_in_head_registry(self) -> None:
        """RegressionHead está registrado con nombre 'regression'."""
        assert "regression" in HEAD_REGISTRY
        assert HEAD_REGISTRY["regression"] is RegressionHead


class TestRegressionHeadGradients:
    def test_backward_computes_gradients(self, dummy_batch: MultimodalBatch) -> None:
        """Los gradientes fluyen a través de RegressionHead."""
        head = RegressionHead(d=8, output_dim=3)
        states = [torch.randn(2, 4, 3, 8, requires_grad=True)]
        result = head(states, dummy_batch, {})
        loss = result.embeddings.sum()
        loss.backward()
        assert head.W_O.weight.grad is not None
        assert head.W_O.bias.grad is not None
        assert states[0].grad is not None


# ===========================================================================
# DenseHead Tests
# ===========================================================================


class TestDenseHeadInit:
    def test_params_stored(self) -> None:
        """Almacena d y output_dim correctamente."""
        head = DenseHead(d=16, output_dim=5)
        assert head.d == 16
        assert head.output_dim == 5

    def test_projection_shape(self) -> None:
        """W_O tiene la shape correcta (output_dim, d) con bias."""
        head = DenseHead(d=16, output_dim=10)
        assert head.W_O.in_features == 16
        assert head.W_O.out_features == 10
        assert head.W_O.bias is not None


class TestDenseHeadForward:
    def test_output_type(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward retorna ModelOutputs."""
        head = DenseHead(d=8, output_dim=5)
        result = head(decoder_states, dummy_batch, {})
        assert isinstance(result, ModelOutputs)

    def test_embeddings_shape_preserves_spatial(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Embeddings preservan estructura espacial: (batch, Y, Z, output_dim)."""
        head = DenseHead(d=8, output_dim=5)
        result = head(decoder_states, dummy_batch, {})
        assert result.embeddings is not None
        # decoder_states shape: (batch=2, Y=4, Z=3, d=8)
        assert result.embeddings.shape == (2, 4, 3, 5)

    def test_embeddings_shape_various_dims(self, dummy_batch: MultimodalBatch) -> None:
        """Verifica shapes para diversas configuraciones."""
        torch.manual_seed(7)
        configs = [
            (4, 2, 1, 2, 2),    # d, output_dim, batch, Y, Z
            (16, 8, 4, 3, 5),
            (64, 1, 1, 1, 1),
            (8, 10, 3, 6, 4),
        ]
        for d, output_dim, batch, Y, Z in configs:
            head = DenseHead(d=d, output_dim=output_dim)
            states = [torch.randn(batch, Y, Z, d)]
            result = head(states, dummy_batch, {})
            assert result.embeddings is not None
            assert result.embeddings.shape == (batch, Y, Z, output_dim)

    def test_operates_on_last_window_only(self, dummy_batch: MultimodalBatch) -> None:
        """DenseHead solo procesa el último window de la lista."""
        torch.manual_seed(10)
        head = DenseHead(d=8, output_dim=3)
        # Crear 2 windows con datos distintos
        states = [torch.zeros(2, 4, 3, 8), torch.ones(2, 4, 3, 8)]
        result = head(states, dummy_batch, {})
        # El resultado debe depender solo del último window (ones)
        # Para verificar, llamar con solo el último
        result_single = head([states[-1]], dummy_batch, {})
        assert torch.equal(result.embeddings, result_single.embeddings)

    def test_metadata_enrichment(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Metadata incluye head_type."""
        head = DenseHead(d=8, output_dim=5)
        result = head(decoder_states, dummy_batch, {"step": 99})
        assert result.metadata["head_type"] == "dense"
        assert result.metadata["step"] == 99

    def test_does_not_mutate_input_metadata(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """Forward no muta el diccionario metadata de entrada."""
        head = DenseHead(d=8, output_dim=5)
        input_metadata: Dict[str, Any] = {"mode": "eval"}
        head(decoder_states, dummy_batch, input_metadata)
        assert "head_type" not in input_metadata

    def test_no_logits_field(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """DenseHead no produce logits."""
        head = DenseHead(d=8, output_dim=5)
        result = head(decoder_states, dummy_batch, {})
        assert result.logits is None
        assert result.aux_losses == {}


class TestDenseHeadReadOnly:
    def test_decoder_states_unchanged(
        self,
        decoder_states: List[Tensor],
        dummy_batch: MultimodalBatch,
    ) -> None:
        """El head no modifica los tensores de decoder_states."""
        head = DenseHead(d=8, output_dim=5)
        originals = [s.clone() for s in decoder_states]
        head(decoder_states, dummy_batch, {})
        for orig, state in zip(originals, decoder_states):
            assert torch.equal(orig, state)


class TestDenseHeadRegistration:
    def test_registered_in_head_registry(self) -> None:
        """DenseHead está registrado con nombre 'dense'."""
        assert "dense" in HEAD_REGISTRY
        assert HEAD_REGISTRY["dense"] is DenseHead


class TestDenseHeadGradients:
    def test_backward_computes_gradients(self, dummy_batch: MultimodalBatch) -> None:
        """Los gradientes fluyen a través de DenseHead."""
        head = DenseHead(d=8, output_dim=5)
        states = [torch.randn(2, 4, 3, 8, requires_grad=True)]
        result = head(states, dummy_batch, {})
        loss = result.embeddings.sum()
        loss.backward()
        assert head.W_O.weight.grad is not None
        assert head.W_O.bias.grad is not None
        assert states[0].grad is not None

    def test_gradients_flow_per_voxel(self, dummy_batch: MultimodalBatch) -> None:
        """Verifica que los gradientes fluyen independientemente por voxel."""
        head = DenseHead(d=4, output_dim=2)
        states = [torch.randn(1, 2, 2, 4, requires_grad=True)]
        result = head(states, dummy_batch, {})
        # Solo backprop desde un voxel específico
        loss = result.embeddings[0, 0, 0, :].sum()
        loss.backward()
        # El gradiente del input debe ser no-cero solo en la posición [0, 0, 0, :]
        grad = states[0].grad
        assert grad is not None
        assert grad[0, 0, 0, :].abs().sum() > 0
        assert grad[0, 1, 1, :].abs().sum() == 0  # Otra posición → sin gradiente
