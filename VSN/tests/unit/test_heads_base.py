"""Tests unitarios para vsn.heads.base — BaseHead, agregaciones y registry."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs
from vsn.heads.base import (
    AGGREGATION_REGISTRY,
    HEAD_REGISTRY,
    BaseHead,
    build_aggregation,
    build_head,
    cls_token,
    last_token,
    max_pool,
    mean_pool,
    register_head,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def decoder_states() -> List[Tensor]:
    """Simula 3 ventanas DGW con shape (batch=2, Y=4, Z=3, d=8)."""
    torch.manual_seed(42)
    return [torch.randn(2, 4, 3, 8) for _ in range(3)]


@pytest.fixture
def single_window_states() -> List[Tensor]:
    """Una sola ventana DGW (batch=1, Y=2, Z=2, d=4)."""
    torch.manual_seed(0)
    return [torch.randn(1, 2, 2, 4)]


# ---------------------------------------------------------------------------
# Tests de funciones de agregación
# ---------------------------------------------------------------------------


class TestLastToken:
    def test_output_shape(self, decoder_states: List[Tensor]) -> None:
        result = last_token(decoder_states)
        # Debe producir (batch=2, d=8)
        assert result.shape == (2, 8)

    def test_uses_last_window(self, decoder_states: List[Tensor]) -> None:
        """Verifica que solo usa el último window."""
        expected = decoder_states[-1].mean(dim=(1, 2))
        result = last_token(decoder_states)
        assert torch.allclose(result, expected)

    def test_single_window(self, single_window_states: List[Tensor]) -> None:
        result = last_token(single_window_states)
        assert result.shape == (1, 4)


class TestMeanPool:
    def test_output_shape(self, decoder_states: List[Tensor]) -> None:
        result = mean_pool(decoder_states)
        assert result.shape == (2, 8)

    def test_averages_all_windows(self, decoder_states: List[Tensor]) -> None:
        """Verifica que promedia sobre todos los windows y dims espaciales."""
        stacked = torch.stack(decoder_states, dim=0)  # (3, 2, 4, 3, 8)
        expected = stacked.mean(dim=(0, 2, 3))  # (2, 8)
        result = mean_pool(decoder_states)
        assert torch.allclose(result, expected)

    def test_single_window(self, single_window_states: List[Tensor]) -> None:
        result = mean_pool(single_window_states)
        assert result.shape == (1, 4)


class TestMaxPool:
    def test_output_shape(self, decoder_states: List[Tensor]) -> None:
        result = max_pool(decoder_states)
        assert result.shape == (2, 8)

    def test_uses_last_window_max(self, decoder_states: List[Tensor]) -> None:
        """Verifica que toma el max espacial del último window."""
        last = decoder_states[-1]  # (2, 4, 3, 8)
        flat = last.reshape(2, 12, 8)
        expected = flat.max(dim=1).values
        result = max_pool(decoder_states)
        assert torch.allclose(result, expected)

    def test_single_window(self, single_window_states: List[Tensor]) -> None:
        result = max_pool(single_window_states)
        assert result.shape == (1, 4)


class TestClsToken:
    def test_output_shape(self, decoder_states: List[Tensor]) -> None:
        result = cls_token(decoder_states)
        assert result.shape == (2, 8)

    def test_uses_position_0_0(self, decoder_states: List[Tensor]) -> None:
        """Verifica que toma posición [0,0] del último window."""
        expected = decoder_states[-1][:, 0, 0, :]
        result = cls_token(decoder_states)
        assert torch.allclose(result, expected)

    def test_single_window(self, single_window_states: List[Tensor]) -> None:
        result = cls_token(single_window_states)
        assert result.shape == (1, 4)


class TestBuildAggregation:
    def test_all_registered(self) -> None:
        """Verifica que las 4 funciones estándar están registradas."""
        assert "last_token" in AGGREGATION_REGISTRY
        assert "mean_pool" in AGGREGATION_REGISTRY
        assert "max_pool" in AGGREGATION_REGISTRY
        assert "cls_token" in AGGREGATION_REGISTRY

    def test_returns_callable(self) -> None:
        fn = build_aggregation("last_token")
        assert callable(fn)
        assert fn is last_token

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            build_aggregation("nonexistent")


# ---------------------------------------------------------------------------
# Tests de BaseHead ABC
# ---------------------------------------------------------------------------


class TestBaseHead:
    def test_cannot_instantiate_directly(self) -> None:
        """BaseHead es abstracto — no se puede instanciar."""
        with pytest.raises(TypeError):
            BaseHead()  # type: ignore[abstract]

    def test_subclass_must_implement_forward(self) -> None:
        """Una subclase sin forward implementado no se puede instanciar."""

        class IncompleteHead(BaseHead):
            pass

        with pytest.raises(TypeError):
            IncompleteHead()  # type: ignore[abstract]

    def test_concrete_subclass_works(self, decoder_states: List[Tensor]) -> None:
        """Una subclase concreta funciona correctamente."""

        class DummyHead(BaseHead):
            def forward(
                self,
                decoder_states: List[Tensor],
                batch: MultimodalBatch,
                metadata: Dict[str, Any],
            ) -> ModelOutputs:
                agg = last_token(decoder_states)
                return ModelOutputs(embeddings=agg)

        head = DummyHead()
        batch = MultimodalBatch(modalities={}, task="test", targets=None, target_mask=None)
        result = head(decoder_states, batch, {})
        assert isinstance(result, ModelOutputs)
        assert result.embeddings is not None
        assert result.embeddings.shape == (2, 8)


# ---------------------------------------------------------------------------
# Tests del registry de heads
# ---------------------------------------------------------------------------


class TestHeadRegistry:
    def test_register_and_build(self) -> None:
        """Registrar un head y construirlo con build_head."""
        # Limpiar si ya existe del test anterior
        HEAD_REGISTRY.pop("_test_dummy", None)

        @register_head("_test_dummy")
        class _TestDummyHead(BaseHead):
            def __init__(self, config: Any):
                super().__init__()
                self.config = config

            def forward(
                self,
                decoder_states: List[Tensor],
                batch: MultimodalBatch,
                metadata: Dict[str, Any],
            ) -> ModelOutputs:
                return ModelOutputs()

        assert "_test_dummy" in HEAD_REGISTRY
        head = build_head("_test_dummy", {"d": 64})
        assert isinstance(head, _TestDummyHead)
        assert head.config == {"d": 64}

        # Cleanup
        HEAD_REGISTRY.pop("_test_dummy", None)

    def test_duplicate_registration_raises(self) -> None:
        """No se permite registrar el mismo nombre dos veces."""
        HEAD_REGISTRY.pop("_test_dup", None)

        @register_head("_test_dup")
        class _HeadA(BaseHead):
            def __init__(self, config: Any):
                super().__init__()

            def forward(self, decoder_states: List[Tensor], batch: MultimodalBatch, metadata: Dict[str, Any]) -> ModelOutputs:
                return ModelOutputs()

        with pytest.raises(ValueError, match="already registered"):

            @register_head("_test_dup")
            class _HeadB(BaseHead):
                def __init__(self, config: Any):
                    super().__init__()

                def forward(self, decoder_states: List[Tensor], batch: MultimodalBatch, metadata: Dict[str, Any]) -> ModelOutputs:
                    return ModelOutputs()

        HEAD_REGISTRY.pop("_test_dup", None)

    def test_register_non_basehead_raises(self) -> None:
        """Registrar una clase que no hereda de BaseHead lanza TypeError."""
        with pytest.raises(TypeError, match="must be a subclass of BaseHead"):
            register_head("_bad")(dict)  # type: ignore[arg-type]

    def test_build_unknown_head_raises(self) -> None:
        """build_head con nombre no registrado lanza KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            build_head("_nonexistent_head", {})
