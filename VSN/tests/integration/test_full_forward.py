"""Integration test: cadena completa IC→Enc→P→Q→Dec→O con batch sintético.

Verifica que el forward end-to-end del VSNModel produce ModelOutputs
con shapes correctas para encoder states, latent H y decoder windows.

Validates: Requirements 13.3, 13.6
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor, nn

from vsn.contracts.outputs import ModelOutputs
from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel


class SimpleRegressionHead(nn.Module):
    """Head simple que sigue el HeadProtocol del VSNModel.

    Acepta (decoder_states, metadata) y produce ModelOutputs con embeddings.
    """

    def __init__(self, d: int, output_dim: int = 1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(d, output_dim)

    def forward(
        self, decoder_states: List[Tensor], metadata: Dict[str, Any]
    ) -> ModelOutputs:
        # Mean pool sobre todas las ventanas y dimensiones espaciales
        # decoder_states: List[Tensor(batch, Y, Z, d)]
        stacked = torch.stack(decoder_states, dim=1)  # (batch, W, Y, Z, d)
        pooled = stacked.mean(dim=(1, 2, 3))  # (batch, d)
        predictions = self.linear(pooled)  # (batch, output_dim)
        return ModelOutputs(
            embeddings=predictions,
            metadata={**metadata, "head_type": "regression"},
        )


class SimpleTextHead(nn.Module):
    """Head simple de texto que sigue el HeadProtocol del VSNModel.

    Acepta (decoder_states, metadata) y produce ModelOutputs con logits.
    """

    def __init__(self, d: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, vocab_size)

    def forward(
        self, decoder_states: List[Tensor], metadata: Dict[str, Any]
    ) -> ModelOutputs:
        # Mean pool sobre la última ventana
        last_window = decoder_states[-1]  # (batch, Y, Z, d)
        pooled = last_window.mean(dim=(1, 2))  # (batch, d)
        logits = self.linear(pooled)  # (batch, vocab_size)
        return ModelOutputs(
            logits=logits,
            metadata={**metadata, "head_type": "text"},
        )


class TestFullForwardChain:
    """Tests de integración para la cadena completa IC→Enc→P→Q→Dec→O."""

    def _make_config(self) -> VSNConfig:
        """Crea una configuración small para regression."""
        return VSNConfig.small(head_type="regression")

    def _make_tokens(self, batch: int, config: VSNConfig) -> Tensor:
        """Genera tokens sintéticos compatibles con la config."""
        num_tokens = config.Y * config.Z  # tokens por plano
        return torch.randn(batch, num_tokens, config.d)

    def test_forward_without_head_produces_model_outputs(self) -> None:
        """Forward sin head retorna ModelOutputs con states."""
        config = self._make_config()
        model = VSNModel(config, head=None)
        model.eval()

        tokens = self._make_tokens(batch=2, config=config)

        with torch.no_grad():
            outputs = model(tokens)

        # Verificar que es un ModelOutputs
        assert isinstance(outputs, ModelOutputs)
        # Sin head, logits debe ser None
        assert outputs.logits is None
        # States debe existir
        assert outputs.states is not None

    def test_forward_states_contain_latent_h(self) -> None:
        """Los states contienen latent_H con shape correcta."""
        config = self._make_config()
        model = VSNModel(config, head=None)
        model.eval()

        batch_size = 2
        tokens = self._make_tokens(batch=batch_size, config=config)

        with torch.no_grad():
            outputs = model(tokens)

        assert "latent_H" in outputs.states
        H = outputs.states["latent_H"]
        assert H.shape == (batch_size, config.Y_H, config.Z_H, config.d_H)

    def test_forward_states_contain_decoder_states(self) -> None:
        """Los states contienen decoder_states como lista con shape correcta."""
        config = self._make_config()
        model = VSNModel(config, head=None, num_windows=2)
        model.eval()

        batch_size = 2
        tokens = self._make_tokens(batch=batch_size, config=config)

        with torch.no_grad():
            outputs = model(tokens, num_windows=2)

        assert "decoder_states" in outputs.states
        decoder_states = outputs.states["decoder_states"]

        # decoder_states es una lista con un tensor por ventana
        assert isinstance(decoder_states, list)
        assert len(decoder_states) == 2  # num_windows=2

        # Cada estado tiene shape (batch, Y_dec, Z_dec, d)
        for state in decoder_states:
            assert state.shape == (
                batch_size,
                config.Y_dec,
                config.Z_dec,
                config.d,
            )

    def test_forward_with_regression_head(self) -> None:
        """Forward con RegressionHead produce embeddings correctos."""
        config = self._make_config()
        head = SimpleRegressionHead(d=config.d, output_dim=4)
        model = VSNModel(config, head=head)
        model.eval()

        batch_size = 2
        tokens = self._make_tokens(batch=batch_size, config=config)

        with torch.no_grad():
            outputs = model(tokens)

        assert isinstance(outputs, ModelOutputs)
        assert outputs.embeddings is not None
        assert outputs.embeddings.shape == (batch_size, 4)
        assert outputs.metadata["head_type"] == "regression"

    def test_forward_with_text_head(self) -> None:
        """Forward con TextHead produce logits de vocabulario."""
        config = self._make_config()
        vocab_size = 1000
        head = SimpleTextHead(d=config.d, vocab_size=vocab_size)
        model = VSNModel(config, head=head)
        model.eval()

        batch_size = 2
        tokens = self._make_tokens(batch=batch_size, config=config)

        with torch.no_grad():
            outputs = model(tokens)

        assert isinstance(outputs, ModelOutputs)
        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, vocab_size)
        assert outputs.metadata["head_type"] == "text"

    def test_forward_metadata_contains_model_info(self) -> None:
        """La metadata del output contiene info del modelo."""
        config = self._make_config()
        model = VSNModel(config, head=None)
        model.eval()

        tokens = self._make_tokens(batch=1, config=config)

        with torch.no_grad():
            outputs = model(tokens)

        assert outputs.metadata["model_family"] == "vsn"
        assert outputs.metadata["vgb_version"] == "v1"
        assert "num_windows" in outputs.metadata

    def test_forward_different_num_windows(self) -> None:
        """El modelo produce diferentes cantidades de decoder states por ventana."""
        config = self._make_config()
        model = VSNModel(config, head=None)
        model.eval()

        tokens = self._make_tokens(batch=2, config=config)

        with torch.no_grad():
            out_1 = model(tokens, num_windows=1)
            out_3 = model(tokens, num_windows=3)

        assert len(out_1.states["decoder_states"]) == 1
        assert len(out_3.states["decoder_states"]) == 3

    def test_forward_is_deterministic(self) -> None:
        """El forward con los mismos inputs produce el mismo output."""
        config = self._make_config()
        model = VSNModel(config, head=None)
        model.eval()

        torch.manual_seed(42)
        tokens = torch.randn(2, config.Y * config.Z, config.d)

        with torch.no_grad():
            out1 = model(tokens)
            out2 = model(tokens)

        H1 = out1.states["latent_H"]
        H2 = out2.states["latent_H"]
        assert torch.allclose(H1, H2, atol=1e-6)
