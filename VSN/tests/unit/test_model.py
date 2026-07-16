"""Tests unitarios para VSNModel.

Verifica:
1. Instanciación desde VSNConfig.small()
2. Forward produce output shapes correctos
3. Backward funciona (gradientes fluyen)
4. Config inválido lanza ConfigurationError en construcción
"""

import pytest
import torch

from vsn.contracts.outputs import ModelOutputs
from vsn.core.config import ConfigurationError, VSNConfig
from vsn.core.model import VSNModel


class TestVSNModelInstantiation:
    """Tests de instanciación del modelo."""

    def test_instantiation_from_small_config(self):
        """El modelo se instancia correctamente desde VSNConfig.small()."""
        config = VSNConfig.small()
        model = VSNModel(config)

        assert model.config is config
        assert model.encoder is not None
        assert model.P is not None
        assert model.Q is not None
        assert model.decoder is not None
        assert model.head is None  # Sin head por defecto

    def test_instantiation_from_base_config(self):
        """El modelo se instancia correctamente desde VSNConfig.base()."""
        config = VSNConfig.base()
        model = VSNModel(config)

        assert model.config is config
        assert model.num_windows == 1

    def test_instantiation_with_custom_num_windows(self):
        """El modelo respeta num_windows configurado."""
        config = VSNConfig.small()
        model = VSNModel(config, num_windows=3)

        assert model.num_windows == 3

    def test_invalid_config_raises_configuration_error(self):
        """Config con dimensiones inválidas lanza ConfigurationError."""
        # d negativo
        with pytest.raises(ConfigurationError):
            VSNModel(VSNConfig(
                X_enc=4, X_dec=4, Y=4, Z=4, d=-1,
                ics=64, Y_H=4, Z_H=4, d_H=64,
                p_mode="identity", Y_dec=4, Z_dec=4, dgw=4,
                head_type="text", vocab_size=32000,
            ))

    def test_invalid_p_mode_raises_configuration_error(self):
        """Config con p_mode inválido lanza ConfigurationError."""
        with pytest.raises(ConfigurationError):
            VSNModel(VSNConfig(
                X_enc=4, X_dec=4, Y=4, Z=4, d=64,
                ics=64, Y_H=4, Z_H=4, d_H=64,
                p_mode="invalid_mode", Y_dec=4, Z_dec=4, dgw=4,
                head_type="text", vocab_size=32000,
            ))

    def test_incompatible_p_mode_compress_raises_error(self):
        """p_mode='compress' con dims iguales lanza ConfigurationError."""
        with pytest.raises(ConfigurationError):
            VSNModel(VSNConfig(
                X_enc=4, X_dec=4, Y=4, Z=4, d=64,
                ics=64, Y_H=4, Z_H=4, d_H=64,
                p_mode="compress",  # identity dims pero modo compress
                Y_dec=4, Z_dec=4, dgw=4,
                head_type="text", vocab_size=32000,
            ))

    def test_text_head_without_vocab_size_raises_error(self):
        """head_type='text' sin vocab_size lanza ConfigurationError."""
        with pytest.raises(ConfigurationError):
            VSNModel(VSNConfig(
                X_enc=4, X_dec=4, Y=4, Z=4, d=64,
                ics=64, Y_H=4, Z_H=4, d_H=64,
                p_mode="identity", Y_dec=4, Z_dec=4, dgw=4,
                head_type="text", vocab_size=None,
            ))

    def test_model_has_parameters(self):
        """El modelo tiene parámetros entrenables."""
        config = VSNConfig.small()
        model = VSNModel(config)

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_repr_does_not_crash(self):
        """repr() del modelo no lanza excepciones."""
        config = VSNConfig.small()
        model = VSNModel(config)

        repr_str = repr(model)
        assert "VSNModel" in repr_str
        assert "encoder" in repr_str


class TestVSNModelForward:
    """Tests del forward pass."""

    @pytest.fixture
    def small_model(self):
        """Modelo small para tests."""
        config = VSNConfig.small()
        return VSNModel(config)

    def test_forward_produces_model_outputs(self, small_model):
        """Forward retorna ModelOutputs."""
        batch_size = 2
        # Para config small: X_enc=4, Y=4, Z=4, d=64
        # num_tokens puede ser hasta X*Y*Z = 64
        num_tokens = 16
        tokens = torch.randn(batch_size, num_tokens, small_model.config.d)

        outputs = small_model(tokens)

        assert isinstance(outputs, ModelOutputs)

    def test_forward_without_head_has_none_logits(self, small_model):
        """Sin head, logits es None."""
        tokens = torch.randn(2, 16, small_model.config.d)

        outputs = small_model(tokens)

        assert outputs.logits is None

    def test_forward_states_contain_decoder_states(self, small_model):
        """Los states contienen decoder_states y latent_H."""
        tokens = torch.randn(2, 16, small_model.config.d)

        outputs = small_model(tokens)

        assert outputs.states is not None
        assert "decoder_states" in outputs.states
        assert "latent_H" in outputs.states

    def test_forward_decoder_states_shape(self, small_model):
        """Los decoder_states tienen shape correcto."""
        batch_size = 2
        tokens = torch.randn(batch_size, 16, small_model.config.d)

        outputs = small_model(tokens)

        decoder_states = outputs.states["decoder_states"]
        assert isinstance(decoder_states, list)
        assert len(decoder_states) == 1  # 1 ventana por defecto

        # Cada estado: (batch, Y_dec, Z_dec, d)
        state = decoder_states[0]
        assert state.shape == (
            batch_size,
            small_model.config.Y_dec,
            small_model.config.Z_dec,
            small_model.config.d,
        )

    def test_forward_latent_H_shape(self, small_model):
        """El plano latente H tiene shape correcto."""
        batch_size = 2
        tokens = torch.randn(batch_size, 16, small_model.config.d)

        outputs = small_model(tokens)

        H = outputs.states["latent_H"]
        assert H.shape == (
            batch_size,
            small_model.config.Y_H,
            small_model.config.Z_H,
            small_model.config.d_H,
        )

    def test_forward_metadata_contains_config_info(self, small_model):
        """Metadata contiene información del config."""
        tokens = torch.randn(2, 16, small_model.config.d)

        outputs = small_model(tokens)

        assert outputs.metadata["model_family"] == "vsn"
        assert outputs.metadata["vgb_version"] == "v1"
        assert outputs.metadata["head_type"] == "text"
        assert outputs.metadata["num_windows"] == 1

    def test_forward_multiple_windows(self, small_model):
        """Forward con múltiples ventanas produce estados por ventana."""
        batch_size = 2
        tokens = torch.randn(batch_size, 16, small_model.config.d)

        outputs = small_model(tokens, num_windows=3)

        decoder_states = outputs.states["decoder_states"]
        assert len(decoder_states) == 3

        # Todas las ventanas tienen la misma shape
        for state in decoder_states:
            assert state.shape == (
                batch_size,
                small_model.config.Y_dec,
                small_model.config.Z_dec,
                small_model.config.d,
            )

    def test_forward_num_windows_override(self):
        """num_windows en forward sobreescribe el default."""
        config = VSNConfig.small()
        model = VSNModel(config, num_windows=1)
        tokens = torch.randn(2, 16, config.d)

        outputs = model(tokens, num_windows=2)

        assert outputs.metadata["num_windows"] == 2
        assert len(outputs.states["decoder_states"]) == 2


class TestVSNModelBackward:
    """Tests de backward pass (gradientes)."""

    def test_backward_produces_gradients(self):
        """Backward produce gradientes en todos los parámetros."""
        config = VSNConfig.small()
        model = VSNModel(config)
        tokens = torch.randn(2, 16, config.d)

        outputs = model(tokens)

        # Usar el último decoder state como proxy de loss
        loss = outputs.states["decoder_states"][0].sum()
        loss.backward()

        # Verificar que al menos algunos parámetros tienen gradientes
        params_with_grad = [
            p for p in model.parameters() if p.grad is not None
        ]
        assert len(params_with_grad) > 0

    def test_gradients_flow_through_all_components(self):
        """Gradientes fluyen a través de encoder, P, Q y decoder."""
        config = VSNConfig.small()
        model = VSNModel(config)
        tokens = torch.randn(2, 16, config.d)

        outputs = model(tokens)
        loss = outputs.states["decoder_states"][0].sum()
        loss.backward()

        # Encoder tiene gradientes
        encoder_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        assert encoder_grads, "Encoder no recibió gradientes"

        # P tiene gradientes
        p_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.P.parameters()
        )
        assert p_grads, "ProjectorP no recibió gradientes"

        # Q tiene gradientes
        q_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.Q.parameters()
        )
        assert q_grads, "TransitionQ no recibió gradientes"

        # Decoder tiene gradientes
        decoder_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.decoder.parameters()
        )
        assert decoder_grads, "Decoder no recibió gradientes"

    def test_backward_with_multiple_windows(self):
        """Gradientes fluyen correctamente con múltiples ventanas DGW."""
        config = VSNConfig.small()
        model = VSNModel(config)
        tokens = torch.randn(2, 16, config.d)

        outputs = model(tokens, num_windows=2)

        # Loss combinada de ambas ventanas
        loss = sum(s.sum() for s in outputs.states["decoder_states"])
        loss.backward()

        # Ψ debe recibir gradientes (conecta ventanas)
        psi_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.decoder.psi.parameters()
        )
        assert psi_grads, "PsiOperator no recibió gradientes con múltiples ventanas"


class TestVSNModelWithHead:
    """Tests con head externo."""

    def test_forward_with_simple_head(self):
        """Forward con head produce logits."""
        config = VSNConfig.small()

        # Head simple que produce logits desde decoder states
        class SimpleHead(torch.nn.Module):
            def __init__(self, d: int, vocab_size: int):
                super().__init__()
                self.proj = torch.nn.Linear(d, vocab_size)

            def forward(self, decoder_states, metadata):
                # Tomar último estado, hacer mean espacial, proyectar
                last_state = decoder_states[-1]  # (batch, Y, Z, d)
                pooled = last_state.mean(dim=(1, 2))  # (batch, d)
                logits = self.proj(pooled)  # (batch, vocab_size)
                return ModelOutputs(
                    logits=logits,
                    metadata=metadata,
                )

        head = SimpleHead(config.d, config.vocab_size)
        model = VSNModel(config, head=head)
        tokens = torch.randn(2, 16, config.d)

        outputs = model(tokens)

        assert outputs.logits is not None
        assert outputs.logits.shape == (2, config.vocab_size)
        # States se preservan cuando head no los incluye
        assert outputs.states is not None
