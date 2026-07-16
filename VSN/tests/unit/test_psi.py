"""Tests unitarios para el Operador Ψ (PsiOperator).

Verifica:
- Correctitud de shapes de salida: V_dec_0_next y M_next como (batch, Y, Z, d)
- Flujo de gradientes (diferenciabilidad)
- Serialización round-trip via state_dict
- Componentes internos correctos
- Validación de atributos y dimensiones
"""

import pytest
import torch

from vsn.core.psi import PsiOperator


class TestPsiOperatorShapes:
    """Verifica que Ψ produce shapes correctas."""

    def test_basic_shape(self):
        """Salidas V_dec_0_next y M_next tienen shape (batch, Y, Z, d)."""
        Y, Z, d = 4, 4, 8
        psi = PsiOperator(Y, Z, d)

        decoder_vol = torch.randn(2, Y, Z, d)
        memory = torch.randn(2, Y, Z, d)
        output_summary = torch.randn(2, d)

        v_next, m_next = psi(decoder_vol, memory, output_summary)

        assert v_next.shape == (2, Y, Z, d)
        assert m_next.shape == (2, Y, Z, d)

    def test_batch_size_one(self):
        """Funciona con batch_size=1."""
        Y, Z, d = 3, 3, 6
        psi = PsiOperator(Y, Z, d)

        decoder_vol = torch.randn(1, Y, Z, d)
        memory = torch.randn(1, Y, Z, d)
        output_summary = torch.randn(1, d)

        v_next, m_next = psi(decoder_vol, memory, output_summary)

        assert v_next.shape == (1, Y, Z, d)
        assert m_next.shape == (1, Y, Z, d)

    def test_large_batch(self):
        """Funciona con batch grande."""
        Y, Z, d = 4, 4, 8
        psi = PsiOperator(Y, Z, d)

        decoder_vol = torch.randn(32, Y, Z, d)
        memory = torch.randn(32, Y, Z, d)
        output_summary = torch.randn(32, d)

        v_next, m_next = psi(decoder_vol, memory, output_summary)

        assert v_next.shape == (32, Y, Z, d)
        assert m_next.shape == (32, Y, Z, d)

    def test_small_dimensions(self):
        """Funciona con dimensiones mínimas Y=1, Z=1, d=1."""
        Y, Z, d = 1, 1, 1
        psi = PsiOperator(Y, Z, d)

        decoder_vol = torch.randn(2, Y, Z, d)
        memory = torch.randn(2, Y, Z, d)
        output_summary = torch.randn(2, d)

        v_next, m_next = psi(decoder_vol, memory, output_summary)

        assert v_next.shape == (2, Y, Z, d)
        assert m_next.shape == (2, Y, Z, d)

    def test_asymmetric_dimensions(self):
        """Funciona con Y != Z."""
        Y, Z, d = 3, 7, 16
        psi = PsiOperator(Y, Z, d)

        decoder_vol = torch.randn(4, Y, Z, d)
        memory = torch.randn(4, Y, Z, d)
        output_summary = torch.randn(4, d)

        v_next, m_next = psi(decoder_vol, memory, output_summary)

        assert v_next.shape == (4, Y, Z, d)
        assert m_next.shape == (4, Y, Z, d)


class TestPsiOperatorGradients:
    """Verifica que los gradientes fluyen a través de Ψ."""

    def test_gradients_flow_to_decoder_volume(self):
        """Los gradientes fluyen hacia decoder_volume_final."""
        psi = PsiOperator(4, 4, 8)
        decoder_vol = torch.randn(2, 4, 4, 8, requires_grad=True)
        memory = torch.randn(2, 4, 4, 8)
        output_summary = torch.randn(2, 8)

        v_next, m_next = psi(decoder_vol, memory, output_summary)
        loss = v_next.sum() + m_next.sum()
        loss.backward()

        assert decoder_vol.grad is not None
        assert decoder_vol.grad.shape == decoder_vol.shape
        assert not torch.all(decoder_vol.grad == 0)

    def test_gradients_flow_to_memory(self):
        """Los gradientes fluyen hacia memory_final."""
        psi = PsiOperator(4, 4, 8)
        decoder_vol = torch.randn(2, 4, 4, 8)
        memory = torch.randn(2, 4, 4, 8, requires_grad=True)
        output_summary = torch.randn(2, 8)

        v_next, m_next = psi(decoder_vol, memory, output_summary)
        loss = v_next.sum() + m_next.sum()
        loss.backward()

        assert memory.grad is not None
        assert memory.grad.shape == memory.shape
        assert not torch.all(memory.grad == 0)

    def test_gradients_flow_to_output_summary(self):
        """Los gradientes fluyen hacia recent_output_summary."""
        psi = PsiOperator(4, 4, 8)
        decoder_vol = torch.randn(2, 4, 4, 8)
        memory = torch.randn(2, 4, 4, 8)
        output_summary = torch.randn(2, 8, requires_grad=True)

        v_next, m_next = psi(decoder_vol, memory, output_summary)
        loss = v_next.sum() + m_next.sum()
        loss.backward()

        assert output_summary.grad is not None
        assert output_summary.grad.shape == output_summary.shape
        assert not torch.all(output_summary.grad == 0)

    def test_gradients_flow_to_all_parameters(self):
        """Todos los parámetros de Ψ reciben gradientes."""
        psi = PsiOperator(4, 4, 8)
        decoder_vol = torch.randn(2, 4, 4, 8)
        memory = torch.randn(2, 4, 4, 8)
        output_summary = torch.randn(2, 8)

        v_next, m_next = psi(decoder_vol, memory, output_summary)
        loss = v_next.sum() + m_next.sum()
        loss.backward()

        for name, param in psi.named_parameters():
            assert param.grad is not None, f"Parámetro {name} no recibió gradiente"
            assert not torch.all(param.grad == 0), f"Gradiente de {name} es todo ceros"


class TestPsiOperatorSerialization:
    """Verifica serialización round-trip via state_dict."""

    def test_state_dict_round_trip(self):
        """Guardar y cargar state_dict preserva parámetros exactamente."""
        Y, Z, d = 4, 4, 8
        psi_original = PsiOperator(Y, Z, d)

        # Modificar parámetros para que no sean los de inicialización
        with torch.no_grad():
            for param in psi_original.parameters():
                param.fill_(0.42)

        # Guardar state_dict
        state = psi_original.state_dict()

        # Crear nueva instancia y cargar
        psi_loaded = PsiOperator(Y, Z, d)
        psi_loaded.load_state_dict(state)

        # Verificar que todos los parámetros son exactamente iguales
        for (name_orig, p_orig), (name_load, p_load) in zip(
            psi_original.named_parameters(), psi_loaded.named_parameters()
        ):
            assert name_orig == name_load
            assert torch.equal(p_orig, p_load), (
                f"Parámetro {name_orig} difiere tras round-trip"
            )

    def test_state_dict_via_file(self, tmp_path):
        """Round-trip completo con guardado a disco."""
        Y, Z, d = 3, 5, 12
        psi_original = PsiOperator(Y, Z, d)

        # Guardar a archivo
        path = tmp_path / "psi_state.pt"
        torch.save(psi_original.state_dict(), path)

        # Cargar desde archivo
        psi_loaded = PsiOperator(Y, Z, d)
        psi_loaded.load_state_dict(torch.load(path, weights_only=True))

        # Verificar igualdad exacta
        for (name, p_orig), (_, p_load) in zip(
            psi_original.named_parameters(), psi_loaded.named_parameters()
        ):
            assert torch.equal(p_orig, p_load), (
                f"Parámetro {name} difiere tras save/load a disco"
            )

    def test_outputs_match_after_load(self, tmp_path):
        """El modelo cargado produce las mismas salidas que el original."""
        Y, Z, d = 4, 4, 8
        psi_original = PsiOperator(Y, Z, d)
        psi_original.eval()

        # Input fijo
        torch.manual_seed(42)
        decoder_vol = torch.randn(2, Y, Z, d)
        memory = torch.randn(2, Y, Z, d)
        output_summary = torch.randn(2, d)

        # Forward original
        with torch.no_grad():
            v_orig, m_orig = psi_original(decoder_vol, memory, output_summary)

        # Guardar y recargar
        path = tmp_path / "psi_state.pt"
        torch.save(psi_original.state_dict(), path)

        psi_loaded = PsiOperator(Y, Z, d)
        psi_loaded.load_state_dict(torch.load(path, weights_only=True))
        psi_loaded.eval()

        # Forward cargado
        with torch.no_grad():
            v_load, m_load = psi_loaded(decoder_vol, memory, output_summary)

        assert torch.equal(v_orig, v_load)
        assert torch.equal(m_orig, m_load)


class TestPsiOperatorComponents:
    """Verifica que los componentes internos son correctos."""

    def test_has_volume_summarizer(self):
        """Tiene componente volume_summarizer con dimensiones correctas."""
        psi = PsiOperator(4, 4, 8)
        assert hasattr(psi, "volume_summarizer")
        assert psi.volume_summarizer.in_features == 4 * 4 * 8
        assert psi.volume_summarizer.out_features == 8

    def test_has_memory_transform(self):
        """Tiene componente memory_transform con dimensiones correctas."""
        psi = PsiOperator(4, 4, 8)
        assert hasattr(psi, "memory_transform")
        assert psi.memory_transform.in_features == 4 * 4 * 8
        assert psi.memory_transform.out_features == 8

    def test_has_output_summarizer(self):
        """Tiene componente output_summarizer con dimensiones correctas."""
        psi = PsiOperator(4, 4, 8)
        assert hasattr(psi, "output_summarizer")
        assert psi.output_summarizer.in_features == 8
        assert psi.output_summarizer.out_features == 8

    def test_has_gate(self):
        """Tiene componente gate con dimensiones correctas (3*d → d)."""
        psi = PsiOperator(4, 4, 8)
        assert hasattr(psi, "gate")
        assert psi.gate.in_features == 3 * 8
        assert psi.gate.out_features == 8

    def test_has_state_projector(self):
        """Tiene componente state_projector con dimensiones correctas."""
        psi = PsiOperator(4, 4, 8)
        assert hasattr(psi, "state_projector")
        assert psi.state_projector.in_features == 8
        assert psi.state_projector.out_features == 4 * 4 * 8

    def test_has_memory_projector(self):
        """Tiene componente memory_projector con dimensiones correctas."""
        psi = PsiOperator(4, 4, 8)
        assert hasattr(psi, "memory_projector")
        assert psi.memory_projector.in_features == 8
        assert psi.memory_projector.out_features == 4 * 4 * 8

    def test_total_parameter_count(self):
        """Verifica que el módulo tiene un número razonable de parámetros."""
        Y, Z, d = 4, 4, 8
        psi = PsiOperator(Y, Z, d)
        flat_dim = Y * Z * d  # 128

        # volume_summarizer: flat_dim*d + d = 128*8 + 8 = 1032
        # memory_transform: flat_dim*d + d = 1032
        # output_summarizer: d*d + d = 72
        # gate: 3*d*d + d = 200
        # state_projector: d*flat_dim + flat_dim = 1152
        # memory_projector: d*flat_dim + flat_dim = 1152
        total = sum(p.numel() for p in psi.parameters())
        assert total > 0


class TestPsiOperatorAttributes:
    """Verifica atributos almacenados."""

    def test_stored_dimensions(self):
        """Las dimensiones Y, Z, d se almacenan como atributos."""
        psi = PsiOperator(3, 5, 7)
        assert psi.Y == 3
        assert psi.Z == 5
        assert psi.d == 7

    def test_is_nn_module(self):
        """PsiOperator es un nn.Module válido."""
        psi = PsiOperator(4, 4, 8)
        assert isinstance(psi, torch.nn.Module)

    def test_eval_mode(self):
        """Se puede poner en modo eval sin error."""
        psi = PsiOperator(4, 4, 8)
        psi.eval()
        # Forward en eval mode
        decoder_vol = torch.randn(1, 4, 4, 8)
        memory = torch.randn(1, 4, 4, 8)
        output_summary = torch.randn(1, 8)
        v_next, m_next = psi(decoder_vol, memory, output_summary)
        assert v_next.shape == (1, 4, 4, 8)

    def test_train_mode(self):
        """Se puede poner en modo train sin error."""
        psi = PsiOperator(4, 4, 8)
        psi.train()
        decoder_vol = torch.randn(1, 4, 4, 8)
        memory = torch.randn(1, 4, 4, 8)
        output_summary = torch.randn(1, 8)
        v_next, m_next = psi(decoder_vol, memory, output_summary)
        assert v_next.shape == (1, 4, 4, 8)
