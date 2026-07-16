"""Tests unitarios para RMSNorm.

Valida:
- Output shape matches input shape
- Scale parameter es entrenable (requires_grad=True)
- Normalización correcta para inputs conocidos
"""

import torch
import math

from vsn.core.rms_norm import RMSNorm


class TestRMSNormShape:
    """Verifica que el output shape coincide con el input shape."""

    def test_2d_input(self):
        norm = RMSNorm(d=8)
        x = torch.randn(4, 8)
        out = norm(x)
        assert out.shape == x.shape

    def test_3d_input(self):
        norm = RMSNorm(d=16)
        x = torch.randn(2, 5, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_4d_input_yz_d(self):
        """Shape (batch, Y, Z, d) — caso principal del VGB."""
        norm = RMSNorm(d=32)
        x = torch.randn(2, 4, 4, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_1d_input(self):
        """Un solo vector de dimensión d."""
        norm = RMSNorm(d=5)
        x = torch.randn(5)
        out = norm(x)
        assert out.shape == x.shape


class TestRMSNormLearnableScale:
    """Verifica que el parámetro scale es entrenable."""

    def test_scale_requires_grad(self):
        norm = RMSNorm(d=16)
        assert norm.scale.requires_grad is True

    def test_scale_shape(self):
        norm = RMSNorm(d=32)
        assert norm.scale.shape == (32,)

    def test_scale_initialized_to_ones(self):
        norm = RMSNorm(d=8)
        assert torch.allclose(norm.scale, torch.ones(8))

    def test_scale_is_nn_parameter(self):
        norm = RMSNorm(d=4)
        params = list(norm.parameters())
        assert len(params) == 1
        assert params[0] is norm.scale


class TestRMSNormCorrectness:
    """Verifica la correctitud numérica de la normalización."""

    def test_known_input_uniform(self):
        """Para un vector constante [c, c, ..., c], RMS = |c|.
        Resultado esperado: sign(c) * scale = scale (para c > 0).
        """
        d = 4
        norm = RMSNorm(d=d, eps=0.0)
        # Con scale = 1 y eps = 0, normalizar un vector constante c > 0
        # da x / RMS(x) = c / c = 1 para cada componente
        x = torch.full((d,), 3.0)
        out = norm(x)
        expected = torch.ones(d)  # 3.0 / sqrt(mean(9.0)) = 3/3 = 1
        assert torch.allclose(out, expected, atol=1e-6)

    def test_known_input_manual(self):
        """Verifica contra cálculo manual."""
        d = 4
        norm = RMSNorm(d=d, eps=1e-6)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # RMS manual: sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + 1e-6)
        rms = math.sqrt(7.5 + 1e-6)
        expected = torch.tensor([1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms])

        out = norm(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_with_custom_scale(self):
        """Verifica que scale multiplica correctamente."""
        d = 4
        norm = RMSNorm(d=d, eps=0.0)
        # Configurar scale = [2, 2, 2, 2]
        with torch.no_grad():
            norm.scale.fill_(2.0)

        x = torch.full((d,), 3.0)
        out = norm(x)
        # x/RMS(x) = 1, * scale = 2
        expected = torch.full((d,), 2.0)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_gradient_flows(self):
        """Verifica que los gradientes fluyen a través de RMSNorm."""
        norm = RMSNorm(d=8)
        x = torch.randn(2, 8, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.scale.grad is not None
