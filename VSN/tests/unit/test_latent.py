"""Tests unitarios para el Operador P (ProjectorP).

Verifica:
- Correctitud de shapes en los tres modos (compress, identity, expand)
- Flujo de gradientes a través de la proyección
- Validación de errores en modos inválidos/inconsistentes
- Inicialización identity-like para mode='identity'
"""

import pytest
import torch

from vsn.core.latent import ProjectorP


class TestProjectorPShapes:
    """Verifica que P transforma (B, Y, Z, d) → (B, Y_H, Z_H, d_H)."""

    def test_compress_mode_shape(self):
        """Compress: dimensiones de salida menores que entrada."""
        Y, Z, d = 4, 4, 8  # in: 4*4*8 = 128
        Y_H, Z_H, d_H = 2, 2, 4  # out: 2*2*4 = 16
        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="compress")

        x = torch.randn(2, Y, Z, d)
        H = P(x)

        assert H.shape == (2, Y_H, Z_H, d_H)

    def test_identity_mode_shape(self):
        """Identity: mismas dimensiones totales."""
        Y, Z, d = 4, 4, 8  # in: 128
        Y_H, Z_H, d_H = 4, 4, 8  # out: 128 (mismas dims)
        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="identity")

        x = torch.randn(3, Y, Z, d)
        H = P(x)

        assert H.shape == (3, Y_H, Z_H, d_H)

    def test_identity_mode_different_arrangement(self):
        """Identity: mismas dimensiones totales pero distinta disposición."""
        Y, Z, d = 4, 4, 8  # in: 128
        Y_H, Z_H, d_H = 8, 2, 8  # out: 128 (misma cantidad total)
        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="identity")

        x = torch.randn(2, Y, Z, d)
        H = P(x)

        assert H.shape == (2, Y_H, Z_H, d_H)

    def test_expand_mode_shape(self):
        """Expand: dimensiones de salida mayores que entrada."""
        Y, Z, d = 2, 2, 4  # in: 2*2*4 = 16
        Y_H, Z_H, d_H = 4, 4, 8  # out: 4*4*8 = 128
        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="expand")

        x = torch.randn(2, Y, Z, d)
        H = P(x)

        assert H.shape == (2, Y_H, Z_H, d_H)

    def test_batch_size_one(self):
        """Funciona con batch_size=1."""
        Y, Z, d = 3, 3, 6
        Y_H, Z_H, d_H = 2, 2, 4
        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="compress")

        x = torch.randn(1, Y, Z, d)
        H = P(x)

        assert H.shape == (1, Y_H, Z_H, d_H)


class TestProjectorPGradients:
    """Verifica que los gradientes fluyen a través de P."""

    def test_gradients_flow_compress(self):
        """Los gradientes fluyen en modo compress."""
        P = ProjectorP(4, 4, 8, 2, 2, 4, mode="compress")
        x = torch.randn(2, 4, 4, 8, requires_grad=True)

        H = P(x)
        loss = H.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)

    def test_gradients_flow_identity(self):
        """Los gradientes fluyen en modo identity."""
        P = ProjectorP(4, 4, 8, 4, 4, 8, mode="identity")
        x = torch.randn(2, 4, 4, 8, requires_grad=True)

        H = P(x)
        loss = H.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradients_flow_expand(self):
        """Los gradientes fluyen en modo expand."""
        P = ProjectorP(2, 2, 4, 4, 4, 8, mode="expand")
        x = torch.randn(2, 2, 2, 4, requires_grad=True)

        H = P(x)
        loss = H.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameters_have_gradients(self):
        """Los parámetros de la capa lineal reciben gradientes."""
        P = ProjectorP(4, 4, 8, 2, 2, 4, mode="compress")
        x = torch.randn(2, 4, 4, 8)

        H = P(x)
        loss = H.sum()
        loss.backward()

        assert P.linear.weight.grad is not None
        assert P.linear.bias.grad is not None


class TestProjectorPValidation:
    """Verifica que errores de configuración se detectan en __init__."""

    def test_invalid_mode_raises(self):
        """Modo inválido lanza ValueError."""
        with pytest.raises(ValueError, match="mode debe ser uno de"):
            ProjectorP(4, 4, 8, 2, 2, 4, mode="invalid")

    def test_compress_with_larger_output_raises(self):
        """mode='compress' con output >= input lanza ValueError."""
        with pytest.raises(ValueError, match="mode='compress'"):
            ProjectorP(2, 2, 4, 4, 4, 8, mode="compress")

    def test_expand_with_smaller_output_raises(self):
        """mode='expand' con output <= input lanza ValueError."""
        with pytest.raises(ValueError, match="mode='expand'"):
            ProjectorP(4, 4, 8, 2, 2, 4, mode="expand")

    def test_identity_with_different_total_raises(self):
        """mode='identity' con total distinto lanza ValueError."""
        with pytest.raises(ValueError, match="mode='identity'"):
            ProjectorP(4, 4, 8, 3, 3, 8, mode="identity")


class TestProjectorPIdentityInit:
    """Verifica inicialización identity-like para mode='identity'."""

    def test_identity_init_near_passthrough(self):
        """Con inicialización identity, la salida es cercana a un reshape de la entrada."""
        Y, Z, d = 4, 4, 8
        P = ProjectorP(Y, Z, d, Y, Z, d, mode="identity")

        x = torch.randn(2, Y, Z, d)
        H = P(x)

        # Con peso = I y bias = 0, H debería ser igual a x (reshape round-trip)
        torch.testing.assert_close(H, x, atol=1e-5, rtol=1e-5)
