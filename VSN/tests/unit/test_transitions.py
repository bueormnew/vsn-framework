"""Tests unitarios para el Operador Q (TransitionQ).

Verifica:
- Correctitud de shapes: (B, Y_H, Z_H, d_H) → (B, Y_dec, Z_dec, d)
- Flujo de gradientes a través de la proyección
- Validación de errores en dimensiones inválidas
- Independencia de parámetros respecto a P
"""

import pytest
import torch

from vsn.core.latent import ProjectorP
from vsn.core.transitions import TransitionQ


class TestTransitionQShapes:
    """Verifica que Q transforma (B, Y_H, Z_H, d_H) → (B, Y_dec, Z_dec, d)."""

    def test_basic_shape(self):
        """Transformación básica con dimensiones distintas."""
        Y_H, Z_H, d_H = 2, 2, 4
        Y_dec, Z_dec, d = 4, 4, 8
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        H = torch.randn(2, Y_H, Z_H, d_H)
        V_dec_0 = Q(H)

        assert V_dec_0.shape == (2, Y_dec, Z_dec, d)

    def test_same_dimensions(self):
        """Funciona cuando entrada y salida tienen mismas dimensiones totales."""
        Y_H, Z_H, d_H = 4, 4, 8
        Y_dec, Z_dec, d = 4, 4, 8
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        H = torch.randn(3, Y_H, Z_H, d_H)
        V_dec_0 = Q(H)

        assert V_dec_0.shape == (3, Y_dec, Z_dec, d)

    def test_expand_dimensions(self):
        """Expande desde H pequeño a decoder grande."""
        Y_H, Z_H, d_H = 2, 2, 4  # in: 16
        Y_dec, Z_dec, d = 8, 8, 16  # out: 1024
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        H = torch.randn(2, Y_H, Z_H, d_H)
        V_dec_0 = Q(H)

        assert V_dec_0.shape == (2, Y_dec, Z_dec, d)

    def test_compress_dimensions(self):
        """Comprime desde H grande a decoder más pequeño."""
        Y_H, Z_H, d_H = 8, 8, 16  # in: 1024
        Y_dec, Z_dec, d = 4, 4, 8  # out: 128
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        H = torch.randn(2, Y_H, Z_H, d_H)
        V_dec_0 = Q(H)

        assert V_dec_0.shape == (2, Y_dec, Z_dec, d)

    def test_batch_size_one(self):
        """Funciona con batch_size=1."""
        Y_H, Z_H, d_H = 3, 3, 6
        Y_dec, Z_dec, d = 4, 4, 8
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        H = torch.randn(1, Y_H, Z_H, d_H)
        V_dec_0 = Q(H)

        assert V_dec_0.shape == (1, Y_dec, Z_dec, d)

    def test_large_batch(self):
        """Funciona con batch grande."""
        Y_H, Z_H, d_H = 2, 2, 4
        Y_dec, Z_dec, d = 4, 4, 8
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        H = torch.randn(32, Y_H, Z_H, d_H)
        V_dec_0 = Q(H)

        assert V_dec_0.shape == (32, Y_dec, Z_dec, d)


class TestTransitionQGradients:
    """Verifica que los gradientes fluyen a través de Q."""

    def test_gradients_flow_to_input(self):
        """Los gradientes fluyen hacia la entrada H."""
        Q = TransitionQ(2, 2, 4, 4, 4, 8)
        H = torch.randn(2, 2, 2, 4, requires_grad=True)

        V_dec_0 = Q(H)
        loss = V_dec_0.sum()
        loss.backward()

        assert H.grad is not None
        assert H.grad.shape == H.shape
        assert not torch.all(H.grad == 0)

    def test_gradients_flow_to_parameters(self):
        """Los parámetros de la capa lineal reciben gradientes."""
        Q = TransitionQ(2, 2, 4, 4, 4, 8)
        H = torch.randn(2, 2, 2, 4)

        V_dec_0 = Q(H)
        loss = V_dec_0.sum()
        loss.backward()

        assert Q.linear.weight.grad is not None
        assert Q.linear.bias.grad is not None
        assert not torch.all(Q.linear.weight.grad == 0)

    def test_gradients_expand_case(self):
        """Los gradientes fluyen en caso de expansión grande."""
        Q = TransitionQ(2, 2, 4, 8, 8, 16)
        H = torch.randn(2, 2, 2, 4, requires_grad=True)

        V_dec_0 = Q(H)
        loss = V_dec_0.mean()
        loss.backward()

        assert H.grad is not None
        assert H.grad.shape == H.shape


class TestTransitionQValidation:
    """Verifica que errores de configuración se detectan en __init__."""

    def test_zero_dimension_raises(self):
        """Dimensión cero lanza ValueError."""
        with pytest.raises(ValueError, match="Y_H debe ser un entero positivo"):
            TransitionQ(0, 2, 4, 4, 4, 8)

    def test_negative_dimension_raises(self):
        """Dimensión negativa lanza ValueError."""
        with pytest.raises(ValueError, match="Z_dec debe ser un entero positivo"):
            TransitionQ(2, 2, 4, 4, -1, 8)

    def test_non_integer_dimension_raises(self):
        """Dimensión no entera lanza ValueError."""
        with pytest.raises(ValueError, match="d debe ser un entero positivo"):
            TransitionQ(2, 2, 4, 4, 4, 2.5)  # type: ignore

    def test_all_zero_dims_raise(self):
        """Cualquier dimensión cero es rechazada."""
        with pytest.raises(ValueError):
            TransitionQ(2, 0, 4, 4, 4, 8)

        with pytest.raises(ValueError):
            TransitionQ(2, 2, 0, 4, 4, 8)

        with pytest.raises(ValueError):
            TransitionQ(2, 2, 4, 0, 4, 8)


class TestTransitionQIndependenceFromP:
    """Verifica que Q es completamente independiente de P (requisito 4.3)."""

    def test_no_shared_parameters(self):
        """P y Q no comparten ningún tensor de parámetros."""
        Y, Z, d = 4, 4, 8
        Y_H, Z_H, d_H = 2, 2, 4
        Y_dec, Z_dec = 4, 4

        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="compress")
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        p_ptrs = {p.data_ptr() for p in P.parameters()}
        q_ptrs = {p.data_ptr() for p in Q.parameters()}

        assert p_ptrs.isdisjoint(q_ptrs), "P y Q comparten parámetros"

    def test_independent_gradients(self):
        """Gradientes de Q no afectan a P y viceversa."""
        Y, Z, d = 4, 4, 8
        Y_H, Z_H, d_H = 2, 2, 4
        Y_dec, Z_dec = 4, 4

        P = ProjectorP(Y, Z, d, Y_H, Z_H, d_H, mode="compress")
        Q = TransitionQ(Y_H, Z_H, d_H, Y_dec, Z_dec, d)

        # Forward a través de P y Q en cadena
        x = torch.randn(2, Y, Z, d, requires_grad=True)
        H = P(x)
        V_dec_0 = Q(H)

        # Solo backprop a Q
        loss_q = V_dec_0.sum()
        loss_q.backward()

        # P también recibe gradientes (cadena completa), pero los pesos son distintos
        assert Q.linear.weight.grad is not None
        assert P.linear.weight.grad is not None

        # Los parámetros son objetos distintos
        assert Q.linear.weight is not P.linear.weight
        assert Q.linear.bias is not P.linear.bias


class TestTransitionQAttributes:
    """Verifica atributos y representación del módulo."""

    def test_in_out_features(self):
        """in_features y out_features se calculan correctamente."""
        Q = TransitionQ(2, 3, 4, 5, 6, 7)

        assert Q.in_features == 2 * 3 * 4  # 24
        assert Q.out_features == 5 * 6 * 7  # 210

    def test_extra_repr_format(self):
        """extra_repr muestra dimensiones de forma legible."""
        Q = TransitionQ(2, 3, 4, 5, 6, 7)
        repr_str = Q.extra_repr()

        assert "(2, 3, 4)" in repr_str
        assert "(5, 6, 7)" in repr_str

    def test_stored_dimensions(self):
        """Las dimensiones se almacenan como atributos accesibles."""
        Q = TransitionQ(2, 3, 4, 5, 6, 7)

        assert Q.Y_H == 2
        assert Q.Z_H == 3
        assert Q.d_H == 4
        assert Q.Y_dec == 5
        assert Q.Z_dec == 6
        assert Q.d == 7
