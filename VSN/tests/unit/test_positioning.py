"""Tests unitarios para el operador de posicionamiento Φ.

Verifica el mapeo determinista de tokens a volumen 3D según la Sección 6.1:
    x = ⌊ j / (Y·Z) ⌋
    i = j mod (Y·Z)
    y = i mod Y
    z = ⌊ i / Y ⌋
"""

import pytest
import torch

from vsn.core.positioning import PositioningOperator


class TestPositioningOperatorInit:
    """Tests de instanciación y validación de parámetros."""

    def test_basic_creation(self) -> None:
        op = PositioningOperator(X=2, Y=3, Z=4)
        assert op.X == 2
        assert op.Y == 3
        assert op.Z == 4
        assert op.capacity == 24

    def test_invalid_X_raises(self) -> None:
        with pytest.raises(ValueError, match="X debe ser positivo"):
            PositioningOperator(X=0, Y=3, Z=4)

    def test_invalid_Y_raises(self) -> None:
        with pytest.raises(ValueError, match="Y debe ser positivo"):
            PositioningOperator(X=2, Y=0, Z=4)

    def test_invalid_Z_raises(self) -> None:
        with pytest.raises(ValueError, match="Z debe ser positivo"):
            PositioningOperator(X=2, Y=3, Z=0)

    def test_negative_dims_raise(self) -> None:
        with pytest.raises(ValueError):
            PositioningOperator(X=-1, Y=3, Z=4)

    def test_no_learnable_parameters(self) -> None:
        op = PositioningOperator(X=2, Y=3, Z=4)
        params = list(op.parameters())
        assert len(params) == 0, "PositioningOperator no debe tener parámetros entrenables"


class TestPositioningOperatorForward:
    """Tests del forward: mapeo correcto y determinismo."""

    def test_output_shape_full(self) -> None:
        """Volumen completo: num_tokens = X*Y*Z."""
        X, Y, Z, d = 2, 3, 4, 8
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(1, X * Y * Z, d)
        volume = op(tokens)
        assert volume.shape == (1, X, Y, Z, d)

    def test_output_shape_partial(self) -> None:
        """Volumen parcial: num_tokens < X*Y*Z."""
        X, Y, Z, d = 2, 3, 4, 8
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(1, 5, d)
        volume = op(tokens)
        assert volume.shape == (1, X, Y, Z, d)

    def test_output_shape_batch(self) -> None:
        """Batch > 1."""
        X, Y, Z, d = 2, 3, 4, 8
        batch = 4
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(batch, X * Y * Z, d)
        volume = op(tokens)
        assert volume.shape == (batch, X, Y, Z, d)

    def test_empty_tokens(self) -> None:
        """Sin tokens: volumen todo ceros."""
        X, Y, Z, d = 2, 3, 4, 8
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(1, 0, d)
        volume = op(tokens)
        assert volume.shape == (1, X, Y, Z, d)
        assert torch.all(volume == 0)

    def test_exceeds_capacity_raises(self) -> None:
        """Más tokens que capacidad debe fallar."""
        X, Y, Z, d = 2, 3, 4, 8
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(1, X * Y * Z + 1, d)
        with pytest.raises(ValueError, match="excede la capacidad"):
            op(tokens)

    def test_invalid_ndim_raises(self) -> None:
        """Input no 3D debe fallar."""
        op = PositioningOperator(X=2, Y=3, Z=4)
        with pytest.raises(ValueError, match="debe ser 3D"):
            op(torch.randn(10, 8))  # 2D

    def test_first_token_at_origin(self) -> None:
        """Token j=0 va a posición (x=0, y=0, z=0)."""
        X, Y, Z, d = 2, 3, 4, 8
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.zeros(1, 1, d)
        tokens[0, 0, :] = 1.0  # token marcado
        volume = op(tokens)
        # j=0: x=0, i=0, y=0, z=0
        assert torch.allclose(volume[0, 0, 0, 0, :], torch.ones(d))

    def test_raster_order_y_first(self) -> None:
        """Dentro de un plano, Y se llena primero (raster order).

        Con Y=3, Z=2:
            j=0 → (x=0, y=0, z=0)
            j=1 → (x=0, y=1, z=0)
            j=2 → (x=0, y=2, z=0)
            j=3 → (x=0, y=0, z=1)
            j=4 → (x=0, y=1, z=1)
            j=5 → (x=0, y=2, z=1)
        """
        X, Y, Z, d = 1, 3, 2, 1
        op = PositioningOperator(X=X, Y=Y, Z=Z)

        # Crear tokens con valores = j para identificación
        num_tokens = X * Y * Z  # 6
        tokens = torch.arange(num_tokens, dtype=torch.float32).view(1, num_tokens, 1)
        volume = op(tokens)

        # Verificar mapeo según fórmula
        assert volume[0, 0, 0, 0, 0].item() == 0.0  # j=0
        assert volume[0, 0, 1, 0, 0].item() == 1.0  # j=1
        assert volume[0, 0, 2, 0, 0].item() == 2.0  # j=2
        assert volume[0, 0, 0, 1, 0].item() == 3.0  # j=3
        assert volume[0, 0, 1, 1, 0].item() == 4.0  # j=4
        assert volume[0, 0, 2, 1, 0].item() == 5.0  # j=5

    def test_plane_transition(self) -> None:
        """Tokens pasan al siguiente plano cuando Y*Z se llena.

        Con X=2, Y=2, Z=2 (plane_size=4):
            j=0..3 → x=0
            j=4..7 → x=1
        """
        X, Y, Z, d = 2, 2, 2, 1
        op = PositioningOperator(X=X, Y=Y, Z=Z)

        num_tokens = X * Y * Z  # 8
        tokens = torch.arange(num_tokens, dtype=torch.float32).view(1, num_tokens, 1)
        volume = op(tokens)

        # Primer plano (x=0): j=0,1,2,3
        assert volume[0, 0, 0, 0, 0].item() == 0.0
        assert volume[0, 0, 1, 0, 0].item() == 1.0
        assert volume[0, 0, 0, 1, 0].item() == 2.0
        assert volume[0, 0, 1, 1, 0].item() == 3.0

        # Segundo plano (x=1): j=4,5,6,7
        assert volume[0, 1, 0, 0, 0].item() == 4.0
        assert volume[0, 1, 1, 0, 0].item() == 5.0
        assert volume[0, 1, 0, 1, 0].item() == 6.0
        assert volume[0, 1, 1, 1, 0].item() == 7.0

    def test_partial_fill_zeros_padding(self) -> None:
        """Posiciones sin token asignado deben ser ceros."""
        X, Y, Z, d = 2, 2, 2, 4
        op = PositioningOperator(X=X, Y=Y, Z=Z)

        # Solo 3 tokens de 8 posibles
        tokens = torch.ones(1, 3, d)
        volume = op(tokens)

        # j=0,1,2 ocupados (valor 1)
        assert torch.all(volume[0, 0, 0, 0, :] == 1.0)  # j=0
        assert torch.all(volume[0, 0, 1, 0, :] == 1.0)  # j=1
        assert torch.all(volume[0, 0, 0, 1, :] == 1.0)  # j=2

        # j=3..7 vacíos (valor 0)
        assert torch.all(volume[0, 0, 1, 1, :] == 0.0)  # j=3
        assert torch.all(volume[0, 1, :, :, :] == 0.0)   # plano x=1 vacío


class TestPositioningOperatorDeterminism:
    """Tests de determinismo: misma entrada → misma salida."""

    def test_same_input_same_output(self) -> None:
        """Ejecutar Φ dos veces con la misma entrada produce resultado idéntico."""
        X, Y, Z, d = 3, 4, 5, 16
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(2, 30, d)

        volume1 = op(tokens)
        volume2 = op(tokens)

        assert torch.equal(volume1, volume2)

    def test_deterministic_across_instances(self) -> None:
        """Dos instancias con mismos parámetros producen mismo resultado."""
        X, Y, Z, d = 3, 4, 5, 16
        op1 = PositioningOperator(X=X, Y=Y, Z=Z)
        op2 = PositioningOperator(X=X, Y=Y, Z=Z)
        tokens = torch.randn(2, 30, d)

        volume1 = op1(tokens)
        volume2 = op2(tokens)

        assert torch.equal(volume1, volume2)

    def test_no_mutable_state(self) -> None:
        """El operador no tiene estado mutable — aplicaciones sucesivas son independientes."""
        X, Y, Z, d = 2, 3, 4, 8
        op = PositioningOperator(X=X, Y=Y, Z=Z)

        tokens_a = torch.randn(1, 10, d)
        tokens_b = torch.randn(1, 15, d)

        # Ejecutar con A, luego B, luego A otra vez
        vol_a1 = op(tokens_a)
        _ = op(tokens_b)
        vol_a2 = op(tokens_a)

        assert torch.equal(vol_a1, vol_a2)


class TestPositioningOperatorFormula:
    """Tests que verifican la fórmula matemática exacta de la spec."""

    @pytest.mark.parametrize("X,Y,Z", [(1, 1, 1), (2, 3, 4), (4, 4, 4), (1, 5, 3)])
    def test_formula_all_positions(self, X: int, Y: int, Z: int) -> None:
        """Verifica que cada token j se mapea a la posición correcta según la fórmula."""
        d = 1
        op = PositioningOperator(X=X, Y=Y, Z=Z)
        num_tokens = X * Y * Z

        # Cada token tiene valor = j+1 (para distinguir de ceros)
        tokens = (torch.arange(num_tokens, dtype=torch.float32) + 1).view(1, num_tokens, d)
        volume = op(tokens)

        for j in range(num_tokens):
            # Fórmula de la spec
            x = j // (Y * Z)
            i = j % (Y * Z)
            y = i % Y
            z = i // Y

            expected_value = float(j + 1)
            actual_value = volume[0, x, y, z, 0].item()
            assert actual_value == expected_value, (
                f"j={j}: esperado volume[0,{x},{y},{z},0]={expected_value}, "
                f"obtenido {actual_value}"
            )
