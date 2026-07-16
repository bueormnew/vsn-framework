"""Feature: vsn-library, Properties 9, 10: Ψ Operator Property Tests

**Validates: Requirements 5.1-5.7, 13.7**

Tests de propiedad que verifican:
- Propiedad 9: Para cualquier estado final de ventana (decoder_volume_final (B,Y,Z,d),
  memory_final (B,Y,Z,d), recent_output_summary (B,d)), PsiOperator SHALL producir
  V_dec_0_next (B,Y,Z,d) y M_next (B,Y,Z,d). Los gradientes SHALL fluir a través de Ψ.
- Propiedad 10: Para cualquier PsiOperator con parámetros entrenados, save/load
  state_dict SHALL producir parámetros exactamente iguales (torch.equal).
"""

import io
import tempfile
from pathlib import Path

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.core.psi import PsiOperator

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones válidas
# ---------------------------------------------------------------------------

batch_strategy = st.integers(min_value=1, max_value=4)
spatial_strategy = st.integers(min_value=1, max_value=6)
d_strategy = st.integers(min_value=2, max_value=16)


@st.composite
def psi_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para PsiOperator.

    Mantiene dimensiones moderadas para que los tests completen
    en tiempo razonable (Y*Z*d cabe en Linear layers).
    """
    B = draw(batch_strategy)
    Y = draw(spatial_strategy)
    Z = draw(spatial_strategy)
    d = draw(d_strategy)
    return {"batch": B, "Y": Y, "Z": Z, "d": d}


# ---------------------------------------------------------------------------
# Propiedad 9: Ψ preserva shapes y es diferenciable
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=psi_dims())
def test_psi_preserves_output_shapes(dims: dict) -> None:
    """Feature: vsn-library, Property 9: Ψ preserva shapes y es diferenciable

    **Validates: Requirements 5.4, 5.5**

    Para cualquier estado (decoder_volume_final (B,Y,Z,d), memory_final (B,Y,Z,d),
    recent_output_summary (B,d)), PsiOperator SHALL producir V_dec_0_next de shape
    (B,Y,Z,d) y M_next de shape (B,Y,Z,d).
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]

    psi = PsiOperator(Y=Y, Z=Z, d=d)
    psi.eval()

    decoder_volume_final = torch.randn(B, Y, Z, d)
    memory_final = torch.randn(B, Y, Z, d)
    recent_output_summary = torch.randn(B, d)

    with torch.no_grad():
        V_dec_0_next, M_next = psi(
            decoder_volume_final, memory_final, recent_output_summary
        )

    expected_volume_shape = (B, Y, Z, d)
    assert V_dec_0_next.shape == expected_volume_shape, (
        f"V_dec_0_next shape {V_dec_0_next.shape} != expected {expected_volume_shape}"
    )
    assert M_next.shape == expected_volume_shape, (
        f"M_next shape {M_next.shape} != expected {expected_volume_shape}"
    )
    assert torch.isfinite(V_dec_0_next).all(), "V_dec_0_next contains NaN or Inf"
    assert torch.isfinite(M_next).all(), "M_next contains NaN or Inf"


@settings(max_examples=100, deadline=None)
@given(dims=psi_dims())
def test_psi_gradients_flow_through(dims: dict) -> None:
    """Feature: vsn-library, Property 9: Ψ preserva shapes y es diferenciable

    **Validates: Requirements 5.4, 5.5**

    Los gradientes SHALL fluir a través de Ψ hasta sus parámetros.
    Verificamos que backward() produce gradientes no nulos en los parámetros de Ψ.
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]

    psi = PsiOperator(Y=Y, Z=Z, d=d)
    psi.train()

    # Inputs requieren grad para verificar flujo completo
    decoder_volume_final = torch.randn(B, Y, Z, d, requires_grad=True)
    memory_final = torch.randn(B, Y, Z, d, requires_grad=True)
    recent_output_summary = torch.randn(B, d, requires_grad=True)

    V_dec_0_next, M_next = psi(
        decoder_volume_final, memory_final, recent_output_summary
    )

    # Crear loss escalar para backward
    loss = V_dec_0_next.sum() + M_next.sum()
    loss.backward()

    # Verificar que los gradientes fluyen hacia los parámetros de Ψ
    params_with_grad = 0
    total_params = 0
    for name, param in psi.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1

    assert params_with_grad > 0, (
        f"Ningún parámetro de Ψ recibió gradientes. "
        f"Total params: {total_params}, con grad: {params_with_grad}"
    )

    # Verificar que los gradientes fluyen hacia los inputs
    assert decoder_volume_final.grad is not None, (
        "decoder_volume_final no recibió gradientes a través de Ψ"
    )
    assert memory_final.grad is not None, (
        "memory_final no recibió gradientes a través de Ψ"
    )
    assert recent_output_summary.grad is not None, (
        "recent_output_summary no recibió gradientes a través de Ψ"
    )


# ---------------------------------------------------------------------------
# Propiedad 10: Serialización round-trip de Ψ
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=psi_dims())
def test_psi_serialization_roundtrip_state_dict(dims: dict) -> None:
    """Feature: vsn-library, Property 10: Serialización round-trip de Ψ

    **Validates: Requirements 5.5, 13.7**

    Para cualquier PsiOperator con parámetros entrenados, guardar y cargar
    state_dict SHALL producir parámetros exactamente iguales (torch.equal).
    """
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]

    # Crear Ψ con parámetros "entrenados" (modificados de su inicialización)
    psi_original = PsiOperator(Y=Y, Z=Z, d=d)

    # Simular entrenamiento: modificar parámetros con valores aleatorios
    with torch.no_grad():
        for param in psi_original.parameters():
            param.copy_(torch.randn_like(param))

    # Guardar state_dict
    original_state = psi_original.state_dict()

    # Crear nueva instancia y cargar state_dict
    psi_loaded = PsiOperator(Y=Y, Z=Z, d=d)
    psi_loaded.load_state_dict(original_state)

    # Verificar que TODOS los parámetros son exactamente iguales
    for key in original_state:
        original_param = original_state[key]
        loaded_param = psi_loaded.state_dict()[key]
        assert torch.equal(original_param, loaded_param), (
            f"Parámetro '{key}' no es idéntico después del round-trip. "
            f"Max diff: {(original_param - loaded_param).abs().max().item()}"
        )


@settings(max_examples=100, deadline=None)
@given(dims=psi_dims())
def test_psi_serialization_roundtrip_file(dims: dict) -> None:
    """Feature: vsn-library, Property 10: Serialización round-trip de Ψ (via archivo)

    **Validates: Requirements 5.5, 13.7**

    Para cualquier PsiOperator, guardar a archivo (torch.save) y recargar
    (torch.load → load_state_dict) SHALL producir parámetros exactamente iguales.
    """
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]

    psi_original = PsiOperator(Y=Y, Z=Z, d=d)

    # Simular entrenamiento
    with torch.no_grad():
        for param in psi_original.parameters():
            param.copy_(torch.randn_like(param))

    # Round-trip via buffer en memoria (equivalente a archivo)
    buffer = io.BytesIO()
    torch.save(psi_original.state_dict(), buffer)
    buffer.seek(0)

    loaded_state = torch.load(buffer, weights_only=True)

    # Crear nueva instancia y cargar
    psi_loaded = PsiOperator(Y=Y, Z=Z, d=d)
    psi_loaded.load_state_dict(loaded_state)

    # Verificar igualdad exacta
    for (name_orig, param_orig), (name_loaded, param_loaded) in zip(
        psi_original.named_parameters(), psi_loaded.named_parameters()
    ):
        assert name_orig == name_loaded, (
            f"Nombre de parámetro cambió: {name_orig} vs {name_loaded}"
        )
        assert torch.equal(param_orig, param_loaded), (
            f"Parámetro '{name_orig}' no es idéntico tras save/load a archivo. "
            f"Max diff: {(param_orig - param_loaded).abs().max().item()}"
        )
