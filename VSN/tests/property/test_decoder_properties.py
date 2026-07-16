"""Feature: vsn-library, Property 21: Decoder genera ventanas DGW correctamente

**Validates: Requirements 5.1-5.7, 13.7**

Tests de propiedad que verifican:
- Propiedad 21: Para cualquier decoder con DGW window size W, el decoder SHALL
  producir exactamente num_windows outputs. Cuando Ψ es invocado entre ventanas,
  SHALL producir estados iniciales válidos (finitos, shapes correctos).
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.core.decoder import VSNDecoder
from vsn.core.psi import PsiOperator

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones válidas del decoder
# ---------------------------------------------------------------------------

batch_strategy = st.integers(min_value=1, max_value=3)
spatial_strategy = st.integers(min_value=1, max_value=4)
d_strategy = st.integers(min_value=2, max_value=8)
x_dec_strategy = st.integers(min_value=1, max_value=4)
dgw_strategy = st.integers(min_value=1, max_value=4)
num_windows_strategy = st.integers(min_value=1, max_value=4)


@st.composite
def decoder_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para VSNDecoder.

    Mantiene dimensiones pequeñas para que las ventanas múltiples
    completen en tiempo razonable (X_dec planos × num_windows ventanas).
    """
    B = draw(batch_strategy)
    Y = draw(spatial_strategy)
    Z = draw(spatial_strategy)
    d = draw(d_strategy)
    X_dec = draw(x_dec_strategy)
    dgw = draw(dgw_strategy)
    num_windows = draw(num_windows_strategy)

    return {
        "batch": B,
        "Y": Y,
        "Z": Z,
        "d": d,
        "X_dec": X_dec,
        "dgw": dgw,
        "num_windows": num_windows,
    }


# ---------------------------------------------------------------------------
# Propiedad 21: Decoder genera exactamente num_windows outputs
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=decoder_dims())
def test_decoder_produces_exact_num_windows(dims: dict) -> None:
    """Feature: vsn-library, Property 21: Decoder genera ventanas DGW correctamente

    **Validates: Requirements 5.3, 5.4**

    Para cualquier configuración con window size W (DGW), el decoder SHALL
    generar exactamente num_windows outputs. Cada output tiene shape (B,Y,Z,d).
    """
    B = dims["batch"]
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]
    X_dec = dims["X_dec"]
    dgw = dims["dgw"]
    num_windows = dims["num_windows"]

    decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)
    decoder.eval()

    V_dec_0 = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        outputs = decoder(V_dec_0, num_windows=num_windows)

    # Verificar que se producen exactamente num_windows salidas
    assert len(outputs) == num_windows, (
        f"Decoder produjo {len(outputs)} outputs, esperado {num_windows}"
    )

    # Verificar shape de cada output
    expected_shape = (B, Y, Z, d)
    for i, output in enumerate(outputs):
        assert output.shape == expected_shape, (
            f"Output de ventana {i} tiene shape {output.shape}, "
            f"esperado {expected_shape}"
        )
        assert torch.isfinite(output).all(), (
            f"Output de ventana {i} contiene NaN o Inf"
        )


@settings(max_examples=100, deadline=None)
@given(dims=decoder_dims())
def test_decoder_psi_invoked_between_windows(dims: dict) -> None:
    """Feature: vsn-library, Property 21: Decoder genera ventanas DGW correctamente

    **Validates: Requirements 5.3, 5.4**

    Cuando Ψ es invocado entre ventanas, SHALL producir estados iniciales
    válidos (tensores finitos con shapes correctos). Verificamos que para
    num_windows > 1, Ψ produce outputs que mantienen la cadena funcional.
    """
    B = dims["batch"]
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]
    X_dec = dims["X_dec"]
    dgw = dims["dgw"]
    num_windows = dims["num_windows"]

    # Si solo hay 1 ventana, Ψ no se invoca — forzar al menos 2
    if num_windows < 2:
        num_windows = 2

    decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)
    decoder.eval()

    V_dec_0 = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        outputs = decoder(V_dec_0, num_windows=num_windows)

    # Si Ψ funciona correctamente, todas las ventanas producen outputs válidos
    assert len(outputs) == num_windows, (
        f"Con Ψ entre ventanas, decoder produjo {len(outputs)} outputs, "
        f"esperado {num_windows}"
    )

    # Verificar que cada output post-Ψ es finito y tiene shape correcta
    for i in range(1, num_windows):
        assert outputs[i].shape == (B, Y, Z, d), (
            f"Output post-Ψ ventana {i} tiene shape {outputs[i].shape}, "
            f"esperado {(B, Y, Z, d)}"
        )
        assert torch.isfinite(outputs[i]).all(), (
            f"Output post-Ψ ventana {i} contiene NaN o Inf. "
            f"Ψ no produce estados iniciales válidos."
        )

    # Verificar que las ventanas sucesivas difieren (Ψ transforma estado)
    # Nota: con params aleatorios, es virtualmente imposible que dos ventanas
    # produzcan exactamente el mismo output
    if num_windows >= 2:
        window_0 = outputs[0]
        window_1 = outputs[1]
        # Los outputs deben ser diferentes (Ψ transforma el estado)
        assert not torch.equal(window_0, window_1), (
            "Ventana 0 y 1 producen outputs idénticos. "
            "Ψ debería transformar el estado entre ventanas."
        )


@settings(max_examples=100, deadline=None)
@given(dims=decoder_dims())
def test_decoder_psi_produces_valid_initial_states(dims: dict) -> None:
    """Feature: vsn-library, Property 21: Decoder genera ventanas DGW correctamente

    **Validates: Requirements 5.3, 5.4**

    Verificación directa de que Ψ produce estados iniciales válidos:
    Dado un estado final de ventana simulado, Ψ produce (V_dec_0_next, M_next)
    con shapes (B,Y,Z,d) y valores finitos.
    """
    B = dims["batch"]
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]

    psi = PsiOperator(Y=Y, Z=Z, d=d)
    psi.eval()

    # Simular estado final de una ventana
    decoder_volume_final = torch.randn(B, Y, Z, d)
    memory_final = torch.randn(B, Y, Z, d)
    recent_output_summary = torch.randn(B, d)

    with torch.no_grad():
        V_dec_0_next, M_next = psi(
            decoder_volume_final, memory_final, recent_output_summary
        )

    # Ψ debe producir estados válidos para la siguiente ventana
    expected_shape = (B, Y, Z, d)
    assert V_dec_0_next.shape == expected_shape, (
        f"Ψ V_dec_0_next shape {V_dec_0_next.shape} != {expected_shape}"
    )
    assert M_next.shape == expected_shape, (
        f"Ψ M_next shape {M_next.shape} != {expected_shape}"
    )
    assert torch.isfinite(V_dec_0_next).all(), (
        "Ψ produce V_dec_0_next con NaN/Inf — estado inválido para siguiente ventana"
    )
    assert torch.isfinite(M_next).all(), (
        "Ψ produce M_next con NaN/Inf — estado inválido para siguiente ventana"
    )

    # Los estados producidos deben ser no triviales (no todos zeros)
    # Con parámetros aleatorios, la probabilidad de output = 0 es ~0
    assert V_dec_0_next.abs().sum() > 0, (
        "Ψ produce V_dec_0_next = 0. El estado debe ser no trivial."
    )
    assert M_next.abs().sum() > 0, (
        "Ψ produce M_next = 0. El estado debe ser no trivial."
    )
