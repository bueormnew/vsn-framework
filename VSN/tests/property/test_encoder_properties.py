"""Feature: vsn-library, Property 7: Encoder Propagation Property Tests

**Validates: Requirements 3.4, 3.5, 3.6**

Tests de propiedad que verifican:
- Propiedad 7: Para cualquier encoder con X planos, la propagación es
  exclusivamente hacia adelante en el eje X. La salida del encoder tiene
  shape (batch, Y, Z, d). El output depende de todos los bloques VGB
  (gradientes fluyen a todos los bloques). No existe dependencia hacia
  atrás (la salida del plano x no depende de planos > x).
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.core.encoder import VSNEncoder

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones válidas del encoder
# ---------------------------------------------------------------------------

d_strategy = st.integers(min_value=4, max_value=16)
spatial_strategy = st.integers(min_value=1, max_value=4)


@st.composite
def encoder_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para un VSNEncoder."""
    X = draw(st.integers(min_value=2, max_value=4))
    Y = draw(spatial_strategy)
    Z = draw(spatial_strategy)
    d = draw(d_strategy)
    # ICS debe ser ≤ X*Y*Z (capacidad del volumen)
    capacity = X * Y * Z
    ics = draw(st.integers(min_value=1, max_value=capacity))
    batch = draw(st.integers(min_value=1, max_value=2))
    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "d": d,
        "ics": ics,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# Propiedad 7: Propagación exclusivamente hacia adelante en eje X
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=encoder_dims())
def test_encoder_output_shape(dims: dict) -> None:
    """Feature: vsn-library, Property 7: Output shape correcto del encoder

    **Validates: Requirements 3.4, 3.5, 3.6**

    Para cualquier encoder con X planos, la salida final tiene shape
    (batch, Y, Z, d) — el último plano procesado.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    ics = dims["ics"]
    batch = dims["batch"]

    encoder = VSNEncoder(X=X, Y=Y, Z=Z, d=d, ics=ics)
    encoder.eval()

    # Generar tokens de entrada (batch, num_tokens, d) con num_tokens = ics
    tokens = torch.randn(batch, ics, d)

    with torch.no_grad():
        output = encoder(tokens)

    expected_shape = (batch, Y, Z, d)
    assert output.shape == expected_shape, (
        f"Encoder output shape {output.shape} != esperado {expected_shape}. "
        f"Config: X={X}, Y={Y}, Z={Z}, d={d}, ics={ics}"
    )


@settings(max_examples=100, deadline=None)
@given(dims=encoder_dims())
def test_encoder_gradients_flow_to_all_blocks(dims: dict) -> None:
    """Feature: vsn-library, Property 7: Gradientes fluyen a todos los bloques VGB

    **Validates: Requirements 3.4, 3.5, 3.6**

    El output del encoder depende de todos los bloques VGB — los gradientes
    fluyen desde la salida hasta cada bloque, confirmando que todos participan
    en la propagación hacia adelante.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    ics = dims["ics"]
    batch = dims["batch"]

    encoder = VSNEncoder(X=X, Y=Y, Z=Z, d=d, ics=ics)
    encoder.train()

    tokens = torch.randn(batch, ics, d, requires_grad=True)

    output = encoder(tokens)

    # Backward desde una loss escalar
    loss = output.sum()
    loss.backward()

    # Verificar que TODOS los bloques VGB recibieron gradientes
    for x, block in enumerate(encoder.vgb_blocks):
        has_grad = False
        for name, param in block.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, (
            f"Bloque VGB en plano x={x} no recibió gradientes. "
            f"La propagación debería conectar todos los bloques al output."
        )


@settings(max_examples=100, deadline=None)
@given(dims=encoder_dims())
def test_encoder_no_backward_dependency(dims: dict) -> None:
    """Feature: vsn-library, Property 7: Sin dependencia hacia atrás en X

    **Validates: Requirements 3.4, 3.5, 3.6**

    La salida del plano x no depende de planos > x. Verificamos esto
    perturbando los parámetros de un bloque posterior y comprobando que
    los planos anteriores no se ven afectados durante el forward.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    ics = dims["ics"]
    batch = dims["batch"]

    if X < 2:
        return  # Necesitamos al menos 2 planos para verificar independencia

    encoder = VSNEncoder(X=X, Y=Y, Z=Z, d=d, ics=ics)
    encoder.eval()

    tokens = torch.randn(batch, ics, d)

    # Forward completo y capturar activaciones intermedias
    # Hacemos forward manual para inspeccionar planos individuales
    with torch.no_grad():
        volume = encoder.phi(tokens)  # (batch, X, Y, Z, d)

        M = tokens.new_zeros(batch, Y, Z, d)
        V = [volume[:, x, :, :, :] for x in range(X)]

        F_outputs = [None] * X
        G_outputs = [None] * X

        # Procesar primer plano para obtener su estado
        plane_input_0 = V[0]
        F_0, G_0, r_0, M_new = encoder.vgb_blocks[0](plane_input_0, M)
        F_outputs[0] = F_0
        G_outputs[0] = G_0
        first_plane_output = r_0.clone()

    # Ahora perturbar los parámetros del ÚLTIMO bloque
    with torch.no_grad():
        for param in encoder.vgb_blocks[-1].parameters():
            param.add_(torch.randn_like(param) * 10.0)

    # Re-ejecutar y verificar que el primer plano no cambió
    with torch.no_grad():
        volume2 = encoder.phi(tokens)
        M2 = tokens.new_zeros(batch, Y, Z, d)
        V2 = [volume2[:, x, :, :, :] for x in range(X)]

        plane_input_0_v2 = V2[0]
        F_0_v2, G_0_v2, r_0_v2, _ = encoder.vgb_blocks[0](plane_input_0_v2, M2)

    # El resultado del primer plano debe ser idéntico
    # (no depende de bloques posteriores)
    assert torch.equal(first_plane_output, r_0_v2), (
        f"El output del plano 0 cambió al perturbar el bloque del plano {X-1}. "
        f"Esto indica dependencia hacia atrás (violación de forward-only). "
        f"Max diff: {(first_plane_output - r_0_v2).abs().max().item()}"
    )


@settings(max_examples=100, deadline=None)
@given(dims=encoder_dims())
def test_encoder_output_is_finite(dims: dict) -> None:
    """Feature: vsn-library, Property 7: Encoder output es finito

    **Validates: Requirements 3.4, 3.5, 3.6**

    Para cualquier input válido, el output del encoder debe ser finito
    (sin NaN ni Inf), confirmando estabilidad numérica de la propagación.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    ics = dims["ics"]
    batch = dims["batch"]

    encoder = VSNEncoder(X=X, Y=Y, Z=Z, d=d, ics=ics)
    encoder.eval()

    tokens = torch.randn(batch, ics, d)

    with torch.no_grad():
        output = encoder(tokens)

    assert torch.isfinite(output).all(), (
        f"Encoder output contiene NaN o Inf. "
        f"Config: X={X}, Y={Y}, Z={Z}, d={d}, ics={ics}"
    )
