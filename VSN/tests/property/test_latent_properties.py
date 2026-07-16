"""Feature: vsn-library, Property 8: Transformación de forma P y Q

**Validates: Requirements 4.1, 4.2, 4.3**

Tests de propiedad que verifican:
1. Para cualquier V_{X-1} de shape (B, Y, Z, d), ProjectorP SHALL producir H
   de shape (B, Y_H, Z_H, d_H).
2. Para cualquier H de shape (B, Y_H, Z_H, d_H), TransitionQ SHALL producir
   V^dec_0 de shape (B, Y_dec, Z_dec, d).
3. P y Q SHALL ser independientes (parámetros distintos, sin data_ptr compartidos).
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.core.latent import ProjectorP
from vsn.core.transitions import TransitionQ

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones válidas
# ---------------------------------------------------------------------------

batch_strategy = st.integers(min_value=1, max_value=4)
spatial_strategy = st.integers(min_value=1, max_value=8)
d_strategy = st.integers(min_value=2, max_value=32)


@st.composite
def compress_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para ProjectorP en modo 'compress'.

    Garantiza que Y_H*Z_H*d_H < Y*Z*d (requisito del modo compress).
    """
    B = draw(batch_strategy)
    Y = draw(st.integers(min_value=2, max_value=8))
    Z = draw(st.integers(min_value=2, max_value=8))
    d = draw(st.integers(min_value=4, max_value=32))

    in_size = Y * Z * d

    # Generar dimensiones de salida cuyo producto sea estrictamente menor
    Y_H = draw(st.integers(min_value=1, max_value=max(1, Y - 1)))
    Z_H = draw(st.integers(min_value=1, max_value=max(1, Z - 1)))
    # d_H debe hacer que Y_H*Z_H*d_H < in_size
    max_d_H = (in_size - 1) // (Y_H * Z_H)
    d_H = draw(st.integers(min_value=1, max_value=max(1, min(max_d_H, 32))))

    # Filtrar si la restricción no se cumple
    out_size = Y_H * Z_H * d_H
    if out_size >= in_size:
        # Forzar un d_H que cumpla
        d_H = max(1, (in_size - 1) // (Y_H * Z_H))
        out_size = Y_H * Z_H * d_H
        if out_size >= in_size:
            # Reducir aún más
            Y_H, Z_H, d_H = 1, 1, max(1, in_size - 1)
            if d_H >= in_size:
                d_H = 1

    # Verificación final
    assert Y_H * Z_H * d_H < in_size, (
        f"compress: {Y_H}*{Z_H}*{d_H}={Y_H*Z_H*d_H} >= {in_size}"
    )

    return {
        "batch": B, "Y": Y, "Z": Z, "d": d,
        "Y_H": Y_H, "Z_H": Z_H, "d_H": d_H,
        "mode": "compress",
    }


@st.composite
def identity_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para ProjectorP en modo 'identity'.

    Garantiza que Y_H*Z_H*d_H == Y*Z*d.
    """
    B = draw(batch_strategy)
    Y = draw(st.integers(min_value=1, max_value=6))
    Z = draw(st.integers(min_value=1, max_value=6))
    d = draw(st.integers(min_value=2, max_value=16))

    in_size = Y * Z * d

    # Para identity, elegir Y_H, Z_H, d_H tal que su producto == in_size
    # Estrategia: usar factorizaciones simples
    Y_H = draw(st.integers(min_value=1, max_value=min(in_size, 8)))
    # Necesitamos Z_H*d_H == in_size // Y_H, con in_size divisible por Y_H
    if in_size % Y_H != 0:
        Y_H = 1  # fallback seguro
    remaining = in_size // Y_H

    Z_H = draw(st.integers(min_value=1, max_value=min(remaining, 8)))
    if remaining % Z_H != 0:
        Z_H = 1
    d_H = remaining // Z_H

    assert Y_H * Z_H * d_H == in_size

    return {
        "batch": B, "Y": Y, "Z": Z, "d": d,
        "Y_H": Y_H, "Z_H": Z_H, "d_H": d_H,
        "mode": "identity",
    }


@st.composite
def expand_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para ProjectorP en modo 'expand'.

    Garantiza que Y_H*Z_H*d_H > Y*Z*d (requisito del modo expand).
    """
    B = draw(batch_strategy)
    Y = draw(st.integers(min_value=1, max_value=4))
    Z = draw(st.integers(min_value=1, max_value=4))
    d = draw(st.integers(min_value=2, max_value=8))

    in_size = Y * Z * d

    # Generar dimensiones de salida cuyo producto sea estrictamente mayor
    Y_H = draw(st.integers(min_value=Y, max_value=Y + 4))
    Z_H = draw(st.integers(min_value=Z, max_value=Z + 4))
    # d_H debe hacer que Y_H*Z_H*d_H > in_size
    min_d_H = (in_size // (Y_H * Z_H)) + 1
    d_H = draw(st.integers(min_value=max(1, min_d_H), max_value=max(min_d_H, 32)))

    out_size = Y_H * Z_H * d_H
    assert out_size > in_size, (
        f"expand: {Y_H}*{Z_H}*{d_H}={out_size} <= {in_size}"
    )

    return {
        "batch": B, "Y": Y, "Z": Z, "d": d,
        "Y_H": Y_H, "Z_H": Z_H, "d_H": d_H,
        "mode": "expand",
    }


@st.composite
def q_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para TransitionQ.

    TransitionQ no tiene restricciones de modo, cualquier combinación positiva
    de dimensiones es válida.
    """
    B = draw(batch_strategy)
    Y_H = draw(spatial_strategy)
    Z_H = draw(spatial_strategy)
    d_H = draw(d_strategy)
    Y_dec = draw(spatial_strategy)
    Z_dec = draw(spatial_strategy)
    d = draw(d_strategy)

    return {
        "batch": B,
        "Y_H": Y_H, "Z_H": Z_H, "d_H": d_H,
        "Y_dec": Y_dec, "Z_dec": Z_dec, "d": d,
    }


# ---------------------------------------------------------------------------
# Propiedad 8.1: P produce H con shape correcta (modo compress)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=compress_dims())
def test_projector_p_compress_shape(dims: dict) -> None:
    """Feature: vsn-library, Property 8: Transformación de forma P (compress)

    **Validates: Requirements 4.1, 4.2**

    Para cualquier V_{X-1} de shape (B, Y, Z, d), ProjectorP en modo compress
    SHALL producir H de shape (B, Y_H, Z_H, d_H) con Y_H*Z_H*d_H < Y*Z*d.
    """
    B = dims["batch"]
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]
    Y_H, Z_H, d_H = dims["Y_H"], dims["Z_H"], dims["d_H"]

    P = ProjectorP(Y=Y, Z=Z, d=d, Y_H=Y_H, Z_H=Z_H, d_H=d_H, mode="compress")
    P.eval()

    V_last = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        H = P(V_last)

    expected_shape = (B, Y_H, Z_H, d_H)
    assert H.shape == expected_shape, (
        f"P(compress) output shape {H.shape} != expected {expected_shape}"
    )
    assert torch.isfinite(H).all(), "P(compress) output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Propiedad 8.1: P produce H con shape correcta (modo identity)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=identity_dims())
def test_projector_p_identity_shape(dims: dict) -> None:
    """Feature: vsn-library, Property 8: Transformación de forma P (identity)

    **Validates: Requirements 4.1, 4.2**

    Para cualquier V_{X-1} de shape (B, Y, Z, d), ProjectorP en modo identity
    SHALL producir H de shape (B, Y_H, Z_H, d_H) con Y_H*Z_H*d_H == Y*Z*d.
    """
    B = dims["batch"]
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]
    Y_H, Z_H, d_H = dims["Y_H"], dims["Z_H"], dims["d_H"]

    P = ProjectorP(Y=Y, Z=Z, d=d, Y_H=Y_H, Z_H=Z_H, d_H=d_H, mode="identity")
    P.eval()

    V_last = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        H = P(V_last)

    expected_shape = (B, Y_H, Z_H, d_H)
    assert H.shape == expected_shape, (
        f"P(identity) output shape {H.shape} != expected {expected_shape}"
    )
    assert torch.isfinite(H).all(), "P(identity) output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Propiedad 8.1: P produce H con shape correcta (modo expand)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=expand_dims())
def test_projector_p_expand_shape(dims: dict) -> None:
    """Feature: vsn-library, Property 8: Transformación de forma P (expand)

    **Validates: Requirements 4.1, 4.2**

    Para cualquier V_{X-1} de shape (B, Y, Z, d), ProjectorP en modo expand
    SHALL producir H de shape (B, Y_H, Z_H, d_H) con Y_H*Z_H*d_H > Y*Z*d.
    """
    B = dims["batch"]
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]
    Y_H, Z_H, d_H = dims["Y_H"], dims["Z_H"], dims["d_H"]

    P = ProjectorP(Y=Y, Z=Z, d=d, Y_H=Y_H, Z_H=Z_H, d_H=d_H, mode="expand")
    P.eval()

    V_last = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        H = P(V_last)

    expected_shape = (B, Y_H, Z_H, d_H)
    assert H.shape == expected_shape, (
        f"P(expand) output shape {H.shape} != expected {expected_shape}"
    )
    assert torch.isfinite(H).all(), "P(expand) output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Propiedad 8.2: Q produce V^dec_0 con shape correcta
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=q_dims())
def test_transition_q_shape(dims: dict) -> None:
    """Feature: vsn-library, Property 8: Transformación de forma Q

    **Validates: Requirements 4.3**

    Para cualquier H de shape (B, Y_H, Z_H, d_H), TransitionQ SHALL producir
    V^dec_0 de shape (B, Y_dec, Z_dec, d).
    """
    B = dims["batch"]
    Y_H, Z_H, d_H = dims["Y_H"], dims["Z_H"], dims["d_H"]
    Y_dec, Z_dec, d = dims["Y_dec"], dims["Z_dec"], dims["d"]

    Q = TransitionQ(Y_H=Y_H, Z_H=Z_H, d_H=d_H, Y_dec=Y_dec, Z_dec=Z_dec, d=d)
    Q.eval()

    H = torch.randn(B, Y_H, Z_H, d_H)

    with torch.no_grad():
        V_dec_0 = Q(H)

    expected_shape = (B, Y_dec, Z_dec, d)
    assert V_dec_0.shape == expected_shape, (
        f"Q output shape {V_dec_0.shape} != expected {expected_shape}"
    )
    assert torch.isfinite(V_dec_0).all(), "Q output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Propiedad 8.3: P y Q son independientes (parámetros distintos)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=compress_dims())
def test_p_and_q_independent_parameters(dims: dict) -> None:
    """Feature: vsn-library, Property 8: P y Q son independientes

    **Validates: Requirements 4.1, 4.2, 4.3**

    Las transformaciones P y Q SHALL ser independientes: parámetros distintos,
    sin data_ptr compartidos. Esto garantiza la separación arquitectónica.
    """
    Y, Z, d = dims["Y"], dims["Z"], dims["d"]
    Y_H, Z_H, d_H = dims["Y_H"], dims["Z_H"], dims["d_H"]

    # Usar Y_H, Z_H, d_H como entrada de Q y dimensiones de decoder arbitrarias
    Y_dec, Z_dec = Y, Z  # decoder puede tener mismas dims que encoder

    P = ProjectorP(Y=Y, Z=Z, d=d, Y_H=Y_H, Z_H=Z_H, d_H=d_H, mode="compress")
    Q = TransitionQ(Y_H=Y_H, Z_H=Z_H, d_H=d_H, Y_dec=Y_dec, Z_dec=Z_dec, d=d)

    # Recolectar todos los data_ptr de P
    p_ptrs = {param.data_ptr() for param in P.parameters()}

    # Recolectar todos los data_ptr de Q
    q_ptrs = {param.data_ptr() for param in Q.parameters()}

    # Verificar que no comparten ningún parámetro
    shared = p_ptrs & q_ptrs
    assert len(shared) == 0, (
        f"P y Q comparten {len(shared)} parámetros (data_ptr compartidos). "
        f"P y Q deben ser completamente independientes."
    )

    # Verificar que ambos tienen parámetros (son entrenables)
    assert len(p_ptrs) > 0, "P no tiene parámetros entrenables"
    assert len(q_ptrs) > 0, "Q no tiene parámetros entrenables"
