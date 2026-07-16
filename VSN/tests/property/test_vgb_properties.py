"""Feature: vsn-library, Properties 2, 3, 4: VGB v1 Property Tests

**Validates: Requirements 2.1-2.7, 5.2**

Tests de propiedad que verifican:
- Propiedad 2: El pipeline VGB v1 de 6 pasos preserva shapes para cualquier
  tensor de entrada (B, Y, Z, d) y memoria M (B, Y, Z, d).
- Propiedad 3: La actualización gated de memoria satisface la semántica formal:
  gate≈1 → M preservada, gate≈0 → M reemplazada por proyección.
- Propiedad 4: Cada bloque VGB v1 en posición x tiene parámetros con data_ptr()
  distinto de cualquier otro bloque en posición x'≠x.
"""

from itertools import combinations

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from vsn.core.vgb import VGBv1

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones válidas
# ---------------------------------------------------------------------------

batch_strategy = st.integers(min_value=1, max_value=4)
spatial_strategy = st.integers(min_value=1, max_value=8)
d_strategy = st.integers(min_value=4, max_value=32)


@st.composite
def vgb_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para un bloque VGB v1."""
    return {
        "batch": draw(batch_strategy),
        "Y": draw(spatial_strategy),
        "Z": draw(spatial_strategy),
        "d": draw(d_strategy),
    }


# ---------------------------------------------------------------------------
# Propiedad 2: Pipeline VGB v1 de 6 pasos preserva shapes
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=vgb_dims())
def test_vgb_v1_pipeline_preserves_shapes(dims: dict) -> None:
    """Feature: vsn-library, Property 2: Pipeline VGB v1 de 6 pasos preserva shapes

    **Validates: Requirements 2.1, 2.2, 2.4, 2.5, 2.6**

    Para cualquier tensor de entrada x de shape (B, Y, Z, d) y memoria M de
    shape (B, Y, Z, d), el bloque VGB v1 SHALL producir F, G, r de shape
    (B, Y, Z, d) y M_new de shape (B, Y, Z, d).
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]

    block = VGBv1(d=d, plane_idx=0)
    block.eval()

    x = torch.randn(B, Y, Z, d)
    M = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        F, G, r, M_new = block(x, M)

    expected_shape = (B, Y, Z, d)

    assert F.shape == expected_shape, (
        f"F shape {F.shape} != expected {expected_shape}"
    )
    assert G.shape == expected_shape, (
        f"G shape {G.shape} != expected {expected_shape}"
    )
    assert r.shape == expected_shape, (
        f"r shape {r.shape} != expected {expected_shape}"
    )
    assert M_new.shape == expected_shape, (
        f"M_new shape {M_new.shape} != expected {expected_shape}"
    )


@settings(max_examples=100, deadline=None)
@given(dims=vgb_dims())
def test_vgb_v1_outputs_are_finite(dims: dict) -> None:
    """Feature: vsn-library, Property 2: VGB v1 outputs are finite (no NaN/Inf).

    **Validates: Requirements 2.1, 2.2, 2.4, 2.5, 2.6**

    Para cualquier input válido, todas las salidas del VGB v1 deben ser finitas.
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]

    block = VGBv1(d=d, plane_idx=0)
    block.eval()

    x = torch.randn(B, Y, Z, d)
    M = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        F, G, r, M_new = block(x, M)

    assert torch.isfinite(F).all(), "F contains NaN or Inf"
    assert torch.isfinite(G).all(), "G contains NaN or Inf"
    assert torch.isfinite(r).all(), "r contains NaN or Inf"
    assert torch.isfinite(M_new).all(), "M_new contains NaN or Inf"


# ---------------------------------------------------------------------------
# Propiedad 3: Actualización gated de memoria VGB v1
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=vgb_dims())
def test_vgb_v1_memory_preserved_when_gate_near_one(dims: dict) -> None:
    """Feature: vsn-library, Property 3: Actualización gated de memoria VGB v1

    **Validates: Requirements 2.3**

    Cuando gate ≈ 1, M_new ≈ M (memoria preservada).
    Forzamos W_g bias alto para que sigmoid(W_g·x_norm + b_g) ≈ 1.
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]

    block = VGBv1(d=d, plane_idx=0)
    block.eval()

    # Forzar gate ≈ 1: establecer W_g.weight = 0 y W_g.bias = valor alto positivo
    with torch.no_grad():
        block.W_g.weight.zero_()
        block.W_g.bias.fill_(20.0)  # sigmoid(20) ≈ 1.0

    x = torch.randn(B, Y, Z, d)
    M = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        _, _, _, M_new = block(x, M)

    # Con gate ≈ 1: M_new = g*M + (1-g)*m ≈ 1*M + 0*m = M
    assert torch.allclose(M_new, M, atol=1e-4), (
        f"Con gate≈1, M_new debería ≈ M. "
        f"Max diff: {(M_new - M).abs().max().item():.6f}"
    )


@settings(max_examples=100, deadline=None)
@given(dims=vgb_dims())
def test_vgb_v1_memory_replaced_when_gate_near_zero(dims: dict) -> None:
    """Feature: vsn-library, Property 3: Actualización gated de memoria VGB v1

    **Validates: Requirements 2.3**

    Cuando gate ≈ 0, M_new ≈ W_m·norm(x) (memoria reemplazada por nueva proyección).
    Forzamos W_g bias muy negativo para que sigmoid(W_g·x_norm + b_g) ≈ 0.
    """
    B, Y, Z, d = dims["batch"], dims["Y"], dims["Z"], dims["d"]

    block = VGBv1(d=d, plane_idx=0)
    block.eval()

    # Forzar gate ≈ 0: establecer W_g.weight = 0 y W_g.bias = valor muy negativo
    with torch.no_grad():
        block.W_g.weight.zero_()
        block.W_g.bias.fill_(-20.0)  # sigmoid(-20) ≈ 0.0

    x = torch.randn(B, Y, Z, d)
    M = torch.randn(B, Y, Z, d)

    with torch.no_grad():
        _, _, _, M_new = block(x, M)

    # Con gate ≈ 0: M_new = g*M + (1-g)*m ≈ 0*M + 1*m = W_m(norm(x))
    # Calcular manualmente W_m(norm(x)) para verificar
    x_norm = block.norm(x)
    m_expected = block.W_m(x_norm)

    assert torch.allclose(M_new, m_expected, atol=1e-4), (
        f"Con gate≈0, M_new debería ≈ W_m·norm(x). "
        f"Max diff: {(M_new - m_expected).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Propiedad 4: Independencia de parámetros entre planos
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    d=d_strategy,
    num_planes=st.integers(min_value=2, max_value=6),
)
def test_vgb_v1_independent_parameters_across_planes(d: int, num_planes: int) -> None:
    """Feature: vsn-library, Property 4: Independencia de parámetros entre planos

    **Validates: Requirements 2.7, 5.2**

    Para cualquier X planos, cada VGB v1 block en posición x SHALL tener
    data_ptr() distinto de cualquier otro bloque en posición x'≠x.
    """
    # Crear múltiples bloques VGB como haría un Encoder/Decoder
    blocks = nn.ModuleList([VGBv1(d=d, plane_idx=x) for x in range(num_planes)])

    # Recolectar todos los data_ptr por bloque
    ptrs_per_block: list[set[int]] = []
    for block in blocks:
        block_ptrs = set()
        for param in block.parameters():
            block_ptrs.add(param.data_ptr())
        ptrs_per_block.append(block_ptrs)

    # Verificar que ningún par de bloques comparte data_ptr
    for (i, ptrs_i), (j, ptrs_j) in combinations(enumerate(ptrs_per_block), 2):
        shared = ptrs_i & ptrs_j
        assert len(shared) == 0, (
            f"Bloques en posición {i} y {j} comparten {len(shared)} "
            f"parámetros (data_ptr compartidos). Los bloques VGB v1 deben "
            f"tener parámetros completamente independientes."
        )
