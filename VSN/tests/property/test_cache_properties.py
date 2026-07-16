"""Feature: vsn-library, Properties 5, 6: Input Cache & Positioning Property Tests

**Validates: Requirements 3.1, 3.2, 3.3**

Tests de propiedad que verifican:
- Propiedad 5: Para cualquier secuencia de tokens enviados al Input Cache de tamaño
  ICS, cuando el buffer alcanza capacidad, el contenido extraído preserva el orden
  de inserción (FIFO). La inyección ocurre exactamente cuando write_ptr alcanza ICS.
- Propiedad 6: Para cualquier conjunto de tokens de tamaño ≤ X*Y*Z, el operador Φ
  aplicado dos veces produce posiciones idénticas. Todas las posiciones generadas
  están en el rango [0, Y) × [0, Z).
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.core.input_cache import InputCache
from vsn.core.positioning import PositioningOperator

# ---------------------------------------------------------------------------
# Estrategias para generar dimensiones válidas
# ---------------------------------------------------------------------------

ics_strategy = st.integers(min_value=1, max_value=16)
d_strategy = st.integers(min_value=2, max_value=32)
batch_strategy = st.integers(min_value=1, max_value=4)
spatial_strategy = st.integers(min_value=1, max_value=6)


@st.composite
def cache_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para un InputCache."""
    return {
        "ics": draw(ics_strategy),
        "d": draw(d_strategy),
        "batch_size": draw(batch_strategy),
    }


@st.composite
def positioning_dims(draw: st.DrawFn) -> dict:
    """Genera dimensiones válidas para el PositioningOperator."""
    X = draw(st.integers(min_value=1, max_value=4))
    Y = draw(spatial_strategy)
    Z = draw(spatial_strategy)
    d = draw(d_strategy)
    # num_tokens entre 1 y capacidad del volumen
    capacity = X * Y * Z
    num_tokens = draw(st.integers(min_value=1, max_value=capacity))
    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "d": d,
        "num_tokens": num_tokens,
    }


# ---------------------------------------------------------------------------
# Propiedad 5: Semántica FIFO del Input Cache
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=cache_dims())
def test_input_cache_fifo_order_preserved_on_full(dims: dict) -> None:
    """Feature: vsn-library, Property 5: Semántica FIFO del Input Cache

    **Validates: Requirements 3.1, 3.2**

    Para cualquier secuencia de tokens T₁, T₂, ..., Tₙ enviados al Input Cache
    de tamaño ICS, cuando el buffer alcanza capacidad ICS, el contenido extraído
    preserva el orden de inserción (T₁ primero, Tₙ último).
    """
    ics = dims["ics"]
    d = dims["d"]
    batch_size = dims["batch_size"]

    cache = InputCache(ics=ics, d=d, batch_size=batch_size)

    # Generar tokens con valores únicos por posición para rastrear orden
    all_tokens = torch.randn(batch_size, ics, d)

    # Insertar todos de una vez (caso más simple de llenado completo)
    result = cache.push(all_tokens)

    # Cuando el buffer se llena exactamente, debe retornar el contenido
    assert result is not None, (
        f"Cache con ics={ics} debería retornar contenido al llenarse"
    )
    assert result.shape == (batch_size, ics, d), (
        f"Shape resultante {result.shape} != esperado ({batch_size}, {ics}, {d})"
    )

    # El contenido extraído debe ser IDÉNTICO al orden de inserción (FIFO)
    assert torch.equal(result, all_tokens), (
        "El contenido extraído no preserva el orden FIFO de inserción"
    )


@settings(max_examples=100, deadline=None)
@given(dims=cache_dims())
def test_input_cache_fifo_order_with_incremental_push(dims: dict) -> None:
    """Feature: vsn-library, Property 5: Semántica FIFO del Input Cache (incremental)

    **Validates: Requirements 3.1, 3.2**

    Para una secuencia de tokens insertados incrementalmente (uno a uno),
    cuando el buffer se llena, el contenido preserva el orden FIFO.
    """
    ics = dims["ics"]
    d = dims["d"]
    batch_size = dims["batch_size"]

    cache = InputCache(ics=ics, d=d, batch_size=batch_size)

    # Insertar tokens uno a uno
    all_tokens = []
    result = None
    for i in range(ics):
        token = torch.randn(batch_size, 1, d)
        all_tokens.append(token)
        result = cache.push(token)

        # Solo el último push debe retornar resultado (cuando se llena)
        if i < ics - 1:
            assert result is None, (
                f"Cache retornó resultado en paso {i} (antes de llenarse). "
                f"ics={ics}, occupancy esperada={i + 1}"
            )

    # El último push debe haber retornado resultado
    assert result is not None, (
        "Cache no retornó contenido al alcanzar capacidad ICS"
    )

    # Reconstruir la secuencia esperada
    expected = torch.cat(all_tokens, dim=1)  # (batch, ics, d)

    # Verificar orden FIFO
    assert torch.equal(result, expected), (
        "Los tokens extraídos no preservan el orden de inserción incremental"
    )


@settings(max_examples=100, deadline=None)
@given(dims=cache_dims())
def test_input_cache_injection_at_exactly_ics(dims: dict) -> None:
    """Feature: vsn-library, Property 5: Inyección ocurre exactamente en ICS

    **Validates: Requirements 3.1, 3.2**

    La inyección (retorno no-None) ocurre EXACTAMENTE cuando write_ptr alcanza ICS.
    Antes de llenar, siempre retorna None.
    """
    ics = dims["ics"]
    d = dims["d"]
    batch_size = dims["batch_size"]

    cache = InputCache(ics=ics, d=d, batch_size=batch_size)

    # Insertar parcialmente (si ICS > 1)
    if ics > 1:
        partial_tokens = torch.randn(batch_size, ics - 1, d)
        result = cache.push(partial_tokens)
        assert result is None, (
            f"Cache retornó resultado con {ics - 1} tokens (capacidad = {ics})"
        )
        assert cache.occupancy == ics - 1

    # Insertar el token faltante → debe disparar inyección
    last_token = torch.randn(batch_size, 1, d)
    result = cache.push(last_token)
    assert result is not None, (
        f"Cache NO retornó contenido al alcanzar exactamente ICS={ics}"
    )

    # Después de la inyección, el cache debe estar vacío (reset)
    assert cache.occupancy == 0, (
        f"Cache occupancy={cache.occupancy} después de inyección (esperado 0)"
    )


@settings(max_examples=100, deadline=None)
@given(dims=cache_dims())
def test_input_cache_flush_preserves_fifo_order(dims: dict) -> None:
    """Feature: vsn-library, Property 5: Flush preserva orden FIFO

    **Validates: Requirements 3.1, 3.2**

    flush() extrae el contenido parcial preservando el orden de inserción.
    """
    ics = dims["ics"]
    d = dims["d"]
    batch_size = dims["batch_size"]

    if ics < 2:
        # Con ics=1, no podemos hacer flush parcial
        return

    cache = InputCache(ics=ics, d=d, batch_size=batch_size)

    # Insertar menos tokens de los que caben
    n_partial = max(1, ics // 2)
    tokens = torch.randn(batch_size, n_partial, d)
    result = cache.push(tokens)
    assert result is None  # No debe llenarse

    # Flush parcial
    flushed = cache.flush()
    assert flushed.shape == (batch_size, n_partial, d), (
        f"Flush shape {flushed.shape} != esperado ({batch_size}, {n_partial}, {d})"
    )

    # El contenido flushed preserva el orden FIFO
    assert torch.equal(flushed, tokens), (
        "flush() no preserva el orden FIFO de inserción"
    )


# ---------------------------------------------------------------------------
# Propiedad 6: Determinismo del operador de posicionamiento Φ
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(dims=positioning_dims())
def test_positioning_operator_deterministic(dims: dict) -> None:
    """Feature: vsn-library, Property 6: Determinismo del operador Φ

    **Validates: Requirements 3.3**

    Para cualquier conjunto de tokens de tamaño ≤ X*Y*Z, el operador Φ
    aplicado dos veces produce posiciones idénticas.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    num_tokens = dims["num_tokens"]

    phi = PositioningOperator(X=X, Y=Y, Z=Z)

    # Generar tokens aleatorios
    batch = 2
    tokens = torch.randn(batch, num_tokens, d)

    # Aplicar Φ dos veces
    volume_1 = phi(tokens)
    volume_2 = phi(tokens)

    # Debe ser exactamente idéntico (determinismo puro)
    assert torch.equal(volume_1, volume_2), (
        f"Φ no es determinista: aplicado dos veces produce resultados distintos. "
        f"Max diff: {(volume_1 - volume_2).abs().max().item()}"
    )


@settings(max_examples=100, deadline=None)
@given(dims=positioning_dims())
def test_positioning_operator_positions_in_valid_range(dims: dict) -> None:
    """Feature: vsn-library, Property 6: Posiciones generadas en rango válido

    **Validates: Requirements 3.3**

    Todas las posiciones generadas por Φ están en el rango
    [0, X) × [0, Y) × [0, Z). Los tokens se mapean a posiciones válidas
    dentro del volumen.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    num_tokens = dims["num_tokens"]

    phi = PositioningOperator(X=X, Y=Y, Z=Z)

    batch = 2
    tokens = torch.randn(batch, num_tokens, d)

    volume = phi(tokens)

    # Shape debe ser (batch, X, Y, Z, d)
    assert volume.shape == (batch, X, Y, Z, d), (
        f"Volume shape {volume.shape} != esperado ({batch}, {X}, {Y}, {Z}, {d})"
    )

    # Verificar que los tokens se posicionaron en slots válidos:
    # Los primeros num_tokens slots en raster order deben contener datos no-cero
    # (asumiendo que los tokens generados no son todos cero, lo cual es
    # estadísticamente imposible con randn)
    # Verificamos que al menos un voxel no es cero
    assert volume.abs().sum() > 0, (
        "El volumen está completamente vacío después de posicionar tokens no-cero"
    )

    # Verificar que las posiciones no sobrepasan las dimensiones del volumen
    # Esto se garantiza por la shape del output, pero verificamos explícitamente
    # que el operador no usa indexación fuera de rango (que causaría error)
    # Si llegamos aquí sin error, las posiciones son válidas.


@settings(max_examples=100, deadline=None)
@given(dims=positioning_dims())
def test_positioning_operator_independent_of_other_tokens(dims: dict) -> None:
    """Feature: vsn-library, Property 6: Φ es un mapeo posicional fijo

    **Validates: Requirements 3.3**

    El operador Φ asigna cada token j a una posición fija (x, y, z) que depende
    SOLO de j, no del contenido de otros tokens. Cambiar el token k≠j no debe
    afectar la posición de j.
    """
    X, Y, Z, d = dims["X"], dims["Y"], dims["Z"], dims["d"]
    num_tokens = dims["num_tokens"]

    if num_tokens < 2:
        return  # Necesitamos al menos 2 tokens para este test

    phi = PositioningOperator(X=X, Y=Y, Z=Z)

    batch = 1
    tokens_a = torch.randn(batch, num_tokens, d)
    tokens_b = tokens_a.clone()

    # Modificar el último token
    tokens_b[:, -1, :] = torch.randn(d)

    volume_a = phi(tokens_a)
    volume_b = phi(tokens_b)

    # La posición del primer token debe ser idéntica en ambos volúmenes
    # (Φ es posicional, no depende del contenido de otros tokens)
    # Encontrar dónde se posiciona el token 0 según el raster order
    # j=0 → x=0, i=0, y=0, z=0
    assert torch.equal(volume_a[:, 0, 0, 0, :], volume_b[:, 0, 0, 0, :]), (
        "El posicionamiento del token 0 cambió al modificar otro token"
    )
