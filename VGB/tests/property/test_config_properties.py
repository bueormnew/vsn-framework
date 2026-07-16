"""Feature: vsn-library, Property 19: Sistema de configuración — merge y validación

**Validates: Requirements 11.2, 11.3**

Tests de propiedad que verifican que:
- El merge de configuraciones respeta precedencia (CLI > specific config > base profile).
- Los CLI overrides siempre ganan sobre configuraciones de archivo.
- Las configuraciones finales inválidas son rechazadas con errores descriptivos.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from vgb.config.loader import apply_overrides, merge_configs
from vgb.config.schema import (
    FullConfig,
    InferConfig,
    ModelConfig,
    RuntimeConfig,
    TrainConfig,
    VGBConfigError,
)
from vsn.core.config import VSNConfig


# ---------------------------------------------------------------------------
# Estrategias para generar datos de configuración
# ---------------------------------------------------------------------------

# Claves simples para dicts de configuración
config_key_strategy = st.text(
    min_size=1, max_size=8, alphabet=st.characters(categories=("Ll",))
)

# Valores simples (int, float, str, bool) para dicts de configuración
config_value_strategy = st.one_of(
    st.integers(min_value=1, max_value=1000),
    st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("Ll",))),
    st.booleans(),
)

# Dicts de configuración planos (1 nivel)
flat_config_strategy = st.dictionaries(
    keys=config_key_strategy,
    values=config_value_strategy,
    min_size=1,
    max_size=5,
)


@st.composite
def two_configs_with_shared_key(draw: st.DrawFn) -> tuple:
    """Genera dos dicts que comparten al menos una clave con valores distintos."""
    shared_key = draw(config_key_strategy)
    val_base = draw(st.integers(min_value=1, max_value=500))
    val_override = draw(
        st.integers(min_value=501, max_value=1000)
    )

    base = draw(flat_config_strategy)
    override = draw(flat_config_strategy)

    base[shared_key] = val_base
    override[shared_key] = val_override

    return base, override, shared_key


@st.composite
def nested_configs_with_shared_key(draw: st.DrawFn) -> tuple:
    """Genera dos dicts anidados que comparten una clave en un sub-dict."""
    section = draw(config_key_strategy)
    field = draw(config_key_strategy)
    val_base = draw(st.integers(min_value=1, max_value=500))
    val_override = draw(st.integers(min_value=501, max_value=1000))

    base = {section: {field: val_base}}
    override = {section: {field: val_override}}

    return base, override, section, field


# Estrategia para generar valores de precision inválidos
invalid_precision_strategy = st.text(
    min_size=1, max_size=8, alphabet=st.characters(categories=("Ll", "Nd"))
).filter(lambda x: x not in ("bf16", "fp16", "fp32"))

# Estrategia para generar valores de strategy inválidos
invalid_strategy_strategy = st.text(
    min_size=1, max_size=10, alphabet=st.characters(categories=("Ll", "Nd"))
).filter(lambda x: x not in ("single", "fsdp2"))


# ---------------------------------------------------------------------------
# Helper: crear un FullConfig válido para mutar
# ---------------------------------------------------------------------------


def _make_valid_full_config() -> FullConfig:
    """Crea un FullConfig completamente válido como punto de partida."""
    vsn = VSNConfig(
        X_enc=4, X_dec=4, Y=4, Z=4, d=64, ics=64,
        Y_H=4, Z_H=4, d_H=64, p_mode="identity",
        Y_dec=4, Z_dec=4, dgw=4,
        head_type="text", vocab_size=32000, num_classes=None,
    )
    return FullConfig(
        model=ModelConfig(vsn=vsn),
        train=TrainConfig(),
        infer=InferConfig(),
        runtime=RuntimeConfig(),
    )


# ---------------------------------------------------------------------------
# Propiedad 19.1: Merge de configs respeta precedencia — later dict wins
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=two_configs_with_shared_key())
def test_merge_precedence_later_dict_wins(data: tuple) -> None:
    """Para cualquier par de dicts con una clave compartida,
    merge_configs retorna el valor del dict posterior para esa clave.

    Feature: vsn-library, Property 19: Sistema de configuración — merge y validación
    """
    base, override, shared_key = data

    merged = merge_configs(base, override)

    # El valor del override (segundo dict) debe ganar
    assert merged[shared_key] == override[shared_key]


# ---------------------------------------------------------------------------
# Propiedad 19.2: CLI overrides siempre ganan sobre configuraciones de archivo
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=nested_configs_with_shared_key())
def test_cli_overrides_always_win(data: tuple) -> None:
    """Para cualquier jerarquía profile → config → CLI,
    el valor CLI siempre prevalece.

    Feature: vsn-library, Property 19: Sistema de configuración — merge y validación
    """
    base, override, section, field = data

    # Simular: profile tiene un valor, config tiene otro, CLI override pone un tercero
    cli_value = 9999
    profile_dict = base
    config_dict = override

    # Merge profile + config
    merged_file = merge_configs(profile_dict, config_dict)

    # Aplicar CLI override
    cli_key = f"{section}.{field}={cli_value}"
    final = apply_overrides(merged_file, [cli_key])

    # CLI debe ganar siempre
    assert final[section][field] == cli_value


# ---------------------------------------------------------------------------
# Propiedad 19.3: Configuraciones inválidas son rechazadas con error descriptivo
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(bad_precision=invalid_precision_strategy)
def test_invalid_precision_rejected_with_descriptive_error(bad_precision: str) -> None:
    """Para cualquier valor de precision inválido, el sistema rechaza
    la configuración con un error que nombra el campo inválido.

    Feature: vsn-library, Property 19: Sistema de configuración — merge y validación
    """
    config = _make_valid_full_config()
    config.train.precision = bad_precision

    try:
        config.validate()
        assert False, "Se esperaba VGBConfigError pero la validación pasó"
    except VGBConfigError as e:
        error_msg = str(e)
        # El error debe mencionar el campo inválido
        assert "train.precision" in error_msg


@settings(max_examples=50, deadline=None)
@given(bad_strategy=invalid_strategy_strategy)
def test_invalid_strategy_rejected_with_descriptive_error(bad_strategy: str) -> None:
    """Para cualquier valor de runtime.strategy inválido, el sistema rechaza
    la configuración con un error que nombra el campo inválido.

    Feature: vsn-library, Property 19: Sistema de configuración — merge y validación
    """
    config = _make_valid_full_config()
    config.runtime.strategy = bad_strategy

    try:
        config.validate()
        assert False, "Se esperaba VGBConfigError pero la validación pasó"
    except VGBConfigError as e:
        error_msg = str(e)
        # El error debe mencionar el campo inválido
        assert "runtime.strategy" in error_msg
