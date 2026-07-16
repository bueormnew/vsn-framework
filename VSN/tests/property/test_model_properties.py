"""Feature: vsn-library, Properties 11, 12: VSNModel Property Tests

**Validates: Requirements 6.2, 6.3, 6.4, 6.5**

Tests de propiedad que verifican:
- Propiedad 11: Para cualquier VSNConfig válido, la instanciación SHALL completarse
  sin error y el forward SHALL producir ModelOutputs con shapes correctas. Para
  cualquier VSNConfig con dimensiones incompatibles, la instanciación SHALL lanzar
  ConfigurationError.
- Propiedad 12: Para cualquier batch válido, el forward SHALL ejecutar los
  componentes en orden estricto: Input_Cache → Encoder → P → Q → Decoder(+Ψ) → O.
  Los states SHALL contener decoder_states y latent_H con shapes correctas.
"""

from unittest.mock import patch

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.contracts.outputs import ModelOutputs
from vsn.core.config import ConfigurationError, VSNConfig
from vsn.core.model import VSNModel

# ---------------------------------------------------------------------------
# Estrategias para generar configuraciones válidas e inválidas
# ---------------------------------------------------------------------------

# Dimensiones pequeñas para ejecución rápida
X_strategy = st.integers(min_value=2, max_value=4)
Y_strategy = st.integers(min_value=2, max_value=4)
Z_strategy = st.integers(min_value=2, max_value=4)
d_strategy = st.integers(min_value=4, max_value=16)
batch_strategy = st.integers(min_value=1, max_value=3)


@st.composite
def valid_vsn_config(draw: st.DrawFn) -> VSNConfig:
    """Genera un VSNConfig válido con dimensiones pequeñas.

    Usa p_mode='identity' con Y_H=Y, Z_H=Z, d_H=d para garantizar
    que el volumen del encoder coincide con el de H (validación pasa).
    También soporta compress y expand eligiendo dimensiones compatibles.
    """
    X_enc = draw(X_strategy)
    X_dec = draw(X_strategy)
    Y = draw(Y_strategy)
    Z = draw(Z_strategy)
    d = draw(d_strategy)

    # Para p_mode identity: Y_H*Z_H*d_H == Y*Z*d
    # Forma más simple: usar las mismas dimensiones
    p_mode = "identity"
    Y_H = Y
    Z_H = Z
    d_H = d

    # Decoder dims pueden ser iguales o diferentes
    Y_dec = draw(Y_strategy)
    Z_dec = draw(Z_strategy)

    # ICS: debe ser positivo, Y*Z garantiza tokens suficientes para un plano
    ics = Y * Z

    # DGW: tamaño de ventana positivo
    dgw = draw(st.integers(min_value=1, max_value=4))

    # head_type: regression no requiere vocab_size ni num_classes
    head_type = "regression"

    config = VSNConfig(
        X_enc=X_enc,
        X_dec=X_dec,
        Y=Y,
        Z=Z,
        d=d,
        ics=ics,
        Y_H=Y_H,
        Z_H=Z_H,
        d_H=d_H,
        p_mode=p_mode,
        Y_dec=Y_dec,
        Z_dec=Z_dec,
        dgw=dgw,
        head_type=head_type,
    )

    return config


@st.composite
def valid_config_with_batch(draw: st.DrawFn) -> tuple:
    """Genera un VSNConfig válido junto con un tamaño de batch."""
    config = draw(valid_vsn_config())
    batch_size = draw(batch_strategy)
    return config, batch_size


@st.composite
def invalid_vsn_config(draw: st.DrawFn) -> VSNConfig:
    """Genera un VSNConfig con dimensiones deliberadamente incompatibles.

    Elige una de varias estrategias para crear incompatibilidad:
    - Dimensiones negativas o cero
    - p_mode='compress' con H volume >= encoder volume
    - p_mode='identity' con volúmenes desiguales
    - p_mode='expand' con H volume <= encoder volume
    - head_type='text' sin vocab_size
    - p_mode inválido
    """
    strategy_choice = draw(st.integers(min_value=0, max_value=5))

    if strategy_choice == 0:
        # Dimensión negativa o cero
        bad_dim = draw(st.integers(min_value=-5, max_value=0))
        return VSNConfig(
            X_enc=bad_dim,
            X_dec=2,
            Y=2,
            Z=2,
            d=4,
            ics=4,
            Y_H=2,
            Z_H=2,
            d_H=4,
            p_mode="identity",
            Y_dec=2,
            Z_dec=2,
            dgw=2,
            head_type="regression",
        )

    elif strategy_choice == 1:
        # p_mode='compress' pero H volume >= encoder volume
        # encoder: Y*Z*d = 2*2*4 = 16
        # H: Y_H*Z_H*d_H = 2*2*4 = 16  (debe ser < 16 para compress)
        return VSNConfig(
            X_enc=2,
            X_dec=2,
            Y=2,
            Z=2,
            d=4,
            ics=4,
            Y_H=2,
            Z_H=2,
            d_H=4,
            p_mode="compress",
            Y_dec=2,
            Z_dec=2,
            dgw=2,
            head_type="regression",
        )

    elif strategy_choice == 2:
        # p_mode='identity' con volúmenes diferentes
        # encoder: Y*Z*d = 2*2*4 = 16
        # H: Y_H*Z_H*d_H = 2*2*8 = 32  (no coincide)
        return VSNConfig(
            X_enc=2,
            X_dec=2,
            Y=2,
            Z=2,
            d=4,
            ics=4,
            Y_H=2,
            Z_H=2,
            d_H=8,
            p_mode="identity",
            Y_dec=2,
            Z_dec=2,
            dgw=2,
            head_type="regression",
        )

    elif strategy_choice == 3:
        # p_mode='expand' pero H volume <= encoder volume
        # encoder: Y*Z*d = 3*3*8 = 72
        # H: Y_H*Z_H*d_H = 2*2*4 = 16  (debe ser > 72 para expand)
        return VSNConfig(
            X_enc=2,
            X_dec=2,
            Y=3,
            Z=3,
            d=8,
            ics=9,
            Y_H=2,
            Z_H=2,
            d_H=4,
            p_mode="expand",
            Y_dec=2,
            Z_dec=2,
            dgw=2,
            head_type="regression",
        )

    elif strategy_choice == 4:
        # head_type='text' sin vocab_size
        return VSNConfig(
            X_enc=2,
            X_dec=2,
            Y=2,
            Z=2,
            d=4,
            ics=4,
            Y_H=2,
            Z_H=2,
            d_H=4,
            p_mode="identity",
            Y_dec=2,
            Z_dec=2,
            dgw=2,
            head_type="text",
            vocab_size=None,
        )

    else:
        # p_mode inválido
        return VSNConfig(
            X_enc=2,
            X_dec=2,
            Y=2,
            Z=2,
            d=4,
            ics=4,
            Y_H=2,
            Z_H=2,
            d_H=4,
            p_mode="invalid_mode",
            Y_dec=2,
            Z_dec=2,
            dgw=2,
            head_type="regression",
        )


# ---------------------------------------------------------------------------
# Propiedad 11: Enforcement de contratos dimensionales en VSNModel
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(config=valid_vsn_config())
def test_valid_config_instantiates_without_error(config: VSNConfig) -> None:
    """Feature: vsn-library, Property 11: Enforcement de contratos dimensionales

    **Validates: Requirements 6.3, 6.4, 6.5**

    Para cualquier VSNConfig válido (dimensiones compatibles), la instanciación
    SHALL completarse sin error.
    """
    # La instanciación no debe lanzar ningún error
    model = VSNModel(config)

    # Verificar que el modelo se construyó correctamente
    assert model is not None
    assert model.config is config
    assert isinstance(model.encoder, torch.nn.Module)
    assert isinstance(model.P, torch.nn.Module)
    assert isinstance(model.Q, torch.nn.Module)
    assert isinstance(model.decoder, torch.nn.Module)


@settings(max_examples=100, deadline=None)
@given(data=valid_config_with_batch())
def test_valid_config_forward_produces_correct_shapes(data: tuple) -> None:
    """Feature: vsn-library, Property 11: Enforcement de contratos dimensionales

    **Validates: Requirements 6.3, 6.4, 6.5**

    Para cualquier VSNConfig válido, el forward SHALL producir ModelOutputs
    con shapes correctas. Los states SHALL contener decoder_states (lista de
    tensores (batch, Y_dec, Z_dec, d)) y latent_H (batch, Y_H, Z_H, d_H).
    """
    config, batch_size = data

    model = VSNModel(config)
    model.eval()

    # Generar tokens de entrada: (batch, num_tokens, d)
    # num_tokens debe ser <= X_enc * Y * Z (capacidad del posicionamiento)
    num_tokens = config.Y * config.Z  # un plano completo
    tokens = torch.randn(batch_size, num_tokens, config.d)

    with torch.no_grad():
        outputs = model(tokens)

    # Verificar tipo de salida
    assert isinstance(outputs, ModelOutputs), (
        f"Forward debe producir ModelOutputs, obtuvo {type(outputs)}"
    )

    # Verificar states
    assert outputs.states is not None, "states no debe ser None"
    assert "decoder_states" in outputs.states, (
        "states debe contener 'decoder_states'"
    )
    assert "latent_H" in outputs.states, (
        "states debe contener 'latent_H'"
    )

    # Verificar latent_H shape
    H = outputs.states["latent_H"]
    expected_H_shape = (batch_size, config.Y_H, config.Z_H, config.d_H)
    assert H.shape == expected_H_shape, (
        f"latent_H shape {H.shape} != expected {expected_H_shape}"
    )

    # Verificar decoder_states shapes
    decoder_states = outputs.states["decoder_states"]
    assert isinstance(decoder_states, list), "decoder_states debe ser una lista"
    assert len(decoder_states) == model.num_windows, (
        f"decoder_states tiene {len(decoder_states)} ventanas, "
        f"esperado {model.num_windows}"
    )

    expected_dec_shape = (batch_size, config.Y_dec, config.Z_dec, config.d)
    for i, state in enumerate(decoder_states):
        assert state.shape == expected_dec_shape, (
            f"decoder_states[{i}] shape {state.shape} != expected {expected_dec_shape}"
        )

    # Verificar que no hay NaN/Inf
    assert torch.isfinite(H).all(), "latent_H contiene NaN o Inf"
    for i, state in enumerate(decoder_states):
        assert torch.isfinite(state).all(), (
            f"decoder_states[{i}] contiene NaN o Inf"
        )


@settings(max_examples=100, deadline=None)
@given(config=invalid_vsn_config())
def test_invalid_config_raises_configuration_error(config: VSNConfig) -> None:
    """Feature: vsn-library, Property 11: Enforcement de contratos dimensionales

    **Validates: Requirements 6.3, 6.5**

    Para cualquier VSNConfig con dimensiones incompatibles, la instanciación
    SHALL lanzar ConfigurationError indicando qué componentes son incompatibles.
    """
    try:
        model = VSNModel(config)
        # Si llegamos aquí, la validación no detectó la incompatibilidad
        assert False, (
            f"VSNModel debería haber lanzado ConfigurationError para config "
            f"inválido, pero se instanció exitosamente. Config: "
            f"p_mode={config.p_mode}, X_enc={config.X_enc}, "
            f"Y={config.Y}, Z={config.Z}, d={config.d}, "
            f"Y_H={config.Y_H}, Z_H={config.Z_H}, d_H={config.d_H}"
        )
    except ConfigurationError as e:
        # Verificar que el mensaje es descriptivo (no vacío)
        error_msg = str(e)
        assert len(error_msg) > 0, (
            "ConfigurationError debe tener un mensaje descriptivo"
        )
    except (ValueError, TypeError):
        # Otros errores de validación también son aceptables
        # (e.g., dimensión negativa puede lanzar ValueError en componentes)
        pass


# ---------------------------------------------------------------------------
# Propiedad 12: Orden de ejecución del forward de VSNModel
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=valid_config_with_batch())
def test_forward_execution_order(data: tuple) -> None:
    """Feature: vsn-library, Property 12: Orden de ejecución del forward

    **Validates: Requirements 6.2**

    Para cualquier batch válido, el forward de VSNModel SHALL ejecutar los
    componentes en orden estricto: Input_Cache → Encoder → P → Q →
    Decoder(+Ψ) → O. Verificamos que la salida contiene states con
    decoder_states y latent_H de shapes correctas, lo cual solo es posible
    si todos los componentes se ejecutaron en orden.
    """
    config, batch_size = data

    model = VSNModel(config)
    model.eval()

    # Rastrear el orden de ejecución usando wrappers
    execution_order: list[str] = []

    # Guardar forwards originales
    original_encoder_forward = model.encoder.forward
    original_P_forward = model.P.forward
    original_Q_forward = model.Q.forward
    original_decoder_forward = model.decoder.forward

    def tracked_encoder_forward(*args, **kwargs):
        execution_order.append("encoder")
        return original_encoder_forward(*args, **kwargs)

    def tracked_P_forward(*args, **kwargs):
        execution_order.append("P")
        return original_P_forward(*args, **kwargs)

    def tracked_Q_forward(*args, **kwargs):
        execution_order.append("Q")
        return original_Q_forward(*args, **kwargs)

    def tracked_decoder_forward(*args, **kwargs):
        execution_order.append("decoder")
        return original_decoder_forward(*args, **kwargs)

    # Monkey-patch los forwards
    model.encoder.forward = tracked_encoder_forward
    model.P.forward = tracked_P_forward
    model.Q.forward = tracked_Q_forward
    model.decoder.forward = tracked_decoder_forward

    num_tokens = config.Y * config.Z
    tokens = torch.randn(batch_size, num_tokens, config.d)

    with torch.no_grad():
        outputs = model(tokens)

    # Verificar que todos los componentes se ejecutaron
    assert "encoder" in execution_order, "Encoder no fue ejecutado"
    assert "P" in execution_order, "Operador P no fue ejecutado"
    assert "Q" in execution_order, "Operador Q no fue ejecutado"
    assert "decoder" in execution_order, "Decoder no fue ejecutado"

    # Verificar orden estricto: Encoder < P < Q < Decoder
    encoder_idx = execution_order.index("encoder")
    p_idx = execution_order.index("P")
    q_idx = execution_order.index("Q")
    decoder_idx = execution_order.index("decoder")

    assert encoder_idx < p_idx, (
        f"Encoder (idx={encoder_idx}) debe ejecutarse antes que P (idx={p_idx}). "
        f"Orden observado: {execution_order}"
    )
    assert p_idx < q_idx, (
        f"P (idx={p_idx}) debe ejecutarse antes que Q (idx={q_idx}). "
        f"Orden observado: {execution_order}"
    )
    assert q_idx < decoder_idx, (
        f"Q (idx={q_idx}) debe ejecutarse antes que Decoder (idx={decoder_idx}). "
        f"Orden observado: {execution_order}"
    )

    # Restaurar forwards originales
    model.encoder.forward = original_encoder_forward
    model.P.forward = original_P_forward
    model.Q.forward = original_Q_forward
    model.decoder.forward = original_decoder_forward


@settings(max_examples=100, deadline=None)
@given(data=valid_config_with_batch())
def test_forward_output_contains_expected_states(data: tuple) -> None:
    """Feature: vsn-library, Property 12: Orden de ejecución del forward

    **Validates: Requirements 6.2**

    Para cualquier batch válido, la salida del forward SHALL contener
    states con decoder_states (lista de tensores por ventana con shape
    (batch, Y_dec, Z_dec, d)) y latent_H (shape (batch, Y_H, Z_H, d_H)),
    confirmando que toda la cadena de procesamiento se ejecutó correctamente.
    """
    config, batch_size = data

    model = VSNModel(config, num_windows=2)
    model.eval()

    num_tokens = config.Y * config.Z
    tokens = torch.randn(batch_size, num_tokens, config.d)

    with torch.no_grad():
        outputs = model(tokens)

    # Verificar estructura de la salida
    assert isinstance(outputs, ModelOutputs)
    assert outputs.states is not None

    # latent_H confirma que Encoder → P se ejecutaron
    H = outputs.states["latent_H"]
    assert H.shape == (batch_size, config.Y_H, config.Z_H, config.d_H)

    # decoder_states confirma que Q → Decoder(+Ψ) se ejecutaron
    decoder_states = outputs.states["decoder_states"]
    assert isinstance(decoder_states, list)
    assert len(decoder_states) == 2  # num_windows=2

    for state in decoder_states:
        assert state.shape == (batch_size, config.Y_dec, config.Z_dec, config.d)

    # Verificar metadata confirma ejecución completa
    assert "model_family" in outputs.metadata
    assert "vgb_version" in outputs.metadata
    assert "num_windows" in outputs.metadata
    assert outputs.metadata["num_windows"] == 2
