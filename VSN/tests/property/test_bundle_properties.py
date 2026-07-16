"""Feature: vsn-library, Properties 17, 18: Bundle de Inferencia Property Tests

**Validates: Requirements 9.3, 9.4, 13.6**

Tests de propiedad que verifican:
- Propiedad 17: Para cualquier VSNModel entrenado (random params) y cualquier input
  batch válido, export_bundle luego load_bundle SHALL producir outputs numéricamente
  idénticos (dentro de tolerancia fp) al modelo original.
- Propiedad 18: Para cualquier Bundle_Inferencia válido, si se corrompe cualquier
  componente (schema_version, checksum, dimensiones, pesos), la carga SHALL fallar
  con BundleIntegrityError indicando qué componente es inválido.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.formats.bundle import (
    BundleIntegrityError,
    export_bundle,
    load_bundle,
)


# ---------------------------------------------------------------------------
# Estrategias para generar configuraciones válidas con dims pequeñas
# ---------------------------------------------------------------------------

@st.composite
def small_vsn_config(draw: st.DrawFn) -> VSNConfig:
    """Genera un VSNConfig válido con dimensiones pequeñas para tests rápidos.

    Usa regression head (no necesita vocab_size ni num_classes).
    Dimensiones: X=2-3, Y=2-3, Z=2-3, d=4-8.
    """
    X_enc = draw(st.integers(min_value=2, max_value=3))
    X_dec = draw(st.integers(min_value=2, max_value=3))
    Y = draw(st.integers(min_value=2, max_value=3))
    Z = draw(st.integers(min_value=2, max_value=3))
    d = draw(st.integers(min_value=4, max_value=8))

    # Para identity mode: Y_H*Z_H*d_H == Y*Z*d
    # Usamos compress mode con H más pequeño para simplicidad
    # O identity con mismas dims
    p_mode = "identity"
    Y_H = Y
    Z_H = Z
    d_H = d

    Y_dec = Y
    Z_dec = Z
    ics = Y * Z  # Input cache size = fits the plane
    dgw = draw(st.integers(min_value=1, max_value=2))

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
        head_type="regression",
        vocab_size=None,
        num_classes=None,
    )
    return config


@st.composite
def model_and_input(draw: st.DrawFn) -> dict:
    """Genera un VSNModel con parámetros aleatorios y un batch de input válido."""
    config = draw(small_vsn_config())
    batch_size = draw(st.integers(min_value=1, max_value=2))

    # Crear modelo
    model = VSNModel(config, head=None, num_windows=1)
    model.eval()

    # Generar tokens de input: (batch, num_tokens, d)
    # num_tokens debe ser al menos ics para que el encoder funcione
    num_tokens = config.ics
    tokens = torch.randn(batch_size, num_tokens, config.d)

    return {
        "config": config,
        "model": model,
        "tokens": tokens,
        "batch_size": batch_size,
    }


# Estrategia para el tipo de corrupción
corruption_type = st.sampled_from([
    "schema_version",
    "checksum",
    "dimensions",
    "weights",
])


# ---------------------------------------------------------------------------
# Propiedad 17: Bundle de inferencia round-trip
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=model_and_input())
def test_bundle_roundtrip_produces_identical_outputs(data: dict) -> None:
    """Feature: vsn-library, Property 17: Bundle de inferencia round-trip

    **Validates: Requirements 9.3, 13.6**

    Para cualquier VSNModel entrenado (random params) y cualquier input batch
    válido, export_bundle luego load_bundle SHALL producir outputs numéricamente
    idénticos (dentro de tolerancia fp) al modelo original.
    """
    model = data["model"]
    tokens = data["tokens"]

    # Crear directorio temporal para el bundle
    tmp_dir = tempfile.mkdtemp()
    bundle_dir = Path(tmp_dir) / "bundle"

    try:
        # Obtener output del modelo original
        model.eval()
        with torch.no_grad():
            original_output = model(tokens)

        # Exportar bundle
        export_bundle(model, bundle_dir)

        # Cargar modelo desde bundle (sin head, igual que el original)
        loaded_model = load_bundle(bundle_dir, device="cpu", head=None)
        loaded_model.eval()

        # Obtener output del modelo cargado
        with torch.no_grad():
            loaded_output = loaded_model(tokens)

        # Verificar que los outputs son numéricamente idénticos
        # Comparamos los decoder states (ya que no hay head)
        assert original_output.states is not None, "Original output missing states"
        assert loaded_output.states is not None, "Loaded output missing states"

        orig_states = original_output.states["decoder_states"]
        load_states = loaded_output.states["decoder_states"]

        assert len(orig_states) == len(load_states), (
            f"Number of decoder windows differs: {len(orig_states)} vs {len(load_states)}"
        )

        for i, (orig_s, load_s) in enumerate(zip(orig_states, load_states)):
            assert torch.allclose(orig_s, load_s, atol=1e-6, rtol=1e-5), (
                f"Decoder state window {i} differs after bundle round-trip. "
                f"Max abs diff: {(orig_s - load_s).abs().max().item():.2e}"
            )

        # Verificar latent_H también
        orig_H = original_output.states["latent_H"]
        load_H = loaded_output.states["latent_H"]
        assert torch.allclose(orig_H, load_H, atol=1e-6, rtol=1e-5), (
            f"Latent H differs after bundle round-trip. "
            f"Max abs diff: {(orig_H - load_H).abs().max().item():.2e}"
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Propiedad 18: Validación de bundle detecta corrupción
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(config=small_vsn_config(), corrupt=corruption_type)
def test_bundle_detects_corruption(config: VSNConfig, corrupt: str) -> None:
    """Feature: vsn-library, Property 18: Validación de bundle detecta corrupción

    **Validates: Requirements 9.4**

    Para cualquier Bundle_Inferencia válido, si se corrompe cualquier componente
    (schema_version, checksum, dimensiones, pesos), la carga SHALL fallar con
    BundleIntegrityError indicando qué componente es inválido.
    """
    # Crear modelo y exportar bundle válido
    model = VSNModel(config, head=None, num_windows=1)
    model.eval()

    tmp_dir = tempfile.mkdtemp()
    bundle_dir = Path(tmp_dir) / "bundle"

    try:
        export_bundle(model, bundle_dir)

        # Aplicar corrupción según el tipo
        if corrupt == "schema_version":
            _corrupt_schema_version(bundle_dir)
        elif corrupt == "checksum":
            _corrupt_checksum(bundle_dir)
        elif corrupt == "dimensions":
            _corrupt_dimensions(bundle_dir)
        elif corrupt == "weights":
            _corrupt_weights(bundle_dir)

        # Intentar cargar el bundle corrupto — DEBE lanzar BundleIntegrityError
        raised = False
        error_msg = ""
        try:
            load_bundle(bundle_dir, device="cpu")
        except BundleIntegrityError as e:
            raised = True
            error_msg = str(e)

        assert raised, (
            f"load_bundle did NOT raise BundleIntegrityError for corruption "
            f"type '{corrupt}'. Expected failure but load succeeded."
        )

        # Verificar que el mensaje de error es descriptivo (no vacío)
        assert len(error_msg) > 0, (
            f"BundleIntegrityError for corruption '{corrupt}' has empty message"
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Funciones auxiliares de corrupción
# ---------------------------------------------------------------------------


def _corrupt_schema_version(bundle_dir: Path) -> None:
    """Corrompe el schema_version en el manifest a una versión no soportada."""
    manifest_path = bundle_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["schema"]["schema_version"] = "99.99"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )


def _corrupt_checksum(bundle_dir: Path) -> None:
    """Corrompe un checksum en el manifest para que no coincida con el archivo."""
    manifest_path = bundle_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Corromper el checksum del primer archivo registrado
    checksums = manifest_data["checksums"]
    if checksums:
        first_key = next(iter(checksums))
        checksums[first_key] = "0" * 64  # SHA-256 falso
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )


def _corrupt_dimensions(bundle_dir: Path) -> None:
    """Corrompe las dimensiones en model_config.json para que no coincidan."""
    config_path = bundle_dir / "model_config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    # Cambiar una dimensión a valor inválido (0 o negativo)
    config_data["d"] = -1
    config_path.write_text(
        json.dumps(config_data, indent=2), encoding="utf-8"
    )


def _corrupt_weights(bundle_dir: Path) -> None:
    """Corrompe el archivo de pesos escribiendo datos basura."""
    weights_path = bundle_dir / "weights.pt"
    if weights_path.exists():
        # Escribir bytes aleatorios para corromper el archivo
        weights_path.write_bytes(b"CORRUPTED_DATA_" * 100)
