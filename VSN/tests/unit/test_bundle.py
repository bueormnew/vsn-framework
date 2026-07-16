"""Tests unitarios para vsn.formats.bundle.

Verifica:
- export_bundle crea el directorio y archivos correctos
- load_bundle carga y reconstruye el modelo correctamente
- Round-trip: export → load produce modelo con outputs idénticos
- Validación de integridad detecta corrupción de archivos
- Soporte para safetensors (si disponible)

Validates: Requirements 9.3, 9.4
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.formats.bundle import (
    BundleIntegrityError,
    BundleManifest,
    _compute_sha256,
    export_bundle,
    load_bundle,
)


@pytest.fixture
def small_config() -> VSNConfig:
    """Config small para tests rápidos."""
    return VSNConfig.small()


@pytest.fixture
def small_model(small_config: VSNConfig) -> VSNModel:
    """Modelo small instanciado para tests."""
    return VSNModel(small_config)


class TestBundleManifest:
    """Tests para BundleManifest dataclass."""

    def test_to_dict_contains_all_fields(self, small_model: VSNModel):
        """Serialización a dict incluye todos los campos requeridos."""
        from vsn.io.state_schema import StateSchema

        total_params = sum(p.numel() for p in small_model.parameters())
        schema = StateSchema.from_config(small_model.config, total_params)

        manifest = BundleManifest(
            schema=schema,
            weight_format="pytorch",
            weight_files=["weights.pt"],
            config_file="model_config.json",
            metadata={"export_version": "1.0"},
            checksums={"weights.pt": "abc123"},
        )

        d = manifest.to_dict()
        assert "schema" in d
        assert d["weight_format"] == "pytorch"
        assert d["weight_files"] == ["weights.pt"]
        assert d["config_file"] == "model_config.json"
        assert d["metadata"]["export_version"] == "1.0"
        assert d["checksums"]["weights.pt"] == "abc123"

    def test_round_trip_json(self, small_model: VSNModel):
        """Serialización JSON round-trip preserva datos."""
        from vsn.io.state_schema import StateSchema

        total_params = sum(p.numel() for p in small_model.parameters())
        schema = StateSchema.from_config(small_model.config, total_params)

        original = BundleManifest(
            schema=schema,
            weight_format="pytorch",
            weight_files=["weights.pt"],
            config_file="model_config.json",
            metadata={"key": "value"},
            checksums={"weights.pt": "deadbeef"},
        )

        json_str = original.to_json()
        loaded = BundleManifest.from_json(json_str)

        assert loaded.weight_format == original.weight_format
        assert loaded.weight_files == original.weight_files
        assert loaded.config_file == original.config_file
        assert loaded.metadata == original.metadata
        assert loaded.checksums == original.checksums
        assert loaded.schema.schema_version == original.schema.schema_version

    def test_from_dict_missing_fields_raises(self):
        """from_dict lanza BundleIntegrityError si faltan campos."""
        with pytest.raises(BundleIntegrityError, match="missing required fields"):
            BundleManifest.from_dict({"weight_format": "pytorch"})

    def test_from_json_invalid_json_raises(self):
        """from_json lanza BundleIntegrityError con JSON inválido."""
        with pytest.raises(BundleIntegrityError, match="Invalid manifest JSON"):
            BundleManifest.from_json("not-valid-json{{{")


class TestExportBundle:
    """Tests para export_bundle."""

    def test_creates_all_files(self, tmp_path: Path, small_model: VSNModel):
        """export_bundle crea manifest, config, weights y schema."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "model_config.json").exists()
        assert (bundle_dir / "weights.pt").exists()
        assert (bundle_dir / "schema.json").exists()

    def test_manifest_contains_valid_checksums(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """El manifest contiene checksums SHA-256 válidos."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        manifest_data = json.loads(
            (bundle_dir / "manifest.json").read_text(encoding="utf-8")
        )
        checksums = manifest_data["checksums"]

        # Verificar que los checksums coinciden con los archivos reales
        for filename, expected_hash in checksums.items():
            actual_hash = _compute_sha256(bundle_dir / filename)
            assert actual_hash == expected_hash, (
                f"Checksum mismatch for {filename}"
            )

    def test_config_json_matches_model_config(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """model_config.json refleja la config del modelo exportado."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        config_data = json.loads(
            (bundle_dir / "model_config.json").read_text(encoding="utf-8")
        )
        assert config_data["X_enc"] == small_model.config.X_enc
        assert config_data["Y"] == small_model.config.Y
        assert config_data["d"] == small_model.config.d
        assert config_data["head_type"] == small_model.config.head_type

    def test_invalid_weight_format_raises(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Formato de pesos no soportado lanza BundleIntegrityError."""
        with pytest.raises(BundleIntegrityError, match="Unsupported weight_format"):
            export_bundle(small_model, tmp_path / "bad", weight_format="onnx")

    def test_custom_metadata_preserved(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Metadata custom se incluye en el manifest."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(
            small_model, bundle_dir, metadata={"author": "test", "version": "2.0"}
        )

        manifest_data = json.loads(
            (bundle_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_data["metadata"]["author"] == "test"
        assert manifest_data["metadata"]["version"] == "2.0"

    def test_returns_output_dir_path(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """export_bundle retorna la ruta al directorio del bundle."""
        bundle_dir = tmp_path / "test_bundle"
        result = export_bundle(small_model, bundle_dir)
        assert result == bundle_dir


class TestLoadBundle:
    """Tests para load_bundle."""

    def test_loads_model_successfully(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """load_bundle reconstruye el modelo correctamente."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        loaded_model = load_bundle(bundle_dir)

        assert isinstance(loaded_model, VSNModel)
        assert loaded_model.config.X_enc == small_model.config.X_enc
        assert loaded_model.config.Y == small_model.config.Y
        assert loaded_model.config.d == small_model.config.d

    def test_round_trip_numerical_equivalence(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Export → load produce outputs numéricamente idénticos."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        loaded_model = load_bundle(bundle_dir)

        # Crear input de prueba
        batch_size = 2
        num_tokens = small_model.config.ics
        d = small_model.config.d
        tokens = torch.randn(batch_size, num_tokens, d)

        # Comparar outputs
        small_model.eval()
        loaded_model.eval()
        with torch.no_grad():
            out_original = small_model(tokens)
            out_loaded = loaded_model(tokens)

        # Verificar equivalencia numérica
        for orig_state, load_state in zip(
            out_original.states["decoder_states"],
            out_loaded.states["decoder_states"],
        ):
            assert torch.allclose(orig_state, load_state, atol=1e-6), (
                "Decoder states differ after round-trip"
            )

    def test_nonexistent_dir_raises(self, tmp_path: Path):
        """Directorio inexistente lanza BundleIntegrityError."""
        with pytest.raises(BundleIntegrityError, match="not found"):
            load_bundle(tmp_path / "nonexistent")

    def test_missing_manifest_raises(self, tmp_path: Path):
        """Directorio sin manifest lanza BundleIntegrityError."""
        bundle_dir = tmp_path / "empty_bundle"
        bundle_dir.mkdir()
        with pytest.raises(BundleIntegrityError, match="Manifest file not found"):
            load_bundle(bundle_dir)

    def test_corrupted_weights_detected(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Pesos corruptos detectados por checksum validation."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        # Corromper el archivo de pesos
        weights_path = bundle_dir / "weights.pt"
        weights_path.write_bytes(b"corrupted data")

        with pytest.raises(BundleIntegrityError, match="Checksum mismatch"):
            load_bundle(bundle_dir)

    def test_corrupted_config_detected(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Config corrupta detectada por checksum validation."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        # Corromper el archivo de config
        config_path = bundle_dir / "model_config.json"
        config_path.write_text('{"corrupted": true}', encoding="utf-8")

        with pytest.raises(BundleIntegrityError, match="Checksum mismatch"):
            load_bundle(bundle_dir)

    def test_unsupported_schema_version_detected(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Schema version no soportada lanza BundleIntegrityError."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        # Modificar el schema_version en el manifest
        manifest_path = bundle_dir / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_data["schema"]["schema_version"] = "99.0"
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )

        with pytest.raises(BundleIntegrityError, match="Unsupported schema_version"):
            load_bundle(bundle_dir)

    def test_missing_weight_file_detected(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Archivo de pesos faltante detectado."""
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(small_model, bundle_dir)

        # Eliminar el archivo de pesos
        (bundle_dir / "weights.pt").unlink()

        with pytest.raises(BundleIntegrityError, match="Bundle file missing"):
            load_bundle(bundle_dir)


class TestSafetensorsSupport:
    """Tests para soporte opcional de safetensors."""

    def test_safetensors_export_if_available(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Si safetensors está instalado, exporta en ese formato."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        bundle_dir = tmp_path / "st_bundle"
        export_bundle(small_model, bundle_dir, weight_format="safetensors")

        assert (bundle_dir / "weights.safetensors").exists()
        assert not (bundle_dir / "weights.pt").exists()

        # Verificar manifest
        manifest_data = json.loads(
            (bundle_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_data["weight_format"] == "safetensors"
        assert "weights.safetensors" in manifest_data["weight_files"]

    def test_safetensors_round_trip_if_available(
        self, tmp_path: Path, small_model: VSNModel
    ):
        """Round-trip con safetensors produce outputs equivalentes."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        bundle_dir = tmp_path / "st_bundle"
        export_bundle(small_model, bundle_dir, weight_format="safetensors")

        loaded_model = load_bundle(bundle_dir)

        # Comparar parámetros
        for (name, p_orig), (_, p_load) in zip(
            small_model.named_parameters(), loaded_model.named_parameters()
        ):
            assert torch.equal(p_orig, p_load), (
                f"Parameter {name} differs after safetensors round-trip"
            )
