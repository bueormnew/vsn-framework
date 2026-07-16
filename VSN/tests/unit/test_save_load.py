"""Unit tests para vsn.io.save_load — save/load del core VSN.

Verifica:
- Round-trip save/load produce outputs idénticos
- schema_version se valida al cargar
- Manejo correcto de errores (archivo inexistente, checkpoint corrupto)

Validates: Requirements 9.3, 9.4
"""

import torch
import pytest

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.io.save_load import SaveLoadError, load_model, save_model


@pytest.fixture
def small_config() -> VSNConfig:
    """Config small para tests rápidos."""
    return VSNConfig.small(head_type="dense")


@pytest.fixture
def small_model(small_config: VSNConfig) -> VSNModel:
    """Modelo small sin head para tests."""
    return VSNModel(small_config)


class TestSaveModel:
    """Tests para save_model."""

    def test_save_creates_file(self, small_model: VSNModel, tmp_path):
        """save_model crea el archivo .pt en la ruta indicada."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, small_model: VSNModel, tmp_path):
        """save_model crea directorios intermedios si no existen."""
        path = tmp_path / "sub" / "dir" / "model.pt"
        save_model(small_model, path)
        assert path.exists()

    def test_save_checkpoint_structure(self, small_model: VSNModel, tmp_path):
        """El checkpoint guardado contiene state_dict, config y schema."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "state_dict" in checkpoint
        assert "config" in checkpoint
        assert "schema" in checkpoint

    def test_save_schema_has_correct_version(
        self, small_model: VSNModel, tmp_path
    ):
        """El schema guardado tiene schema_version='1.0'."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["schema"]["schema_version"] == "1.0"

    def test_save_config_matches_model(
        self, small_model: VSNModel, small_config: VSNConfig, tmp_path
    ):
        """La config guardada coincide con la del modelo."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["config"]["X_enc"] == small_config.X_enc
        assert checkpoint["config"]["d"] == small_config.d
        assert checkpoint["config"]["model_family"] == small_config.model_family


class TestLoadModel:
    """Tests para load_model."""

    def test_load_reconstructs_model(self, small_model: VSNModel, tmp_path):
        """load_model reconstruye un VSNModel funcional."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        loaded = load_model(path)
        assert isinstance(loaded, VSNModel)

    def test_load_preserves_config(
        self, small_model: VSNModel, small_config: VSNConfig, tmp_path
    ):
        """El modelo cargado tiene la misma config que el original."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        loaded = load_model(path)
        assert loaded.config.X_enc == small_config.X_enc
        assert loaded.config.d == small_config.d
        assert loaded.config.Y == small_config.Y

    def test_load_nonexistent_file_raises(self, tmp_path):
        """load_model lanza SaveLoadError si el archivo no existe."""
        path = tmp_path / "nonexistent.pt"
        with pytest.raises(SaveLoadError, match="not found"):
            load_model(path)

    def test_load_unsupported_schema_version_raises(
        self, small_model: VSNModel, tmp_path
    ):
        """load_model lanza SaveLoadError si schema_version no es soportado."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        # Corromper schema_version
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["schema"]["schema_version"] = "99.0"
        torch.save(checkpoint, path)

        with pytest.raises(SaveLoadError, match="Unsupported schema_version"):
            load_model(path)

    def test_load_corrupted_checkpoint_raises(self, tmp_path):
        """load_model lanza SaveLoadError si faltan keys en el checkpoint."""
        path = tmp_path / "model.pt"
        # Guardar un checkpoint incompleto
        torch.save({"state_dict": {}}, path)

        with pytest.raises(SaveLoadError, match="missing keys"):
            load_model(path)

    def test_load_with_custom_head(self, small_model: VSNModel, tmp_path):
        """load_model acepta un head personalizado al reconstruir."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)

        # El head pasado se asigna al modelo cargado
        custom_head = torch.nn.Linear(10, 5)
        loaded = load_model(path, head=custom_head)
        assert loaded.head is custom_head


class TestRoundTrip:
    """Tests de round-trip: save → load produce outputs idénticos."""

    def test_round_trip_identical_forward(
        self, small_model: VSNModel, small_config: VSNConfig, tmp_path
    ):
        """Forward del modelo cargado produce los mismos outputs que el original."""
        path = tmp_path / "model.pt"

        # Forward original
        small_model.eval()
        torch.manual_seed(42)
        tokens = torch.randn(1, small_config.ics, small_config.d)

        with torch.no_grad():
            original_out = small_model(tokens)

        # Save → Load
        save_model(small_model, path)
        loaded = load_model(path)
        loaded.eval()

        with torch.no_grad():
            loaded_out = loaded(tokens)

        # Comparar outputs
        for orig_state, load_state in zip(
            original_out.states["decoder_states"],
            loaded_out.states["decoder_states"],
        ):
            assert torch.allclose(orig_state, load_state, atol=1e-6), (
                "Forward outputs differ after round-trip save/load"
            )

        # Comparar latent H
        assert torch.allclose(
            original_out.states["latent_H"],
            loaded_out.states["latent_H"],
            atol=1e-6,
        )

    def test_round_trip_preserves_parameters(
        self, small_model: VSNModel, tmp_path
    ):
        """Los parámetros del modelo cargado son exactamente iguales."""
        path = tmp_path / "model.pt"
        save_model(small_model, path)
        loaded = load_model(path)

        for (name, orig_param), (_, load_param) in zip(
            small_model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.equal(orig_param, load_param), (
                f"Parameter '{name}' differs after round-trip"
            )

    def test_round_trip_string_path(
        self, small_model: VSNModel, tmp_path
    ):
        """save/load aceptan string como path (no solo Path)."""
        path = str(tmp_path / "model.pt")
        save_model(small_model, path)
        loaded = load_model(path)
        assert isinstance(loaded, VSNModel)
