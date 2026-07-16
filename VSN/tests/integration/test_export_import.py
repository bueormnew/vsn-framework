"""Integration test: exportar bundle → reimportar → comparar outputs.

Verifica round-trip de persistencia:
- export_bundle → load_bundle produce outputs idénticos
- save_model → load_model produce outputs idénticos
- Los pesos y configuración se preservan en el round-trip

Validates: Requirements 13.3, 13.6
"""

from __future__ import annotations

import pytest
import torch

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.formats.bundle import export_bundle, load_bundle
from vsn.io.save_load import load_model, save_model


class TestExportImportBundle:
    """Tests de integración para export/import de bundles."""

    def _make_model_and_tokens(self):
        """Crea modelo y tokens sintéticos para tests de round-trip."""
        config = VSNConfig.small(head_type="regression")
        model = VSNModel(config, head=None)
        model.eval()

        num_tokens = config.Y * config.Z
        torch.manual_seed(42)
        tokens = torch.randn(2, num_tokens, config.d)

        return model, tokens

    def test_bundle_export_import_outputs_match(self, tmp_path) -> None:
        """export_bundle → load_bundle produce outputs idénticos (atol=1e-6)."""
        model, tokens = self._make_model_and_tokens()

        # Forward de referencia
        with torch.no_grad():
            ref_output = model(tokens)

        # Export bundle
        bundle_dir = tmp_path / "test_bundle"
        export_bundle(model, bundle_dir)

        # Import bundle
        loaded_model = load_bundle(bundle_dir)
        loaded_model.eval()

        # Forward con modelo cargado
        with torch.no_grad():
            loaded_output = loaded_model(tokens)

        # Comparar latent_H
        ref_H = ref_output.states["latent_H"]
        loaded_H = loaded_output.states["latent_H"]
        assert torch.allclose(ref_H, loaded_H, atol=1e-6), (
            f"latent_H mismatch: max diff = {(ref_H - loaded_H).abs().max().item()}"
        )

        # Comparar decoder_states
        ref_states = ref_output.states["decoder_states"]
        loaded_states = loaded_output.states["decoder_states"]
        assert len(ref_states) == len(loaded_states)

        for i, (ref_s, loaded_s) in enumerate(zip(ref_states, loaded_states)):
            assert torch.allclose(ref_s, loaded_s, atol=1e-6), (
                f"decoder_state[{i}] mismatch: "
                f"max diff = {(ref_s - loaded_s).abs().max().item()}"
            )

    def test_bundle_preserves_config(self, tmp_path) -> None:
        """El bundle preserva la configuración del modelo."""
        model, _ = self._make_model_and_tokens()

        bundle_dir = tmp_path / "config_bundle"
        export_bundle(model, bundle_dir)

        loaded_model = load_bundle(bundle_dir)

        # Comparar configs
        assert loaded_model.config.X_enc == model.config.X_enc
        assert loaded_model.config.X_dec == model.config.X_dec
        assert loaded_model.config.Y == model.config.Y
        assert loaded_model.config.Z == model.config.Z
        assert loaded_model.config.d == model.config.d
        assert loaded_model.config.Y_H == model.config.Y_H
        assert loaded_model.config.Z_H == model.config.Z_H
        assert loaded_model.config.d_H == model.config.d_H

    def test_bundle_multiple_windows(self, tmp_path) -> None:
        """Round-trip con num_windows > 1 produce outputs consistentes."""
        model, tokens = self._make_model_and_tokens()

        with torch.no_grad():
            ref_output = model(tokens, num_windows=3)

        bundle_dir = tmp_path / "multi_window_bundle"
        export_bundle(model, bundle_dir)

        loaded_model = load_bundle(bundle_dir)
        loaded_model.eval()

        with torch.no_grad():
            loaded_output = loaded_model(tokens, num_windows=3)

        ref_states = ref_output.states["decoder_states"]
        loaded_states = loaded_output.states["decoder_states"]
        assert len(ref_states) == 3
        assert len(loaded_states) == 3

        for i in range(3):
            assert torch.allclose(ref_states[i], loaded_states[i], atol=1e-6)


class TestSaveLoadModel:
    """Tests de integración para save_model/load_model round-trip."""

    def _make_model_and_tokens(self):
        """Crea modelo y tokens sintéticos."""
        config = VSNConfig.small(head_type="regression")
        model = VSNModel(config, head=None)
        model.eval()

        num_tokens = config.Y * config.Z
        torch.manual_seed(99)
        tokens = torch.randn(2, num_tokens, config.d)

        return model, tokens

    def test_save_load_outputs_match(self, tmp_path) -> None:
        """save_model → load_model produce outputs idénticos."""
        model, tokens = self._make_model_and_tokens()

        # Forward de referencia
        with torch.no_grad():
            ref_output = model(tokens)

        # Save
        model_path = tmp_path / "model.pt"
        save_model(model, model_path)

        # Load
        loaded_model = load_model(model_path)
        loaded_model.eval()

        # Forward con modelo cargado
        with torch.no_grad():
            loaded_output = loaded_model(tokens)

        # Comparar latent_H
        ref_H = ref_output.states["latent_H"]
        loaded_H = loaded_output.states["latent_H"]
        assert torch.allclose(ref_H, loaded_H, atol=1e-6)

        # Comparar decoder_states
        ref_states = ref_output.states["decoder_states"]
        loaded_states = loaded_output.states["decoder_states"]
        assert len(ref_states) == len(loaded_states)

        for i, (ref_s, loaded_s) in enumerate(zip(ref_states, loaded_states)):
            assert torch.allclose(ref_s, loaded_s, atol=1e-6)

    def test_save_load_preserves_parameter_count(self, tmp_path) -> None:
        """save/load preserva el número total de parámetros."""
        model, _ = self._make_model_and_tokens()

        original_params = sum(p.numel() for p in model.parameters())

        model_path = tmp_path / "model_params.pt"
        save_model(model, model_path)

        loaded_model = load_model(model_path)
        loaded_params = sum(p.numel() for p in loaded_model.parameters())

        assert original_params == loaded_params

    def test_save_load_weight_values_identical(self, tmp_path) -> None:
        """Los valores de pesos son bit-a-bit idénticos tras save/load."""
        model, _ = self._make_model_and_tokens()

        model_path = tmp_path / "model_weights.pt"
        save_model(model, model_path)

        loaded_model = load_model(model_path)

        for (name1, p1), (name2, p2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} != {name2}"
            assert torch.equal(p1.data, p2.data), (
                f"Weight values differ for '{name1}'"
            )
