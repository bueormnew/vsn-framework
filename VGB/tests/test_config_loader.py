"""Tests for vgb.config.loader — merge, overrides, and full config build.

Validates: Requirements 11.2, 11.3
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from vgb.config.loader import (
    ConfigLoadError,
    apply_overrides,
    build_full_config,
    load_config,
    merge_configs,
)
from vgb.config.schema import VGBConfigError


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_yaml(self, tmp_path: Path):
        cfg = {"train": {"learning_rate": 0.001, "batch_size": 16}}
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(cfg), encoding="utf-8")

        result = load_config(yaml_file)
        assert result == cfg

    def test_load_yml_extension(self, tmp_path: Path):
        cfg = {"runtime": {"strategy": "fsdp2"}}
        yml_file = tmp_path / "config.yml"
        yml_file.write_text(yaml.dump(cfg), encoding="utf-8")

        result = load_config(yml_file)
        assert result == cfg

    def test_load_json(self, tmp_path: Path):
        cfg = {"train": {"learning_rate": 0.01}}
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg), encoding="utf-8")

        result = load_config(json_file)
        assert result == cfg

    def test_file_not_found(self):
        with pytest.raises(ConfigLoadError, match="not found"):
            load_config("/nonexistent/path.yaml")

    def test_unsupported_extension(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[train]\nlr = 0.01", encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="Unsupported"):
            load_config(toml_file)

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")

        result = load_config(yaml_file)
        assert result == {}

    def test_invalid_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(":\n  - [invalid\n", encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="YAML parse error"):
            load_config(yaml_file)

    def test_invalid_json(self, tmp_path: Path):
        json_file = tmp_path / "bad.json"
        json_file.write_text("{not valid json}", encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="JSON parse error"):
            load_config(json_file)


# ---------------------------------------------------------------------------
# merge_configs tests
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"train": {"lr": 0.001, "epochs": 10}, "runtime": {"seed": 42}}
        override = {"train": {"lr": 0.01}, "model": {"d": 128}}

        result = merge_configs(base, override)
        assert result == {
            "train": {"lr": 0.01, "epochs": 10},
            "runtime": {"seed": 42},
            "model": {"d": 128},
        }

    def test_three_way_merge(self):
        profile = {"train": {"lr": 0.001, "batch_size": 8}}
        config = {"train": {"batch_size": 16}}
        cli = {"train": {"lr": 0.01}}

        result = merge_configs(profile, config, cli)
        assert result == {"train": {"lr": 0.01, "batch_size": 16}}

    def test_none_values_do_not_override(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"a": None, "b": {"c": None}}

        result = merge_configs(base, override)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_empty_merge(self):
        result = merge_configs({}, {})
        assert result == {}

    def test_single_config(self):
        cfg = {"a": 1}
        result = merge_configs(cfg)
        assert result == {"a": 1}

    def test_none_config_skipped(self):
        base = {"a": 1}
        result = merge_configs(base, None)
        assert result == {"a": 1}

    def test_override_replaces_non_dict_with_dict(self):
        base = {"a": 1}
        override = {"a": {"nested": True}}

        result = merge_configs(base, override)
        assert result == {"a": {"nested": True}}


# ---------------------------------------------------------------------------
# apply_overrides tests
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_simple_override(self):
        config = {"train": {"learning_rate": 0.001}}
        result = apply_overrides(config, ["train.learning_rate=0.01"])
        assert result["train"]["learning_rate"] == 0.01

    def test_scientific_notation(self):
        config = {"train": {}}
        result = apply_overrides(config, ["train.learning_rate=3e-4"])
        assert result["train"]["learning_rate"] == 3e-4

    def test_int_value(self):
        config = {"train": {}}
        result = apply_overrides(config, ["train.batch_size=32"])
        assert result["train"]["batch_size"] == 32
        assert isinstance(result["train"]["batch_size"], int)

    def test_bool_values(self):
        config = {}
        result = apply_overrides(config, ["flag=true", "other=false"])
        assert result["flag"] is True
        assert result["other"] is False

    def test_string_value(self):
        config = {}
        result = apply_overrides(config, ["runtime.strategy=fsdp2"])
        assert result["runtime"]["strategy"] == "fsdp2"

    def test_creates_nested_keys(self):
        config = {}
        result = apply_overrides(config, ["a.b.c=42"])
        assert result == {"a": {"b": {"c": 42}}}

    def test_multiple_overrides(self):
        config = {"train": {"lr": 0.001, "bs": 8}}
        result = apply_overrides(
            config, ["train.lr=0.01", "train.bs=16", "runtime.seed=123"]
        )
        assert result["train"]["lr"] == 0.01
        assert result["train"]["bs"] == 16
        assert result["runtime"]["seed"] == 123

    def test_malformed_override_raises(self):
        with pytest.raises(ConfigLoadError, match="Invalid override"):
            apply_overrides({}, ["no_equals_sign"])

    def test_empty_key_raises(self):
        with pytest.raises(ConfigLoadError, match="Empty key"):
            apply_overrides({}, ["=value"])

    def test_null_value(self):
        config = {"a": 1}
        result = apply_overrides(config, ["a=null"])
        assert result["a"] is None

    def test_original_not_modified(self):
        config = {"train": {"lr": 0.001}}
        _ = apply_overrides(config, ["train.lr=0.01"])
        assert config["train"]["lr"] == 0.001


# ---------------------------------------------------------------------------
# build_full_config tests
# ---------------------------------------------------------------------------


class TestBuildFullConfig:
    def _write_profile(self, profiles_dir: Path, name: str, data: dict):
        profiles_dir.mkdir(parents=True, exist_ok=True)
        path = profiles_dir / f"{name}.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")

    def _make_vsn_small_dict(self) -> dict:
        """Return a valid VSN small config as a dict."""
        return {
            "model": {
                "vsn": {
                    "X_enc": 4,
                    "X_dec": 4,
                    "Y": 4,
                    "Z": 4,
                    "d": 64,
                    "ics": 64,
                    "Y_H": 4,
                    "Z_H": 4,
                    "d_H": 64,
                    "p_mode": "identity",
                    "Y_dec": 4,
                    "Z_dec": 4,
                    "dgw": 4,
                    "head_type": "text",
                    "vocab_size": 32000,
                }
            },
            "train": {"learning_rate": 0.001, "batch_size": 8},
            "runtime": {"strategy": "single", "seed": 42},
        }

    def test_build_from_config_file(self, tmp_path: Path):
        cfg_data = self._make_vsn_small_dict()
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

        full = build_full_config(config_path=config_file)
        assert full.model.vsn.d == 64
        assert full.train.learning_rate == 0.001
        assert full.runtime.strategy == "single"

    def test_build_with_profile_and_overrides(self, tmp_path: Path):
        profiles_dir = tmp_path / "profiles"
        profile_data = self._make_vsn_small_dict()
        self._write_profile(profiles_dir, "vsn_small", profile_data)

        full = build_full_config(
            profile="vsn_small",
            overrides=["train.learning_rate=3e-4", "runtime.strategy=single"],
            profiles_dir=profiles_dir,
        )
        assert full.train.learning_rate == 3e-4
        assert full.runtime.strategy == "single"

    def test_profile_overridden_by_config(self, tmp_path: Path):
        profiles_dir = tmp_path / "profiles"
        profile_data = self._make_vsn_small_dict()
        profile_data["train"]["learning_rate"] = 0.01
        self._write_profile(profiles_dir, "vsn_small", profile_data)

        config_data = {"train": {"learning_rate": 0.005}}
        config_file = tmp_path / "override.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        full = build_full_config(
            config_path=config_file,
            profile="vsn_small",
            profiles_dir=profiles_dir,
        )
        # config overrides profile
        assert full.train.learning_rate == 0.005

    def test_cli_overrides_config(self, tmp_path: Path):
        cfg_data = self._make_vsn_small_dict()
        cfg_data["train"]["learning_rate"] = 0.01
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

        full = build_full_config(
            config_path=config_file,
            overrides=["train.learning_rate=0.001"],
        )
        # CLI overrides config file
        assert full.train.learning_rate == 0.001

    def test_invalid_config_raises_vgb_error(self, tmp_path: Path):
        # Invalid: strategy not in valid set
        cfg_data = self._make_vsn_small_dict()
        cfg_data["runtime"]["strategy"] = "invalid_strategy"
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

        with pytest.raises(VGBConfigError, match="runtime.strategy"):
            build_full_config(config_path=config_file)

    def test_profile_not_found_raises(self, tmp_path: Path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        with pytest.raises(ConfigLoadError, match="not found"):
            build_full_config(
                profile="nonexistent",
                profiles_dir=profiles_dir,
            )

    def test_full_precedence_chain(self, tmp_path: Path):
        """Profile < config < CLI overrides (full chain)."""
        profiles_dir = tmp_path / "profiles"
        profile_data = self._make_vsn_small_dict()
        profile_data["train"]["learning_rate"] = 0.1
        profile_data["train"]["batch_size"] = 4
        self._write_profile(profiles_dir, "base", profile_data)

        config_data = {"train": {"learning_rate": 0.01}}
        config_file = tmp_path / "specific.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        full = build_full_config(
            config_path=config_file,
            profile="base",
            overrides=["train.learning_rate=0.001"],
            profiles_dir=profiles_dir,
        )
        # CLI wins over config which wins over profile
        assert full.train.learning_rate == 0.001
        # config didn't set batch_size, so profile value persists
        assert full.train.batch_size == 4
