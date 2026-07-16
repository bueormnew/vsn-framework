"""Cargador de configuración jerárquico para el framework VGB.

Soporta carga desde archivos YAML/JSON, merge jerárquico con precedencia
(CLI overrides > specific config > base profile), y validación contra
el schema tipado FullConfig.

Validates: Requirements 11.2, 11.3
"""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from vsn.core.config import VSNConfig
from vgb.config.schema import (
    FullConfig,
    ModelConfig,
    TrainConfig,
    InferConfig,
    RuntimeConfig,
    VGBConfigError,
)


class ConfigLoadError(Exception):
    """Error raised when a configuration file cannot be loaded.

    Indicates file not found, parse error, or unsupported format.
    """

    pass


def load_config(path: str | Path) -> dict:
    """Load a configuration file (YAML or JSON) and return as dict.

    Determines format from file extension:
    - .yaml, .yml → YAML
    - .json → JSON

    Args:
        path: Path to configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigLoadError: If file not found, unsupported extension, or parse error.
    """
    path = Path(path)

    if not path.exists():
        raise ConfigLoadError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(
                f"YAML parse error in {path}: {e}"
            ) from e
    elif suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigLoadError(
                f"JSON parse error in {path}: {e}"
            ) from e
    else:
        raise ConfigLoadError(
            f"Unsupported config format '{suffix}' for {path}. "
            f"Supported: .yaml, .yml, .json"
        )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigLoadError(
            f"Configuration file {path} must contain a mapping (dict) "
            f"at the top level, got {type(data).__name__}"
        )

    return data


def merge_configs(*configs: dict) -> dict:
    """Deep-merge multiple config dicts. Later dicts override earlier ones.

    Merge rules:
    - Dicts are merged recursively (keys are combined).
    - Non-dict values are overwritten by the later config.
    - None values in later configs do NOT override existing values.

    Args:
        *configs: Variable number of config dicts, in order of increasing
                  precedence.

    Returns:
        New dict with merged configuration.
    """
    result: Dict[str, Any] = {}

    for config in configs:
        if config is None:
            continue
        result = _deep_merge(result, config)

    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if value is None:
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def apply_overrides(config: dict, overrides: List[str]) -> dict:
    """Apply CLI overrides in dotted.key.path=value format.

    Each override string must be in the format "dotted.key.path=value".
    Values are parsed as:
    - int if all digits (or negative)
    - float if numeric with decimal point or scientific notation
    - bool if 'true'/'false' (case-insensitive)
    - None if 'null'/'none' (case-insensitive)
    - str otherwise

    Args:
        config: Base configuration dictionary.
        overrides: List of override strings, e.g. ["train.learning_rate=3e-4"].

    Returns:
        New dict with overrides applied.

    Raises:
        ConfigLoadError: If an override string is malformed.
    """
    result = copy.deepcopy(config)

    for override in overrides:
        if "=" not in override:
            raise ConfigLoadError(
                f"Invalid override format: {override!r}. "
                f"Expected 'dotted.key.path=value'."
            )

        key_path, value_str = override.split("=", 1)
        key_path = key_path.strip()
        value_str = value_str.strip()

        if not key_path:
            raise ConfigLoadError(
                f"Empty key in override: {override!r}."
            )

        keys = key_path.split(".")
        parsed_value = _parse_value(value_str)

        # Navigate to the correct nested dict and set the value
        current = result
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = parsed_value

    return result


def _parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    # Bool
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # None/null
    if value_str.lower() in ("null", "none"):
        return None

    # Int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # String (strip quotes if present)
    if (
        (value_str.startswith('"') and value_str.endswith('"'))
        or (value_str.startswith("'") and value_str.endswith("'"))
    ):
        return value_str[1:-1]

    return value_str


def build_full_config(
    config_path: Optional[Path] = None,
    profile: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    profiles_dir: Optional[Path] = None,
) -> FullConfig:
    """Build and validate a complete FullConfig from sources.

    Merge order (precedence): CLI overrides > specific config > base profile.

    Steps:
    1. Load base profile from profiles_dir/{profile}.yaml if provided.
    2. Load specific config file from config_path if provided.
    3. Deep-merge: profile → config (config wins).
    4. Apply CLI overrides on top.
    5. Construct FullConfig from the merged dict.
    6. Call validate() on the result.
    7. Return the validated FullConfig.

    Args:
        config_path: Path to a specific YAML/JSON config file.
        profile: Name of a base profile (e.g. 'vsn_small', 'vsn_base').
        overrides: List of CLI override strings.
        profiles_dir: Directory containing profile YAML files.

    Returns:
        Validated FullConfig instance.

    Raises:
        ConfigLoadError: If any file cannot be loaded or parsed.
        VGBConfigError: If the final merged config fails validation.
    """
    if overrides is None:
        overrides = []

    # 1. Load base profile
    profile_config: dict = {}
    if profile is not None:
        if profiles_dir is None:
            profiles_dir = Path(__file__).parent / "profiles"

        # Try .yaml first, then .yml, then .json
        profile_path = None
        for ext in (".yaml", ".yml", ".json"):
            candidate = profiles_dir / f"{profile}{ext}"
            if candidate.exists():
                profile_path = candidate
                break

        if profile_path is None:
            available = [
                f.stem
                for f in profiles_dir.iterdir()
                if f.suffix in (".yaml", ".yml", ".json")
            ] if profiles_dir.exists() else []
            raise ConfigLoadError(
                f"Profile '{profile}' not found in {profiles_dir}. "
                f"Available profiles: {available}"
            )

        profile_config = load_config(profile_path)

    # 2. Load specific config file
    specific_config: dict = {}
    if config_path is not None:
        specific_config = load_config(config_path)

    # 3. Deep-merge: profile → specific config
    merged = merge_configs(profile_config, specific_config)

    # 4. Apply CLI overrides
    if overrides:
        merged = apply_overrides(merged, overrides)

    # 5. Construct FullConfig from merged dict
    full_config = _dict_to_full_config(merged)

    # 6. Validate
    full_config.validate()

    # 7. Return
    return full_config


def _dict_to_full_config(data: dict) -> FullConfig:
    """Convert a merged dict to a FullConfig instance.

    Expects top-level keys: 'model', 'train', 'infer', 'runtime'.
    The 'model' section must contain a 'vsn' sub-dict with VSNConfig fields.

    Raises:
        ConfigLoadError: If required fields are missing or type conversion fails.
    """
    try:
        # Model config (required)
        model_data = data.get("model", {})
        vsn_data = model_data.get("vsn", model_data)

        vsn_config = VSNConfig(
            X_enc=int(vsn_data.get("X_enc", 4)),
            X_dec=int(vsn_data.get("X_dec", 4)),
            Y=int(vsn_data.get("Y", 4)),
            Z=int(vsn_data.get("Z", 4)),
            d=int(vsn_data.get("d", 64)),
            ics=int(vsn_data.get("ics", 64)),
            Y_H=int(vsn_data.get("Y_H", 4)),
            Z_H=int(vsn_data.get("Z_H", 4)),
            d_H=int(vsn_data.get("d_H", 64)),
            p_mode=str(vsn_data.get("p_mode", "identity")),
            Y_dec=int(vsn_data.get("Y_dec", 4)),
            Z_dec=int(vsn_data.get("Z_dec", 4)),
            dgw=int(vsn_data.get("dgw", 4)),
            head_type=str(vsn_data.get("head_type", "text")),
            vocab_size=_opt_int(vsn_data.get("vocab_size", 32000)),
            num_classes=_opt_int(vsn_data.get("num_classes")),
        )

        model_config = ModelConfig(vsn=vsn_config)

        # Train config (optional, has defaults)
        train_data = data.get("train", {})
        train_config = TrainConfig(
            **{k: v for k, v in train_data.items() if v is not None}
        )

        # Infer config (optional, has defaults)
        infer_data = data.get("infer", {})
        infer_config = InferConfig(
            **{k: v for k, v in infer_data.items() if v is not None}
        )

        # Runtime config (optional, has defaults)
        runtime_data = data.get("runtime", {})
        runtime_config = RuntimeConfig(
            **{k: v for k, v in runtime_data.items() if v is not None}
        )

        return FullConfig(
            model=model_config,
            train=train_config,
            infer=infer_config,
            runtime=runtime_config,
        )

    except (TypeError, ValueError, KeyError) as e:
        raise ConfigLoadError(
            f"Failed to construct FullConfig from merged dict: {e}"
        ) from e


def _opt_int(value: Any) -> Optional[int]:
    """Convert a value to Optional[int]."""
    if value is None:
        return None
    return int(value)
