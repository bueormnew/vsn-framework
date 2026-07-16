"""Save/Load de modelos VSN — persistencia del core.

Provee funciones para guardar y cargar modelos VSN completos:
- save_model: guarda state_dict + config + schema en un .pt
- load_model: carga y reconstruye un VSNModel desde un .pt

El formato del checkpoint es un diccionario con:
    {
        "state_dict": model.state_dict(),
        "config": dataclasses.asdict(model.config),
        "schema": schema.to_dict(),
    }

Valida schema_version al cargar para detectar incompatibilidades.

Validates: Requirements 9.3, 9.4
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.io.state_schema import StateSchema, StateSchemaError

# Versiones de schema soportadas para carga
_SUPPORTED_SCHEMA_VERSIONS = ("1.0",)


class SaveLoadError(Exception):
    """Error durante operaciones de save/load de modelos."""

    pass


def save_model(model: VSNModel, path: str | Path) -> None:
    """Guarda un modelo VSN completo en un archivo .pt.

    El checkpoint contiene:
        - state_dict: pesos del modelo
        - config: configuración completa como dict
        - schema: StateSchema serializado con metadata de versionado

    Args:
        model: Instancia de VSNModel a guardar.
        path: Ruta del archivo destino (.pt).

    Raises:
        SaveLoadError: Si ocurre un error durante el guardado.
    """
    path = Path(path)

    # Crear directorio padre si no existe
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Calcular total de parámetros
        total_params = sum(p.numel() for p in model.parameters())

        # Construir schema desde config
        schema = StateSchema.from_config(model.config, total_params)

        # Construir checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(model.config),
            "schema": schema.to_dict(),
        }

        torch.save(checkpoint, path)

    except Exception as e:
        raise SaveLoadError(f"Failed to save model to '{path}': {e}") from e


def load_model(
    path: str | Path,
    device: str = "cpu",
    head: Optional[nn.Module] = None,
) -> VSNModel:
    """Carga y reconstruye un modelo VSN completo desde un archivo .pt.

    Valida que el schema_version del checkpoint sea soportado antes de
    reconstruir el modelo.

    Args:
        path: Ruta al archivo .pt del checkpoint.
        device: Dispositivo destino para el modelo ('cpu', 'cuda', etc.).
        head: Módulo head opcional para el modelo reconstruido.

    Returns:
        VSNModel reconstruido con pesos cargados y en el dispositivo indicado.

    Raises:
        SaveLoadError: Si el archivo no existe, el schema_version no es
            soportado, o el checkpoint está corrupto.
    """
    path = Path(path)

    if not path.exists():
        raise SaveLoadError(f"Checkpoint file not found: '{path}'")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise SaveLoadError(
            f"Failed to load checkpoint from '{path}': {e}"
        ) from e

    # Validar estructura del checkpoint
    required_keys = {"state_dict", "config", "schema"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise SaveLoadError(
            f"Corrupted checkpoint — missing keys: {sorted(missing)}"
        )

    # Validar schema_version
    schema_data = checkpoint["schema"]
    schema_version = schema_data.get("schema_version", "unknown")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise SaveLoadError(
            f"Unsupported schema_version '{schema_version}'. "
            f"Supported versions: {_SUPPORTED_SCHEMA_VERSIONS}"
        )

    # Validar schema completo
    try:
        schema = StateSchema.from_dict(schema_data)
        schema.validate()
    except StateSchemaError as e:
        raise SaveLoadError(
            f"Schema validation failed during load: {e}"
        ) from e

    # Reconstruir VSNConfig desde el dict guardado
    config_data = checkpoint["config"]
    try:
        config = VSNConfig(**config_data)
    except TypeError as e:
        raise SaveLoadError(
            f"Failed to reconstruct VSNConfig: {e}"
        ) from e

    # Crear modelo con la config reconstruida
    model = VSNModel(config, head=head)

    # Cargar state_dict — usar strict=False si se inyecta un head distinto
    # al guardado, ya que el state_dict no contendrá los pesos del nuevo head
    # (o viceversa, el head original tenía pesos que el nuevo no tiene).
    strict = head is None
    model.load_state_dict(checkpoint["state_dict"], strict=strict)

    # Mover al dispositivo destino
    model = model.to(device)

    return model
