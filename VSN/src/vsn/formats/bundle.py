"""Formato de bundle de inferencia para modelos VSN.

Provee exportación e importación de bundles de inferencia como directorios:
    bundle_dir/
    ├── manifest.json       # BundleManifest serializado
    ├── model_config.json   # VSNConfig como JSON
    ├── weights.pt          # model state_dict (PyTorch format)
    └── schema.json         # StateSchema serializado

Funciones principales:
    - export_bundle: exporta modelo a bundle con pesos + config + checksums
    - load_bundle: carga bundle con validación de integridad (SHA-256)

Soporte opcional para safetensors como formato alternativo de pesos.

Validates: Requirements 9.3, 9.4
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.io.state_schema import StateSchema

# Versiones de schema soportadas para carga de bundles
_SUPPORTED_SCHEMA_VERSIONS = ("1.0",)

# Nombre de archivos estándar dentro del bundle
_MANIFEST_FILE = "manifest.json"
_CONFIG_FILE = "model_config.json"
_WEIGHTS_PT_FILE = "weights.pt"
_WEIGHTS_ST_FILE = "weights.safetensors"
_SCHEMA_FILE = "schema.json"


class BundleIntegrityError(Exception):
    """Error cuando la validación de integridad de un bundle falla.

    Se lanza cuando:
        - Un checksum SHA-256 no coincide con el archivo
        - El schema_version no es soportado
        - Faltan archivos requeridos en el bundle
        - El manifest está corrupto o incompleto
    """

    pass


@dataclass
class BundleManifest:
    """Manifiesto de un bundle de inferencia.

    Contiene toda la metadata necesaria para validar y cargar un bundle:
    schema del modelo, formato de pesos, archivos incluidos y checksums.

    Attributes:
        schema: StateSchema con metadata del modelo.
        weight_format: Formato de pesos ('pytorch' o 'safetensors').
        weight_files: Lista de archivos de pesos incluidos.
        config_file: Nombre del archivo de configuración.
        metadata: Metadata adicional (versión de exportación, etc.).
        checksums: Mapa filename → SHA-256 hex digest.
    """

    schema: StateSchema
    weight_format: str  # 'pytorch' | 'safetensors'
    weight_files: List[str]
    config_file: str
    metadata: Dict[str, Any]
    checksums: Dict[str, str]  # filename → sha256

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el manifest a un diccionario JSON-compatible."""
        return {
            "schema": self.schema.to_dict(),
            "weight_format": self.weight_format,
            "weight_files": list(self.weight_files),
            "config_file": self.config_file,
            "metadata": dict(self.metadata),
            "checksums": dict(self.checksums),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializa el manifest a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BundleManifest:
        """Construye un BundleManifest desde un diccionario.

        Args:
            data: Diccionario con los campos del manifest.

        Returns:
            Instancia de BundleManifest.

        Raises:
            BundleIntegrityError: Si faltan campos requeridos.
        """
        required_keys = {
            "schema",
            "weight_format",
            "weight_files",
            "config_file",
            "metadata",
            "checksums",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise BundleIntegrityError(
                f"Manifest missing required fields: {sorted(missing)}"
            )

        schema = StateSchema.from_dict(data["schema"])

        return cls(
            schema=schema,
            weight_format=data["weight_format"],
            weight_files=list(data["weight_files"]),
            config_file=data["config_file"],
            metadata=dict(data.get("metadata", {})),
            checksums=dict(data.get("checksums", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> BundleManifest:
        """Construye un BundleManifest desde una cadena JSON.

        Args:
            json_str: Cadena JSON con los campos del manifest.

        Returns:
            Instancia de BundleManifest.

        Raises:
            BundleIntegrityError: Si el JSON es inválido o faltan campos.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise BundleIntegrityError(f"Invalid manifest JSON: {e}") from e

        return cls.from_dict(data)


def _compute_sha256(file_path: Path) -> str:
    """Computa el SHA-256 hex digest de un archivo.

    Args:
        file_path: Ruta al archivo.

    Returns:
        SHA-256 hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _has_safetensors() -> bool:
    """Verifica si la librería safetensors está disponible."""
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


def export_bundle(
    model: VSNModel,
    output_dir: str | Path,
    weight_format: str = "pytorch",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Exporta un modelo VSN como bundle de inferencia.

    Crea un directorio con todos los archivos necesarios para cargar
    el modelo en inferencia: pesos, configuración, schema y manifest
    con checksums SHA-256.

    Args:
        model: Instancia de VSNModel a exportar.
        output_dir: Directorio destino del bundle.
        weight_format: Formato de pesos — 'pytorch' o 'safetensors'.
        metadata: Metadata adicional opcional para el manifest.

    Returns:
        Path al directorio del bundle creado.

    Raises:
        BundleIntegrityError: Si el weight_format no es soportado o
            safetensors no está instalado cuando se solicita.
        ValueError: Si el modelo no tiene config válido.
    """
    output_dir = Path(output_dir)

    # Validar weight_format
    valid_formats = ("pytorch", "safetensors")
    if weight_format not in valid_formats:
        raise BundleIntegrityError(
            f"Unsupported weight_format '{weight_format}'. "
            f"Must be one of: {valid_formats}"
        )

    if weight_format == "safetensors" and not _has_safetensors():
        raise BundleIntegrityError(
            "safetensors package not installed. "
            "Install with: pip install safetensors"
        )

    # 1. Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Guardar pesos
    state_dict = model.state_dict()
    weight_files: List[str] = []

    if weight_format == "pytorch":
        weights_path = output_dir / _WEIGHTS_PT_FILE
        torch.save(state_dict, weights_path)
        weight_files.append(_WEIGHTS_PT_FILE)
    else:
        # safetensors format
        from safetensors.torch import save_file

        weights_path = output_dir / _WEIGHTS_ST_FILE
        save_file(state_dict, str(weights_path))
        weight_files.append(_WEIGHTS_ST_FILE)

    # 3. Guardar VSNConfig como JSON
    config_path = output_dir / _CONFIG_FILE
    config_dict = dataclasses.asdict(model.config)
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # 4. Guardar StateSchema como JSON
    total_params = sum(p.numel() for p in model.parameters())
    schema = StateSchema.from_config(model.config, total_params)
    schema_path = output_dir / _SCHEMA_FILE
    schema_path.write_text(schema.to_json(), encoding="utf-8")

    # 5. Computar checksums SHA-256
    checksums: Dict[str, str] = {}
    for wf in weight_files:
        checksums[wf] = _compute_sha256(output_dir / wf)
    checksums[_CONFIG_FILE] = _compute_sha256(config_path)
    checksums[_SCHEMA_FILE] = _compute_sha256(schema_path)

    # 6. Crear BundleManifest
    bundle_metadata = metadata or {}
    bundle_metadata.setdefault("export_version", "1.0")

    manifest = BundleManifest(
        schema=schema,
        weight_format=weight_format,
        weight_files=weight_files,
        config_file=_CONFIG_FILE,
        metadata=bundle_metadata,
        checksums=checksums,
    )

    # 7. Guardar manifest.json
    manifest_path = output_dir / _MANIFEST_FILE
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")

    return output_dir


def load_bundle(
    bundle_dir: str | Path,
    device: str = "cpu",
    head: Optional[nn.Module] = None,
) -> VSNModel:
    """Carga un modelo VSN desde un bundle de inferencia.

    Realiza validación completa de integridad antes de instanciar el modelo:
    1. Verifica que el manifest existe y es válido
    2. Valida schema_version soportada
    3. Verifica checksums SHA-256 de todos los archivos
    4. Reconstruye el modelo desde la configuración y pesos

    Args:
        bundle_dir: Directorio del bundle a cargar.
        device: Dispositivo destino ('cpu', 'cuda', etc.).
        head: Módulo head opcional para el modelo reconstruido.

    Returns:
        VSNModel reconstruido con pesos cargados.

    Raises:
        BundleIntegrityError: Si el bundle no es válido, los checksums
            no coinciden, o el schema_version no es soportado.
    """
    bundle_dir = Path(bundle_dir)

    # 1. Verificar que el directorio existe
    if not bundle_dir.is_dir():
        raise BundleIntegrityError(
            f"Bundle directory not found: '{bundle_dir}'"
        )

    # 2. Cargar manifest.json
    manifest_path = bundle_dir / _MANIFEST_FILE
    if not manifest_path.exists():
        raise BundleIntegrityError(
            f"Manifest file not found: '{manifest_path}'"
        )

    try:
        manifest_json = manifest_path.read_text(encoding="utf-8")
        manifest = BundleManifest.from_json(manifest_json)
    except BundleIntegrityError:
        raise
    except Exception as e:
        raise BundleIntegrityError(
            f"Failed to parse manifest: {e}"
        ) from e

    # 3. Validar schema_version
    schema_version = manifest.schema.schema_version
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise BundleIntegrityError(
            f"Unsupported schema_version '{schema_version}'. "
            f"Supported versions: {_SUPPORTED_SCHEMA_VERSIONS}"
        )

    # 4. Validar checksums de todos los archivos registrados
    for filename, expected_hash in manifest.checksums.items():
        file_path = bundle_dir / filename
        if not file_path.exists():
            raise BundleIntegrityError(
                f"Bundle file missing: '{filename}'"
            )
        actual_hash = _compute_sha256(file_path)
        if actual_hash != expected_hash:
            raise BundleIntegrityError(
                f"Checksum mismatch for '{filename}': "
                f"expected {expected_hash}, got {actual_hash}"
            )

    # 5. Cargar VSNConfig desde model_config.json
    config_path = bundle_dir / manifest.config_file
    if not config_path.exists():
        raise BundleIntegrityError(
            f"Config file not found: '{manifest.config_file}'"
        )

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config = VSNConfig(**config_data)
    except (json.JSONDecodeError, TypeError) as e:
        raise BundleIntegrityError(
            f"Failed to load config: {e}"
        ) from e

    # 6. Crear VSNModel desde config
    model = VSNModel(config, head=head)

    # 7. Cargar pesos
    if not manifest.weight_files:
        raise BundleIntegrityError("No weight files specified in manifest")

    weight_file = manifest.weight_files[0]
    weight_path = bundle_dir / weight_file

    if not weight_path.exists():
        raise BundleIntegrityError(
            f"Weight file not found: '{weight_file}'"
        )

    if manifest.weight_format == "pytorch":
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    elif manifest.weight_format == "safetensors":
        if not _has_safetensors():
            raise BundleIntegrityError(
                "safetensors package required to load this bundle. "
                "Install with: pip install safetensors"
            )
        from safetensors.torch import load_file

        state_dict = load_file(str(weight_path), device=device)
    else:
        raise BundleIntegrityError(
            f"Unknown weight_format: '{manifest.weight_format}'"
        )

    # Cargar state_dict — strict=False si se inyecta head distinto
    strict = head is None
    model.load_state_dict(state_dict, strict=strict)

    # Mover al dispositivo destino
    model = model.to(device)

    return model
