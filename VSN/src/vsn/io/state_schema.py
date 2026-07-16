"""Esquema de estado para serialización de modelos VSN.

Define StateSchema con todos los campos de versionado,
serialización JSON y validación de schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Required dimension keys in the dims dict
_REQUIRED_DIM_KEYS = frozenset(
    {"X_enc", "X_dec", "Y", "Z", "d", "Y_H", "Z_H", "d_H", "Y_dec", "Z_dec"}
)

# Supported schema versions
_SUPPORTED_SCHEMA_VERSIONS = ("1.0",)


class StateSchemaError(Exception):
    """Error raised when StateSchema validation fails."""

    pass


@dataclass
class StateSchema:
    """Esquema de estado para serialización de modelos VSN.

    Contiene metadata de versionado, dimensiones del modelo y
    checksum para verificar integridad.
    """

    schema_version: str
    model_family: str
    vgb_version: str
    psi_version: str
    head_type: str
    dims: Dict[str, int]
    ics: int
    dgw: int
    total_params: int
    created_at: str
    checksum: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el schema a un diccionario serializable a JSON."""
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "vgb_version": self.vgb_version,
            "psi_version": self.psi_version,
            "head_type": self.head_type,
            "dims": dict(self.dims),
            "ics": self.ics,
            "dgw": self.dgw,
            "total_params": self.total_params,
            "created_at": self.created_at,
            "checksum": self.checksum,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializa el schema a una cadena JSON.

        Args:
            indent: Niveles de indentación para pretty-printing.

        Returns:
            Cadena JSON representando el schema.
        """
        return json.dumps(self.to_dict(), indent=indent)

    # ------------------------------------------------------------------
    # Deserialización (classmethods)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateSchema:
        """Construye un StateSchema desde un diccionario.

        Args:
            data: Diccionario con los campos del schema.

        Returns:
            Instancia de StateSchema.

        Raises:
            StateSchemaError: Si faltan campos requeridos.
        """
        required_keys = {
            "schema_version",
            "model_family",
            "vgb_version",
            "psi_version",
            "head_type",
            "dims",
            "ics",
            "dgw",
            "total_params",
            "created_at",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise StateSchemaError(
                f"Missing required fields in schema data: {sorted(missing)}"
            )

        return cls(
            schema_version=data["schema_version"],
            model_family=data["model_family"],
            vgb_version=data["vgb_version"],
            psi_version=data["psi_version"],
            head_type=data["head_type"],
            dims=dict(data["dims"]),
            ics=data["ics"],
            dgw=data["dgw"],
            total_params=data["total_params"],
            created_at=data["created_at"],
            checksum=data.get("checksum"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> StateSchema:
        """Construye un StateSchema desde una cadena JSON.

        Args:
            json_str: Cadena JSON con los campos del schema.

        Returns:
            Instancia de StateSchema.

        Raises:
            StateSchemaError: Si el JSON es inválido o faltan campos.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise StateSchemaError(f"Invalid JSON: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_config(cls, config: Any, total_params: int) -> StateSchema:
        """Construye un StateSchema desde un VSNConfig y el conteo de parámetros.

        Args:
            config: Instancia de VSNConfig con la configuración del modelo.
            total_params: Número total de parámetros del modelo.

        Returns:
            Instancia de StateSchema con los valores extraídos del config.
        """
        dims = {
            "X_enc": config.X_enc,
            "X_dec": config.X_dec,
            "Y": config.Y,
            "Z": config.Z,
            "d": config.d,
            "Y_H": config.Y_H,
            "Z_H": config.Z_H,
            "d_H": config.d_H,
            "Y_dec": config.Y_dec,
            "Z_dec": config.Z_dec,
        }

        return cls(
            schema_version=config.schema_version,
            model_family=config.model_family,
            vgb_version=config.vgb_version,
            psi_version=config.psi_version,
            head_type=config.head_type,
            dims=dims,
            ics=config.ics,
            dgw=config.dgw,
            total_params=total_params,
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=None,
        )

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Valida la consistencia del schema.

        Verifica:
        - schema_version soportada
        - Campos requeridos presentes y con tipos correctos
        - Todas las dimensiones requeridas presentes en dims
        - Todas las dimensiones son enteros positivos
        - ics y dgw son enteros positivos
        - total_params es un entero no-negativo

        Raises:
            StateSchemaError: Si alguna validación falla, con detalle
                de todos los errores encontrados.
        """
        errors: list[str] = []

        # Validar schema_version
        if self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"Unsupported schema_version '{self.schema_version}', "
                f"supported: {_SUPPORTED_SCHEMA_VERSIONS}"
            )

        # Validar campos string no vacíos
        string_fields = {
            "model_family": self.model_family,
            "vgb_version": self.vgb_version,
            "psi_version": self.psi_version,
            "head_type": self.head_type,
            "created_at": self.created_at,
        }
        for name, value in string_fields.items():
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string, got {value!r}")

        # Validar dims keys
        if not isinstance(self.dims, dict):
            errors.append(f"dims must be a dict, got {type(self.dims).__name__}")
        else:
            missing_dims = _REQUIRED_DIM_KEYS - set(self.dims.keys())
            if missing_dims:
                errors.append(
                    f"dims missing required keys: {sorted(missing_dims)}"
                )

            # Validar que todas las dimensiones son enteros positivos
            for key, value in self.dims.items():
                if not isinstance(value, int) or value <= 0:
                    errors.append(
                        f"dims['{key}'] must be a positive integer, got {value!r}"
                    )

        # Validar ics
        if not isinstance(self.ics, int) or self.ics <= 0:
            errors.append(f"ics must be a positive integer, got {self.ics!r}")

        # Validar dgw
        if not isinstance(self.dgw, int) or self.dgw <= 0:
            errors.append(f"dgw must be a positive integer, got {self.dgw!r}")

        # Validar total_params
        if not isinstance(self.total_params, int) or self.total_params < 0:
            errors.append(
                f"total_params must be a non-negative integer, got {self.total_params!r}"
            )

        if errors:
            raise StateSchemaError(
                "StateSchema validation failed:\n  - " + "\n  - ".join(errors)
            )
