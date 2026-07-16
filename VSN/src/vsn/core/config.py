"""Configuración tipada para modelos VSN.

Define VSNConfig con validación dimensional y factory methods para
configuraciones predeterminadas (small, base, large).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class ConfigurationError(Exception):
    """Error raised when VSNConfig validation fails.

    Provides descriptive messages indicating which fields or component
    relationships are incompatible.
    """

    pass


@dataclass
class VSNConfig:
    """Configuración completa para un modelo VSN.

    Contiene todas las dimensiones del volumen, encoder, decoder,
    plano latente H, head de salida y metadata del modelo.
    """

    # Dimensiones del volumen
    X_enc: int  # Profundidad encoder (número de planos)
    X_dec: int  # Profundidad decoder
    Y: int  # Alto del plano
    Z: int  # Ancho del plano
    d: int  # Dimensión de embedding

    # Input Cache
    ics: int  # Tamaño del buffer FIFO

    # Plano latente H
    Y_H: int  # Alto de H
    Z_H: int  # Ancho de H
    d_H: int  # Dimensión de H
    p_mode: str  # 'compress', 'identity', 'expand'

    # Decoder
    Y_dec: int  # Alto decoder (puede diferir de encoder)
    Z_dec: int  # Ancho decoder
    dgw: int  # Tamaño de ventana DGW

    # Head
    head_type: str  # 'text', 'classification', 'regression', 'dense'
    vocab_size: Optional[int] = None  # Solo para TextHead
    num_classes: Optional[int] = None  # Solo para ClassificationHead

    # Metadata
    schema_version: str = "1.0"
    model_family: str = "vsn"
    vgb_version: str = "v1"
    psi_version: str = "v1"

    def validate(self) -> None:
        """Verifica compatibilidad entre componentes.

        Raises:
            ConfigurationError: Si alguna dimensión es inválida o hay
                incompatibilidad entre componentes.
        """
        errors: list[str] = []

        # Verificar que todas las dimensiones son positivas
        dim_fields = {
            "X_enc": self.X_enc,
            "X_dec": self.X_dec,
            "Y": self.Y,
            "Z": self.Z,
            "d": self.d,
            "Y_H": self.Y_H,
            "Z_H": self.Z_H,
            "d_H": self.d_H,
            "Y_dec": self.Y_dec,
            "Z_dec": self.Z_dec,
            "dgw": self.dgw,
        }

        for name, value in dim_fields.items():
            if not isinstance(value, int) or value <= 0:
                errors.append(
                    f"{name} must be a positive integer, got {value!r}"
                )

        # Verificar ics > 0
        if not isinstance(self.ics, int) or self.ics <= 0:
            errors.append(f"ics must be a positive integer, got {self.ics!r}")

        # Verificar p_mode válido
        valid_p_modes = ("compress", "identity", "expand")
        if self.p_mode not in valid_p_modes:
            errors.append(
                f"p_mode must be one of {valid_p_modes}, got {self.p_mode!r}"
            )

        # Verificar que p_mode coincide con la relación dimensional real
        if not errors:  # Solo si las dims son válidas
            enc_volume = self.Y * self.Z * self.d
            h_volume = self.Y_H * self.Z_H * self.d_H

            if self.p_mode == "compress" and h_volume >= enc_volume:
                errors.append(
                    f"p_mode='compress' requires Y_H*Z_H*d_H < Y*Z*d, "
                    f"but {h_volume} >= {enc_volume}"
                )
            elif self.p_mode == "identity" and h_volume != enc_volume:
                errors.append(
                    f"p_mode='identity' requires Y_H*Z_H*d_H == Y*Z*d, "
                    f"but {h_volume} != {enc_volume}"
                )
            elif self.p_mode == "expand" and h_volume <= enc_volume:
                errors.append(
                    f"p_mode='expand' requires Y_H*Z_H*d_H > Y*Z*d, "
                    f"but {h_volume} <= {enc_volume}"
                )

        # Verificar head_type válido
        valid_head_types = ("text", "classification", "regression", "dense")
        if self.head_type not in valid_head_types:
            errors.append(
                f"head_type must be one of {valid_head_types}, "
                f"got {self.head_type!r}"
            )

        # Verificar vocab_size cuando head_type='text'
        if self.head_type == "text":
            if self.vocab_size is None or self.vocab_size <= 0:
                errors.append(
                    "vocab_size must be a positive integer when "
                    f"head_type='text', got {self.vocab_size!r}"
                )

        # Verificar num_classes cuando head_type='classification'
        if self.head_type == "classification":
            if self.num_classes is None or self.num_classes <= 0:
                errors.append(
                    "num_classes must be a positive integer when "
                    f"head_type='classification', got {self.num_classes!r}"
                )

        if errors:
            raise ConfigurationError(
                "VSNConfig validation failed:\n  - " + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------
    # Factory methods para configuraciones predeterminadas
    # ------------------------------------------------------------------

    @classmethod
    def small(
        cls,
        head_type: str = "text",
        vocab_size: Optional[int] = 32000,
        num_classes: Optional[int] = None,
    ) -> VSNConfig:
        """Configuración small: X=4, Y=4, Z=4, d=64, ics=64, dgw=4."""
        return cls(
            X_enc=4,
            X_dec=4,
            Y=4,
            Z=4,
            d=64,
            ics=64,
            Y_H=4,
            Z_H=4,
            d_H=64,
            p_mode="identity",
            Y_dec=4,
            Z_dec=4,
            dgw=4,
            head_type=head_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
        )

    @classmethod
    def base(
        cls,
        head_type: str = "text",
        vocab_size: Optional[int] = 32000,
        num_classes: Optional[int] = None,
    ) -> VSNConfig:
        """Configuración base: X=8, Y=8, Z=8, d=128, ics=256, dgw=8."""
        return cls(
            X_enc=8,
            X_dec=8,
            Y=8,
            Z=8,
            d=128,
            ics=256,
            Y_H=8,
            Z_H=8,
            d_H=128,
            p_mode="identity",
            Y_dec=8,
            Z_dec=8,
            dgw=8,
            head_type=head_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
        )

    @classmethod
    def large(
        cls,
        head_type: str = "text",
        vocab_size: Optional[int] = 32000,
        num_classes: Optional[int] = None,
    ) -> VSNConfig:
        """Configuración large: X=16, Y=16, Z=16, d=256, ics=1024, dgw=16."""
        return cls(
            X_enc=16,
            X_dec=16,
            Y=16,
            Z=16,
            d=256,
            ics=1024,
            Y_H=16,
            Z_H=16,
            d_H=256,
            p_mode="identity",
            Y_dec=16,
            Z_dec=16,
            dgw=16,
            head_type=head_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
        )
