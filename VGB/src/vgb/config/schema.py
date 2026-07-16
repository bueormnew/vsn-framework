"""Esquema tipado de configuración para el framework VGB.

Define dataclasses tipadas para modelo, entrenamiento, inferencia y runtime,
con validación cruzada de campos. Envuelve VSNConfig del core y añade
parámetros operativos.

Validates: Requirements 11.2, 11.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vsn.core.config import VSNConfig, ConfigurationError


class VGBConfigError(Exception):
    """Error raised when VGB configuration validation fails.

    Provides descriptive messages indicating which fields are invalid
    and why, including cross-field incompatibilities.
    """

    pass


@dataclass
class ModelConfig:
    """Wraps VSNConfig with a source reference.

    Delegates all model-architecture configuration to the underlying
    VSNConfig instance, which defines dimensions, head type, etc.
    """

    vsn: VSNConfig


@dataclass
class TrainConfig:
    """Training configuration.

    Controls optimizer hyperparameters, gradient handling, precision,
    loss function, and checkpoint/eval intervals.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100_000
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    batch_size: int = 8
    eval_interval: int = 1000
    save_interval: int = 5000
    precision: str = "bf16"  # 'bf16', 'fp16', 'fp32'
    loss_type: str = "cross_entropy"  # 'cross_entropy', 'mse', 'l1'


@dataclass
class InferConfig:
    """Inference configuration.

    Controls batch size, precision, number of decoder windows,
    and target device for inference.
    """

    batch_size: int = 1
    precision: str = "bf16"  # 'bf16', 'fp16', 'fp32'
    num_windows: int = 1
    device: str = "cuda"


@dataclass
class RuntimeConfig:
    """Distributed runtime configuration.

    Controls parallelism strategy, data loading workers, RNG seed,
    and output directories for logs and checkpoints.
    """

    strategy: str = "single"  # 'single', 'fsdp2'
    num_workers: int = 4
    seed: int = 42
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class FullConfig:
    """Complete configuration combining model, training, inference, and runtime.

    Provides cross-validation between all sub-configs to catch
    incompatibilities early (fail-fast).
    """

    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        """Cross-validates all config components.

        Checks:
        - model.vsn validates (dimensional consistency)
        - precision values are valid ('bf16', 'fp16', 'fp32')
        - loss_type is compatible with head_type
        - runtime strategy is valid ('single', 'fsdp2')

        Raises:
            VGBConfigError: If any cross-validation check fails.
        """
        errors: list[str] = []

        # 1. Validate underlying VSNConfig
        try:
            self.model.vsn.validate()
        except ConfigurationError as e:
            errors.append(f"model.vsn: {e}")

        # 2. Validate precision fields
        valid_precisions = ("bf16", "fp16", "fp32")

        if self.train.precision not in valid_precisions:
            errors.append(
                f"train.precision must be one of {valid_precisions}, "
                f"got {self.train.precision!r}"
            )

        if self.infer.precision not in valid_precisions:
            errors.append(
                f"infer.precision must be one of {valid_precisions}, "
                f"got {self.infer.precision!r}"
            )

        # 3. Validate loss_type compatibility with head_type
        valid_loss_types = ("cross_entropy", "mse", "l1")
        if self.train.loss_type not in valid_loss_types:
            errors.append(
                f"train.loss_type must be one of {valid_loss_types}, "
                f"got {self.train.loss_type!r}"
            )
        else:
            head_type = self.model.vsn.head_type
            # cross_entropy requires discrete output heads (text, classification)
            if self.train.loss_type == "cross_entropy" and head_type not in (
                "text",
                "classification",
            ):
                errors.append(
                    f"train.loss_type='cross_entropy' is incompatible with "
                    f"head_type={head_type!r}; cross_entropy requires "
                    f"'text' or 'classification' head"
                )
            # mse/l1 are regression losses, incompatible with text head
            if self.train.loss_type in ("mse", "l1") and head_type == "text":
                errors.append(
                    f"train.loss_type={self.train.loss_type!r} is incompatible "
                    f"with head_type='text'; use 'cross_entropy' for text heads"
                )

        # 4. Validate runtime strategy
        valid_strategies = ("single", "fsdp2")
        if self.runtime.strategy not in valid_strategies:
            errors.append(
                f"runtime.strategy must be one of {valid_strategies}, "
                f"got {self.runtime.strategy!r}"
            )

        # 5. Validate numeric ranges
        if self.train.learning_rate <= 0:
            errors.append(
                f"train.learning_rate must be positive, "
                f"got {self.train.learning_rate}"
            )

        if self.train.batch_size < 1:
            errors.append(
                f"train.batch_size must be >= 1, got {self.train.batch_size}"
            )

        if self.train.grad_accum_steps < 1:
            errors.append(
                f"train.grad_accum_steps must be >= 1, "
                f"got {self.train.grad_accum_steps}"
            )

        if self.infer.batch_size < 1:
            errors.append(
                f"infer.batch_size must be >= 1, got {self.infer.batch_size}"
            )

        if self.infer.num_windows < 1:
            errors.append(
                f"infer.num_windows must be >= 1, got {self.infer.num_windows}"
            )

        if errors:
            raise VGBConfigError(
                "VGB configuration validation failed:\n  - "
                + "\n  - ".join(errors)
            )
