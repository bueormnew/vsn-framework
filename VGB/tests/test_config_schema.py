"""Tests for vgb.config.schema — typed config dataclasses and validation.

Validates: Requirements 11.2, 11.3
"""

import pytest
from vsn.core.config import VSNConfig

from vgb.config.schema import (
    FullConfig,
    InferConfig,
    ModelConfig,
    RuntimeConfig,
    TrainConfig,
    VGBConfigError,
)


def _valid_vsn_config(**overrides) -> VSNConfig:
    """Helper to create a valid VSNConfig with sensible defaults."""
    defaults = dict(
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
        head_type="text",
        vocab_size=32000,
    )
    defaults.update(overrides)
    return VSNConfig(**defaults)


class TestModelConfig:
    def test_wraps_vsn_config(self):
        vsn = _valid_vsn_config()
        model = ModelConfig(vsn=vsn)
        assert model.vsn is vsn
        assert model.vsn.d == 64

    def test_access_nested_fields(self):
        vsn = _valid_vsn_config(head_type="classification", num_classes=10)
        model = ModelConfig(vsn=vsn)
        assert model.vsn.head_type == "classification"
        assert model.vsn.num_classes == 10


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.max_steps == 100_000
        assert cfg.warmup_steps == 1000
        assert cfg.grad_clip_norm == 1.0
        assert cfg.grad_accum_steps == 1
        assert cfg.batch_size == 8
        assert cfg.eval_interval == 1000
        assert cfg.save_interval == 5000
        assert cfg.precision == "bf16"
        assert cfg.loss_type == "cross_entropy"

    def test_custom_values(self):
        cfg = TrainConfig(learning_rate=3e-4, batch_size=32, precision="fp32")
        assert cfg.learning_rate == 3e-4
        assert cfg.batch_size == 32
        assert cfg.precision == "fp32"


class TestInferConfig:
    def test_defaults(self):
        cfg = InferConfig()
        assert cfg.batch_size == 1
        assert cfg.precision == "bf16"
        assert cfg.num_windows == 1
        assert cfg.device == "cuda"

    def test_custom_values(self):
        cfg = InferConfig(batch_size=4, device="cpu", num_windows=3)
        assert cfg.batch_size == 4
        assert cfg.device == "cpu"
        assert cfg.num_windows == 3


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.strategy == "single"
        assert cfg.num_workers == 4
        assert cfg.seed == 42
        assert cfg.log_dir == "logs"
        assert cfg.checkpoint_dir == "checkpoints"

    def test_fsdp2_strategy(self):
        cfg = RuntimeConfig(strategy="fsdp2")
        assert cfg.strategy == "fsdp2"


class TestFullConfig:
    def test_valid_config_passes_validation(self):
        vsn = _valid_vsn_config()
        full = FullConfig(model=ModelConfig(vsn=vsn))
        # Should not raise
        full.validate()

    def test_invalid_precision_train(self):
        vsn = _valid_vsn_config()
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(precision="tf32"),
        )
        with pytest.raises(VGBConfigError, match="train.precision"):
            full.validate()

    def test_invalid_precision_infer(self):
        vsn = _valid_vsn_config()
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            infer=InferConfig(precision="int8"),
        )
        with pytest.raises(VGBConfigError, match="infer.precision"):
            full.validate()

    def test_cross_entropy_with_regression_head_fails(self):
        vsn = _valid_vsn_config(head_type="regression")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="cross_entropy"),
        )
        with pytest.raises(VGBConfigError, match="cross_entropy.*incompatible"):
            full.validate()

    def test_cross_entropy_with_dense_head_fails(self):
        vsn = _valid_vsn_config(head_type="dense")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="cross_entropy"),
        )
        with pytest.raises(VGBConfigError, match="cross_entropy.*incompatible"):
            full.validate()

    def test_mse_with_text_head_fails(self):
        vsn = _valid_vsn_config(head_type="text")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="mse"),
        )
        with pytest.raises(VGBConfigError, match="mse.*incompatible.*text"):
            full.validate()

    def test_l1_with_text_head_fails(self):
        vsn = _valid_vsn_config(head_type="text")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="l1"),
        )
        with pytest.raises(VGBConfigError, match="l1.*incompatible.*text"):
            full.validate()

    def test_cross_entropy_with_text_head_passes(self):
        vsn = _valid_vsn_config(head_type="text")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="cross_entropy"),
        )
        full.validate()  # Should not raise

    def test_cross_entropy_with_classification_head_passes(self):
        vsn = _valid_vsn_config(head_type="classification", num_classes=10)
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="cross_entropy"),
        )
        full.validate()  # Should not raise

    def test_mse_with_regression_head_passes(self):
        vsn = _valid_vsn_config(head_type="regression")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="mse"),
        )
        full.validate()  # Should not raise

    def test_l1_with_dense_head_passes(self):
        vsn = _valid_vsn_config(head_type="dense")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(loss_type="l1"),
        )
        full.validate()  # Should not raise

    def test_invalid_strategy(self):
        vsn = _valid_vsn_config()
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            runtime=RuntimeConfig(strategy="ddp"),
        )
        with pytest.raises(VGBConfigError, match="runtime.strategy"):
            full.validate()

    def test_invalid_vsn_config_propagates(self):
        # Create a VSNConfig with bad dimensions
        vsn = VSNConfig(
            X_enc=-1,
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
            head_type="text",
            vocab_size=32000,
        )
        full = FullConfig(model=ModelConfig(vsn=vsn))
        with pytest.raises(VGBConfigError, match="model.vsn"):
            full.validate()

    def test_negative_learning_rate(self):
        vsn = _valid_vsn_config()
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(learning_rate=-0.01),
        )
        with pytest.raises(VGBConfigError, match="learning_rate"):
            full.validate()

    def test_zero_batch_size_train(self):
        vsn = _valid_vsn_config()
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(batch_size=0),
        )
        with pytest.raises(VGBConfigError, match="train.batch_size"):
            full.validate()

    def test_multiple_errors_collected(self):
        vsn = _valid_vsn_config(head_type="regression")
        full = FullConfig(
            model=ModelConfig(vsn=vsn),
            train=TrainConfig(precision="wrong", loss_type="cross_entropy"),
            runtime=RuntimeConfig(strategy="invalid"),
        )
        with pytest.raises(VGBConfigError) as exc_info:
            full.validate()
        msg = str(exc_info.value)
        # Should collect all errors
        assert "train.precision" in msg
        assert "cross_entropy" in msg
        assert "runtime.strategy" in msg
