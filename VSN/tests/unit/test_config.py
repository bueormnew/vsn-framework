"""Unit tests para VSNConfig — validación y factory methods."""

import pytest

from vsn.core.config import ConfigurationError, VSNConfig


class TestVSNConfigCreation:
    """Tests de creación e instanciación."""

    def test_create_valid_config(self):
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="text", vocab_size=32000,
        )
        config.validate()
        assert config.X_enc == 4
        assert config.d == 64

    def test_default_metadata_fields(self):
        config = VSNConfig.small()
        assert config.schema_version == "1.0"
        assert config.model_family == "vsn"
        assert config.vgb_version == "v1"
        assert config.psi_version == "v1"


class TestVSNConfigValidation:
    """Tests de validate() — verifica detección de errores."""

    def test_negative_dimension_raises(self):
        config = VSNConfig(
            X_enc=-1, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        with pytest.raises(ConfigurationError, match="X_enc must be a positive"):
            config.validate()

    def test_zero_ics_raises(self):
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=0,
            Y_H=4, Z_H=4, d_H=64, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        with pytest.raises(ConfigurationError, match="ics must be a positive"):
            config.validate()

    def test_invalid_p_mode_raises(self):
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="invalid",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        with pytest.raises(ConfigurationError, match="p_mode must be one of"):
            config.validate()

    def test_p_mode_compress_mismatch_raises(self):
        # p_mode='compress' but H volume == enc volume (identity dims)
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="compress",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        with pytest.raises(ConfigurationError, match="p_mode='compress'"):
            config.validate()

    def test_p_mode_identity_mismatch_raises(self):
        # p_mode='identity' but H volume != enc volume
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=2, Z_H=2, d_H=32, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        with pytest.raises(ConfigurationError, match="p_mode='identity'"):
            config.validate()

    def test_p_mode_expand_mismatch_raises(self):
        # p_mode='expand' but H volume == enc volume
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="expand",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        with pytest.raises(ConfigurationError, match="p_mode='expand'"):
            config.validate()

    def test_p_mode_compress_valid(self):
        # H volume < enc volume — should pass
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=2, Z_H=2, d_H=32, p_mode="compress",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        config.validate()  # No error

    def test_p_mode_expand_valid(self):
        # H volume > enc volume — should pass
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=8, Z_H=8, d_H=128, p_mode="expand",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="regression",
        )
        config.validate()  # No error

    def test_invalid_head_type_raises(self):
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="unknown",
        )
        with pytest.raises(ConfigurationError, match="head_type must be one of"):
            config.validate()

    def test_text_head_without_vocab_size_raises(self):
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="text", vocab_size=None,
        )
        with pytest.raises(ConfigurationError, match="vocab_size must be a positive"):
            config.validate()

    def test_classification_head_without_num_classes_raises(self):
        config = VSNConfig(
            X_enc=4, X_dec=4, Y=4, Z=4, d=64,
            ics=64,
            Y_H=4, Z_H=4, d_H=64, p_mode="identity",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="classification", num_classes=None,
        )
        with pytest.raises(ConfigurationError, match="num_classes must be a positive"):
            config.validate()

    def test_multiple_errors_reported(self):
        config = VSNConfig(
            X_enc=-1, X_dec=0, Y=4, Z=4, d=64,
            ics=0,
            Y_H=4, Z_H=4, d_H=64, p_mode="invalid",
            Y_dec=4, Z_dec=4, dgw=4,
            head_type="unknown",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert "X_enc" in msg
        assert "X_dec" in msg
        assert "ics" in msg
        assert "p_mode" in msg
        assert "head_type" in msg


class TestVSNConfigFactoryMethods:
    """Tests para factory methods small(), base(), large()."""

    def test_small_config(self):
        config = VSNConfig.small()
        assert config.X_enc == 4
        assert config.X_dec == 4
        assert config.Y == 4
        assert config.Z == 4
        assert config.d == 64
        assert config.ics == 64
        assert config.p_mode == "identity"
        assert config.dgw == 4
        assert config.Y_H == 4
        assert config.Z_H == 4
        assert config.d_H == 64
        config.validate()

    def test_base_config(self):
        config = VSNConfig.base()
        assert config.X_enc == 8
        assert config.X_dec == 8
        assert config.Y == 8
        assert config.Z == 8
        assert config.d == 128
        assert config.ics == 256
        assert config.p_mode == "identity"
        assert config.dgw == 8
        config.validate()

    def test_large_config(self):
        config = VSNConfig.large()
        assert config.X_enc == 16
        assert config.X_dec == 16
        assert config.Y == 16
        assert config.Z == 16
        assert config.d == 256
        assert config.ics == 1024
        assert config.p_mode == "identity"
        assert config.dgw == 16
        config.validate()

    def test_factory_with_classification_head(self):
        config = VSNConfig.small(
            head_type="classification", vocab_size=None, num_classes=10
        )
        assert config.head_type == "classification"
        assert config.num_classes == 10
        config.validate()

    def test_factory_with_regression_head(self):
        config = VSNConfig.base(head_type="regression", vocab_size=None)
        assert config.head_type == "regression"
        config.validate()

    def test_factory_with_dense_head(self):
        config = VSNConfig.large(head_type="dense", vocab_size=None)
        assert config.head_type == "dense"
        config.validate()
