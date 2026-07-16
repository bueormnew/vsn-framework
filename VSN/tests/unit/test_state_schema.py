"""Unit tests para StateSchema — serialización round-trip y validación."""

import json

import pytest

from vsn.core.config import VSNConfig
from vsn.io.state_schema import StateSchema, StateSchemaError


def _make_valid_schema(**overrides) -> StateSchema:
    """Helper para crear un StateSchema válido con posibles overrides."""
    defaults = dict(
        schema_version="1.0",
        model_family="vsn",
        vgb_version="v1",
        psi_version="v1",
        head_type="text",
        dims={
            "X_enc": 4, "X_dec": 4,
            "Y": 4, "Z": 4, "d": 64,
            "Y_H": 4, "Z_H": 4, "d_H": 64,
            "Y_dec": 4, "Z_dec": 4,
        },
        ics=64,
        dgw=4,
        total_params=100000,
        created_at="2024-01-01T00:00:00+00:00",
        checksum=None,
    )
    defaults.update(overrides)
    return StateSchema(**defaults)


class TestStateSchemaCreation:
    """Tests de creación e instanciación."""

    def test_create_valid_schema(self):
        schema = _make_valid_schema()
        assert schema.schema_version == "1.0"
        assert schema.model_family == "vsn"
        assert schema.dims["X_enc"] == 4
        assert schema.total_params == 100000

    def test_checksum_defaults_to_none(self):
        schema = _make_valid_schema()
        assert schema.checksum is None

    def test_checksum_can_be_set(self):
        schema = _make_valid_schema(checksum="abc123")
        assert schema.checksum == "abc123"


class TestStateSchemaSerializationRoundTrip:
    """Tests de serialización y deserialización."""

    def test_to_dict_round_trip(self):
        schema = _make_valid_schema()
        d = schema.to_dict()
        restored = StateSchema.from_dict(d)
        assert restored == schema

    def test_to_json_round_trip(self):
        schema = _make_valid_schema(checksum="sha256:abc123")
        json_str = schema.to_json()
        restored = StateSchema.from_json(json_str)
        assert restored == schema

    def test_to_json_produces_valid_json(self):
        schema = _make_valid_schema()
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0"
        assert parsed["dims"]["d"] == 64

    def test_to_dict_returns_plain_dict(self):
        schema = _make_valid_schema()
        d = schema.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["dims"], dict)

    def test_from_dict_missing_field_raises(self):
        data = _make_valid_schema().to_dict()
        del data["schema_version"]
        with pytest.raises(StateSchemaError, match="Missing required fields"):
            StateSchema.from_dict(data)

    def test_from_json_invalid_json_raises(self):
        with pytest.raises(StateSchemaError, match="Invalid JSON"):
            StateSchema.from_json("{not valid json")

    def test_from_dict_with_extra_fields_ignored(self):
        data = _make_valid_schema().to_dict()
        data["extra_field"] = "ignored"
        schema = StateSchema.from_dict(data)
        assert schema.schema_version == "1.0"


class TestStateSchemaFromConfig:
    """Tests del factory method from_config."""

    def test_from_config_small(self):
        config = VSNConfig.small()
        schema = StateSchema.from_config(config, total_params=50000)
        assert schema.schema_version == "1.0"
        assert schema.model_family == "vsn"
        assert schema.vgb_version == "v1"
        assert schema.psi_version == "v1"
        assert schema.head_type == "text"
        assert schema.dims["X_enc"] == 4
        assert schema.dims["Y"] == 4
        assert schema.dims["d"] == 64
        assert schema.ics == 64
        assert schema.dgw == 4
        assert schema.total_params == 50000
        assert schema.created_at  # non-empty string
        assert schema.checksum is None

    def test_from_config_base(self):
        config = VSNConfig.base()
        schema = StateSchema.from_config(config, total_params=500000)
        assert schema.dims["X_enc"] == 8
        assert schema.dims["d"] == 128
        assert schema.ics == 256
        assert schema.dgw == 8

    def test_from_config_produces_valid_schema(self):
        config = VSNConfig.small()
        schema = StateSchema.from_config(config, total_params=50000)
        # Should not raise
        schema.validate()


class TestStateSchemaValidation:
    """Tests de validate() — detección de errores."""

    def test_valid_schema_passes(self):
        schema = _make_valid_schema()
        schema.validate()  # Should not raise

    def test_unsupported_schema_version_raises(self):
        schema = _make_valid_schema(schema_version="99.0")
        with pytest.raises(StateSchemaError, match="Unsupported schema_version"):
            schema.validate()

    def test_empty_model_family_raises(self):
        schema = _make_valid_schema(model_family="")
        with pytest.raises(StateSchemaError, match="model_family"):
            schema.validate()

    def test_missing_dim_key_raises(self):
        dims = {"X_enc": 4, "X_dec": 4, "Y": 4, "Z": 4, "d": 64}
        # Missing Y_H, Z_H, d_H, Y_dec, Z_dec
        schema = _make_valid_schema(dims=dims)
        with pytest.raises(StateSchemaError, match="dims missing required keys"):
            schema.validate()

    def test_negative_dim_value_raises(self):
        schema = _make_valid_schema()
        schema.dims["d"] = -1
        with pytest.raises(StateSchemaError, match="dims\\['d'\\]"):
            schema.validate()

    def test_zero_ics_raises(self):
        schema = _make_valid_schema(ics=0)
        with pytest.raises(StateSchemaError, match="ics must be a positive integer"):
            schema.validate()

    def test_zero_dgw_raises(self):
        schema = _make_valid_schema(dgw=0)
        with pytest.raises(StateSchemaError, match="dgw must be a positive integer"):
            schema.validate()

    def test_negative_total_params_raises(self):
        schema = _make_valid_schema(total_params=-1)
        with pytest.raises(StateSchemaError, match="total_params"):
            schema.validate()

    def test_zero_total_params_valid(self):
        schema = _make_valid_schema(total_params=0)
        schema.validate()  # 0 params is valid (edge case)

    def test_multiple_errors_reported(self):
        schema = _make_valid_schema(
            schema_version="99.0",
            model_family="",
            ics=-1,
        )
        with pytest.raises(StateSchemaError) as exc_info:
            schema.validate()
        error_msg = str(exc_info.value)
        assert "schema_version" in error_msg
        assert "model_family" in error_msg
        assert "ics" in error_msg
