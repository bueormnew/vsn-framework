"""Tests for distributed checkpointing (vgb.runtime.checkpointing).

Verifies import, CheckpointState dataclass, RNG capture/restore,
and proper error handling when distributed is not initialized.

Validates: Requirements 9.1, 9.2, 8.4, 8.5
"""

import random

import torch
import torch.nn as nn
import pytest

from vgb.runtime.checkpointing import (
    CheckpointState,
    save_distributed_checkpoint,
    load_distributed_checkpoint,
    _capture_rng_states,
    _restore_rng_states,
    _build_state_dict,
    _check_distributed_available,
)


class TestCheckpointState:
    """Tests for CheckpointState dataclass."""

    def test_default_all_none(self):
        """All fields default to None."""
        state = CheckpointState()
        assert state.model is None
        assert state.optimizer is None
        assert state.scheduler is None
        assert state.scaler is None
        assert state.trainer_state is None
        assert state.rng is None

    def test_with_model(self):
        """Can be constructed with a model."""
        model = nn.Linear(8, 8)
        state = CheckpointState(model=model)
        assert state.model is model

    def test_with_trainer_state(self):
        """Can store arbitrary trainer state dict."""
        trainer_state = {"step": 100, "epoch": 2, "best_metric": 0.95}
        state = CheckpointState(trainer_state=trainer_state)
        assert state.trainer_state["step"] == 100
        assert state.trainer_state["epoch"] == 2

    def test_with_all_fields(self):
        """Can construct with all fields populated."""
        model = nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler()
        trainer_state = {"step": 0}
        rng = {"cpu": torch.random.get_rng_state()}

        state = CheckpointState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            trainer_state=trainer_state,
            rng=rng,
        )
        assert state.model is model
        assert state.optimizer is optimizer
        assert state.scheduler is scheduler
        assert state.scaler is scaler
        assert state.trainer_state is trainer_state
        assert state.rng is rng


class TestCaptureRngStates:
    """Tests for _capture_rng_states."""

    def test_captures_cpu_state(self):
        """Should always capture CPU RNG state."""
        rng = _capture_rng_states()
        assert "cpu" in rng
        assert isinstance(rng["cpu"], torch.Tensor)

    def test_captures_python_state(self):
        """Should capture Python random state."""
        rng = _capture_rng_states()
        assert "python" in rng

    def test_captures_numpy_if_available(self):
        """Should capture numpy state if numpy is installed."""
        try:
            import numpy  # noqa: F401
            rng = _capture_rng_states()
            assert "numpy" in rng
        except ImportError:
            pass


class TestRestoreRngStates:
    """Tests for _restore_rng_states."""

    def test_round_trip_cpu(self):
        """Capturing and restoring CPU RNG should reproduce random values."""
        torch.manual_seed(42)
        rng = _capture_rng_states()

        # Generate some random values
        vals_before = torch.randn(10)

        # Restore state
        _restore_rng_states(rng)
        vals_after = torch.randn(10)

        # After restoring, same sequence should be generated
        assert torch.equal(vals_before, vals_after)

    def test_round_trip_python(self):
        """Capturing and restoring Python RNG should reproduce values."""
        random.seed(42)
        rng = _capture_rng_states()

        vals_before = [random.random() for _ in range(5)]

        _restore_rng_states(rng)
        vals_after = [random.random() for _ in range(5)]

        assert vals_before == vals_after


class TestBuildStateDict:
    """Tests for _build_state_dict."""

    def test_with_model_only(self):
        """Should include model in state dict."""
        model = nn.Linear(4, 4)
        state = CheckpointState(model=model)
        sd = _build_state_dict(state)
        assert "model" in sd
        assert sd["model"] is model

    def test_auto_captures_rng_when_not_provided(self):
        """Should auto-capture RNG states if state.rng is None."""
        state = CheckpointState()
        sd = _build_state_dict(state)
        assert "rng" in sd
        assert "cpu" in sd["rng"]

    def test_uses_provided_rng(self):
        """Should use explicit RNG states if provided."""
        rng = {"cpu": torch.random.get_rng_state()}
        state = CheckpointState(rng=rng)
        sd = _build_state_dict(state)
        assert sd["rng"] is rng

    def test_scheduler_state_dict(self):
        """Should call state_dict() on scheduler."""
        model = nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        state = CheckpointState(scheduler=scheduler)
        sd = _build_state_dict(state)
        assert "scheduler" in sd
        assert isinstance(sd["scheduler"], dict)

    def test_scaler_state_dict(self):
        """Should call state_dict() on scaler."""
        scaler = torch.amp.GradScaler()
        state = CheckpointState(scaler=scaler)
        sd = _build_state_dict(state)
        assert "scaler" in sd
        assert isinstance(sd["scaler"], dict)


class TestSaveDistributedCheckpoint:
    """Tests for save_distributed_checkpoint — error handling."""

    def test_raises_without_distributed(self):
        """Should raise clear error without torchrun."""
        if not torch.distributed.is_initialized():
            state = CheckpointState(model=nn.Linear(4, 4))
            with pytest.raises(RuntimeError, match="not initialized"):
                save_distributed_checkpoint(state, "/tmp/test_ckpt")


class TestLoadDistributedCheckpoint:
    """Tests for load_distributed_checkpoint — error handling."""

    def test_raises_without_distributed(self):
        """Should raise clear error without torchrun."""
        if not torch.distributed.is_initialized():
            state = CheckpointState(model=nn.Linear(4, 4))
            with pytest.raises(RuntimeError, match="not initialized"):
                load_distributed_checkpoint(state, "/tmp/test_ckpt")
