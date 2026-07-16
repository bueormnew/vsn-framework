"""Tests for vgb.training — loops, metrics, and trainer.

Validates basic behavior on CPU with a simple model, single process.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vgb.config.schema import TrainConfig, RuntimeConfig
from vgb.training.loops import train_step
from vgb.training.metrics import MetricTracker, compute_perplexity
from vgb.training.trainer import Trainer


# ---------- Fixtures ----------


class SimpleModel(nn.Module):
    """A trivially simple model for testing the training loop."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class SimpleLoss(nn.Module):
    """MSE loss against zeros."""

    def forward(self, logits, targets):
        return nn.functional.mse_loss(logits, targets)


@pytest.fixture
def simple_model():
    return SimpleModel(d=16)


@pytest.fixture
def train_config():
    return TrainConfig(
        learning_rate=1e-3,
        max_steps=5,
        grad_clip_norm=1.0,
        grad_accum_steps=1,
        precision="fp32",
    )


# ---------- Tests: MetricTracker ----------


class TestMetricTracker:
    def test_empty_tracker(self):
        tracker = MetricTracker()
        assert tracker.get_averages() == {}
        assert tracker.count == 0

    def test_single_update(self):
        tracker = MetricTracker()
        tracker.update({"loss": 2.0, "grad_norm": 1.0})
        avgs = tracker.get_averages()
        assert avgs["loss"] == 2.0
        assert avgs["grad_norm"] == 1.0

    def test_multiple_updates(self):
        tracker = MetricTracker()
        tracker.update({"loss": 2.0})
        tracker.update({"loss": 4.0})
        avgs = tracker.get_averages()
        assert abs(avgs["loss"] - 3.0) < 1e-6

    def test_reset(self):
        tracker = MetricTracker()
        tracker.update({"loss": 5.0})
        tracker.reset()
        assert tracker.get_averages() == {}
        assert tracker.count == 0


# ---------- Tests: compute_perplexity ----------


class TestPerplexity:
    def test_zero_loss(self):
        assert compute_perplexity(0.0) == 1.0

    def test_normal_loss(self):
        ppl = compute_perplexity(2.0)
        assert abs(ppl - math.exp(2.0)) < 1e-4

    def test_high_loss_returns_inf(self):
        ppl = compute_perplexity(100.0)
        assert ppl == float("inf")


# ---------- Tests: train_step ----------


class TestTrainStep:
    def test_basic_train_step(self, simple_model, train_config):
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=1e-3)
        batch = torch.randn(2, 16)
        targets = torch.zeros(2, 16)
        loss_fn = SimpleLoss()

        result = train_step(
            model=simple_model,
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            config=train_config,
            step=0,
            loss_fn=loss_fn,
            targets=targets,
        )

        assert "loss" in result
        assert "grad_norm" in result
        assert result["loss"] > 0
        assert result["grad_norm"] > 0

    def test_gradient_accumulation(self, simple_model, train_config):
        """With grad_accum_steps=2, grad_norm should be 0 on odd steps."""
        train_config.grad_accum_steps = 2
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=1e-3)
        batch = torch.randn(2, 16)
        targets = torch.zeros(2, 16)
        loss_fn = SimpleLoss()

        # Step 0 (first accumulation step, not boundary)
        result = train_step(
            model=simple_model,
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            config=train_config,
            step=0,
            loss_fn=loss_fn,
            targets=targets,
        )
        # step 0: (0+1) % 2 == 1 != 0, so NOT boundary
        assert result["grad_norm"] == 0.0

        # Step 1 (boundary: (1+1) % 2 == 0)
        result = train_step(
            model=simple_model,
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            config=train_config,
            step=1,
            loss_fn=loss_fn,
            targets=targets,
        )
        assert result["grad_norm"] > 0


# ---------- Tests: Trainer ----------


class TestTrainer:
    def test_fit_runs_to_completion(self, simple_model, train_config):
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=1e-3)
        loss_fn = SimpleLoss()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                checkpoint_dir=str(Path(tmpdir) / "ckpts")
            )
            trainer = Trainer(
                model=simple_model,
                optimizer=optimizer,
                config=train_config,
                runtime_config=runtime_config,
                loss_fn=loss_fn,
            )

            # Create a simple dataloader
            data = torch.randn(10, 16)
            targets = torch.zeros(10, 16)
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=2)

            results = trainer.fit(dataloader)

        assert results["final_step"] == 5
        assert "final_loss" in results
        assert "avg_metrics" in results

    def test_resume_from_checkpoint(self, simple_model, train_config):
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=1e-3)
        loss_fn = SimpleLoss()
        train_config.max_steps = 4
        train_config.save_interval = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                checkpoint_dir=str(Path(tmpdir) / "ckpts")
            )

            # First run: train 4 steps, saves at step 2 and 4
            trainer = Trainer(
                model=simple_model,
                optimizer=optimizer,
                config=train_config,
                runtime_config=runtime_config,
                loss_fn=loss_fn,
            )

            data = torch.randn(10, 16)
            targets = torch.zeros(10, 16)
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=2)

            results = trainer.fit(dataloader)
            assert results["final_step"] == 4

            # Second run: should resume and find max_steps already reached
            train_config.max_steps = 4
            optimizer2 = torch.optim.SGD(simple_model.parameters(), lr=1e-3)
            trainer2 = Trainer(
                model=simple_model,
                optimizer=optimizer2,
                config=train_config,
                runtime_config=runtime_config,
                loss_fn=loss_fn,
            )
            results2 = trainer2.fit(dataloader)
            # Already at max_steps, so should exit immediately
            assert results2["final_step"] == 4
