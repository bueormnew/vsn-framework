"""Trainer class for the VGB framework.

Coordinates the full training loop: iterates steps, calls train_step,
runs periodic evaluation, saves periodic checkpoints, and supports
resuming from interrupted training.

Validates: Requirements 8.3, 8.4, 8.5
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from vgb.config.schema import TrainConfig, RuntimeConfig
from vgb.runtime.amp import create_grad_scaler
from vgb.training.loops import train_step
from vgb.training.metrics import MetricTracker, compute_perplexity

logger = logging.getLogger(__name__)


class Trainer:
    """Coordinates training: step loop, eval, checkpointing, resume.

    Args:
        model: The model to train (nn.Module).
        optimizer: Optimizer instance (created after FSDP if applicable).
        scheduler: Optional learning rate scheduler.
        config: TrainConfig with training hyperparameters.
        runtime_config: RuntimeConfig with checkpoint/log directories.
        loss_fn: Optional loss function module. If None, model must
            return loss directly.
        scaler: Optional GradScaler. If None, one will be created
            based on config.precision.

    Example:
        trainer = Trainer(model, optimizer, scheduler, train_config, runtime_config)
        results = trainer.fit(train_dataloader, eval_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[TrainConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainConfig()
        self.runtime_config = runtime_config or RuntimeConfig()
        self.loss_fn = loss_fn

        # Create scaler based on precision if not provided
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = create_grad_scaler(self.config.precision)

        # Training state
        self.global_step = 0
        self.best_metric = float("inf")

        # Metric tracking
        self.metric_tracker = MetricTracker()

        # Checkpoint directory
        self.checkpoint_dir = Path(self.runtime_config.checkpoint_dir)

    def fit(
        self,
        train_dataloader: DataLoader | Iterator,
        eval_dataloader: Optional[DataLoader | Iterator] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Iterates for max_steps, calling train_step at each iteration.
        Performs periodic evaluation and checkpoint saves.

        Supports resume: if a checkpoint exists in checkpoint_dir,
        training continues from the interrupted step.

        Args:
            train_dataloader: DataLoader or iterator yielding training batches.
                Each batch should be a Tensor or a tuple (inputs, targets).
            eval_dataloader: Optional DataLoader for periodic evaluation.

        Returns:
            Dictionary with training results:
                - 'final_step': last completed step
                - 'final_loss': last recorded loss
                - 'best_metric': best evaluation metric achieved
                - 'avg_metrics': average metrics over the entire run
        """
        # Attempt to resume from checkpoint
        self._maybe_resume()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        data_iter = iter(train_dataloader)

        logger.info(
            "Starting training from step %d to %d.",
            self.global_step,
            self.config.max_steps,
        )

        final_loss = 0.0

        while self.global_step < self.config.max_steps:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the data iterator (epoch boundary)
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Unpack batch: could be (inputs,) or (inputs, targets)
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                targets = None

            # Execute one training step
            step_metrics = train_step(
                model=self.model,
                batch=inputs,
                optimizer=self.optimizer,
                scaler=self.scaler,
                config=self.config,
                step=self.global_step,
                loss_fn=self.loss_fn,
                targets=targets,
            )

            final_loss = step_metrics["loss"]
            self.metric_tracker.update(step_metrics)

            # Step scheduler (if provided) at accumulation boundaries
            if (
                self.scheduler is not None
                and (self.global_step + 1) % self.config.grad_accum_steps == 0
            ):
                self.scheduler.step()

            self.global_step += 1

            # Periodic evaluation
            if (
                eval_dataloader is not None
                and self.config.eval_interval > 0
                and self.global_step % self.config.eval_interval == 0
            ):
                eval_loss = self._evaluate(eval_dataloader)
                if eval_loss < self.best_metric:
                    self.best_metric = eval_loss
                logger.info(
                    "Step %d | Eval loss: %.4f | Best: %.4f",
                    self.global_step,
                    eval_loss,
                    self.best_metric,
                )
                self.model.train()

            # Periodic checkpoint save
            if (
                self.config.save_interval > 0
                and self.global_step % self.config.save_interval == 0
            ):
                self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()

        avg_metrics = self.metric_tracker.get_averages()

        return {
            "final_step": self.global_step,
            "final_loss": final_loss,
            "best_metric": self.best_metric,
            "avg_metrics": avg_metrics,
        }

    def _evaluate(self, eval_dataloader: DataLoader | Iterator) -> float:
        """Run evaluation and return average loss.

        Args:
            eval_dataloader: DataLoader yielding evaluation batches.

        Returns:
            Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch
                    targets = None

                outputs = self.model(inputs)

                if self.loss_fn is not None and targets is not None:
                    logits = (
                        outputs.logits
                        if hasattr(outputs, "logits")
                        else outputs
                    )
                    loss = self.loss_fn(logits, targets)
                elif hasattr(outputs, "logits") and outputs.logits is not None:
                    loss = outputs.logits.sum() * 0.0
                else:
                    break

                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            return float("inf")

        return total_loss / num_batches

    def _save_checkpoint(self) -> None:
        """Save a local (non-distributed) training checkpoint.

        Saves model, optimizer, scheduler, scaler, and trainer state
        to a .pt file in the checkpoint directory.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / f"step_{self.global_step}.pt"

        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": {
                "learning_rate": self.config.learning_rate,
                "precision": self.config.precision,
                "max_steps": self.config.max_steps,
                "grad_accum_steps": self.config.grad_accum_steps,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)

    def _maybe_resume(self) -> None:
        """Detect and load the latest checkpoint if available.

        Looks for step_*.pt files in checkpoint_dir, selects the one
        with the highest step number, and restores all state.
        """
        if not self.checkpoint_dir.exists():
            return

        # Find all checkpoint files
        ckpt_files = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

        if not ckpt_files:
            return

        latest_ckpt = ckpt_files[-1]
        logger.info("Resuming from checkpoint: %s", latest_ckpt)

        checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)

        # Restore model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore scaler
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore trainer state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float("inf"))

        logger.info("Resumed at step %d.", self.global_step)
