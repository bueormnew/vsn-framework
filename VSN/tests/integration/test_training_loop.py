"""Integration test: paso de entrenamiento single GPU con datos sintéticos.

Verifica que el modelo:
- Permite backward sobre loss computada desde decoder states
- Los parámetros cambian después de optimizer.step()
- No hay NaN en parámetros tras el entrenamiento
- La loss decrece tras múltiples pasos de entrenamiento

Validates: Requirements 13.3, 13.6
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from vsn.contracts.outputs import ModelOutputs
from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel


class MeanPoolHead(nn.Module):
    """Head que mean-pools decoder states para calcular una loss.

    Sigue el HeadProtocol del VSNModel: (decoder_states, metadata) → ModelOutputs.
    """

    def __init__(self, d: int, output_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(d, output_dim)

    def forward(
        self, decoder_states: List[Tensor], metadata: Dict[str, Any]
    ) -> ModelOutputs:
        # Mean pool sobre ventanas y dimensiones espaciales
        stacked = torch.stack(decoder_states, dim=1)  # (batch, W, Y, Z, d)
        pooled = stacked.mean(dim=(1, 2, 3))  # (batch, d)
        predictions = self.linear(pooled)  # (batch, output_dim)
        return ModelOutputs(
            embeddings=predictions,
            metadata={**metadata, "head_type": "regression"},
        )


class TestTrainingLoop:
    """Tests de integración para el loop de entrenamiento."""

    def _make_model_and_data(self, batch_size: int = 2):
        """Crea modelo, optimizer, datos sintéticos y targets."""
        config = VSNConfig.small(head_type="regression")
        head = MeanPoolHead(d=config.d, output_dim=4)
        model = VSNModel(config, head=head)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Datos sintéticos
        num_tokens = config.Y * config.Z
        tokens = torch.randn(batch_size, num_tokens, config.d)
        targets = torch.randn(batch_size, 4)

        return model, optimizer, tokens, targets

    def test_backward_computes_gradients(self) -> None:
        """backward() produce gradientes en todos los parámetros."""
        model, optimizer, tokens, targets = self._make_model_and_data()

        outputs = model(tokens)
        predictions = outputs.embeddings  # (batch, 4)

        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()

        # Verificar que al menos algunos parámetros tienen gradientes
        params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())

        assert params_with_grad > 0, "No parameters received gradients"
        # La mayoría de parámetros deben recibir gradientes
        assert params_with_grad >= total_params * 0.5

    def test_optimizer_step_changes_parameters(self) -> None:
        """optimizer.step() modifica los parámetros del modelo."""
        model, optimizer, tokens, targets = self._make_model_and_data()

        # Guardar snapshot de parámetros antes
        params_before = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # Training step
        outputs = model(tokens)
        loss = nn.functional.mse_loss(outputs.embeddings, targets)
        loss.backward()
        optimizer.step()

        # Verificar que al menos algunos params cambiaron
        changed_count = 0
        for name, p in model.named_parameters():
            if name in params_before and not torch.equal(p.data, params_before[name]):
                changed_count += 1

        assert changed_count > 0, "No parameters changed after optimizer.step()"

    def test_no_nan_after_training(self) -> None:
        """Ningún parámetro contiene NaN después de 3 pasos de entrenamiento."""
        model, optimizer, tokens, targets = self._make_model_and_data()

        for step in range(3):
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = nn.functional.mse_loss(outputs.embeddings, targets)
            loss.backward()
            optimizer.step()

        # Verificar que ningún parámetro tiene NaN
        for name, p in model.named_parameters():
            assert not torch.isnan(p.data).any(), (
                f"NaN found in parameter '{name}' after training"
            )
            assert not torch.isinf(p.data).any(), (
                f"Inf found in parameter '{name}' after training"
            )

    def test_loss_decreases_over_steps(self) -> None:
        """La loss decrece tras múltiples pasos de entrenamiento con datos fijos."""
        torch.manual_seed(123)

        config = VSNConfig.small(head_type="regression")
        head = MeanPoolHead(d=config.d, output_dim=4)
        model = VSNModel(config, head=head)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Datos fijos para entrenamiento
        num_tokens = config.Y * config.Z
        tokens = torch.randn(2, num_tokens, config.d)
        targets = torch.randn(2, 4)

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = nn.functional.mse_loss(outputs.embeddings, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # La loss final debe ser menor que la inicial
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.6f}, "
            f"final={losses[-1]:.6f}"
        )

    def test_training_without_head_using_states(self) -> None:
        """Entrenamiento sin head: loss computada directamente sobre states."""
        config = VSNConfig.small(head_type="regression")
        model = VSNModel(config, head=None)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        num_tokens = config.Y * config.Z
        tokens = torch.randn(2, num_tokens, config.d)

        # Loss basada en mean-pool de decoder states
        outputs = model(tokens)
        decoder_states = outputs.states["decoder_states"]
        pooled = torch.stack(decoder_states).mean(dim=(0, 2, 3))  # (batch, d)
        target = torch.zeros_like(pooled)
        loss = nn.functional.mse_loss(pooled, target)

        # Backward debe funcionar
        loss.backward()

        grads_exist = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert grads_exist, "No gradients computed for model without head"
