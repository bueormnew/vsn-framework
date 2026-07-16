"""Tests unitarios para vsn.losses — CrossEntropy, MSE, L1, MultiTaskLoss."""

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from vsn.losses import (
    MaskedCrossEntropyLoss,
    MaskedL1Loss,
    MaskedMSELoss,
    MultiTaskLoss,
)


# ============================================================================
# MaskedCrossEntropyLoss
# ============================================================================


class TestMaskedCrossEntropyLoss:
    """Tests para MaskedCrossEntropyLoss."""

    def test_without_mask_matches_pytorch(self) -> None:
        """Sin máscara, el resultado debe ser idéntico a F.cross_entropy."""
        torch.manual_seed(42)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss_fn = MaskedCrossEntropyLoss(reduction="mean")
        result = loss_fn(logits, targets)
        expected = F.cross_entropy(logits, targets, reduction="mean")

        assert torch.allclose(result, expected, atol=1e-6)

    def test_without_mask_sum_reduction(self) -> None:
        """reduction='sum' sin máscara coincide con PyTorch."""
        torch.manual_seed(7)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))

        loss_fn = MaskedCrossEntropyLoss(reduction="sum")
        result = loss_fn(logits, targets)
        expected = F.cross_entropy(logits, targets, reduction="sum")

        assert torch.allclose(result, expected, atol=1e-6)

    def test_without_mask_none_reduction(self) -> None:
        """reduction='none' sin máscara devuelve tensor element-wise."""
        torch.manual_seed(3)
        logits = torch.randn(6, 3)
        targets = torch.randint(0, 3, (6,))

        loss_fn = MaskedCrossEntropyLoss(reduction="none")
        result = loss_fn(logits, targets)
        expected = F.cross_entropy(logits, targets, reduction="none")

        assert torch.allclose(result, expected, atol=1e-6)
        assert result.shape == (6,)

    def test_with_mask_zeros_masked_positions(self) -> None:
        """Con máscara, las posiciones inválidas no contribuyen a la pérdida."""
        torch.manual_seed(5)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Solo posiciones 0 y 2

        loss_fn = MaskedCrossEntropyLoss(reduction="mean")
        result = loss_fn(logits, targets, mask)

        # Calcular manualmente: loss solo en posiciones 0 y 2
        unreduced = F.cross_entropy(logits, targets, reduction="none")
        expected = (unreduced[0] + unreduced[2]) / 2.0

        assert torch.allclose(result, expected, atol=1e-6)

    def test_with_mask_sum_reduction(self) -> None:
        """Con máscara y sum, solo se suman posiciones válidas."""
        torch.manual_seed(9)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss_fn = MaskedCrossEntropyLoss(reduction="sum")
        result = loss_fn(logits, targets, mask)

        unreduced = F.cross_entropy(logits, targets, reduction="none")
        expected = (unreduced * mask).sum()

        assert torch.allclose(result, expected, atol=1e-6)

    def test_all_masked_returns_zero(self) -> None:
        """Si toda la máscara es cero, la pérdida es 0."""
        logits = torch.randn(3, 4)
        targets = torch.randint(0, 4, (3,))
        mask = torch.zeros(3)

        loss_fn = MaskedCrossEntropyLoss(reduction="mean")
        result = loss_fn(logits, targets, mask)

        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_invalid_reduction_raises(self) -> None:
        """Reducción inválida lanza ValueError."""
        with pytest.raises(ValueError, match="reduction"):
            MaskedCrossEntropyLoss(reduction="invalid")

    def test_is_nn_module(self) -> None:
        """Es una subclase de nn.Module para compatibilidad con state_dict."""
        loss_fn = MaskedCrossEntropyLoss()
        assert isinstance(loss_fn, torch.nn.Module)

    def test_label_smoothing(self) -> None:
        """Label smoothing se propaga correctamente."""
        torch.manual_seed(11)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss_fn = MaskedCrossEntropyLoss(reduction="mean", label_smoothing=0.1)
        result = loss_fn(logits, targets)
        expected = F.cross_entropy(
            logits, targets, reduction="mean", label_smoothing=0.1
        )

        assert torch.allclose(result, expected, atol=1e-6)


# ============================================================================
# MaskedMSELoss
# ============================================================================


class TestMaskedMSELoss:
    """Tests para MaskedMSELoss."""

    def test_without_mask_matches_pytorch(self) -> None:
        """Sin máscara, el resultado debe ser idéntico a F.mse_loss."""
        torch.manual_seed(42)
        preds = torch.randn(8, 4)
        targets = torch.randn(8, 4)

        loss_fn = MaskedMSELoss(reduction="mean")
        result = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="mean")

        assert torch.allclose(result, expected, atol=1e-6)

    def test_without_mask_sum_reduction(self) -> None:
        """reduction='sum' sin máscara."""
        torch.manual_seed(7)
        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)

        loss_fn = MaskedMSELoss(reduction="sum")
        result = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="sum")

        assert torch.allclose(result, expected, atol=1e-6)

    def test_without_mask_none_reduction(self) -> None:
        """reduction='none' devuelve tensor element-wise."""
        torch.manual_seed(3)
        preds = torch.randn(6, 2)
        targets = torch.randn(6, 2)

        loss_fn = MaskedMSELoss(reduction="none")
        result = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="none")

        assert torch.allclose(result, expected, atol=1e-6)
        assert result.shape == (6, 2)

    def test_with_mask_batch_dimension(self) -> None:
        """Máscara sobre dimensión batch, expande a features."""
        torch.manual_seed(5)
        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Por batch element

        loss_fn = MaskedMSELoss(reduction="mean")
        result = loss_fn(preds, targets, mask)

        # Calcular manualmente: MSE solo en posiciones 0 y 2
        unreduced = F.mse_loss(preds, targets, reduction="none")
        mask_expanded = mask.unsqueeze(-1)  # (4, 1) → broadcasts a (4, 3)
        masked = unreduced * mask_expanded
        num_valid = mask_expanded.expand_as(unreduced).sum()
        expected = masked.sum() / num_valid

        assert torch.allclose(result, expected, atol=1e-6)

    def test_with_mask_sum_reduction(self) -> None:
        """Con máscara y sum reduction."""
        torch.manual_seed(9)
        preds = torch.randn(4, 2)
        targets = torch.randn(4, 2)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss_fn = MaskedMSELoss(reduction="sum")
        result = loss_fn(preds, targets, mask)

        unreduced = F.mse_loss(preds, targets, reduction="none")
        mask_expanded = mask.unsqueeze(-1)
        expected = (unreduced * mask_expanded).sum()

        assert torch.allclose(result, expected, atol=1e-6)

    def test_all_masked_returns_zero(self) -> None:
        """Si toda la máscara es cero, la pérdida es 0."""
        preds = torch.randn(3, 4)
        targets = torch.randn(3, 4)
        mask = torch.zeros(3)

        loss_fn = MaskedMSELoss(reduction="mean")
        result = loss_fn(preds, targets, mask)

        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_invalid_reduction_raises(self) -> None:
        """Reducción inválida lanza ValueError."""
        with pytest.raises(ValueError, match="reduction"):
            MaskedMSELoss(reduction="invalid")

    def test_is_nn_module(self) -> None:
        """Es una subclase de nn.Module."""
        loss_fn = MaskedMSELoss()
        assert isinstance(loss_fn, torch.nn.Module)


# ============================================================================
# MaskedL1Loss
# ============================================================================


class TestMaskedL1Loss:
    """Tests para MaskedL1Loss."""

    def test_without_mask_matches_pytorch(self) -> None:
        """Sin máscara, el resultado debe ser idéntico a F.l1_loss."""
        torch.manual_seed(42)
        preds = torch.randn(8, 4)
        targets = torch.randn(8, 4)

        loss_fn = MaskedL1Loss(reduction="mean")
        result = loss_fn(preds, targets)
        expected = F.l1_loss(preds, targets, reduction="mean")

        assert torch.allclose(result, expected, atol=1e-6)

    def test_without_mask_sum_reduction(self) -> None:
        """reduction='sum' sin máscara."""
        torch.manual_seed(7)
        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)

        loss_fn = MaskedL1Loss(reduction="sum")
        result = loss_fn(preds, targets)
        expected = F.l1_loss(preds, targets, reduction="sum")

        assert torch.allclose(result, expected, atol=1e-6)

    def test_without_mask_none_reduction(self) -> None:
        """reduction='none' devuelve tensor element-wise."""
        torch.manual_seed(3)
        preds = torch.randn(6, 2)
        targets = torch.randn(6, 2)

        loss_fn = MaskedL1Loss(reduction="none")
        result = loss_fn(preds, targets)
        expected = F.l1_loss(preds, targets, reduction="none")

        assert torch.allclose(result, expected, atol=1e-6)
        assert result.shape == (6, 2)

    def test_with_mask_batch_dimension(self) -> None:
        """Máscara sobre dimensión batch, expande a features."""
        torch.manual_seed(5)
        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss_fn = MaskedL1Loss(reduction="mean")
        result = loss_fn(preds, targets, mask)

        # Calcular manualmente
        unreduced = F.l1_loss(preds, targets, reduction="none")
        mask_expanded = mask.unsqueeze(-1)
        masked = unreduced * mask_expanded
        num_valid = mask_expanded.expand_as(unreduced).sum()
        expected = masked.sum() / num_valid

        assert torch.allclose(result, expected, atol=1e-6)

    def test_with_mask_sum_reduction(self) -> None:
        """Con máscara y sum reduction."""
        torch.manual_seed(9)
        preds = torch.randn(4, 2)
        targets = torch.randn(4, 2)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss_fn = MaskedL1Loss(reduction="sum")
        result = loss_fn(preds, targets, mask)

        unreduced = F.l1_loss(preds, targets, reduction="none")
        mask_expanded = mask.unsqueeze(-1)
        expected = (unreduced * mask_expanded).sum()

        assert torch.allclose(result, expected, atol=1e-6)

    def test_all_masked_returns_zero(self) -> None:
        """Si toda la máscara es cero, la pérdida es 0."""
        preds = torch.randn(3, 4)
        targets = torch.randn(3, 4)
        mask = torch.zeros(3)

        loss_fn = MaskedL1Loss(reduction="mean")
        result = loss_fn(preds, targets, mask)

        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_invalid_reduction_raises(self) -> None:
        """Reducción inválida lanza ValueError."""
        with pytest.raises(ValueError, match="reduction"):
            MaskedL1Loss(reduction="invalid")

    def test_is_nn_module(self) -> None:
        """Es una subclase de nn.Module."""
        loss_fn = MaskedL1Loss()
        assert isinstance(loss_fn, torch.nn.Module)


# ============================================================================
# MultiTaskLoss
# ============================================================================


class TestMultiTaskLoss:
    """Tests para MultiTaskLoss."""

    def test_weighted_sum_correctness(self) -> None:
        """La suma ponderada Σ(w_i × loss_i) es correcta."""
        torch.manual_seed(42)
        preds_a = torch.randn(4, 3)
        targets_a = torch.randn(4, 3)
        preds_b = torch.randn(4, 5)
        targets_b = torch.randint(0, 5, (4,))

        losses = {
            "regression": MaskedMSELoss(reduction="mean"),
            "classification": MaskedCrossEntropyLoss(reduction="mean"),
        }
        weights = {"regression": 0.7, "classification": 0.3}

        combiner = MultiTaskLoss(losses, weights)
        inputs = {
            "regression": (preds_a, targets_a, None),
            "classification": (preds_b, targets_b, None),
        }
        total, individual = combiner(inputs)

        # Verificar suma ponderada
        expected_total = (
            0.7 * individual["regression"] + 0.3 * individual["classification"]
        )
        assert torch.allclose(total, expected_total, atol=1e-6)

    def test_individual_losses_correct(self) -> None:
        """Los valores individuales coinciden con calcular cada loss por separado."""
        torch.manual_seed(11)
        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)

        mse_fn = MaskedMSELoss(reduction="mean")
        l1_fn = MaskedL1Loss(reduction="mean")

        losses = {"mse": MaskedMSELoss(reduction="mean"), "l1": MaskedL1Loss(reduction="mean")}
        weights = {"mse": 1.0, "l1": 1.0}

        combiner = MultiTaskLoss(losses, weights)
        inputs = {
            "mse": (preds, targets, None),
            "l1": (preds, targets, None),
        }
        total, individual = combiner(inputs)

        expected_mse = mse_fn(preds, targets)
        expected_l1 = l1_fn(preds, targets)

        assert torch.allclose(individual["mse"], expected_mse, atol=1e-6)
        assert torch.allclose(individual["l1"], expected_l1, atol=1e-6)

    def test_with_masks_forwarded(self) -> None:
        """Las máscaras se propagan correctamente a los loss individuales."""
        torch.manual_seed(7)
        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        mask = torch.tensor([1.0, 0.0, 1.0, 1.0])

        losses = {"task": MaskedMSELoss(reduction="mean")}
        weights = {"task": 2.0}

        combiner = MultiTaskLoss(losses, weights)
        inputs = {"task": (preds, targets, mask)}
        total, individual = combiner(inputs)

        # Verificar que la máscara se aplicó
        expected_loss = MaskedMSELoss(reduction="mean")(preds, targets, mask)
        assert torch.allclose(individual["task"], expected_loss, atol=1e-6)
        assert torch.allclose(total, 2.0 * expected_loss, atol=1e-6)

    def test_mismatched_keys_raises(self) -> None:
        """Claves inconsistentes entre losses y weights lanza ValueError."""
        losses = {"a": MaskedMSELoss()}
        weights = {"b": 1.0}

        with pytest.raises(ValueError, match="coincidir"):
            MultiTaskLoss(losses, weights)

    def test_mismatched_input_keys_raises(self) -> None:
        """Claves de input que no coinciden con las registradas lanza ValueError."""
        losses = {"a": MaskedMSELoss()}
        weights = {"a": 1.0}
        combiner = MultiTaskLoss(losses, weights)

        preds = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        inputs = {"wrong_key": (preds, targets, None)}

        with pytest.raises(ValueError, match="coincidir"):
            combiner(inputs)

    def test_is_nn_module(self) -> None:
        """Es una subclase de nn.Module con sub-módulos registrados."""
        losses = {
            "a": MaskedMSELoss(),
            "b": MaskedL1Loss(),
        }
        weights = {"a": 0.5, "b": 0.5}
        combiner = MultiTaskLoss(losses, weights)

        assert isinstance(combiner, torch.nn.Module)
        # Los sub-módulos deben estar registrados (accesibles via state_dict)
        state_dict = combiner.state_dict()
        # ModuleDict no tiene parámetros en losses sin parámetros,
        # pero la estructura debe existir
        assert hasattr(combiner, "losses")

    def test_three_tasks_combined(self) -> None:
        """Tres tareas combinadas producen la suma ponderada correcta."""
        torch.manual_seed(99)
        losses = {
            "mse": MaskedMSELoss(reduction="mean"),
            "l1": MaskedL1Loss(reduction="mean"),
            "ce": MaskedCrossEntropyLoss(reduction="mean"),
        }
        weights = {"mse": 0.5, "l1": 0.3, "ce": 0.2}

        combiner = MultiTaskLoss(losses, weights)

        preds_reg = torch.randn(4, 8)
        targets_reg = torch.randn(4, 8)
        preds_ce = torch.randn(4, 10)
        targets_ce = torch.randint(0, 10, (4,))

        inputs = {
            "mse": (preds_reg, targets_reg, None),
            "l1": (preds_reg, targets_reg, None),
            "ce": (preds_ce, targets_ce, None),
        }

        total, individual = combiner(inputs)

        expected = (
            0.5 * individual["mse"]
            + 0.3 * individual["l1"]
            + 0.2 * individual["ce"]
        )
        assert torch.allclose(total, expected, atol=1e-6)

    def test_gradient_flows(self) -> None:
        """Los gradientes fluyen a través del combinador."""
        preds = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)

        losses = {"mse": MaskedMSELoss(reduction="mean")}
        weights = {"mse": 1.0}
        combiner = MultiTaskLoss(losses, weights)

        inputs = {"mse": (preds, targets, None)}
        total, _ = combiner(inputs)
        total.backward()

        assert preds.grad is not None
        assert preds.grad.shape == preds.shape
