"""Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

**Validates: Requirements 8.1, 8.2**

Tests de propiedad que verifican:
1. Para cualquier (logits, targets) par, MaskedCrossEntropyLoss sin máscara SHALL
   producir valores idénticos (dentro de tolerancia fp) a F.cross_entropy.
2. Para cualquier (predictions, targets) par, MaskedMSELoss sin máscara SHALL
   producir valores idénticos a F.mse_loss.
3. Para cualquier (predictions, targets) par, MaskedL1Loss sin máscara SHALL
   producir valores idénticos a F.l1_loss.
4. Para cualquier conjunto de losses individuales y pesos, MultiTaskLoss SHALL
   producir Σ(w_i × loss_i).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from vsn.losses.cross_entropy import MaskedCrossEntropyLoss
from vsn.losses.l1 import MaskedL1Loss
from vsn.losses.mse import MaskedMSELoss
from vsn.losses.combiner import MultiTaskLoss

# ---------------------------------------------------------------------------
# Estrategias para generar tensores válidos con dimensiones pequeñas
# ---------------------------------------------------------------------------

batch_strategy = st.integers(min_value=1, max_value=8)
seq_strategy = st.integers(min_value=1, max_value=16)
num_classes_strategy = st.integers(min_value=2, max_value=10)
feature_strategy = st.integers(min_value=1, max_value=8)
num_tasks_strategy = st.integers(min_value=1, max_value=4)


@st.composite
def cross_entropy_inputs(draw: st.DrawFn) -> dict:
    """Genera pares (logits, targets) válidos para cross entropy."""
    B = draw(batch_strategy)
    C = draw(num_classes_strategy)
    # Logits: (B, C) con valores razonables para estabilidad numérica
    logits = torch.randn(B, C) * 2.0
    # Targets: (B,) con índices de clase válidos
    targets = torch.randint(0, C, (B,))
    return {"logits": logits, "targets": targets, "B": B, "C": C}


@st.composite
def regression_inputs(draw: st.DrawFn) -> dict:
    """Genera pares (predictions, targets) válidos para MSE/L1."""
    B = draw(batch_strategy)
    D = draw(feature_strategy)
    # Predicciones y targets con valores razonables
    predictions = torch.randn(B, D)
    targets = torch.randn(B, D)
    return {"predictions": predictions, "targets": targets, "B": B, "D": D}


@st.composite
def multitask_inputs(draw: st.DrawFn) -> dict:
    """Genera un conjunto de losses individuales con pesos para MultiTaskLoss."""
    num_tasks = draw(num_tasks_strategy)
    B = draw(batch_strategy)
    D = draw(feature_strategy)

    task_names = [f"task_{i}" for i in range(num_tasks)]
    # Pesos positivos para cada tarea
    weights = {
        name: draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
        for name in task_names
    }
    # Generar predicciones y targets para cada tarea (todas MSE para simplicidad)
    task_data = {}
    for name in task_names:
        preds = torch.randn(B, D)
        tgts = torch.randn(B, D)
        task_data[name] = (preds, tgts)

    return {
        "task_names": task_names,
        "weights": weights,
        "task_data": task_data,
        "B": B,
        "D": D,
    }


# ---------------------------------------------------------------------------
# Propiedad 14.1: MaskedCrossEntropyLoss sin máscara ≡ F.cross_entropy
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=cross_entropy_inputs())
def test_cross_entropy_matches_pytorch_reference(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier (logits, targets) par, MaskedCrossEntropyLoss sin máscara
    SHALL producir valores idénticos (dentro de tolerancia fp) a F.cross_entropy.
    """
    logits = data["logits"]
    targets = data["targets"]

    loss_fn = MaskedCrossEntropyLoss(reduction="mean")
    our_loss = loss_fn(logits, targets, mask=None)

    ref_loss = F.cross_entropy(logits, targets, reduction="mean")

    assert torch.isfinite(our_loss), "MaskedCrossEntropyLoss produjo NaN o Inf"
    assert torch.allclose(our_loss, ref_loss, atol=1e-6, rtol=1e-5), (
        f"MaskedCrossEntropyLoss(mean) = {our_loss.item():.8f} != "
        f"F.cross_entropy(mean) = {ref_loss.item():.8f}, "
        f"diff = {(our_loss - ref_loss).abs().item():.2e}"
    )


@settings(max_examples=100, deadline=None)
@given(data=cross_entropy_inputs())
def test_cross_entropy_sum_matches_pytorch_reference(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier (logits, targets) par, MaskedCrossEntropyLoss(reduction='sum') sin
    máscara SHALL producir valores idénticos a F.cross_entropy(reduction='sum').
    """
    logits = data["logits"]
    targets = data["targets"]

    loss_fn = MaskedCrossEntropyLoss(reduction="sum")
    our_loss = loss_fn(logits, targets, mask=None)

    ref_loss = F.cross_entropy(logits, targets, reduction="sum")

    assert torch.isfinite(our_loss), "MaskedCrossEntropyLoss(sum) produjo NaN o Inf"
    assert torch.allclose(our_loss, ref_loss, atol=1e-6, rtol=1e-5), (
        f"MaskedCrossEntropyLoss(sum) = {our_loss.item():.8f} != "
        f"F.cross_entropy(sum) = {ref_loss.item():.8f}, "
        f"diff = {(our_loss - ref_loss).abs().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Propiedad 14.2: MaskedMSELoss sin máscara ≡ F.mse_loss
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=regression_inputs())
def test_mse_loss_matches_pytorch_reference(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier (predictions, targets) par, MaskedMSELoss sin máscara SHALL
    producir valores idénticos a F.mse_loss.
    """
    predictions = data["predictions"]
    targets = data["targets"]

    loss_fn = MaskedMSELoss(reduction="mean")
    our_loss = loss_fn(predictions, targets, mask=None)

    ref_loss = F.mse_loss(predictions, targets, reduction="mean")

    assert torch.isfinite(our_loss), "MaskedMSELoss produjo NaN o Inf"
    assert torch.allclose(our_loss, ref_loss, atol=1e-6, rtol=1e-5), (
        f"MaskedMSELoss(mean) = {our_loss.item():.8f} != "
        f"F.mse_loss(mean) = {ref_loss.item():.8f}, "
        f"diff = {(our_loss - ref_loss).abs().item():.2e}"
    )


@settings(max_examples=100, deadline=None)
@given(data=regression_inputs())
def test_mse_loss_sum_matches_pytorch_reference(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier (predictions, targets) par, MaskedMSELoss(reduction='sum') sin
    máscara SHALL producir valores idénticos a F.mse_loss(reduction='sum').
    """
    predictions = data["predictions"]
    targets = data["targets"]

    loss_fn = MaskedMSELoss(reduction="sum")
    our_loss = loss_fn(predictions, targets, mask=None)

    ref_loss = F.mse_loss(predictions, targets, reduction="sum")

    assert torch.isfinite(our_loss), "MaskedMSELoss(sum) produjo NaN o Inf"
    assert torch.allclose(our_loss, ref_loss, atol=1e-6, rtol=1e-5), (
        f"MaskedMSELoss(sum) = {our_loss.item():.8f} != "
        f"F.mse_loss(sum) = {ref_loss.item():.8f}, "
        f"diff = {(our_loss - ref_loss).abs().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Propiedad 14.3: MaskedL1Loss sin máscara ≡ F.l1_loss
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=regression_inputs())
def test_l1_loss_matches_pytorch_reference(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier (predictions, targets) par, MaskedL1Loss sin máscara SHALL
    producir valores idénticos a F.l1_loss.
    """
    predictions = data["predictions"]
    targets = data["targets"]

    loss_fn = MaskedL1Loss(reduction="mean")
    our_loss = loss_fn(predictions, targets, mask=None)

    ref_loss = F.l1_loss(predictions, targets, reduction="mean")

    assert torch.isfinite(our_loss), "MaskedL1Loss produjo NaN o Inf"
    assert torch.allclose(our_loss, ref_loss, atol=1e-6, rtol=1e-5), (
        f"MaskedL1Loss(mean) = {our_loss.item():.8f} != "
        f"F.l1_loss(mean) = {ref_loss.item():.8f}, "
        f"diff = {(our_loss - ref_loss).abs().item():.2e}"
    )


@settings(max_examples=100, deadline=None)
@given(data=regression_inputs())
def test_l1_loss_sum_matches_pytorch_reference(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier (predictions, targets) par, MaskedL1Loss(reduction='sum') sin
    máscara SHALL producir valores idénticos a F.l1_loss(reduction='sum').
    """
    predictions = data["predictions"]
    targets = data["targets"]

    loss_fn = MaskedL1Loss(reduction="sum")
    our_loss = loss_fn(predictions, targets, mask=None)

    ref_loss = F.l1_loss(predictions, targets, reduction="sum")

    assert torch.isfinite(our_loss), "MaskedL1Loss(sum) produjo NaN o Inf"
    assert torch.allclose(our_loss, ref_loss, atol=1e-6, rtol=1e-5), (
        f"MaskedL1Loss(sum) = {our_loss.item():.8f} != "
        f"F.l1_loss(sum) = {ref_loss.item():.8f}, "
        f"diff = {(our_loss - ref_loss).abs().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Propiedad 14.4: MultiTaskLoss produce Σ(w_i × loss_i)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=multitask_inputs())
def test_multitask_loss_is_weighted_sum(data: dict) -> None:
    """Feature: vsn-library, Property 14: Losses computadas correctamente y combinables

    **Validates: Requirements 8.1, 8.2**

    Para cualquier conjunto de losses individuales y pesos, MultiTaskLoss SHALL
    producir Σ(w_i × loss_i).
    """
    task_names = data["task_names"]
    weights = data["weights"]
    task_data = data["task_data"]

    # Crear loss functions (todas MSE para esta propiedad)
    losses = {name: MaskedMSELoss(reduction="mean") for name in task_names}

    combiner = MultiTaskLoss(losses=losses, weights=weights)

    # Preparar inputs para el combiner
    inputs = {
        name: (task_data[name][0], task_data[name][1], None)
        for name in task_names
    }

    total_loss, individual_losses = combiner(inputs)

    # Calcular la referencia manual: Σ(w_i × loss_i)
    expected_total = torch.tensor(0.0)
    for name in task_names:
        preds, tgts = task_data[name]
        ref_loss = F.mse_loss(preds, tgts, reduction="mean")
        expected_total = expected_total + weights[name] * ref_loss

    # 1. Verificar que total_loss es finito
    assert torch.isfinite(total_loss), "MultiTaskLoss produjo NaN o Inf"

    # 2. Verificar que total_loss ≈ Σ(w_i × loss_i)
    assert torch.allclose(total_loss, expected_total, atol=1e-5, rtol=1e-4), (
        f"MultiTaskLoss total = {total_loss.item():.8f} != "
        f"Σ(w_i × loss_i) = {expected_total.item():.8f}, "
        f"diff = {(total_loss - expected_total).abs().item():.2e}"
    )

    # 3. Verificar que cada loss individual coincide con la referencia
    for name in task_names:
        preds, tgts = task_data[name]
        ref_loss = F.mse_loss(preds, tgts, reduction="mean")
        assert torch.allclose(individual_losses[name], ref_loss, atol=1e-6, rtol=1e-5), (
            f"MultiTaskLoss individual['{name}'] = {individual_losses[name].item():.8f} != "
            f"referencia = {ref_loss.item():.8f}"
        )
