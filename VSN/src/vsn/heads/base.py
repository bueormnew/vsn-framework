"""Base head, utilidades de agregación y registry para Output Heads O.

Define:
- BaseHead: ABC para todos los heads de salida. Sigue la descomposición:
  aggregation A → projection (W_O, b_O) → task function.
- Funciones de agregación: last_token, mean_pool, max_pool, cls_token.
- Registry de heads para extensibilidad: HEAD_REGISTRY, register_head, build_head.

Los heads reciben decoder state como READ-ONLY y producen ModelOutputs
sin efectos secundarios sobre el estado interno del modelo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from torch import Tensor, nn

from vsn.contracts.multimodal import MultimodalBatch
from vsn.contracts.outputs import ModelOutputs


# ---------------------------------------------------------------------------
# Funciones de agregación
# ---------------------------------------------------------------------------


def last_token(states: List[Tensor]) -> Tensor:
    """Toma el último window y promedia sobre dimensiones espaciales (Y, Z).

    Args:
        states: Lista de tensores, uno por ventana DGW.
            Cada tensor tiene shape (batch, Y, Z, d).

    Returns:
        Tensor de shape (batch, d) — mean-pool espacial del último window.
    """
    last = states[-1]  # (batch, Y, Z, d)
    # Mean-pool sobre Y (dim=1) y Z (dim=2)
    return last.mean(dim=(1, 2))  # (batch, d)


def mean_pool(states: List[Tensor]) -> Tensor:
    """Promedia sobre todos los windows y dimensiones espaciales.

    Args:
        states: Lista de tensores, uno por ventana DGW.
            Cada tensor tiene shape (batch, Y, Z, d).

    Returns:
        Tensor de shape (batch, d) — mean global.
    """
    # Apilar todos los windows: (num_windows, batch, Y, Z, d)
    stacked = torch.stack(states, dim=0)
    # Mean sobre windows (0), Y (2), Z (3)
    return stacked.mean(dim=(0, 2, 3))  # (batch, d)


def max_pool(states: List[Tensor]) -> Tensor:
    """Max-pool sobre dimensiones espaciales del último window.

    Args:
        states: Lista de tensores, uno por ventana DGW.
            Cada tensor tiene shape (batch, Y, Z, d).

    Returns:
        Tensor de shape (batch, d) — max-pool espacial del último window.
    """
    last = states[-1]  # (batch, Y, Z, d)
    # Reshape a (batch, Y*Z, d) para hacer max sobre la dimensión espacial
    batch, Y, Z, d = last.shape
    flat = last.reshape(batch, Y * Z, d)
    # Max sobre posiciones espaciales (dim=1)
    return flat.max(dim=1).values  # (batch, d)


def cls_token(states: List[Tensor]) -> Tensor:
    """Toma la posición [0, 0] del último window como representación CLS.

    Args:
        states: Lista de tensores, uno por ventana DGW.
            Cada tensor tiene shape (batch, Y, Z, d).

    Returns:
        Tensor de shape (batch, d) — posición [0,0] del último window.
    """
    last = states[-1]  # (batch, Y, Z, d)
    return last[:, 0, 0, :]  # (batch, d)


# Mapa de nombre → función de agregación
AGGREGATION_REGISTRY: Dict[str, Callable[[List[Tensor]], Tensor]] = {
    "last_token": last_token,
    "mean_pool": mean_pool,
    "max_pool": max_pool,
    "cls_token": cls_token,
}


def build_aggregation(name: str) -> Callable[[List[Tensor]], Tensor]:
    """Obtiene una función de agregación por nombre.

    Args:
        name: Nombre de la agregación ('last_token', 'mean_pool', 'max_pool', 'cls_token').

    Returns:
        Función de agregación que mapea List[Tensor] → Tensor.

    Raises:
        KeyError: Si el nombre no está registrado.
    """
    if name not in AGGREGATION_REGISTRY:
        available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
        raise KeyError(
            f"Aggregation '{name}' not found. Available: {available}"
        )
    return AGGREGATION_REGISTRY[name]


# ---------------------------------------------------------------------------
# BaseHead ABC
# ---------------------------------------------------------------------------


class BaseHead(nn.Module, ABC):
    """Clase base abstracta para todos los Output Heads O.

    Cada head sigue la descomposición:
        aggregation A → projection (W_O, b_O) → task-specific output function

    Los heads:
    - Reciben decoder state como READ-ONLY (no modifican el tensor).
    - No participan en la propagación del eje X.
    - Producen ModelOutputs sin efectos secundarios.
    """

    @abstractmethod
    def forward(
        self,
        decoder_states: List[Tensor],
        batch: MultimodalBatch,
        metadata: Dict[str, Any],
    ) -> ModelOutputs:
        """Procesa estados del decoder y produce salidas tipadas.

        Args:
            decoder_states: Lista de tensores del decoder (uno por ventana DGW).
                Cada tensor tiene shape (batch, Y, Z, d).
            batch: Batch multimodal con targets y metadata de tarea.
            metadata: Metadatos adicionales (e.g. mode, step).

        Returns:
            ModelOutputs con los campos relevantes para la tarea del head.
        """
        ...


# ---------------------------------------------------------------------------
# Registry de heads
# ---------------------------------------------------------------------------

# Registro global: nombre → clase de head
HEAD_REGISTRY: Dict[str, Type[BaseHead]] = {}


def register_head(name: str) -> Callable[[Type[BaseHead]], Type[BaseHead]]:
    """Decorador para registrar un head en el registry global.

    Uso:
        @register_head("text")
        class TextHead(BaseHead):
            ...

    Args:
        name: Nombre único para identificar el head en configuraciones.

    Returns:
        Decorador que registra la clase y la retorna sin modificar.

    Raises:
        ValueError: Si el nombre ya está registrado.
        TypeError: Si la clase no hereda de BaseHead.
    """

    def decorator(cls: Type[BaseHead]) -> Type[BaseHead]:
        if name in HEAD_REGISTRY:
            raise ValueError(
                f"Head '{name}' already registered by {HEAD_REGISTRY[name].__name__}. "
                f"Cannot register {cls.__name__}."
            )
        if not (isinstance(cls, type) and issubclass(cls, BaseHead)):
            raise TypeError(
                f"Cannot register {cls}: must be a subclass of BaseHead."
            )
        HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def build_head(name: str, config: Any) -> BaseHead:
    """Instancia un head registrado a partir de su nombre y configuración.

    Args:
        name: Nombre del head registrado (e.g. 'text', 'classification').
        config: Objeto de configuración que el head usará para inicializarse.
            Típicamente un VSNConfig o similar con los campos necesarios.

    Returns:
        Instancia del head registrado.

    Raises:
        KeyError: Si el nombre no está en el registry.
    """
    if name not in HEAD_REGISTRY:
        available = ", ".join(sorted(HEAD_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Head '{name}' not found in registry. Available: {available}"
        )
    head_cls = HEAD_REGISTRY[name]
    return head_cls(config)
