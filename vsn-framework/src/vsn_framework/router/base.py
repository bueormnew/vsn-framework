"""Base DataRouter — clase base para todos los enrutadores de datos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


class DataRouter(ABC):
    """Clase base abstracta para enrutadores de datos.
    
    Transforma datos crudos en tensores listos para entrenamiento VSN.
    Cada router implementa la lógica específica de su dominio (tokenización,
    padding, formateo, etc.)
    """
    
    @abstractmethod
    def prepare(self, data: List[Any], **kwargs) -> Tuple[Tensor, Tensor]:
        """Prepara datos crudos para entrenamiento.
        
        Args:
            data: Lista de datos crudos (formato depende del router).
            
        Returns:
            Tuple (inputs, targets) listos para model.fit() o DataLoader.
        """
        ...
    
    @abstractmethod
    def encode(self, sample: Any) -> Tensor:
        """Codifica un solo sample para inferencia."""
        ...
    
    @abstractmethod
    def decode(self, tensor: Tensor) -> Any:
        """Decodifica un tensor de output a formato legible."""
        ...
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Tamaño del vocabulario del router."""
        ...
    
    @property
    @abstractmethod
    def max_seq_len(self) -> int:
        """Longitud máxima de secuencia."""
        ...
    
    @staticmethod
    def for_task(task: str, **kwargs) -> "DataRouter":
        """Factory method para crear un router según la tarea.
        
        Args:
            task: Nombre de la tarea ("arithmetic", "text").
            **kwargs: Argumentos específicos del router.
            
        Returns:
            Instancia del router apropiado.
        """
        from vsn_framework.router.arithmetic import ArithmeticRouter
        from vsn_framework.router.text import TextRouter
        
        routers = {
            "arithmetic": ArithmeticRouter,
            "math": ArithmeticRouter,
            "text": TextRouter,
        }
        
        if task not in routers:
            raise ValueError(f"Task '{task}' no soportada. Opciones: {list(routers.keys())}")
        
        return routers[task](**kwargs)
