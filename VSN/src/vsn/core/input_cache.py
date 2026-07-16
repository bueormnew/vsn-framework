"""Input Cache — Buffer FIFO temporal para el encoder VSN.

Implementación de la Definición 5.1 de la especificación formal:

    C es una cola FIFO de tamaño configurable ICS.
    Los embeddings e_j se almacenan en orden de llegada.
    La extracción respeta el mismo orden (first-in, first-out).
    L' = min(L, ICS) — acotado por la capacidad ICS.
    Cuando el buffer alcanza capacidad ICS, inyecta su contenido al encoder.

La InputCache opera con semántica batched:
    - Buffer shape: (batch, ICS, d)
    - push(tokens) acepta tokens de shape (batch, num_tokens, d)
    - Retorna el batch completo cuando write_ptr alcanza ICS
    - flush() extrae el contenido actual sin importar si está lleno
    - reset() limpia el buffer para una nueva secuencia
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class InputCache(nn.Module):
    """Buffer FIFO de tamaño ICS con semántica batched.

    Almacena tokens/embeddings secuencialmente hasta alcanzar capacidad,
    luego entrega el batch completo al encoder para procesamiento.

    Args:
        ics: Capacidad máxima del buffer (Input Cache Size).
        d: Dimensión del embedding por token.
        batch_size: Tamaño del batch. Default 1.
    """

    def __init__(self, ics: int, d: int, batch_size: int = 1) -> None:
        super().__init__()

        if ics <= 0:
            raise ValueError(f"ics debe ser positivo, recibido: {ics}")
        if d <= 0:
            raise ValueError(f"d debe ser positivo, recibido: {d}")
        if batch_size <= 0:
            raise ValueError(f"batch_size debe ser positivo, recibido: {batch_size}")

        self.ics = ics
        self.d = d
        self.batch_size = batch_size

        # Buffer principal: almacena tokens en orden FIFO
        self.register_buffer(
            "buffer", torch.zeros(batch_size, ics, d)
        )
        # Puntero de escritura: indica la siguiente posición libre
        self.register_buffer(
            "write_ptr", torch.tensor(0, dtype=torch.long)
        )

    def push(self, tokens: Tensor) -> Optional[Tensor]:
        """Añade tokens al buffer FIFO. Retorna batch completo cuando se llena.

        Args:
            tokens: Tensor de shape (batch, num_tokens, d) o (num_tokens, d).
                    Si se pasa (num_tokens, d), se expande a (1, num_tokens, d).

        Returns:
            Tensor de shape (batch, ICS, d) cuando el buffer alcanza capacidad ICS.
            None si el buffer aún no está lleno después de la inserción.

        Raises:
            ValueError: Si num_tokens excede el espacio disponible en el buffer.
            ValueError: Si la dimensión d no coincide con la configurada.
        """
        # Normalizar a 3D: (batch, num_tokens, d)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)

        if tokens.ndim != 3:
            raise ValueError(
                f"tokens debe ser 2D o 3D, recibido shape: {tokens.shape}"
            )

        batch, num_tokens, dim = tokens.shape

        if dim != self.d:
            raise ValueError(
                f"Dimensión de tokens ({dim}) no coincide con d={self.d}"
            )
        if batch != self.batch_size:
            raise ValueError(
                f"Batch size de tokens ({batch}) no coincide con "
                f"batch_size configurado ({self.batch_size})"
            )

        current_ptr = self.write_ptr.item()
        available = self.ics - current_ptr

        if num_tokens > available:
            raise ValueError(
                f"num_tokens ({num_tokens}) excede espacio disponible "
                f"({available}). write_ptr={current_ptr}, ics={self.ics}"
            )

        # Escritura FIFO: insertar tokens en posiciones [write_ptr, write_ptr + num_tokens)
        end_ptr = current_ptr + num_tokens
        self.buffer[:, current_ptr:end_ptr, :] = tokens
        self.write_ptr.fill_(end_ptr)

        # Si el buffer alcanzó capacidad, retornar el contenido completo
        if end_ptr == self.ics:
            result = self.buffer.clone()
            self._reset_state()
            return result

        return None

    def flush(self) -> Tensor:
        """Extrae el contenido actual del buffer (parcial o completo).

        Retorna los tokens almacenados hasta write_ptr y resetea el buffer.
        Si el buffer está vacío, retorna un tensor de shape (batch, 0, d).

        Returns:
            Tensor de shape (batch, L', d) donde L' = write_ptr actual.
        """
        current_ptr = self.write_ptr.item()
        if current_ptr == 0:
            result = self.buffer[:, :0, :]  # (batch, 0, d)
        else:
            result = self.buffer[:, :current_ptr, :].clone()
        self._reset_state()
        return result

    def reset(self) -> None:
        """Resetea el buffer para una nueva secuencia.

        Limpia el contenido y resetea el puntero de escritura a 0.
        """
        self._reset_state()

    def _reset_state(self) -> None:
        """Helper interno para resetear buffer y puntero."""
        self.buffer.zero_()
        self.write_ptr.fill_(0)

    @property
    def occupancy(self) -> int:
        """Número de tokens actualmente almacenados en el buffer."""
        return self.write_ptr.item()

    @property
    def is_full(self) -> bool:
        """True si el buffer alcanzó capacidad ICS."""
        return self.write_ptr.item() == self.ics

    @property
    def is_empty(self) -> bool:
        """True si el buffer está vacío."""
        return self.write_ptr.item() == 0

    @property
    def available_space(self) -> int:
        """Espacio disponible en el buffer."""
        return self.ics - self.write_ptr.item()

    def __repr__(self) -> str:
        return (
            f"InputCache(ics={self.ics}, d={self.d}, "
            f"batch_size={self.batch_size}, "
            f"occupancy={self.occupancy}/{self.ics})"
        )
