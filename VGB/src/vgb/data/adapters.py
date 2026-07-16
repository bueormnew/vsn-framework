"""Token adapter for converting external tokenized batches to VSN model format.

Converts externally tokenized sequences (token IDs or pre-computed embeddings)
into the expected (batch, num_tokens, d) tensor format for the VSN Input Cache.

Validates: Requirements 11.1 (data pipeline), 12.3 (input validation)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class TokenAdapter(nn.Module):
    """Adapts external token batches to VSN model input format.

    Handles two input modalities:
    - Token IDs: integer tensor → looked up in an embedding table → (batch, seq, d)
    - Pre-computed embeddings: float tensor already in (batch, seq, d) shape

    Sequences longer than max_tokens are truncated; shorter are left as-is
    (padding is handled by the collate function).

    Args:
        d: Model embedding dimension.
        max_tokens: Maximum number of tokens per sequence.
        vocab_size: Vocabulary size for embedding lookup (required if using token IDs).
    """

    def __init__(self, d: int, max_tokens: int, vocab_size: Optional[int] = None):
        super().__init__()
        self.d = d
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d)
        else:
            self.embedding = None

    def adapt(
        self,
        token_ids: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Convert token IDs or embeddings to model input format.

        Exactly one of token_ids or embeddings must be provided.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len) with token indices.
            embeddings: Float tensor of shape (batch, seq_len, d) with pre-computed embeddings.

        Returns:
            Tensor of shape (batch, min(seq_len, max_tokens), d).

        Raises:
            ValueError: If neither or both inputs are provided, or shapes are invalid.
        """
        if token_ids is None and embeddings is None:
            raise ValueError("Either token_ids or embeddings must be provided.")
        if token_ids is not None and embeddings is not None:
            raise ValueError(
                "Only one of token_ids or embeddings should be provided, not both."
            )

        if token_ids is not None:
            if self.embedding is None:
                raise ValueError(
                    "TokenAdapter was created without vocab_size; "
                    "cannot process token_ids. Provide vocab_size at init "
                    "or pass embeddings directly."
                )
            if token_ids.ndim != 2:
                raise ValueError(
                    f"token_ids must be 2D (batch, seq_len), got shape {token_ids.shape}"
                )
            # Truncate to max_tokens
            token_ids = token_ids[:, : self.max_tokens]
            result = self.embedding(token_ids)  # (batch, seq, d)
        else:
            assert embeddings is not None
            if embeddings.ndim != 3:
                raise ValueError(
                    f"embeddings must be 3D (batch, seq_len, d), got shape {embeddings.shape}"
                )
            if embeddings.shape[-1] != self.d:
                raise ValueError(
                    f"embeddings last dim must be {self.d}, got {embeddings.shape[-1]}"
                )
            # Truncate to max_tokens
            result = embeddings[:, : self.max_tokens, :]

        return result

    def forward(
        self,
        token_ids: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass — alias for adapt().

        Allows TokenAdapter to be used as an nn.Module in a pipeline.
        """
        return self.adapt(token_ids=token_ids, embeddings=embeddings)
