"""Collate function for VSN DataLoaders.

Pads variable-length sequences to the same length within a batch
and creates attention masks for padded positions.

Validates: Requirements 11.1 (data pipeline)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def vsn_collate_fn(
    batch: List[Dict[str, Tensor]],
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Collate function for VSN DataLoaders.

    Pads variable-length sequences to the longest sequence in the batch
    and creates a boolean mask indicating valid (non-padded) positions.

    Each sample in the batch is a dict with:
    - "tokens": Tensor of shape (seq_len, d) — token embeddings
    - "targets" (optional): Tensor of shape (seq_len,) or (seq_len, d) — targets

    Args:
        batch: List of sample dicts from the dataset.

    Returns:
        Tuple of:
        - padded_tokens: (batch_size, max_seq_len, d) — zero-padded embeddings
        - targets_or_none: (batch_size, max_seq_len, ...) — padded targets, or None
        - mask: (batch_size, max_seq_len) — bool mask, True for valid positions
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    # Extract tokens and determine dimensions
    tokens_list = [sample["tokens"] for sample in batch]
    d = tokens_list[0].shape[-1]
    seq_lens = [t.shape[0] for t in tokens_list]
    max_seq_len = max(seq_lens)
    batch_size = len(batch)

    # Pad tokens to max_seq_len
    padded_tokens = torch.zeros(batch_size, max_seq_len, d, dtype=tokens_list[0].dtype)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    for i, (tokens, length) in enumerate(zip(tokens_list, seq_lens)):
        padded_tokens[i, :length, :] = tokens
        mask[i, :length] = True

    # Handle targets (optional)
    has_targets = "targets" in batch[0] and batch[0]["targets"] is not None
    targets: Optional[Tensor] = None

    if has_targets:
        targets_list = [sample["targets"] for sample in batch]
        target_shape = targets_list[0].shape[1:] if targets_list[0].ndim > 1 else ()
        target_dtype = targets_list[0].dtype

        if target_shape:
            # Multi-dimensional targets (seq_len, ...)
            padded_targets = torch.zeros(
                batch_size, max_seq_len, *target_shape, dtype=target_dtype
            )
        else:
            # Scalar targets per position (seq_len,)
            padded_targets = torch.zeros(
                batch_size, max_seq_len, dtype=target_dtype
            )

        for i, (tgt, length) in enumerate(zip(targets_list, seq_lens)):
            if target_shape:
                padded_targets[i, :length, ...] = tgt
            else:
                padded_targets[i, :length] = tgt

        targets = padded_targets

    return padded_tokens, targets, mask
