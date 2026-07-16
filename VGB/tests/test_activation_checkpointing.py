"""Tests for activation checkpointing (vgb.runtime.activation_checkpointing).

Verifies import, function signatures, and selective checkpointing on CPU
with a mock model structure that mimics the VGB model layout.

Validates: Requirements 10.3
"""

import torch
import torch.nn as nn
import pytest

from vgb.runtime.activation_checkpointing import (
    CheckpointSegment,
    apply_activation_checkpointing,
    _wrap_forward_with_checkpoint,
    _find_vgb_blocks,
)


class FakeVGBBlock(nn.Module):
    """A minimal block that simulates a VGB block for testing."""

    def __init__(self, d: int = 8):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FakeEncoder(nn.Module):
    """Simulates an encoder with vgb_blocks."""

    def __init__(self, num_blocks: int = 3, d: int = 8):
        super().__init__()
        self.vgb_blocks = nn.ModuleList(
            [FakeVGBBlock(d) for _ in range(num_blocks)]
        )


class FakeDecoder(nn.Module):
    """Simulates a decoder with vgb_blocks."""

    def __init__(self, num_blocks: int = 2, d: int = 8):
        super().__init__()
        self.vgb_blocks = nn.ModuleList(
            [FakeVGBBlock(d) for _ in range(num_blocks)]
        )


class FakeModel(nn.Module):
    """Simulates a model with encoder and decoder containing VGB blocks."""

    def __init__(self, enc_blocks: int = 3, dec_blocks: int = 2, d: int = 8):
        super().__init__()
        self.encoder = FakeEncoder(enc_blocks, d)
        self.decoder = FakeDecoder(dec_blocks, d)


class TestCheckpointSegmentEnum:
    """Tests for CheckpointSegment enum."""

    def test_encoder_value(self):
        assert CheckpointSegment.ENCODER == "encoder"

    def test_decoder_value(self):
        assert CheckpointSegment.DECODER == "decoder"

    def test_both_value(self):
        assert CheckpointSegment.BOTH == "both"

    def test_construction_from_string(self):
        assert CheckpointSegment("encoder") == CheckpointSegment.ENCODER
        assert CheckpointSegment("decoder") == CheckpointSegment.DECODER
        assert CheckpointSegment("both") == CheckpointSegment.BOTH


class TestFindVGBBlocks:
    """Tests for _find_vgb_blocks helper."""

    def test_finds_encoder_blocks(self):
        model = FakeModel(enc_blocks=4, dec_blocks=2)
        blocks = _find_vgb_blocks(model, "encoder")
        assert len(blocks) == 4

    def test_finds_decoder_blocks(self):
        model = FakeModel(enc_blocks=3, dec_blocks=5)
        blocks = _find_vgb_blocks(model, "decoder")
        assert len(blocks) == 5

    def test_missing_segment_returns_empty(self):
        model = nn.Linear(8, 8)  # No encoder/decoder
        blocks = _find_vgb_blocks(model, "encoder")
        assert blocks == []


class TestApplyActivationCheckpointing:
    """Tests for apply_activation_checkpointing."""

    def test_checkpoints_both_segments(self):
        """Should wrap all VGB blocks in encoder + decoder."""
        model = FakeModel(enc_blocks=3, dec_blocks=2)
        n = apply_activation_checkpointing(model, segments="both")
        assert n == 5  # 3 encoder + 2 decoder

    def test_checkpoints_encoder_only(self):
        """Should wrap only encoder blocks."""
        model = FakeModel(enc_blocks=3, dec_blocks=2)
        n = apply_activation_checkpointing(model, segments="encoder")
        assert n == 3

    def test_checkpoints_decoder_only(self):
        """Should wrap only decoder blocks."""
        model = FakeModel(enc_blocks=3, dec_blocks=2)
        n = apply_activation_checkpointing(model, segments="decoder")
        assert n == 2

    def test_with_block_filter(self):
        """Should respect the block_filter callable."""
        model = FakeModel(enc_blocks=4, dec_blocks=0)
        # Only checkpoint even-indexed blocks
        n = apply_activation_checkpointing(
            model,
            segments="encoder",
            block_filter=lambda m, i: i % 2 == 0,
        )
        assert n == 2  # indices 0, 2

    def test_checkpointed_forward_still_produces_output(self):
        """Wrapped forward should still compute correct output."""
        model = FakeModel(enc_blocks=2, dec_blocks=0, d=8)
        apply_activation_checkpointing(model, segments="encoder")

        x = torch.randn(4, 8, requires_grad=True)
        # Run through the first encoder block (now checkpointed)
        out = model.encoder.vgb_blocks[0](x)
        assert out.shape == (4, 8)
        # Gradients should flow through
        out.sum().backward()
        assert x.grad is not None

    def test_returns_zero_for_empty_model(self):
        """Model without vgb_blocks returns 0."""
        model = nn.Linear(8, 8)
        n = apply_activation_checkpointing(model, segments="both")
        assert n == 0

    def test_accepts_enum_value(self):
        """Should accept CheckpointSegment enum directly."""
        model = FakeModel(enc_blocks=2, dec_blocks=1)
        n = apply_activation_checkpointing(
            model, segments=CheckpointSegment.BOTH
        )
        assert n == 3
