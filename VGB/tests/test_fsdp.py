"""Tests for FSDP2 integration (vgb.runtime.fsdp).

Verifies import, configuration dataclass, and error handling
when distributed is not initialized (CPU-only testing).

Validates: Requirements 10.1, 10.5
"""

import torch
import torch.nn as nn
import pytest

from vgb.runtime.fsdp import (
    FSDP2Config,
    apply_fsdp2,
    _check_distributed_available,
    _find_shardable_blocks,
)


class FakeVGBBlock(nn.Module):
    """A minimal block simulating a VGB block."""

    def __init__(self, d: int = 8):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FakeEncoder(nn.Module):
    def __init__(self, num_blocks: int = 3, d: int = 8):
        super().__init__()
        self.vgb_blocks = nn.ModuleList(
            [FakeVGBBlock(d) for _ in range(num_blocks)]
        )


class FakeDecoder(nn.Module):
    def __init__(self, num_blocks: int = 2, d: int = 8):
        super().__init__()
        self.vgb_blocks = nn.ModuleList(
            [FakeVGBBlock(d) for _ in range(num_blocks)]
        )


class FakeModel(nn.Module):
    def __init__(self, enc_blocks: int = 3, dec_blocks: int = 2, d: int = 8):
        super().__init__()
        self.encoder = FakeEncoder(enc_blocks, d)
        self.decoder = FakeDecoder(dec_blocks, d)


class TestFSDP2Config:
    """Tests for FSDP2Config dataclass."""

    def test_default_values(self):
        config = FSDP2Config()
        assert config.precision == "bf16"
        assert config.reshard_after_forward is True
        assert config.prefetch_num == 1

    def test_custom_values(self):
        config = FSDP2Config(
            precision="fp16",
            reshard_after_forward=False,
            prefetch_num=2,
        )
        assert config.precision == "fp16"
        assert config.reshard_after_forward is False
        assert config.prefetch_num == 2


class TestCheckDistributedAvailable:
    """Tests for _check_distributed_available."""

    def test_raises_when_not_initialized(self):
        """Should raise RuntimeError when distributed is not initialized."""
        # In CI/local without torchrun, distributed is not initialized
        if not torch.distributed.is_initialized():
            with pytest.raises(RuntimeError, match="not initialized"):
                _check_distributed_available()


class TestFindShardableBlocks:
    """Tests for _find_shardable_blocks helper."""

    def test_finds_all_vgb_blocks(self):
        model = FakeModel(enc_blocks=3, dec_blocks=2)
        blocks = _find_shardable_blocks(model)
        assert len(blocks) == 5  # 3 encoder + 2 decoder

    def test_encoder_only_model(self):
        model = nn.Module()
        model.encoder = FakeEncoder(num_blocks=4)
        blocks = _find_shardable_blocks(model)
        assert len(blocks) == 4

    def test_model_without_vgb_blocks(self):
        model = nn.Linear(8, 8)
        blocks = _find_shardable_blocks(model)
        assert len(blocks) == 0


class TestApplyFSDP2:
    """Tests for apply_fsdp2 — verifies error when distributed not initialized."""

    def test_raises_without_distributed(self):
        """apply_fsdp2 should raise clear error without torchrun."""
        if not torch.distributed.is_initialized():
            model = FakeModel()
            with pytest.raises(RuntimeError, match="not initialized"):
                apply_fsdp2(model)

    def test_raises_with_custom_config(self):
        """Should raise same error regardless of config."""
        if not torch.distributed.is_initialized():
            model = FakeModel()
            config = FSDP2Config(precision="fp16")
            with pytest.raises(RuntimeError, match="not initialized"):
                apply_fsdp2(model, config)

    def test_accepts_none_config(self):
        """Should use default config when None is passed."""
        if not torch.distributed.is_initialized():
            model = FakeModel()
            with pytest.raises(RuntimeError, match="not initialized"):
                apply_fsdp2(model, None)
