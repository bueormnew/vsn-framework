"""Tests for vgb.runtime.bootstrap module.

Verifies bootstrap works in single-GPU mode (no torchrun env vars)
and that distributed environment detection works correctly.
"""

import os
import random
from unittest.mock import patch

import torch
import pytest

from vgb.config.schema import RuntimeConfig
from vgb.runtime.bootstrap import (
    RuntimeContext,
    bootstrap,
    _set_seeds,
    _detect_distributed_env,
    _select_device,
    _configure_logging,
)


class TestBootstrapSingleGPU:
    """Tests for single-GPU mode (no torchrun env vars)."""

    def test_bootstrap_returns_runtime_context(self):
        """bootstrap() should return a RuntimeContext dataclass."""
        config = RuntimeConfig(seed=123)
        ctx = bootstrap(config)
        assert isinstance(ctx, RuntimeContext)

    def test_bootstrap_single_gpu_defaults(self):
        """In single-GPU mode, rank=0, local_rank=0, world_size=1."""
        # Ensure no distributed env vars are set
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig(seed=42)
            ctx = bootstrap(config)

        assert ctx.rank == 0
        assert ctx.local_rank == 0
        assert ctx.world_size == 1
        assert ctx.is_main is True
        assert ctx.distributed is False

    def test_bootstrap_device_is_valid(self):
        """Device should be cuda:0 if CUDA available, else cpu."""
        config = RuntimeConfig(seed=7)
        ctx = bootstrap(config)

        if torch.cuda.is_available():
            assert ctx.device == torch.device("cuda:0")
        else:
            assert ctx.device == torch.device("cpu")

    def test_bootstrap_sets_torch_seed(self):
        """bootstrap should set torch seed for reproducibility."""
        config = RuntimeConfig(seed=99)
        bootstrap(config)

        # Generate a tensor — should be deterministic after seeding
        t1 = torch.randn(5)

        # Re-seed and generate again
        torch.manual_seed(99)
        t2 = torch.randn(5)

        assert torch.equal(t1, t2)

    def test_bootstrap_sets_random_seed(self):
        """bootstrap should set Python random seed."""
        config = RuntimeConfig(seed=55)
        bootstrap(config)
        val1 = random.random()

        random.seed(55)
        val2 = random.random()

        assert val1 == val2


class TestSetSeeds:
    """Tests for _set_seeds helper."""

    def test_set_seeds_deterministic_torch(self):
        """Torch random should produce same values after same seed."""
        _set_seeds(42)
        a = torch.randn(10)
        _set_seeds(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_set_seeds_deterministic_random(self):
        """Python random should produce same values after same seed."""
        _set_seeds(42)
        a = random.random()
        _set_seeds(42)
        b = random.random()
        assert a == b

    def test_set_seeds_numpy_if_available(self):
        """If numpy is available, it should also be seeded."""
        try:
            import numpy as np

            _set_seeds(77)
            a = np.random.random()
            _set_seeds(77)
            b = np.random.random()
            assert a == b
        except ImportError:
            pytest.skip("numpy not available")


class TestDetectDistributedEnv:
    """Tests for _detect_distributed_env helper."""

    def test_no_env_vars_returns_defaults(self):
        """Without RANK/LOCAL_RANK/WORLD_SIZE, returns (0, 0, 1)."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        with patch.dict(os.environ, env, clear=True):
            rank, local_rank, world_size = _detect_distributed_env()

        assert rank == 0
        assert local_rank == 0
        assert world_size == 1

    def test_with_env_vars_detects_distributed(self):
        """With torchrun env vars set, should detect the distributed context."""
        env_patch = {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
        with patch.dict(os.environ, env_patch):
            rank, local_rank, world_size = _detect_distributed_env()

        assert rank == 2
        assert local_rank == 1
        assert world_size == 4


class TestSelectDevice:
    """Tests for _select_device helper."""

    def test_cpu_when_no_cuda(self):
        """When CUDA is not available, should return cpu device."""
        with patch("torch.cuda.is_available", return_value=False):
            device = _select_device(0)
        assert device == torch.device("cpu")

    def test_cuda_device_with_local_rank(self):
        """When CUDA is available, should return cuda:{local_rank}."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = _select_device(0)
        assert device == torch.device("cuda:0")


class TestConfigureLogging:
    """Tests for _configure_logging helper."""

    def test_main_process_info_level(self):
        """Main process should have INFO level logging."""
        import logging

        _configure_logging(is_main=True)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_non_main_process_warning_level(self):
        """Non-main processes should have WARNING level logging."""
        import logging

        _configure_logging(is_main=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING
