"""Tests unitarios para InputCache.

Valida:
- Semántica FIFO estricta con write_ptr
- push() retorna None cuando no está lleno
- push() retorna batch completo cuando alcanza capacidad ICS
- flush() extrae contenido actual y resetea
- reset() limpia el buffer
- Manejo de batch dimensions
- Validación de errores
"""

import torch
import pytest

from vsn.core.input_cache import InputCache


class TestInputCacheInit:
    """Verifica la inicialización correcta del InputCache."""

    def test_buffer_shape(self):
        cache = InputCache(ics=8, d=16, batch_size=2)
        assert cache.buffer.shape == (2, 8, 16)

    def test_buffer_initialized_to_zeros(self):
        cache = InputCache(ics=4, d=8)
        assert torch.all(cache.buffer == 0)

    def test_write_ptr_starts_at_zero(self):
        cache = InputCache(ics=4, d=8)
        assert cache.write_ptr.item() == 0

    def test_invalid_ics_raises(self):
        with pytest.raises(ValueError, match="ics debe ser positivo"):
            InputCache(ics=0, d=8)

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError, match="d debe ser positivo"):
            InputCache(ics=4, d=0)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size debe ser positivo"):
            InputCache(ics=4, d=8, batch_size=0)

    def test_buffers_are_registered(self):
        cache = InputCache(ics=4, d=8)
        buffers = dict(cache.named_buffers())
        assert "buffer" in buffers
        assert "write_ptr" in buffers


class TestInputCachePush:
    """Verifica la operación push con semántica FIFO."""

    def test_push_returns_none_when_not_full(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        tokens = torch.randn(1, 2, 8)
        result = cache.push(tokens)
        assert result is None

    def test_push_updates_write_ptr(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        tokens = torch.randn(1, 2, 8)
        cache.push(tokens)
        assert cache.write_ptr.item() == 2

    def test_push_returns_batch_when_full(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        tokens = torch.randn(1, 4, 8)
        result = cache.push(tokens)
        assert result is not None
        assert result.shape == (1, 4, 8)

    def test_push_multiple_fills_buffer(self):
        """Múltiples push que llenan el buffer."""
        cache = InputCache(ics=4, d=8, batch_size=1)
        t1 = torch.randn(1, 2, 8)
        t2 = torch.randn(1, 2, 8)

        result1 = cache.push(t1)
        assert result1 is None

        result2 = cache.push(t2)
        assert result2 is not None
        assert result2.shape == (1, 4, 8)

    def test_push_preserves_fifo_order(self):
        """Los tokens se almacenan en el orden en que llegan (FIFO)."""
        cache = InputCache(ics=4, d=2, batch_size=1)

        # Tokens distinguibles
        t1 = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])  # (1, 2, 2)
        t2 = torch.tensor([[[3.0, 0.0], [4.0, 0.0]]])  # (1, 2, 2)

        cache.push(t1)
        result = cache.push(t2)

        # FIFO: t1 primero, t2 después
        assert torch.allclose(result[0, 0], torch.tensor([1.0, 0.0]))
        assert torch.allclose(result[0, 1], torch.tensor([2.0, 0.0]))
        assert torch.allclose(result[0, 2], torch.tensor([3.0, 0.0]))
        assert torch.allclose(result[0, 3], torch.tensor([4.0, 0.0]))

    def test_push_resets_after_full(self):
        """Después de retornar batch completo, el buffer se resetea."""
        cache = InputCache(ics=2, d=4, batch_size=1)
        tokens = torch.randn(1, 2, 4)
        cache.push(tokens)

        # Buffer debería estar vacío después de entregar
        assert cache.write_ptr.item() == 0
        assert cache.is_empty

    def test_push_2d_input_auto_expands(self):
        """Tokens 2D se expanden automáticamente a (1, num_tokens, d)."""
        cache = InputCache(ics=4, d=8, batch_size=1)
        tokens = torch.randn(2, 8)  # (num_tokens, d)
        result = cache.push(tokens)
        assert result is None
        assert cache.write_ptr.item() == 2

    def test_push_wrong_d_raises(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        tokens = torch.randn(1, 2, 16)  # d=16 != 8
        with pytest.raises(ValueError, match="Dimensión de tokens"):
            cache.push(tokens)

    def test_push_overflow_raises(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        cache.push(torch.randn(1, 3, 8))
        with pytest.raises(ValueError, match="excede espacio disponible"):
            cache.push(torch.randn(1, 2, 8))  # Solo queda 1

    def test_push_wrong_batch_size_raises(self):
        cache = InputCache(ics=4, d=8, batch_size=2)
        tokens = torch.randn(3, 2, 8)  # batch=3 != 2
        with pytest.raises(ValueError, match="Batch size"):
            cache.push(tokens)


class TestInputCacheFlush:
    """Verifica la operación flush."""

    def test_flush_empty_returns_zero_length(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        result = cache.flush()
        assert result.shape == (1, 0, 8)

    def test_flush_partial_returns_stored_tokens(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        tokens = torch.randn(1, 2, 8)
        cache.push(tokens)
        result = cache.flush()
        assert result.shape == (1, 2, 8)
        assert torch.allclose(result, tokens)

    def test_flush_resets_buffer(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        cache.push(torch.randn(1, 2, 8))
        cache.flush()
        assert cache.write_ptr.item() == 0
        assert cache.is_empty

    def test_flush_preserves_fifo_order(self):
        """flush() preserva el orden FIFO."""
        cache = InputCache(ics=6, d=2, batch_size=1)
        t1 = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        t2 = torch.tensor([[[3.0, 0.0]]])

        cache.push(t1)
        cache.push(t2)
        result = cache.flush()

        assert result.shape == (1, 3, 2)
        assert torch.allclose(result[0, 0], torch.tensor([1.0, 0.0]))
        assert torch.allclose(result[0, 1], torch.tensor([2.0, 0.0]))
        assert torch.allclose(result[0, 2], torch.tensor([3.0, 0.0]))


class TestInputCacheReset:
    """Verifica la operación reset."""

    def test_reset_clears_buffer(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        cache.push(torch.randn(1, 3, 8))
        cache.reset()
        assert cache.write_ptr.item() == 0
        assert torch.all(cache.buffer == 0)

    def test_reset_allows_reuse(self):
        """Después de reset, se pueden insertar nuevos tokens."""
        cache = InputCache(ics=4, d=8, batch_size=1)
        cache.push(torch.randn(1, 3, 8))
        cache.reset()

        tokens = torch.randn(1, 4, 8)
        result = cache.push(tokens)
        assert result is not None


class TestInputCacheProperties:
    """Verifica propiedades auxiliares."""

    def test_occupancy(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        assert cache.occupancy == 0
        cache.push(torch.randn(1, 2, 8))
        assert cache.occupancy == 2

    def test_is_full(self):
        cache = InputCache(ics=2, d=8, batch_size=1)
        assert not cache.is_full
        # After full push, cache resets so it's empty again
        cache.push(torch.randn(1, 2, 8))
        assert not cache.is_full  # After full, it resets

    def test_is_empty(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        assert cache.is_empty
        cache.push(torch.randn(1, 1, 8))
        assert not cache.is_empty

    def test_available_space(self):
        cache = InputCache(ics=4, d=8, batch_size=1)
        assert cache.available_space == 4
        cache.push(torch.randn(1, 3, 8))
        assert cache.available_space == 1

    def test_repr(self):
        cache = InputCache(ics=4, d=8, batch_size=2)
        s = repr(cache)
        assert "ics=4" in s
        assert "d=8" in s
        assert "batch_size=2" in s


class TestInputCacheBatched:
    """Verifica comportamiento con batch_size > 1."""

    def test_batched_push_fills_all_batches(self):
        cache = InputCache(ics=3, d=4, batch_size=2)
        tokens = torch.randn(2, 3, 4)
        result = cache.push(tokens)
        assert result is not None
        assert result.shape == (2, 3, 4)

    def test_batched_flush_partial(self):
        cache = InputCache(ics=4, d=4, batch_size=3)
        tokens = torch.randn(3, 2, 4)
        cache.push(tokens)
        result = cache.flush()
        assert result.shape == (3, 2, 4)

    def test_batched_preserves_batch_independence(self):
        """Cada batch es independiente pero comparten write_ptr."""
        cache = InputCache(ics=2, d=2, batch_size=2)
        t = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],  # batch 0
            [[5.0, 6.0], [7.0, 8.0]],  # batch 1
        ])
        result = cache.push(t)
        assert result is not None
        assert torch.allclose(result[0, 0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[1, 0], torch.tensor([5.0, 6.0]))


class TestInputCacheNNModule:
    """Verifica la integración como nn.Module."""

    def test_state_dict_includes_buffers(self):
        cache = InputCache(ics=4, d=8)
        state = cache.state_dict()
        assert "buffer" in state
        assert "write_ptr" in state

    def test_load_state_dict(self):
        """Round-trip: save → load preserva estado."""
        cache1 = InputCache(ics=4, d=8, batch_size=1)
        cache1.push(torch.randn(1, 2, 8))
        state = cache1.state_dict()

        cache2 = InputCache(ics=4, d=8, batch_size=1)
        cache2.load_state_dict(state)

        assert cache2.write_ptr.item() == cache1.write_ptr.item()
        assert torch.allclose(cache2.buffer, cache1.buffer)

    def test_to_device(self):
        """El cache se puede mover a otro device (test con CPU)."""
        cache = InputCache(ics=4, d=8)
        cache = cache.to("cpu")
        assert cache.buffer.device.type == "cpu"
