"""Tests unitarios para VSNDecoder.

Verificaciones:
1. Output shapes correctos para ventana única y multi-ventana
2. Gradientes fluyen a través del decoder completo
3. Ψ se invoca correctamente entre ventanas
4. Parámetros θ^dec son independientes
5. Validación de entradas
"""

import pytest
import torch

from vsn.core.decoder import VSNDecoder


class TestVSNDecoderShape:
    """Verificar que el decoder produce shapes correctas."""

    def test_single_window_output_shape(self):
        """Output shape es lista de un tensor (batch, Y, Z, d) con 1 ventana."""
        X_dec, Y, Z, d, dgw = 4, 3, 3, 16, 4
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        V_dec_0 = torch.randn(batch, Y, Z, d)

        outputs = decoder(V_dec_0, num_windows=1)

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0].shape == (batch, Y, Z, d), (
            f"Expected shape ({batch}, {Y}, {Z}, {d}), got {outputs[0].shape}"
        )

    def test_multi_window_output_shape(self):
        """Output tiene tantos tensores como ventanas solicitadas."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        num_windows = 4
        V_dec_0 = torch.randn(batch, Y, Z, d)

        outputs = decoder(V_dec_0, num_windows=num_windows)

        assert len(outputs) == num_windows
        for k, out in enumerate(outputs):
            assert out.shape == (batch, Y, Z, d), (
                f"Ventana {k}: expected ({batch}, {Y}, {Z}, {d}), got {out.shape}"
            )

    def test_single_plane_decoder(self):
        """Funciona con X_dec=1 (un solo plano en el decoder)."""
        X_dec, Y, Z, d, dgw = 1, 2, 2, 8, 1
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 1
        V_dec_0 = torch.randn(batch, Y, Z, d)

        outputs = decoder(V_dec_0, num_windows=2)

        assert len(outputs) == 2
        assert outputs[0].shape == (batch, Y, Z, d)
        assert outputs[1].shape == (batch, Y, Z, d)

    def test_default_num_windows_is_one(self):
        """Por defecto genera una sola ventana."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        V_dec_0 = torch.randn(1, Y, Z, d)
        outputs = decoder(V_dec_0)

        assert len(outputs) == 1


class TestVSNDecoderGradient:
    """Verificar que gradientes fluyen a través del decoder."""

    def test_gradient_flows_to_input(self):
        """Gradientes fluyen desde la salida hasta V_dec_0."""
        X_dec, Y, Z, d, dgw = 4, 3, 3, 16, 4
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        V_dec_0 = torch.randn(batch, Y, Z, d, requires_grad=True)

        outputs = decoder(V_dec_0, num_windows=1)
        loss = outputs[0].sum()
        loss.backward()

        assert V_dec_0.grad is not None, "Gradientes no fluyen a V_dec_0"
        assert V_dec_0.grad.shape == V_dec_0.shape
        assert not torch.all(V_dec_0.grad == 0), "Gradientes son todos cero"

    def test_gradient_flows_through_psi_multi_window(self):
        """Gradientes fluyen a través de Ψ cuando hay múltiples ventanas."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        V_dec_0 = torch.randn(batch, Y, Z, d, requires_grad=True)

        outputs = decoder(V_dec_0, num_windows=3)
        # La loss usa la última ventana — gradientes deben fluir a través de Ψ
        loss = outputs[-1].sum()
        loss.backward()

        assert V_dec_0.grad is not None, (
            "Gradientes no fluyen a V_dec_0 a través de Ψ"
        )
        assert not torch.all(V_dec_0.grad == 0), (
            "Gradientes son cero — Ψ no propaga gradientes correctamente"
        )

    def test_gradient_flows_to_all_vgb_blocks(self):
        """Cada bloque VGB del decoder recibe gradientes."""
        X_dec, Y, Z, d, dgw = 4, 3, 3, 16, 4
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        V_dec_0 = torch.randn(batch, Y, Z, d)

        outputs = decoder(V_dec_0, num_windows=1)
        loss = outputs[0].sum()
        loss.backward()

        for x, block in enumerate(decoder.vgb_blocks):
            has_grad = any(
                p.grad is not None and not torch.all(p.grad == 0)
                for p in block.parameters()
            )
            assert has_grad, f"VGB dec block {x} no recibió gradientes"

    def test_gradient_flows_to_psi_params(self):
        """Los parámetros de Ψ reciben gradientes cuando hay multi-ventana."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        V_dec_0 = torch.randn(batch, Y, Z, d)

        outputs = decoder(V_dec_0, num_windows=2)
        # Loss en la segunda ventana obliga gradientes a través de Ψ
        loss = outputs[1].sum()
        loss.backward()

        psi_has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in decoder.psi.parameters()
        )
        assert psi_has_grad, "Ψ no recibió gradientes en multi-ventana"


class TestVSNDecoderPsi:
    """Verificar comportamiento del operador Ψ entre ventanas."""

    def test_psi_produces_different_initial_states(self):
        """Ψ produce V_dec_0 diferente para cada ventana (no copias)."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 2
        V_dec_0 = torch.randn(batch, Y, Z, d)

        outputs = decoder(V_dec_0, num_windows=3)

        # Los outputs de cada ventana deben ser diferentes
        assert not torch.allclose(outputs[0], outputs[1], atol=1e-5), (
            "Ventanas 0 y 1 producen output idéntico — Ψ no genera transición"
        )
        assert not torch.allclose(outputs[1], outputs[2], atol=1e-5), (
            "Ventanas 1 y 2 producen output idéntico"
        )

    def test_single_window_does_not_invoke_psi(self):
        """Con 1 ventana, Ψ no se ejecuta (no hay transición)."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        batch = 1
        V_dec_0 = torch.randn(batch, Y, Z, d)

        # Con 1 ventana, psi nunca se invoca — verificamos que psi
        # no tiene gradientes
        outputs = decoder(V_dec_0, num_windows=1)
        loss = outputs[0].sum()
        loss.backward()

        # Psi params should have no gradient (not used)
        for p in decoder.psi.parameters():
            assert p.grad is None or torch.all(p.grad == 0), (
                "Ψ recibió gradientes con una sola ventana — no debería"
            )


class TestVSNDecoderIndependence:
    """Verificar independencia de parámetros θ^dec."""

    def test_decoder_blocks_have_independent_params(self):
        """Cada bloque VGB del decoder tiene parámetros independientes."""
        X_dec, Y, Z, d, dgw = 4, 2, 2, 8, 4
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)

        # Verificar que los data_ptr de parámetros son distintos entre bloques
        for i in range(X_dec):
            for j in range(i + 1, X_dec):
                params_i = list(decoder.vgb_blocks[i].parameters())
                params_j = list(decoder.vgb_blocks[j].parameters())
                for p_i, p_j in zip(params_i, params_j):
                    assert p_i.data_ptr() != p_j.data_ptr(), (
                        f"Bloques {i} y {j} comparten parámetros (data_ptr igual)"
                    )


class TestVSNDecoderValidation:
    """Verificar validación de entradas."""

    def test_rejects_non_4d_input(self):
        """Rechaza V_dec_0 que no sea 4D."""
        decoder = VSNDecoder(X_dec=3, Y=2, Z=2, d=8, dgw=3)

        V_dec_0 = torch.randn(2, 8)  # 2D
        with pytest.raises(ValueError, match="4D"):
            decoder(V_dec_0)

    def test_rejects_invalid_constructor_params(self):
        """Rechaza parámetros inválidos en constructor."""
        with pytest.raises(ValueError):
            VSNDecoder(X_dec=0, Y=2, Z=2, d=8, dgw=3)
        with pytest.raises(ValueError):
            VSNDecoder(X_dec=3, Y=-1, Z=2, d=8, dgw=3)
        with pytest.raises(ValueError):
            VSNDecoder(X_dec=3, Y=2, Z=0, d=8, dgw=3)
        with pytest.raises(ValueError):
            VSNDecoder(X_dec=3, Y=2, Z=2, d=0, dgw=3)
        with pytest.raises(ValueError):
            VSNDecoder(X_dec=3, Y=2, Z=2, d=8, dgw=0)


class TestVSNDecoderDeterminism:
    """Verificar determinismo del decoder."""

    def test_same_input_same_output(self):
        """Misma entrada produce misma salida (no hay estado persistente)."""
        X_dec, Y, Z, d, dgw = 3, 2, 2, 8, 3
        decoder = VSNDecoder(X_dec=X_dec, Y=Y, Z=Z, d=d, dgw=dgw)
        decoder.eval()

        V_dec_0 = torch.randn(1, Y, Z, d)

        outputs1 = decoder(V_dec_0, num_windows=2)
        outputs2 = decoder(V_dec_0, num_windows=2)

        for k in range(2):
            assert torch.allclose(outputs1[k], outputs2[k], atol=1e-6), (
                f"Ventana {k}: outputs difieren entre forwards — "
                "estado persistente detectado"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
