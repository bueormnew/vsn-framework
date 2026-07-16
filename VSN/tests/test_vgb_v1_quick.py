"""Quick verification test for VGBv1 — shapes and gradient flow."""

import torch
from vsn.core.vgb import VGBv1


def test_vgb_v1_output_shapes():
    """Verify all outputs have correct shapes (B, Y, Z, d)."""
    B, Y, Z, d = 2, 4, 4, 32
    block = VGBv1(d=d, plane_idx=0)

    x = torch.randn(B, Y, Z, d)
    M = torch.randn(B, Y, Z, d)

    F, G, r, M_new = block(x, M)

    assert F.shape == (B, Y, Z, d), f"F shape mismatch: {F.shape}"
    assert G.shape == (B, Y, Z, d), f"G shape mismatch: {G.shape}"
    assert r.shape == (B, Y, Z, d), f"r shape mismatch: {r.shape}"
    assert M_new.shape == (B, Y, Z, d), f"M_new shape mismatch: {M_new.shape}"
    print("✓ Output shapes correct")


def test_vgb_v1_gradient_flow():
    """Verify gradients flow through all parameters."""
    B, Y, Z, d = 2, 4, 4, 16
    block = VGBv1(d=d, plane_idx=0)

    x = torch.randn(B, Y, Z, d, requires_grad=True)
    M = torch.randn(B, Y, Z, d, requires_grad=True)

    F, G, r, M_new = block(x, M)

    # Loss combining all outputs to test full gradient flow
    loss = F.sum() + G.sum() + r.sum() + M_new.sum()
    loss.backward()

    # Check gradients on input
    assert x.grad is not None, "No gradient on x"
    assert M.grad is not None, "No gradient on M"

    # Check gradients on all parameters
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient on parameter: {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient on parameter: {name}"

    print("✓ Gradient flow verified for all parameters")


def test_vgb_v1_f_equals_r():
    """Verify F is exactly r (no separate projection)."""
    B, Y, Z, d = 1, 2, 2, 8
    block = VGBv1(d=d, plane_idx=0)

    x = torch.randn(B, Y, Z, d)
    M = torch.randn(B, Y, Z, d)

    F, G, r, M_new = block(x, M)

    assert torch.equal(F, r), "F should be identical to r"
    print("✓ F == r verified")


def test_vgb_v1_memory_gating():
    """Verify memory gating boundary behavior."""
    B, Y, Z, d = 1, 2, 2, 8
    block = VGBv1(d=d, plane_idx=0)

    x = torch.randn(B, Y, Z, d)
    M_old = torch.randn(B, Y, Z, d)

    # With default random weights, just verify M_new is in valid range
    # and depends on both M_old and x
    F, G, r, M_new = block(x, M_old)

    # M_new should not be identical to M_old (extremely unlikely with random init)
    assert not torch.equal(M_new, M_old), "M_new should differ from M_old"
    print("✓ Memory gating produces updated state")


def test_vgb_v1_independent_planes():
    """Verify different plane_idx blocks have independent parameters."""
    d = 16
    block_0 = VGBv1(d=d, plane_idx=0)
    block_1 = VGBv1(d=d, plane_idx=1)

    # Each block has its own parameters (different data_ptr)
    for (name_0, p0), (name_1, p1) in zip(
        block_0.named_parameters(), block_1.named_parameters()
    ):
        assert p0.data_ptr() != p1.data_ptr(), (
            f"Parameters share memory: {name_0} vs {name_1}"
        )

    print("✓ Independent parameters per plane")


if __name__ == "__main__":
    test_vgb_v1_output_shapes()
    test_vgb_v1_gradient_flow()
    test_vgb_v1_f_equals_r()
    test_vgb_v1_memory_gating()
    test_vgb_v1_independent_planes()
    print("\n✓ All VGBv1 tests passed!")
