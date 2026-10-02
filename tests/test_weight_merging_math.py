import pytest
import torch


def test_lora_merge_math():
    """Validates the exact mathematical equivalence of additive LoRA merge W_eff = W + (alpha / r) * B @ A."""
    d_out = 64
    d_in = 64
    r = 16
    alpha = 32
    scaling = alpha / r

    torch.manual_seed(42)
    W_base = torch.randn(d_out, d_in)
    A = torch.randn(r, d_in)
    B = torch.randn(d_out, r)

    # Effective merged weight
    delta_W = scaling * (B @ A)
    W_merged = W_base + delta_W

    # Test with arbitrary input batch
    x = torch.randn(8, d_in)

    # Dynamic adapter computation: x @ W_base^T + scaling * (x @ A^T @ B^T)
    y_base = x @ W_base.T
    y_lora = (x @ A.T) @ B.T
    y_dynamic = y_base + scaling * y_lora

    # Merged model computation: x @ W_merged^T
    y_merged = x @ W_merged.T

    # For float32, associative floating-point arithmetic differences ((x @ A.T) @ B.T vs x @ (B @ A).T) are within 1e-4
    max_diff_fp32 = torch.max(torch.abs(y_dynamic - y_merged)).item()
    assert max_diff_fp32 < 1e-3, f"Discrepancy in float32 merged computation: {max_diff_fp32}"

    # In float64, precision difference is negligible (< 1e-7)
    W_base_d = W_base.double()
    A_d = A.double()
    B_d = B.double()
    x_d = x.double()

    W_merged_d = W_base_d + scaling * (B_d @ A_d)
    y_dynamic_d = x_d @ W_base_d.T + scaling * ((x_d @ A_d.T) @ B_d.T)
    y_merged_d = x_d @ W_merged_d.T

    max_diff_fp64 = torch.max(torch.abs(y_dynamic_d - y_merged_d)).item()
    assert max_diff_fp64 < 1e-7, f"Discrepancy in float64 merged computation: {max_diff_fp64}"

