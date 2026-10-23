"""CUDA numerical regression tests for the student's non-causal forward kernel."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU and Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_scale", [1.0, 4.0])
def test_flash_forward_precision(dtype, input_scale):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This GPU does not support BF16")

    from cs336_systems.flash_forward_triton import FlashForwardTriton

    torch.manual_seed(42)
    # Aligned tiles isolate precision from tail masking and causal masking.
    q = (torch.randn(2, 64, 64, device="cuda", dtype=dtype) * input_scale).requires_grad_()
    k = (torch.randn(2, 256, 64, device="cuda", dtype=dtype) * input_scale).requires_grad_()
    v = torch.randn(2, 256, 64, device="cuda", dtype=dtype, requires_grad=True)
    output = FlashForwardTriton.apply(q, k, v, False)
    lse = output.grad_fn.saved_tensors[-1]

    # FP64 reference avoids TF32 and low-precision intermediate rounding.
    with torch.no_grad():
        scores = (q.double() @ k.double().transpose(-1, -2)) / 8.0
        expected_lse = torch.logsumexp(scores, dim=-1)
        expected_output = scores.softmax(dim=-1) @ v.double()

    assert output.dtype == dtype
    assert lse.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
    tolerance = {torch.float32: 1e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(output.double(), expected_output, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse.double(), expected_lse, atol=1e-4, rtol=1e-5)
