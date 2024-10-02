# test_ops.py — Correctness tests for custom ops
import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_swish_correctness():
    try:
        import custom_ext_cuda
    except ImportError:
        pytest.skip("custom_ext not built")
    x = torch.randn(1024, device="cuda")
    ref = torch.nn.functional.silu(x)
    out = custom_ext_cuda.swish(x)
    assert torch.allclose(ref, out, atol=1e-5), "Swish output mismatch"
