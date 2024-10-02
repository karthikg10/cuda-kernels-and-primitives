# bench_activations.py — Custom ops vs PyTorch native
# Research / Exploratory

import torch
import time

def bench(fn, x, n=1000):
    for _ in range(50): fn(x)  # warmup
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000  # ms

if __name__ == "__main__":
    x = torch.randn(1, 4096, device="cuda")
    native_swish = torch.nn.functional.silu

    print(f"PyTorch SiLU (BS=1):   {bench(native_swish, x):.3f} ms")
    try:
        import custom_ext_cuda
        custom_swish = custom_ext_cuda.swish
        print(f"Custom CUDA Swish (BS=1): {bench(custom_swish, x):.3f} ms")
    except ImportError:
        print("custom_ext_cuda not built yet — run: pip install -e .")
