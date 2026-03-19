"""Benchmark: MPS-Accel kernel vs native PyTorch for a single large GEMM.
This isolates kernel speed from all Python/dispatch overhead."""
import torch
import torch.nn.functional as F
import mps_accel_core
import os
import time

device = torch.device("mps")
lib_path = os.path.join(os.getcwd(), "default.metallib")

# Typical Flux shapes
shapes = [
    (3072, 4096, 12288, "QKV projection"),
    (3072, 4096, 4096,  "Attention output"),
    (512, 12288, 4096,  "txt linear"),
    (3072, 4096, 24576, "MLP up (largest)"),
]

print("=" * 80)
print("BENCHMARK: MPS-Accel kernel vs native PyTorch F.linear")
print("=" * 80)

for M, K, N, label in shapes:
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(3):
        o = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
        mps_accel_core.linear_mps(x, w, o, False, lib_path)
        torch.mps.synchronize()
    
    # Benchmark MPS-Accel
    torch.mps.synchronize()
    t0 = time.perf_counter()
    N_ITER = 20
    for _ in range(N_ITER):
        o = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
        mps_accel_core.linear_mps(x, w, o, False, lib_path)
    torch.mps.synchronize()
    amx_time = (time.perf_counter() - t0) / N_ITER * 1000  # ms

    # Warmup PyTorch
    for _ in range(3):
        _ = F.linear(x, w)
        torch.mps.synchronize()
    
    # Benchmark PyTorch native
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = F.linear(x, w)
    torch.mps.synchronize()
    pyt_time = (time.perf_counter() - t0) / N_ITER * 1000  # ms

    gflops = 2 * M * K * N / 1e9
    amx_tflops = gflops / amx_time
    pyt_tflops = gflops / pyt_time
    
    print(f"\n[{label}] [{M}x{K}] * [{K}x{N}]  ({gflops:.1f} GFLOPS)")
    print(f"  MPS-Accel:    {amx_time:7.2f} ms  ({amx_tflops:.2f} TFLOPS)")
    print(f"  PyTorch:{pyt_time:7.2f} ms  ({pyt_tflops:.2f} TFLOPS)")
    print(f"  Ratio:  MPS-Accel is {pyt_time/amx_time:.2f}x {'FASTER' if amx_time < pyt_time else 'SLOWER'}")

print("\n" + "=" * 80)
print("DISPATCH OVERHEAD TEST (empty-ish kernel vs PyTorch)")
print("=" * 80)

# Small matrix to measure pure dispatch overhead
x_s = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
w_s = torch.randn(128, 128, device=device, dtype=torch.bfloat16)

torch.mps.synchronize()
t0 = time.perf_counter()
N_ITER = 100
for _ in range(N_ITER):
    o_s = torch.zeros(128, 128, device=device, dtype=torch.bfloat16)
    mps_accel_core.linear_mps(x_s, w_s, o_s, False, lib_path)
torch.mps.synchronize()
amx_small = (time.perf_counter() - t0) / N_ITER * 1000

torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = F.linear(x_s, w_s)
torch.mps.synchronize()
pyt_small = (time.perf_counter() - t0) / N_ITER * 1000

print(f"  MPS-Accel dispatch:    {amx_small:.3f} ms per call")
print(f"  PyTorch dispatch:{pyt_small:.3f} ms per call")
print(f"  Overhead diff:   {amx_small - pyt_small:.3f} ms")
print(f"  × 100 calls/step = {(amx_small - pyt_small) * 100:.1f} ms extra per step")
