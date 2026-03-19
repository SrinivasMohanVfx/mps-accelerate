"""Test mps_accel_core.linear_mps across multiple matrix shapes."""
import torch, os, mps_accel_core
device = torch.device("mps")
lib_path = os.path.join(os.getcwd(), "default.metallib")

shapes = [
    (3072, 4096, 128),
    (3072, 128, 4096),
    (512, 12288, 4096),
    (512, 8192, 128),
    (1536, 12288, 4096)
]

for M, K, N in shapes:
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    o = torch.zeros(M, N, device=device, dtype=torch.float16)

    try:
        mps_accel_core.linear_mps(x, w, o, False, lib_path)
        torch.mps.synchronize()
        truth = torch.nn.functional.linear(x, w)
        torch.mps.synchronize()
        diff = (o - truth).abs().max().item()
        print(f"Test [{M}x{K}] * [{K}x{N}] -> Max Diff: {diff:.5f} | NaN? {torch.isnan(o).any().item()}")
    except Exception as e:
        print(f"Test [{M}x{K}] * [{K}x{N}] -> FAILED: {e}")
