import torch, os, mps_accel_core
device = torch.device("mps")
M, K, N = 3072, 4096, 128
x = torch.randn(M, K, device=device, dtype=torch.float32).bfloat16()
w = torch.randn(N, K, device=device, dtype=torch.float32).bfloat16()
o = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
lib_path = os.path.join(os.getcwd(), "default.metallib")
mps_accel_core.linear_mps(x, w, o, False, lib_path)
truth = torch.nn.functional.linear(x, w)
diff = (o - truth).abs().max().item()

print(f"Test Scale: [{M}x{K}] * [{K}x{N}] -> Max Diff: {diff}")
print(f"MPS-Accel Output NaN? {torch.isnan(o).any().item()}, Inf? {torch.isinf(o).any().item()}")
