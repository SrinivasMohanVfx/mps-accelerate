# MPS-Accelerate

**Drop-in PyTorch acceleration for Apple Silicon** — 22% faster inference by calling Apple's `MPSMatrixMultiplication` directly from C++, bypassing PyTorch's dispatch overhead.

```
Native PyTorch MPS:  10.6 s/it
MPS-Accelerate:       8.3 s/it  ← 22% faster
```

> Tested on MacBook Pro M2 Max (64GB) running Flux.1-Dev 12B (bfloat16, 5 steps).

---

## How It Works

PyTorch's MPS backend routes every `F.linear` call through Python → MPSGraph → GPU. MPS-Accelerate short-circuits this:

```
PyTorch default:   Python → PyTorch dispatch (0.97ms) → MPSGraph → GPU GEMM
MPS-Accelerate:    Python → pybind11 (0.08ms) → MPSMatrixMultiplication → GPU GEMM
                                 ↑ 12× faster dispatch, same Apple-optimized GEMM
```

The `mps_accel_core` C++ module calls Apple's `MPSMatrixMultiplication` directly on PyTorch's MPS command buffer — zero-copy, fully asynchronous, no graph compilation overhead.

---

## Quick Start

### Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4, any variant)
- Python 3.11 (pre-built) or 3.10+ (build from source)
- PyTorch 2.0+ with MPS support
- Xcode 15+ Command Line Tools

### Build from Source

```bash
git clone https://github.com/SrinivasMohanVfx/mps-accelerate.git
cd mps-accelerate
pip install -r requirements.txt
make clean && make all
```

This produces:
- `mps_accel_core.cpython-3XX-darwin.so` — the C++ acceleration module
- `default.metallib` — compiled Metal compute shaders (used for SDPA attention)

### Verify

```bash
PYTHONPATH=. python tests/test_all_shapes.py
```

---

## Usage

### ComfyUI (Drop-in Custom Node)

1. Copy the integration folder to ComfyUI (pre-built binaries included — no compilation needed):
```bash
cp -r integrations/ComfyUI-MPSAccel /path/to/ComfyUI/custom_nodes/ComfyUI-MPSAccel
```

2. Restart ComfyUI
3. Add the **"MPS Accelerate"** node to your workflow (connects to your model loader)
4. Look for `>> [MPS-Accel] Acceleration ENABLED.` in the console to confirm it's active

> **Note:** Pre-built binaries are for Python 3.11. If you're on a different Python version, rebuild from source: `make clean && make all`, then copy the new `.so` and `.metallib` into the integration folder.

### Any PyTorch Script

```python
import torch
import mps_accel_core

# Create tensors on MPS
x = torch.randn(3072, 4096, device="mps", dtype=torch.float16)
w = torch.randn(12288, 4096, device="mps", dtype=torch.float16)
out = torch.zeros(3072, 12288, device="mps", dtype=torch.float16)

# Direct MPSMatrixMultiplication — faster than F.linear
mps_accel_core.linear_mps(x, w, out, False, "default.metallib")
```

### Monkey-Patch F.linear Globally

```python
import torch.nn.functional as F
import mps_accel_core, os

LIB = os.path.join(os.path.dirname(__file__), "default.metallib")
_original_linear = F.linear

def accelerated_linear(input, weight, bias=None):
    if input.device.type == "mps" and input.numel() > 16384:
        in_c = input.contiguous().half()
        w_c = weight.half()
        out = torch.zeros(*input.shape[:-1], weight.shape[0],
                         device=input.device, dtype=torch.float16)
        mps_accel_core.linear_mps(in_c, w_c, out, False, LIB)
        if input.dtype == torch.bfloat16:
            out = out.bfloat16()
        if bias is not None:
            out.add_(bias.to(out.dtype))
        return out
    return _original_linear(input, weight, bias)

F.linear = accelerated_linear  # All nn.Linear layers now accelerated
```

---

## Benchmarks

Tested on MacBook Pro M2 Max (38-core GPU, 64GB unified memory):

| Shape (M×K × K×N) | PyTorch MPS | MPS-Accelerate | Speedup |
|----|----|----|---|
| 3072×4096 × 4096×12288 | 46.2 ms | 46.2 ms | Same GEMM |
| 3072×4096 × 4096×4096 | 15.8 ms | 15.8 ms | Same GEMM |
| **Per-call dispatch** | **0.97 ms** | **0.08 ms** | **12× faster** |
| **Full Flux.1 (5 steps)** | **10.6 s/it** | **8.3 s/it** | **22% faster** |

The GEMM throughput is identical (both use Apple's `MPSMatrixMultiplication`). The speedup comes entirely from eliminating Python-level dispatch overhead across ~100 linear operations per step.

---

## Architecture

```
mps-accelerate/
├── src/
│   ├── bindings.mm             ← pybind11 → MPSMatrixMultiplication
│   ├── flux_graph.h            ← C API declarations
│   ├── flux_graph.mm           ← Custom Metal SDPA kernel dispatch
│   └── flux_kernels.metal      ← Metal compute shaders (SDPA + reference GEMM)
├── integrations/
│   └── ComfyUI-MPSAccel/       ← Drop-in ComfyUI custom node (with pre-built binaries)
│       ├── __init__.py         ← Node registration + early patching
│       ├── mps_ops.py          ← F.linear monkey-patch + bf16↔fp16 handling
│       ├── mps_accel_core.*.so ← Pre-built C++ module (Python 3.11)
│       └── default.metallib    ← Pre-built Metal shaders
├── tests/
│   ├── test_all_shapes.py      ← Correctness across 5 matrix shapes
│   ├── test_scale.py           ← Single large shape test
│   └── benchmark_kernel.py     ← Performance comparison vs PyTorch
├── Makefile                    ← Build system
├── requirements.txt            ← Python dependencies
└── README.md
```

### C++ API (`mps_accel_core`)

| Function | Description |
|----------|-------------|
| `linear_mps(x, w, out, accumulate, lib_path)` | GEMM via `MPSMatrixMultiplication` (float16) |
| `sdpa_mps(q, k, v, out, scale, lib_path)` | Scaled Dot-Product Attention via custom Metal kernel |

---

## Supported Dtypes

| Dtype | Linear (`linear_mps`) | SDPA (`sdpa_mps`) |
|-------|----------------------|-------------------|
| float16 | ✅ Native | ✅ Native |
| bfloat16 | ✅ Auto-cast to fp16 | ✅ Native |
| float32 | ✅ Via MPSMatrixMultiplication | ❌ Not supported |

> **Note:** `MPSMatrixMultiplication` doesn't support bfloat16 natively. The ComfyUI integration handles bf16↔fp16 conversion automatically.

---

## Known Limitations

- **macOS only** — requires Metal and MPS frameworks
- **bfloat16 conversion overhead** — bf16 models incur a small type-cast cost per linear call
- **Minimum dimensions** — operations smaller than 128×128×128 fall back to PyTorch (dispatch overhead would negate GEMM savings)

---

## License

MIT
