# ComfyUI-MPSAccel

**MPS-Accelerate** custom node for ComfyUI — 22% faster Flux inference on Apple Silicon.

## Installation

### Option 1: From Pre-built (Fastest)

```bash
# Copy this entire folder to ComfyUI custom nodes
cp -r integrations/ComfyUI-MPSAccel /path/to/ComfyUI/custom_nodes/

# Copy the pre-built binaries from the mps-accelerate root
cp mps_accel_core.cpython-*-darwin.so default.metallib \
   /path/to/ComfyUI/custom_nodes/ComfyUI-MPSAccel/
```

### Option 2: Build from Source

```bash
cd /path/to/mps-accelerate
make clean && make all
cp mps_accel_core.cpython-*-darwin.so default.metallib \
   integrations/comfyui/*.py \
   /path/to/ComfyUI/custom_nodes/ComfyUI-MPSAccel/
```

## Usage

1. Restart ComfyUI
2. In your workflow, add the **"MPS Accelerate"** node (Category: MPS-Accelerate)
3. Connect it between your model loader and sampler
4. Console shows `[MPS-Accel] Acceleration active.` when working

## Requirements

- macOS 13+ (Ventura)
- Apple Silicon (M1/M2/M3/M4)
- ComfyUI with PyTorch 2.0+ MPS support

## Files

| File | Description |
|------|-------------|
| `__init__.py` | ComfyUI node registration |
| `mps_ops.py` | F.linear monkey-patch, bf16↔fp16 conversion |
| `mps_accel_core.*.so` | C++ pybind11 module (MPSMatrixMultiplication) |
| `default.metallib` | Compiled Metal shaders (SDPA kernels) |
| `requirements.txt` | Python dependencies |
