# ComfyUI-MPSAccel

**MPS-Accelerate** custom node for ComfyUI — ~22% faster Flux inference on Apple Silicon.

## Installation

```bash
# Clone into ComfyUI custom_nodes
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/SrinivasMohanVfx/mps-accelerate.git

# Copy the node folder
cp -r mps-accelerate/integrations/ComfyUI-MPSAccel ./ComfyUI-MPSAccel
```

Pre-built binaries (`.so` and `.metallib`) are included — **no compilation needed**.

> To rebuild from source (optional): `cd mps-accelerate && make clean && make all`

## Usage

1. Restart ComfyUI
2. In your workflow, add the **"MPS Accelerate"** node (Category: MPS-Accelerate)
3. Connect it between your model loader and sampler
4. Console shows `>> [MPS-Accel] Acceleration ENABLED.` when working

## Requirements

- macOS 13+ (Ventura)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11
- ComfyUI with PyTorch 2.0+ MPS support

## Files

| File | Description |
|------|-------------|
| `__init__.py` | ComfyUI node registration + early patching |
| `mps_ops.py` | F.linear monkey-patch, bf16↔fp16 conversion |
| `mps_accel_core.*.so` | C++ pybind11 module (MPSMatrixMultiplication) |
| `default.metallib` | Compiled Metal shaders |
| `requirements.txt` | Python dependencies |
