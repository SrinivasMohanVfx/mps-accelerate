import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve local library path — metallib sits beside this module
LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.metallib")
try:
    from . import mps_accel_core
except ImportError:
    import mps_accel_core

def mps_sdpa(q, k, v, scale=None, theta=10000.0):
    """Scaled Dot-Product Attention via custom Metal SIMD kernel."""
    original_dtype = q.dtype
    is_shd = False
    
    if q.dim() == 4 and q.size(1) < q.size(2):  # [B, H, S, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        is_shd = False
    else:
        is_shd = True
        
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
        
    out = torch.empty((B, S, H, D), device=q.device, dtype=q.dtype)
    
    for b in range(B):
        mps_accel_core.sdpa_mps(q[b], k[b], v[b], out[b].view(-1), scale, LIB_PATH)
    
    if not is_shd:
        out = out.transpose(1, 2)  # Back to [B, H, S, D]
        
    return out

def patch_model_attention():
    """
    Hooks optimized MPS linear kernel into PyTorch's F.linear.
    Safe to call multiple times — only patches once per session.
    """
    if hasattr(patch_model_attention, '_patched'):
        return  # Already patched this session
    patch_model_attention._patched = True
    
    import sys
    import torch.nn.functional as F
    
    # 1. Linear wrapper — MPSMatrixMultiplication (the main acceleration)
    old_linear = F.linear

    def mps_linear_wrapper(input, weight, bias=None):
        if not hasattr(mps_linear_wrapper, "call_count"): mps_linear_wrapper.call_count = 0
        mps_linear_wrapper.call_count += 1
        
        if input.device.type == "mps" and weight.device.type == "mps":
            import math
            K = input.size(-1)
            M = math.prod(input.shape[:-1])
            N = weight.size(0)
            
            # Only dispatch dense operations to MPS GEMM
            if M >= 128 and N >= 128 and K >= 128:
                try:
                    in_c = input.contiguous()
                    byte_offset = weight.storage_offset() * weight.element_size()
                    if byte_offset % 4 != 0 or not weight.is_contiguous():
                        w_c = weight.clone().contiguous()
                    else:
                        w_c = weight
                    
                    # Convert bfloat16 → float32 (NOT float16!)
                    # float16 max is only 65504 — models like Lumina2 overflow it
                    # float32 is fully supported by MPSMatrixMultiplication
                    original_dtype = input.dtype
                    if in_c.dtype == torch.bfloat16:
                        in_c = in_c.float()
                    if w_c.dtype == torch.bfloat16:
                        w_c = w_c.float()
                    
                    compute_dtype = in_c.dtype
                    out_shape = list(input.shape[:-1]) + [weight.shape[0]]
                    out = torch.zeros(out_shape, device=input.device, dtype=compute_dtype)
                    
                    mps_accel_core.linear_mps(in_c, w_c, out, False, LIB_PATH)
                    
                    if original_dtype == torch.bfloat16:
                        out = out.bfloat16()
                    
                    # Tether async pointers to prevent GC
                    out._metal_retain_x = in_c
                    out._metal_retain_w = w_c
                    
                    if bias is not None:
                        out.add_(bias.to(out.dtype))

                    return out
                except Exception as e:
                    print(f">> [MPS-Accel] Linear Error: {e}")
                    return old_linear(input, weight, bias)
                    
        return old_linear(input, weight, bias)

    # 2. Apply F.linear patch only (SDPA left to native PyTorch for all models)
    F.linear = mps_linear_wrapper
    
    # 3. Global scan: patch all modules that imported F.linear
    patch_count = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("comfy") or mod_name.startswith("torch"):
            if hasattr(mod, "linear"):
                attr = getattr(mod, "linear")
                if attr is old_linear:
                    setattr(mod, "linear", mps_linear_wrapper)
                    patch_count += 1
                    
    print(f">> [MPS-Accel] Patched F.linear in {patch_count} modules. Acceleration active.")

class MPSAccelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def convert_weights(self):
        self.weight.data = self.weight.data.half().contiguous()
        if self.bias is not None:
            self.bias.data = self.bias.data.float().contiguous()

print(">> [MPS-Accel] Initialized.")
