import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve local library path — metallib sits beside this module
LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.metallib")
# Global enable flag — patching happens early (import time) so all modules
# get consistent references, but acceleration only activates when the node runs.
_accel_enabled = False
_weight_cache = {}  # Cache weight alignment + bf16 safety per weight id()

try:
    from . import mps_accel_core
except ImportError:
    try:
        import mps_accel_core
    except ImportError:
        import sys
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f">> [MPS-Accel] ERROR: Cannot load mps_accel_core.")
        print(f">> [MPS-Accel] Pre-built binary is for Python 3.11, you have Python {py_ver}.")
        print(f">> [MPS-Accel] Please rebuild: cd mps-accelerate && make clean && make all")
        raise


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
        # Bypass if acceleration not activated by node
        if not _accel_enabled:
            return old_linear(input, weight, bias)
        
        if input.device.type == "mps" and weight.device.type == "mps":
            K = input.size(-1)
            M = input.numel() // K
            N = weight.size(0)
            
            # Only dispatch dense operations to MPS GEMM
            if M >= 128 and N >= 128 and K >= 128:
                try:
                    in_c = input if input.is_contiguous() else input.contiguous()
                    
                    # Cache weight alignment + bf16 check per weight tensor
                    w_id = id(weight)
                    cached = _weight_cache.get(w_id)
                    if cached is None:
                        byte_offset = weight.storage_offset() * weight.element_size()
                        needs_clone = (byte_offset % 4 != 0 or not weight.is_contiguous())
                        bf16_safe = None
                        if weight.dtype == torch.bfloat16:
                            w_tmp = weight.clone().contiguous() if needs_clone else weight
                            w_max = w_tmp.abs().max().item()
                            bf16_safe = w_max < 65500
                        _weight_cache[w_id] = (needs_clone, bf16_safe)
                        cached = _weight_cache[w_id]
                    
                    needs_clone, bf16_safe = cached
                    w_c = weight.clone().contiguous() if needs_clone else weight
                    
                    # dtype conversion
                    original_dtype = input.dtype
                    if original_dtype == torch.bfloat16:
                        if bf16_safe:
                            in_c = in_c.half()
                            w_c = w_c.half() if w_c.dtype == torch.bfloat16 else w_c
                        else:
                            in_c = in_c.float()
                            w_c = w_c.float() if w_c.dtype == torch.bfloat16 else w_c
                    
                    out = torch.empty((*input.shape[:-1], N), device=input.device, dtype=in_c.dtype)
                    mps_accel_core.linear_mps(in_c, w_c, out, False, LIB_PATH)
                    
                    if original_dtype == torch.bfloat16:
                        out = out.to(original_dtype)
                    
                    if bias is not None:
                        out.add_(bias)

                    return out
                except Exception as e:
                    print(f">> [MPS-Accel] Linear Error: {e}")
                    return old_linear(input, weight, bias)
                    
        return old_linear(input, weight, bias)

    # 2. Apply F.linear patch
    F.linear = mps_linear_wrapper
    
    # 3. Global scan: patch all modules that imported F.linear
    linear_count = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("comfy") or mod_name.startswith("torch"):
            if hasattr(mod, "linear"):
                attr = getattr(mod, "linear")
                if attr is old_linear:
                    setattr(mod, "linear", mps_linear_wrapper)
                    linear_count += 1
                    
    print(f">> [MPS-Accel] Patched F.linear in {linear_count} modules. Acceleration active.")

print(">> [MPS-Accel] Initialized.")
