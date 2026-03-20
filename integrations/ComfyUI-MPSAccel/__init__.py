import sys
import os

# Add the current directory to sys.path so local modules can be found
node_dir = os.path.dirname(__file__)
if node_dir not in sys.path:
    sys.path.append(node_dir)

import torch
import torch.nn as nn
from . import mps_ops
from .mps_ops import patch_model_attention

# ─── Early patch at import time ───
# Installs our wrapper on F.linear BEFORE any model loads, so all modules
# get consistent references. The wrapper defaults to bypass mode (no overhead)
# until the node activates it.
if torch.backends.mps.is_available():
    patch_model_attention()

class PatchModelWithMPS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MPS-Accelerate"

    def patch(self, model):
        # Activate acceleration — flips the global flag so the wrapper
        # dispatches to MPSMatrixMultiplication instead of native F.linear
        mps_ops._accel_enabled = True
        print(f">> [MPS-Accel] Acceleration ENABLED. (Restart ComfyUI to disable)")
        return (model,)

NODE_CLASS_MAPPINGS = {
    "PatchModelWithMPS": PatchModelWithMPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchModelWithMPS": "MPS Accelerate",
}
