import sys
import os

# Add the current directory to sys.path so local modules can be found
node_dir = os.path.dirname(__file__)
if node_dir not in sys.path:
    sys.path.append(node_dir)

import torch
import torch.nn as nn
from .mps_ops import MPSAccelLinear, patch_model_attention

class PatchModelWithMPS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MPS-Accelerate"

    def patch(self, model):
        diffusion_model = model.model.diffusion_model
        print(f">> [MPS-Accel] Patching model with MPS acceleration...")
        
        # Patch F.linear and attention functions globally
        patch_model_attention()
        
        print(f">> [MPS-Accel] Patching complete. Ready for generation.")
        return (model,)

NODE_CLASS_MAPPINGS = {
    "PatchModelWithMPS": PatchModelWithMPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchModelWithMPS": "MPS Accelerate",
}
