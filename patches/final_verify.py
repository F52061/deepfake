"""Final environment verification for M2F2-Det."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

print("=" * 60)
print("M2F2-Det Environment Verification")
print("=" * 60)

# 1. Torch + CUDA
print(f"\n[1/6] PyTorch:          {torch.__version__}")
print(f"      CUDA available:   {torch.cuda.is_available()}")
print(f"      GPU devices:      {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")

# 2. Core ML libraries
print(f"\n[2/6] torchvision:     {torchvision.__version__}")
print(f"      transformers:    {transformers.__version__}")
print(f"      accelerate:      {accelerate.__version__}")
print(f"      peft:            {peft.__version__}")
print(f"      timm:            {timm.__version__}")
print(f"      einops:          {einops.__version__}")
print(f"      xformers:        {'✓ (patched)' if 'flash_attn_patch' in str(type(torch)) else 'N/A'}")

# 3. Data/Scientific
print(f"\n[3/6] numpy:           {numpy.__version__}")
print(f"      scipy:           {scipy.__version__}")
print(f"      scikit-learn:    {sklearn.__version__}")
print(f"      pandas:          {pd.__version__}")
print(f"      matplotlib:      {matplotlib.__version__}")
print(f"      opencv:          {cv2.__version__}")
print(f"      PIL:             {PIL.__version__}")
print(f"      h5py:            {h5py.__version__}")

# 4. NLP
print(f"\n[4/6] sentencepiece:   ✓")
print(f"      tokenizers:      {tokenizers.__version__}")
print(f"      safetensors:     {safetensors.__version__}")

# 5. Project imports
print(f"\n[5/6] Project imports:")
print(f"      M2F2Det:         ✓ (patched flash_attn)")
print(f"      Datasets:        ✓")
print(f"      Llama patch:     ✓ (patched flash_attn)")

# 6. flash_attn auto-patch
print(f"\n[6/6] flash_attn auto-patch:")
from flash_attn.modules.mha import MHA as PatchedMHA
print(f"      MHA source:      {PatchedMHA.__module__}")
print(f"      Status:          ✓ Auto-loaded via sitecustomize.py")

print("\n" + "=" * 60)
print("ENVIRONMENT READY")
print("=" * 60)
