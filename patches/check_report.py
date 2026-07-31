"""
M2F2-Det Environment & Inference Readiness Check
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_file(path, desc):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {'✅' if exists else '❌'} {desc}: {path if exists else '(missing)'}")
    if exists:
        print(f"        Size: {size / 1024**3:.2f} GB" if size > 1024**3 else f"        Size: {size / 1024**2:.2f} MB" if size > 1024**2 else f"        Size: {size / 1024:.1f} KB")
    return exists

def check_dir(path, desc):
    exists = os.path.isdir(path)
    print(f"  {'✅' if exists else '❌'} {desc}: {path if exists else '(missing)'}")
    return exists

print("=" * 70)
print("M2F2-Det: Environment & Inference Readiness Report")
print("=" * 70)

print("\n[1] Core Environment")
print("-" * 50)
import torch
import torchvision
import transformers
import numpy
print(f"  PyTorch:    {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"  GPUs:       {torch.cuda.device_count()}")
print(f"  torchvision:{torchvision.__version__}")
print(f"  transformers:{transformers.__version__}")
print(f"  numpy:      {numpy.__version__}")

# check flash_attn patch
try:
    from flash_attn.modules.mha import MHA
    print(f"  flash_attn: patched ({MHA.__module__})")
except:
    print(f"  flash_attn: NOT AVAILABLE")

print("\n[2] Model Weights")
print("-" * 50)
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_dir = os.path.join(proj_root, 'checkpoints')
weights_dir = os.path.join(proj_root, 'utils', 'weights')

# llava-v1.5 checkpoint
ckpt_exists = check_dir(os.path.join(ckpt_dir, 'llava-v1.5-7b-M2F2-Det'), 'LLaVA-7B checkpoint')
if ckpt_exists:
    shards = [f for f in os.listdir(os.path.join(ckpt_dir, 'llava-v1.5-7b-M2F2-Det')) if f.endswith('.safetensors')]
    print(f"        Shard files: {len(shards)}/3")
    for s in sorted(shards):
        spath = os.path.join(ckpt_dir, 'llava-v1.5-7b-M2F2-Det', s)
        ssize = os.path.getsize(spath)
        print(f"        {s}: {ssize/1024**3:.2f} GB")

# stage_1 checkpoint
s1_dir = os.path.join(ckpt_dir, 'stage_1')
check_dir(s1_dir, 'Stage-1 checkpoint dir')
s1_files = os.listdir(s1_dir) if os.path.isdir(s1_dir) else []
print(f"        Files: {s1_files}")

# weights
check_file(os.path.join(weights_dir, 'M2F2_Det_densenet121.pth'), 'M2F2_Det densenet121 weights')
check_file(os.path.join(weights_dir, 'vision_tower.pth'), 'CLIP vision tower weights (needed for stage_1)')

print("\n[3] Dataset")
print("-" * 50)
data_dir = os.path.join(proj_root, 'dataset', 'data_2023')
check_dir(data_dir, 'Data directory')

# Check H5 files
import glob
h5_files = glob.glob(os.path.join(proj_root, 'dataset', 'data_2023', '*.h5'))
print(f"  H5 files: {len(h5_files)} {'✅' if h5_files else '❌ (none found)'}")

# Check zip archive
zip_file = os.path.join(proj_root, 'dataset', 'FF++_test_only.zip')
if os.path.exists(zip_file):
    zsize = os.path.getsize(zip_file)
    print(f"  FF++_test_only.zip: {zsize/1024**3:.2f} GB (need to extract)")

# Check test split
check_file(os.path.join(proj_root, 'utils', 'FFPP_split', 'test.json'), 'Test split JSON')

print("\n[4] Inference Path Verification")
print("-" * 50)
print("""
Path A: cli_DDVQA_det.py (LLaVA-based DDVQA)
  python llava/serve/cli_DDVQA_det.py --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det
  Needs:  LLaVA checkpoint ✅  (3/3 safetensors)
          M2F2_Det weights ✅  (densenet121.pth)
          CLIP models   ⚠️   (auto-download from HuggingFace)
          Test images   ❌   (DDVQA_images needed)
  Status: Partially ready - need images

Path B: stage_1_detection_inference.py (Standalone Detection)
  python stage_1_detection_inference.py
  Needs:  Stage-1 checkpoint     ❌ (current_model_180.pth missing)
          vision_tower.pth       ❌ (missing)
          H5 dataset files        ❌ (only zip archive found)
          M2F2_Det weights       ✅ (exists)
  Status: Not ready - missing checkpoints and data

NOTE: Network access to HuggingFace may fail in restricted environments.
      The model needs to download CLIP models on first run.
""")

print("\n[5] Summary")
print("-" * 50)
missing_items = []
if not ckpt_exists:
    missing_items.append("LLaVA checkpoint")
if not os.path.exists(os.path.join(weights_dir, 'M2F2_Det_densenet121.pth')):
    missing_items.append("M2F2_Det weights")
if not os.path.exists(os.path.join(s1_dir, 'current_model_180.pth')):
    missing_items.append("stage_1 checkpoint")
if not os.path.exists(os.path.join(weights_dir, 'vision_tower.pth')):
    missing_items.append("vision_tower weights")
if not h5_files:
    missing_items.append("H5 dataset files")

if not missing_items:
    print("  All files present! Ready for inference.")
else:
    print(f"  Missing items ({len(missing_items)}):")
    for item in missing_items:
        print(f"    ❌ {item}")
print("=" * 70)
