"""修复版 LoRA 合并 — 处理 key 前缀和 config 缺失问题。"""
import os, sys, torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only merge (safe, prevents OOM)
print("CUDA_VISIBLE_DEVICES= (CPU only)", flush=True)

# CLIP offline redirect
clip_local = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '..', 'checkpoints', 'clip-vit-large-patch14-336'))
if os.path.isdir(clip_local):
    from transformers import (CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor,
        CLIPTextConfig, CLIPTextModel, AutoConfig, AutoTokenizer)
    def rdr(f):
        def w(p,*a,**k):
            if 'openai/clip' in str(p): return f(clip_local,*a,**k)
            return f(p,*a,**k)
        return w
    for cls in [CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, AutoConfig, AutoTokenizer]:
        cls.from_pretrained = rdr(cls.from_pretrained)

BASE = "./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
LORA = "./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta"
OUTPUT = "./checkpoints/llava-v1.5-7b-M2F2-Det-bridge"

print(f"Base: {BASE}", flush=True)
print(f"LoRA: {LORA}", flush=True)
print(f"Output: {OUTPUT}", flush=True)

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake as LMD
from peft import PeftModel
import torch

# Step 1: Load base model
print("\n[1/4] Loading base model...", flush=True)
model = LMD.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='cpu',  # CPU: 安全, 不会 OOM
)
print("  Base model loaded on CPU", flush=True)

# Step 2: Load non_lora_trainables (mm_projector + deepfake_projector)
print("[2/4] Loading projector weights...", flush=True)
proj_path = os.path.join(LORA, "non_lora_trainables.bin")
ckpt = torch.load(proj_path, map_location='cpu')
# Strip peft base_model.model. prefix
fixed = {}
for k, v in ckpt.items():
    new_k = k
    # strip "base_model.model.model." → "model."
    if new_k.startswith('base_model.model.model.'):
        new_k = new_k[21:]
    elif new_k.startswith('base_model.model.'):
        new_k = new_k[15:]
    fixed[new_k] = v
missing, unexpected = model.load_state_dict(fixed, strict=False)
print(f"  Projector: loaded {len(fixed)} keys, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
for mk in missing[:5]:
    print(f"    missing: {mk}", flush=True)

# Step 3: Load LoRA
print("[3/4] Loading LoRA...", flush=True)
model = PeftModel.from_pretrained(model, LORA)
print("  LoRA loaded", flush=True)

# Merge and unload LoRA into base weights
print("  Merging LoRA into base weights...", flush=True)
model = model.merge_and_unload()
print("  Merge complete", flush=True)

# Step 4: Save
print("[4/4] Saving merged model...", flush=True)
os.makedirs(OUTPUT, exist_ok=True)

# Fix config
model.config._name_or_path = 'llava-1.5-7b-M2F2-Det-bridge'
model.config.deepfake_model_name = 'vit'
model.config.deepfake_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'checkpoints', 'stage_1', 'bridge_v2_phase1.pth')
model.config.deepfake_model_path = os.path.normpath(model.config.deepfake_model_path)
model.config.mm_vision_select_feature = 'cls_patch'
model.config.tune_deepfake_mlp_adapter = True

model.save_pretrained(OUTPUT, max_shard_size='5GB')

# Also save tokenizer from base
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
tok.save_pretrained(OUTPUT)

print(f"\n[OK] Merged model saved to: {OUTPUT}", flush=True)
total = sum(p.numel() for p in model.parameters()) / 1e9
print(f"  Total params: {total:.1f}B", flush=True)
