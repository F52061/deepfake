"""
Hybrid merge: combine HF M2F2-Det (trained LoRA) with our bridge_v2 detector.

Strategy:
  1. Load HF M2F2-Det (llava-v1.5-7b-M2F2-Det) ← LM + LoRA + mm_projector 已训练
  2. Load checkpoint-1700 adapter ← deepfake_projector (trained for bridge_v2)
  3. Replace deepfake_encoder in HF model with our bridge_v2 detector

Result: hybrid model with all 3 components trained
  - HF LoRA: trained on DD-VQA (same data as our Stage-3)
  - checkpoint-1700 projector: trained to map bridge_v2 → LLM embedding
  - bridge_v2 detector: our Stage-1 best result

Usage (must use bat for GPU visibility):
  vit_module\hybrid_merge.bat
"""
import os, sys, torch

# CLIP offline
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

print('=' * 60, flush=True)
print('Hybrid Merge: HF LoRA + bridge_v2 detector + checkpoint-1700 projector', flush=True)
print('=' * 60, flush=True)

# ═══════════════════════════════════════════════════════════
# Step 1: Load HF M2F2-Det (has trained LoRA + LLM + mm_projector)
# ═══════════════════════════════════════════════════════════
print('[1/5] Loading HF M2F2-Det (trained LoRA)...', flush=True)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake as LMD

model = LMD.from_pretrained(
    './checkpoints/llava-v1.5-7b-M2F2-Det',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={i: '10GB' for i in range(torch.cuda.device_count())},
)
print(f'  loaded, device_map entries: {len(getattr(model, "hf_device_map", {}))}', flush=True)

# Verify: HF model should have LoRA in its state
lora_keys = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
print(f'  LoRA keys in HF model: {len(lora_keys)}', flush=True)
if lora_keys:
    print(f'  sample: {lora_keys[0]}', flush=True)

# ═══════════════════════════════════════════════════════════
# Step 2: Load checkpoint-1700 adapter (deepfake_projector trained for bridge_v2)
# ═══════════════════════════════════════════════════════════
print('[2/5] Loading checkpoint-1700 adapter...', flush=True)
adapter_path = './checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-1700/mm_projector.bin'
adapter_sd = torch.load(adapter_path, map_location='cpu')
print(f'  adapter keys: {len(adapter_sd)}', flush=True)

# Only load deepfake_projector keys (not mm_projector — HF already has that)
proj_sd = {k: v for k, v in adapter_sd.items() if 'deepfake_projector' in k}
print(f'  deepfake_projector keys: {len(proj_sd)}', flush=True)
missing, unexpected = model.load_state_dict(proj_sd, strict=False)
print(f'  missing (OK): {len(missing)}, unexpected (OK): {len(unexpected)}', flush=True)

# ═══════════════════════════════════════════════════════════
# Step 3: Replace deepfake_encoder (DenseNet → bridge_v2)
# ═══════════════════════════════════════════════════════════
print('[3/5] Loading bridge_v2 detector...', flush=True)
from vit_module.vit_m2f2_detector_unified import ViT_M2F2Det_Unified

new_encoder = ViT_M2F2Det_Unified(
    fusion_mode='bridge',
    clip_text_encoder_name=clip_local,
    clip_vision_encoder_name=clip_local,
    hidden_size=768,
    load_vision_encoder=True,
    vision_dtype=torch.float16,
    text_dtype=torch.float16,
    deepfake_dtype=torch.float16,
).to('cpu')

# Load bridge_v2 trained weights
bridge_ckpt = torch.load('./checkpoints/stage_1/bridge_v2_phase1.pth', map_location='cpu')
missing, unexpected = new_encoder.load_state_dict(bridge_ckpt, strict=False)
print(f'  bridge_v2 loaded: missing={len(missing)}, unexpected={len(unexpected)}', flush=True)

# Replace in model
old_encoder = model.deepfake_encoder
model.deepfake_encoder = new_encoder
del old_encoder
print('  deepfake_encoder replaced [OK]', flush=True)

# ═══════════════════════════════════════════════════════════
# Step 4: Save the hybrid model
# ═══════════════════════════════════════════════════════════
print('[4/5] Saving hybrid model...', flush=True)
output_dir = './checkpoints/llava-v1.5-7b-bridge-hybrid'
os.makedirs(output_dir, exist_ok=True)

model.config._name_or_path = 'llava-1.5-7b-bridge-hybrid'
model.config.deepfake_model_name = 'vit'
model.config.deepfake_model_path = './checkpoints/stage_1/bridge_v2_phase1.pth'
model.config.tune_deepfake_mlp_adapter = True
model.config.mm_vision_select_feature = 'cls_patch'
model.save_pretrained(output_dir, max_shard_size='5GB')
print(f'  saved to {output_dir}', flush=True)

# ═══════════════════════════════════════════════════════════
# Step 5: Quick inference test
# ═══════════════════════════════════════════════════════════
print('[5/5] Verifying hybrid model...', flush=True)
print(f'  Total params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B', flush=True)
print(f'  deepfake_encoder name: {type(model.deepfake_encoder).__name__}', flush=True)
print(f'  deepfake_encoder.fusion_mode: {getattr(model.deepfake_encoder, "fusion_mode", "N/A")}', flush=True)
print(f'  deepfake_projector: {type(model.deepfake_projector).__name__}', flush=True)
print()
print('=' * 60, flush=True)
print('Hybrid merge complete!', flush=True)
print(f'Model saved to: {output_dir}', flush=True)
print('=' * 60, flush=True)
