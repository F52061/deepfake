"""从 checkpoint-1700 重建完整模型并保存 LoRA 权重。

流程:
1. 加载骨架 (llava-1.5-7b-deepfake-rand-proj-v1)  ← 主模型 + detector
2. 加载 checkpoint-1700/mm_projector.bin          ← 训练的 adapter
3. 提取并保存 LoRA state_dict + non_lora_trainables
"""
import os, sys, torch

# CLIP 离线
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

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake as LMD

# Step 1: 加载骨架（含 LoRA adapter）
print('[1/3] Loading skeleton + LoRA...', flush=True)
model = LMD.from_pretrained(
    './checkpoints/llava-1.5-7b-deepfake-rand-proj-v1',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={i: '11GB' for i in range(torch.cuda.device_count())},
)
print(f'  loaded on {torch.cuda.device_count()} GPUs', flush=True)

# Step 2: 加载训练的 adapter (mm_projector + deepfake_projector)
print('[2/3] Loading checkpoint-1700 adapter...', flush=True)
adapter_path = './checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-1700/mm_projector.bin'
sd = torch.load(adapter_path, map_location='cpu')
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'  adapter keys: {len(sd)}, missing: {len(missing)}, unexpected: {len(unexpected)}', flush=True)

# Step 3: 提取并保存 LoRA + non_lora
print('[3/3] Extracting and saving LoRA state...', flush=True)
output_dir = './checkpoints/llava-v1.5-7b-deepfake_stage-3-delta'

# LoRA 状态
lora_sd = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if "lora_" in k}
# 非 LoRA 可训练参数
non_lora_sd = {k: v.detach().cpu().clone() for k, v in model.named_parameters()
               if "lora_" not in k and v.requires_grad}

model.config.save_pretrained(output_dir)
model.save_pretrained(output_dir, state_dict=lora_sd)
torch.save(non_lora_sd, os.path.join(output_dir, 'non_lora_trainables.bin'))

print(f'  LoRA keys: {len(lora_sd)}  ({sum(v.numel() for v in lora_sd.values())/1e6:.1f}M params)', flush=True)
print(f'  Non-LoRA keys: {len(non_lora_sd)}  ({sum(v.numel() for v in non_lora_sd.values())/1e6:.1f}M params)', flush=True)
print(f'  Saved to: {output_dir}', flush=True)
print('Done!', flush=True)
