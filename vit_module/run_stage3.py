"""Stage-3 LoRA fine-tuning launcher.
3-GPU fp32 (avoids fp16 GradScaler NaN issue on frozen-backbone models,
and 1080 Ti has no bf16 support). 7B fp32 = ~28GB over 3x11.8GB cards.
Starts from the rand-proj skeleton (contains bridge_v2 detector + LLaVA base).

Usage:
    python vit_module/run_stage3.py   (run via run_stage3.bat)
"""
import os, sys

# NOTE: CUDA_VISIBLE_DEVICES must be set via the bat launcher (run_stage3.bat).
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)

import torch
print(f"GPUs: {torch.cuda.device_count()}", flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1e9:.1f}GB", flush=True)

# ═══════════════════════════════════════════════════════════
# Redirect CLIP models to local cache (offline environment)
# ═══════════════════════════════════════════════════════════
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
    print(f'CLIP offline redirect: {clip_local}')
else:
    print(f'WARNING: CLIP local not found: {clip_local}')

# ═══════════════════════════════════════════════════════════
# fp16 training with REAL GradScaler (not monkeypatched).
# The fix in train_deepfake.py casts all TRAINABLE params (LoRA +
# projectors) to fp32, so GradScaler can properly scale gradients.
# Frozen LLaMA backbone stays fp16. This prevents the gradient
# overflow/nan that occurred with the no-op scaler.
# ═══════════════════════════════════════════════════════════

sys.argv = [
    "train_deepfake.py",
    # ---- 模型 ----
    "--model_name_or_path", "./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1",
    "--version", "v1",
    # ---- 数据 (Stage-3: 完整 DD-VQA 含解释) ----
    "--data_path", "./utils/DDVQA_split/c40/train_DDVQA_format.json",
    "--image_folder", "./utils/DDVQA_images/c40/train",
    # ---- 视觉/检测器 ----
    "--vision_tower", "openai/clip-vit-large-patch14-336",
    "--deepfake_ckpt_path", "./checkpoints/stage_1/bridge_v2_phase1.pth",
    # ---- 训练配置 ----
    "--lora_enable", "True",
    "--lora_r", "128",
    "--lora_alpha", "256",
    "--mm_projector_lr", "2e-5",
    "--tune_mm_mlp_adapter", "True",
    "--tune_deepfake_mlp_adapter", "True",
    "--mm_projector_type", "mlp2x_gelu",
    "--mm_vision_select_layer", "-2",
    "--mm_vision_select_feature", "cls_patch",
    "--mm_use_im_start_end", "False",
    "--mm_use_im_patch_token", "False",
    "--fp16", "True",
    "--output_dir", "./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", "16",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "100",
    "--save_total_limit", "2",
    "--learning_rate", "2e-5",
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--model_max_length", "2048",
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "0",
    "--lazy_preprocess", "True",
]

exec(open("llava/train/train_deepfake.py", encoding="utf-8").read())
