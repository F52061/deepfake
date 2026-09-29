"""单张推理测试 — 修复 processors=None 与 vision tower 未加载的问题。

根因(2026-08-08):
1. generate() 直接访问 self.processors['clip_processor'/'deepfake_processor']，
   但推理路径 from_pretrained 不会初始化 processors（训练时才通过
   initialize_vision_modules 设置）。已在 llava_llama.py generate() 里加
   _ensure_processors() 惰性初始化。
2. 合并 checkpoint 不含 CLIP vision tower 权重（delay_load=True 初始化时无参数，
   自然没保存）。CLIP 全程冻结，需用 load_model() 从本地 checkpoint 重新加载。
3. deepfake 输入要传原始 PIL 图，让 deepfake_processor 统一预处理（再传已处理
   的 tensor 会被二次归一化）。

GPU 说明:GPU 0 被 deepfake_1 的进程占用（~9.8GB），本脚本用 1,2,3。
"""
import os, sys, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# 把项目根目录加入 sys.path (脚本在 vit_module/ 下, 直接运行会找不到 llava 包)
_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

# CLIP 离线重定向:所有 from_pretrained('openai/clip*') 指向本地 checkpoint
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

MODEL_PATH = "./checkpoints/llava-v1.5-7b-M2F2-Det-bridge"

print(f"Loading model: {MODEL_PATH}", flush=True)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake as LMD

model = LMD.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map="auto",
    max_memory={i: "9GB" for i in range(torch.cuda.device_count())})
model.eval()
print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B", flush=True)

# --- 关键:加载 CLIP vision tower（checkpoint 里没有它的权重）---
vt = model.get_model().get_vision_tower()
if vt is not None and not vt.is_loaded:
    print("Loading CLIP vision tower (not saved in merged checkpoint)...", flush=True)
    vt.load_model(device_map="cuda:0")
    vt.vision_tower.half()  # checkpoint 里 detector/LLM 是 fp16/bf16, tower 默认 fp32 会 dtype 不匹配
    print("  vision tower loaded (fp16)", flush=True)
model._ensure_processors()
print("  processors initialized:", {k: type(v).__name__ for k, v in model.processors.items()}, flush=True)

from PIL import Image
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_hybrid_token

tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# 测试图片(DDVQA test 集中的伪造样本)
img_paths = [
    "utils/DDVQA_images/c40/test/0_024_073.jpg",
    "utils/DDVQA_images/c40/test/0_035_036.jpg",
    "utils/DDVQA_images/c40/test/0_044_945.jpg",
]

for img_path in img_paths:
    if not os.path.exists(img_path):
        continue
    print(f"\n{'='*50}", flush=True)
    print(f"Image: {img_path}", flush=True)

    image = Image.open(img_path).convert('RGB')
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], "<image>\n<deepfake>\nDetermine the authenticity of this image.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_hybrid_token(prompt, tok, return_tensors='pt').unsqueeze(0).cuda()

    with torch.no_grad():
        # deepfake_inputs 传原始 PIL 图,让 deepfake_processor 统一预处理
        out = model.generate(inputs=input_ids, images=[image],
            deepfake_inputs=[image], max_new_tokens=128, do_sample=False)

    text = tok.decode(out[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    print(f"Output: {text[:400]}", flush=True)

print(f"\nDone.", flush=True)
