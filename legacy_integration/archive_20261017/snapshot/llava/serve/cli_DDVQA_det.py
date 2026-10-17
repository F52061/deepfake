import argparse
import torch
import os
# ── Force offline mode to prevent HuggingFace Hub timeouts ─────
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import gc
import json
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torchvision.transforms import transforms
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEEPFAKE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_deepfake_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, tokenizer_hybrid_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    model_name = get_model_name_from_path(args.model_path)

    # ── CLIP 离线重定向: from_pretrained('openai/clip*') → 本地 checkpoint ──
    _clip_local = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'checkpoints', 'clip-vit-large-patch14-336'))
    if os.path.isdir(_clip_local):
        from transformers import (CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor,
            CLIPTextConfig, CLIPTextModel, AutoConfig, AutoTokenizer)
        def _rdr(f):
            def w(p, *a, **k):
                if 'openai/clip' in str(p): return f(_clip_local, *a, **k)
                return f(p, *a, **k)
            return w
        for _cls in [CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, AutoConfig, AutoTokenizer]:
            _cls.from_pretrained = _rdr(_cls.from_pretrained)
        print(f"CLIP offline redirect: {_clip_local}")

    # ── Step 1: Load model on CPU (then move vision tower to GPU) ──
    print("Loading model (CPU first)...")
    gc.collect()
    torch.cuda.empty_cache()

    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLMDeepfake.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,  # 合并 checkpoint 含 bf16 权重, Pascal 不支持, 统一转 fp16
    )
    model.eval()
    print("Model loaded on CPU. Loading vision tower...")

    # ── Load vision tower from local CLIP ──
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor
    print("Vision tower loaded on GPU.")

    # Output directory (used for heatmaps and results)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Step 2: Load deepfake encoder weights (Phase 1 or ViT backbone)
    phase1_path = args.phase1_ckpt
    if phase1_path and os.path.exists(phase1_path):
        print(f"Loading Phase 1 trained detector from: {phase1_path}")
        ckpt = torch.load(phase1_path, map_location='cpu')
        model.deepfake_encoder.load_state_dict(ckpt, strict=False)
        print(f"  Loaded {len(ckpt)} weight tensors.")
    else:
        print("Loading ViT backbone weights...")
        model.load_deepfake_encoder(args.vit_ckpt_path, verbose=True)

    # Enable heatmap visualization for the detector
#    if hasattr(model.deepfake_encoder, 'save_heatmap'):
#        model.deepfake_encoder.save_heatmap = True
#        model.deepfake_encoder.heatmap_dir = os.path.join(out_dir, 'heatmaps')
#        print(f"Heatmaps will be saved to: {model.deepfake_encoder.heatmap_dir}")

    # Step 2b: Load Phase 2 bridge MLP (if available)
    if args.phase2_ckpt and os.path.exists(args.phase2_ckpt):
        print(f"Loading Phase 2 bridge MLP from: {args.phase2_ckpt}")
        ckpt = torch.load(args.phase2_ckpt, map_location='cpu')
        model.deepfake_projector.load_state_dict(ckpt, strict=False)
        print(f"  Loaded {len(ckpt)} bridge MLP weight tensors.")

    # Step 3: Patch DynamicCache for cross-GPU KV cache (needed before dispatch)
    from transformers.cache_utils import DynamicCache
    def _patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if cache_kwargs is None:
            cache_kwargs = {}
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states
        if self.key_cache[layer_idx] is not None:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    DynamicCache.update = _patched_update

    # Step 4: Dispatch across all GPUs using accelerate (adds cross-device hooks)
    num_gpus = torch.cuda.device_count()
    print(f"Dispatching across {num_gpus} GPUs...")
    from accelerate import dispatch_model
    layers_per_gpu = (32 + num_gpus - 1) // num_gpus
    device_map = {}
    for i in range(32):
        device_map[f'model.layers.{i}'] = i // layers_per_gpu
    device_map.update({
        'model.embed_tokens': 0,
        'model.norm': max(0, num_gpus - 1),
        'lm_head': max(0, num_gpus - 1),
        'model.vision_tower': 1 if num_gpus > 1 else 0,
        'model.mm_projector': max(0, num_gpus - 1),
        'deepfake_encoder': 1 if num_gpus > 1 else 0,
        'deepfake_projector': 0,
    })
    model = dispatch_model(model, device_map=device_map)
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(num_gpus):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {(total-free)/1024**3:.1f} GiB / {total/1024**3:.1f} GiB")

    # ====== Dataset ======
    eccv_dataset_root = './utils/DDVQA_images/c40/test'

    # Get list of all images
    image_files = []
    for root, dirs, files in os.walk(eccv_dataset_root):
        for file in files:
            if file.lower().endswith(('.jpg')):
                image_files.append(os.path.join(root, file))

    # Conv mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # ====== Inference ======
    output_file = os.path.join(out_dir, 'DDVQA_det_c40.jsonl')
    print(f"\nRunning inference on {len(image_files)} images...")
    print(f"Results will be saved to: {output_file}\n")

    for image_idx, image_fn in enumerate(tqdm(image_files)):
        image = load_image(image_fn)
        image_size = image.size

        prompt = f"Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n <deepfake>\n Determine the authenticity. Is the image real or fake? ###Assistant:"
        input_ids = tokenizer_hybrid_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=[image],
                image_sizes=[image_size],
                deepfake_inputs=[image],
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True)
        output_ids = output['sequences']

        outputs = tokenizer.decode(output_ids[0]).strip()
        answer = {
            "image": image_fn,
            "prompt": prompt,
            "text": outputs,
            "metadata": {}
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(answer) + '\n')

    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/llava-v1.5-7b-M2F2-Det")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vit-ckpt-path", type=str,
        default=r"E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth",
        help="ViT backbone checkpoint (net_XXX.pth from train_Ama_aps.py)")
    parser.add_argument("--phase1-ckpt", type=str, default="./vit_module/vit_m2f2_phase1.pth",
        help="Phase 1 trained detector (vit_m2f2_phase1.pth)")
    parser.add_argument("--phase2-ckpt", type=str, default="./vit_module/deepfake_projector.pth",
        help="Phase 2 trained bridge MLP (deepfake_projector.pth)")
    parser.add_argument("--output-dir", type=str, default="./outputs/DDVQA",
        help="Output directory for detection results")
    args = parser.parse_args()
    main(args)
