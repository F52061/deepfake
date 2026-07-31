"""
M2F2-Det Explanatory Inference (with ViT backbone support)

Two weight-loading modes:
  Mode A: Only ViT backbone (no fine-tuning)  ─ uses --vit-ckpt-path
  Mode B: Full pipeline (Phase 1 + Phase 2)   ─ uses --phase1-ckpt + --phase2-ckpt

Usage (Mode B - after both training phases):
  python llava/serve/cli_DDVQA_exp.py

Usage (Mode A - ViT backbone only, no fine-tuning):
  python llava/serve/cli_DDVQA_exp.py --vit-ckpt-path "path/to/net_050.pth"
"""

import argparse
import json
import os
import gc
import torch
# ── Force offline mode to prevent HuggingFace Hub timeouts ─────
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX
from llava.mm_utils import tokenizer_hybrid_token, get_model_name_from_path
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════
# Default paths (modify these to match your environment)
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_MODEL_PATH = "./checkpoints/llava-v1.5-7b-M2F2-Det"
DEFAULT_PHASE1_CKPT = "./vit_module/vit_m2f2_phase1.pth"
DEFAULT_PHASE2_CKPT = "./vit_module/deepfake_projector.pth"


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # ── 1. Load model on CPU (then move vision tower to GPU) ──
    print("Loading model (CPU first)...")
    gc.collect()
    torch.cuda.empty_cache()

    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake
    from transformers import AutoTokenizer

    model_name = get_model_name_from_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLMDeepfake.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
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

    # ── 2. Load deepfake encoder weights ────────────────────────────────
    # Priority: --phase1-ckpt > --vit-ckpt-path > default ViT checkpoint
    phase1_path = args.phase1_ckpt
    vit_path = args.vit_ckpt_path

    if phase1_path and os.path.exists(phase1_path):
        # Mode B: Full Phase 1 trained detector
        print(f"Loading Phase 1 trained detector from: {phase1_path}")
        ckpt = torch.load(phase1_path, map_location='cpu')
        model.deepfake_encoder.load_state_dict(ckpt, strict=False)
        print(f"  Loaded {len(ckpt)} weight tensors.")
    elif vit_path and os.path.exists(vit_path):
        # Mode A: ViT backbone only (new layers randomly initialized)
        print(f"Loading ViT backbone from: {vit_path}")
        model.load_deepfake_encoder(vit_path, verbose=True)
    else:
        print("WARNING: No deepfake encoder weights found. "
              "Check --vit-ckpt-path or --phase1-ckpt.")

    # ── 3. Load Phase 2 bridge MLP (if available) ───────────────────────
    if args.phase2_ckpt and os.path.exists(args.phase2_ckpt):
        print(f"Loading Phase 2 bridge MLP from: {args.phase2_ckpt}")
        ckpt = torch.load(args.phase2_ckpt, map_location='cpu')
        model.deepfake_projector.load_state_dict(ckpt, strict=False)
        print(f"  Loaded {len(ckpt)} bridge MLP weight tensors.")

    # ── 4. Patch DynamicCache for cross-GPU ────────────────────────────
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

    # ── 5. Dispatch across GPUs ─────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Dispatching model across {num_gpus} GPUs...")
        from accelerate import dispatch_model
        layers_per_gpu = (32 + num_gpus - 1) // num_gpus
        device_map = {}
        for i in range(32):
            device_map[f'model.layers.{i}'] = i // layers_per_gpu
        device_map['model.embed_tokens'] = 0
        device_map['model.norm'] = max(0, num_gpus - 1)
        device_map['lm_head'] = max(0, num_gpus - 1)
        device_map['model.vision_tower'] = 1 if num_gpus > 1 else 0
        device_map['model.mm_projector'] = max(0, num_gpus - 1)
        device_map['deepfake_encoder'] = 1 if num_gpus > 1 else 0
        device_map['deepfake_projector'] = 0
        model = dispatch_model(model, device_map=device_map)
        print("Model dispatched.")
    else:
        model = model.cuda()
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(num_gpus):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {(total-free)/1024**3:.1f} GiB / {total/1024**3:.1f} GiB")

    # ── 6. Inference loop ──────────────────────────────────────────────
    eccv_dataset_root = args.image_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    json_path = args.test_json
    output_file = os.path.join(out_dir, 'DDVQA_exp_c40.jsonl')

    conv_mode = "llava_v1"
    if args.conv_mode is not None:
        conv_mode = args.conv_mode

    print(f"\nRunning inference on {json_path}...")
    print(f"Results → {output_file}\n")

    with open(json_path, 'r') as f:
        for image_idx, line in enumerate(tqdm(f)):
            data = json.loads(line)
            key = list(data.keys())[0]
            img_id = "_".join(key.split('_')[:-1])
            image_fn = img_id + ".jpg"
            image_path = os.path.join(eccv_dataset_root, image_fn)
            if not os.path.exists(image_path):
                continue

            question = data[key]['question'].strip()
            image = load_image(image_path)
            image_size = image.size

            prompt = (
                "Assistant: A chat between a curious human and an artificial "
                "intelligence assistant. The assistant gives helpful, detailed, "
                "and polite answers to the human's questions."
                f"###Human: <image>\n<deepfake>\n{question} ###Assistant:"
            )
            input_ids = tokenizer_hybrid_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(model.device)

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
                "question": question,
                "text": outputs,
            }
            with open(output_file, 'a') as fout:
                fout.write(json.dumps(answer) + '\n')

    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2F2-Det inference with ViT backbone")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    # ── Weight paths (with defaults) ──────────────────────────────────
    parser.add_argument("--vit-ckpt-path", type=str, default=None,
        help="ViT backbone weights from train_Ama_aps.py (net_XXX.pth)")
    parser.add_argument("--phase1-ckpt", type=str, default=r'./vit_module/vit_m2f2_phase1.pth',
        help="Phase 1 trained detector (deepfake_encoder_phase1.pth)")
    parser.add_argument("--phase2-ckpt", type=str, default=r'./vit_module/deepfake_projector.pth',
        help="Phase 2 trained bridge MLP (mm_projector.bin)")

    # ── Data paths ────────────────────────────────────────────────────
    parser.add_argument("--test-json", type=str,
        default=r"./utils/DDVQA_eval/c40/test.jsonl",
        help="Test JSONL file with questions")
    parser.add_argument("--image-dir", type=str,
        default=r"./utils/DDVQA_images/c40/test",
        help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="./outputs/DDVQA_exp",
        help="Output directory for results")

    args = parser.parse_args()
    main(args)
