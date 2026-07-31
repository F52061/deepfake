"""Full inference test: CPU load + accelerate dispatch to GPUs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import patches.flash_attn_patch
import patches.clip_offline_patch

import torch, gc
from PIL import Image
from llava.model.builder import load_deepfake_model
from llava.mm_utils import get_model_name_from_path, tokenizer_hybrid_token
from llava.constants import IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX

# 1. Load model entirely on CPU (device='cpu' passes device_map={"":"cpu"} to from_pretrained)
model_path = './checkpoints/llava-v1.5-7b-M2F2-Det'
print("Loading model on CPU...")
tokenizer, model, image_processor, context_len = load_deepfake_model(
    model_path, None, get_model_name_from_path(model_path), device='cpu'
)
model.eval()
print("Model loaded on CPU.")

# 2. Load deepfake encoder weights
print("Loading deepfake encoder...")
ckpt = torch.load(model.config.deepfake_model_path, map_location='cpu')
state_dict = {}
for k, v in model.deepfake_encoder.state_dict().items():
    if k in ckpt:
        state_dict[k] = ckpt[k].to(v.dtype)
missing, unexpected = model.deepfake_encoder.load_state_dict(state_dict, strict=False)
print(f'Missing: {missing}, Unexpected: {unexpected}')

# 3. Patch KV cache to handle cross-device
# Patch: keep KV cache on same device as the layer's output
from transformers.cache_utils import DynamicCache
_original_update = DynamicCache.update
def patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    if cache_kwargs is None:
        cache_kwargs = {}
    if len(self.key_cache) <= layer_idx:
        # New entry - just append
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return key_states, value_states
    if self.key_cache[layer_idx] is not None:
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
    # Now do the original cat
    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
    return self.key_cache[layer_idx], self.value_cache[layer_idx]
DynamicCache.update = patched_update

# 4. Dispatch to GPUs with accelerate
gc.collect()
from accelerate import dispatch_model
print("Dispatching to GPUs...")

# Manual device map for reliability
device_map = {
    'model.embed_tokens': 0,
    'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1,
    'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1,
    'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1,
    'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1,
    'model.layers.16': 2, 'model.layers.17': 2, 'model.layers.18': 2, 'model.layers.19': 2,
    'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2,
    'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2,
    'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 2,
    'model.norm': 2,
    'model.vision_tower': 0,
    'model.mm_projector': 0,
    'lm_head': 0,
    'deepfake_encoder': 0,
    'deepfake_projector': 0,
}
model = dispatch_model(model, device_map=device_map)
torch.cuda.empty_cache()

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {(total-free)/1024**3:.1f} GiB / {total/1024**3:.1f} GiB")
print("\n=== MODEL READY ===\n")

# 4. Run inference
test_img_path = "./utils/DDVQA_images/c40/test/0_024_073.jpg"
image = Image.open(test_img_path).convert('RGB')
image_size = image.size
print(f"Test image: {test_img_path}")

prompt = ("Assistant: A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions."
          "###Human: <image>\n <deepfake>\n Determine the authenticity. Is the image real or fake? ###Assistant:")

input_ids = tokenizer_hybrid_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).cuda()

print("Running inference...")
with torch.inference_mode():
    output = model.generate(
        input_ids,
        images=[image],
        image_sizes=[image_size],
        deepfake_inputs=[image],
        do_sample=False,
        num_beams=1,
        max_new_tokens=512,
        use_cache=True,
        return_dict_in_generate=True
    )

output_text = tokenizer.decode(output['sequences'][0]).strip()
print(f"\n{'='*60}")
print("MODEL OUTPUT:")
print(f"{'='*60}")
print(output_text)
print(f"{'='*60}")

os.makedirs('outputs/DDVQA', exist_ok=True)
import json
with open('outputs/DDVQA/test_result.json', 'w', encoding='utf-8') as f:
    json.dump({"image": test_img_path, "text": output_text}, f, indent=2)
print("Result saved!")
