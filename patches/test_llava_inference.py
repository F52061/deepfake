"""Quick test: load LLaVA model with deepfake encoder."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llava.model.builder import load_deepfake_model
from llava.mm_utils import get_model_name_from_path

model_path = "./checkpoints/llava-v1.5-7b-M2F2-Det"
model_name = get_model_name_from_path(model_path)

print(f"Model path: {model_path}")
print(f"Model name: {model_name}")
print(f"Loading model (this may take a few minutes)...")

tokenizer, model, image_processor, context_len = load_deepfake_model(
    model_path,
    model_base=None,
    model_name=model_name,
    load_8bit=False,
    load_4bit=False,
    device="cuda"
)

model.eval()
print(f"Model loaded successfully!")
print(f"Context length: {context_len}")

# Load deepfake encoder weights
print(f"Loading deepfake encoder from config path...")
model.load_deepfake_encoder(model.config.deepfake_model_path, verbose=True)
print(f"Deepfake encoder loaded successfully!")

# Test with a single image
from PIL import Image
test_img_path = "./utils/DDVQA_images/c40/test/0_024_073.jpg"
print(f"\nTest inference with: {test_img_path}")
image = Image.open(test_img_path).convert('RGB')
image_size = image.size

from llava.constants import IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX
from llava.mm_utils import tokenizer_hybrid_token

prompt = "Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n <deepfake>\n Determine the authenticity. Is the image real or fake? ###Assistant:"
input_ids = tokenizer_hybrid_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
print(f"Input IDs shape: {input_ids.shape}")

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
        output_hidden_states=True,
        return_dict_in_generate=True
    )
output_ids = output['sequences']
outputs = tokenizer.decode(output_ids[0]).strip()
print(f"\n=== Model Output ===")
print(outputs)
print("=== End ===")

# Save result
os.makedirs('outputs/DDVQA', exist_ok=True)
import json
result = {"image": test_img_path, "text": outputs}
with open('outputs/DDVQA/test_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"Result saved to outputs/DDVQA/test_result.json")
