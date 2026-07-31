"""Debug: run 3 real training batches through the model, print loss + label mask stats."""
import os, sys, torch
sys.path.insert(0, '.')

clip_local = os.path.normpath('./checkpoints/clip-vit-large-patch14-336')
from transformers import (CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor,
    CLIPTextConfig, CLIPTextModel, AutoConfig, AutoTokenizer)
def rdr(f):
    def w(p,*a,**k):
        if 'openai/clip' in str(p): return f(clip_local,*a,**k)
        return f(p,*a,**k)
    return w
for cls in [CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, AutoConfig, AutoTokenizer]:
    cls.from_pretrained = rdr(cls.from_pretrained)

print('Loading model...', flush=True)
import torch
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLMDeepfake
model = LlavaLlamaForCausalLMDeepfake.from_pretrained(
    './checkpoints/llava-1.5-7b-deepfake-rand-proj-v1',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={i: '11GB' for i in range(torch.cuda.device_count())},
)
model.eval()
print('Model loaded.', flush=True)

from llava.train.train_deepfake import LazySupervisedDataset, DataCollatorForSupervisedDataset, DataArguments
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1', use_fast=False)

da = DataArguments()
da.data_path = './utils/DDVQA_split/c40/train_DDVQA_format.json'
da.image_folder = './utils/DDVQA_images/c40/train'
da.is_multimodal = True
da.image_processor = CLIPImageProcessor.from_pretrained(clip_local)  # fix: set image_processor
da.mm_use_im_start_end = False
da.mm_use_im_patch_token = False

ds = LazySupervisedDataset(data_path=da.data_path, tokenizer=tok, data_args=da)
print('Dataset size:', len(ds), flush=True)
collate = DataCollatorForSupervisedDataset(tokenizer=tok)

for i in range(3):
    sample = ds[i]
    batch = collate([sample])
    # Move tensors to device; keep images list on CPU (model handles)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    # Count non-IGNORE labels
    labels = batch['labels']
    n_masked = (labels == -100).sum().item()
    print(f'Batch {i}: input_ids={batch["input_ids"].shape} labels_unmasked={labels.numel()-n_masked}/{labels.numel()}', flush=True)
    with torch.no_grad():
        try:
            out = model(**batch)
            print(f'  loss={out.loss}', flush=True)
        except Exception as e:
            print(f'  ERROR {type(e).__name__}: {str(e)[:250]}', flush=True)
print('Done.', flush=True)
