"""
Quick test: Run 10 training steps with ViT_M2F2Det_Bridge to verify pipeline works.

Usage:
    python vit_module/quick_test_train.py
"""
import os, sys, gc, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, torch.nn as nn
import numpy as np
from torch.optim import AdamW
from albumentations import Compose, Normalize, ToTensorV2
import cv2
from tqdm import tqdm

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def load_sample_batch(txt_path, n=8):
    """Load n samples from txt file."""
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and os.path.exists(parts[0]):
                data.append((parts[0], int(parts[1])))
            if len(data) >= n:
                break
    tf = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])
    images, labels = [], []
    for fn, lb in data:
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (336, 336))
        img = tf(image=img)['image']
        images.append(img)
        labels.append(torch.tensor(lb, dtype=torch.long))
    return torch.stack(images), torch.stack(labels)


def main():
    device = torch.device('cuda:0')
    clip_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'clip-vit-large-patch14-336')

    print('=' * 60)
    print('Step 1: Building ViT_M2F2Det_Bridge...')
    print('=' * 60)
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge

    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=clip_path,
        clip_vision_encoder_name=clip_path,
        hidden_size=768, load_vision_encoder=True,
        pretrained=False,
    ).cuda()

    print('\nStep 2: Loading ViT backbone from net_050.pth...')
    model.load_vit_backbone('./vit_module/net_050.pth', verbose=True)

    # Freeze
    for p in model.clip_vision_encoder.parameters():
        p.requires_grad = False
    for p in model.clip_text_encoder.model.parameters():
        p.requires_grad = False
    model.clip_text_encoder.prompt_tokens.requires_grad = True

    gc.collect(); torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    print(f'GPU: {(total-free)/1024**3:.1f} GiB used / {total/1024**3:.1f} GiB')

    # Optimizer
    param_groups = model.assign_lr_dict_list(lr=1e-4)
    opt = AdamW(param_groups)
    criterion = nn.CrossEntropyLoss()

    # Load data
    print('\nStep 3: Loading sample batch...')
    images, labels = load_sample_batch('./dataset/data_2023/ffpp_train_split.txt', n=6)
    images, labels = images.cuda(), labels.cuda()
    print(f'Images: {images.shape}, Labels: {labels.tolist()}')

    # Train 10 steps
    print('\nStep 4: Training 10 steps...')
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    t_start = time.time()
    for step in range(10):
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        acc = (out.argmax(1) == labels).float().mean().item()
        print(f'  Step {step+1:2d}: loss={loss.item():.4f} acc={acc:.2%}')
    t_elapsed = time.time() - t_start
    print(f'\n10 steps in {t_elapsed:.1f}s ({t_elapsed/10:.1f}s/step)')

    # Save
    out_path = './checkpoints/stage_1/bridge_phase1_test.pth'
    torch.save(model.state_dict(), out_path)
    print(f'\nStep 5: Saved to {out_path}')
    print('\nPipeline verified OK! Ready for full training.')


if __name__ == '__main__':
    main()
