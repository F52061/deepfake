"""
Quick evaluation: evaluates cosine model on key datasets with fp16 for speed.
Limits each dataset to 3000 samples for quick results.
"""
import os, sys, gc, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from albumentations import Compose, Normalize, ToTensorV2
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
MAX_SAMPLES = 3000

class QuickDataset(Dataset):
    def __init__(self, txt_path):
        data = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append((parts[0], int(parts[1])))
        self.data = [(p, l) for p, l in data if os.path.exists(p)][:MAX_SAMPLES]
        self.tfm = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])
        self.name = os.path.splitext(os.path.basename(txt_path))[0]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        fn, label = self.data[idx]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (336, 336))
        img = self.tfm(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long)

device = torch.device('cuda:0')
print('Loading model...', flush=True)
from vit_module.vit_m2f2_detector import ViT_M2F2Det
model = ViT_M2F2Det(
    clip_text_encoder_name='./checkpoints/clip-vit-large-patch14-336',
    clip_vision_encoder_name='./checkpoints/clip-vit-large-patch14-336',
    hidden_size=1024, load_vision_encoder=True,
).to(device)
model.load_state_dict(torch.load('./vit_module/vit_m2f2_phase1.pth', map_location='cpu'), strict=False)
model.eval()

datasets = [
    ('FF++', 'ffpp_test_split'),
    ('Celeb-DF', 'CD1_test'),
    ('Celeb-DF', 'CD2_test'),
    ('DFD/DFR', 'DFD_test'),
    ('DFD/DFR', 'DFR_test'),
    ('DFDC', 'dfdc_test_lip'),
    ('DFDC', 'dfdcp_test'),
    ('FFIW', 'FFIW_test'),
    ('Wild', 'wild_test'),
    ('Diff', 'diff_test'),
]
txt_dir = './dataset/data_2023'

results = {}
for cat, name in datasets:
    p = os.path.join(txt_dir, name + '.txt')
    if not os.path.exists(p):
        continue
    ds = QuickDataset(p)
    if len(ds) == 0:
        continue
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    print(f'[{cat}] {name} ({len(ds)} samples)...', flush=True)
    all_probs, all_labels = [], []
    t0 = time.time()
    for images, labels in tqdm(dl, leave=False):
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    elapsed = time.time() - t0
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs) * 100
    acc = accuracy_score(all_labels, preds) * 100
    f1 = f1_score(all_labels, preds, zero_division=0) * 100
    results[name] = {'n': len(ds), 'auc': auc, 'acc': acc, 'f1': f1, 'time': f'{elapsed:.0f}s'}
    print(f'  AUC={auc:.2f}%  Acc={acc:.2f}%  F1={f1:.2f}%  ({elapsed:.0f}s)', flush=True)
    gc.collect(); torch.cuda.empty_cache()

print('\n' + '=' * 60)
print('RESULTS')
print('=' * 60)
print(f'{"Dataset":25s} {"N":>6s} {"AUC%":>7s} {"Acc%":>7s} {"F1%":>7s} {"Time":>8s}')
print('-' * 60)
for cat, name in datasets:
    r = results.get(name)
    if r:
        print(f'{name:25s} {r["n"]:>6d} {r["auc"]:6.2f}% {r["acc"]:6.2f}% {r["f1"]:6.2f}% {r["time"]:>8s}')
print('-' * 60)

with open('./outputs/cosine_results.json', 'w') as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
print(f'\nSaved to ./outputs/cosine_results.json')
