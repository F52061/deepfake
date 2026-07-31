"""
Run all 3 model evaluations sequentially on the SAME benchmark datasets.
No LLM loaded — only detector + CLIP.

Models:
  1. Original pure ViT (net_050.pth)
  2. Cosine integrated (vit_m2f2_phase1.pth)
  3. BridgeAdapter (bridge_phase1.pth)

Output: ./outputs/{original_vit,cosine,bridge}_results.json + console summary
"""
import os, sys, gc, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2, warnings, numpy as np
warnings.filterwarnings('ignore')
from albumentations import Compose, Normalize, ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ── Datasets (same as original eval) ──
TXT_DIR = './dataset/data_2023'
TESTS = [
    ('FF++',  'ffpp_test_split'),
    ('CD1',   'CD1_test'),
    ('CD2',   'CD2_test'),
    ('DFD',   'DFD_test'),
    ('DFR',   'DFR_test'),
    ('DFDC',  'dfdc_test_lip'),
    ('dfdcp', 'dfdcp_test'),
    ('FFIW',  'FFIW_test'),
    ('Wild',  'wild_test'),
    ('Diff',  'diff_test'),
]

class OrigDataset(Dataset):
    """224x224, [-1,1] norm — for original ViT model."""
    def __init__(self, txt_path):
        data = [(p,int(l)) for p,l in [l.strip().split() for l in open(txt_path) if l.strip()] if os.path.exists(p)]
        self.data = data
        self.tfm = Compose([Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)), ToTensorV2()])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx][0], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self.tfm(image=img)['image']
        return img, torch.tensor(self.data[idx][1], dtype=torch.long)

class CLIPDataset(Dataset):
    """336x336, CLIP norm — for cosine/bridge integrated models."""
    def __init__(self, txt_path):
        data = [(p,int(l)) for p,l in [l.strip().split() for l in open(txt_path) if l.strip()] if os.path.exists(p)]
        self.data = data
        self.tfm = Compose([Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]), ToTensorV2()])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx][0], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (336, 336))
        img = self.tfm(image=img)['image']
        return img, torch.tensor(self.data[idx][1], dtype=torch.long)

def evaluate(model, dl, device, use_autocast=True):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in tqdm(dl, leave=False):
        images = images.to(device)
        with torch.no_grad():
            if use_autocast:
                with torch.cuda.amp.autocast():
                    logits = model(images)
            else:
                logits = model(images)
        all_probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().numpy())
        all_labels.extend(labels.numpy())
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    preds = (all_probs>=0.5).astype(int)
    return {
        'n': len(all_labels),
        'n_pos': int((all_labels==1).sum()), 'n_neg': int((all_labels==0).sum()),
        'auc': round(roc_auc_score(all_labels, all_probs)*100, 2),
        'acc': round(accuracy_score(all_labels, preds)*100, 2),
        'f1': round(f1_score(all_labels, preds, zero_division=0)*100, 2),
    }

device = torch.device('cuda:0')

# ── Model configs ──
models = []

# 1. Original ViT
print('='*60, flush=True)
print('Model 1/3: Original ViT (net_050.pth)', flush=True)
print('='*60, flush=True)
from vit_module.vit_adaptive_mattn_aps import vit_base_patch16_224
import torch.nn as nn
net1 = vit_base_patch16_224(pretrained=False, num_classes=2)
net1.load_state_dict(torch.load('./vit_module/net_050.pth', map_location='cpu')['model'] if 'model' in torch.load('./vit_module/net_050.pth', map_location='cpu') else torch.load('./vit_module/net_050.pth', map_location='cpu'))
net1 = net1.to(device).eval()
models.append(('original_vit', net1, OrigDataset, False))

# 2. Cosine integrated
print('='*60, flush=True)
print('Model 2/3: Cosine Integrated (vit_m2f2_phase1.pth)', flush=True)
print('='*60, flush=True)
from vit_module.vit_m2f2_detector import ViT_M2F2Det
clip_local = './checkpoints/clip-vit-large-patch14-336'
net2 = ViT_M2F2Det(clip_text_encoder_name=clip_local, clip_vision_encoder_name=clip_local, hidden_size=1024, load_vision_encoder=True).to(device)
sd2 = torch.load('./vit_module/vit_m2f2_phase1.pth', map_location='cpu')
net2.load_state_dict(sd2, strict=False)
net2.eval()
models.append(('cosine', net2, CLIPDataset, True))

# 3. BridgeAdapter
print('='*60, flush=True)
print('Model 3/3: BridgeAdapter (bridge_phase1.pth)', flush=True)
print('='*60, flush=True)
from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
net3 = ViT_M2F2Det_Bridge(clip_text_encoder_name=clip_local, clip_vision_encoder_name=clip_local, hidden_size=768, load_vision_encoder=True, pretrained=False).to(device)
sd3 = torch.load('./checkpoints/stage_1/bridge_phase1.pth', map_location='cpu')
if 'model_state_dict' in sd3:
    sd3 = sd3['model_state_dict']
sd3 = {k.replace('module.',''):v for k,v in sd3.items()}
net3.load_state_dict(sd3, strict=False)
net3.eval()
models.append(('bridge', net3, CLIPDataset, True))

# ── Run evaluations ──
all_results = {}
for name, model, ds_cls, use_amp in models:
    print(f'\n--- {name} ---', flush=True)
    results = {}
    t0 = time.time()
    for cat, dname in TESTS:
        p = os.path.join(TXT_DIR, dname+'.txt')
        if not os.path.exists(p):
            continue
        ds = ds_cls(p)
        if len(ds)==0:
            continue
        dl = DataLoader(ds, batch_size=8 if 'bridge' in name else (16 if 'cosine' in name else 128), shuffle=False, num_workers=0)
        print(f'  [{cat}] {dname} ({len(ds)})...', flush=True)
        r = evaluate(model, dl, device, use_amp)
        results[dname] = r
        print(f'    AUC={r["auc"]:.2f}%  Acc={r["acc"]:.2f}%  F1={r["f1"]:.2f}%', flush=True)
        gc.collect(); torch.cuda.empty_cache()
    all_results[name] = results
    elapsed = (time.time()-t0)/60
    print(f'  [{name}] Done in {elapsed:.1f} min', flush=True)

# ── Summary ──
print('\n\n'+'='*110)
print('FINAL COMPARISON — 3 Models on Same Datasets')
print('='*110)
header = f'{"Dataset":<22s} {"ORIG AUC":>9s} {"COS AUC":>9s} {"BRG AUC":>9s} {"ORIG Acc":>9s} {"COS Acc":>9s} {"BRG Acc":>9s}'
print(header)
print('-'*len(header))
for cat, dname in TESTS:
    o = all_results.get('original_vit',{}).get(dname,{})
    c = all_results.get('cosine',{}).get(dname,{})
    b = all_results.get('bridge',{}).get(dname,{})
    if not o and not c and not b:
        continue
    o_auc = f'{o["auc"]:.2f}%' if o else '  N/A '
    c_auc = f'{c["auc"]:.2f}%' if c else '  N/A '
    b_auc = f'{b["auc"]:.2f}%' if b else '  N/A '
    o_acc = f'{o["acc"]:.2f}%' if o else '  N/A '
    c_acc = f'{c["acc"]:.2f}%' if c else '  N/A '
    b_acc = f'{b["acc"]:.2f}%' if b else '  N/A '
    print(f'{dname:<22s} {o_auc:>9s} {c_auc:>9s} {b_auc:>9s} {o_acc:>9s} {c_acc:>9s} {b_acc:>9s}')
print('-'*len(header))

# Save
for name in ['original_vit','cosine','bridge']:
    out = f'./outputs/{name}_results.json'
    with open(out,'w') as f:
        json.dump(all_results.get(name,{}), f, indent=2)
    print(f'\nSaved: {out}')
print(f'\nAll done!')
