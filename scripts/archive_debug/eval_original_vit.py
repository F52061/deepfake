"""
Evaluate original MainModel (pure ViT classifier) on benchmark datasets.
Loads net_050.pth, runs on 224x224 [-1,1] normalized images.

Usage:
    python vit_module/eval_original_vit.py
"""
import os, sys, gc, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from albumentations import Compose, Normalize, ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# ── ViT backbone + classifier (same as MainModel in train_Ama_aps.py) ──
from vit_module.vit_adaptive_mattn_aps import vit_base_patch16_224
import torch.nn as nn

class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_base_patch16_224(pretrained=False, num_classes=2)
    def forward(self, x):
        return self.model.forward(x)

# ── Dataset (224x224, [-1,1] normalization) ──
class OrigDataset(Dataset):
    def __init__(self, txt_path, max_samples=None):
        data = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append((parts[0], int(parts[1])))
        self.data = [(p, l) for p, l in data if os.path.exists(p)]
        if max_samples:
            self.data = self.data[:max_samples]
        self.tfm = Compose([Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)), ToTensorV2()])
        self.name = os.path.splitext(os.path.basename(txt_path))[0]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        fn, label = self.data[idx]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self.tfm(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long)

def compute_eer(labels, scores):
    fpr_list, tpr_list = [], []
    idx = np.argsort(scores)[::-1]
    labels_s = labels[idx]
    n_pos, n_neg = (labels==1).sum(), (labels==0).sum()
    if n_pos==0 or n_neg==0:
        return float('nan')
    tp, fp, fn, tn = 0, 0, n_pos, n_neg
    for i in range(len(labels_s)):
        if labels_s[i]==1: tp+=1; fn-=1
        else: fp+=1; tn-=1
        fpr_list.append(fp/n_neg)
        tpr_list.append(tp/n_pos)
    diff = np.abs(np.array(fpr_list) - (1.0-np.array(tpr_list)))
    return (fpr_list[np.argmin(diff)] + (1.0-tpr_list[np.argmin(diff)]))/2*100

device = torch.device('cuda:0')

# ── Build original model & load weights ──
print('Building original MainModel...', flush=True)
net = MainModel()
ckpt = torch.load('./vit_module/net_050.pth', map_location='cpu')
# net_050.pth saved as net.state_dict() → keys have 'model.' prefix
net.load_state_dict(ckpt, strict=True)
net = net.to(device)
net.eval()
print(f'Loaded net_050.pth ({sum(p.numel() for p in net.parameters())/1e6:.1f}M params)', flush=True)

# ── Test datasets ──
txt_dir = './dataset/data_2023'
tests = [
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

results = {}
for cat, name in tests:
    p = os.path.join(txt_dir, name+'.txt')
    if not os.path.exists(p):
        continue
    ds = OrigDataset(p)
    if len(ds)==0:
        continue
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    print(f'[{cat}] {name} ({len(ds)} samples)...', flush=True)

    all_probs, all_labels = [], []
    for images, labels in tqdm(dl, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = net(images)
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs>=0.5).astype(int)

    n = len(all_labels)
    n_pos = (all_labels==1).sum()
    n_neg = (all_labels==0).sum()
    auc = roc_auc_score(all_labels, all_probs)*100
    acc = accuracy_score(all_labels, preds)*100
    f1 = f1_score(all_labels, preds, zero_division=0)*100
    prec = precision_score(all_labels, preds, zero_division=0)*100
    rec = recall_score(all_labels, preds, zero_division=0)*100
    eer = compute_eer(all_labels, all_probs)

    results[name] = {'n':n,'n_pos':int(n_pos),'n_neg':int(n_neg),'auc':auc,'acc':acc,'f1':f1,'prec':prec,'rec':rec,'eer':eer}
    print(f'  AUC={auc:.2f}%  Acc={acc:.2f}%  F1={f1:.2f}%  EER={eer:.2f}%', flush=True)
    gc.collect(); torch.cuda.empty_cache()

# ── Summary ──
print('\n' + '='*90)
print('ORIGINAL MAIN MODEL (pure ViT, net_050.pth)')
print('='*90)
print(f'{"Dataset":<25s} {"N":>7s} {"AUC%":>8s} {"Acc%":>7s} {"F1%":>7s} {"EER%":>7s} {"Prec%":>7s} {"Rec%":>7s}')
print('-'*90)
for cat, name in tests:
    r = results.get(name)
    if r:
        print(f'{name:<25s} {r["n"]:>7d} {r["auc"]:7.2f}% {r["acc"]:6.2f}% {r["f1"]:6.2f}% {r["eer"]:6.2f}% {r["prec"]:6.2f}% {r["rec"]:6.2f}%')
print('-'*90)
vals = [r for r in results.values()]
if vals:
    o_auc = np.mean([x['auc'] for x in vals])
    o_acc = np.mean([x['acc'] for x in vals])
    o_f1  = np.mean([x['f1'] for x in vals])
    o_eer = np.mean([x['eer'] for x in vals if not np.isnan(x['eer'])])
    print(f'{"AVERAGE":<25s} {"":>7s} {o_auc:7.2f}% {o_acc:6.2f}% {o_f1:6.2f}% {o_eer:6.2f}%')

with open('./outputs/original_vit_results.json','w') as f:
    json.dump({k:{kk:float(vv) if isinstance(vv,(float,np.floating)) else vv for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)
print(f'\nSaved to ./outputs/original_vit_results.json')
