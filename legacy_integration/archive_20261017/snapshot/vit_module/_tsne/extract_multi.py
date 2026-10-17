# -*- coding: utf-8 -*-
"""
Task #3 — feature extraction for cross-domain t-SNE visualisation.

Loads the Stage-1 detector (bridge_v2_phase1.pth) and, per image, extracts:
    F : classifier head INPUT = cat[clip_vision_cls(768, proj, *alpha),
                                  bridge_adapter_embed(128, *text_alpha),
                                  vit_features(768, proj)]   -> 1664-d
    V : raw PDI-ViT final CLS token (forward_features[:,0,:])  -> 768-d
    C : raw CLIP vision CLS token (hidden_states[-2][:,0,:])   -> 1024-d
plus label y (1=real, 0=fake), domain, vid and full image path.

Datasets (txt label: real=1, fake=0):
    FF++: ffpp_test_split   real 400 + fake 400
    CD1_test / CD2_test / dfdcp_test / FFIW_test / wild_test : real ~150 + fake ~150 each

CPU/GPU discipline: OMP..=1 etc. set before imports; only GPU 0; fp32; no_grad;
DataLoader num_workers=0. The flash-attn MHA is replaced by the pure-PyTorch
shim (same parameter names) so the bridge can run on this GPU.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import random
from collections import defaultdict

import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)
from albumentations import Compose, Normalize, ToTensorV2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

TXT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'data_2023')
CKPT = os.path.join(PROJECT_ROOT, 'checkpoints', 'stage_1', 'bridge_v2_phase1.pth')
CLIP_LOCAL = os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336')
OUT_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feats_multi.npz')

# domain -> (txt name, real_cap, fake_cap)
DOMAINS = [
    ('ffpp',  'ffpp_test_split', 400, 400),
    ('cd1',   'CD1_test',        150, 150),
    ('cd2',   'CD2_test',        150, 150),
    ('dfdcp', 'dfdcp_test',      150, 150),
    ('ffiw',  'FFIW_test',       150, 150),
    ('wild',  'wild_test',       150, 150),
]


# ---------------------------------------------------------------------------
# sampling / dataset
# ---------------------------------------------------------------------------
def vid_key(path):
    leaf = os.path.basename(os.path.dirname(path))
    return leaf.split('_')[0]


def load_lines(txt_name):
    p = os.path.join(TXT_DIR, txt_name + '.txt')
    if not os.path.exists(p):
        return None, 'missing-txt'
    rows = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rows.append((parts[0], int(parts[1])))
    return rows, None


def sample_class(rows_class, cap, rng):
    """rows_class: list of (path,label). Spread across videos; return list of (path,label,vid)."""
    by_vid = defaultdict(list)
    for path, lab in rows_class:
        by_vid[vid_key(path)].append((path, lab))
    vids = list(by_vid.keys())
    rng.shuffle(vids)
    idx = {v: 0 for v in vids}
    pick = []
    while len(pick) < cap:
        added = False
        for v in vids:
            if idx[v] < len(by_vid[v]):
                path, lab = by_vid[v][idx[v]]
                idx[v] += 1
                pick.append((path, lab, v))
                added = True
                if len(pick) >= cap:
                    break
        if not added:
            break
    return pick


class ImgDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries  # (path, label, vid)
        self.transform = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label, vid = self.entries[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((336, 336, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (336, 336))
        t = self.transform(image=img)['image']
        return t, torch.tensor(label, dtype=torch.long), path, vid


def _collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
            [b[2] for b in batch], [b[3] for b in batch])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model():
    import vit_module.vit_m2f2_detector_bridge as br
    from vit_module.flash_attn_shim.mha import MHA as ShimMHA
    br.MHA = ShimMHA  # force pure-PyTorch MHA (same param names) for this GPU
    model = br.ViT_M2F2Det_Bridge(
        clip_text_encoder_name=CLIP_LOCAL,
        clip_vision_encoder_name=CLIP_LOCAL,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
    )
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in ck.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f'[model] load missing={len(miss)} unexpected={len(unexp)}')
    return model


@torch.no_grad()
def extract_batch_feats(model, images):
    """Return F (classifier input 1664-d), V (raw ViT CLS 768-d), C (raw CLIP CLS 1024-d)."""
    B = images.shape[0]
    device = images.device

    vit_input = model._preprocess_for_vit(images).to(model.vit_dtype)
    vit_out = model.vit.forward_features(vit_input)            # [B,197,768]
    vit_cls_token = vit_out[:, 0, :]                           # [B,768]
    V = vit_cls_token.float()
    vit_features = model.deepfake_proj(vit_cls_token)          # [B,768]

    vit_feat_0 = model.vit_block_outputs['b_1'][:, 1:, :]
    vit_feat_1 = model.vit_block_outputs['b_2'][:, 1:, :]
    vit_feat_2 = model.vit_block_outputs['b_3'][:, 1:, :]

    clip_0, clip_1, clip_2, clip_vis_feats = model.clip_vision_encoder(images)
    C = clip_vis_feats[:, 0, :].float()                        # [B,1024]
    clip_vision_cls = model.vision_proj(clip_vis_feats.float())[:, 0, :]  # [B,768]

    vit_feat_lst = [vit_feat_0, vit_feat_1, vit_feat_2]
    clip_feat_lst = [clip_0, clip_1, clip_2]
    bridge_out = None
    for i, (vf, cf) in enumerate(zip(vit_feat_lst, clip_feat_lst)):
        cf = model.clip_reduction(cf.to(device))               # [B,576,64]
        vf = model.linear_vit_lst[i](vf.to(device))            # [B,196,64]
        if bridge_out is None:
            combined = torch.cat((vf, cf), dim=1)              # [B,772,64]
        else:
            bridge_out = bridge_out.permute(1, 0, 2)           # [B,prev,64]
            combined = torch.cat((bridge_out, vf, cf), dim=1)
        combined = combined.permute(1, 0, 2)                   # [seq,B,64]
        bridge_out = model.bridge_adapter[i](combined)

    clip_adapt_embed = model.bridge_adapter_proj(bridge_out, B)     # [B,128]
    clip_adapt_embed = model.clip_text_alpha * clip_adapt_embed

    clip_vision_cls = model.clip_vision_alpha * clip_vision_cls.to(model.deepfake_dtype)
    F = torch.cat([clip_vision_cls, clip_adapt_embed, vit_features], dim=-1)  # [B,1664]
    return F.float().cpu(), V.cpu(), C.cpu()


def extract_domain(model, entries, device, batch_size):
    ds = ImgDataset(entries)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    collate_fn=_collate)
    Fs, Vs, Cs, ys, paths, vids = [], [], [], [], [], []
    for images, labels, pth, vid in tqdm(dl, desc='extract', leave=False):
        images = images.to(device)
        F, V, C = extract_batch_feats(model, images)
        Fs.append(F.numpy().astype(np.float32))
        Vs.append(V.numpy().astype(np.float32))
        Cs.append(C.numpy().astype(np.float32))
        ys.append(labels.numpy())
        paths.extend(pth)
        vids.extend(vid)
    return (np.concatenate(Fs).astype(np.float32),
            np.concatenate(Vs).astype(np.float32),
            np.concatenate(Cs).astype(np.float32),
            np.concatenate(ys).astype(np.int64),
            np.array(paths), np.array(vids))


def main():
    seed = 1234
    rng = random.Random(seed)
    device = torch.device('cuda:0')
    print('[info] build model')
    model = build_model()
    model.to(device).eval()
    print(f'[info] device={device}')

    F_all, V_all, C_all, y_all = [], [], [], []
    domain_all, path_all, vid_all = [], [], []
    batch_size = 8

    for dname, txt, real_cap, fake_cap in DOMAINS:
        rows, err = load_lines(txt)
        if err:
            print(f'[skip] {dname}: {err}')
            continue
        # rows might contain missing files -> filter
        exist = [(p, l) for (p, l) in rows if os.path.exists(p)]
        real = sample_class([(p, l) for (p, l) in exist if l == 1], real_cap, rng)
        fake = sample_class([(p, l) for (p, l) in exist if l == 0], fake_cap, rng)
        entries = real + fake
        rng.shuffle(entries)
        if not entries:
            print(f'[skip] {dname}: no usable files')
            continue
        print(f'[extract] {dname:6s} n={len(entries)} (real={sum(1 for e in entries if e[1]==1)}, '
              f'fake={sum(1 for e in entries if e[1]==0)})')
        F, V, C, y, paths, vids = extract_domain(model, entries, device, batch_size)
        F_all.append(F); V_all.append(V); C_all.append(C); y_all.append(y)
        domain_all.extend([dname] * len(y))
        path_all.extend(paths); vid_all.extend(vids)

    F_all = np.concatenate(F_all).astype(np.float32)
    V_all = np.concatenate(V_all).astype(np.float32)
    C_all = np.concatenate(C_all).astype(np.float32)
    y_all = np.concatenate(y_all).astype(np.int64)
    domain_all = np.array(domain_all)
    path_all = np.array(path_all)
    vid_all = np.array(vid_all)

    print(f'\n[total] {len(y_all)} images  F{F_all.shape} V{V_all.shape} C{C_all.shape}')
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        F=F_all, V=V_all, C=C_all, y=y_all,
        domain=domain_all, vid=vid_all, path=path_all,
        domains=np.array([d[0] for d in DOMAINS]),
    )
    print(f'[save] {OUT_NPZ}')


if __name__ == '__main__':
    main()
