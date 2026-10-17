# -*- coding: utf-8 -*-
"""
R-test / residual probe for the M2F2-Det Stage-1 detector (ViT_M2F2Det_Bridge).

Goal: test the hypothesis
    "the frozen deepfake ViT branch carries fake-discriminative residual
     information that the frozen CLIP vision representation cannot linearly
     cover (complementarity)."

We extract two feature vectors per image from the two frozen towers BEFORE
the BridgeAdapter / before the trained projection heads (the "raw" branch
representations), plus their projected (head-input) versions:

    C = raw CLIP vision tower final/pooled CLS feature   (hs[-2], [B,577,1024] -> CLS, 1024-d)
    V = raw ViT (PDI-initialized) final CLS feature      (forward_features[:,0,:], 768-d)

Then:
    - probe(V): L2-logistic AUC on V  (does ViT branch alone separate fake?)
    - probe(C): L2-logistic AUC on C  (CLIP-alone baseline)
    - Ridge C->V  ->  residual R = V - pred(V|C)  (fit on train segment, applied everywhere)
    - probe(R): L2-logistic AUC on R  (** core criterion **)
    - CKA(V,C)  and effective ranks of V, C, R

Only GPU 0 is used (CUDA_VISIBLE_DEVICES=0), fp32, torch.no_grad, batch<=32.

CPU discipline (must be before any numpy/torch/cv2/sklearn import):
    OMP/MKL/OPENBLAS/NUMEXPR/JOBLIB threads forced to 1,
    torch.set_num_threads(1), cv2.setNumThreads(0), DataLoader num_workers=0,
    sklearn solvers single-threaded.

Outputs (in this dir):
    probe_feats.npz          (V, C, R, y, paths, train_mask, video_keys, ... )
    residual_probe_report.txt
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
import argparse
from collections import defaultdict

import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)

from albumentations import Compose, Normalize, ToTensorV2

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Dataset (mirrors vit_module/eval_all_datasets.py EvalDataset: CLIP norm 336)
# ---------------------------------------------------------------------------
class FeatDataset(Dataset):
    def __init__(self, entries, size=336):
        # entries: list of (path, label, video_key)
        self.entries = entries
        self.size = size
        self.transform = Compose([
            Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, vid = self.entries[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:                      # fallback for broken images
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        t = self.transform(image=image)['image']
        return t, torch.tensor(label, dtype=torch.long), path, vid


# ---------------------------------------------------------------------------
# Video-level split helpers
# ---------------------------------------------------------------------------
def load_entries(txt_path):
    """Read txt (path label). Returns list of (path, label, video_key).

    video_key: the source-video identity = leaf dir prefix before '_'.
    Real:     .../original_sequences/c23/faces23/<id>/<frame>.png      -> id
    Fake:     .../manipulated_sequences/<m>/c23/faces23/<id>_<t>/f.png -> id
    This keeps frames of the same real identity (real + its fakes) on one side.
    """
    entries = []
    n_skip = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            p, lab = parts[0], int(parts[1])
            if not os.path.exists(p):
                n_skip += 1
                continue
            leaf = os.path.basename(os.path.dirname(p))
            vid = leaf.split('_')[0]
            entries.append((p, lab, vid))
    if n_skip:
        print(f'[load] skipped {n_skip} missing files from {txt_path}')
    print(f'[load] {txt_path}: {len(entries)} existing entries')
    return entries


def video_split(entries, train_frac=0.7, seed=1234):
    """Group entries by video_key, split video_keys 70/30, return train/test entries."""
    groups = defaultdict(list)
    for e in entries:
        groups[e[2]].append(e)
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_tr = int(round(len(keys) * train_frac))
    tr_keys = set(keys[:n_tr])
    te_keys = set(keys[n_tr:])
    tr = [e for k in tr_keys for e in groups[k]]
    te = [e for k in te_keys for e in groups[k]]
    return tr, te, len(tr_keys), len(te_keys)


def cap_per_class(entries, cap, seed=1234):
    rng = random.Random(seed)
    by_lab = defaultdict(list)
    for e in entries:
        by_lab[e[1]].append(e)
    out = []
    for lab in sorted(by_lab.keys()):
        arr = by_lab[lab]
        if len(arr) > cap:
            arr = rng.sample(arr, cap)
        out.extend(arr)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(clip_local, ckpt_path):
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=clip_local,
        clip_vision_encoder_name=clip_local,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    else:
        sd = {k.replace('module.', ''): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[model] loaded {ckpt_path}')
    print(f'[model] missing {len(missing)}  unexpected {len(unexpected)}')
    if missing:
        print('  sample missing:', missing[:6])
    return model


# ---------------------------------------------------------------------------
# Feature extraction (two towers, raw + projected head-inputs)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    vids = [b[3] for b in batch]
    return imgs, labs, paths, vids


@torch.no_grad()
def extract_features(model, entries, device, batch_size=16):
    ds = FeatDataset(entries)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=_collate)

    C_list, V_list, Cp_list, Vp_list = [], [], [], []
    y_list, path_list, vid_list = [], [], []

    model.eval()
    for images, labels, paths, vids in tqdm(loader, desc='extract', leave=False):
        images = images.to(device)
        B = images.shape[0]

        # ---- CLIP tower: C = raw CLS of hs[-2] (the level the model feeds the head)
        clip_0, clip_1, clip_2, clip_vis_feats = model.clip_vision_encoder(images)
        C = clip_vis_feats[:, 0, :].float()                      # [B,1024]
        # projected head-input CLIP CLS (768) -- only for sensitivity check
        Cp = model.vision_proj(clip_vis_feats.float())[:, 0, :].float()   # [B,768]

        # ---- ViT tower: V = final CLS token (after final LayerNorm)
        vit_in = model._preprocess_for_vit(images).to(model.vit_dtype)
        vit_out = model.vit.forward_features(vit_in)             # [B,197,768]
        V = vit_out[:, 0, :].float()                             # [B,768]
        Vp = model.deepfake_proj(V).float()                      # [B,768] head-input

        C_list.append(C.cpu().numpy())
        Cp_list.append(Cp.cpu().numpy())
        V_list.append(V.cpu().numpy())
        Vp_list.append(Vp.cpu().numpy())
        y_list.append(labels.numpy())
        path_list.extend(paths)
        vid_list.extend(vids)

    C = np.concatenate(C_list, axis=0).astype(np.float32)
    Cp = np.concatenate(Cp_list, axis=0).astype(np.float32)
    V = np.concatenate(V_list, axis=0).astype(np.float32)
    Vp = np.concatenate(Vp_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)          # 1=real, 0=fake (as in txt)
    return {'C': C, 'Cp': Cp, 'V': V, 'Vp': Vp, 'y': y,
            'paths': np.array(path_list), 'vids': np.array(vid_list)}


# ---------------------------------------------------------------------------
# Probes / metrics
# ---------------------------------------------------------------------------
def l2_logistic_auc(Xtr, ytr, Xte, yte, seed=0, C=1.0):
    """Standardized L2 logistic regression fit on train, AUC on test."""
    ytr = np.asarray(ytr).ravel()
    yte = np.asarray(yte).ravel()
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float('nan'), None, None
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)
    clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs',
                             random_state=seed)
    clf.fit(Xtr_s, ytr)
    s = clf.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(yte, s)
    return auc, s, clf


def fit_residual(Ctr, Vtr, alpha_grid=None):
    """Ridge regression C->V fit on train. Returns (resid_train, transform_fn)."""
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 13)
    sc = StandardScaler().fit(Ctr)
    Ctr_s = sc.transform(Ctr)
    ridge = RidgeCV(alphas=alpha_grid).fit(Ctr_s, Vtr)
    return ridge, sc


def apply_residual(ridge, sc, C, V):
    return V - ridge.predict(sc.transform(C))


def bootstrap_auc(y, s, n_iter=2000, seed=0):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_iter):
        ii = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[ii])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[ii], s[ii]))
        except ValueError:
            pass
    if not aucs:
        return (float('nan'), float('nan'))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def linear_cka(X, Y):
    """Linear CKA between feature matrices (centered over samples)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))


def effective_ranks(X, energy=0.9):
    """Column-centered data -> SVD. Returns (participation_ratio, n_dims_90%_energy, numerical_rank)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s * s
    tot = s2.sum()
    if tot <= 0 or s[0] <= 0:
        return (float('nan'), float('nan'), int(0))
    pr = float((s.sum() ** 2) / tot)
    cum = np.cumsum(s2) / tot
    n90 = int(np.searchsorted(cum, energy) + 1)
    nnum = int((s > s[0] * 1e-6).sum())
    return pr, n90, nnum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--txt', default=os.path.join(PROJECT_ROOT, 'dataset', 'data_2023', 'ffpp_test_split.txt'))
    ap.add_argument('--ckpt', default=os.path.join(PROJECT_ROOT, 'checkpoints', 'stage_1', 'bridge_v2_phase1.pth'))
    ap.add_argument('--clip', default=os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336'))
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--cap-train', type=int, default=1100, help='max frames per class on probe-train side')
    ap.add_argument('--cap-test', type=int, default=400, help='max frames per class on probe-test side')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # ---------- load entries & video-level 70/30 split, cap per class ----------
    entries = load_entries(args.txt)
    tr_all, te_all, ntr_k, nte_k = video_split(entries, train_frac=0.7, seed=args.seed)
    # Re-balance classes toward ~cap each (total per class <= cap_train+cap_test = 1500)
    tr = cap_per_class(tr_all, args.cap_train, seed=args.seed)
    te = cap_per_class(te_all, args.cap_test, seed=args.seed)
    print(f'[split] video groups: train={ntr_k}, test={nte_k}')
    print(f'[split] after cap: train={len(tr)} (real={sum(1 for e in tr if e[1]==1)}, '
          f'fake={sum(1 for e in tr if e[1]==0)}), '
          f'test={len(te)} (real={sum(1 for e in te if e[1]==1)}, '
          f'fake={sum(1 for e in te if e[1]==0)})')

    # ---------- build model ----------
    model = build_model(args.clip, args.ckpt)
    model.to(device).eval()
    print(f'[model] on {device}, fp32')

    # ---------- extract ----------
    Ftr = extract_features(model, tr, device, args.batch_size)
    Fte = extract_features(model, te, device, args.batch_size)

    Ctr, Vtr, ytr = Ftr['C'], Ftr['V'], Ftr['y']
    Cte, Vte, yte = Fte['C'], Fte['V'], Fte['y']

    # npz label is fake-positive for probe classification
    # txt label: 1=real (original), 0=fake (manipulated)  -> target=1 for fake
    ztr = (ytr == 0).astype(np.int64)
    zte = (yte == 0).astype(np.int64)

    # ---------- Ridge C -> V on train ----------
    ridge, sc_C = fit_residual(Ctr, Vtr)
    Rtr = apply_residual(ridge, sc_C, Ctr, Vtr)
    Rte = apply_residual(ridge, sc_C, Cte, Vte)

    # ---------- probes ----------
    print('\n[probe] fitting L2 logistic probes ...')
    auc_V, s_V, _ = l2_logistic_auc(Vtr, ztr, Vte, zte)
    auc_C, s_C, _ = l2_logistic_auc(Ctr, ztr, Cte, zte)
    auc_R, s_R, _ = l2_logistic_auc(Rtr, ztr, Rte, zte)
    ci_V = bootstrap_auc(zte, s_V)
    ci_C = bootstrap_auc(zte, s_C)
    ci_R = bootstrap_auc(zte, s_R)

    # sensitivity: head-input projected pair (Cp 768, Vp 768)
    Cp_tr, Vp_tr = Ftr['Cp'], Ftr['Vp']
    Cp_te, Vp_te = Fte['Cp'], Fte['Vp']
    ridge_p, sc_Cp = fit_residual(Cp_tr, Vp_tr)
    Rp_tr = apply_residual(ridge_p, sc_Cp, Cp_tr, Vp_tr)
    Rp_te = apply_residual(ridge_p, sc_Cp, Cp_te, Vp_te)
    auc_Vp, _, _ = l2_logistic_auc(Vp_tr, ztr, Vp_te, zte)
    auc_Cp, _, _ = l2_logistic_auc(Cp_tr, ztr, Cp_te, zte)
    auc_Rp, s_Rp, _ = l2_logistic_auc(Rp_tr, ztr, Rp_te, zte)
    ci_Rp = bootstrap_auc(zte, s_Rp)

    # ---------- CKA & effective ranks (pooled train+test, R is train-fitted) ----------
    Cp_all = np.concatenate([Ctr, Cte], axis=0)
    Vp_all = np.concatenate([Vtr, Vte], axis=0)
    Rp_all = np.concatenate([Rtr, Rte], axis=0)
    cka_VC = linear_cka(Vp_all, Cp_all)
    cka_VC_tr = linear_cka(Vtr, Ctr)
    er_V = effective_ranks(Vp_all)
    er_C = effective_ranks(Cp_all)
    er_R = effective_ranks(Rp_all)

    # ---------- decision ----------
    decision = None
    reason = ''
    if auc_R >= 0.60 and ci_R[0] > 0.50:
        decision = 'COMPLEMENTARY (residual carries fake-discriminative info)'
        reason = (f'probe(R) AUC = {auc_R:.4f} (95% CI {ci_R[0]:.4f}-{ci_R[1]:.4f}), '
                  f'clearly > 0.5 -> ViT branch adds info CLIP cannot linearly predict.')
    elif abs(auc_R - 0.5) < 0.03 or (ci_R[0] <= 0.50 and auc_R < 0.60):
        decision = 'REDUNDANT (no reliable residual signal)'
        reason = (f'probe(R) AUC = {auc_R:.4f} (95% CI {ci_R[0]:.4f}-{ci_R[1]:.4f}) '
                  f'~ 0.5 -> ViT residual is linearly redundant w.r.t. CLIP.')
    else:
        decision = 'WEAK / INCONCLUSIVE'
        reason = f'probe(R) AUC = {auc_R:.4f} (95% CI {ci_R[0]:.4f}-{ci_R[1]:.4f})'

    # ---------- save npz ----------
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, 'probe_feats.npz')
    # NOTE: R stored here is the train-fitted residual (see ridge info in txt report)
    R_all = np.concatenate([Rtr, Rte], axis=0)
    np.savez_compressed(
        npz_path,
        V=Vp_all, C=Cp_all, R=R_all,
        y=np.concatenate([ytr, yte], axis=0),
        paths=np.concatenate([Ftr['paths'], Fte['paths']], axis=0),
        vids=np.concatenate([Ftr['vids'], Fte['vids']], axis=0),
        train_mask=np.concatenate([np.ones(len(ytr), bool), np.zeros(len(yte), bool)], axis=0),
        V_proj=np.concatenate([Vp_tr, Vp_te], axis=0),
        C_proj=np.concatenate([Cp_tr, Cp_te], axis=0),
        feature_dims=np.array([Cp_all.shape[1], Vp_all.shape[1], R_all.shape[1]]),
        auc_R=auc_R,
        seed=args.seed,
    )
    print(f'[save] npz -> {npz_path}')

    # ---------- text report ----------
    report_path = os.path.join(out_dir, 'residual_probe_report.txt')
    lines = []
    lines.append('=' * 78)
    lines.append('R-TEST / RESIDUAL PROBE REPORT  (M2F2-Det Stage-1 detector)')
    lines.append('=' * 78)
    lines.append('')
    lines.append(f'Date              : 2026-09-05')
    lines.append(f'Detector ckpt     : {args.ckpt}')
    lines.append(f'Txt / split       : {args.txt}')
    lines.append(f'Split method      : video-level (source-id group) 70/30, then cap per class '
                 f'train={args.cap_train}, test={args.cap_test}')
    lines.append(f'Sample counts     : probe-train {len(ytr)} (real {int((ytr==1).sum())}, '
                 f'fake {int((ytr==0).sum())}); probe-test {len(yte)} '
                 f'(real {int((yte==1).sum())}, fake {int((yte==0).sum())})')
    lines.append('')
    lines.append('Feature definitions')
    lines.append('  C : raw CLIP vision tower CLS feature  (hidden_states[-2] layer, position 0), '
                 f'{Cp_all.shape[1]}-d, BEFORE vision_proj')
    lines.append('  V : raw ViT (PDI-initialized) final CLS token after final LayerNorm '
                 f'(vit.forward_features[:,0]), {Vp_all.shape[1]}-d, BEFORE deepfake_proj')
    lines.append('  R : V - Ridge_pred(V|C), Ridge fit on probe-train, applied to all')
    lines.append('  (sensitivity: C_proj/V_proj = the projected 768-d head-input pair)')
    lines.append('')
    lines.append('AUC (L2 logistic probe; target=fake==1; CI = 2.5/97.5 percentile bootstrap on test)')
    lines.append(f'  probe(C)      AUC = {auc_C:.4f}   95% CI [{ci_C[0]:.4f}, {ci_C[1]:.4f}]')
    lines.append(f'  probe(V)      AUC = {auc_V:.4f}   95% CI [{ci_V[0]:.4f}, {ci_V[1]:.4f}]')
    lines.append(f'  probe(R)      AUC = {auc_R:.4f}   95% CI [{ci_R[0]:.4f}, {ci_R[1]:.4f}]   <-- CORE')
    lines.append('')
    lines.append('Sensitivity (head-input projected features, 768-d)')
    lines.append(f'  probe(C_proj) AUC = {auc_Cp:.4f}   probe(V_proj) AUC = {auc_Vp:.4f}   '
                 f'probe(R_proj) AUC = {auc_Rp:.4f}   95% CI [{ci_Rp[0]:.4f}, {ci_Rp[1]:.4f}]')
    lines.append('')
    lines.append(f'Linear CKA (V, C) : all-data = {cka_VC:.4f}    probe-train = {cka_VC_tr:.4f}')
    lines.append('Effective rank (column-centered SVD; all pooled data; R is train-fitted):')
    lines.append(f'  metric format: (participation_ratio, n_dims_for_90%_energy, numerical_rank)')
    lines.append(f'  V : {er_V}')
    lines.append(f'  C : {er_C}')
    lines.append(f'  R : {er_R}')
    lines.append('')
    lines.append('CKA/rank methodology notes:')
    lines.append('  * linear CKA on centered (per-feature mean subtracted over samples) Gram matrices.')
    lines.append('  * effective rank computed on column-centered data matrix SVD; participation ratio = '
                 'sum(s)^2/sum(s^2); numerical_rank = # singular values > 1e-6*s_max.')
    lines.append('')
    lines.append(f'DECISION : {decision}')
    lines.append(f'REASON  : {reason}')
    lines.append('')
    lines.append(f'npz output : {npz_path}')
    lines.append('=' * 78)
    txt = '\n'.join(lines)
    print(txt)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f'[save] report -> {report_path}')

    # console summary for caller
    print('\n===== SUMMARY =====')
    print(f'probe(C)={auc_C:.4f}  probe(V)={auc_V:.4f}  probe(R)={auc_R:.4f}  (CI {ci_R[0]:.4f}-{ci_R[1]:.4f})')
    print(f'CKA(V,C)={cka_VC:.4f}   effrank V={er_V}  C={er_C}  R={er_R}')
    print(f'DECISION: {decision}')


if __name__ == '__main__':
    main()
