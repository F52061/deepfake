# -*- coding: utf-8 -*-
"""
G19a -- LoRA fine-tuning of the (previously frozen) ViT branch for CROSS-DOMAIN
deepfake transfer.  HELD-OUT FOLD = **wild**  (sibling agent owns dfdcp on GPU 1).

Science question
----------------
The frozen ViT branch (bridge_v2_phase1.pth) yields a cross-domain source-label
LR probe of cd1 0.8286 / cd2 0.8633 / dfdcp 0.8261 / ffiw 0.8244 / wild 0.8090
(mean 0.8303), in-domain FF++ 0.9852.  Does *unfreezing* the ViT branch (via
LoRA adapters) and retraining it on multi-domain data improve cross-domain
transfer, and does an explicit domain-invariance objective beat plain CE?

Arms (5)
--------
    R0   no training -- features from the frozen ViT (pipeline sanity gate)
    R1   CE with SHUFFLED labels (control for "gains are not from label info")
    O1   CE only
    O3a  CE + lambda * sum_{d,c} n_{d,c} ||mu_{d,c} - mu_c||^2 / sum n_{d,c},  lambda=0.3
    O3b  same, lambda=1.0
mu_{d,c} = batch mean of the 768-d CLS feature over (domain,class) cell (d,c);
mu_c = mean over domains d of mu_{d,c}, **detached** (stop-gradient);
penalty skipped when any (domain,class) cell in the batch has < 2 samples.

Training sources for fold=wild : FF++ (official splits/train.json videos) +
cd2 + dfdcp + ffiw.  Held-out domain `wild` is NEVER touched (no target stats,
no TTA, no pseudo-labels, no BN updates).  `cd1` is excluded ALWAYS (its 49
folder names are 100% contained in cd2 -> leakage).

Hard exclusion of the probe set: every FF++ video folder used by
`_probe/probe_feats.npz` (BOTH probe-train and probe-test) is removed from
training, identified by PATH-DERIVED DIRECTORY-CHAIN identity (full normalised
absolute directory path), never by bare video-name strings (bare names collide
across datasets, e.g. FF++ `161` != WildDeepfake `161`).

Feature = raw ViT CLS token (768-d, after the final LayerNorm) produced by the
model's own `_preprocess_for_vit` path (albumentations Normalize(CLIP means) ->
ToTensorV2 -> F.interpolate(224) -> [-1,1]).

Resource discipline: CUDA_VISIBLE_DEVICES=2 (physical GPU 2, verified idle),
fp32, batch 32; OMP/MKL/OPENBLAS/NUMEXPR=4, torch.set_num_threads(4),
cv2.setNumThreads(0), num_workers<=2.

Writes ONLY inside vit_module/_g19/ with the `_wild` suffix.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["BLIS_NUM_THREADS"] = "4"
os.environ["JOBLIB_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
cv2.setNumThreads(0)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from albumentations import Compose, Normalize, ToTensorV2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ------------------------------------------------------------------- paths ----
PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
G17_CNN_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_g17", "cnn_feats.npz")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "stage_1", "bridge_v2_phase1.pth")

FFPP_RAW = "F:/zhj/data/FaceForensic++_raw"
FFPP_TRAIN_JSON = os.path.join(FFPP_RAW, "splits", "train.json")
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

HELD_OUT = "wild"                                     # this agent owns `wild`
TRAIN_DOMAINS = ["ffpp", "cd2", "dfdcp", "ffiw"]      # cd1 excluded ALWAYS
ALL_TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
CONTAMINATED = ["cd1", "cd2", "dfdcp", "ffiw"]

# ---------------------------------------------------------------- constants --
SEED = 20260910
IMG_SIZE = 336
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

REAL_FRAMES = 15            # per FF++ real video   (identical to _g18/train_g18b.py)
FAKE_FRAMES = 3             # per FF++ manipulated video folder
TARGET_MAX_PER_VIDEO = 40
TARGET_CLASS_CAP = 2000
VAL_MAX_PER_VIDEO = 8
VAL_FRAC = 0.10
VAL_VIDEOS_PER_CELL = 24

BATCH = 32
STEPS_PER_EPOCH = 250
MAX_EPOCHS = 12
PATIENCE = 4
LORA_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LAMBDA_O3A = 0.3
LAMBDA_O3B = 1.0

ARMS = ["R0", "R1", "O1", "O3a", "O3b"]

# anchors (prior experiments, identical probe protocol)
ANCHOR_FROZEN = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261,
                 "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_INDOM = 0.9852
ANCHOR_SRCLR_MEAN = 0.8318
ANCHOR_ORACLE = 0.9111
GATE_R0_TOL = 0.005

OUT_DIR = HERE
LOG_PATH = os.path.join(OUT_DIR, "run_log_g19a_wild.txt")
REPORT_PATH = os.path.join(OUT_DIR, "g19a_wild_report.txt")
STATS_PATH = os.path.join(OUT_DIR, "g19a_wild_stats.npz")


# ------------------------------------------------------------------- logger --
class Logger:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")
        self.lines = []

    def __call__(self, msg=""):
        s = str(msg)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()
        self.lines.append(s)

    def close(self):
        self.fh.close()


log = Logger(LOG_PATH)
T_START = time.time()


# ------------------------------------------------------------------ helpers --
def query_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    rows.append((int(parts[0]), int(float(parts[1])), int(float(parts[2]))))
                except ValueError:
                    continue
        return rows
    except Exception:
        return []


def _norm_dir(x):
    return os.path.abspath(str(x)).replace("\\", "/").rstrip("/").lower()


def vid_id_of_image(p):
    """Path-derived directory-chain identity of the VIDEO FOLDER that holds image `p`.

    Never a bare video-name string: two datasets sharing a leaf folder name
    (FF++ `161` vs WildDeepfake `161`) map to different ids because the dataset
    root differs.
    """
    return _norm_dir(os.path.dirname(str(p)))


def vid_id_of_folder(d):
    """Same identity, addressed by the video folder itself (NOT its parent)."""
    return _norm_dir(d)


def list_frames(folder):
    try:
        names = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg"))]
    except Exception:
        return []

    def key(f):
        mm = re.search(r"(\d+)", f)
        return (int(mm.group(1)) if mm else 0, f)
    names.sort(key=key)
    return [os.path.join(folder, f) for f in names]


def sample_frames(folder, max_n):
    frames = list_frames(folder)
    if not frames:
        return []
    n = min(len(frames), max_n)
    idx = np.unique(np.linspace(0, len(frames) - 1, n).astype(np.int64))
    return [frames[i] for i in idx]


def balance_by_class3(samples, rng=None):
    """samples: list of (path,label,domain). Cap so real==fake (verbatim _g18 logic)."""
    rng = rng or random.Random(SEED)
    real = [s for s in samples if s[1] == 1]
    fake = [s for s in samples if s[1] == 0]
    n = min(len(real), len(fake))
    if len(real) > n:
        rng.shuffle(real)
        real = real[:n]
    if len(fake) > n:
        rng.shuffle(fake)
        fake = fake[:n]
    out = real + fake
    rng.shuffle(out)
    return out


# ------------------------------------------------------------- preprocessing --
def _worker_init(_wid):
    torch.set_num_threads(1)
    cv2.setNumThreads(0)


class PoolDataset(Dataset):
    """Returns (CLIP-normalised 336 tensor, index) for one image path."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return _feat_tensor(self.paths[i]), i


class FeatDataset(Dataset):
    """Bit-identical preprocessing to _probe/residual_probe.py FeatDataset:
    cv2.imread -> BGR2RGB -> resize(336) -> albumentations Normalize(CLIP) -> ToTensorV2."""

    def __init__(self, paths):
        self.paths = paths
        self.transform = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        return self.transform(image=image)["image"]


def _feat_tensor(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    z = image.astype(np.float32) / 255.0
    z = (z - np.array(CLIP_MEAN, np.float32)) / np.array(CLIP_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))


class _PreprocessStub:
    """Calls the detector's OWN `_preprocess_for_vit` implementation (unbound, on a
    stub receiver -- the method only touches module-level CLIP_MEAN/CLIP_STD)."""

    def __init__(self):
        from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
        self._fn = ViT_M2F2Det_Bridge._preprocess_for_vit

    def __call__(self, images):
        return self._fn(self, images)


_PREPROCESS_STUB = None


def preprocess_for_vit(images):
    return _PREPROCESS_STUB(images)


# ------------------------------------------------------------- data build ----
def load_eval_set():
    """5300 rows in _g17/cnn_feats.npz native order."""
    P = np.load(PROBE_NPZ, allow_pickle=True)
    paths_p = np.array([str(s) for s in P["paths"]], dtype=object)
    vids_p = np.array([str(s) for s in P["vids"]], dtype=object)
    y_p = P["y"].astype(np.int64)
    tr_mask = P["train_mask"].astype(bool)
    n_p = len(paths_p)
    assert n_p == 3000 and int(tr_mask.sum()) == 2200

    M = np.load(MULTI_NPZ, allow_pickle=True)
    paths_m = np.array([str(s) for s in M["path"]], dtype=object)
    vids_m = np.array([str(s) for s in M["vid"]], dtype=object)
    y_m = M["y"].astype(np.int64)
    dom_m = np.array([str(s) for s in M["domain"]], dtype=object)
    n_m = len(paths_m)
    assert n_m == 2300

    y = np.concatenate([y_p, y_m]).astype(np.int64)
    paths = np.concatenate([paths_p, paths_m]).astype(object)
    vids = np.concatenate([vids_p, vids_m]).astype(object)
    dom = np.concatenate([np.array(["ffpp_probe"] * n_p, dtype=object), dom_m]).astype(object)
    split = np.concatenate([np.where(tr_mask, "train", "test").astype(object), dom_m]).astype(object)
    assert len(paths) == 5300

    g = np.load(G17_CNN_NPZ, allow_pickle=True)
    gpaths = np.array([str(s) for s in g["paths"]], dtype=object)
    assert gpaths.shape == paths.shape, "row count mismatch vs _g17/cnn_feats.npz"
    aligned = bool(np.array_equal(paths.astype(str), gpaths.astype(str)))
    assert aligned, "FATAL: row-order mismatch vs _g17/cnn_feats.npz"
    # full-length (5300) mask: True only for the 2200 FF++ probe-train rows
    train_mask_full = np.concatenate([tr_mask, np.zeros(n_m, dtype=bool)])
    assert int(train_mask_full.sum()) == 2200
    return dict(paths=paths, vids=vids, y=y, domain=dom, split=split,
                train_mask=tr_mask, train_mask_full=train_mask_full,
                n_probe=n_p, aligned=aligned)


def build_probe_ffpp_dirchain(ev):
    """Video-folder identities of EVERY FF++ video used by the probe set
    (BOTH probe-train and probe-test rows)."""
    return set(vid_id_of_image(p) for p in ev["paths"][:ev["n_probe"]])


def build_ffpp_videos(probe_dirchain):
    """FF++ train videos (bidirectional convention, identical to _g18/train_g18b.py)
    minus every folder whose video-folder identity is used by the probe set."""
    with open(FFPP_TRAIN_JSON, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    real_ids = sorted(set([a for a, _ in pairs]) | set([b for _, b in pairs]))
    videos = []          # (folder, label, vkey, domain)
    n_missing = n_excluded = 0
    for vid in real_ids:
        folder = os.path.join(FFPP_RAW, "original_sequences", "c23", "faces23", vid)
        if not os.path.isdir(folder):
            n_missing += 1
            continue
        if vid_id_of_folder(folder) in probe_dirchain:
            n_excluded += 1
            continue
        videos.append((folder, 1, "ffpp_real_%s" % vid, "ffpp"))
    for a, b in pairs:
        for method in FFPP_METHODS:
            for vid in ("%s_%s" % (a, b), "%s_%s" % (b, a)):     # both directions
                folder = os.path.join(FFPP_RAW, "manipulated_sequences", method,
                                      "c23", "faces23", vid)
                if not os.path.isdir(folder):
                    n_missing += 1
                    continue
                if vid_id_of_folder(folder) in probe_dirchain:
                    n_excluded += 1
                    continue
                videos.append((folder, 0, "ffpp_fake_%s_%s" % (method, vid), "ffpp"))
    return videos, n_missing, n_excluded


def build_target_videos(domains):
    """Target-domain training videos from feats_multi.npz folders (verbatim _g18)."""
    m = np.load(MULTI_NPZ, allow_pickle=True)
    dom = m["domain"].astype(str)
    path = m["path"].astype(str)
    y = m["y"].astype(np.int64)
    videos = []
    for d in domains:
        idx = np.where(dom == d)[0]
        folder_label = {}
        for i in idx:
            fld = os.path.dirname(path[i])
            if fld in folder_label:
                assert folder_label[fld] == int(y[i]), "mixed-label folder %s" % fld
            folder_label[fld] = int(y[i])
        for fld, lab in folder_label.items():
            videos.append((fld, lab, "%s_%s" % (d, os.path.basename(fld)), d))
    return videos


# ------------------------------------------------------------------ model ----
def load_vit_state():
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    vit_sd = {k[4:]: v for k, v in sd.items() if k.startswith("vit.")}
    assert len(vit_sd) == 162, len(vit_sd)
    return vit_sd


def make_vit(vit_sd, device):
    from vit_module.vit_adaptive_mattn_aps import vit_base_patch16_224
    m = vit_base_patch16_224(pretrained=False, num_classes=0)
    missing, unexpected = m.load_state_dict(vit_sd, strict=True)
    assert not missing and not unexpected
    for blk in m.blocks:
        # the deployment (eval) forward never takes the stochastic dynamic-mask
        # branch; force attn_drop=0 so training matches evaluation exactly.
        blk.attn.attn_drop = 0.0
    m.to(device)
    return m


def attach_lora(vit):
    from peft import LoraConfig, get_peft_model
    targets = ["blocks.%d.attn.qkv" % i for i in range(len(vit.blocks))] + \
              ["blocks.%d.attn.proj" % i for i in range(len(vit.blocks))]
    for _n, p in vit.named_parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=targets, bias="none")
    lm = get_peft_model(vit, cfg)
    n_lora = sum(p.numel() for n, p in lm.named_parameters()
                 if "lora_" in n and p.requires_grad)
    return lm, targets, n_lora


@torch.no_grad()
def extract_feats(vit, device, paths, batch=BATCH, nworkers=2, tag="", quiet=False):
    """Raw ViT CLS features (N,768) via the exact deployment preprocessing path."""
    vit.eval()
    dl = DataLoader(FeatDataset(list(paths)), batch_size=batch, shuffle=False,
                    num_workers=nworkers, pin_memory=True, worker_init_fn=_worker_init)
    out = []
    t0 = time.time()
    for i, x in enumerate(dl):
        x = x.to(device, non_blocking=True)
        z = preprocess_for_vit(x).to(torch.float32)
        f = vit.forward_features(z)[:, 0, :]
        out.append(f.float().cpu().numpy())
        if (not quiet) and i % 25 == 0:
            log("    [extract %s] %d/%d wall=%.0fs"
                % (tag, min((i + 1) * batch, len(dl.dataset)), len(dl.dataset), time.time() - t0))
    feats = np.concatenate(out, axis=0).astype(np.float32)
    assert feats.shape[1] == 768, feats.shape
    return feats


# ---------------------------------------------------------------- sampling ---
def build_cells(samples):
    """samples: list of (path,label,domain) -> (cells dict cell->indices, cell names)."""
    cells = defaultdict(list)
    for i, (_p, lab, dom) in enumerate(samples):
        cells[(dom, int(lab))].append(i)
    return cells, sorted(cells.keys())


class BalancedCellSampler:
    """Balanced sampler over (domain,class) cells: each batch holds `per` samples
    from every cell (with replacement when a cell is smaller than `per`)."""

    def __init__(self, cells, names, batch=BATCH, steps=STEPS_PER_EPOCH, seed=SEED):
        self.cells = cells
        self.names = list(names)
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.per = max(2, batch // max(1, len(self.names)))
        self.bs = self.per * len(self.names)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            idx = []
            for nm in self.names:
                pool = self.cells[nm]
                replace = len(pool) < self.per
                idx.extend(self.rng.choice(pool, size=self.per, replace=replace).tolist())
            self.rng.shuffle(idx)
            yield idx


def _collate_pool(batch):
    x = torch.stack([b[0] for b in batch])
    i = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return x, i


# ------------------------------------------------------------------ training --
def _invariance_penalty(feat, idx, lab_arr, dom_arr, names):
    """sum_{d,c} n_{d,c} ||mu_{d,c} - mu_c||^2 / sum n_{d,c} ; mu_c detached.
    Returns None when any (domain,class) cell in the batch has < 2 samples."""
    mus, counts = {}, {}
    for (d, c) in names:
        sel = np.where((dom_arr[idx] == d) & (lab_arr[idx] == c))[0]
        counts[(d, c)] = len(sel)
        if len(sel) < 2:
            return None
        mus[(d, c)] = sel
    dev = feat.device
    mu = {k: feat[torch.from_numpy(v).to(dev)].mean(dim=0) for k, v in mus.items()}
    tot = float(sum(counts.values()))
    pen = feat.new_zeros(())
    classes = sorted(set(c for (_d, c) in names))
    for c in classes:
        doms = [d for (d, cc) in names if cc == c]
        mu_c = torch.stack([mu[(d, c)] for d in doms], dim=0).mean(dim=0).detach()
        for d in doms:
            pen = pen + (counts[(d, c)] / tot) * ((mu[(d, c)] - mu_c) ** 2).sum()
    return pen


@torch.no_grad()
def _eval_val(lm, head, device, val_paths, batch=BATCH):
    lm.eval()
    head.eval()
    out = []
    for i in range(0, len(val_paths), batch):
        chunk = val_paths[i:i + batch]
        x = torch.stack([_feat_tensor(p) for p in chunk]).to(device)
        z = preprocess_for_vit(x).to(torch.float32)
        f = lm.forward_features(z)[:, 0, :]
        out.append(F.softmax(head(f), dim=1)[:, 1].float().cpu().numpy())   # P(real)
    return np.concatenate(out)


def train_arm(arm, pool, val_paths, val_y, val_dom, device, vit_sd,
              steps_per_epoch=STEPS_PER_EPOCH, max_epochs=MAX_EPOCHS,
              patience=PATIENCE, nworkers=2, shuffle_labels=False, log_steps=0):
    log("")
    log("=" * 92)
    log("ARM %s  (held-out=%s)" % (arm, HELD_OUT))
    log("=" * 92)
    t0 = time.time()

    labels = np.array([s[1] for s in pool], dtype=np.int64)
    if shuffle_labels:
        rng = np.random.default_rng(SEED + 777)
        labels = rng.permutation(labels)
        log("[R1] TRAINING LABELS SHUFFLED (random permutation of the real label vector)")
    dom_arr = np.array([s[2] for s in pool], dtype=object)
    paths_all = [s[0] for s in pool]

    cells, names = build_cells([(paths_all[i], int(labels[i]), dom_arr[i])
                                for i in range(len(pool))])
    log("[pool] n=%d ; cells: %s%s"
        % (len(pool), ", ".join("%s/%s=%d" % (d, "real" if c == 1 else "fake", len(cells[(d, c)]))
                                for (d, c) in names),
           "" if len(names) == 8 else "   *** WARNING: expected 8 (domain,class) cells ***"))

    vit = make_vit(vit_sd, device)
    lm, targets, n_lora = attach_lora(vit)
    head = nn.Linear(768, 2).to(device)
    log("[lora] target modules=%d (blocks[*].attn.qkv + blocks[*].attn.proj) r=%d alpha=%d dropout=%.1f"
        % (len(targets), LORA_R, LORA_ALPHA, LORA_DROPOUT))
    log("[lora] LoRA trainable params=%d (%.4fM) ; head trainable params=%d"
        % (n_lora, n_lora / 1e6, sum(p.numel() for p in head.parameters())))

    opt = torch.optim.AdamW(
        [{"params": [p for p in lm.parameters() if p.requires_grad], "lr": LORA_LR},
         {"params": head.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps_per_epoch)
    crit = nn.CrossEntropyLoss()

    lam = {"O3a": LAMBDA_O3A, "O3b": LAMBDA_O3B}.get(arm, 0.0)
    sampler = BalancedCellSampler(cells, names, batch=BATCH, steps=steps_per_epoch,
                                  seed=SEED + (0 if not shuffle_labels else 1))
    log("[obj] lambda=%s -> objective = CE%s ; balanced sampler: %d cells x %d = batch %d, %d steps/epoch"
        % (lam, "" if lam == 0 else " + lambda*domain-invariance", len(names), sampler.per,
           sampler.bs, steps_per_epoch))
    dl = DataLoader(PoolDataset(paths_all), batch_sampler=sampler, num_workers=nworkers,
                    pin_memory=True, collate_fn=_collate_pool, worker_init_fn=_worker_init)

    best_auc, best_epoch = -1.0, -1
    best_lora = best_head = None
    patience_left = patience
    curve = []
    n_skip = n_pen = 0

    for epoch in range(1, max_epochs + 1):
        lm.train()
        head.train()
        te0 = time.time()
        tot = corr = 0
        loss_sum = ce_sum = pen_sum = 0.0
        step = 0
        for x, idx in dl:
            step += 1
            x = x.to(device, non_blocking=True)
            idx_np = idx.numpy()
            yb = torch.from_numpy(labels[idx_np]).to(device)

            z = preprocess_for_vit(x).to(torch.float32)
            feat = lm.forward_features(z)[:, 0, :]
            logits = head(feat)
            ce = crit(logits, yb)
            if lam > 0:
                pen = _invariance_penalty(feat, idx_np, labels, dom_arr, names)
                if pen is None:
                    n_skip += 1
                    loss = ce
                else:
                    n_pen += 1
                    pen_sum += float(pen.item())
                    loss = ce + lam * pen
            else:
                loss = ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log_steps and (step % log_steps == 0):
                log("    [%s epoch %d step %d/%d] ce=%.4f%s lr_lora=%.2e"
                    % (arm, epoch, step, steps_per_epoch, float(ce.item()),
                       "" if lam == 0 else " pen=%.4f" % (float(pen.item()) if pen is not None else -1),
                       opt.param_groups[0]["lr"]))
            bs = yb.size(0)
            tot += bs
            corr += int((logits.argmax(1) == yb).sum().item())
            loss_sum += float(loss.item()) * bs
            ce_sum += float(ce.item()) * bs

        train_loss = loss_sum / max(1, tot)
        train_ce = ce_sum / max(1, tot)
        train_acc = corr / max(1, tot)

        vf = _eval_val(lm, head, device, val_paths)
        vy = np.asarray(val_y)
        vdom = np.asarray(val_dom)
        val_auc = float(roc_auc_score(vy, vf)) if len(np.unique(vy)) > 1 else float("nan")
        per_dom = {}
        for d in TRAIN_DOMAINS:
            m = vdom == d
            if m.sum() > 0 and len(np.unique(vy[m])) > 1:
                per_dom[d] = float(roc_auc_score(vy[m], vf[m]))
        curve.append(dict(epoch=epoch, loss=train_loss, ce=train_ce, acc=train_acc,
                          val_auc=val_auc, per_dom=per_dom))
        improved = (not np.isnan(val_auc)) and val_auc > best_auc
        log("[%s epoch %02d/%d] loss=%.4f ce=%.4f acc=%.4f val_auc=%.4f | %s%s (%.0fs)"
            % (arm, epoch, max_epochs, train_loss, train_ce, train_acc, val_auc,
               " ".join("%s=%.4f" % (k, v) for k, v in sorted(per_dom.items())),
               "  *BEST*" if improved else "", time.time() - te0))
        if improved:
            best_auc, best_epoch = val_auc, epoch
            best_lora = {k: v.detach().cpu().clone() for k, v in lm.named_parameters()
                         if "lora_" in k}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log("[%s early-stop] no val_auc improvement for %d epochs at epoch %d"
                    % (arm, patience, epoch))
                break

    wall = time.time() - t0
    log("[%s done] best val_auc=%.4f at epoch %d ; penalty applied %d / skipped %d ; wall=%.1fs"
        % (arm, best_auc, best_epoch, n_pen, n_skip, wall))

    named = dict(lm.named_parameters())
    for k, v in best_lora.items():
        named[k].data.copy_(v.to(device))
    head.load_state_dict({k: v.to(device) for k, v in best_head.items()})

    save_path = os.path.join(OUT_DIR, "lora_g19a_wild_%s.pt" % arm)
    torch.save({"arm": arm, "held_out": HELD_OUT, "lora": best_lora, "head": best_head,
                "targets": targets, "r": LORA_R, "alpha": LORA_ALPHA,
                "best_val_auc": best_auc, "best_epoch": best_epoch, "n_lora": n_lora,
                "wall_s": wall, "curve": curve, "lambda": lam,
                "shuffle_labels": bool(shuffle_labels)}, save_path)
    log("[%s save] %s" % (arm, save_path))
    return dict(lm=lm, head=head, best_val_auc=best_auc, best_epoch=best_epoch,
                n_lora=n_lora, wall=wall, curve=curve, save_path=save_path, lam=lam,
                n_pen=n_pen, n_skip=n_skip)


# ------------------------------------------------------------- probe / ratio -
def probe_eval(feat, ev):
    """Standard probe: StandardScaler(fit probe-train 2200) + LR(C=1e-3, lbfgs, 3000)."""
    trf = ev["train_mask_full"]
    Xtr = feat[trf].astype(np.float64)
    ytr = ev["y"][trf]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    res = {}
    te = np.zeros(len(ev["y"]), bool)
    te[:ev["n_probe"]] = ~ev["train_mask"]
    res["indom"] = float(roc_auc_score(ev["y"][te], clf.decision_function(sc.transform(feat[te]))))
    for d in ALL_TARGET_DOMAINS:
        m = ev["domain"] == d
        res[d] = float(roc_auc_score(ev["y"][m], clf.decision_function(sc.transform(feat[m]))))
    return res


def ratio_eval(feat, ev):
    """G12 formula: s=sqrt(mean_j Var_j(Xtr,ddof=1)); class_gap=||mu_fake-mu_real||/s;
    dom_gap(d)=||mu_d-mu_src||/s ; ratio=mean_d(dom_gap)/class_gap."""
    X = feat.astype(np.float64)
    trf = ev["train_mask_full"]
    Xtr = X[trf]
    ys = ev["y"][trf]
    mu_src = Xtr.mean(axis=0)
    s = float(np.sqrt(np.var(Xtr, axis=0, ddof=1).mean()))
    class_gap = float(np.linalg.norm(Xtr[ys == 0].mean(axis=0) - Xtr[ys == 1].mean(axis=0)) / s)
    ratios, gaps = {}, {}
    for d in ALL_TARGET_DOMAINS:
        gaps[d] = float(np.linalg.norm(X[ev["domain"] == d].mean(axis=0) - mu_src) / s)
        ratios[d] = gaps[d] / class_gap
    return dict(s=s, class_gap=class_gap, dom_gap=gaps, ratio=ratios,
                ratio_mean=float(np.mean([ratios[d] for d in ALL_TARGET_DOMAINS])))


def smoke_selection(ev, n=80):
    """80 probe-train + 80 probe-test rows, class-balanced (sklearn needs 2 classes)."""
    idx_tr = np.where(ev["train_mask"])[0]
    idx_te = np.where(~ev["train_mask"])[0]
    a = np.concatenate([idx_tr[ev["y"][idx_tr] == 1][:n], idx_tr[ev["y"][idx_tr] == 0][:n]])
    b = np.concatenate([idx_te[ev["y"][idx_te] == 1][:n], idx_te[ev["y"][idx_te] == 0][:n]])
    return np.concatenate([a, b])


def fmt_tab(rows):
    widths = [max(len(str(r[i])) for r in rows) + 2 for i in range(len(rows[0]))]
    return "\n".join("  " + "".join(("%-" + str(w) + "s") % str(c) for c, w in zip(r, widths))
                     for r in rows)


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="R0,R1,O1,O3a,O3b")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    smoke = args.smoke

    gpus = query_gpu()
    log("[gpu] nvidia-smi (index,mem.used,util)=%s" % (gpus,))
    log("[gpu] CUDA_VISIBLE_DEVICES=%s (target physical GPU index 2)"
        % os.environ.get("CUDA_VISIBLE_DEVICES"))
    g2 = [g for g in gpus if g[0] == 2]
    if g2:
        log("[gpu] physical GPU 2 mem.used=%d MiB -> %s"
            % (g2[0][1], "IDLE (<=100 MiB)" if g2[0][1] <= 100 else "BUSY (>100 MiB)"))
    nw = 0 if smoke else 2
    log("[threads] OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s ; torch.set_num_threads -> %d ; "
        "cv2.setNumThreads(0) ; DataLoader num_workers=%d"
        % (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"),
           os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("NUMEXPR_NUM_THREADS"),
           torch.get_num_threads(), nw))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    log("[torch] %s ; device=%s ; fp32 ; threads=%d"
        % (torch.__version__, torch.cuda.get_device_name(0), torch.get_num_threads()))

    global _PREPROCESS_STUB
    _PREPROCESS_STUB = _PreprocessStub()
    log("[preproc] using detector's own ViT_M2F2Det_Bridge._preprocess_for_vit")

    ev = load_eval_set()
    log("[eval] %d rows ; probe=%d (train_mask %d) ; multi=%d ; alignment vs _g17/cnn_feats.npz = %s"
        % (len(ev["paths"]), ev["n_probe"], int(ev["train_mask"].sum()),
           len(ev["paths"]) - ev["n_probe"], ev["aligned"]))

    # ---------------- data audit ----------------
    probe_dirchain = build_probe_ffpp_dirchain(ev)
    log("[probe] distinct FF++ video-folder identities in the 3000 probe rows = %d"
        % len(probe_dirchain))
    # positive control: the two identity accessors must agree on the same folder
    _chk = sorted(probe_dirchain)[:200]
    assert all(vid_id_of_folder(c) == c for c in _chk), "directory-identity accessors disagree"
    log("[probe] identity positive-control: vid_id_of_folder(x)==x for 200/%d probe chains -> OK"
        % len(probe_dirchain))
    ffpp_videos, ffpp_missing, ffpp_excluded = build_ffpp_videos(probe_dirchain)
    log("[ffpp] train videos: real=%d fake=%d total=%d missing=%d excluded_by_probe_dirchain=%d"
        % (sum(1 for v in ffpp_videos if v[1] == 1), sum(1 for v in ffpp_videos if v[1] == 0),
           len(ffpp_videos), ffpp_missing, ffpp_excluded))

    # --- POSITIVE CONTROL for the probe-exclusion mechanism ---
    # Build the FF++ training-folder path for probe source ids and confirm it *would*
    # be caught.  (FF++ official train/test ids are disjoint here, so the real
    # exclusion count is 0 -- this control proves the key format is not degenerate.)
    _ptest_ids = sorted(set(os.path.basename(os.path.dirname(str(p))).split("_")[0]
                            for p in ev["paths"][:ev["n_probe"]]))
    _hit = _miss = 0
    for vid in _ptest_ids[:40]:
        fld = os.path.join(FFPP_RAW, "original_sequences", "c23", "faces23", vid)
        if os.path.isdir(fld):
            if vid_id_of_folder(fld) in probe_dirchain:
                _hit += 1
            else:
                _miss += 1
    log("[probe] exclusion POSITIVE CONTROL: %d/%d constructed FF++ folders for probe source ids "
        "would be caught by the directory-chain filter (%d miss)" % (_hit, _hit + _miss, _miss))
    assert _miss == 0, "probe-exclusion mechanism is degenerate"

    target_videos = build_target_videos(TRAIN_DOMAINS[1:])
    for d in TRAIN_DOMAINS[1:]:
        vv = [v for v in target_videos if v[3] == d]
        log("[target] %-5s training videos=%d (real %d / fake %d)"
            % (d, len(vv), sum(1 for v in vv if v[1] == 1), sum(1 for v in vv if v[1] == 0)))

    train_chains = set(vid_id_of_folder(v[0]) for v in ffpp_videos) | \
                   set(vid_id_of_folder(v[0]) for v in target_videos)
    ho_chains = set(vid_id_of_image(p) for p in ev["paths"][ev["domain"] == HELD_OUT])
    log("[leak %s] held-out video folders=%d ; intersection with training folders=%d"
        % (HELD_OUT, len(ho_chains), len(ho_chains & train_chains)))
    assert len(ho_chains & train_chains) == 0, "HELD-OUT %s LEAKED INTO TRAINING" % HELD_OUT

    cd1_chains = set(vid_id_of_image(p) for p in ev["paths"][ev["domain"] == "cd1"])
    log("[leak cd1] cd1 video folders=%d ; intersection with training folders=%d"
        % (len(cd1_chains), len(cd1_chains & train_chains)))
    assert len(cd1_chains & train_chains) == 0, "cd1 leaked into training"

    probe_all_chains = set(vid_id_of_image(p) for p in ev["paths"][:ev["n_probe"]])
    log("[leak probe] 3000 probe rows -> %d video folders ; intersection with training folders=%d"
        % (len(probe_all_chains), len(probe_all_chains & train_chains)))
    assert len(probe_all_chains & train_chains) == 0, "probe videos leaked into training"

    cd1_names = set(os.path.basename(c) for c in cd1_chains)
    cd2_names = set(os.path.basename(c) for c in
                    set(vid_id_of_image(p) for p in ev["paths"][ev["domain"] == "cd2"]))
    log("[audit] cd1 basenames subset of cd2 basenames: %s (cd1=%d cd2=%d inter=%d) -> cd1 excluded always"
        % (cd1_names <= cd2_names, len(cd1_names), len(cd2_names), len(cd1_names & cd2_names)))

    # ---------------- video-level val split (TRAINING domains only) ----------------
    all_videos = ffpp_videos + target_videos
    real_vids = [v for v in all_videos if v[1] == 1]
    fake_vids = [v for v in all_videos if v[1] == 0]
    rng = random.Random(SEED)
    rng.shuffle(real_vids)
    rng.shuffle(fake_vids)
    n_val_real = max(1, int(round(len(real_vids) * VAL_FRAC)))
    n_val_fake = max(1, int(round(len(fake_vids) * VAL_FRAC)))
    val_videos = real_vids[:n_val_real] + fake_vids[:n_val_fake]
    train_videos = real_vids[n_val_real:] + fake_vids[n_val_fake:]
    log("[split] train videos=%d (real %d / fake %d) ; val videos=%d (real %d / fake %d) ; "
        "NO frame of `%s` is present in either"
        % (len(train_videos), sum(1 for v in train_videos if v[1] == 1),
           sum(1 for v in train_videos if v[1] == 0), len(val_videos),
           sum(1 for v in val_videos if v[1] == 1), sum(1 for v in val_videos if v[1] == 0),
           HELD_OUT))

    # ---------------- frame sampling ----------------
    rng_s = random.Random(SEED)
    pool = []
    for folder, label, vkey, dom in train_videos:
        cap = (REAL_FRAMES if label == 1 else FAKE_FRAMES) if dom == "ffpp" \
            else TARGET_MAX_PER_VIDEO
        for fp in sample_frames(folder, cap):
            pool.append((fp, label, dom))
    n_before = len(pool)
    pool = balance_by_class3(pool, rng=rng_s)
    log("[pool] frames before balance=%d after=%d" % (n_before, len(pool)))

    probe_chain_all = set(vid_id_of_image(p) for p in ev["paths"])
    bad_ffpp = [p for (p, _l, d) in pool if d == "ffpp" and vid_id_of_image(p) in probe_chain_all]
    n_tgt_overlap = sum(1 for (p, _l, d) in pool
                        if d != "ffpp" and vid_id_of_image(p) in probe_chain_all)
    log("[leak frame-level] FF++ training frames from the 5300 eval set: %d (must be 0) ; "
        "target-domain training frames that ARE eval rows: %d (expected: cd2/dfdcp/ffiw are "
        "CONTAMINATED training domains)" % (len(bad_ffpp), n_tgt_overlap))
    assert len(bad_ffpp) == 0, "FF++ training frame from the probe/eval set: %d" % len(bad_ffpp)

    # ---------------- val pool (balanced per cell, capped) ----------------
    rng_v = random.Random(SEED + 1)
    by_cell = defaultdict(list)
    for v in val_videos:
        by_cell[(v[3], v[1])].append(v)
    val_paths, val_y, val_dom = [], [], []
    for key in sorted(by_cell.keys()):
        vids = list(by_cell[key])
        rng_v.shuffle(vids)
        for folder, label, vkey, dom in vids[:VAL_VIDEOS_PER_CELL]:
            for fp in sample_frames(folder, VAL_MAX_PER_VIDEO):
                val_paths.append(fp)
                val_y.append(label)
                val_dom.append(dom)
    log("[val] samples=%d over %d cells: %s"
        % (len(val_paths), len(by_cell),
           ", ".join("%s/%s=%d" % (k[0], "real" if k[1] == 1 else "fake",
                                   sum(1 for d in val_dom if d == k[0]))
                     for k in sorted(by_cell.keys()))))

    smoke_steps, smoke_epochs = 20, 1
    if smoke:
        log("")
        log("### SMOKE MODE: tiny pool (<=40/cell), 1 epoch, O1 only ###")
        rng_sm = random.Random(SEED + 5)
        by_cell_sm = defaultdict(list)
        for s in pool:
            by_cell_sm[(s[2], s[1])].append(s)
        pool = []
        for k in sorted(by_cell_sm.keys()):
            v = list(by_cell_sm[k])
            rng_sm.shuffle(v)
            pool.extend(v[:40])
        log("[smoke] pool=%d cells=%s"
            % (len(pool), ", ".join("%s/%d=%d" % (k[0], k[1], sum(1 for s in pool if (s[2], s[1]) == k))
                                    for k in sorted(by_cell_sm.keys()))))
        arms = ["O1"]
        val_paths = val_paths[:256]
        val_y = val_y[:256]
        val_dom = val_dom[:256]

    vit_sd = load_vit_state()
    log("[ckpt] ViT state keys=%d (91.4M params, num_classes=0)" % len(vit_sd))

    results, feats_store = {}, {}

    # ---------------- R0 : frozen features + pipeline sanity gate ----------------
    log("")
    log("=" * 92)
    log("ARM R0 -- frozen ViT (no training)")
    log("=" * 92)
    if smoke:
        t0 = time.time()
        vit = make_vit(vit_sd, device)
        sel = smoke_selection(ev, 80)
        sub = extract_feats(vit, device, ev["paths"][sel], batch=32, nworkers=0,
                            tag="R0-smoke", quiet=True)
        log("[smoke R0] feats shape=%s dtype=%s" % (sub.shape, sub.dtype))
        assert sub.shape == (320, 768)
        ysm = ev["y"][sel]
        sc = StandardScaler().fit(sub[:160])
        clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs").fit(
            sc.transform(sub[:160]), ysm[:160])
        auc = float(roc_auc_score(ysm[160:], clf.decision_function(sc.transform(sub[160:]))))
        log("[smoke R0] probe AUC on the 160/160 subset = %.4f" % auc)
        results["R0"] = dict(smoke_auc=auc, wall=time.time() - t0)
        del vit
        torch.cuda.empty_cache()
    else:
        t0 = time.time()
        vit = make_vit(vit_sd, device)
        f = extract_feats(vit, device, ev["paths"], tag="R0")
        wall = time.time() - t0
        p = probe_eval(f, ev)
        log("[R0] in-domain FF++800 AUC = %.4f (anchor %.4f)" % (p["indom"], ANCHOR_INDOM))
        for d in ALL_TARGET_DOMAINS:
            log("  [R0] %-6s AUC = %.4f (anchor %.4f, dev %+.4f)"
                % (d, p[d], ANCHOR_FROZEN[d], p[d] - ANCHOR_FROZEN[d]))
        P = np.load(PROBE_NPZ, allow_pickle=True)
        M = np.load(MULTI_NPZ, allow_pickle=True)
        dP = float(np.abs(f[:ev["n_probe"]] - P["V"].astype(np.float32)).max())
        dM = float(np.abs(f[ev["n_probe"]:] - M["V"].astype(np.float32)).max())
        log("[R0] max|feat - probe_feats.npz V| = %.3e ; max|feat - feats_multi.npz V| = %.3e"
            % (dP, dM))
        devs = [abs(p[d] - ANCHOR_FROZEN[d]) for d in ALL_TARGET_DOMAINS]
        dev_indom = abs(p["indom"] - ANCHOR_INDOM)
        gate_ok = (max(devs) <= GATE_R0_TOL) and (dev_indom <= GATE_R0_TOL)
        log("[R0 GATE] max|dev| over 5 domains=%.4f ; |dev| in-domain=%.4f ; tol=%.3f -> %s"
            % (max(devs), dev_indom, GATE_R0_TOL,
               "PASS (extraction reproduces the frozen anchor)" if gate_ok
               else "FAIL (extraction pipeline is WRONG)"))
        np.savez(os.path.join(OUT_DIR, "feats_g19a_wild_R0.npz"), feat=f, paths=ev["paths"])
        results["R0"] = dict(probe=p, wall=wall, ratio=ratio_eval(f, ev), gate_ok=bool(gate_ok),
                             max_dev=float(max(devs)), dev_indom=float(dev_indom),
                             dev_stored_probe=dP, dev_stored_multi=dM)
        feats_store["R0"] = f
        del vit
        torch.cuda.empty_cache()
        gc.collect()
        if not gate_ok:
            log("[R0 GATE] *** STOPPING: pipeline sanity gate FAILED; training arms NOT run ***")
            _finish(ev, results, feats_store, aborted=True)
            return

    # ---------------- trainable arms ----------------
    for arm in arms:
        if arm == "R0":
            continue
        shuffle = (arm == "R1")
        t0 = time.time()
        tr = train_arm(arm, pool, val_paths, val_y, val_dom, device, vit_sd,
                       steps_per_epoch=smoke_steps if smoke else STEPS_PER_EPOCH,
                       max_epochs=smoke_epochs if smoke else MAX_EPOCHS,
                       patience=PATIENCE, nworkers=nw, shuffle_labels=shuffle,
                       log_steps=5 if smoke else 0)
        lm = tr["lm"]
        if smoke:
            sel = smoke_selection(ev, 80)
            sub = extract_feats(lm, device, ev["paths"][sel], batch=32, nworkers=0,
                                tag=arm + "-smoke", quiet=True)
            log("[smoke %s] feats shape=%s dtype=%s" % (arm, sub.shape, sub.dtype))
            assert sub.shape == (320, 768)
            ysm = ev["y"][sel]
            sc = StandardScaler().fit(sub[:160])
            clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs").fit(
                sc.transform(sub[:160]), ysm[:160])
            auc = float(roc_auc_score(ysm[160:], clf.decision_function(sc.transform(sub[160:]))))
            log("[smoke %s] probe AUC on the 160/160 subset = %.4f" % (arm, auc))
            results[arm] = dict(smoke_auc=auc, wall=time.time() - t0,
                                n_lora=tr["n_lora"], best_val_auc=tr["best_val_auc"],
                                n_pen=tr["n_pen"], n_skip=tr["n_skip"])
            del lm
            torch.cuda.empty_cache()
            continue

        f = extract_feats(lm, device, ev["paths"], tag=arm)
        p = probe_eval(f, ev)
        r = ratio_eval(f, ev)
        wall = time.time() - t0
        log("[%s] in-domain FF++800 AUC = %.4f" % (arm, p["indom"]))
        for d in ALL_TARGET_DOMAINS:
            log("  [%s] %-6s AUC = %.4f%s" % (arm, d, p[d],
               "   <-- HELD-OUT (valid transfer)" if d == HELD_OUT else "   [CONTAMINATED]"))
        log("[%s] ratio: s=%.3f class_gap=%.4f mean_ratio=%.4f"
            % (arm, r["s"], r["class_gap"], r["ratio_mean"]))
        np.savez(os.path.join(OUT_DIR, "feats_g19a_wild_%s.npz" % arm), feat=f, paths=ev["paths"])
        results[arm] = dict(probe=p, wall=wall, ratio=r, n_lora=tr["n_lora"],
                            best_val_auc=tr["best_val_auc"], best_epoch=tr["best_epoch"],
                            train_wall=tr["wall"], lam=tr["lam"],
                            n_pen=tr["n_pen"], n_skip=tr["n_skip"])
        feats_store[arm] = f
        del lm
        torch.cuda.empty_cache()
        gc.collect()

    if smoke:
        log("")
        log("[smoke] SMOKE COMPLETE results=%s"
            % json.dumps({k: v for k, v in results.items()}, default=str))
        log.close()
        return

    _finish(ev, results, feats_store, aborted=False)


def _finish(ev, results, feats_store, aborted=False):
    L = log
    L("")
    L("=" * 92)
    L("G19a  (held-out fold = %s, physical GPU 2)  SUMMARY" % HELD_OUT)
    L("=" * 92)
    if aborted:
        L("ABORTED: the R0 pipeline sanity gate failed.  No training arms were run.")

    have = [a for a in ARMS if a in results and "probe" in results[a]]

    if have:
        rows = [["arm"] + ALL_TARGET_DOMAINS + ["indom800", "valid(=wild)"]]
        for a in have:
            p = results[a]["probe"]
            rows.append([a] + ["%.4f" % p[d] for d in ALL_TARGET_DOMAINS] +
                        ["%.4f" % p["indom"], "%.4f" % p[HELD_OUT]])
        rows.append(["frozen-anchor"] + ["%.4f" % ANCHOR_FROZEN[d] for d in ALL_TARGET_DOMAINS] +
                    ["%.4f" % ANCHOR_INDOM, "%.4f" % ANCHOR_FROZEN[HELD_OUT]])
        L("")
        L("TABLE 1  standard probe AUC (StandardScaler fit on probe-train 2200 + LR C=1e-3 lbfgs 3000)")
        L(fmt_tab(rows))
        L("  CONTAMINATION: cd1/cd2/dfdcp/ffiw are TRAINING domains for fold=%s -> those columns are" % HELD_OUT)
        L("    CONTAMINATED and are EXCLUDED from every aggregate; ONLY the %s column is a valid" % HELD_OUT)
        L("    cross-domain transfer measurement.  'valid(=wild)' repeats that column.")
        L("  cd1 is a duplicate of cd2 (cd1 basenames subset of cd2 basenames) and is excluded from")
        L("    training AND from held-out selection at all times.")
        L("  References: frozen ViT mean 0.8303 ; source-label cross-domain anchor 0.8318 ; target ORACLE 0.9111")

        rows = [["arm", "dim", "s", "class_gap"] + ALL_TARGET_DOMAINS + ["mean_ratio"]]
        for a in have:
            r = results[a]["ratio"]
            rows.append([a, "768", "%.3f" % r["s"], "%.4f" % r["class_gap"]] +
                        ["%.4f" % r["ratio"][d] for d in ALL_TARGET_DOMAINS] +
                        ["%.4f" % r["ratio_mean"]])
        L("")
        L("TABLE 2  domain-sensitivity ratio (G12 formula), source = probe-train 2200")
        L(fmt_tab(rows))

        rows = [["arm", "lora_params", "lambda", "best_val_auc", "best_epoch", "train_wall_s", "total_wall_s"]]
        for a in have:
            r = results[a]
            rows.append([a, str(r.get("n_lora", "-")), str(r.get("lam", "-")),
                         "%.4f" % r.get("best_val_auc", float("nan")),
                         str(r.get("best_epoch", "-")), "%.1f" % r.get("train_wall", 0.0),
                         "%.1f" % r.get("wall", 0.0)])
        L("")
        L("TABLE 3  LoRA trainable params / schedule / wall time")
        L(fmt_tab(rows))

    L("")
    L("GATES (mechanical; raw verdicts, no interpretation)")
    gates = {}
    if have:
        wild = {a: results[a]["probe"][HELD_OUT] for a in have}
        cand = [a for a in ["O1", "O3a", "O3b"] if a in wild]
        best_arm = max(cand, key=lambda a: wild[a]) if cand else None
        best = wild[best_arm] if best_arm else float("nan")
        r0 = wild.get("R0", float("nan"))
        r1 = wild.get("R1", float("nan"))
        o1 = wild.get("O1", float("nan"))
        indom = {a: results[a]["probe"]["indom"] for a in have}

        g1 = (o1 - r0) if ("O1" in wild and "R0" in wild) else float("nan")
        g2 = (best - o1) if ("O1" in wild and best_arm) else float("nan")
        g3 = (best - r1) if ("R1" in wild and best_arm) else float("nan")
        g4 = indom.get(best_arm, float("nan")) if best_arm else float("nan")
        g5 = (best - ANCHOR_SRCLR_MEAN) / (ANCHOR_ORACLE - ANCHOR_SRCLR_MEAN) if best_arm else float("nan")
        kill = ("O1" in wild and "R1" in wild and "R0" in wild and
                abs(o1 - r0) < 0.005 and abs(r1 - r0) < 0.005)

        L("  held-out(%s) AUC: R0=%.4f R1=%.4f O1=%.4f %s"
          % (HELD_OUT, r0, r1, o1,
             " ".join("%s=%.4f" % (a, wild[a]) for a in ["O3a", "O3b"] if a in wild)))
        L("  best of {O1,O3a,O3b} = %s (%.4f)" % (best_arm, best))
        L("  G19_UNFREEZE_HELPS   : O1 - R0 = %+.4f  (>= +0.0100) -> %s"
          % (g1, "PASS" if g1 >= 0.01 else "FAIL"))
        L("  G19_INVARIANCE_HELPS : best - O1 = %+.4f (>= +0.0200) -> %s"
          % (g2, "PASS" if g2 >= 0.02 else "FAIL"))
        L("  G19_VS_RANDLABEL     : best - R1 = %+.4f (>= +0.0200) -> %s   [HARD]"
          % (g3, "PASS" if g3 >= 0.02 else "FAIL"))
        L("  G19_NO_FORGETTING    : in-domain(best)=%.4f (>= %.4f) -> %s"
          % (g4, ANCHOR_INDOM - 0.01, "PASS" if g4 >= ANCHOR_INDOM - 0.01 else "FAIL"))
        L("  G19_CEILING_RECOVERY : (best-0.8318)/(0.9111-0.8318) = %.4f (>= 0.30) -> %s"
          % (g5, "PASS" if g5 >= 0.30 else "FAIL"))
        L("  KILL SIGNAL          : |O1-R0|=%.4f |R1-R0|=%.4f -> %s"
          % (abs(o1 - r0), abs(r1 - r0),
             "TRIGGERED (unfreezing did not move the representation at all)" if kill
             else "not triggered"))
        gates = dict(unfreeze=g1, invariance=g2, vs_randlabel=g3, no_forgetting=g4,
                     ceiling=g5, kill=bool(kill))
        L("  best arm for the gates = %s" % best_arm)

    L("")
    L("[gpu final] %s" % (query_gpu(),))
    L("[wall] total = %.1f s" % (time.time() - T_START))

    stat = {"arms": np.array(have), "held_out": np.array([HELD_OUT]),
            "target_domains": np.array(ALL_TARGET_DOMAINS),
            "anchor_frozen": np.array([ANCHOR_FROZEN[d] for d in ALL_TARGET_DOMAINS]),
            "anchor_indom": np.array([ANCHOR_INDOM]),
            "contaminated": np.array(CONTAMINATED)}
    for a in have:
        stat["auc_%s" % a] = np.array([results[a]["probe"][d] for d in ALL_TARGET_DOMAINS])
        stat["indom_%s" % a] = np.array([results[a]["probe"]["indom"]])
        stat["ratio_%s" % a] = np.array([results[a]["ratio"]["ratio"][d]
                                         for d in ALL_TARGET_DOMAINS])
        stat["classgap_%s" % a] = np.array([results[a]["ratio"]["class_gap"]])
        stat["s_%s" % a] = np.array([results[a]["ratio"]["s"]])
        if "n_lora" in results[a]:
            stat["lora_params_%s" % a] = np.array([results[a]["n_lora"]])
    if "R0" in results and "probe" in results["R0"]:
        stat["r0_gate_ok"] = np.array([float(results["R0"]["gate_ok"])])
        stat["r0_max_dev"] = np.array([results["R0"]["max_dev"]])
        stat["r0_dev_stored_probe"] = np.array([results["R0"]["dev_stored_probe"]])
        stat["r0_dev_stored_multi"] = np.array([results["R0"]["dev_stored_multi"]])
    for k, v in gates.items():
        if isinstance(v, (int, float, bool)):
            stat["gate_%s" % k] = np.array([float(v)])
    np.savez(STATS_PATH, **stat)

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log.lines) + "\n")
    log("[save] %s" % REPORT_PATH)
    log("[save] %s" % STATS_PATH)
    log.close()


if __name__ == "__main__":
    main()
