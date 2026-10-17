# -*- coding: utf-8 -*-
"""
G19a -- does unfreezing/retraining the ViT branch (LoRA) improve CROSS-DOMAIN
fake/real transfer, and does an explicit domain-invariance objective beat plain
cross-entropy?

Held-out fold: dfdcp.  Training sources: FF++ (official splits/train.json videos)
+ cd2 + ffiw + wild.  cd1 is ALWAYS excluded (its folder set is a subset of cd2),
dfdcp is the held-out domain and is never touched by training or model selection.

Arms (5):
    R0   no training -- frozen ViT features (pipeline sanity / anchor reproduction)
    R1   CE with labels SHUFFLED (control: gains must not come from label info)
    O1   CE only
    O3a  CE + lambda * sum_{d,c} n_{d,c} ||mu_{d,c} - mu_c||^2 / sum n_{d,c},  lambda=0.3
    O3b  same with lambda=1.0

Trainable: LoRA (peft, r=16 alpha=32 dropout=0, blocks[*].attn.qkv + blocks[*].attn.proj)
+ a 768->2 classification head.  Everything else frozen.

Pipeline reuse (verbatim conventions from G18b / G17c / _probe / _tsne):
  * data enumeration / video-level split / leakage audit / probe-test-folder
    exclusion -> copied from vit_module/_g18/train_g18b.py, with the FF++ exclusion
    HARDENED to exclude BOTH probe-train and probe-test videos, identified by
    path-derived DIRECTORY CHAIN (never bare video-name strings).
  * preprocessing -> cv2.imread BGR2RGB -> cv2.resize(336, INTER_LINEAR) ->
    albumentations Normalize(CLIP_MEAN/STD) + ToTensorV2 -> model._preprocess_for_vit
    (denorm -> F.interpolate(224) -> [-1,1]) -> vit.forward_features[:,0,:] (768-d).
    Verified byte-level: this reproduces probe_feats.npz['V'] and feats_multi.npz['V']
    exactly (maxabs 0.0 / 1.1e-05).
  * feature matrix rows 0..2999 = probe_feats order, 3000..5299 = feats_multi order,
    asserted byte-equal to _g17/cnn_feats.npz['paths'].
  * probe head: StandardScaler(fit probe-train, 2200) + LogisticRegression(
    C=1e-3, solver='lbfgs', max_iter=3000, random_state=0); y 1=real 0=fake,
    AUC positive=fake (yb = (y==0)).
  * G12 ratio: s=sqrt(mean_j Var_j(Xtr,ddof=1)); class_gap=||mu_fake-mu_real||/s;
    dom_gap(d)=||mu_d-mu_src||/s; ratio=mean_d(dom_gap)/class_gap, d over the 5
    multi target domains.  Baseline V ratio = 0.1983 (G17c anchor).

Resource discipline: CUDA_VISIBLE_DEVICES=1 (verified idle before start),
OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS=4, torch.set_num_threads(4),
cv2.setNumThreads(0), num_workers=0 (all images pre-decoded to RAM once), fp32,
batch 32.

Writes ONLY inside vit_module/_g19/.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["BLIS_NUM_THREADS"] = "4"

import argparse
import gc
import json
import random
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
import cv2
cv2.setNumThreads(0)
from albumentations import Compose, Normalize, ToTensorV2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ------------------------------------------------------------------- paths ----
PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
G17_CNN_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_g17", "cnn_feats.npz")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_1", "bridge_v2_phase1.pth")
CLIP_LOCAL = os.path.join(PROJECT_ROOT, "checkpoints", "clip-vit-large-patch14-336")

LOG_PATH = os.path.join(HERE, "run_log_g19a_dfdcp.txt")
REPORT_PATH = os.path.join(HERE, "g19a_dfdcp_report.txt")
STATS_PATH = os.path.join(HERE, "g19a_dfdcp_stats.npz")

FFPP_RAW = "F:/zhj/data/FaceForensic++_raw"
FFPP_TRAIN_JSON = os.path.join(FFPP_RAW, "splits", "train.json")
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# ---------------------------------------------------------------- constants --
SEED = 20260910
IMG_SIZE = 336
NUM_THREADS = 4
HOLD_OUT = "dfdcp"
TRAIN_DOMAINS = ["ffpp", "cd2", "ffiw", "wild"]
ALL_TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
MEAN_T = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
STD_T = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)

REAL_FRAMES = 15            # per FF++ real video   (G18b convention)
FAKE_FRAMES = 3             # per FF++ fake folder  (G18b convention)
TARGET_MAX_PER_VIDEO = 40
VAL_MAX_PER_VIDEO = 4
VAL_FRAC = 0.10
BATCH = 32
MAX_ITERS_PER_EPOCH = 250
MAX_EPOCHS = 12
PATIENCE = 4
LORA_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
AUG_CROP_MIN = 0.89
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# baseline anchors (prior experiments, identical probe protocol)
ANCHOR_FROZEN = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_FROZEN_MEAN = 0.8303
ANCHOR_IN_DOMAIN = 0.9852
ANCHOR_SRC_LR_XDOM = 0.8318
ANCHOR_ORACLE_TARGET = 0.9111
ANCHOR_RATIO_V = 0.1983

ARMS_O3 = {"O3a": 0.3, "O3b": 1.0}
ARMS_ALL = ["R0", "R1", "O1", "O3a", "O3b"]

CONTAMINATED_COLUMNS = ["cd2", "ffiw", "wild"]


# ------------------------------------------------------------------- logger --
class Logger:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")
        self.buf = []

    def __call__(self, msg=""):
        msg = str(msg)
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()
        self.buf.append(msg)

    def close(self):
        self.fh.close()


log = Logger(LOG_PATH)
t_start = time.time()


def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


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


# ------------------------------------------------------------- file helpers --
def path_key(p):
    """Path-derived identity for a directory chain (never a bare basename)."""
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def ck(p):
    """Cache key: normalized path so / and \\ forms of the same file collide."""
    return path_key(p)


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


def cap_by_class(samples, cap, rng):
    real = [s for s in samples if s[1] == 1]
    fake = [s for s in samples if s[1] == 0]
    if cap is not None:
        if len(real) > cap:
            rng.shuffle(real)
            real = real[:cap]
        if len(fake) > cap:
            rng.shuffle(fake)
            fake = fake[:cap]
    out = real + fake
    rng.shuffle(out)
    return out


# ------------------------------------------------------------- 5300 layout ---
def load_5300_meta():
    """Rows 0..2999 = probe_feats order ; 3000..5299 = feats_multi order."""
    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths_p = np.array([str(s) for s in p["paths"]], dtype=object)
    vids_p = np.array([str(s) for s in p["vids"]], dtype=object)
    y_p = p["y"].astype(np.int64)
    tm = p["train_mask"].astype(bool)

    m = np.load(MULTI_NPZ, allow_pickle=True)
    paths_m = np.array([str(s) for s in m["path"]], dtype=object)
    vids_m = np.array([str(s) for s in m["vid"]], dtype=object)
    y_m = m["y"].astype(np.int64)
    dom_m = np.array([str(s) for s in m["domain"]], dtype=object)

    n = len(paths_p) + len(paths_m)
    assert n == 5300, n
    assert int(tm.sum()) == 2200
    paths = np.concatenate([paths_p, paths_m]).astype(object)
    vids = np.concatenate([vids_p, vids_m]).astype(object)
    y = np.concatenate([y_p, y_m]).astype(np.int64)
    domain = np.array(["ffpp_probe"] * len(paths_p) + list(dom_m), dtype=object)
    split = np.array((["train"] * int(tm.sum()) + ["test"] * int((~tm).sum())) + list(dom_m),
                     dtype=object)
    tr_mask = np.concatenate([tm, np.zeros(len(paths_m), bool)])

    # ---- alignment assertion against _g17/cnn_feats.npz (row-aligned byte-equal) ----
    g = np.load(G17_CNN_NPZ, allow_pickle=True)
    gpaths = np.array([str(s) for s in g["paths"]], dtype=object)
    gpaths_bytes = np.array([s.encode("utf-8") for s in gpaths])
    paths_bytes = np.array([str(s).encode("utf-8") for s in paths])
    assert gpaths_bytes.shape == paths_bytes.shape
    assert np.array_equal(gpaths_bytes, paths_bytes), "row-order mismatch vs _g17/cnn_feats.npz"
    gy = g["y"].astype(np.int64)
    assert np.array_equal(gy, y), "y mismatch vs _g17/cnn_feats.npz"
    meta = dict(paths=paths, vids=vids, y=y, domain=domain, split=split,
                train_mask=tr_mask, align_g17=True)
    return meta


def read_rgb336(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


_TF = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])


def to_clip_tensor(rgb336):
    """Exactly the _probe/_tsne path: albumentations Normalize(CLIP) + ToTensorV2."""
    return _TF(image=np.ascontiguousarray(rgb336))["image"]


def aug_train(rgb336):
    """Light train augmentation from the cached 336 uint8 (G18b-style, output stays 336)."""
    s = random.randint(int(IMG_SIZE * AUG_CROP_MIN), IMG_SIZE)
    top = random.randint(0, IMG_SIZE - s)
    left = random.randint(0, IMG_SIZE - s)
    img = rgb336[top:top + s, left:left + s]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
    return img


# ----------------------------------------------------------- data build ------
def build_probe_folder_set(meta):
    """Directory-chain identity of EVERY probe image folder (train AND test)."""
    probe_rows = np.where(np.array([str(d) for d in meta["domain"]]) == "ffpp_probe")[0]
    keys = set(path_key(os.path.dirname(str(meta["paths"][i]))) for i in probe_rows)
    return keys, probe_rows


def build_ffpp_videos(probe_folder_keys):
    """FF++ official train.json, bidirectional fake convention (G18b), but exclude by
    DIRECTORY CHAIN (covers probe-train *and* probe-test videos)."""
    with open(FFPP_TRAIN_JSON, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    real_ids = sorted(set([a for a, _ in pairs]) | set([b for _, b in pairs]))
    videos = []          # (folder, label, vkey)
    n_missing = 0
    n_excluded = 0

    def consider(folder, label, vkey):
        nonlocal n_missing, n_excluded
        if not os.path.isdir(folder):
            n_missing += 1
            return
        if path_key(folder) in probe_folder_keys:
            n_excluded += 1
            return
        videos.append((folder, label, vkey))

    for vid in real_ids:
        consider(os.path.join(FFPP_RAW, "original_sequences", "c23", "faces23", vid),
                 1, "ffpp_real_%s" % vid)
    for a, b in pairs:
        for method in FFPP_METHODS:
            for vid in ("%s_%s" % (a, b), "%s_%s" % (b, a)):     # both directions
                consider(os.path.join(FFPP_RAW, "manipulated_sequences", method,
                                      "c23", "faces23", vid),
                         0, "ffpp_fake_%s_%s" % (method, vid))
    return videos, n_missing, n_excluded


def build_target_videos(meta, domains):
    dom = np.array([str(d) for d in meta["domain"]])
    path = meta["paths"]
    y = meta["y"]
    videos = []
    for d in domains:
        idx = np.where(dom == d)[0]
        folder_label = {}
        for i in idx:
            fld = os.path.dirname(str(path[i]))
            if fld in folder_label:
                assert folder_label[fld] == int(y[i]), "mixed-label folder %s" % fld
            folder_label[fld] = int(y[i])
        for fld, lab in folder_label.items():
            videos.append((fld, lab, "%s_%s" % (d, os.path.basename(fld))))
    return videos


# ------------------------------------------------------------- datasets ------
class MemDataset(Dataset):
    def __init__(self, samples, cache, train=False):
        self.samples = samples          # (path, label, cell_id)
        self.cache = cache
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, cell = self.samples[i]
        rgb = self.cache[ck(path)]
        if self.train:
            rgb = aug_train(rgb)
        t = to_clip_tensor(rgb)
        return t, torch.tensor(label, dtype=torch.long), torch.tensor(cell, dtype=torch.long)


class BalancedBatchSampler(Sampler):
    """Draw BATCH//n_cells samples from every (domain,class) cell each step."""

    def __init__(self, cells, batch=BATCH, seed=SEED, max_iters=MAX_ITERS_PER_EPOCH):
        self.cells = sorted(cells.keys())
        self.by_cell = {c: np.asarray(cells[c], dtype=np.int64) for c in self.cells}
        self.n_cells = len(self.cells)
        self.per = max(1, batch // self.n_cells)
        self.bs = self.per * self.n_cells
        self.n_total = int(sum(len(v) for v in self.by_cell.values()))
        self.n_batches = max(1, min(self.n_total // self.bs, max_iters))
        self.seed = seed
        self.epoch = 0
        self.smallest = min(len(v) for v in self.by_cell.values())

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        for _ in range(self.n_batches):
            batch = []
            for c in self.cells:
                arr = self.by_cell[c]
                replace = len(arr) < self.per
                batch.extend(rng.choice(arr, size=self.per, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ------------------------------------------------------------- model ---------
def build_bridge():
    import vit_module.vit_m2f2_detector_bridge as br
    from vit_module.flash_attn_shim.mha import MHA as ShimMHA
    br.MHA = ShimMHA
    model = br.ViT_M2F2Det_Bridge(
        clip_text_encoder_name=CLIP_LOCAL, clip_vision_encoder_name=CLIP_LOCAL,
        hidden_size=768, load_vision_encoder=True, pretrained=False,
        vision_dtype=torch.float32, text_dtype=torch.float32, deepfake_dtype=torch.float32)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert len(miss) == 0, "unexpected missing keys: %s" % miss[:5]
    return model, miss, unexp


def build_lora_vit(vit_state):
    """Fresh timm ViT (same constructor as the bridge) + peft LoRA + linear head."""
    from vit_module.vit_adaptive_mattn_aps import vit_base_patch16_224
    from peft import LoraConfig, get_peft_model
    vit = vit_base_patch16_224(pretrained=False, num_classes=0)
    missing, unexpected = vit.load_state_dict(vit_state, strict=True)
    for p in vit.parameters():
        p.requires_grad_(False)
    targets = ["blocks.%d.attn.qkv" % i for i in range(12)] + \
              ["blocks.%d.attn.proj" % i for i in range(12)]
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     bias="none", target_modules=targets)
    lora_vit = get_peft_model(vit, cfg)
    head = nn.Linear(768, 2)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.constant_(head.bias, 0.0)
    return lora_vit, head


def is_adapter_key(k):
    """True for LoRA A/B matrices and the classification head (NOT the `lora_vit.` prefix)."""
    return (".lora_A." in k or ".lora_B." in k or k.endswith(".lora_A")
            or k.endswith(".lora_B") or k.startswith("head."))


class LoRADetector(nn.Module):
    def __init__(self, lora_vit, head):
        super().__init__()
        self.lora_vit = lora_vit
        self.head = head

    def forward(self, x224):
        out = self.lora_vit.base_model.model.forward_features(x224)
        cls = out[:, 0, :]
        return self.head(cls), cls


# ---------------------------------------------------------- evaluation -------
@torch.no_grad()
def extract_feats(model, bridge_ref, samples, cache, device, batch=BATCH, progress=None,
                  tag=""):
    """samples: list of (path, [_label]) ; returns (N,768) float32 features."""
    model.eval()
    outs = []
    t0 = time.time()
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        rgbs = [cache[ck(p)] for (p, _l) in chunk]
        x = torch.stack([to_clip_tensor(r) for r in rgbs]).to(device)
        vi = bridge_ref._preprocess_for_vit(x).to(torch.float32)
        _, cls = model(vi)
        outs.append(cls.float().cpu().numpy())
        if progress is not None and (i // batch) % 25 == 0:
            progress("  [extract %s] %d/%d wall=%.0fs" % (tag, i + len(chunk), len(samples),
                                                          time.time() - t0))
    return np.concatenate(outs, axis=0).astype(np.float32)


@torch.no_grad()
def predict_val(model, bridge_ref, samples, cache, device, batch=BATCH):
    model.eval()
    ys, ss = [], []
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        x = torch.stack([to_clip_tensor(cache[ck(p)]) for (p, _l, _c) in chunk]).to(device)
        vi = bridge_ref._preprocess_for_vit(x).to(torch.float32)
        logits, _ = model(vi)
        # score must be P(class 1) = P(real), because roc_auc_score() treats the
        # higher label (=1=real) as the positive class.
        p_real = F.softmax(logits, dim=1)[:, 1]
        ys.append(np.array([l for (_p, l, _c) in chunk], dtype=np.int64))
        ss.append(p_real.cpu().numpy())
    model.train()
    return np.concatenate(ys), np.concatenate(ss)


def auc_of(y, s):
    y = np.asarray(y).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------- train one arm ----
def train_arm(arm, meta, cfg, cache, cells_samples, val_samples, bridge_ref, device):
    lam = ARMS_O3.get(arm, None)
    log("")
    log("=" * 90)
    log("TRAIN ARM %s   (lambda=%s)" % (arm, lam))
    log("=" * 90)

    rng = random.Random(SEED + 7)
    samples = []
    for d in TRAIN_DOMAINS:
        for lab in (1, 0):
            cell = cfg["cell_id"][(d, lab)]
            arr = cells_samples[(d, lab)]
            for (p, l) in arr:
                samples.append((p, l, cell))
    log("[arm %s] train samples=%d over %d (domain,class) cells" %
        (arm, len(samples), len(cfg["cell_id"])))
    for (d, lab) in sorted(cfg["cell_id"]):
        log("   cell %-5s lab=%d id=%d n=%d" % (d, lab, cfg["cell_id"][(d, lab)],
                                                len(cells_samples[(d, lab)])))

    # ---- label shuffle for R1 (control) ----
    if arm == "R1":
        ys = [s[1] for s in samples]
        rng.shuffle(ys)
        samples = [(s[0], ys[i], s[2]) for i, s in enumerate(samples)]
        log("[arm R1] training labels SHUFFLED (seed=%d) ; n_real=%d n_fake=%d"
            % (SEED + 7, sum(1 for s in samples if s[1] == 1),
               sum(1 for s in samples if s[1] == 0)))

    cells = {}
    for i, s in enumerate(samples):
        cells.setdefault(s[2], []).append(i)
    sampler = BalancedBatchSampler(cells, batch=BATCH, seed=SEED,
                                   max_iters=cfg["max_iters"])
    log("[arm %s] balanced batch sampler: %d cells x %d / batch of %d ; batches/epoch=%d "
        "(iters/epoch cap=%d) ; smallest cell=%d -> ~%.1fx draws/epoch)"
        % (arm, sampler.n_cells, sampler.per, sampler.bs, sampler.n_batches,
           cfg["max_iters"], sampler.smallest,
           sampler.n_batches * sampler.per / max(1, sampler.smallest)))

    ds = MemDataset(samples, cache, train=True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    lora_vit, head = build_lora_vit(cfg["vit_state"])
    model = LoRADetector(lora_vit, head).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log("[arm %s] trainable params=%d / %d  (%.4f%%)" %
        (arm, n_train, n_total, 100.0 * n_train / n_total))

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and is_adapter_key(n) and not n.startswith("head.")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("head.")]
    log("[arm %s] lora trainable=%d  head trainable=%d"
        % (arm, sum(p.numel() for p in lora_params), sum(p.numel() for p in head_params)))

    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["max_epochs"], eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE
    curve = []
    t_arm = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        sampler.set_epoch(epoch)
        t0 = time.time()
        total = correct = 0
        loss_sum = 0.0
        ce_sum = 0.0
        pen_sum = 0.0
        n_skipped = 0
        for x, y, cell in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            cell = cell.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            vi = bridge_ref._preprocess_for_vit(x).to(torch.float32)
            logits, cls = model(vi)
            ce = crit(logits, y)
            loss = ce
            pen_val = 0.0
            if lam is not None:
                pen, ok = domain_invariance_penalty(cls, cell, cfg["n_domains"], cfg["n_classes"])
                if ok:
                    loss = ce + lam * pen
                    pen_val = float(pen.item())
                else:
                    n_skipped += 1
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            loss_sum += float(loss.item()) * y.size(0)
            ce_sum += float(ce.item()) * y.size(0)
            pen_sum += pen_val * y.size(0)
        sched.step()
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        vy, vs = predict_val(model, bridge_ref, val_samples, cache, device)
        val_auc = auc_of(vy, vs)
        curve.append((epoch, train_loss, ce_sum / max(1, total), pen_sum / max(1, total),
                      train_acc, val_auc))
        improved = val_auc > best_auc
        tag = " *BEST*" if improved else ""
        log("[arm %s epoch %02d/%d] loss=%.4f ce=%.4f pen=%.4f acc=%.4f val_auc=%.4f "
            "skip=%d lr=%.2e (%.0fs)%s"
            % (arm, epoch, cfg["max_epochs"], train_loss, ce_sum / max(1, total),
               pen_sum / max(1, total), train_acc, val_auc, n_skipped,
               opt.param_groups[0]["lr"], time.time() - t0, tag))
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if is_adapter_key(k)}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                log("[arm %s early-stop] no val_auc improvement for %d epochs at epoch %d"
                    % (arm, PATIENCE, epoch))
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                      if is_adapter_key(k)}
    model.load_state_dict(best_state, strict=False)
    wall = time.time() - t_arm
    log("[arm %s done] best_val_auc=%.4f (epoch %d) wall=%.0fs" % (arm, best_auc, best_epoch, wall))

    adapter_path = os.path.join(HERE, "lora_g19a_dfdcp_%s.pt" % arm)
    torch.save(best_state, adapter_path)
    log("[arm %s save] %s" % (arm, adapter_path))

    return dict(model=model, best_val_auc=best_auc, best_epoch=best_epoch,
                n_trainable=n_train, wall=wall, curve=curve,
                adapter_path=adapter_path, samples=samples)


def domain_invariance_penalty(cls, cell, n_domains, n_classes):
    """lambda * sum_{d,c} n_{d,c} ||mu_{d,c} - mu_c||^2 / sum n_{d,c}.

    mu_c = mean over domains of mu_{d,c}, DETACHED (stop-gradient).
    Returns (penalty, ok).  ok=False (skip) when any (domain,class) cell has <2 samples.
    """
    B, D = cls.shape
    sums = cls.new_zeros((n_domains * n_classes, D))
    cnt = cls.new_zeros((n_domains * n_classes,))
    ones = cls.new_ones((B,))
    sums.index_add_(0, cell, cls)
    cnt.index_add_(0, cell, ones)
    cnt_mat = cnt.view(n_domains, n_classes)
    if int((cnt_mat < 2).sum()) > 0:
        return cls.new_zeros(()), False
    mu_dc = (sums / cnt.unsqueeze(1)).view(n_domains, n_classes, D)
    mu_c = mu_dc.mean(dim=0).detach()                       # (n_classes, D) stop-grad
    d2 = ((mu_dc - mu_c.unsqueeze(0)) ** 2).sum(dim=-1)     # (n_domains, n_classes)
    pen = (cnt_mat * d2).sum() / cnt_mat.sum()
    return pen, True


# ------------------------------------------------------------- probe ---------
def probe_protocol(feat, meta, logline):
    """StandardScaler(fit probe-train) + LR(C=1e-3, lbfgs, max_iter=3000)."""
    tm = meta["train_mask"]
    tr = np.where(tm)[0]
    assert len(tr) == 2200 and bool(tm[:2200].all()) and not bool(tm[2200:].any())
    te = np.arange(2200, 3000)          # in-domain FF++ probe-test = rows 2200..2999
    yb = (meta["y"] == 0).astype(np.int64)      # positive = fake
    sc = StandardScaler().fit(feat[tr])
    clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000, random_state=0)
    clf.fit(sc.transform(feat[tr]), yb[tr])

    res = {}
    # in-domain FF++ probe-test (800)
    s_te = clf.decision_function(sc.transform(feat[te]))
    res["in_domain"] = float(roc_auc_score(yb[te], s_te))
    # 5 target domains (300 each)
    dom = np.array([str(d) for d in meta["domain"]])
    for d in ALL_TARGET_DOMAINS:
        idx = np.where(dom == d)[0]
        s = clf.decision_function(sc.transform(feat[idx]))
        res[d] = float(roc_auc_score(yb[idx], s))
        res["n_" + d] = int(len(idx))
    res["cd_mean_all5"] = float(np.mean([res[d] for d in ALL_TARGET_DOMAINS]))
    res["cd_mean_clean"] = float(np.mean([res[HOLD_OUT]]))
    return res


def g12_ratio(feat, meta):
    """G12 domain-sensitivity ratio over the 5 multi target domains."""
    tm = meta["train_mask"]
    tr = np.where(tm)[0]
    Xtr = feat[tr].astype(np.float64)
    ytr = meta["y"][tr]
    s = float(np.sqrt(np.var(Xtr, axis=0, ddof=1).mean()))
    mu_src = Xtr.mean(axis=0)
    mu_real = Xtr[ytr == 1].mean(axis=0)
    mu_fake = Xtr[ytr == 0].mean(axis=0)
    class_gap = float(np.linalg.norm(mu_fake - mu_real) / s)
    dom = np.array([str(d) for d in meta["domain"]])
    per = {}
    for d in ALL_TARGET_DOMAINS:
        Xd = feat[np.where(dom == d)[0]].astype(np.float64)
        per[d] = float(np.linalg.norm(Xd.mean(axis=0) - mu_src) / s) / class_gap
    return per, float(np.mean(list(per.values()))), s, class_gap


# ------------------------------------------------------------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="R0,R1,O1,O3a,O3b")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--cap-ffpp", type=int, default=4000)
    ap.add_argument("--cap-target", type=int, default=4000)
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS_PER_EPOCH)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = query_gpu()
    log("[gpu] nvidia-smi query (index, mem_used MiB, util%%)=%s" % (gpus,))
    log("[gpu] CUDA_VISIBLE_DEVICES=%s (target physical GPU index 1)" %
        os.environ.get("CUDA_VISIBLE_DEVICES"))
    g1 = [g for g in gpus if g[0] == 1]
    if g1:
        used = g1[0][1]
        log("[gpu] physical GPU 1 mem.used=%dMiB -> %s" %
            (used, "IDLE (<=100MiB)" if used <= 100 else "BUSY (>100MiB)"))
        if used > 100:
            log("[gpu] ABORT: GPU 1 is not free. Not switching cards.")
            log.close()
            sys.exit(2)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    log("[torch] %s ; device=%s ; current_device=%s ; num_threads=%d ; cudnn.benchmark=%s"
        % (torch.__version__, torch.cuda.get_device_name(0),
           torch.cuda.current_device(), torch.get_num_threads(),
           torch.backends.cudnn.benchmark))
    log("[cpu] OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s VECLIB=%s BLIS=%s ; cv2 threads=%d ; "
        "num_workers=0" % (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"),
                           os.environ.get("OPENBLAS_NUM_THREADS"),
                           os.environ.get("NUMEXPR_NUM_THREADS"),
                           os.environ.get("VECLIB_MAXIMUM_THREADS"),
                           os.environ.get("BLIS_NUM_THREADS"), cv2.getNumThreads()))

    # ---------------- 5300-row metadata + alignment ----------------
    meta = load_5300_meta()
    log("[align] 5300-row paths byte-equal to _g17/cnn_feats.npz: True (assert passed)")
    log("[align] probe rows 0..2999 (train 0..2199, test 2200..2999) ; "
        "multi rows 3000..5299")

    probe_keys, probe_rows = build_probe_folder_set(meta)
    log("[probe] probe folder directory-chain keys = %d (from all %d probe rows)"
        % (len(probe_keys), len(probe_rows)))

    ffpp_videos, n_missing, n_excluded = build_ffpp_videos(probe_keys)
    log("[ffpp] train.json videos: real=%d fake=%d total=%d (missing_dirs=%d "
        "excluded_by_probe_dirchain=%d)"
        % (sum(1 for v in ffpp_videos if v[1] == 1),
           sum(1 for v in ffpp_videos if v[1] == 0), len(ffpp_videos), n_missing, n_excluded))

    target_videos = build_target_videos(meta, ["cd2", "ffiw", "wild"])
    log("[target] training-domain videos: %d (cd2=%d ffiw=%d wild=%d)"
        % (len(target_videos),
           sum(1 for v in target_videos if v[2].startswith("cd2_")),
           sum(1 for v in target_videos if v[2].startswith("ffiw_")),
           sum(1 for v in target_videos if v[2].startswith("wild_"))))

    # ---------------- leakage audits ----------------
    train_folders = set(path_key(v[0]) for v in ffpp_videos) | \
                    set(path_key(v[0]) for v in target_videos)
    dom = np.array([str(d) for d in meta["domain"]])
    for hd in [HOLD_OUT, "cd1"]:
        ho = set(path_key(os.path.dirname(str(meta["paths"][i])))
                 for i in np.where(dom == hd)[0])
        inter = train_folders & ho
        log("[leak] held-out domain %-5s folders present in training (dir-chain): %d"
            % (hd, len(inter)))
        assert len(inter) == 0, "LEAK: %s folders in training" % hd
    log("[leak] probe folder dir-chain keys present in FF++ training: %d"
        % len(train_folders & probe_keys))
    assert len(train_folders & probe_keys) == 0
    for d in ["cd2", "ffiw", "wild"]:
        dd = set(path_key(os.path.dirname(str(meta["paths"][i])))
                 for i in np.where(dom == d)[0])
        log("[leak] CONTAMINATION (expected) %-5s eval folders also in training: %d/%d"
            % (d, len(dd & train_folders), len(dd)))

    # ---------------- build model, load ckpt, keep a vit weight snapshot ----------
    log("")
    log("[model] building ViT_M2F2Det_Bridge + loading bridge_v2_phase1.pth")
    bridge, miss, unexp = build_bridge()
    log("[model] load_state_dict strict=False -> missing=%d unexpected=%d" % (len(miss), len(unexp)))
    log("[model] model.vit = %s ; params=%.1fM ; blocks=%d ; attn has qkv/proj"
        % (type(bridge.vit).__name__,
           sum(p.numel() for p in bridge.vit.parameters()) / 1e6,
           len(bridge.vit.blocks)))
    for p in bridge.vit.parameters():
        p.requires_grad_(False)
    vit_state = {k: v.detach().clone() for k, v in bridge.vit.state_dict().items()}
    probe_vit = None                       # keep bridge.vit on CPU; move later

    # ---------------- R0 : frozen extraction (sanity gate) ----------------
    log("")
    log("=" * 90)
    log("ARM R0 -- frozen ViT feature extraction (pipeline sanity gate)")
    log("=" * 90)
    n_all = 5300
    all_paths = [str(p) for p in meta["paths"]]
    cache = {}
    t0 = time.time()
    for p in all_paths:
        if ck(p) not in cache:
            cache[ck(p)] = read_rgb336(p)
    log("[preload] decoded all %d eval images in %.1fs" % (len(cache), time.time() - t0))

    # R0 uses bridge.vit directly (identical to model._preprocess_for_vit)
    f0_path = os.path.join(HERE, "feats_g19a_dfdcp_R0.npz")

    def frozen_extract():
        bridge.vit.to(device).eval()
        outs = []
        t0 = time.time()
        for i in range(0, n_all, BATCH):
            chunk = all_paths[i:i + BATCH]
            x = torch.stack([to_clip_tensor(cache[ck(p)]) for p in chunk]).to(device)
            vi = bridge.vit.forward_features(bridge._preprocess_for_vit(x).to(torch.float32))
            outs.append(vi[:, 0, :].float().cpu().numpy())
            if (i // BATCH) % 25 == 0:
                log("  [extract R0] %d/%d wall=%.0fs" % (i + len(chunk), n_all, time.time() - t0))
        return np.concatenate(outs, axis=0).astype(np.float32)

    if "R0" in args.arms.split(","):
        t_r0 = time.time()
        feat_R0 = frozen_extract()
        assert feat_R0.shape == (5300, 768), feat_R0.shape
        wall_R0 = time.time() - t_r0
        np.savez(f0_path, feat=feat_R0, paths=np.array(all_paths, dtype=object))
        log("[R0] feat shape=%s wall=%.0fs saved %s" % (feat_R0.shape, wall_R0, f0_path))

        # cross-check vs stored probe/multi V
        pz = np.load(PROBE_NPZ, allow_pickle=True)
        mz = np.load(MULTI_NPZ, allow_pickle=True)
        d_probe = float(np.abs(feat_R0[:3000] - pz["V"].astype(np.float32)).max())
        d_multi = float(np.abs(feat_R0[3000:] - mz["V"].astype(np.float32)).max())
        log("[R0] maxabs deviation vs probe_feats.npz['V'] (3000 rows) = %.3e" % d_probe)
        log("[R0] maxabs deviation vs feats_multi.npz['V'] (2300 rows) = %.3e" % d_multi)

        r0 = probe_protocol(feat_R0, meta, log)
        log("[R0] in-domain FF++ probe-test AUC=%.4f (anchor %.4f, |dev|=%.4f)"
            % (r0["in_domain"], ANCHOR_IN_DOMAIN, abs(r0["in_domain"] - ANCHOR_IN_DOMAIN)))
        devs = {}
        for d in ALL_TARGET_DOMAINS:
            devs[d] = abs(r0[d] - ANCHOR_FROZEN[d])
            log("[R0] target %-5s AUC=%.4f (frozen anchor %.4f, |dev|=%.4f) n=%d"
                % (d, r0[d], ANCHOR_FROZEN[d], devs[d], r0["n_" + d]))
        maxdev = max(devs.values())
        log("[R0] max anchor deviation over 5 domains = %.4f (gate 0.005) -> %s"
            % (maxdev, "PASS" if maxdev <= 0.005 else "FAIL"))
        ratio_R0_per, ratio_R0, s_R0, cg_R0 = g12_ratio(feat_R0, meta)
        log("[R0] G12 s=%.4f class_gap=%.4f ratio=%.4f (G17c anchor ratio_V=%.4f, |dev|=%.4f)"
            % (s_R0, cg_R0, ratio_R0, ANCHOR_RATIO_V, abs(ratio_R0 - ANCHOR_RATIO_V)))
        for d in ALL_TARGET_DOMAINS:
            log("      ratio[%s]=%.4f" % (d, ratio_R0_per[d]))
        if maxdev > 0.005:
            log("[GATE] PIPELINE SANITY GATE FAILED (maxdev %.4f > 0.005) -- STOPPING" % maxdev)
            log.close()
            sys.exit(3)
        log("[GATE] PIPELINE SANITY GATE PASSED")
        del feat_R0
        gc.collect()

    # ---------------- training data assembly ----------------
    rng = random.Random(SEED)
    cells_samples = {}
    counts = {}
    for d in TRAIN_DOMAINS:
        for lab in (1, 0):
            counts[(d, lab)] = 0
    # FF++ frames
    ffpp_cache = {}
    for folder, label, vkey in ffpp_videos:
        cap = REAL_FRAMES if label == 1 else FAKE_FRAMES
        cells_samples.setdefault(("ffpp", label), []).extend(
            [(fp, label) for fp in sample_frames(folder, cap)])
    for d in ["cd2", "ffiw", "wild"]:
        for folder, label, vkey in target_videos:
            if not vkey.startswith(d + "_"):
                continue
            cells_samples.setdefault((d, label), []).extend(
                [(fp, label) for fp in sample_frames(folder, TARGET_MAX_PER_VIDEO)])
    for k in list(cells_samples.keys()):
        counts[k] = len(cells_samples[k])
    log("")
    log("[data] raw per-(domain,class) frame counts: %s"
        % {("%s/%d" % k): v for k, v in sorted(counts.items())})
    cap = args.cap_ffpp
    for k in list(cells_samples.keys()):
        cap_k = args.cap_target if k[0] != "ffpp" else args.cap_ffpp
        cells_samples[k] = cap_by_class(cells_samples[k], cap_k, rng)
    log("[data] capped per-cell counts (ffpp cap=%d, target cap=%d): %s"
        % (args.cap_ffpp, args.cap_target,
           {("%s/%d" % k): len(v) for k, v in sorted(cells_samples.items())}))
    log("[data] total training frames = %d" % sum(len(v) for v in cells_samples.values()))

    cell_id = {}
    nxt = 0
    for d in TRAIN_DOMAINS:
        for lab in (1, 0):
            cell_id[(d, lab)] = nxt
            nxt += 1
    cfg = dict(cell_id=cell_id, n_domains=len(TRAIN_DOMAINS), n_classes=2,
               vit_state=vit_state, max_epochs=args.max_epochs,
               patience=args.patience, max_iters=args.max_iters)

    if args.smoke:
        log("[smoke] sub-sampling each cell to <= %d frames" % args.smoke_n)
        for k in list(cells_samples.keys()):
            arr = cells_samples[k]
            rng.shuffle(arr)
            cells_samples[k] = arr[:args.smoke_n]
        log("[smoke] per-cell n = %s" % {("%s/%d" % k): len(v)
                                         for k, v in sorted(cells_samples.items())})

    # ---------------- val split (training domains ONLY) ----------------
    all_train_videos = ffpp_videos + target_videos
    real_vids = [v for v in all_train_videos if v[1] == 1]
    fake_vids = [v for v in all_train_videos if v[1] == 0]
    rng_v = random.Random(SEED)
    rng_v.shuffle(real_vids)
    rng_v.shuffle(fake_vids)
    n_val_real = max(1, int(round(len(real_vids) * VAL_FRAC)))
    n_val_fake = max(1, int(round(len(fake_vids) * VAL_FRAC)))
    val_videos = real_vids[:n_val_real] + fake_vids[:n_val_fake]
    train_videos = set(v[0] for v in (real_vids[n_val_real:] + fake_vids[n_val_fake:]))
    log("[split] val videos=%d (real %d / fake %d) ; held-out fold=%s ; "
        "NO dfdcp/cd1/probe video is used for training or selection"
        % (len(val_videos), n_val_real, n_val_fake, HOLD_OUT))
    assert not (set(v[0] for v in val_videos) & set(v[0] for v in target_videos
                                                   if v[2].startswith(HOLD_OUT + "_")))
    # val must not use any frame that is in the training cells
    train_paths = set()
    for k, v in cells_samples.items():
        for (p, _l) in v:
            train_paths.add(path_key(p))
    val_samples = []
    for folder, label, vkey in val_videos:
        for fp in sample_frames(folder, VAL_MAX_PER_VIDEO):
            if path_key(fp) in train_paths:
                continue
            val_samples.append((fp, label, cell_id[("ffpp" if vkey.startswith("ffpp") else
                                                    vkey.split("_")[0], label)]))
    log("[split] val frames=%d (real %d / fake %d), video-held-out"
        % (len(val_samples), sum(1 for s in val_samples if s[1] == 1),
           sum(1 for s in val_samples if s[1] == 0)))
    # balance val classes so val AUC is not dominated by the (huge) FF++ fake pool
    vr = [s for s in val_samples if s[1] == 1]
    vf = [s for s in val_samples if s[1] == 0]
    nv = min(len(vr), len(vf))
    rng_v.shuffle(vr)
    rng_v.shuffle(vf)
    val_samples = vr[:nv] + vf[:nv]
    rng_v.shuffle(val_samples)
    log("[split] val balanced -> %d frames (real %d / fake %d)"
        % (len(val_samples), sum(1 for s in val_samples if s[1] == 1),
           sum(1 for s in val_samples if s[1] == 0)))
    if args.dry_run:
        log("")
        log("[dry-run] data audit complete; stopping before model build.")
        log("[dry-run] planned training frames = %d"
            % sum(len(v) for v in cells_samples.values()))
        log.close()
        return

    # ---- pre-decode ALL training + val frames once to RAM (uint8 336) ----
    all_train_paths = set()
    for _k, _v in cells_samples.items():
        for (pp, _ll) in _v:
            all_train_paths.add(pp)
    for pp in all_train_paths:
        if ck(pp) not in cache:
            cache[ck(pp)] = read_rgb336(pp)
    for (p, _l, _c) in val_samples:
        if ck(p) not in cache:
            cache[ck(p)] = read_rgb336(p)
    log("[preload] cache total = %d images (~%.2f GB uint8 336x336x3)"
        % (len(cache), len(cache) * IMG_SIZE * IMG_SIZE * 3 / 1e9))

    # ---------------- run arms ----------------
    arms = args.arms.split(",")
    results = {}
    feat_paths = {}
    for arm in arms:
        if arm == "R0":
            continue
        if args.smoke and arm != "O1":
            continue
        res = train_arm(arm, meta, cfg, cache, cells_samples, val_samples, bridge, device)
        model = res.pop("model")
        eval_samples = [(str(meta["paths"][i]), int(meta["y"][i])) for i in range(n_all)]
        feat = extract_feats(model, bridge, eval_samples, cache, device,
                             tag=arm, progress=log)
        assert feat.shape == (5300, 768), feat.shape
        fp = os.path.join(HERE, "feats_g19a_dfdcp_%s.npz" % arm)
        np.savez(fp, feat=feat, paths=np.array(all_paths, dtype=object))
        log("[arm %s] feats saved %s shape=%s" % (arm, fp, feat.shape))
        results[arm] = res
        feat_paths[arm] = (fp, feat)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if args.smoke:
        log("")
        log("[smoke] DONE. arms run: %s" % ",".join(results.keys()))
        log("[wall] total = %.1f s" % (time.time() - t_start))
        log.close()
        return

    # ---------------- R0 features (reuse if already computed) ----------------
    if "R0" in arms and "R0" not in feat_paths:
        feat_paths["R0"] = (f0_path, np.load(f0_path, allow_pickle=True)["feat"])

    # ---------------- probes / ratios / gates ----------------
    log("")
    log("=" * 90)
    log("G19a RESULTS -- held-out fold = %s" % HOLD_OUT)
    log("=" * 90)

    probe_res = {}
    ratio_res = {}
    for arm in ARMS_ALL:
        if arm not in feat_paths:
            continue
        fp, feat = feat_paths[arm]
        pr = probe_protocol(feat, meta, log)
        rr_per, rr, ss, cg = g12_ratio(feat, meta)
        probe_res[arm] = pr
        ratio_res[arm] = dict(per=rr_per, mean=rr, s=ss, class_gap=cg)

    hdr = "%-6s | %8s |" % ("arm", "in-dom")
    for d in ALL_TARGET_DOMAINS:
        hdr += " %9s" % d
    hdr += " | %9s" % "mean5"
    log("")
    log("PRIMARY TABLE -- cross-domain AUC (probe: StandardScaler(probe-train)+LR C=1e-3)")
    log(hdr)
    log("-" * len(hdr))
    for arm in ARMS_ALL:
        if arm not in probe_res:
            continue
        r = probe_res[arm]
        line = "%-6s | %8.4f |" % (arm, r["in_domain"])
        for d in ALL_TARGET_DOMAINS:
            line += " %9.4f" % r[d]
        line += " | %9.4f" % r["cd_mean_all5"]
        log(line)
    log("-" * len(hdr))
    line = "%-6s | %8.4f |" % ("ANCHOR", ANCHOR_IN_DOMAIN)
    for d in ALL_TARGET_DOMAINS:
        line += " %9.4f" % ANCHOR_FROZEN[d]
    line += " | %9.4f" % ANCHOR_FROZEN_MEAN
    log(line + "   (prior frozen-ViT anchors, same protocol)")

    log("")
    log("CONTAMINATION FLAGS for fold=%s: training domains = %s ; held-out = %s"
        % (HOLD_OUT, TRAIN_DOMAINS, HOLD_OUT))
    log("  columns marked CONTAMINATED (excluded from aggregates): %s"
        % ", ".join(CONTAMINATED_COLUMNS))
    log("  VALID transfer column: %s" % HOLD_OUT)

    log("")
    log("IN-DOMAIN GUARD (FF++ probe-test 800 rows, train_mask False)")
    for arm in ARMS_ALL:
        if arm in probe_res:
            log("  %-5s in-domain AUC=%.4f   (frozen baseline %.4f)"
                % (arm, probe_res[arm]["in_domain"], ANCHOR_IN_DOMAIN))

    log("")
    log("G12 DOMAIN-SENSITIVITY RATIO (s / class_gap printed for cross-check)")
    log("%-6s %10s %10s | %s | %10s" % ("arm", "s", "class_gap",
                                        " ".join("%9s" % d for d in ALL_TARGET_DOMAINS),
                                        "mean"))
    for arm in ARMS_ALL:
        if arm not in ratio_res:
            continue
        rr = ratio_res[arm]
        log("%-6s %10.4f %10.4f | %s | %10.4f"
            % (arm, rr["s"], rr["class_gap"],
               " ".join("%9.4f" % rr["per"][d] for d in ALL_TARGET_DOMAINS), rr["mean"]))
    log("  (G17c frozen-ViT anchor: ratio_V=%.4f)" % ANCHOR_RATIO_V)

    # ---------------- gates ----------------
    have = [a for a in ARMS_ALL if a in probe_res]
    hold = {}
    for a in ["R0", "R1", "O1", "O3a", "O3b"]:
        if a in probe_res:
            hold[a] = probe_res[a][HOLD_OUT]
    log("")
    log("HELD-OUT DOMAIN (%s) AUC BY ARM: %s" % (HOLD_OUT, {k: round(v, 4) for k, v in hold.items()}))

    gates = {}
    r0 = hold.get("R0", float("nan"))
    o1 = hold.get("O1", float("nan"))
    r1 = hold.get("R1", float("nan"))
    o3 = [hold[a] for a in ["O3a", "O3b"] if a in hold]
    best_o3 = max(o3) if o3 else float("nan")
    trained = [a for a in ["R1", "O1", "O3a", "O3b"] if a in hold]
    best_arm = max(trained, key=lambda a: hold[a]) if trained else None
    best = hold[best_arm] if best_arm else float("nan")
    best_o3_arm = max(["O3a", "O3b"], key=lambda a: hold.get(a, -1)) if o3 else None
    log("  best-overall trained arm = %s (%.4f) ; best O3 arm = %s (%.4f)"
        % (best_arm, best, best_o3_arm, best_o3))

    if "O1" in hold and "R0" in hold:
        gates["G19_UNFREEZE_HELPS"] = (o1 - r0, o1 - r0 >= 0.01)
    if o3 and "O1" in hold:
        gates["G19_INVARIANCE_HELPS"] = (best_o3 - o1, best_o3 - o1 >= 0.02)
    if trained and "R1" in hold:
        gates["G19_VS_RANDLABEL"] = (best - r1, best - r1 >= 0.02)
        if o3:
            gates["G19_VS_RANDLABEL[bestO3]"] = (best_o3 - r1, best_o3 - r1 >= 0.02)
    if best_arm is not None:
        ind = probe_res[best_arm]["in_domain"]
        gates["G19_NO_FORGETTING"] = (ind, ind >= ANCHOR_IN_DOMAIN - 0.01)
        for _a in trained:
            log("  [no-forgetting detail] arm %-4s in-domain=%.4f (baseline-0.01=%.4f) -> %s"
                % (_a, probe_res[_a]["in_domain"], ANCHOR_IN_DOMAIN - 0.01,
                   "OK" if probe_res[_a]["in_domain"] >= ANCHOR_IN_DOMAIN - 0.01 else "FORGETTING"))
    if trained:
        crm = (best - ANCHOR_SRC_LR_XDOM) / (ANCHOR_ORACLE_TARGET - ANCHOR_SRC_LR_XDOM)
        gates["G19_CEILING_RECOVERY"] = (crm, crm >= 0.30)
        if o3:
            crm3 = (best_o3 - ANCHOR_SRC_LR_XDOM) / (ANCHOR_ORACLE_TARGET - ANCHOR_SRC_LR_XDOM)
            gates["G19_CEILING_RECOVERY[bestO3]"] = (crm3, crm3 >= 0.30)

    log("")
    log("GATE VERDICTS (mechanical)")
    for k, (v, ok) in gates.items():
        log("  %-24s value=%+.4f  ->  %s" % (k, v, "PASS" if ok else "FAIL"))

    kill = None
    if "O1" in hold and "R1" in hold and "R0" in hold:
        dO1 = abs(o1 - r0)
        dR1 = abs(r1 - r0)
        kill = (dO1 <= 0.005 and dR1 <= 0.005)
        log("")
        log("KILL SIGNAL CHECK: |O1-R0|=%.4f  |R1-R0|=%.4f  both <=0.005 -> %s"
            % (dO1, dR1, "KILL (unfreezing did not move the representation)"
               if kill else "not triggered"))

    log("")
    log("WALL / RESOURCE")
    for arm in ARMS_ALL:
        if arm in results:
            log("  arm %-4s wall=%.0fs best_val_auc=%.4f best_epoch=%d trainable=%d"
                % (arm, results[arm]["wall"], results[arm]["best_val_auc"],
                   results[arm]["best_epoch"], results[arm]["n_trainable"]))
    if "R0" in feat_paths:
        pass
    log("  threads: torch=%d OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s ; cv2=%d ; num_workers=0"
        % (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"),
           os.environ.get("MKL_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"),
           os.environ.get("NUMEXPR_NUM_THREADS"), cv2.getNumThreads()))
    log("[wall] total = %.1f s" % (time.time() - t_start))

    # ---------------- stats npz ----------------
    stats = {}
    for arm in ARMS_ALL:
        if arm not in probe_res:
            continue
        r = probe_res[arm]
        stats["in_domain_%s" % arm] = np.float64(r["in_domain"])
        for d in ALL_TARGET_DOMAINS:
            stats["auc_%s_%s" % (arm, d)] = np.float64(r[d])
        stats["auc_mean5_%s" % arm] = np.float64(r["cd_mean_all5"])
        rr = ratio_res[arm]
        stats["ratio_%s" % arm] = np.float64(rr["mean"])
        stats["class_gap_%s" % arm] = np.float64(rr["class_gap"])
        stats["s_%s" % arm] = np.float64(rr["s"])
        for d in ALL_TARGET_DOMAINS:
            stats["ratio_%s_%s" % (arm, d)] = np.float64(rr["per"][d])
        if arm in results:
            stats["wall_%s" % arm] = np.float64(results[arm]["wall"])
            stats["trainable_%s" % arm] = np.int64(results[arm]["n_trainable"])
            stats["best_val_auc_%s" % arm] = np.float64(results[arm]["best_val_auc"])
            stats["best_epoch_%s" % arm] = np.int64(results[arm]["best_epoch"])
    for k, (v, ok) in gates.items():
        stats["gate_%s" % k] = np.float64(v)
        stats["gate_%s_pass" % k] = np.bool_(ok)
    stats["holdout_auc_best_arm"] = np.float64(best)
    np.savez(STATS_PATH, **stats)
    log("[save] stats -> %s" % STATS_PATH)

    # ---------------- final nvidia-smi ----------------
    gpus_end = query_gpu()
    log("[gpu-end] nvidia-smi query (index, mem_used MiB, util%%)=%s" % (gpus_end,))

    # ---------------- report ----------------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log.buf) + "\n")
    log("[save] report -> %s" % REPORT_PATH)
    log.close()


if __name__ == "__main__":
    main()
