# -*- coding: utf-8 -*-
"""
G19b -- DECISIVE ABLATION for the G19a result.

G19a (held-out fold = dfdcp) found: frozen R0 = 0.8261 ; arm O1 (CE, LoRA r=16,
sources = FF++ + cd2 + ffiw + wild) = 0.9320 (+0.106).  That gain can come from
    (a) unfreezing/retraining the ViT branch per se, or
    (b) merely from having a training source (cd2 = Celeb-DF-v2) that is
        distributionally CLOSE to DFDC-P.
Only an FF++-ONLY arm can separate (a) from (b).  This script runs that ablation.

Held-out fold: dfdcp (identical to G19a).  cd1 is ALWAYS excluded (its folder set
is a subset of cd2).  dfdcp is never used for training or for any model-selection
decision.  No TTA / no target-domain BN update / no pseudo-labels.

Arms (7 trained + R0 anchor):
    O1multi_s0/s1/s2   CE, LoRA r=16 alpha=32, sources = FF++ + cd2 + ffiw + wild, seed 0/1/2
    O1ffonly_s0/s1/s2  CE, LoRA r=16 alpha=32, sources = FF++ (official splits/train.json) ONLY, seed 0/1/2
    R2multi_s0         CE, LoRA r=32 alpha=64 (rank doubled), sources = MULTI, seed 0
    R0                 frozen ViT features (pipeline sanity anchor; must reproduce
                       dfdcp 0.8261 within 0.005)

New in G19b vs G19a
-------------------
* sources parameterised per arm (FFONLY / MULTI) ; the FF++ training pool is built
  ONCE and shared verbatim by every arm, so the *only* difference between
  O1multi_* and O1ffonly_* is the presence of the cd2/ffiw/wild cells.
* one deliberate, documented deviation from G19a: the early-stopping val split is
  drawn from FF++ videos ONLY (10% of FF++ real videos + 10% of FF++ fake folders,
  video-disjoint from the training pool), for EVERY arm.  G19a's val pool included
  cd2/ffiw/wild videos because all of them were training domains there; that is
  not a *training*-domain val for the FFONLY arm, so using it would leak target
  domains into model selection.  A single shared FF++ val keeps the source
  ablation unconfounded.  Consequence: the O1multi numbers here are not a bitwise
  reproduction of G19a's O1 (which also used a different seed, 20260910).
* per-arm ORACLE (own-label, video-disjoint 70/30 within the target domain) and
  the ALIGNMENT RATIO = (source-label AUC on domain) / (oracle AUC on domain).

Unchanged, reused verbatim from train_g19a.py:
  * LoRA target modules (blocks[*].attn.qkv + .proj), AdamW, lr LoRA 1e-4 / head
    1e-3, cosine, batch 32, <=12 epochs, patience 4, CE, balanced (domain,class)
    batch sampler, deployment-identical preprocessing, the 5300-image layout and
    its byte-level assertion against _g17/cnn_feats.npz, the LODO enumeration and
    leakage audits (from _g18b), the standard probe
    (StandardScaler(fit probe-train 2200) + LogisticRegression(C=1e-3, lbfgs,
    max_iter=3000, random_state=0)), and the G12 ratio definition.

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
import hashlib
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

LOG_PATH = os.path.join(HERE, "run_log_g19b_dfdcp.txt")
REPORT_PATH = os.path.join(HERE, "g19b_dfdcp_report.txt")
STATS_PATH = os.path.join(HERE, "g19b_dfdcp_stats.npz")

FFPP_RAW = "F:/zhj/data/FaceForensic++_raw"
FFPP_TRAIN_JSON = os.path.join(FFPP_RAW, "splits", "train.json")
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# ---------------------------------------------------------------- constants --
SEED = 20260910                 # data-enumeration / val-split seed (G19a value)
SPLIT_SEED = 12345              # fixed seed for the within-domain oracle 70/30 split
IMG_SIZE = 336
NUM_THREADS = 4
HOLD_OUT = "dfdcp"
DOMAIN_ORDER = ["ffpp", "cd2", "ffiw", "wild"]      # cell-id order, G19a order
MULTI_SOURCES = ["ffpp", "cd2", "ffiw", "wild"]
FFONLY_SOURCES = ["ffpp"]
ALL_TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
CONTAMINATED_COLUMNS = ["cd2", "ffiw", "wild"]
VALID_TRANSFER_COLUMN = "dfdcp"
ORACLE_DOMAINS = ["dfdcp", "wild"]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

REAL_FRAMES = 15            # per FF++ real video   (G18b / G19a convention)
FAKE_FRAMES = 3             # per FF++ fake folder  (G18b / G19a convention)
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
LORA_DROPOUT = 0.0

# baseline anchors (prior experiments, identical probe protocol)
ANCHOR_FROZEN = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_FROZEN_MEAN = 0.8303
ANCHOR_IN_DOMAIN = 0.9852
ANCHOR_RATIO_V = 0.1983
G19A_O1_DFDCP = 0.9320      # G19a arm O1 (seed 20260910, multi-domain val) -- reference only
G19A_R0_DFDCP = 0.8261

# ------------------------------------------------------------------- arms -----
def _spec(name, sources, seed, r=16, alpha=32):
    return dict(name=name, sources=list(sources), seed=int(seed), r=int(r), alpha=int(alpha))


ARM_SPECS = [
    _spec("O1multi_s0", MULTI_SOURCES, 0),
    _spec("O1multi_s1", MULTI_SOURCES, 1),
    _spec("O1multi_s2", MULTI_SOURCES, 2),
    _spec("O1ffonly_s0", FFONLY_SOURCES, 0),
    _spec("O1ffonly_s1", FFONLY_SOURCES, 1),
    _spec("O1ffonly_s2", FFONLY_SOURCES, 2),
    _spec("R2multi_s0", MULTI_SOURCES, 0, r=32, alpha=64),
]
ARM_NAMES = [s["name"] for s in ARM_SPECS]
ALL_RUNS = ["R0"] + ARM_NAMES


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


def sha1_params(pairs):
    h = hashlib.sha1()
    for _n, t in pairs:
        h.update(np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tobytes())
    return h.hexdigest()[:12]


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
    return _TF(image=np.ascontiguousarray(rgb336))["image"]


def aug_train(rgb336):
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
    videos = []
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
            for vid in ("%s_%s" % (a, b), "%s_%s" % (b, a)):
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
        self.samples = samples
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
    ck_ = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck_.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert len(miss) == 0, "unexpected missing keys: %s" % miss[:5]
    return model, miss, unexp


def build_lora_vit(vit_state, r, alpha):
    """Fresh timm ViT (same constructor as the bridge) + peft LoRA + linear head.

    NOTE: the caller MUST set torch.manual_seed(arm_seed) before this call --
    peft initialises lora_A with kaiming_uniform_ (global RNG) and lora_B to ZERO,
    so the seed changes the LoRA init (through lora_A, and through the head below).
    """
    from vit_module.vit_adaptive_mattn_aps import vit_base_patch16_224
    from peft import LoraConfig, get_peft_model
    vit = vit_base_patch16_224(pretrained=False, num_classes=0)
    vit.load_state_dict(vit_state, strict=True)
    for p in vit.parameters():
        p.requires_grad_(False)
    targets = ["blocks.%d.attn.qkv" % i for i in range(12)] + \
              ["blocks.%d.attn.proj" % i for i in range(12)]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=LORA_DROPOUT,
                     bias="none", target_modules=targets)
    lora_vit = get_peft_model(vit, cfg)
    head = nn.Linear(768, 2)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.constant_(head.bias, 0.0)
    return lora_vit, head


def is_adapter_key(k):
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
def train_arm(spec, meta, cells_all, val_samples, bridge_ref, device, vit_state, cache,
              max_iters=MAX_ITERS_PER_EPOCH, smoke=False):
    arm = spec["name"]
    seed = spec["seed"]
    domains = spec["sources"]
    r = spec["r"]
    alpha = spec["alpha"]

    log("")
    log("=" * 90)
    log("TRAIN ARM %s   (objective=CE, sources=%s, seed=%d, LoRA r=%d alpha=%d)"
        % (arm, "+".join(domains), seed, r, alpha))
    log("=" * 90)

    # ---- seed everything that must change per seed (LoRA init, head init, order) ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cell_id = {}
    nxt = 0
    for d in [x for x in DOMAIN_ORDER if x in domains]:
        for lab in (1, 0):
            cell_id[(d, lab)] = nxt
            nxt += 1
    n_domains = len([x for x in DOMAIN_ORDER if x in domains])

    samples = []
    for d in [x for x in DOMAIN_ORDER if x in domains]:
        for lab in (1, 0):
            for (p, l) in cells_all[(d, lab)]:
                samples.append((p, l, cell_id[(d, lab)]))
    log("[arm %s] train samples=%d over %d (domain,class) cells" %
        (arm, len(samples), len(cell_id)))
    for (d, lab) in sorted(cell_id):
        log("   cell %-5s lab=%d id=%d n=%d" % (d, lab, cell_id[(d, lab)],
                                                len(cells_all[(d, lab)])))

    cells = {}
    for i, s in enumerate(samples):
        cells.setdefault(s[2], []).append(i)
    sampler = BalancedBatchSampler(cells, batch=BATCH, seed=seed,
                                   max_iters=max_iters)
    log("[arm %s] balanced batch sampler: %d cells x %d / batch of %d ; batches/epoch=%d "
        "(iters/epoch cap=%d) ; smallest cell=%d -> ~%.1fx draws/epoch) ; sampler seed=%d"
        % (arm, sampler.n_cells, sampler.per, sampler.bs, sampler.n_batches,
           max_iters, sampler.smallest,
           sampler.n_batches * sampler.per / max(1, sampler.smallest), seed))

    ds = MemDataset(samples, cache, train=True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    lora_vit, head = build_lora_vit(vit_state, r, alpha)
    model = LoRADetector(lora_vit, head).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    loraA_hash = sha1_params([(n, p) for n, p in model.named_parameters()
                              if ".lora_A." in n])
    loraB_hash = sha1_params([(n, p) for n, p in model.named_parameters()
                              if ".lora_B." in n])
    head_hash = sha1_params([(n, p) for n, p in model.named_parameters()
                             if n.startswith("head.")])
    log("[arm %s] trainable params=%d / %d  (%.4f%%)" %
        (arm, n_train, n_total, 100.0 * n_train / n_total))
    log("[arm %s] INIT HASHES (seed=%d): lora_A=%s lora_B=%s head=%s"
        % (arm, seed, loraA_hash, loraB_hash, head_hash))

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
    max_epochs = 1 if smoke else MAX_EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE
    curve = []
    t_arm = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        t0 = time.time()
        total = correct = 0
        loss_sum = 0.0
        for x, y, cell in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            vi = bridge_ref._preprocess_for_vit(x).to(torch.float32)
            logits, cls = model(vi)
            ce = crit(logits, y)
            ce.backward()
            opt.step()
            total += y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            loss_sum += float(ce.item()) * y.size(0)
        sched.step()
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        vy, vs = predict_val(model, bridge_ref, val_samples, cache, device)
        val_auc = auc_of(vy, vs)
        curve.append((epoch, train_loss, train_acc, val_auc))
        improved = val_auc > best_auc
        tag = " *BEST*" if improved else ""
        log("[arm %s epoch %02d/%d] loss=%.4f acc=%.4f val_auc=%.4f lr=%.2e (%.0fs)%s"
            % (arm, epoch, max_epochs, train_loss, train_acc, val_auc,
               opt.param_groups[0]["lr"], time.time() - t0, tag))
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if is_adapter_key(k)}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0 and not smoke:
                log("[arm %s early-stop] no val_auc improvement for %d epochs at epoch %d"
                    % (arm, PATIENCE, epoch))
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                      if is_adapter_key(k)}
    model.load_state_dict(best_state, strict=False)
    wall = time.time() - t_arm
    log("[arm %s done] best_val_auc=%.4f (epoch %d) wall=%.0fs" % (arm, best_auc, best_epoch, wall))

    suffix = "_smoke" if smoke else ""
    adapter_path = os.path.join(HERE, "lora_g19b_dfdcp_%s%s.pt" % (arm, suffix))
    torch.save(best_state, adapter_path)
    log("[arm %s save] %s" % (arm, adapter_path))

    return dict(model=model, best_val_auc=best_auc, best_epoch=best_epoch,
                n_trainable=n_train, n_lora=sum(p.numel() for p in lora_params),
                n_head=sum(p.numel() for p in head_params),
                lora_r=r, lora_alpha=alpha, seed=seed, sources=list(domains),
                loraA_hash=loraA_hash, loraB_hash=loraB_hash, head_hash=head_hash,
                wall=wall, curve=curve, adapter_path=adapter_path, samples=samples)


# ------------------------------------------------------------- probe ---------
def probe_protocol(feat, meta):
    """StandardScaler(fit probe-train) + LR(C=1e-3, lbfgs, max_iter=3000)."""
    tm = meta["train_mask"]
    tr = np.where(tm)[0]
    assert len(tr) == 2200 and bool(tm[:2200].all()) and not bool(tm[2200:].any())
    te = np.arange(2200, 3000)
    yb = (meta["y"] == 0).astype(np.int64)      # positive = fake
    sc = StandardScaler().fit(feat[tr])
    clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000, random_state=0)
    clf.fit(sc.transform(feat[tr]), yb[tr])

    res = {}
    s_te = clf.decision_function(sc.transform(feat[te]))
    res["in_domain"] = float(roc_auc_score(yb[te], s_te))
    dom = np.array([str(d) for d in meta["domain"]])
    for d in ALL_TARGET_DOMAINS:
        idx = np.where(dom == d)[0]
        s = clf.decision_function(sc.transform(feat[idx]))
        res[d] = float(roc_auc_score(yb[idx], s))
        res["n_" + d] = int(len(idx))
    res["cd_mean_all5"] = float(np.mean([res[d] for d in ALL_TARGET_DOMAINS]))
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


# ------------------------------------------------------------- oracle --------
def oracle_split(meta, domain, seed=SPLIT_SEED, frac=0.7):
    """Video-directory-identity, class-stratified 70/30 split of one target domain."""
    dom = np.array([str(d) for d in meta["domain"]])
    idx = np.where(dom == domain)[0]
    yb = (meta["y"] == 0).astype(np.int64)
    paths = meta["paths"]
    off = 1000 * ORACLE_DOMAINS.index(domain)
    rng = np.random.default_rng(seed + off)
    tr, te = [], []
    for c in (0, 1):
        dirs = sorted(set(os.path.dirname(str(paths[i])) for i in idx if yb[i] == c))
        perm = rng.permutation(len(dirs))
        n = int(round(len(dirs) * frac))
        n = min(max(n, 1), len(dirs) - 1)
        trc = set(dirs[j] for j in perm[:n])
        for i in idx:
            if int(yb[i]) != c:
                continue
            (tr if os.path.dirname(str(paths[i])) in trc else te).append(i)
    return np.array(sorted(tr)), np.array(sorted(te))


def oracle_auc(feat, meta, domain, seed=SPLIT_SEED):
    """Own-label probe on a video-disjoint 70% split of `domain`, tested on the 30%."""
    tr, te = oracle_split(meta, domain, seed=seed)
    yb = (meta["y"] == 0).astype(np.int64)
    assert len(np.unique(yb[tr])) == 2 and len(np.unique(yb[te])) == 2, \
        "oracle split for %s is not class-complete (tr=%d te=%d)" % (domain, len(tr), len(te))
    sc = StandardScaler().fit(feat[tr])
    clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000, random_state=0)
    clf.fit(sc.transform(feat[tr]), yb[tr])
    auc = float(roc_auc_score(yb[te], clf.decision_function(sc.transform(feat[te]))))
    return dict(auc=auc, n_tr=int(len(tr)), n_te=int(len(te)),
                n_tr_dirs=len(set(os.path.dirname(str(meta["paths"][i])) for i in tr)),
                n_te_dirs=len(set(os.path.dirname(str(meta["paths"][i])) for i in te)),
                n_tr_fake=int((yb[tr] == 1).sum()), n_te_fake=int((yb[te] == 1).sum()))


# ------------------------------------------------------------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARM_NAMES))
    ap.add_argument("--run-r0", action="store_true", default=True)
    ap.add_argument("--no-r0", dest="run_r0", action="store_false")
    ap.add_argument("--cap-ffpp", type=int, default=4000)
    ap.add_argument("--cap-target", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=64)
    ap.add_argument("--smoke-iters", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = query_gpu()
    log("[gpu] nvidia-smi query (index, mem_used MiB, util%%)=%s  [BEFORE]" % (gpus,))
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
        "num_workers=0 ; torch.set_num_threads=%d"
        % (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"),
           os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("NUMEXPR_NUM_THREADS"),
           os.environ.get("VECLIB_MAXIMUM_THREADS"), os.environ.get("BLIS_NUM_THREADS"),
           cv2.getNumThreads(), torch.get_num_threads()))

    # ---------------- 5300-row metadata + alignment ----------------
    meta = load_5300_meta()
    log("[align] 5300-row paths byte-equal to _g17/cnn_feats.npz: True (assert passed)")
    log("[align] probe rows 0..2999 (train 0..2199, test 2200..2999) ; multi rows 3000..5299")

    probe_keys, probe_rows = build_probe_folder_set(meta)
    log("[probe] probe folder directory-chain keys = %d (from all %d probe rows)"
        % (len(probe_keys), len(probe_rows)))

    ffpp_videos, n_missing, n_excluded = build_ffpp_videos(probe_keys)
    log("[ffpp] train.json videos: real=%d fake=%d total=%d (missing_dirs=%d "
        "excluded_by_probe_dirchain=%d)"
        % (sum(1 for v in ffpp_videos if v[1] == 1),
           sum(1 for v in ffpp_videos if v[1] == 0), len(ffpp_videos), n_missing, n_excluded))

    target_videos = build_target_videos(meta, ["cd2", "ffiw", "wild"])
    log("[target] candidate training-domain videos: %d (cd2=%d ffiw=%d wild=%d)"
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
        log("[leak] CONTAMINATION (expected, MULTI arms only) %-5s eval folders also in "
            "training pool: %d/%d" % (d, len(dd & train_folders), len(dd)))

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

    # ---------------- preload eval images ----------------
    n_all = 5300
    all_paths = [str(p) for p in meta["paths"]]
    cache = {}
    t0 = time.time()
    for p in all_paths:
        if ck(p) not in cache:
            cache[ck(p)] = read_rgb336(p)
    log("[preload] decoded all %d eval images in %.1fs" % (len(cache), time.time() - t0))

    # ---------------- R0 : frozen extraction (sanity gate) ----------------
    feat_R0 = None
    wall_R0 = float("nan")
    if args.run_r0:
        log("")
        log("=" * 90)
        log("ARM R0 -- frozen ViT feature extraction (pipeline sanity gate)")
        log("=" * 90)
        suffix = "_smoke" if args.smoke else ""
        f0_path = os.path.join(HERE, "feats_g19b_dfdcp_R0%s.npz" % suffix)

        bridge.vit.to(device).eval()
        outs = []
        t_r0 = time.time()
        n_ext = min(n_all, args.smoke_n) if args.smoke else n_all
        with torch.no_grad():
            for i in range(0, n_ext, BATCH):
                chunk = all_paths[i:i + BATCH]
                x = torch.stack([to_clip_tensor(cache[ck(p)]) for p in chunk]).to(device)
                vi = bridge.vit.forward_features(bridge._preprocess_for_vit(x).to(torch.float32))
                outs.append(vi[:, 0, :].float().cpu().numpy())
                if (i // BATCH) % 25 == 0:
                    log("  [extract R0] %d/%d wall=%.0fs" % (i + len(chunk), n_ext,
                                                             time.time() - t_r0))
        feat_R0 = np.concatenate(outs, axis=0).astype(np.float32)
        wall_R0 = time.time() - t_r0
        log("[R0] feat shape=%s wall=%.0fs (extracted %d rows)" % (feat_R0.shape, wall_R0, n_ext))
        np.savez(f0_path, feat=feat_R0,
                 paths=np.array(all_paths[:n_ext], dtype=object))
        if not args.smoke:
            assert feat_R0.shape == (5300, 768), feat_R0.shape
            pz = np.load(PROBE_NPZ, allow_pickle=True)
            mz = np.load(MULTI_NPZ, allow_pickle=True)
            d_probe = float(np.abs(feat_R0[:3000] - pz["V"].astype(np.float32)).max())
            d_multi = float(np.abs(feat_R0[3000:] - mz["V"].astype(np.float32)).max())
            log("[R0] maxabs deviation vs probe_feats.npz['V'] (3000 rows) = %.3e" % d_probe)
            log("[R0] maxabs deviation vs feats_multi.npz['V'] (2300 rows) = %.3e" % d_multi)

            r0 = probe_protocol(feat_R0, meta)
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
            log("[R0] saved %s" % f0_path)

    # ---------------- shared training pool (built ONCE, shared by ALL arms) --------
    # Deliberate G19b deviation: the early-stopping val split is drawn from FF++
    # videos ONLY, for every arm (see module docstring).
    rng_v = random.Random(SEED)
    ffpp_real = [v for v in ffpp_videos if v[1] == 1]
    ffpp_fake = [v for v in ffpp_videos if v[1] == 0]
    rng_v.shuffle(ffpp_real)
    rng_v.shuffle(ffpp_fake)
    n_val_real = max(1, int(round(len(ffpp_real) * VAL_FRAC)))
    n_val_fake = max(1, int(round(len(ffpp_fake) * VAL_FRAC)))
    val_videos = ffpp_real[:n_val_real] + ffpp_fake[:n_val_fake]
    train_ffpp = ffpp_real[n_val_real:] + ffpp_fake[n_val_fake:]
    val_folders = set(path_key(v[0]) for v in val_videos)
    assert not (val_folders & set(path_key(v[0]) for v in target_videos))
    log("")
    log("[split] FF++-ONLY, video-disjoint val for EVERY arm: val videos=%d "
        "(real %d / fake %d) ; FF++ train videos=%d ; held-out fold=%s ; "
        "NO dfdcp/cd1/probe video is used for training or selection"
        % (len(val_videos), n_val_real, n_val_fake, len(train_ffpp), HOLD_OUT))

    # raw per-cell frames (val videos excluded from FF++ training pool)
    cells_raw = {}
    for d in DOMAIN_ORDER:
        for lab in (1, 0):
            cells_raw[(d, lab)] = []
    for folder, label, vkey in train_ffpp:
        cap = REAL_FRAMES if label == 1 else FAKE_FRAMES
        cells_raw[("ffpp", label)].extend([(fp, label) for fp in sample_frames(folder, cap)])
    for folder, label, vkey in target_videos:
        d = vkey.split("_")[0]
        cells_raw[(d, label)].extend(
            [(fp, label) for fp in sample_frames(folder, TARGET_MAX_PER_VIDEO)])

    log("[data] raw per-(domain,class) frame counts: %s"
        % {("%s/%d" % k): len(v) for k, v in sorted(cells_raw.items())})
    if args.smoke:
        rng_s = random.Random(SEED)
        for k in cells_raw:
            arr = list(cells_raw[k])
            rng_s.shuffle(arr)
            cells_raw[k] = arr[:args.smoke_n]
        log("[smoke] per-cell n = %s" % {("%s/%d" % k): len(v)
                                         for k, v in sorted(cells_raw.items())})

    rng = random.Random(SEED)
    cells_all = {}
    for d in DOMAIN_ORDER:                      # fixed order -> shared across arms
        for lab in (1, 0):
            cap_k = args.cap_ffpp if d == "ffpp" else args.cap_target
            cells_all[(d, lab)] = cap_by_class(cells_raw[(d, lab)], cap_k, rng)
    log("[data] capped per-cell counts (ffpp cap=%d, target cap=%d): %s"
        % (args.cap_ffpp, args.cap_target,
           {("%s/%d" % k): len(v) for k, v in sorted(cells_all.items())}))
    log("[data] total shared training pool = %d frames"
        % sum(len(v) for v in cells_all.values()))

    # val frames (video-disjoint)
    val_samples = []
    for folder, label, vkey in val_videos:
        for fp in sample_frames(folder, VAL_MAX_PER_VIDEO):
            val_samples.append((fp, label, 0))
    vr = [s for s in val_samples if s[1] == 1]
    vf = [s for s in val_samples if s[1] == 0]
    nv = min(len(vr), len(vf))
    rng_v.shuffle(vr)
    rng_v.shuffle(vf)
    val_samples = vr[:nv] + vf[:nv]
    rng_v.shuffle(val_samples)
    log("[split] val frames=%d (real %d / fake %d), video-held-out"
        % (len(val_samples), sum(1 for s in val_samples if s[1] == 1),
           sum(1 for s in val_samples if s[1] == 0)))

    if args.dry_run:
        log("")
        log("[dry-run] data audit complete; stopping before model build.")
        log.close()
        return

    # ---- pre-decode ALL pooled training + val frames once to RAM ----
    all_pool_paths = set()
    for _k, _v in cells_all.items():
        for (pp, _ll) in _v:
            all_pool_paths.add(pp)
    for pp in all_pool_paths:
        if ck(pp) not in cache:
            cache[ck(pp)] = read_rgb336(pp)
    for (p, _l, _c) in val_samples:
        if ck(p) not in cache:
            cache[ck(p)] = read_rgb336(p)
    log("[preload] cache total = %d images (~%.2f GB uint8 336x336x3)"
        % (len(cache), len(cache) * IMG_SIZE * IMG_SIZE * 3 / 1e9))

    # ---------------- run arms ----------------
    arms = [a for a in args.arms.split(",") if a in ARM_NAMES]
    if args.smoke:
        arms = arms[:1]
        log("[smoke] single-arm 1-epoch smoke: %s" % arms)
    spec_by_name = {s["name"]: s for s in ARM_SPECS}
    results = {}
    feat_paths = {}
    for arm in arms:
        res = train_arm(spec_by_name[arm], meta, cells_all, val_samples, bridge, device,
                        vit_state, cache,
                        max_iters=(args.smoke_iters if args.smoke else MAX_ITERS_PER_EPOCH),
                        smoke=args.smoke)
        model = res.pop("model")
        n_ext = min(n_all, args.smoke_n) if args.smoke else n_all
        eval_samples = [(str(meta["paths"][i]), int(meta["y"][i])) for i in range(n_ext)]
        feat = extract_feats(model, bridge, eval_samples, cache, device,
                             tag=arm, progress=log)
        assert feat.shape == (n_ext, 768), feat.shape
        suffix = "_smoke" if args.smoke else ""
        fp = os.path.join(HERE, "feats_g19b_dfdcp_%s%s.npz" % (arm, suffix))
        np.savez(fp, feat=feat, paths=np.array(all_paths[:n_ext], dtype=object))
        log("[arm %s] feats saved %s shape=%s" % (arm, fp, feat.shape))
        results[arm] = res
        feat_paths[arm] = (fp, feat)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if args.smoke:
        log("")
        log("[smoke] DONE. arms run: %s" % ",".join(results.keys()))
        for a in results:
            log("[smoke] %s best_val_auc=%.4f wall=%.0fs trainable=%d"
                % (a, results[a]["best_val_auc"], results[a]["wall"],
                   results[a]["n_trainable"]))
        log("[wall] total = %.1f s" % (time.time() - t_start))
        log.close()
        return

    # ---------------- R0 features ----------------
    if args.run_r0:
        feat_paths["R0"] = (os.path.join(HERE, "feats_g19b_dfdcp_R0.npz"), feat_R0)

    # ---------------- probes / ratios / oracles ----------------
    log("")
    log("=" * 90)
    log("G19b RESULTS -- held-out fold = %s   (DECISIVE: O1multi vs O1ffonly)" % HOLD_OUT)
    log("=" * 90)

    probe_res = {}
    ratio_res = {}
    oracle_res = {}
    for arm in ALL_RUNS:
        if arm not in feat_paths:
            continue
        _fp, feat = feat_paths[arm]
        probe_res[arm] = probe_protocol(feat, meta)
        rr_per, rr, ss, cg = g12_ratio(feat, meta)
        ratio_res[arm] = dict(per=rr_per, mean=rr, s=ss, class_gap=cg)
        oracle_res[arm] = {}
        for d in ORACLE_DOMAINS:
            oracle_res[arm][d] = oracle_auc(feat, meta, d)

    hdr = "%-14s | %8s |" % ("run", "in-dom")
    for d in ALL_TARGET_DOMAINS:
        hdr += " %9s" % d
    hdr += " | %9s" % "mean5"
    log("")
    log("PRIMARY TABLE -- cross-domain AUC (probe: StandardScaler(probe-train)+LR C=1e-3)")
    log(hdr)
    log("-" * len(hdr))
    for arm in ALL_RUNS:
        if arm not in probe_res:
            continue
        r = probe_res[arm]
        line = "%-14s | %8.4f |" % (arm, r["in_domain"])
        for d in ALL_TARGET_DOMAINS:
            line += " %9.4f" % r[d]
        line += " | %9.4f" % r["cd_mean_all5"]
        log(line)
    log("-" * len(hdr))
    line = "%-14s | %8.4f |" % ("ANCHOR_G19a", ANCHOR_IN_DOMAIN)
    for d in ALL_TARGET_DOMAINS:
        line += " %9.4f" % ANCHOR_FROZEN[d]
    line += " | %9.4f" % ANCHOR_FROZEN_MEAN
    log(line + "   (prior frozen-ViT anchors, same protocol)")

    log("")
    log("CONTAMINATION FLAGS for fold=%s: MULTI training sources = %s ; FFONLY training "
        "source = %s ; held-out = %s" % (HOLD_OUT, MULTI_SOURCES, FFONLY_SOURCES, HOLD_OUT))
    log("  columns marked CONTAMINATED (excluded from aggregates): %s"
        % ", ".join(CONTAMINATED_COLUMNS))
    log("  VALID transfer column: %s" % VALID_TRANSFER_COLUMN)

    log("")
    log("IN-DOMAIN GUARD (FF++ probe-test 800 rows, train_mask False; frozen baseline %.4f)"
        % ANCHOR_IN_DOMAIN)
    for arm in ALL_RUNS:
        if arm in probe_res:
            log("  %-14s in-domain AUC=%.4f   (baseline %.4f, delta=%+.4f)"
                % (arm, probe_res[arm]["in_domain"], ANCHOR_IN_DOMAIN,
                   probe_res[arm]["in_domain"] - ANCHOR_IN_DOMAIN))

    log("")
    log("G12 DOMAIN-SENSITIVITY RATIO (s / class_gap printed for cross-check)")
    log("%-14s %10s %10s | %s | %10s" % ("run", "s", "class_gap",
                                         " ".join("%9s" % d for d in ALL_TARGET_DOMAINS),
                                         "mean"))
    for arm in ALL_RUNS:
        if arm not in ratio_res:
            continue
        rr = ratio_res[arm]
        log("%-14s %10.4f %10.4f | %s | %10.4f"
            % (arm, rr["s"], rr["class_gap"],
               " ".join("%9.4f" % rr["per"][d] for d in ALL_TARGET_DOMAINS), rr["mean"]))
    log("  (G17c frozen-ViT anchor: ratio_V=%.4f)" % ANCHOR_RATIO_V)

    # ---------------- oracle + alignment ----------------
    log("")
    log("ORACLE (own-label, video-dir-identity class-stratified 70/30 split INSIDE the "
        "target domain; StandardScaler+LR(C=1e-3) fit on the 70%%, tested on the 30%%)")
    log("  split seed=%d (per-domain offset +1000*idx) ; oracle AUC moves with the "
        "representation" % SPLIT_SEED)
    log("%-14s | %-22s | %-22s" % ("run", "dfdcp (oracle)", "wild (oracle)"))
    for arm in ALL_RUNS:
        if arm not in oracle_res:
            continue
        o = oracle_res[arm]
        log("%-14s | AUC=%.4f tr=%3d/te=%3d (dirs %3d/%3d, fake %3d/%3d) | "
            "AUC=%.4f tr=%3d/te=%3d (dirs %3d/%3d, fake %3d/%3d)"
            % (arm, o["dfdcp"]["auc"], o["dfdcp"]["n_tr"], o["dfdcp"]["n_te"],
               o["dfdcp"]["n_tr_dirs"], o["dfdcp"]["n_te_dirs"],
               o["dfdcp"]["n_tr_fake"], o["dfdcp"]["n_te_fake"],
               o["wild"]["auc"], o["wild"]["n_tr"], o["wild"]["n_te"],
               o["wild"]["n_tr_dirs"], o["wild"]["n_te_dirs"],
               o["wild"]["n_tr_fake"], o["wild"]["n_te_fake"]))

    log("")
    log("ALIGNMENT RATIO = (source-label AUC on domain) / (oracle AUC on domain)")
    log("  high = source-trained direction aligns with the target's own discriminative "
        "direction ; low = representation is discriminable but misaligned")
    log("%-14s | %9s %9s %9s | %9s %9s %9s" %
        ("run", "src_d", "orc_d", "ratio_d", "src_w", "orc_w", "ratio_w"))
    align = {}
    for arm in ALL_RUNS:
        if arm not in oracle_res:
            continue
        a = {}
        for d, short in (("dfdcp", "d"), ("wild", "w")):
            src = probe_res[arm][d]
            orc = oracle_res[arm][d]["auc"]
            a[d] = float(src / orc) if orc > 0 else float("nan")
        align[arm] = a
        log("%-14s | %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f"
            % (arm, probe_res[arm]["dfdcp"], oracle_res[arm]["dfdcp"]["auc"], a["dfdcp"],
               probe_res[arm]["wild"], oracle_res[arm]["wild"]["auc"], a["wild"]))

    # ---------------- per-seed summary + gates ----------------
    def dcol(a):
        return probe_res[a][HOLD_OUT] if a in probe_res else float("nan")

    m_vals = [dcol("O1multi_s%d" % s) for s in (0, 1, 2)]
    f_vals = [dcol("O1ffonly_s%d" % s) for s in (0, 1, 2)]
    r0v = dcol("R0")
    r2v = dcol("R2multi_s0")
    m_mean, m_std = float(np.mean(m_vals)), float(np.std(m_vals, ddof=0))
    f_mean, f_std = float(np.mean(f_vals)), float(np.std(f_vals, ddof=0))

    log("")
    log("PER-SEED SUMMARY on the VALID transfer column (%s)" % HOLD_OUT)
    log("  R0 (frozen)                = %.4f   (G19a R0 = %.4f)" % (r0v, G19A_R0_DFDCP))
    for s in (0, 1, 2):
        log("  O1multi_s%d                 = %.4f   (delta vs R0 = %+.4f)"
            % (s, m_vals[s], m_vals[s] - r0v))
    log("  O1multi   mean+-std        = %.4f +- %.4f" % (m_mean, m_std))
    for s in (0, 1, 2):
        log("  O1ffonly_s%d                = %.4f   (delta vs R0 = %+.4f)"
            % (s, f_vals[s], f_vals[s] - r0v))
    log("  O1ffonly  mean+-std        = %.4f +- %.4f" % (f_mean, f_std))
    log("  R2multi_s0 (r=32, alpha=64)= %.4f   (delta vs R0 = %+.4f)" % (r2v, r2v - r0v))
    log("")
    log("PAIRED COMPARISON  O1multi - O1ffonly (same seed):")
    for s in (0, 1, 2):
        log("  seed %d : %.4f - %.4f = %+.4f" % (s, m_vals[s], f_vals[s], m_vals[s] - f_vals[s]))
    log("  MEAN   : %.4f - %.4f = %+.4f" % (m_mean, f_mean, m_mean - f_mean))
    log("  (G19a reference: O1 seed 20260910 = %.4f, different seed + multi-domain val)"
        % G19A_O1_DFDCP)

    trained_runs = [a for a in ARM_NAMES if a in probe_res]
    best_arm = max(trained_runs, key=lambda a: dcol(a)) if trained_runs else None
    best_val = dcol(best_arm) if best_arm else float("nan")
    log("")
    log("BEST TRAINED ARM by %s = %s (%.4f)" % (HOLD_OUT, best_arm, best_val))

    gates = {}
    # G19B_UNFREEZE_REPLICATES
    cond_a = (m_mean - r0v) >= 0.01
    cond_b = all(v > r0v for v in m_vals)
    gates["G19B_UNFREEZE_REPLICATES"] = (m_mean - r0v, bool(cond_a and cond_b))
    log("  [detail] O1multi all 3 seeds > R0: %s (deltas %s)"
        % (cond_b, ["%+.4f" % (v - r0v) for v in m_vals]))
    # G19B_NEEDS_MULTISOURCE (decisive)
    c1 = (f_mean - r0v) <= 0.01
    c2 = (m_mean - f_mean) >= 0.05
    gates["G19B_NEEDS_MULTISOURCE"] = (m_mean - f_mean, bool(c1 and c2))
    log("  [detail] O1ffonly mean - R0 = %+.4f (<= +0.0100 -> %s) ; "
        "O1multi mean - O1ffonly mean = %+.4f (>= +0.0500 -> %s)"
        % (f_mean - r0v, c1, m_mean - f_mean, c2))
    # G19B_VS_CAPACITY
    gates["G19B_VS_CAPACITY"] = (m_mean - r2v, bool((m_mean - r2v) >= 0.01))
    # G19B_NO_FORGETTING (+ per arm)
    if best_arm is not None:
        ind = probe_res[best_arm]["in_domain"]
        gates["G19B_NO_FORGETTING"] = (ind, bool(ind >= 0.9752))
    log("  [no-forgetting detail] threshold 0.9752 (frozen in-domain %.4f - 0.01)"
        % ANCHOR_IN_DOMAIN)
    for a in trained_runs:
        log("    arm %-14s in-domain=%.4f -> %s"
            % (a, probe_res[a]["in_domain"],
               "OK" if probe_res[a]["in_domain"] >= 0.9752 else "FORGETTING"))
    # G19B_ALIGNMENT_SHIFT
    if best_arm is not None and "R0" in align:
        dlt = align[best_arm][HOLD_OUT] - align["R0"][HOLD_OUT]
        gates["G19B_ALIGNMENT_SHIFT"] = (dlt, bool(dlt > 0))
        log("  [detail] alignment ratio on dfdcp: best arm %s = %.4f ; R0 = %.4f ; "
            "shift = %+.4f" % (best_arm, align[best_arm][HOLD_OUT], align["R0"][HOLD_OUT], dlt))

    log("")
    log("GATE VERDICTS (mechanical)")
    for k in ["G19B_UNFREEZE_REPLICATES", "G19B_NEEDS_MULTISOURCE", "G19B_VS_CAPACITY",
              "G19B_NO_FORGETTING", "G19B_ALIGNMENT_SHIFT"]:
        if k in gates:
            v, ok = gates[k]
            log("  %-28s value=%+.4f  ->  %s" % (k, v, "PASS" if ok else "FAIL"))
        else:
            log("  %-28s NOT EVALUATED" % k)

    log("")
    log("WALL / RESOURCE")
    log("  R0 frozen extraction wall=%.0fs" % wall_R0)
    for arm in ARM_NAMES:
        if arm in results:
            rres = results[arm]
            log("  arm %-14s wall=%.0fs best_val_auc=%.4f best_epoch=%d trainable=%d "
                "(lora=%d head=%d) r=%d alpha=%d seed=%d sources=%s"
                % (arm, rres["wall"], rres["best_val_auc"], rres["best_epoch"],
                   rres["n_trainable"], rres["n_lora"], rres["n_head"],
                   rres["lora_r"], rres["lora_alpha"], rres["seed"],
                   "+".join(rres["sources"])))
    log("  init hashes (proves per-seed LoRA init):")
    for arm in ARM_NAMES:
        if arm in results:
            rres = results[arm]
            log("    %-14s lora_A=%s lora_B=%s head=%s"
                % (arm, rres["loraA_hash"], rres["loraB_hash"], rres["head_hash"]))
    log("  threads: torch=%d OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s ; cv2=%d ; num_workers=0"
        % (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"),
           os.environ.get("MKL_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"),
           os.environ.get("NUMEXPR_NUM_THREADS"), cv2.getNumThreads()))
    log("[wall] total = %.1f s (%.2f h)" % (time.time() - t_start, (time.time() - t_start) / 3600.0))

    # ---------------- stats npz ----------------
    stats = {}
    for arm in ALL_RUNS:
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
        for d in ORACLE_DOMAINS:
            stats["oracle_%s_%s" % (arm, d)] = np.float64(oracle_res[arm][d]["auc"])
            stats["oracle_n_tr_%s_%s" % (arm, d)] = np.int64(oracle_res[arm][d]["n_tr"])
            stats["oracle_n_te_%s_%s" % (arm, d)] = np.int64(oracle_res[arm][d]["n_te"])
            stats["align_%s_%s" % (arm, d)] = np.float64(align[arm][d])
        if arm in results:
            rres = results[arm]
            stats["wall_%s" % arm] = np.float64(rres["wall"])
            stats["trainable_%s" % arm] = np.int64(rres["n_trainable"])
            stats["lora_trainable_%s" % arm] = np.int64(rres["n_lora"])
            stats["head_trainable_%s" % arm] = np.int64(rres["n_head"])
            stats["best_val_auc_%s" % arm] = np.float64(rres["best_val_auc"])
            stats["best_epoch_%s" % arm] = np.int64(rres["best_epoch"])
            stats["lora_r_%s" % arm] = np.int64(rres["lora_r"])
            stats["lora_alpha_%s" % arm] = np.int64(rres["lora_alpha"])
            stats["seed_%s" % arm] = np.int64(rres["seed"])
            stats["sources_%s" % arm] = np.array("+".join(rres["sources"]), dtype=object)
    for s in (0, 1, 2):
        stats["dfdcp_O1multi_s%d" % s] = np.float64(m_vals[s])
        stats["dfdcp_O1ffonly_s%d" % s] = np.float64(f_vals[s])
        stats["dfdcp_pairdiff_s%d" % s] = np.float64(m_vals[s] - f_vals[s])
    stats["dfdcp_O1multi_mean"] = np.float64(m_mean)
    stats["dfdcp_O1multi_std"] = np.float64(m_std)
    stats["dfdcp_O1ffonly_mean"] = np.float64(f_mean)
    stats["dfdcp_O1ffonly_std"] = np.float64(f_std)
    stats["dfdcp_R0"] = np.float64(r0v)
    stats["dfdcp_R2multi_s0"] = np.float64(r2v)
    stats["dfdcp_pairdiff_mean"] = np.float64(m_mean - f_mean)
    for k, (v, ok) in gates.items():
        stats["gate_%s" % k] = np.float64(v)
        stats["gate_%s_pass" % k] = np.bool_(ok)
    stats["holdout_auc_best_arm"] = np.float64(best_val)
    stats["best_arm_name"] = np.array(str(best_arm), dtype=object)
    stats["window_val_videos"] = np.int64(len(val_videos))
    stats["window_val_frames"] = np.int64(len(val_samples))
    stats["wall_R0"] = np.float64(wall_R0)
    np.savez(STATS_PATH, **stats)
    log("[save] stats -> %s" % STATS_PATH)

    # ---------------- final nvidia-smi ----------------
    gpus_end = query_gpu()
    log("[gpu] nvidia-smi query (index, mem_used MiB, util%%)=%s  [AFTER]" % (gpus_end,))
    log("[save] report -> %s" % REPORT_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log.buf) + "\n")
    log.close()


if __name__ == "__main__":
    main()
