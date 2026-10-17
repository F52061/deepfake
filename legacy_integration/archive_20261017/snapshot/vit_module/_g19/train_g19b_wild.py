# -*- coding: utf-8 -*-
"""
G19b -- DECISIVE SOURCE-SIMILARITY ABLATION for cross-domain deepfake transfer.
HELD-OUT FOLD = **wild**  (sibling agent owns dfdcp on GPU 1).
Follow-up to G19a (vit_module/_g19/train_g19a_wild.py), whose code is REUSED here
(LoRA r=16/alpha=32 on vit.blocks[*].attn.qkv/.proj, LODO enumeration + leakage
audits from _g18/train_g18b.py, balanced (domain,class) sampling, deployment-
identical preprocessing, the 5300-image extraction, the standard probe).

G19a (wild fold) gave:  R0 frozen = 0.8090 ; O1 (CE, MULTI sources) = 0.8384 ;
R1 (shuffled labels) = 0.7130 ; O3a/O3b (domain-invariance) = 0.8090/0.7797
-> the invariance objectives FAILED and are DROPPED here.

Science question
----------------
Is G19a's CE gain on the held-out domain caused by *unfreezing per se*, or merely
by *having training sources distributionally close to the target*?  Only an
FF++-only arm can separate the two.  `wild` is a FAR domain (G19a moved it by
only +0.029, vs +0.106 on the dfdcp fold), so if the gain is source-similarity
driven the FFONLY arm should collapse back toward the frozen anchor here.

Runs (7 trainings + the R0 frozen anchor)
-----------------------------------------
  O1multi_s0/s1/s2   CE, MULTI sources (FF++ train.json + cd2 + dfdcp + ffiw)
  O1ffonly_s0/s1/s2  CE, FFONLY source  (FF++ train.json only)
  R2multi_s0         CE, MULTI, LoRA rank DOUBLED to r=32 / alpha=64

All runs: held-out domain `wild` never touched (no target stats / TTA / pseudo-
labels / BN updates); `cd1` excluded ALWAYS (its 49 folder names are 100% contained
in cd2 -> leakage); every FF++ video folder used by probe_feats.npz (probe-train AND
probe-test) hard-excluded from training by PATH-DERIVED DIRECTORY-CHAIN identity
(never bare video-name strings: FF++ `161` and WildDeepfake `161` are different
videos).  Model selection = early stop on TRAINING-DOMAIN val AUC only.

Seeds change the LoRA init, the head init and the balanced-sampler data order.
The frame pool, the video train/val split and the val sample list are held FIXED
(seed SEED) across seeds so that seed variance is not confounded with data variance.

Source-similarity control: the FF++ portion of the pool is split ONCE (seed SEED)
and is therefore IDENTICAL between the MULTI and FFONLY arms -- the only difference
between them is the presence of the cd2/dfdcp/ffiw videos.

NEW in G19b -- oracle / alignment ratio
---------------------------------------
The ORACLE moves with the representation, so it is recomputed per trained arm.
For domains `wild` and `dfdcp`: that domain's 300 rows are split 70/30 BY VIDEO
DIRECTORY IDENTITY (stratified by class, fixed seed), a StandardScaler +
LogisticRegression(C=1e-3) is trained on the 70% split with that domain's OWN
labels and tested on the 30% split.  Then
    alignment_ratio(d) = (source-label probe AUC on d) / (oracle AUC on d)
High = the source-trained direction aligns with the target's own discriminative
direction ; low = the representation is discriminable but misaligned.

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
ALL_TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
CONTAMINATED = ["cd1", "cd2", "dfdcp", "ffiw"]
TARGET_SOURCES = ["cd2", "dfdcp", "ffiw"]             # cd1 excluded ALWAYS

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

ORACLE_DOMAINS = ["wild", "dfdcp"]
ORACLE_FRAC = 0.70
ORACLE_SEED = SEED

# -------------------------------------------------------------- run table -----
# objective is plain CE for every run (the G19a invariance arms O3a/O3b failed).
RUNS = [
    dict(name="O1multi_s0",  sources="multi",  seed=0, r=16, alpha=32),
    dict(name="O1multi_s1",  sources="multi",  seed=1, r=16, alpha=32),
    dict(name="O1multi_s2",  sources="multi",  seed=2, r=16, alpha=32),
    dict(name="O1ffonly_s0", sources="ffonly", seed=0, r=16, alpha=32),
    dict(name="O1ffonly_s1", sources="ffonly", seed=1, r=16, alpha=32),
    dict(name="O1ffonly_s2", sources="ffonly", seed=2, r=16, alpha=32),
    dict(name="R2multi_s0",  sources="multi",  seed=0, r=32, alpha=64),
]
RUN_NAMES = [r["name"] for r in RUNS]

# anchors (prior experiments, identical probe protocol)
ANCHOR_FROZEN = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261,
                 "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_INDOM = 0.9852
ANCHOR_SRCLR_MEAN = 0.8318
ANCHOR_ORACLE = 0.9111
GATE_R0_TOL = 0.005

OUT_DIR = HERE
LOG_PATH = os.path.join(OUT_DIR, "run_log_g19b_wild.txt")
REPORT_PATH = os.path.join(OUT_DIR, "g19b_wild_report.txt")
STATS_PATH = os.path.join(OUT_DIR, "g19b_wild_stats.npz")


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


class _NullLogger:
    """Stand-in used inside spawned DataLoader worker processes."""

    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        return None

    def close(self):
        return None


# NOTE: on Windows the DataLoader uses the `spawn` start method, which RE-IMPORTS this
# module inside every worker process.  A module-level `Logger(LOG_PATH)` would then be
# constructed in each worker and would re-open the run log with "w", TRUNCATING it from
# under the parent.  Only the real main process writes the log.
log = Logger(LOG_PATH) if __name__ == "__main__" else _NullLogger()
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


def fmt_tab(rows):
    widths = [max(len(str(r[i])) for r in rows) + 2 for i in range(len(rows[0]))]
    return "\n".join("  " + "".join(("%-" + str(w) + "s") % str(c) for c, w in zip(r, widths))
                     for r in rows)


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
                n_probe=n_p, aligned=aligned, g17_paths=gpaths)


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


def split_videos(videos, seed, frac=VAL_FRAC):
    """Video-level train/val split with a fixed seed (verbatim G19a logic)."""
    real = [v for v in videos if v[1] == 1]
    fake = [v for v in videos if v[1] == 0]
    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(fake)
    nvr = max(1, int(round(len(real) * frac)))
    nvf = max(1, int(round(len(fake) * frac)))
    val = real[:nvr] + fake[:nvf]
    tr = real[nvr:] + fake[nvf:]
    return tr, val


def build_val_pool(val_videos, seed):
    """Balanced-per-(domain,class)-cell val sample list, capped (verbatim G19a)."""
    rng_v = random.Random(seed)
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
    return val_paths, val_y, val_dom, by_cell


def build_pool(train_videos, seed):
    """Frame sampling + real/fake balancing over the TRAIN video list (verbatim G19a)."""
    rng_s = random.Random(seed)
    pool = []
    for folder, label, vkey, dom in train_videos:
        cap = (REAL_FRAMES if label == 1 else FAKE_FRAMES) if dom == "ffpp" \
            else TARGET_MAX_PER_VIDEO
        for fp in sample_frames(folder, cap):
            pool.append((fp, label, dom))
    n_before = len(pool)
    pool = balance_by_class3(pool, rng=rng_s)
    return pool, n_before


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


def attach_lora(vit, r=LORA_R, alpha=LORA_ALPHA):
    from peft import LoraConfig, get_peft_model
    targets = ["blocks.%d.attn.qkv" % i for i in range(len(vit.blocks))] + \
              ["blocks.%d.attn.proj" % i for i in range(len(vit.blocks))]
    for _n, p in vit.named_parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=LORA_DROPOUT,
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
        if (not quiet) and i % 50 == 0:
            log("    [extract %s] %d/%d wall=%.0fs"
                % (tag, min((i + 1) * batch, len(dl.dataset)), len(dl.dataset), time.time() - t0))
    feats = np.concatenate(out, axis=0).astype(np.float32)
    assert feats.shape[1] == 768, feats.shape
    return feats


# ---------------------------------------------------------------- sampling ---
def build_cells(samples):
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


def train_run(run, pool, val_paths, val_y, val_dom, val_domains, device, vit_sd,
              steps_per_epoch=STEPS_PER_EPOCH, max_epochs=MAX_EPOCHS,
              patience=PATIENCE, nworkers=2, log_steps=0):
    name = run["name"]
    """CE training of one G19b run; early stop on TRAINING-DOMAIN val AUC only."""
    log("")
    log("=" * 92)
    log("RUN %s  (held-out=%s | sources=%s | seed=%d | r=%d alpha=%d)"
        % (name, HELD_OUT, run["sources"], run["seed"], run["r"], run["alpha"]))
    log("=" * 92)
    t0 = time.time()

    labels = np.array([s[1] for s in pool], dtype=np.int64)
    dom_arr = np.array([s[2] for s in pool], dtype=object)
    paths_all = [s[0] for s in pool]

    cells, names = build_cells([(paths_all[i], int(labels[i]), dom_arr[i])
                                for i in range(len(pool))])
    log("[pool] n=%d ; cells: %s"
        % (len(pool), ", ".join("%s/%s=%d" % (d, "real" if c == 1 else "fake", len(cells[(d, c)]))
                                for (d, c) in names)))

    # ---- seed everything that the seed is supposed to move ----
    run_seed = int(run["seed"])
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)
    log("[seed] run_seed=%d -> torch/numpy/random manual_seed + sampler order "
        "(LoRA init, head init, data order)" % run_seed)

    vit = make_vit(vit_sd, device)
    lm, targets, n_lora = attach_lora(vit, r=run["r"], alpha=run["alpha"])
    head = nn.Linear(768, 2).to(device)
    log("[lora] target modules=%d (blocks[*].attn.qkv + blocks[*].attn.proj) r=%d alpha=%d dropout=%.1f"
        % (len(targets), run["r"], run["alpha"], LORA_DROPOUT))
    log("[lora] LoRA trainable params=%d (%.4fM) ; head trainable params=%d"
        % (n_lora, n_lora / 1e6, sum(p.numel() for p in head.parameters())))

    opt = torch.optim.AdamW(
        [{"params": [p for p in lm.parameters() if p.requires_grad], "lr": LORA_LR},
         {"params": head.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps_per_epoch)
    crit = nn.CrossEntropyLoss()

    sampler = BalancedCellSampler(cells, names, batch=BATCH, steps=steps_per_epoch,
                                  seed=run_seed)
    log("[obj] lambda=0 -> objective = CE ; balanced sampler: %d cells x %d = batch %d, "
        "%d steps/epoch" % (len(names), sampler.per, sampler.bs, steps_per_epoch))
    dl = DataLoader(PoolDataset(paths_all), batch_sampler=sampler, num_workers=nworkers,
                    pin_memory=True, collate_fn=_collate_pool, worker_init_fn=_worker_init)

    best_auc, best_epoch = -1.0, -1
    best_lora = best_head = None
    patience_left = patience
    curve = []

    for epoch in range(1, max_epochs + 1):
        lm.train()
        head.train()
        te0 = time.time()
        tot = corr = 0
        loss_sum = ce_sum = 0.0
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
            loss = ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log_steps and (step % log_steps == 0):
                log("    [%s epoch %d step %d/%d] ce=%.4f lr_lora=%.2e"
                    % (name, epoch, step, steps_per_epoch, float(ce.item()),
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
        for d in val_domains:
            m = vdom == d
            if m.sum() > 0 and len(np.unique(vy[m])) > 1:
                per_dom[d] = float(roc_auc_score(vy[m], vf[m]))
        curve.append(dict(epoch=epoch, loss=train_loss, ce=train_ce, acc=train_acc,
                          val_auc=val_auc, per_dom=per_dom))
        improved = (not np.isnan(val_auc)) and val_auc > best_auc
        log("[%s epoch %02d/%d] loss=%.4f ce=%.4f acc=%.4f val_auc=%.4f | %s%s (%.0fs)"
            % (name, epoch, max_epochs, train_loss, train_ce, train_acc, val_auc,
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
                    % (name, patience, epoch))
                break

    wall = time.time() - t0
    log("[%s done] best val_auc=%.4f at epoch %d ; wall=%.1fs" % (name, best_auc, best_epoch, wall))

    named = dict(lm.named_parameters())
    for k, v in best_lora.items():
        named[k].data.copy_(v.to(device))
    head.load_state_dict({k: v.to(device) for k, v in best_head.items()})

    save_path = os.path.join(OUT_DIR, "lora_g19b_wild_%s.pt" % name)
    torch.save({"run": name, "held_out": HELD_OUT, "lora": best_lora, "head": best_head,
                "targets": targets, "r": run["r"], "alpha": run["alpha"],
                "seed": run_seed, "sources": run["sources"],
                "best_val_auc": best_auc, "best_epoch": best_epoch, "n_lora": n_lora,
                "wall_s": wall, "curve": curve}, save_path)
    log("[%s save] %s" % (name, save_path))
    return dict(lm=lm, head=head, best_val_auc=best_auc, best_epoch=best_epoch,
                n_lora=n_lora, wall=wall, curve=curve, save_path=save_path)


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


def oracle_split(ev, dom, frac=ORACLE_FRAC, seed=ORACLE_SEED):
    """Split one target domain's rows 70/30 BY VIDEO DIRECTORY IDENTITY, stratified
    by class, with a fixed seed.  Returns (train_idx, test_idx)."""
    idx = np.where(ev["domain"] == dom)[0]
    groups = defaultdict(list)
    for i in idx:
        groups[vid_id_of_image(ev["paths"][i])].append(int(i))
    for k in groups:
        labs = set(int(ev["y"][j]) for j in groups[k])
        assert len(labs) == 1, "video folder with mixed labels: %s" % k
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for lab in (0, 1):
        keys = sorted(k for k in groups if int(ev["y"][groups[k][0]]) == lab)
        assert len(keys) >= 4, "domain %s class %d has only %d videos" % (dom, lab, len(keys))
        keys = np.array(keys, dtype=object)
        rng.shuffle(keys)
        n_tr = int(round(len(keys) * frac))
        n_tr = min(max(1, n_tr), len(keys) - 1)
        for k in keys[:n_tr]:
            tr.extend(groups[k])
        for k in keys[n_tr:]:
            te.extend(groups[k])
    return np.array(sorted(tr)), np.array(sorted(te))


def oracle_eval(feat, ev, dom):
    """Train StandardScaler+LR(C=1e-3) on the 70% VIDEO split of `dom` with that
    domain's OWN labels; AUC on the held-out 30% video split."""
    tr, te = oracle_split(ev, dom)
    sc = StandardScaler().fit(feat[tr].astype(np.float64))
    clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000)
    clf.fit(sc.transform(feat[tr].astype(np.float64)), ev["y"][tr])
    auc = float(roc_auc_score(ev["y"][te],
                              clf.decision_function(sc.transform(feat[te].astype(np.float64)))))
    return dict(auc=auc, n_train=int(len(tr)), n_test=int(len(te)),
                n_vid_train=int(len(set(vid_id_of_image(ev["paths"][i]) for i in tr))),
                n_vid_test=int(len(set(vid_id_of_image(ev["paths"][i]) for i in te))))


def oracle_bundle(feat, ev, tag):
    """Per-arm oracle AUCs + alignment ratios for ORACLE_DOMAINS."""
    out = {}
    for d in ORACLE_DOMAINS:
        o = oracle_eval(feat, ev, d)
        out[d] = o
    return out


def smoke_selection(ev, n=80):
    """80 probe-train + 80 probe-test rows, class-balanced (sklearn needs 2 classes)."""
    idx_tr = np.where(ev["train_mask"])[0]
    idx_te = np.where(~ev["train_mask"])[0]
    a = np.concatenate([idx_tr[ev["y"][idx_tr] == 1][:n], idx_tr[ev["y"][idx_tr] == 0][:n]])
    b = np.concatenate([idx_te[ev["y"][idx_te] == 1][:n], idx_te[ev["y"][idx_te] == 0][:n]])
    return np.concatenate([a, b])


# ---------------------------------------------------------------------- main --
def _mean_std(vals):
    v = np.asarray(vals, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUN_NAMES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    smoke = args.smoke
    sel_runs = [r for r in RUNS if r["name"] in run_names]

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
    _chk = sorted(probe_dirchain)[:200]
    assert all(vid_id_of_folder(c) == c for c in _chk), "directory-identity accessors disagree"
    log("[probe] identity positive-control: vid_id_of_folder(x)==x for 200/%d probe chains -> OK"
        % len(probe_dirchain))
    ffpp_videos, ffpp_missing, ffpp_excluded = build_ffpp_videos(probe_dirchain)
    log("[ffpp] train videos: real=%d fake=%d total=%d missing=%d excluded_by_probe_dirchain=%d"
        % (sum(1 for v in ffpp_videos if v[1] == 1), sum(1 for v in ffpp_videos if v[1] == 0),
           len(ffpp_videos), ffpp_missing, ffpp_excluded))

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

    target_videos = build_target_videos(TARGET_SOURCES)
    for d in TARGET_SOURCES:
        vv = [v for v in target_videos if v[3] == d]
        log("[target] %-5s training videos=%d (real %d / fake %d)"
            % (d, len(vv), sum(1 for v in vv if v[1] == 1), sum(1 for v in vv if v[1] == 0)))

    # ---------------- FF++ train/val split: computed ONCE and SHARED ------------
    # so that the only difference between the MULTI and FFONLY arms is the presence
    # of the cd2/dfdcp/ffiw videos.
    ffpp_train, ffpp_val = split_videos(ffpp_videos, SEED)
    tg_train, tg_val = split_videos(target_videos, SEED + 13)
    log("[split] FF++ only  : train videos=%d (real %d / fake %d) ; val videos=%d (real %d / fake %d)"
        % (len(ffpp_train), sum(1 for v in ffpp_train if v[1] == 1),
           sum(1 for v in ffpp_train if v[1] == 0), len(ffpp_val),
           sum(1 for v in ffpp_val if v[1] == 1), sum(1 for v in ffpp_val if v[1] == 0)))
    log("[split] targets    : train videos=%d (real %d / fake %d) ; val videos=%d (real %d / fake %d)"
        % (len(tg_train), sum(1 for v in tg_train if v[1] == 1),
           sum(1 for v in tg_train if v[1] == 0), len(tg_val),
           sum(1 for v in tg_val if v[1] == 1), sum(1 for v in tg_val if v[1] == 0)))
    log("[split] the FF++ portion is IDENTICAL between MULTI and FFONLY (split computed once, "
        "seed=%d) ; NO frame of `%s` is present in either" % (SEED, HELD_OUT))

    SOURCE_SETS = {
        "ffonly": dict(train=ffpp_train, val=ffpp_val, val_domains=["ffpp"]),
        "multi": dict(train=ffpp_train + tg_train, val=ffpp_val + tg_val,
                      val_domains=["ffpp"] + TARGET_SOURCES),
    }

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

    # ---------------- pools + val sets ----------------
    PACK = {}
    for skey, s in SOURCE_SETS.items():
        pool, n_before = build_pool(s["train"], SEED)
        vp, vy, vd, by_cell = build_val_pool(s["val"], SEED + 1)
        bad_ffpp = [p for (p, _l, d) in pool if d == "ffpp" and vid_id_of_image(p) in probe_all_chains]
        assert len(bad_ffpp) == 0, "FF++ training frame from the probe/eval set: %d" % len(bad_ffpp)
        log("")
        log("[sources=%s] train videos=%d -> frames before balance=%d after=%d ; "
            "FF++ training frames from the 5300 eval set: %d (must be 0)"
            % (skey, len(s["train"]), n_before, len(pool), len(bad_ffpp)))
        log("[sources=%s] val samples=%d over %d cells: %s"
            % (skey, len(vp), len(by_cell),
               ", ".join("%s/%s=%d" % (k[0], "real" if k[1] == 1 else "fake",
                                       sum(1 for d in vd if d == k[0]))
                         for k in sorted(by_cell.keys()))))
        PACK[skey] = dict(pool=pool, val_paths=vp, val_y=vy, val_dom=vd,
                          val_domains=s["val_domains"], n_cells=len(by_cell))

    smoke_steps, smoke_epochs = 20, 1
    if smoke:
        log("")
        log("### SMOKE MODE: tiny pool (<=40/cell), 20 steps, 1 epoch ###")
        for skey in PACK:
            p = PACK[skey]["pool"]
            rng_sm = random.Random(SEED + 5)
            by_cell_sm = defaultdict(list)
            for s in p:
                by_cell_sm[(s[2], s[1])].append(s)
            newp = []
            for k in sorted(by_cell_sm.keys()):
                v = list(by_cell_sm[k])
                rng_sm.shuffle(v)
                newp.extend(v[:40])
            PACK[skey]["pool"] = newp
            # val: take up to 32 per (domain,class) cell so both classes survive
            rng_sv = random.Random(SEED + 6)
            vc = defaultdict(list)
            for j in range(len(PACK[skey]["val_paths"])):
                vc[(PACK[skey]["val_dom"][j], PACK[skey]["val_y"][j])].append(j)
            keep = []
            for k in sorted(vc.keys()):
                v = list(vc[k])
                rng_sv.shuffle(v)
                keep.extend(v[:32])
            keep.sort()
            PACK[skey]["val_paths"] = [PACK[skey]["val_paths"][j] for j in keep]
            PACK[skey]["val_y"] = [PACK[skey]["val_y"][j] for j in keep]
            PACK[skey]["val_dom"] = [PACK[skey]["val_dom"][j] for j in keep]
            log("[smoke %s] pool=%d val=%d (%s)" % (skey, len(newp), len(keep),
                ", ".join("%s/%s=%d" % (k[0], k[1], len(vc[k])) for k in sorted(vc.keys()))))
        sel_runs = [r for r in RUNS if r["name"] in
                    ("O1multi_s0", "O1ffonly_s0", "R2multi_s0")]
        log("[smoke] runs=%s" % [r["name"] for r in sel_runs])

    # ---------------- oracle split (reported, deterministic) ----------------
    for d in ORACLE_DOMAINS:
        tr, te = oracle_split(ev, d)
        log("[oracle split %s] rows train=%d test=%d ; videos train=%d test=%d (70/30 by video "
            "directory identity, stratified by class, seed=%d)"
            % (d, len(tr), len(te),
               len(set(vid_id_of_image(ev["paths"][i]) for i in tr)),
               len(set(vid_id_of_image(ev["paths"][i]) for i in te)), ORACLE_SEED))
        assert len(set(tr) & set(te)) == 0

    vit_sd = load_vit_state()
    log("[ckpt] ViT state keys=%d (91.4M params, num_classes=0)" % len(vit_sd))

    results, feats_store = {}, {}
    path_ok = True

    def save_feats(tag, f):
        nonlocal path_ok
        p = os.path.join(OUT_DIR, "feats_g19b_wild_%s.npz" % tag)
        np.savez(p, feat=f.astype(np.float32), paths=ev["paths"])
        chk = np.load(p, allow_pickle=True)
        eq = bool(np.array_equal(np.array([str(s) for s in chk["paths"]], dtype=object),
                                 ev["g17_paths"]))
        assert chk["feat"].shape == (5300, 768) and chk["feat"].dtype == np.float32
        assert eq, "saved paths are NOT byte-equal to _g17/cnn_feats.npz paths"
        log("[%s save] %s (feat %s %s ; paths byte-equal to _g17/cnn_feats.npz = %s)"
            % (tag, p, chk["feat"].shape, chk["feat"].dtype, eq))
        return p

    # ---------------- R0 : frozen features + pipeline sanity gate ----------------
    log("")
    log("=" * 92)
    log("RUN R0 -- frozen ViT (no training)")
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
        # exercise the oracle code path on the frozen features of the target rows
        tgt = np.where(np.isin(ev["domain"], ORACLE_DOMAINS))[0]
        fsub = extract_feats(vit, device, ev["paths"][tgt], batch=32, nworkers=0,
                             tag="R0-oracle-smoke", quiet=True)
        ev_t = dict(paths=np.array([ev["paths"][i] for i in tgt], dtype=object),
                    y=ev["y"][tgt], domain=ev["domain"][tgt])
        for d in ORACLE_DOMAINS:
            oo = oracle_eval(fsub, ev_t, d)
            log("[smoke oracle %s] oracle AUC=%.4f (train %d rows/%d vids, test %d rows/%d vids)"
                % (d, oo["auc"], oo["n_train"], oo["n_vid_train"], oo["n_test"], oo["n_vid_test"]))
        results["R0"] = dict(smoke_auc=auc, wall=time.time() - t0)
        del vit
        torch.cuda.empty_cache()
        feats_store["R0_smoke"] = sub
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
        save_feats("R0", f)
        o = oracle_bundle(f, ev, "R0")
        r = ratio_eval(f, ev)
        align = {}
        for d in ORACLE_DOMAINS:
            align[d] = p[d] / o[d]["auc"]
            log("[R0 oracle %-6s] AUC=%.4f (train %d rows / %d videos, test %d rows / %d videos) ; "
                "source-label=%.4f ; ALIGNMENT RATIO=%.4f"
                % (d, o[d]["auc"], o[d]["n_train"], o[d]["n_vid_train"],
                   o[d]["n_test"], o[d]["n_vid_test"], p[d], align[d]))
        results["R0"] = dict(probe=p, wall=wall, ratio=r, oracle=o, align=align,
                             gate_ok=bool(gate_ok), max_dev=float(max(devs)),
                             dev_indom=float(dev_indom), dev_stored_probe=dP,
                             dev_stored_multi=dM, r=0, alpha=0, seed=-1,
                             sources="none", n_lora=0, best_val_auc=float("nan"),
                             best_epoch=-1, train_wall=0.0)
        feats_store["R0"] = f
        results["R0"]["feat_path"] = os.path.join(OUT_DIR, "feats_g19b_wild_R0.npz")
        del vit
        torch.cuda.empty_cache()
        gc.collect()
        if not gate_ok:
            log("[R0 GATE] *** STOPPING: pipeline sanity gate FAILED; training runs NOT run ***")
            _finish(ev, results, feats_store, sel_runs, aborted=True)
            return

    # ---------------- training runs ----------------
    for run in sel_runs:
        pack = PACK[run["sources"]]
        tr = train_run(run, pack["pool"], pack["val_paths"], pack["val_y"], pack["val_dom"],
                       pack["val_domains"], device, vit_sd,
                       steps_per_epoch=smoke_steps if smoke else STEPS_PER_EPOCH,
                       max_epochs=smoke_epochs if smoke else MAX_EPOCHS,
                       patience=PATIENCE, nworkers=nw,
                       log_steps=5 if smoke else 0)
        lm = tr["lm"]
        if smoke:
            sel = smoke_selection(ev, 80)
            sub = extract_feats(lm, device, ev["paths"][sel], batch=32, nworkers=0,
                                tag=run["name"] + "-smoke", quiet=True)
            log("[smoke %s] feats shape=%s dtype=%s" % (run["name"], sub.shape, sub.dtype))
            assert sub.shape == (320, 768)
            ysm = ev["y"][sel]
            sc = StandardScaler().fit(sub[:160])
            clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs").fit(
                sc.transform(sub[:160]), ysm[:160])
            auc = float(roc_auc_score(ysm[160:], clf.decision_function(sc.transform(sub[160:]))))
            log("[smoke %s] probe AUC on the 160/160 subset = %.4f" % (run["name"], auc))
            results[run["name"]] = dict(smoke_auc=auc, wall=time.time() - 0,
                                        n_lora=tr["n_lora"], best_val_auc=tr["best_val_auc"],
                                        r=run["r"], alpha=run["alpha"], seed=run["seed"],
                                        sources=run["sources"])
            feats_store[run["name"]] = sub
            del lm
            torch.cuda.empty_cache()
            continue

        t0 = time.time()
        f = extract_feats(lm, device, ev["paths"], tag=run["name"])
        p = probe_eval(f, ev)
        r = ratio_eval(f, ev)
        o = oracle_bundle(f, ev, run["name"])
        align = {d: p[d] / o[d]["auc"] for d in ORACLE_DOMAINS}
        wall = time.time() - t0 + tr["wall"]
        log("[%s] in-domain FF++800 AUC = %.4f" % (run["name"], p["indom"]))
        for d in ALL_TARGET_DOMAINS:
            log("  [%s] %-6s AUC = %.4f%s" % (run["name"], d, p[d],
               "   <-- HELD-OUT (valid transfer)" if d == HELD_OUT else "   [CONTAMINATED]"))
        log("[%s] ratio: s=%.3f class_gap=%.4f mean_ratio=%.4f"
            % (run["name"], r["s"], r["class_gap"], r["ratio_mean"]))
        for d in ORACLE_DOMAINS:
            log("[%s oracle %-6s] AUC=%.4f (train %d rows / %d videos, test %d rows / %d videos) ; "
                "source-label=%.4f ; ALIGNMENT RATIO=%.4f"
                % (run["name"], d, o[d]["auc"], o[d]["n_train"], o[d]["n_vid_train"],
                   o[d]["n_test"], o[d]["n_vid_test"], p[d], align[d]))
        fp = save_feats(run["name"], f)
        results[run["name"]] = dict(probe=p, wall=wall, ratio=r, oracle=o, align=align,
                                    n_lora=tr["n_lora"], best_val_auc=tr["best_val_auc"],
                                    best_epoch=tr["best_epoch"], train_wall=tr["wall"],
                                    r=run["r"], alpha=run["alpha"], seed=run["seed"],
                                    sources=run["sources"], feat_path=fp)
        feats_store[run["name"]] = f
        del lm
        torch.cuda.empty_cache()
        gc.collect()

    if smoke:
        log("")
        log("[smoke] SMOKE COMPLETE")
        for k, v in results.items():
            log("  %s -> %s" % (k, json.dumps({kk: vv for kk, vv in v.items()
                                               if kk in ("smoke_auc", "n_lora", "best_val_auc",
                                                         "r", "alpha", "seed", "sources")},
                                              default=str)))
        log("[gpu final] %s" % (query_gpu(),))
        log("[wall] total = %.1f s" % (time.time() - T_START))
        log.close()
        return

    _finish(ev, results, feats_store, sel_runs, aborted=False)


def _finish(ev, results, feats_store, sel_runs, aborted=False):
    L = log
    L("")
    L("=" * 92)
    L("G19b  (held-out fold = %s, physical GPU 2)  SUMMARY" % HELD_OUT)
    L("=" * 92)
    if aborted:
        L("ABORTED: the R0 pipeline sanity gate failed.  No training runs were run.")

    have = [r["name"] for r in RUNS if r["name"] in results and "probe" in results[r["name"]]]
    if "R0" in results and "probe" in results["R0"]:
        have = ["R0"] + have

    MULTI_NAMES = [n for n in have if n.startswith("O1multi_")]
    FFONLY_NAMES = [n for n in have if n.startswith("O1ffonly_")]
    R2_NAMES = [n for n in have if n.startswith("R2")]
    r0 = results["R0"]["probe"][HELD_OUT] if "R0" in have else float("nan")

    if have:
        rows = [["run"] + ALL_TARGET_DOMAINS + ["indom800", "valid(=wild)"]]
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

        # ---- per-seed + mean/-std ----
        L("")
        L("TABLE 2  per-seed held-out(%s) AUC, mean +- std, and the PAIRED comparison" % HELD_OUT)
        rows = [["seed", "O1multi", "O1ffonly", "O1multi - O1ffonly"]]
        for s in (0, 1, 2):
            m = "O1multi_s%d" % s
            fo = "O1ffonly_s%d" % s
            mv = results[m]["probe"][HELD_OUT] if m in have else float("nan")
            fv = results[fo]["probe"][HELD_OUT] if fo in have else float("nan")
            rows.append([str(s), "%.4f" % mv, "%.4f" % fv, "%+.4f" % (mv - fv)])
        if MULTI_NAMES:
            mm, ms = _mean_std([results[n]["probe"][HELD_OUT] for n in MULTI_NAMES])
            rows.append(["MEAN", "%.4f" % mm, "-", "-"])
            rows.append(["STD", "%.4f" % ms, "-", "-"])
        if FFONLY_NAMES:
            fm, fs = _mean_std([results[n]["probe"][HELD_OUT] for n in FFONLY_NAMES])
            rows.append(["MEAN(f)", "-", "%.4f" % fm, "-"])
            rows.append(["STD(f)", "-", "%.4f" % fs, "-"])
        mm = _mean_std([results[n]["probe"][HELD_OUT] for n in MULTI_NAMES])[0] if MULTI_NAMES else float("nan")
        fm = _mean_std([results[n]["probe"][HELD_OUT] for n in FFONLY_NAMES])[0] if FFONLY_NAMES else float("nan")
        r2v = results[R2_NAMES[0]]["probe"][HELD_OUT] if R2_NAMES else float("nan")
        rows.append(["MEAN diff", "%.4f" % (mm - fm), "R0=%.4f" % r0, "R2=%.4f" % r2v])
        L(fmt_tab(rows))

        # ---- oracle / alignment ----
        L("")
        L("TABLE 3  ORACLE per representation (70/30 VIDEO-identity split of the domain's own 300 "
          "rows, StandardScaler+LR C=1e-3 fit on that domain's OWN labels) and ALIGNMENT RATIO = "
          "source-label AUC / oracle AUC")
        rows = [["run", "oracle_wild", "srclabel_wild", "align_wild",
                 "oracle_dfdcp", "srclabel_dfdcp", "align_dfdcp"]]
        for a in have:
            o = results[a]["oracle"]
            p = results[a]["probe"]
            rows.append([a, "%.4f" % o["wild"]["auc"], "%.4f" % p["wild"], "%.4f" % results[a]["align"]["wild"],
                         "%.4f" % o["dfdcp"]["auc"], "%.4f" % p["dfdcp"], "%.4f" % results[a]["align"]["dfdcp"]])
        L(fmt_tab(rows))

        # ---- G12 ratio ----
        L("")
        L("TABLE 4  domain-sensitivity ratio (G12 formula), source = probe-train 2200")
        rows = [["run", "dim", "s", "class_gap"] + ALL_TARGET_DOMAINS + ["mean_ratio"]]
        for a in have:
            r = results[a]["ratio"]
            rows.append([a, "768", "%.4f" % r["s"], "%.4f" % r["class_gap"]] +
                        ["%.4f" % r["ratio"][d] for d in ALL_TARGET_DOMAINS] +
                        ["%.4f" % r["ratio_mean"]])
        L(fmt_tab(rows))

        # ---- params / wall ----
        L("")
        L("TABLE 5  LoRA trainable params / schedule / wall time")
        rows = [["run", "sources", "seed", "r", "alpha", "lora_params", "best_val_auc",
                 "best_epoch", "train_wall_s", "total_wall_s"]]
        for a in have:
            r = results[a]
            rows.append([a, str(r.get("sources", "-")), str(r.get("seed", "-")),
                         str(r.get("r", "-")), str(r.get("alpha", "-")),
                         str(r.get("n_lora", "-")),
                         "%.4f" % r.get("best_val_auc", float("nan")),
                         str(r.get("best_epoch", "-")),
                         "%.1f" % r.get("train_wall", 0.0), "%.1f" % r.get("wall", 0.0)])
        L(fmt_tab(rows))

        # ---- in-domain / forgetting ----
        L("")
        L("TABLE 6  in-domain FF++ probe-test (800) AUC  [frozen anchor %.4f ; G19B_NO_FORGETTING "
          "threshold %.4f]" % (ANCHOR_INDOM, ANCHOR_INDOM - 0.01))
        rows = [["run", "indom800", ">= 0.9752"]]
        for a in have:
            v = results[a]["probe"]["indom"]
            rows.append([a, "%.4f" % v, "PASS" if v >= ANCHOR_INDOM - 0.01 else "FAIL"])
        L(fmt_tab(rows))

    # ---------------------------------------------------------------- gates ----
    L("")
    L("GATES (mechanical; raw verdicts, no interpretation)")
    gates = {}
    if have and MULTI_NAMES and FFONLY_NAMES:
        mm, ms = _mean_std([results[n]["probe"][HELD_OUT] for n in MULTI_NAMES])
        fm, fs = _mean_std([results[n]["probe"][HELD_OUT] for n in FFONLY_NAMES])
        signs = [results[n]["probe"][HELD_OUT] - r0 for n in MULTI_NAMES]
        r2v = results[R2_NAMES[0]]["probe"][HELD_OUT] if R2_NAMES else float("nan")

        # best arm = highest MEAN held-out AUC among the trained configurations
        cand = {"O1multi": mm, "O1ffonly": fm}
        if R2_NAMES:
            cand["R2multi"] = r2v
        best_arm = max(cand, key=lambda k: cand[k])
        best_names = {"O1multi": MULTI_NAMES, "O1ffonly": FFONLY_NAMES,
                      "R2multi": R2_NAMES}[best_arm]
        best_indom = max(results[n]["probe"]["indom"] for n in best_names)

        g1 = mm - r0
        g1_same_sign = all(s > 0 for s in signs)
        g2a = fm - r0
        g2b = mm - fm
        g3 = mm - r2v
        g4 = best_indom
        align_best = max(results[n]["align"]["wild"] for n in best_names)
        align_r0 = results["R0"]["align"]["wild"] if "R0" in have else float("nan")
        g5 = align_best - align_r0

        L("  held-out(%s) AUC: R0=%.4f | O1multi mean=%.4f (n=%d) | O1ffonly mean=%.4f (n=%d) | R2multi=%.4f"
          % (HELD_OUT, r0, mm, len(MULTI_NAMES), fm, len(FFONLY_NAMES), r2v))
        L("  per-seed O1multi - R0 = %s" % " ".join("%+.4f" % s for s in signs))
        L("  O1multi std=%.4f ; O1ffonly std=%.4f ; paired O1multi - O1ffonly = %s"
          % (ms, fs, " ".join("%+.4f" % (results[m]["probe"][HELD_OUT] -
                                         results[f]["probe"][HELD_OUT])
                              for m, f in zip(MULTI_NAMES, FFONLY_NAMES))))
        L("  G19B_UNFREEZE_REPLICATES: O1multi mean - R0 = %+.4f (>= +0.0100) AND all 3 seeds same sign "
          "(%s) -> %s" % (g1, g1_same_sign, "PASS" if (g1 >= 0.01 and g1_same_sign) else "FAIL"))
        L("  G19B_NEEDS_MULTISOURCE  : O1ffonly mean - R0 = %+.4f (<= +0.0100) AND O1multi mean - "
          "O1ffonly mean = %+.4f (>= +0.0500) -> %s"
          % (g2a, g2b, "PASS" if (g2a <= 0.01 and g2b >= 0.05) else "FAIL"))
        L("  G19B_VS_CAPACITY       : O1multi mean - R2multi = %+.4f (>= +0.0100) -> %s"
          % (g3, "PASS" if g3 >= 0.01 else "FAIL"))
        L("  G19B_NO_FORGETTING     : in-domain(best=%s)=%.4f (>= %.4f) -> %s"
          % (best_arm, g4, ANCHOR_INDOM - 0.01, "PASS" if g4 >= ANCHOR_INDOM - 0.01 else "FAIL"))
        L("    per-arm in-domain: %s"
          % " ".join("%s=%.4f(%s)" % (a, results[a]["probe"]["indom"],
                                      "P" if results[a]["probe"]["indom"] >= ANCHOR_INDOM - 0.01 else "F")
                     for a in have))
        L("  G19B_ALIGNMENT_SHIFT   : align_wild(best=%s)=%.4f - align_wild(R0)=%.4f = %+.4f"
          % (best_arm, align_best, align_r0, g5))
        L("  best trained configuration = %s (mean held-out %.4f)" % (best_arm, cand[best_arm]))
        gates = dict(unfreeze_replicates=g1, unfreeze_same_sign=bool(g1_same_sign),
                     needs_multisource_a=g2a, needs_multisource_b=g2b,
                     needs_multisource_pass=bool(g2a <= 0.01 and g2b >= 0.05),
                     vs_capacity=g3, no_forgetting=g4, alignment_shift=g5,
                     alignment_best=align_best, alignment_r0=align_r0,
                     multi_mean=mm, multi_std=ms, ffonly_mean=fm, ffonly_std=fs,
                     r2=r2v, r0=r0)

    L("")
    L("[gpu final] %s" % (query_gpu(),))
    L("[wall] total = %.1f s" % (time.time() - T_START))

    # ---------------------------------------------------------------- stats ----
    stat = {"arms": np.array(have), "held_out": np.array([HELD_OUT]),
            "target_domains": np.array(ALL_TARGET_DOMAINS),
            "anchor_frozen": np.array([ANCHOR_FROZEN[d] for d in ALL_TARGET_DOMAINS]),
            "anchor_indom": np.array([ANCHOR_INDOM]),
            "contaminated": np.array(CONTAMINATED),
            "oracle_domains": np.array(ORACLE_DOMAINS),
            "multi_runs": np.array(MULTI_NAMES), "ffonly_runs": np.array(FFONLY_NAMES),
            "r2_runs": np.array(R2_NAMES)}
    for a in have:
        stat["auc_%s" % a] = np.array([results[a]["probe"][d] for d in ALL_TARGET_DOMAINS])
        stat["indom_%s" % a] = np.array([results[a]["probe"]["indom"]])
        stat["ratio_%s" % a] = np.array([results[a]["ratio"]["ratio"][d]
                                         for d in ALL_TARGET_DOMAINS])
        stat["classgap_%s" % a] = np.array([results[a]["ratio"]["class_gap"]])
        stat["s_%s" % a] = np.array([results[a]["ratio"]["s"]])
        for d in ORACLE_DOMAINS:
            stat["oracle_%s_%s" % (a, d)] = np.array([results[a]["oracle"][d]["auc"]])
            stat["align_%s_%s" % (a, d)] = np.array([results[a]["align"][d]])
        if "n_lora" in results[a]:
            stat["lora_params_%s" % a] = np.array([results[a]["n_lora"]])
        if "seed" in results[a]:
            stat["seed_%s" % a] = np.array([results[a]["seed"]])
        if "r" in results[a]:
            stat["rank_%s" % a] = np.array([results[a]["r"]])
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
