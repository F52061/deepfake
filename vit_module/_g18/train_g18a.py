# -*- coding: utf-8 -*-
"""
G18a -- specialized DenseNet121 forgery detector trained on FF++ (train.json only).

Goal (this agent's one job): train a DenseNet121 real/fake classifier SPECIALIZED
for forgery detection (unlike G17b which used ImageNet-pretrained weights without
fine-tuning), select the VAL-AUC-optimal checkpoint, then extract GAP features for
the downstream complementarity analysis (G18c).

Feature arms (all GAP-pooled, AdaptiveAvgPool2d(1), fp32):
    df_ffpp_db2    = features.denseblock2 output  (512)
    df_ffpp_db3    = features.denseblock3 output  (1024)
    df_ffpp_final  = whole features output         (1024)

Row order of df_ffpp_feats.npz must be byte-identical in `paths` to
vit_module/_g17/cnn_feats.npz (probe 3000 + multi 2300).

Training data:
    - ONLY official split F:/zhj/data/FaceForensic++_raw/splits/train.json (360 pairs).
    - pair [a,b] -> real videos a,b (original_sequences/c23/faces23/<id>)
                 -> fake videos a_b, b_a (manipulated_sequences/<method>/c23/faces23/<id>_<id>)
                   for method in {Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures}.
    - Video identity (for the 10% video-level val split) = SOURCE id (leaf prefix before '_'),
      same convention as _probe/residual_probe.py.
    - Leak exclusion (hard): any training video folder whose `faces23/<folder>` appears in
      probe-feats test segment (train_mask==False) is dropped.  Audit printed below.

Input pipeline (identical to G17b DenseNet arm for eval/feature extraction):
    cv2.imread -> BGR2RGB -> cv2.resize(336) -> cv2.resize(224, INTER_LINEAR) -> ImageNet norm.
Train-only augmentation: light random crop (scale ~0.89-1.0 square, random offset) + horizontal
flip p=0.5.  NO color jitter / blur / strong photometric aug (would destroy forgery cues).

Label convention: y=1 real, y=0 fake.  Final AUC positive = fake (P(fake)=softmax[:,0]).

GPU discipline (hard): use GPU 1 only; if GPU 1 occupied (>100 MiB) -> STOP, do not switch.
CPU discipline: OMP/MKL/etc = 4, torch.set_num_threads(4), cv2.setNumThreads(0),
num_workers=0 (in-memory dataset, no DataLoader workers).

Outputs (this dir): densenet_ffpp_best.pth, densenet_ffpp_last.pth,
df_ffpp_feats.npz, train_samples.npz, run_log_g18a.txt (via shell redirect).
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["JOBLIB_NUM_THREADS"] = "4"

import subprocess
import sys
import time
import json
import re
import glob
import argparse

import numpy as np


# --------------------------------------------------------------- GPU pick ----
def query_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append(tuple(int(float(x)) for x in parts[:4]))
                except ValueError:
                    continue
        return gpus
    except Exception as e:
        print(f"[gpu] query failed: {e}", flush=True)
        return []


GPU_QUERY = query_gpus()
_g1 = [g for g in GPU_QUERY if g[0] == 1]
if not _g1:
    print("[gpu] FATAL: GPU index 1 not present in nvidia-smi. STOP.", flush=True)
    sys.exit(1)
G1 = _g1[0]
if G1[1] > 100:
    print(f"[gpu] FATAL: GPU 1 occupied ({G1[1]} MiB > 100 MiB). "
          f"STOP and report; will NOT switch to another card.", flush=True)
    sys.exit(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MODE = "gpu"
print(f"[gpu] GPU 1 free (used={G1[1]} MiB), selected. CUDA_VISIBLE_DEVICES=1", flush=True)

import torch
torch.set_num_threads(4)
import cv2
cv2.setNumThreads(0)
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from sklearn.metrics import roc_auc_score
except Exception as e:
    print(f"[import] sklearn missing: {e}", flush=True)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_ROOT = "F:/zhj/data/FaceForensic++_raw"
SPLIT_TRAIN = os.path.join(DATA_ROOT, "splits", "train.json")
PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
G17_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_g17", "cnn_feats.npz")

OUT_DIR = HERE
BEST_PTH = os.path.join(OUT_DIR, "densenet_ffpp_best.pth")
LAST_PTH = os.path.join(OUT_DIR, "densenet_ffpp_last.pth")
FEAT_NPZ = os.path.join(OUT_DIR, "df_ffpp_feats.npz")
SAMPLES_NPZ = os.path.join(OUT_DIR, "train_samples.npz")

METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

IMG_SIZE = 336
CNN_SIZE = 224
SEED = 20260910
VAL_FRAC = 0.10

# sampling budget (balanced real/fake, ~21.5k total)
REAL_FRAMES = 15      # per real video
FAKE_FRAMES = 3       # per (method, manipulated folder) video

BATCH = 32
LR = 1e-4
MAX_EPOCHS = 20
PATIENCE = 6

AUDIT = {}


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


# ------------------------------------------------------------- pipeline ----
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_336(rgb):
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def normalize_224(img224_list):
    """list of uint8 (224,224,3) -> float32 tensor [B,3,224,224] ImageNet norm."""
    tensors = []
    for img in img224_list:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).div_(255.0)
        t = (t - MEAN_T) / STD_T
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def aug_train(img336, rng):
    """Light random crop (square, scale ~0.89-1.0) + hflip p=0.5 -> uint8 (224,224,3)."""
    H = W = IMG_SIZE
    s = rng.randint(int(IMG_SIZE * 0.89), IMG_SIZE)          # 299..336
    top = rng.randint(0, H - s)
    left = rng.randint(0, W - s)
    img = img336[top:top + s, left:left + s]
    img = cv2.resize(img, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        img = img[:, ::-1, :]
    return img


def eval_224(img336):
    """Exact G17b eval path: resize 336 -> 224, no crop/flip."""
    return cv2.resize(img336, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------- densenet build ----
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
_DENSENET_PATTERN = re.compile(
    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")


def _remap_densenet(state_dict):
    out = {k: v for k, v in state_dict.items()}
    for key in list(out.keys()):
        res = _DENSENET_PATTERN.match(key)
        if res:
            out[res.group(1) + res.group(2)] = out[key]
            del out[key]
    return out


def build_densenet2(n_classes=2):
    cache_path = os.path.join(_CACHE_DIR, "densenet121-a639ec97.pth")
    src = None
    try:
        m = densenet121(weights=DenseNet121_Weights.DEFAULT)
        src = "weights-enum cache hit (densenet121-a639ec97.pth)"
    except Exception as e1:
        sd = torch.load(cache_path, map_location="cpu")
        sd = _remap_densenet(sd)
        m = densenet121(weights=None)
        m.load_state_dict(sd)
        src = f"torch.load fallback after enum err {type(e1).__name__}"
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, n_classes)
    return m, src


def make_hook(cap, key):
    def hk(module, inp, out):
        cap[key] = out
    return hk


def gap(t):
    return F.adaptive_avg_pool2d(t, 1).flatten(1).float().cpu().numpy()


# ------------------------------------------------------------ data prep ----
def folder_of(path):
    m = re.search(r"faces23/([^/]+)/", path.replace("\\", "/"))
    return m.group(1) if m else None


def source_of(folder):
    """video identity = source id (leaf prefix before '_'), matching residual_probe.py."""
    return folder.split("_")[0]


def list_frames_sorted(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.png"))
    def key_fn(f):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            return int(stem)
        except ValueError:
            return stem
    return sorted(files, key=key_fn)


def sample_frames(files, k, rng):
    n = len(files)
    if n == 0:
        return []
    if k >= n:
        return list(files)
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    idx = np.unique(np.clip(idx, 0, n - 1))
    return [files[i] for i in idx]


def prepare_data():
    """Enumerate training videos from train.json, exclude probe-test folders,
    sample frames, return parallel lists (path, label, video_key)."""
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    with open(SPLIT_TRAIN, "r") as f:
        pairs = json.load(f)
    AUDIT["train_json_pairs"] = len(pairs)

    # probe-test folder set (hard leak-exclusion set)
    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths_p = np.array([str(s) for s in p["paths"]], dtype=object)
    tr_mask = p["train_mask"].astype(bool)
    te_paths = paths_p[~tr_mask]
    AUDIT["probe_train_n"] = int(tr_mask.sum())
    AUDIT["probe_test_n"] = int((~tr_mask).sum())
    probe_test_folders = set(folder_of(x) for x in te_paths)
    AUDIT["probe_test_folder_set_size"] = len(probe_test_folders)

    # enumerate training videos
    real_folders = set()
    fake_pairs = set()          # (method, folder)
    for a, b in pairs:
        real_folders.add(a)
        real_folders.add(b)
        for m in METHODS:
            fake_pairs.add((m, a + "_" + b))
            fake_pairs.add((m, b + "_" + a))

    AUDIT["train_real_folder_ids"] = len(real_folders)
    AUDIT["train_fake_unique_folders"] = len(set(f[1] for f in fake_pairs))
    AUDIT["train_fake_method_folder_pairs"] = len(fake_pairs)

    # leak audit
    inter_real = real_folders & probe_test_folders
    fake_folders_set = set(f[1] for f in fake_pairs)
    inter_fake = fake_folders_set & probe_test_folders
    AUDIT["leak_intersect_real"] = len(inter_real)
    AUDIT["leak_intersect_fake"] = len(inter_fake)
    AUDIT["leak_intersect_total"] = len(inter_real | inter_fake)
    print("[leak] probe-test folder set size = %d" % AUDIT["probe_test_folder_set_size"], flush=True)
    print("[leak] train.json real folders  %d, intersect with probe-test = %d"
          % (len(real_folders), len(inter_real)), flush=True)
    print("[leak] train.json fake folders  %d, intersect with probe-test = %d"
          % (len(fake_folders_set), len(inter_fake)), flush=True)
    print("[leak] TOTAL intersect = %d (MUST be 0)" % AUDIT["leak_intersect_total"], flush=True)

    # build sample list (exclude probe-test folders + missing dirs)
    paths, labels, keys = [], [], []
    n_missing_real = 0
    n_missing_fake = 0
    n_excluded = 0

    for fo in sorted(real_folders):
        if fo in probe_test_folders:
            n_excluded += 1
            continue
        d = os.path.join(DATA_ROOT, "original_sequences", "c23", "faces23", fo)
        if not os.path.isdir(d):
            n_missing_real += 1
            continue
        files = list_frames_sorted(d)
        sel = sample_frames(files, REAL_FRAMES, rng)
        for fp in sel:
            paths.append(fp); labels.append(1); keys.append(source_of(fo))

    for (m, fo) in sorted(fake_pairs):
        if fo in probe_test_folders:
            n_excluded += 1
            continue
        d = os.path.join(DATA_ROOT, "manipulated_sequences", m, "c23", "faces23", fo)
        if not os.path.isdir(d):
            n_missing_fake += 1
            continue
        files = list_frames_sorted(d)
        sel = sample_frames(files, FAKE_FRAMES, rng)
        for fp in sel:
            paths.append(fp); labels.append(0); keys.append(source_of(fo))

    AUDIT["excluded_videos"] = n_excluded
    AUDIT["missing_real_videos"] = n_missing_real
    AUDIT["missing_fake_method_videos"] = n_missing_fake

    paths = np.array(paths, dtype=object)
    labels = np.array(labels, dtype=np.int64)
    keys = np.array(keys, dtype=object)

    n_real = int((labels == 1).sum())
    n_fake = int((labels == 0).sum())
    AUDIT["n_samples_total"] = int(len(paths))
    AUDIT["n_real"] = n_real
    AUDIT["n_fake"] = n_fake
    AUDIT["prep_wall_s"] = time.time() - t0

    print("[data] samples: total=%d real=%d fake=%d (balanced)" % (len(paths), n_real, n_fake), flush=True)
    print("[data] excluded (leak) videos=%d, missing real=%d, missing fake-method=%d"
          % (n_excluded, n_missing_real, n_missing_fake), flush=True)
    print("[data] prep wall=%.1fs" % AUDIT["prep_wall_s"], flush=True)

    return paths, labels, keys


def video_split(keys, seed=SEED, val_frac=VAL_FRAC):
    rng = np.random.RandomState(seed)
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_frac))
    val_keys = set(uniq[:n_val])
    tr_idx = [i for i, k in enumerate(keys) if k not in val_keys]
    va_idx = [i for i, k in enumerate(keys) if k in val_keys]
    return np.array(tr_idx, dtype=np.int64), np.array(va_idx, dtype=np.int64), len(uniq), n_val


# -------------------------------------------------------------- training ----
@torch.no_grad()
def predict_batch(model, x, device):
    logits = model(x.to(device))
    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


def run_training(paths, labels, keys, device):
    t0 = time.time()
    tr_idx, va_idx, n_videos, n_val_videos = video_split(keys)
    AUDIT["n_videos"] = n_videos
    AUDIT["n_val_videos"] = n_val_videos
    print("[split] videos=%d val=%d  train_samples=%d val_samples=%d"
          % (n_videos, n_val_videos, len(tr_idx), len(va_idx)), flush=True)

    # pre-load images (decode once) -> uint8 (336,336,3)
    print("[load] pre-decoding %d training images to 336x336 ..." % len(paths), flush=True)
    imgs = [None] * len(paths)
    n_fail = 0
    for i, fp in enumerate(paths):
        rgb = read_rgb(fp)
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            n_fail += 1
        imgs[i] = to_336(rgb)
    AUDIT["preload_fail"] = n_fail
    print("[load] done, decode_fail=%d, wall=%.1fs" % (n_fail, time.time() - t0), flush=True)

    model, wsrc = build_densenet2(2)
    print("[weight] densenet121 : " + wsrc, flush=True)
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    rng = np.random.RandomState(SEED + 1)
    n_tr = len(tr_idx)
    y_tr = labels[tr_idx].astype(np.int64)
    y_va = labels[va_idx].astype(np.int64)
    fake_label_va = (y_va == 0).astype(np.int64)

    best_auc = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0
    val_curve = []

    for epoch in range(1, MAX_EPOCHS + 1):
        ep0 = time.time()
        model.train()
        order = np.arange(n_tr)
        rng.shuffle(order)
        total_loss = 0.0
        total_correct = 0
        n_seen = 0
        for b0 in range(0, n_tr, BATCH):
            bidx = order[b0:b0 + BATCH]
            batch_img224 = []
            for i in bidx:
                batch_img224.append(aug_train(imgs[tr_idx[i]], rng))
            x = normalize_224(batch_img224).to(device)
            yb = torch.from_numpy(y_tr[bidx]).to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = crit(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bidx)
            total_correct += int((logits.argmax(1) == yb).sum().item())
            n_seen += len(bidx)
        train_loss = total_loss / max(n_seen, 1)
        train_acc = total_correct / max(n_seen, 1)

        # ---- validation (eval pipeline, no aug) ----
        model.eval()
        va_scores = np.empty(len(va_idx), dtype=np.float32)
        for b0 in range(0, len(va_idx), BATCH):
            bidx = va_idx[b0:b0 + BATCH]
            batch_img224 = [eval_224(imgs[i]) for i in bidx]
            x = normalize_224(batch_img224)
            probs = predict_batch(model, x, device)
            va_scores[b0:b0 + len(bidx)] = probs[:, 0]   # P(fake)
        val_auc = roc_auc_score(fake_label_va, va_scores)
        val_curve.append(float(val_auc))

        improved = val_auc > best_auc
        if improved:
            best_auc = float(val_auc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print("[epoch %02d/%d] lr=%.1e loss=%.4f acc=%.4f val_auc=%.4f %s  wall=%.0fs"
              % (epoch, MAX_EPOCHS, LR, train_loss, train_acc, val_auc,
                 "*BEST*" if improved else "", time.time() - ep0), flush=True)

        if no_improve >= PATIENCE:
            print("[train] early stop: no val-AUC improvement for %d epochs" % PATIENCE, flush=True)
            break

    # save best + last
    if best_state is not None:
        torch.save(best_state, BEST_PTH)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, LAST_PTH)

    # downstream (probe-test AUC + feature extraction) MUST use val-optimal weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device).eval()

    AUDIT["best_val_auc"] = best_auc
    AUDIT["best_epoch"] = best_epoch
    AUDIT["last_val_auc"] = float(val_curve[-1])
    AUDIT["val_curve"] = val_curve
    AUDIT["n_epochs_run"] = epoch
    AUDIT["train_wall_s"] = time.time() - t0

    print("[train] BEST val_auc=%.4f @ epoch %d (curve=%s)" % (best_auc, best_epoch,
          ",".join("%.4f" % v for v in val_curve)), flush=True)
    print("[train] saved best -> %s" % BEST_PTH, flush=True)
    print("[train] saved last -> %s" % LAST_PTH, flush=True)
    return model, best_state


# ------------------------------------------------- probe-test in-domain AUC --
@torch.no_grad()
def probe_test_auc(model, device):
    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths = np.array([str(s) for s in p["paths"]], dtype=object)
    y = p["y"].astype(np.int64)
    tr_mask = p["train_mask"].astype(bool)
    te_idx = np.where(~tr_mask)[0]
    n = len(te_idx)
    model.eval()
    scores = np.empty(n, dtype=np.float32)
    for b0 in range(0, n, BATCH):
        bidx = te_idx[b0:b0 + BATCH]
        batch = []
        for i in bidx:
            rgb = read_rgb(paths[i])
            if rgb is None:
                rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            batch.append(eval_224(to_336(rgb)))
        x = normalize_224(batch)
        probs = predict_batch(model, x, device)
        scores[b0:b0 + len(bidx)] = probs[:, 0]
    fake_label = (y[te_idx] == 0).astype(np.int64)
    auc = roc_auc_score(fake_label, scores)
    AUDIT["probe_test_auc"] = float(auc)
    print("[probe-test] in-domain AUC (n=%d, positive=fake) = %.4f" % (n, auc), flush=True)
    return float(auc)


# ------------------------------------------------------ feature extraction --
def extract_features(model, device):
    t0 = time.time()
    # order: probe 3000 (train_mask order) + multi 2300 (native order)
    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths_p = np.array([str(s) for s in p["paths"]], dtype=object)
    vids_p = np.array([str(s) for s in p["vids"]], dtype=object)
    y_p = p["y"].astype(np.int64)
    tr_mask = p["train_mask"].astype(bool)

    m = np.load(MULTI_NPZ, allow_pickle=True)
    paths_m = np.array([str(s) for s in m["path"]], dtype=object)
    vids_m = np.array([str(s) for s in m["vid"]], dtype=object)
    y_m = m["y"].astype(np.int64)
    dom_m = np.array([str(s) for s in m["domain"]], dtype=object)

    samples = []
    for i in range(len(paths_p)):
        split = "train" if tr_mask[i] else "test"
        samples.append((paths_p[i], vids_p[i], int(y_p[i]), "ffpp_probe", split, "probe"))
    for i in range(len(paths_m)):
        d = dom_m[i]
        samples.append((paths_m[i], vids_m[i], int(y_m[i]), d, d, "multi"))
    n_total = len(samples)
    assert n_total == 5300, n_total

    paths_all = np.array([s[0] for s in samples], dtype=object)
    vids_all = np.array([s[1] for s in samples], dtype=object)
    y_all = np.array([s[2] for s in samples], dtype=np.int64)
    domain_all = np.array([s[3] for s in samples], dtype=object)
    split_all = np.array([s[4] for s in samples], dtype=object)
    source_all = np.array([s[5] for s in samples], dtype=object)

    # hard gate: verify against g17 cnn_feats.npz paths (row order)
    g = np.load(G17_NPZ, allow_pickle=True)
    paths_g = np.array([str(s) for s in g["paths"]], dtype=object)
    order_match = bool(np.array_equal(paths_all, paths_g))
    print("[verify] row-order vs _g17/cnn_feats.npz paths: %s (n=%d)" % (order_match, n_total), flush=True)
    AUDIT["row_order_match"] = order_match

    cap = {}
    model.features.denseblock2.register_forward_hook(make_hook(cap, "db2"))
    model.features.denseblock3.register_forward_hook(make_hook(cap, "db3"))
    model.eval()

    feats = {"db2": [], "db3": [], "final": []}
    n_fail = 0
    for b0 in range(0, n_total, BATCH):
        chunk = samples[b0:b0 + BATCH]
        batch = []
        for (path, *_rest) in chunk:
            rgb = read_rgb(path)
            if rgb is None:
                rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
                n_fail += 1
            batch.append(eval_224(to_336(rgb)))
        x = normalize_224(batch).to(device)
        with torch.no_grad():
            final = model.features(x)
            feats["db2"].append(gap(cap["db2"]))
            feats["db3"].append(gap(cap["db3"]))
            feats["final"].append(gap(final))
        if (b0 // BATCH) % 25 == 0:
            print("  [extract] %d/%d wall=%.0fs" % (b0 + len(chunk), n_total, time.time() - t0), flush=True)

    feats = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in feats.items()}
    for k in feats:
        assert feats[k].shape[0] == n_total, (k, feats[k].shape)

    np.savez(FEAT_NPZ,
             df_ffpp_db2=feats["db2"],
             df_ffpp_db3=feats["db3"],
             df_ffpp_final=feats["final"],
             y=y_all, paths=paths_all, vids=vids_all,
             domain=domain_all, split=split_all, source=source_all)
    AUDIT["extract_read_fail"] = n_fail
    AUDIT["extract_wall_s"] = time.time() - t0
    AUDIT["feat_dims"] = {k: int(feats[k].shape[1]) for k in feats}
    print("[save] df_ffpp_feats.npz -> %s" % FEAT_NPZ, flush=True)
    print("[feat] dims db2=%d db3=%d final=%d read_fail=%d wall=%.0fs"
          % (feats["db2"].shape[1], feats["db3"].shape[1], feats["final"].shape[1],
             n_fail, time.time() - t0), flush=True)
    return order_match


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print("=" * 90, flush=True)
    print("[machine] device=%s  torch=%s  CUDA_VISIBLE_DEVICES=%s"
          % (dev_name, torch.__version__, os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
    print("[machine] GPU_QUERY=%s" % (GPU_QUERY,), flush=True)

    # ---------------- stage 1: data prep + audit ----------------
    paths, labels, keys = prepare_data()

    # save sample list for reproducibility
    np.savez(SAMPLES_NPZ, paths=paths, labels=labels, keys=keys)
    print("[save] train_samples.npz -> %s" % SAMPLES_NPZ, flush=True)

    # ---------------- stage 2: train ----------------
    model = None
    if not args.skip_train:
        model, best_state = run_training(paths, labels, keys, device)
    else:
        model, _ = build_densenet2(2)
        model.to(device)
        sd = torch.load(BEST_PTH, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        print("[skip-train] loaded best weights from %s" % BEST_PTH, flush=True)

    # ---------------- stage 3: probe-test in-domain AUC ----------------
    if model is not None:
        probe_test_auc(model, device)

    # ---------------- stage 4: feature extraction ----------------
    if not args.skip_extract and model is not None:
        extract_features(model, device)

    # ---------------- audit block ----------------
    peak_torch = 0.0
    try:
        peak_torch = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)) if torch.cuda.is_available() else 0.0
    except Exception:
        pass
    wall = time.time() - t0
    print("=" * 90, flush=True)
    print("G18A MACHINE_BLOCK", flush=True)
    print("G18A_MODE=%s" % MODE, flush=True)
    print("G18A_GPU_INDEX=1", flush=True)
    print("G18A_DEVICE=%s" % dev_name, flush=True)
    print("G18A_GPU_QUERY=%s" % (GPU_QUERY,), flush=True)
    print("G18A_TORCH=%s" % torch.__version__, flush=True)
    print("G18A_BATCH=%d" % BATCH, flush=True)
    print("G18A_LR=%.1e" % LR, flush=True)
    print("G18A_MAX_EPOCHS=%d PATIENCE=%d" % (MAX_EPOCHS, PATIENCE), flush=True)
    print("G18A_REAL_FRAMES=%d FAKE_FRAMES=%d" % (REAL_FRAMES, FAKE_FRAMES), flush=True)
    for k in sorted(AUDIT.keys()):
        v = AUDIT[k]
        if isinstance(v, list):
            print("G18A_%s=%s" % (k, ",".join(fmt(x) for x in v)), flush=True)
        else:
            print("G18A_%s=%s" % (k, fmt(v)), flush=True)
    print("G18A_PEAK_TORCH_MIB=%s" % fmt(peak_torch, 0), flush=True)
    print("G18A_WALL_S=%.1f" % wall, flush=True)
    print("G18A_AUG=train:light_random_crop(0.89-1.0)+hflip0.5; eval:336->224(no aug)", flush=True)
    print("G18A_PIPELINE=cv2.imread->BGR2RGB->resize336->resize224(INTER_LINEAR)->ImageNet_norm", flush=True)
    print("G18A_LABEL=1:real,0:fake; AUC positive=fake", flush=True)
    print("=" * 90, flush=True)
    print("[done] wall=%.1fs" % wall, flush=True)


if __name__ == "__main__":
    main()
