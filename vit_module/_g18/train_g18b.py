# -*- coding: utf-8 -*-
"""
G18b -- multi-domain (leave-one-domain-out, LODO) supervised DenseNet121.

Control arm vs G18a (FF++ single-source specialization).  Science question: does
writing cross-domain invariance into the training objective (multi-domain joint
supervision) lower the domain-sensitivity of CNN features (G12 ratio =
dom_gap/class_gap) and make the CNN complementary to the ViT branch?

Three models df_lodo_<domain>.pth for held-out domains {dfdcp, cd2, wild}:
    leave dfdcp : train = FF++ train + {cd1, cd2, ffiw, wild}
    leave cd2   : train = FF++ train + {dfdcp, ffiw, wild}   (cd1 also excluded:
                   cd1's 49 folder names are 100% contained in cd2 -> real leakage)
    leave wild  : train = FF++ train + {cd1, cd2, dfdcp, ffiw}
Each training set contains NO image of its held-out domain.

FF++ train convention (IDENTICAL to G18a for source-domain comparability):
    real = original_sequences/c23/faces23/<id>/  for every 3-digit id in
           splits/train.json (source ids AND target ids), label 1
    fake = manipulated_sequences/<method>/c23/faces23/<id>_<id>/ for BOTH
           directions (a_b and b_a) of every [a,b] pair in train.json, method in
           the 5 FF++ methods, label 0
    Excluded: any folder whose basename is in the probe-test folder set
           (parsed from probe_feats.npz where train_mask==False).

Target-domain data: vit_module/_tsne/feats_multi.npz path/domain/y.  Group the
300 base images per domain by their video folder and re-sample more frames from
those same folders (never outside).

Leakage identity = full folder path (dataset root + video folder name).  The
cd1-subset-cd2 rule is applied by construction (see LODO_TRAIN_DOMAINS).

Model / pipeline (match G18a / G17b):
    torchvision densenet121(DenseNet121_Weights.DEFAULT, local cache), classifier
    replaced by 2-class Linear; .features structure kept.  Full fine-tune.
    Input: cv2.imread -> BGR2RGB -> resize(336) -> [train-only: light random crop
    scale ~0.89-1.0 + hflip p=0.5] -> resize(224) -> ImageNet norm.
    Adam lr 1e-4, batch 32, early stop on val AUC (video-level ~10% val).
    Label: 1 = real, 0 = fake.

Feature extraction: best weights of each LODO model -> GAP features of
    dense121_db2 (512) / dense121_db3 (1024) / dense121_final (1024) for all 5300
    images (probe 3000 + multi 2300, native row order), saved to
    df_lodo_<domain>_feats.npz with y/paths/vids/domain/split/source (same field
    names + semantics as _g17/cnn_feats.npz).

Resource discipline: CUDA_VISIBLE_DEVICES=2 (physical GPU 2, verified idle),
    OMP/MKL/OPENBLAS/NUMEXPR=4, torch.set_num_threads(4), num_workers=0,
    cv2.setNumThreads(0).  Training images pre-decoded once to RAM (uint8 336)
    like G18a so epochs are GPU-bound.

Do NOT touch any G18a file.  This file writes only g18b-named artifacts.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["BLIS_NUM_THREADS"] = "4"

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
from torch.utils.data import Dataset, DataLoader
import cv2
cv2.setNumThreads(0)
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ------------------------------------------------------------------- paths ----
PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
G17_CNN_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_g17", "cnn_feats.npz")
OUT_DIR = HERE
LOG_PATH = os.path.join(OUT_DIR, "run_log_g18b.txt")

FFPP_RAW = "F:/zhj/data/FaceForensic++_raw"
FFPP_TRAIN_JSON = os.path.join(FFPP_RAW, "splits", "train.json")
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# ---------------------------------------------------------------- constants --
SEED = 20260910
IMG_SIZE = 336
CNN_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

REAL_FRAMES = 15           # per real video (matches G18a)
FAKE_FRAMES = 3            # per (method, manipulated folder) video (matches G18a)
TARGET_MAX_PER_VIDEO = 40
TARGET_CLASS_CAP = 2000    # per class, per target domain
VAL_MAX_PER_VIDEO = 8
VAL_FRAC = 0.10
BATCH = 32
LR = 1e-4
MAX_EPOCHS = 20            # matches G18a
PATIENCE = 6               # matches G18a
AUG_CROP_MIN = 0.89

ALL_TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
LODO_HELD_OUT = ["dfdcp", "cd2", "wild"]
LODO_TRAIN_DOMAINS = {
    "dfdcp": ["cd1", "cd2", "ffiw", "wild"],
    "cd2": ["dfdcp", "ffiw", "wild"],        # cd1 excluded (subset of cd2)
    "wild": ["cd1", "cd2", "dfdcp", "ffiw"],
}
FEAT_KEYS = ["dense121_db2", "dense121_db3", "dense121_final"]

# ------------------------------------------------------------------- logger --
class Logger:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg):
        print(msg, flush=True)
        self.fh.write(str(msg) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


log = Logger(LOG_PATH)
t_start = time.time()


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


def balance_by_class(samples, cap=None, rng=None):
    """samples: list of (path, label).  Return list capped so real==fake."""
    rng = rng or random.Random(SEED)
    real = [s for s in samples if s[1] == 1]
    fake = [s for s in samples if s[1] == 0]
    n = min(len(real), len(fake))
    if cap is not None:
        n = min(n, cap)
    if len(real) > n:
        rng.shuffle(real)
        real = real[:n]
    if len(fake) > n:
        rng.shuffle(fake)
        fake = fake[:n]
    out = real + fake
    rng.shuffle(out)
    return out


# ------------------------------------------------------------- data build ----
def build_probe_test_folders():
    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths = p["paths"].astype(str)
    tm = p["train_mask"].astype(bool)
    return set(os.path.basename(os.path.dirname(x)) for x in paths[~tm])


def build_ffpp_videos(probe_test_folders):
    """Bidirectional convention, identical to G18a."""
    with open(FFPP_TRAIN_JSON, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    real_ids = sorted(set([a for a, _ in pairs]) | set([b for _, b in pairs]))
    videos = []          # (folder, label, vid_key)
    n_missing = 0
    n_excluded = 0
    for vid in real_ids:
        folder = os.path.join(FFPP_RAW, "original_sequences", "c23", "faces23", vid)
        if not os.path.isdir(folder):
            n_missing += 1
            continue
        if os.path.basename(folder) in probe_test_folders:
            n_excluded += 1
            continue
        videos.append((folder, 1, "ffpp_real_%s" % vid))
    for a, b in pairs:
        for method in FFPP_METHODS:
            for vid in ("%s_%s" % (a, b), "%s_%s" % (b, a)):     # both directions
                folder = os.path.join(FFPP_RAW, "manipulated_sequences", method,
                                      "c23", "faces23", vid)
                if not os.path.isdir(folder):
                    n_missing += 1
                    continue
                if os.path.basename(folder) in probe_test_folders:
                    n_excluded += 1
                    continue
                videos.append((folder, 0, "ffpp_fake_%s_%s" % (method, vid)))
    return videos, n_missing, n_excluded


def build_target_videos(domains):
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
            videos.append((fld, lab, "%s_%s" % (d, os.path.basename(fld))))
    return videos


# ------------------------------------------------------------- pipeline ----
def read_rgb336(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def aug_train(img336):
    s = random.randint(int(IMG_SIZE * AUG_CROP_MIN), IMG_SIZE)   # 299..336
    top = random.randint(0, IMG_SIZE - s)
    left = random.randint(0, IMG_SIZE - s)
    img = img336[top:top + s, left:left + s]
    img = cv2.resize(img, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
    return img


def eval_224(img336):
    return cv2.resize(img336, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_LINEAR)


def normalize_224(img224_list):
    tensors = []
    for img in img224_list:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).div_(255.0)
        t = (t - MEAN_T) / STD_T
        tensors.append(t)
    return torch.stack(tensors, dim=0)


class MemDataset(Dataset):
    def __init__(self, samples, cache, train=False):
        self.samples = samples
        self.cache = cache
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img336 = self.cache[path]
        if self.train:
            img224 = aug_train(img336)
        else:
            img224 = eval_224(img336)
        t = torch.from_numpy(img224.astype(np.float32)).permute(2, 0, 1).div_(255.0)
        t = (t - MEAN_T) / STD_T
        return t, torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------- model -------
_DENSENET_PATTERN = re.compile(
    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")


def _remap_densenet(state_dict):
    out = {k: v for k, v in state_dict.items()}
    for key in list(out.keys()):
        res = _DENSENET_PATTERN.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            out[new_key] = out[key]
            del out[key]
    return out


def build_model():
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                              "checkpoints", "densenet121-a639ec97.pth")
    try:
        m = densenet121(weights=DenseNet121_Weights.DEFAULT)
        src = "weights-enum cache hit"
    except Exception as e:
        sd = torch.load(cache_path, map_location="cpu")
        sd = _remap_densenet(sd)
        m = densenet121(weights=None)
        m.load_state_dict(sd)
        src = "torch.load fallback (enum err %s)" % type(e).__name__
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m, src


# ---------------------------------------------------------- evaluation -----
@torch.no_grad()
def predict_scores(model, samples, cache, device, batch=BATCH):
    """Return (y, score_real) for a list of (path, label); cache: {path: uint8 336}."""
    model.eval()
    ys = []
    scores = []
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        imgs224 = [eval_224(cache[p]) for (p, _l) in chunk]
        x = normalize_224(imgs224).to(device)
        out = model(x)
        prob = F.softmax(out, dim=1)[:, 1]
        ys.append(np.array([l for (_p, l) in chunk], dtype=np.int64))
        scores.append(prob.cpu().numpy())
    model.train()
    return np.concatenate(ys), np.concatenate(scores)


def auc_of(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


# ---------------------------------------------------------- training ------
def train_one_model(held_out, train_videos, val_videos, device, weight_src):
    log("=" * 90)
    log("TRAIN model df_lodo_%s" % held_out)
    log("=" * 90)

    rng = random.Random(SEED + LODO_HELD_OUT.index(held_out))
    train_samples = []
    n_real_vid = n_fake_vid = 0
    for folder, label, vkey in train_videos:
        if label == 1:
            cap = REAL_FRAMES if vkey.startswith("ffpp") else TARGET_MAX_PER_VIDEO
            n_real_vid += 1
        else:
            cap = FAKE_FRAMES if vkey.startswith("ffpp") else TARGET_MAX_PER_VIDEO
            n_fake_vid += 1
        for fp in sample_frames(folder, cap):
            train_samples.append((fp, label))
    n_before = len(train_samples)
    train_samples = balance_by_class(train_samples, cap=None, rng=rng)
    log("[train] videos real=%d fake=%d ; frames before balance=%d after=%d"
        % (n_real_vid, n_fake_vid, n_before, len(train_samples)))

    val_samples = []
    for folder, label, vkey in val_videos:
        for fp in sample_frames(folder, VAL_MAX_PER_VIDEO):
            val_samples.append((fp, label))

    # ---- pre-decode all train+val images once to RAM (uint8 336) ----
    cache = {}
    t0 = time.time()
    for (p, _l) in train_samples + val_samples:
        if p not in cache:
            cache[p] = read_rgb336(p)
    log("[preload] decoded %d unique images (train+val) in %.1fs"
        % (len(cache), time.time() - t0))

    train_ds = MemDataset(train_samples, cache, train=True)
    val_ds = MemDataset(val_samples, cache, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0, drop_last=False)

    model, _ = build_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_state = None
    patience_left = PATIENCE
    curve = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += y.size(0)
            correct += int((out.argmax(1) == y).sum().item())
            loss_sum += float(loss.item()) * y.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        vy, vs = predict_scores(model, val_samples, cache, device)
        val_auc = auc_of(vy, vs)
        val_acc = float(((vs >= 0.5).astype(np.int64) == vy).mean())

        curve.append((epoch, train_loss, train_acc, val_auc, val_acc))
        improved = val_auc > best_auc
        tag = " *BEST*" if improved else ""
        log("[epoch %02d/%d] loss=%.4f acc=%.4f val_auc=%.4f val_acc=%.4f%s (%.0fs)"
            % (epoch, MAX_EPOCHS, train_loss, train_acc, val_auc, val_acc, tag,
               time.time() - t0))
        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                log("[early-stop] no val_auc improvement for %d epochs at epoch %d"
                    % (PATIENCE, epoch))
                break

    model.load_state_dict(best_state)
    best_epoch = max(range(len(curve)), key=lambda i: curve[i][3]) + 1
    log("[done] best val_auc=%.4f (epoch %d)  weight_src=%s" % (best_auc, best_epoch, weight_src))

    # ---- held-out domain AUC (base 300 canonical + capped expanded) ----
    m = np.load(MULTI_NPZ, allow_pickle=True)
    dom = m["domain"].astype(str)
    path = m["path"].astype(str)
    y = m["y"].astype(np.int64)
    ho_idx = np.where(dom == held_out)[0]
    ho_base = [(str(path[i]), int(y[i])) for i in ho_idx]
    for (p, _l) in ho_base:
        if p not in cache:
            cache[p] = read_rgb336(p)
    hoy, hos = predict_scores(model, ho_base, cache, device)
    ho_auc_base = auc_of(hoy, hos)

    ho_folders = sorted(set(os.path.dirname(str(path[i])) for i in ho_idx))
    ho_exp = []
    for fld in ho_folders:
        lab = None
        for i in ho_idx:
            if os.path.dirname(str(path[i])) == fld:
                lab = int(y[i])
                break
        for fp in list_frames(fld):
            ho_exp.append((fp, lab))
    # cap expanded eval to keep it fast while still stable (esp. wild ~50k frames)
    rng_exp = random.Random(SEED + 1000 + LODO_HELD_OUT.index(held_out))
    rng_exp.shuffle(ho_exp)
    ho_exp = ho_exp[:8000]
    for (p, _l) in ho_exp:
        if p not in cache:
            cache[p] = read_rgb336(p)
    hoy2, hos2 = predict_scores(model, ho_exp, cache, device)
    ho_auc_exp = auc_of(hoy2, hos2)
    log("[held-out %s] base-300 AUC=%.4f (n=%d) ; expanded AUC=%.4f (n=%d)"
        % (held_out, ho_auc_base, len(ho_base), ho_auc_exp, len(ho_exp)))

    weight_path = os.path.join(OUT_DIR, "df_lodo_%s.pth" % held_out)
    torch.save(best_state, weight_path)
    log("[save] %s" % weight_path)

    del cache
    gc.collect()

    result = {
        "held_out": held_out,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "curve": curve,
        "ho_auc_base": ho_auc_base,
        "ho_auc_exp": ho_auc_exp,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_train_videos": len(train_videos),
        "n_val_videos": len(val_videos),
    }
    return result


# ------------------------------------------------------- feature extract ----
def extract_features(held_out, device):
    weight_path = os.path.join(OUT_DIR, "df_lodo_%s.pth" % held_out)
    model, _ = build_model()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cap = {}
    model.features.denseblock2.register_forward_hook(
        lambda mo, i, o: cap.__setitem__("denseblock2", o))
    model.features.denseblock3.register_forward_hook(
        lambda mo, i, o: cap.__setitem__("denseblock3", o))

    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths_p = np.array([str(s) for s in p["paths"]], dtype=object)
    vids_p = np.array([str(s) for s in p["vids"]], dtype=object)
    y_p = p["y"].astype(np.int64)
    tr_mask = p["train_mask"].astype(bool)
    assert int(tr_mask.sum()) == 2200

    m = np.load(MULTI_NPZ, allow_pickle=True)
    paths_m = np.array([str(s) for s in m["path"]], dtype=object)
    vids_m = np.array([str(s) for s in m["vid"]], dtype=object)
    y_m = m["y"].astype(np.int64)
    dom_m = np.array([str(s) for s in m["domain"]], dtype=object)

    samples = []
    for i in range(len(paths_p)):
        spl = "train" if tr_mask[i] else "test"
        samples.append((paths_p[i], vids_p[i], int(y_p[i]), "ffpp_probe", spl, "probe"))
    for i in range(len(paths_m)):
        d = dom_m[i]
        samples.append((paths_m[i], vids_m[i], int(y_m[i]), d, d, "multi"))
    n_total = len(samples)
    assert n_total == 5300

    paths_all = np.array([s[0] for s in samples], dtype=object)
    vids_all = np.array([s[1] for s in samples], dtype=object)
    y_all = np.array([s[2] for s in samples], dtype=np.int64)
    domain_all = np.array([s[3] for s in samples], dtype=object)
    split_all = np.array([s[4] for s in samples], dtype=object)
    source_all = np.array([s[5] for s in samples], dtype=object)

    feats = {k: [] for k in FEAT_KEYS}
    t0 = time.time()
    for i in range(0, n_total, BATCH):
        chunk = samples[i:i + BATCH]
        rgbs = [read_rgb336(s[0]) for s in chunk]
        x = normalize_224([eval_224(r) for r in rgbs]).to(device)
        with torch.no_grad():
            dfinal = model.features(x)
            feats["dense121_db2"].append(
                F.adaptive_avg_pool2d(cap["denseblock2"], 1).flatten(1).float().cpu().numpy())
            feats["dense121_db3"].append(
                F.adaptive_avg_pool2d(cap["denseblock3"], 1).flatten(1).float().cpu().numpy())
            feats["dense121_final"].append(
                F.adaptive_avg_pool2d(dfinal, 1).flatten(1).float().cpu().numpy())
        if (i // BATCH) % 25 == 0:
            log("  [extract %s] %d/%d wall=%.0fs" % (held_out, i + len(chunk), n_total,
                                                     time.time() - t0))
    feats = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in feats.items()}
    for k in FEAT_KEYS:
        assert feats[k].shape[0] == n_total

    out_path = os.path.join(OUT_DIR, "df_lodo_%s_feats.npz" % held_out)
    np.savez(out_path,
             **feats,
             y=y_all, paths=paths_all, vids=vids_all,
             domain=domain_all, split=split_all, source=source_all)

    g = np.load(G17_CNN_NPZ, allow_pickle=True)
    gpaths = np.array([str(s) for s in g["paths"]], dtype=object)
    align = bool(np.array_equal(paths_all, gpaths))
    log("[align %s] feats paths == _g17/cnn_feats.npz paths: %s" % (held_out, align))
    if not align:
        log("[align %s] FATAL: row-order mismatch" % held_out)
    return out_path, align


# ------------------------------------------------------------- ratio ------
def compute_ratios(held_out):
    npz_path = os.path.join(OUT_DIR, "df_lodo_%s_feats.npz" % held_out)
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"].astype(np.int64)
    split = d["split"].astype(str)
    domain = d["domain"].astype(str)
    tr = (split == "train")
    log("")
    log("[ratio %s] source=probe-train n=%d ; targets=%s" %
        (held_out, int(tr.sum()), ",".join(ALL_TARGET_DOMAINS)))
    log("  arm        dim     s     class_gap | " +
        " ".join("%9s" % dm for dm in ALL_TARGET_DOMAINS) + "   (ratio=dom_gap/class_gap)")
    ratio_table = {}
    for arm in FEAT_KEYS:
        X = d[arm].astype(np.float64)
        Xtr = X[tr]
        ys = y[tr]
        mu_src = Xtr.mean(axis=0)
        mu_real = Xtr[ys == 1].mean(axis=0)
        mu_fake = Xtr[ys == 0].mean(axis=0)
        s = float(np.sqrt(np.var(Xtr, axis=0, ddof=1).mean()))
        class_gap = float(np.linalg.norm(mu_fake - mu_real) / s)
        ratios = {}
        for dm in ALL_TARGET_DOMAINS:
            Xd = X[domain == dm]
            dom_gap = float(np.linalg.norm(Xd.mean(axis=0) - mu_src) / s)
            ratios[dm] = dom_gap / class_gap
        ratio_table[arm] = ratios
        log("  %-10s %4d %7.3f %8.4f  | " % (arm, X.shape[1], s, class_gap) +
            " ".join("%9.4f" % ratios[dm] for dm in ALL_TARGET_DOMAINS) +
            "   | mean=%.4f" % float(np.mean([ratios[dm] for dm in ALL_TARGET_DOMAINS])))
    return ratio_table


# --------------------------------------------------------------- main -----
def main():
    gpus = query_gpu()
    log("[gpu] nvidia-smi query=%s" % (gpus,))
    log("[gpu] CUDA_VISIBLE_DEVICES=%s ; target physical GPU index 2" %
        os.environ.get("CUDA_VISIBLE_DEVICES"))
    g2 = [g for g in gpus if g[0] == 2]
    if g2:
        used = g2[0][1]
        log("[gpu] physical GPU 2 mem.used=%dMiB -> %s" %
            (used, "IDLE (<=100MiB)" if used <= 100 else "BUSY (>100MiB)"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    log("[torch] %s ; device=%s ; threads=%d" %
        (torch.__version__, torch.cuda.get_device_name(0), torch.get_num_threads()))

    # ---------------- data audit ----------------
    probe_test_folders = build_probe_test_folders()
    log("[probe] probe-test folders = %d" % len(probe_test_folders))

    ffpp_videos, ffpp_missing, ffpp_excluded = build_ffpp_videos(probe_test_folders)
    n_ffpp_real = sum(1 for v in ffpp_videos if v[1] == 1)
    n_ffpp_fake = sum(1 for v in ffpp_videos if v[1] == 0)
    log("[ffpp] train videos: real=%d fake=%d total=%d missing=%d excluded_by_probe_test=%d"
        % (n_ffpp_real, n_ffpp_fake, len(ffpp_videos), ffpp_missing, ffpp_excluded))
    ffpp_folder_set = set(v[0] for v in ffpp_videos)
    ffpp_base_set = set(os.path.basename(v[0]) for v in ffpp_videos)
    log("[leak] ffpp train folders (basename) intersect probe-test: %d" %
        len(ffpp_base_set & probe_test_folders))

    m = np.load(MULTI_NPZ, allow_pickle=True)
    dom_all = m["domain"].astype(str)
    path_all = m["path"].astype(str)

    cd1_names = set(os.path.basename(os.path.dirname(str(path_all[i])))
                    for i in np.where(dom_all == "cd1")[0])
    cd2_names = set(os.path.basename(os.path.dirname(str(path_all[i])))
                    for i in np.where(dom_all == "cd2")[0])
    log("[audit] cd1 folder names subset of cd2 folder names: %s (cd1=%d cd2=%d inter=%d)"
        % (cd1_names <= cd2_names, len(cd1_names), len(cd2_names), len(cd1_names & cd2_names)))

    all_results = {}
    all_ratios = {}
    all_align = {}
    for held_out in LODO_HELD_OUT:
        train_domains = LODO_TRAIN_DOMAINS[held_out]
        target_videos = build_target_videos(train_domains)

        # ---- leak audit: held-out domain paths vs training paths ----
        ho_paths = set(str(path_all[i]) for i in np.where(dom_all == held_out)[0])
        train_folders = set(ffpp_folder_set) | set(v[0] for v in target_videos)
        ho_folders = set(os.path.dirname(x) for x in ho_paths)
        inter = train_folders & ho_folders
        log("[leak %s] held-out folders in training (full-path): %d" % (held_out, len(inter)))
        for x in sorted(inter)[:5]:
            log("    LEAK: %s" % x)
        assert len(inter) == 0, "held-out %s leaked into training" % held_out

        if held_out == "cd2":
            cd1_paths = set(str(path_all[i]) for i in np.where(dom_all == "cd1")[0])
            cd1_folders = set(os.path.dirname(x) for x in cd1_paths)
            inter2 = train_folders & cd1_folders
            log("[leak cd2] cd1 folders in training (full-path): %d" % len(inter2))
            assert len(inter2) == 0

        # ---- video-level val split (~10%, stratified) ----
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
        log("[split %s] train videos=%d (real %d/fake %d) ; val videos=%d (real %d/fake %d) "
            "train domains=%s" %
            (held_out, len(train_videos),
             sum(1 for v in train_videos if v[1] == 1),
             sum(1 for v in train_videos if v[1] == 0),
             len(val_videos),
             sum(1 for v in val_videos if v[1] == 1),
             sum(1 for v in val_videos if v[1] == 0),
             train_domains))

        res = train_one_model(held_out, train_videos, val_videos, device, "")
        all_results[held_out] = res

        torch.cuda.empty_cache()
        out_path, align = extract_features(held_out, device)
        all_align[held_out] = align
        all_ratios[held_out] = compute_ratios(held_out)
        torch.cuda.empty_cache()

    # ---------------- summary ----------------
    log("")
    log("=" * 90)
    log("G18B SUMMARY")
    log("=" * 90)
    for held_out in LODO_HELD_OUT:
        r = all_results[held_out]
        log("[%s] best_val_auc=%.4f (epoch %d)  held-out base-300 AUC=%.4f  "
            "expanded AUC=%.4f  n_train=%d n_val=%d  align=%s" %
            (held_out, r["best_val_auc"], r["best_epoch"], r["ho_auc_base"],
             r["ho_auc_exp"], r["n_train"], r["n_val"], all_align[held_out]))
    log("")
    log("[wall] total = %.1f s" % (time.time() - t_start))
    log.close()


if __name__ == "__main__":
    main()
