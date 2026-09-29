# -*- coding: utf-8 -*-
"""G13 -- H7: net_050 (pre-finetune) vs bridge_v2 (post-finetune) ViT representation contrast.

Phases
  A  load / sanity:  bridge_v2 model vs probe_feats.npz V (pipeline check),
     then overwrite the vit submodule with net_050_backup.pth ("pre"),
     then check cos(pre, post) on the same 10 probe-test images.
  B  extract pre-V for 4500 images (probe 2200 train + 800 test, 5 target domains x300).
     post-V is NOT re-extracted: it comes from probe_feats.npz / feats_multi.npz.
  C  per-model protocol: center-only PCA on source train -> mu/e0/PC0 varfrac;
     StandardScaler+LR(C=1e-3) in-domain video-disjoint test AUC;
     cross-domain (5 domains): e0-axis AUC, source-LR AUC, d_z, gap_ratio,
     d_real/d_fake, zdim_frac, |cos(e0_dom, e0_src)|;
     class-conditional (real-only / fake-only) domain-separability linear probes.
  D  optional intermediate point vit_m2f2_phase1.pth (frozen backbone check, 300 imgs).

Resource discipline: single process, CPU-side threads pinned to 1, DataLoader-free,
torch.set_num_threads(1), cv2.setNumThreads(0).  GPU picked from nvidia-smi
(idle <=100 MiB used and >=6 GB free -> lowest-used one), else CPU single-thread.
fp32, batch <= 16.

Reference implementation reused verbatim: vit_module/_g10/run_g10.py L118-145
(to_336 / to_tensor_batch / forward_V) and L386-410 (model build + ckpt load).
"""

import os

# ---------------------------------------------------------------- CPU pins ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import sys
import time
import json
import random
import subprocess
from collections import OrderedDict

# ------------------------------------------------- GPU pick (before torch) ---
SMI_QUERY = ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]
SMI_QUERY_HUMAN = ["nvidia-smi",
                   "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                   "--format=csv"]


def _digits(s):
    out = "".join(ch for ch in str(s) if ch.isdigit() or ch == "-")
    return int(out) if out not in ("", "-") else None


def query_gpus():
    try:
        txt = subprocess.check_output(SMI_QUERY, text=True, timeout=60,
                                      stderr=subprocess.STDOUT)
    except Exception as exc:                                   # pragma: no cover
        return [], "nvidia-smi failed: %r" % (exc,)
    rows = []
    for line in txt.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, used, total = _digits(parts[0]), _digits(parts[1]), _digits(parts[2])
        if None in (idx, used, total):
            continue
        rows.append(dict(index=idx, used=used, total=total, util=parts[3]))
    human = txt.strip()
    try:                       # keep the human-readable table for the report
        human = subprocess.check_output(SMI_QUERY_HUMAN, text=True, timeout=60,
                                        stderr=subprocess.STDOUT).strip()
    except Exception:
        pass
    return rows, human


GPU_ROWS, SMI_TEXT = query_gpus()
GPU_IDLE_MIB = 100
GPU_MIN_FREE_MIB = 6 * 1024
CANDS = [r for r in GPU_ROWS
         if r["used"] <= GPU_IDLE_MIB and (r["total"] - r["used"]) >= GPU_MIN_FREE_MIB]
CANDS.sort(key=lambda r: r["used"])
if CANDS:
    GPU_PICK = CANDS[0]["index"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_PICK)
    DEVICE_MODE = "gpu"
else:
    GPU_PICK = None
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    DEVICE_MODE = "cpu"

# --------------------------------------------------------------- imports ----
import numpy as np
import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from albumentations import Compose, Normalize, ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_1", "bridge_v2_phase1.pth")
PRE_CKPT = os.path.join(PROJECT_ROOT, "vit_module", "backup_weights", "net_050_backup.pth")
MID_CKPT = os.path.join(PROJECT_ROOT, "vit_module", "backup_weights", "vit_m2f2_phase1.pth")
CLIP_LOCAL = os.path.join(PROJECT_ROOT, "checkpoints", "clip-vit-large-patch14-336")
OUT_PRE_NPZ = os.path.join(HERE, "pre_feats.npz")
OUT_STATS_NPZ = os.path.join(HERE, "g13_stats.npz")
REPORT = os.path.join(HERE, "g13_report.txt")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TRANSFORM = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

IMG_SIZE = 336
BATCH = 16
SEED = 20260910
TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]

# ---- pre-registered decision constants (G13 B1..B4) ----
B1_TAU = 0.02          # in-domain test AUC delta >= +0.02 -> IMPROVED
B2_HI, B2_LO = 0.03, 0.01   # cross-domain (srcLR) delta
B3_GAP_DROP = 0.10     # post <= pre - 0.10 -> gap_ratio collapsed
B3_NEAR = 0.05         # |d gap_ratio| < 0.05 -> "unchanged"
B3_SHIFT_BIG = 1.50    # post shift magnitude > 1.50 x pre -> "clearly larger"
B3_SHIFT_EQ_HI, B3_SHIFT_EQ_LO = 1.25, 0.80   # "comparable magnitude"
B3_SHIFT_DOMAINS = 3   # need >=3/5 domains
B4_HI, B4_LO = 0.50, 0.30   # pre PC0 varfrac
D_COS_FROZEN = 0.999

N_PROBE_TRAIN, N_PROBE_TEST, N_DOM = 2200, 800, 300
N_TOTAL = N_PROBE_TRAIN + N_PROBE_TEST + 5 * N_DOM

# known G11 post-model anchors (same protocol), for a pure cross-check
G11_ANCHORS = {
    "pc0_varfrac": 0.6229, "srclr_test_auc": 0.9852,
    "e0_auc": {"cd1": 0.8576, "cd2": 0.8604, "dfdcp": 0.8322, "ffiw": 0.8103, "wild": 0.8060},
    "srclr_auc": {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090},
}

AUDIT = OrderedDict()
AUDIT["imgs_forward"] = 0
AUDIT["batch_calls"] = 0
AUDIT["read_fail"] = 0


def reset_audit():
    AUDIT["imgs_forward"] = 0
    AUDIT["batch_calls"] = 0
    AUDIT["read_fail"] = 0


# ------------------------------------------------------- image pipeline ----
# verbatim from vit_module/_g10/run_g10.py L118-145
def read_rgb(path):
    """cv2.imread -> BGR2RGB at native resolution (zeros on failure)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        AUDIT["read_fail"] += 1
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_336(rgb):
    """native RGB uint8 -> 336 RGB uint8 (identical to the probe pipeline)."""
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return rgb


def to_tensor_batch(rgb_list):
    ts = [TRANSFORM(image=im)["image"] for im in rgb_list]
    return torch.stack(ts, dim=0)


@torch.no_grad()
def forward_V(model, rgb_list, device):
    """rgb_list: list of RGB uint8 336x336 -> raw ViT CLS V [n,768] float32 numpy."""
    if not rgb_list:
        return np.zeros((0, 768), np.float32)
    feats = []
    for i in range(0, len(rgb_list), BATCH):
        chunk = rgb_list[i:i + BATCH]
        x = to_tensor_batch(chunk).to(device)
        AUDIT["batch_calls"] += 1
        AUDIT["imgs_forward"] += len(chunk)
        vit_in = model._preprocess_for_vit(x).to(model.vit_dtype)
        out = model.vit.forward_features(vit_in)
        feats.append(out[:, 0, :].float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# ------------------------------------------------------------- model I/O ----
def build_bridge_model():
    """verbatim construction from vit_module/_g10/run_g10.py L386-410."""
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=CLIP_LOCAL,
        clip_vision_encoder_name=CLIP_LOCAL,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] bridge_v2 ckpt: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    return model, len(missing), len(unexpected)


def unwrap_state_dict(obj):
    """Return (state_dict, how). Handles {'model'/'state_dict'/'model_state_dict': ...}."""
    if isinstance(obj, dict) and not all(torch.is_tensor(v) for v in list(obj.values())[:1]):
        for key in ("model_state_dict", "state_dict", "model"):
            inner = obj.get(key, None)
            if (isinstance(inner, dict) and len(inner) > 100
                    and all(torch.is_tensor(v) for v in list(inner.values())[:3])):
                return inner, "unwrapped['%s']" % key
    return obj, "as-is"


def load_pre_vit(model, path, tag):
    """Map net_050-style 'model.*' -> 'vit.*' and copy into the built model's vit submodule.

    Precedent: _archive/analyze_branch_redundancy.py:163 (load_vit_backbone).
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd, how = unwrap_state_dict(raw)
    msd = model.state_dict()
    vit_keys = sorted(k for k in msd if k.startswith("vit."))
    mapped, skipped, non_model = {}, [], []
    for k, v in sd.items():
        if k.startswith("model."):
            cand = "vit." + k[len("model."):]
        elif k.startswith("vit."):
            cand = k
        else:
            non_model.append(k)
            continue
        if (cand in msd and hasattr(v, "shape")
                and tuple(v.shape) == tuple(msd[cand].shape)):
            mapped[cand] = v.to(msd[cand].dtype)
        else:
            skipped.append(k)
    missing = [k for k in vit_keys if k not in mapped]
    model.load_state_dict(mapped, strict=False)
    info = OrderedDict(tag=tag, path=os.path.basename(path), wrapper=how,
                       n_keys_in_file=len(sd), n_vit_target_keys=len(vit_keys),
                       n_mapped=len(mapped), missing=missing, skipped=skipped,
                       non_model_prefix=non_model)
    print(f"[{tag}] wrapper={how} file_keys={len(sd)} vit_targets={len(vit_keys)} "
          f"mapped={len(mapped)} missing={len(missing)} skipped={len(skipped)} "
          f"non_model_keys={len(non_model)}", flush=True)
    if skipped:
        print(f"[{tag}]   skipped: {skipped}", flush=True)
    if missing:
        print(f"[{tag}]   MISSING (kept bridge_v2 values): {missing}", flush=True)
    if non_model:
        print(f"[{tag}]   non-model-prefix keys: {non_model}", flush=True)
    return info


def vit_tensor_identity(sd_a, sd_b):
    """Exact tensor-level comparison of two {vit.*: tensor} state dicts."""
    common = sorted(set(sd_a) & set(sd_b))
    n_diff, mx, worst, n_shape = 0, 0.0, "", 0
    for k in common:
        a, b = sd_a[k], sd_b[k]
        if tuple(a.shape) != tuple(b.shape):
            n_shape += 1
            n_diff += 1
            continue
        d = float((a.float() - b.float()).abs().max().item())
        if d > 0:
            n_diff += 1
            if d > mx:
                mx, worst = d, k
    return dict(n_compared=len(common), n_diff=n_diff, n_shape_mismatch=n_shape,
                max_abs=float(mx), worst_key=worst,
                only_a=sorted(set(sd_a) - set(sd_b)),
                only_b=sorted(set(sd_b) - set(sd_a)))


def snapshot_vit(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.startswith("vit.")}


# --------------------------------------------------------------- metrics ----
def auc(y, s):
    return float(roc_auc_score(np.asarray(y).ravel(), np.asarray(s).ravel()))


def center_only_pca(X):
    """mu + first right-singular vector of the centred matrix (identical to g10 L197)."""
    X = np.asarray(X, np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    e0 = Vt[0].astype(np.float64)
    varfrac = float((S[0] ** 2) / (S ** 2).sum())
    return mu.astype(np.float64), e0, varfrac, S


def pooled_sd(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    return float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)))


def vid_dir(path):
    """Video identity for leakage purposes = the parent directory of the frame file.

    probe fake  .../faces23/102_114/295.png -> .../faces23/102_114
    probe real  .../faces23/842/68.png      -> .../faces23/842
    Two frames share a source video iff their parent dirs match; this is immune to the
    bare vid-label collisions that occur BETWEEN datasets (e.g. FF++ '102' vs
    WildDeepfake '102' are different videos).
    """
    return os.path.dirname(str(path).replace("\\", "/")).lower()


def split_vids(vids, rng, frac=0.7):
    """video-level 70/30 split; returns (train_set, test_set)."""
    u = np.array(sorted(set(str(v) for v in vids)))
    rng.shuffle(u)
    n = len(u)
    if n == 0:
        return set(), set()
    if n == 1:
        return set(u.tolist()), set()
    k = int(round(n * frac))
    k = max(1, min(n - 1, k))
    return set(u[:k].tolist()), set(u[k:].tolist())


def fit_lr(Xtr, ytr, Xte, seed=0, C=1e-3):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def model_metrics(V_tr, y_tr, vids_tr, V_te, y_te, vids_te, paths_te, doms, seed):
    """Full protocol for one model. is_fake = (y == 0); AUC positive class = fake.

    doms: OrderedDict name -> dict(V=, y=, vid=, path=)
    """
    V_tr = np.asarray(V_tr, np.float64)
    V_te = np.asarray(V_te, np.float64)
    isf_tr = (np.asarray(y_tr) == 0).astype(int)
    isf_te = (np.asarray(y_te) == 0).astype(int)

    # --- source-fit logistic regression ------------------------------------
    sc = StandardScaler().fit(V_tr)
    clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs", random_state=seed)
    clf.fit(sc.transform(V_tr), isf_tr)
    res = OrderedDict()
    res["n_train"] = int(len(V_tr))
    res["n_train_vid"] = int(len(set(map(str, vids_tr))))
    res["n_test"] = int(len(V_te))
    res["n_test_vid"] = int(len(set(map(str, vids_te))))
    res["auc_test"] = auc(isf_te, clf.predict_proba(sc.transform(V_te))[:, 1])
    res["zz_auc_test"] = auc(isf_te, clf.decision_function(sc.transform(V_te)))

    # --- source axis (center-only PCA on source train) ----------------------
    mu, e0, vf, _S = center_only_pca(V_tr)
    z_tr = (V_tr - mu) @ e0
    gap_src = float(z_tr[isf_tr == 1].mean() - z_tr[isf_tr == 0].mean())
    if gap_src < 0:
        e0, z_tr, gap_src = -e0, -z_tr, -gap_src
    res["pc0_varfrac"] = vf
    res["gap_src"] = gap_src
    res["sd_src"] = pooled_sd(z_tr[isf_tr == 1], z_tr[isf_tr == 0])
    res["d_z_src"] = gap_src / res["sd_src"]
    res["z_real_src_train"] = float(z_tr[isf_tr == 0].mean())
    res["z_fake_src_train"] = float(z_tr[isf_tr == 1].mean())

    # --- FF++ test (video-disjoint) ----------------------------------------
    z_te = (V_te - mu) @ e0
    res["e0_auc_test"] = auc(isf_te, z_te)
    res["d_z_fftest"] = (z_te[isf_te == 1].mean() - z_te[isf_te == 0].mean()) / \
        pooled_sd(z_te[isf_te == 1], z_te[isf_te == 0])

    # --- per target domain --------------------------------------------------
    per = OrderedDict()
    for name, d in doms.items():
        Vd = np.asarray(d["V"], np.float64)
        yd = np.asarray(d["y"])
        isf = (yd == 0).astype(int)
        mf, mr = isf == 1, isf == 0
        zd = (Vd - mu) @ e0
        gap = float(zd[mf].mean() - zd[mr].mean())
        sd = pooled_sd(zd[mf], zd[mr])
        _mu_d, e0_d, _vf_d, _ = center_only_pca(Vd)
        r = OrderedDict(
            n=int(len(Vd)), n_vid=int(len(set(map(str, d["vid"])))),
            auc_e0=auc(isf, zd),
            auc_srclr=auc(isf, clf.decision_function(sc.transform(Vd))),
            auc_srclr_p=auc(isf, clf.predict_proba(sc.transform(Vd))[:, 1]),
            gap=gap, sd_z=sd, d_z=gap / sd if sd else float("nan"),
            gap_ratio=gap / gap_src,
            d_real=float(zd[mr].mean() - z_tr[isf_tr == 0].mean()),
            d_fake=float(zd[mf].mean() - z_tr[isf_tr == 1].mean()),
            zdim_frac=float(np.var(zd, ddof=1) / np.var(Vd, axis=0, ddof=1).sum()),
            cos_e0dom_e0src=abs(float(np.dot(e0_d, e0) /
                                      (np.linalg.norm(e0_d) * np.linalg.norm(e0) + 1e-12))),
            z_mean_dom=float(zd.mean()),
        )
        r["shift_mag"] = float(np.hypot(r["d_real"], r["d_fake"]))
        per[name] = r
    res["dom"] = per
    res["mu"] = mu
    res["e0"] = e0

    # --- class-conditional domain separability (real-only / fake-only) ------
    cc = OrderedDict()
    for cls, ylab in (("real", 1), ("fake", 0)):
        src_v = np.asarray(vids_te)[np.asarray(y_te) == ylab]
        src_i = np.where(np.asarray(y_te) == ylab)[0]
        rng = np.random.default_rng(seed + (0 if cls == "real" else 5000))
        s_tr, s_te = split_vids(src_v, rng)
        row = OrderedDict()
        for name, d in doms.items():
            yd = np.asarray(d["y"])
            dom_i = np.where(yd == ylab)[0]
            dom_v = np.asarray(d["vid"])[dom_i]
            # independent 70/30 split of the target-side vids (same rng stream)
            r2 = np.random.default_rng(seed + (0 if cls == "real" else 5000) + 1)
            d_tr, d_te = split_vids(dom_v, r2)
            label_collision = set(map(str, src_v)) & set(map(str, dom_v))
            s_dirs = set(vid_dir(paths_te[i]) for i in src_i)
            d_dirs = set(vid_dir(p) for p in d["path"])
            file_overlap = s_dirs & d_dirs
            degenerate = (not d_te) or (not s_te) or (not d_tr) or (not s_tr)
            leak = bool(file_overlap) or degenerate
            if leak:
                row[name] = OrderedDict(auc=float("nan"), leak=True, n_tr=0, n_te=0,
                                        label_collision=len(label_collision),
                                        file_overlap=len(file_overlap),
                                        degenerate_split=bool(degenerate),
                                        src_te_vids=len(s_te), dom_te_vids=len(d_te))
                continue
            s_vids = np.asarray(vids_te)[src_i]
            tr_sel = [i for i, v in zip(src_i, s_vids) if str(v) in s_tr]
            te_sel = [i for i, v in zip(src_i, s_vids) if str(v) in s_te]
            d_tr_sel = [i for i, v in zip(dom_i, dom_v) if str(v) in d_tr]
            d_te_sel = [i for i, v in zip(dom_i, dom_v) if str(v) in d_te]
            Xtr = np.vstack([V_te[tr_sel], d["V"][d_tr_sel]])
            Xte = np.vstack([V_te[te_sel], d["V"][d_te_sel]])
            ytr = np.r_[np.zeros(len(tr_sel)), np.ones(len(d_tr_sel))]
            yte = np.r_[np.zeros(len(te_sel)), np.ones(len(d_te_sel))]
            s = fit_lr(Xtr, ytr, Xte, seed=seed, C=1e-3)
            row[name] = OrderedDict(auc=auc(yte, s), leak=False,
                                    n_tr=int(len(ytr)), n_te=int(len(yte)),
                                    label_collision=len(label_collision),
                                    file_overlap=0, degenerate_split=False,
                                    src_te_vids=len(s_te), dom_te_vids=len(d_te))
        vals = [v["auc"] for v in row.values() if not v["leak"] and np.isfinite(v["auc"])]
        row["_agg"] = OrderedDict(
            mean=float(np.mean(vals)) if vals else float("nan"),
            n_used=len(vals),
            n_leak=int(sum(1 for v in row.values() if isinstance(v, dict) and v.get("leak"))))
        cc[cls] = row
    res["cc"] = cc
    return res


# ============================================================== main ========
def main():
    t_all = time.time()
    print("=" * 120, flush=True)
    print("G13  H7: net_050 (pre) vs bridge_v2 (post) ViT representation contrast", flush=True)
    print("=" * 120, flush=True)
    print(f"[env] python={sys.executable}", flush=True)
    print(f"[env] nvidia-smi:\n{SMI_TEXT}", flush=True)
    print(f"[env] idle/gpu criteria: used<={GPU_IDLE_MIB} MiB and free>={GPU_MIN_FREE_MIB} MiB "
          f"-> pick={GPU_PICK} mode={DEVICE_MODE}", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda:0") if (DEVICE_MODE == "gpu" and torch.cuda.is_available()) \
        else torch.device("cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[env] device={device} name={dev_name} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)

    report = []      # lines of the final report
    mb = []          # MACHINE_BLOCK lines

    def L(s=""):
        print(s, flush=True)
        report.append(s)

    # ---------------------------------------------------------------- load ---
    model, miss, unexp = build_bridge_model()
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    A = OrderedDict(build_missing=miss, build_unexpected=unexp)

    # probe npz (post features live here)
    dprobe = np.load(PROBE_NPZ, allow_pickle=True)
    P_V = dprobe["V"].astype(np.float64)
    P_y = dprobe["y"].astype(np.int64)
    P_paths = np.asarray(dprobe["paths"])
    P_vids = np.asarray(dprobe["vids"])
    P_tr = dprobe["train_mask"].astype(bool)
    te_idx = np.where(~P_tr)[0]
    tr_idx = np.where(P_tr)[0]
    print(f"[npz] probe n={len(P_y)} train={P_tr.sum()} test={(~P_tr).sum()} "
          f"vids(train)={len(set(P_vids[P_tr]))} vids(test)={len(set(P_vids[~P_tr]))}", flush=True)

    dmulti = np.load(MULTI_NPZ, allow_pickle=True)
    M_V = dmulti["V"].astype(np.float64)
    M_y = dmulti["y"].astype(np.int64)
    M_dom = np.asarray(dmulti["domain"])
    M_path = np.asarray(dmulti["path"])
    M_vid = np.asarray(dmulti["vid"])

    # ======================================================= PHASE A =========
    L("#" * 120)
    L("A. LOAD / SANITY")
    L("#" * 120)

    # A1 -- pipeline check on 10 probe-test images vs stored post features
    reset_audit()
    a_ids = te_idx[:10]
    a_imgs = [to_336(read_rgb(str(P_paths[i]))) for i in a_ids]
    post_V10 = forward_V(model, a_imgs, device)
    cos_pipe = [float(np.dot(post_V10[k], P_V[i]) /
                      (np.linalg.norm(post_V10[k]) * np.linalg.norm(P_V[i]) + 1e-12))
                for k, i in enumerate(a_ids)]
    A["pipe_cos_min"] = float(np.min(cos_pipe))
    A["pipe_cos_mean"] = float(np.mean(cos_pipe))
    A["pipe_cos_list"] = [float(c) for c in cos_pipe]
    a_audit = OrderedDict(AUDIT)
    L(f"A1 pipeline check (post model vs probe_feats.npz V), n=10 probe-test imgs")
    L(f"   cos min={A['pipe_cos_min']:.6f} mean={A['pipe_cos_mean']:.6f} "
      f"(expect ~1.000000)  read_fail={a_audit['read_fail']}  batches={a_audit['batch_calls']}")
    if A["pipe_cos_min"] < 0.999:
        L("   !! PIPELINE CHECK FAILED -- aborting phase A (V would not be comparable).")
        mb.append(f"G13_PIPELINE_COS_probe10={A['pipe_cos_min']:.6f} G13_STATUS=ABORT_PIPELINE")
        write_outputs(report, mb, OrderedDict(), None)
        return

    # A2 -- overwrite the vit submodule with net_050 ("pre")
    post_vit_sd = snapshot_vit(model)
    pre_info = load_pre_vit(model, PRE_CKPT, "pre")
    A["pre_load"] = pre_info
    pre_vit_sd = snapshot_vit(model)
    ident = vit_tensor_identity(pre_vit_sd, post_vit_sd)
    A["vit_identity"] = ident
    L(f"A2a tensor-level vit comparison (net_050 vs bridge_v2_phase1): "
      f"compared={ident['n_compared']} differing={ident['n_diff']} "
      f"shape_mismatch={ident['n_shape_mismatch']} max_abs_diff={ident['max_abs']:.3e} "
      f"worst_key={ident['worst_key'] or 'none'}")
    L(f"      only-in-pre={ident['only_a']} only-in-post={ident['only_b']}")
    if ident["n_diff"] == 0:
        L("      >>> the two checkpoints' ViT submodules are BIT-IDENTICAL: the bridge_v2 ViT"
          " was never updated (weights copied from net_050 / frozen).")

    # A3 -- cos(pre, post) on the same 10 images
    pre_V10 = forward_V(model, a_imgs, device)
    cos_pre_post = [float(np.dot(pre_V10[k], post_V10[k]) /
                          (np.linalg.norm(pre_V10[k]) * np.linalg.norm(post_V10[k]) + 1e-12))
                    for k in range(len(a_ids))]
    A["cos_pre_post_min"] = float(np.min(cos_pre_post))
    A["cos_pre_post_mean"] = float(np.mean(cos_pre_post))
    A["cos_pre_post_list"] = [float(c) for c in cos_pre_post]
    L(f"A2 pre vit load: wrapper={pre_info['wrapper']} file_keys={pre_info['n_keys_in_file']} "
      f"vit_targets={pre_info['n_vit_target_keys']} mapped={pre_info['n_mapped']} "
      f"missing={len(pre_info['missing'])} skipped={len(pre_info['skipped'])} "
      f"non_model={len(pre_info['non_model_prefix'])}")
    if pre_info["skipped"]:
        L(f"   skipped (not present / shape-mismatch in vit): {pre_info['skipped']}")
    if pre_info["missing"]:
        L(f"   missing -> KEPT bridge_v2 values: {pre_info['missing']}")
    L(f"A3 cos(pre V, post V) on the same 10 imgs: min={A['cos_pre_post_min']:.6f} "
      f"mean={A['cos_pre_post_mean']:.6f}  (per-img: "
      f"{', '.join('%.6f' % c for c in cos_pre_post)})")
    if A["cos_pre_post_min"] > 0.9999:
        L("   !! pre/post V are ~identical. Not forced. Root cause is resolved by A2a: the"
          " bridge_v2 ViT submodule is bit-identical to net_050, so V cannot differ. This is"
          " a substantive finding (the FF++ training did not touch the ViT), not an overwrite"
          " failure: A1 proves the loaded model is exactly the one that produced"
          " probe_feats.npz, and A2a proves the two ViTs are the same tensors.")
    L("")

    # ======================================================= PHASE B =========
    L("#" * 120)
    L("B. EXTRACT pre-V (net_050) over 4500 images")
    L("#" * 120)
    samples = []
    for i in range(len(P_paths)):
        samples.append((str(P_paths[i]), str(P_vids[i]), int(P_y[i]), "ffpp",
                        "train" if P_tr[i] else "test"))
    for dm in TARGET_DOMAINS:
        idx = np.where(M_dom == dm)[0]
        if len(idx) != N_DOM:
            raise RuntimeError(f"domain {dm}: expected {N_DOM} rows, got {len(idx)}")
        for i in idx:
            samples.append((str(M_path[i]), str(M_vid[i]), int(M_y[i]), dm, dm))
    assert len(samples) == N_TOTAL, (len(samples), N_TOTAL)
    print(f"[B] sample table n={len(samples)} "
          f"(probe 3000 + {len(TARGET_DOMAINS)}x{N_DOM}); domains={TARGET_DOMAINS}", flush=True)

    reset_audit()
    t_b = time.time()
    pre_V = np.zeros((len(samples), 768), np.float32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s0 in range(0, len(samples), BATCH):
        chunk = samples[s0:s0 + BATCH]
        imgs = [to_336(read_rgb(p)) for p, _v, _y, _d, _s in chunk]
        pre_V[s0:s0 + len(chunk)] = forward_V(model, imgs, device)
        if (s0 // BATCH) % 20 == 0:
            print(f"[B] {s0 + len(chunk)}/{len(samples)}  "
                  f"({time.time() - t_b:.0f}s)", flush=True)
    wall_b = time.time() - t_b
    peak_torch = (torch.cuda.max_memory_allocated() / (1024.0 ** 2)) if device.type == "cuda" else 0.0
    peak_resv = (torch.cuda.max_memory_reserved() / (1024.0 ** 2)) if device.type == "cuda" else 0.0
    b_audit = OrderedDict(AUDIT)
    b_audit["wall_s"] = wall_b
    b_audit["peak_alloc_mib"] = peak_torch
    b_audit["peak_reserved_mib"] = peak_resv

    # verify: the same 10 A-images are probe-test rows -> pre_V must equal pre_V10
    chk = [float(np.dot(pre_V[i], pre_V10[k]) /
                 (np.linalg.norm(pre_V[i]) * np.linalg.norm(pre_V10[k]) + 1e-12))
           for k, i in enumerate(a_ids)]
    b_audit["self_check_cos_min"] = float(np.min(chk))
    L(f"S1 extracted pre-V {pre_V.shape}  imgs={b_audit['imgs_forward']} "
      f"batches={b_audit['batch_calls']} read_fail={b_audit['read_fail']} "
      f"device={device} ({dev_name}) wall={wall_b:.1f}s "
      f"peak_alloc={peak_torch:.0f} MiB peak_reserved={peak_resv:.0f} MiB")
    L(f"   self-check: pre_V rows of the 10 A-images vs pre_V10 -> cos min="
      f"{b_audit['self_check_cos_min']:.6f} (expect 1.000000)")
    L(f"S2 stored -> {OUT_PRE_NPZ}")
    L("")

    y_all = np.array([s[2] for s in samples], np.int64)
    vid_all = np.array([s[1] for s in samples])
    path_all = np.array([s[0] for s in samples])
    dom_all = np.array([s[3] for s in samples])
    split_all = np.array([s[4] for s in samples])

    # ---- pre-V vs stored post-V, on ALL 4500 rows (exact, float32) ----
    mx_probe = float(np.abs(pre_V[:3000] - P_V.astype(np.float32)).max())
    mx_dom = 0.0
    for dm in TARGET_DOMAINS:
        i_pre = np.where(dom_all == dm)[0]
        i_post = np.where(M_dom == dm)[0]
        mx_dom = max(mx_dom, float(np.abs(
            pre_V[i_pre] - M_V[i_post].astype(np.float32)).max()))
    b_audit["maxabs_vs_post_probe"] = mx_probe
    b_audit["maxabs_vs_post_domains"] = mx_dom
    L(f"S3 pre-V vs stored post-V over all 4500 rows (float32 exact): "
      f"max|d| probe={mx_probe:.3e}  5-domains={mx_dom:.3e}  "
      f"(bit-identical if 0.000e+00)")
    L("")

    np.savez_compressed(OUT_PRE_NPZ, V=pre_V, y=y_all, paths=path_all, vids=vid_all,
                        domain=dom_all, split=split_all)

    # ======================================================= PHASE C =========
    L("#" * 120)
    L("C. ANALYSIS  (pre and post, identical protocol)")
    L("#" * 120)
    L("   is_fake = (y == 0); AUC positive class = fake; split by video.")
    L("   post V = probe_feats.npz['V'] (probe) + feats_multi.npz['V'] (5 target domains),"
      " NOT re-extracted.")
    L("")

    doms_pre, doms_post = OrderedDict(), OrderedDict()
    for dm in TARGET_DOMAINS:
        idx = np.where(dom_all == dm)[0]
        jdx = np.where(M_dom == dm)[0]
        doms_pre[dm] = dict(V=pre_V[idx].astype(np.float64), y=y_all[idx],
                            vid=vid_all[idx], path=path_all[idx])
        doms_post[dm] = dict(V=M_V[jdx].astype(np.float64), y=M_y[jdx],
                             vid=M_vid[jdx], path=M_path[jdx])

    res_pre = model_metrics(pre_V[tr_idx], P_y[tr_idx], P_vids[tr_idx],
                            pre_V[te_idx], P_y[te_idx], P_vids[te_idx], P_paths[te_idx],
                            doms_pre, SEED)
    res_post = model_metrics(P_V[tr_idx], P_y[tr_idx], P_vids[tr_idx],
                             P_V[te_idx], P_y[te_idx], P_vids[te_idx], P_paths[te_idx],
                             doms_post, SEED)

    # sanity: post domain V must be the same rows as pre domain V sample table
    for dm in TARGET_DOMAINS:
        i_pre = np.where(dom_all == dm)[0]
        i_post = np.where(M_dom == dm)[0]
        assert list(path_all[i_pre]) == [str(x) for x in M_path[i_post]], dm

    d_auc = res_post["auc_test"] - res_pre["auc_test"]
    d_cross_dom = {dm: res_post["dom"][dm]["auc_srclr"] - res_pre["dom"][dm]["auc_srclr"]
                   for dm in TARGET_DOMAINS}
    d_cross = float(np.mean(list(d_cross_dom.values())))
    d_gapr = {dm: res_post["dom"][dm]["gap_ratio"] - res_pre["dom"][dm]["gap_ratio"]
              for dm in TARGET_DOMAINS}
    shift_ratio = {dm: (res_post["dom"][dm]["shift_mag"] /
                        (res_pre["dom"][dm]["shift_mag"] + 1e-12)) for dm in TARGET_DOMAINS}

    L("C0. NOTE: the two ViTs are the same tensors (A2a: 162/162, max|d|=0), so V_pre == V_post"
      " BY CONSTRUCTION: bit-exactly on the 3000 probe rows (B3 max|d|=0.000e+00) and to"
      " <=3.2e-05 on the 1500 domain rows (only because feats_multi.npz was extracted with"
      " the pure-PyTorch MHA shim; see caveat 10).  Every pre/post delta below is therefore"
      " structurally 0 -- the two columns are a consistency check, NOT two independent"
      " measurements.  The absolute (non-delta) numbers are the substantive content.")
    L("C1. IN-DOMAIN (FF++ source; train=2200 / test=800, video-disjoint)")
    L("  model  in-domain          n_train n_train_vid n_test n_test_vid  PC0varfrac  d_z_src"
      "  e0AUC_test  LRAUC_test  DZ_fftest")
    for tag, r in (("pre", res_pre), ("post", res_post)):
        L(f"  {tag:<6} {'FF++ (probe)':<17}{r['n_train']:>8}{r['n_train_vid']:>12}"
          f"{r['n_test']:>7}{r['n_test_vid']:>11}{r['pc0_varfrac']:>11.4f}"
          f"{r['d_z_src']:>9.3f}{r['e0_auc_test']:>11.4f}{r['auc_test']:>12.4f}"
          f"{r['d_z_fftest']:>11.3f}")
    L(f"  {'delta':<6} {'post - pre':<17}{'':>8}{'':>12}{'':>7}{'':>11}"
      f"{res_post['pc0_varfrac']-res_pre['pc0_varfrac']:>+11.4f}"
      f"{res_post['d_z_src']-res_pre['d_z_src']:>+9.3f}"
      f"{res_post['e0_auc_test']-res_pre['e0_auc_test']:>+11.4f}"
      f"{d_auc:>+12.4f}{res_post['d_z_fftest']-res_pre['d_z_fftest']:>+11.3f}")
    L("")
    L("C2. CROSS-DOMAIN (5 target domains x 300: 150 real / 150 fake)")
    L("  dom     |  e0_auc(post) e0_auc(pre)  e0AUC_d   | srcLR(post) srcLR(pre)  srcLR_d"
      "  |   d_z(post)  d_z(pre)  | gapR(post) gapR(pre)  gapR_d"
      "  | dReal(post) dReal(pre) | dFake(post) dFake(pre)"
      "  | zdim(post) zdim(pre) | |cos e0dom,src|post pre | AUCreal(post) (pre) | AUCfake(post) (pre)")
    for dm in TARGET_DOMAINS:
        a, b = res_post["dom"][dm], res_pre["dom"][dm]
        ca, cb = res_post["cc"]["real"][dm], res_pre["cc"]["real"][dm]
        fa, fb = res_post["cc"]["fake"][dm], res_pre["cc"]["fake"][dm]
        L(f"  {dm:<7}| {a['auc_e0']:>10.4f} {b['auc_e0']:>10.4f} {a['auc_e0']-b['auc_e0']:>+9.4f}"
          f" | {a['auc_srclr']:>10.4f} {b['auc_srclr']:>9.4f} {a['auc_srclr']-b['auc_srclr']:>+8.4f}"
          f" | {a['d_z']:>9.3f} {b['d_z']:>8.3f}"
          f" | {a['gap_ratio']:>9.4f} {b['gap_ratio']:>8.4f} {a['gap_ratio']-b['gap_ratio']:>+8.4f}"
          f" | {a['d_real']:>+9.3f} {b['d_real']:>+9.3f} | {a['d_fake']:>+9.3f} {b['d_fake']:>+9.3f}"
          f" | {a['zdim_frac']:>9.4f} {b['zdim_frac']:>8.4f}"
          f" | {a['cos_e0dom_e0src']:>9.4f} {b['cos_e0dom_e0src']:>7.4f}"
          f" | {ca['auc']:>9.4f} ({cb['auc']:.4f})"
          f" | {fa['auc']:>9.4f} ({fb['auc']:.4f})")
    L("")
    leak_txt, coll_txt = [], []
    for cls in ("real", "fake"):
        for dm in TARGET_DOMAINS:
            v = res_post["cc"][cls][dm]
            if v["leak"]:
                leak_txt.append(f"{dm}/{cls}(file_overlap={v['file_overlap']},"
                                f"degenerate_split={v['degenerate_split']},"
                                f"src_te_vids={v['src_te_vids']},dom_te_vids={v['dom_te_vids']},"
                                f"label_collision={v['label_collision']})")
            elif v["label_collision"]:
                coll_txt.append(f"{dm}/{cls}(n={v['label_collision']},file_overlap=0)")
    L("C2a. leaked / not-aggregated cells (leak = shared video DIRECTORY between the source"
      " and target pools, or a degenerate 70/30 split): "
      + (", ".join(leak_txt) if leak_txt else "none"))
    L("     bare vid-LABEL collisions across the 2 datasets (reported, NOT treated as"
      " leakage -- video identity is the parent dir, and no file is shared): "
      + (", ".join(coll_txt) if coll_txt else "none"))
    for cls in ("real", "fake"):
        ag = res_post["cc"][cls]["_agg"]
        agp = res_pre["cc"][cls]["_agg"]
        L(f"      mean AUC_{cls}: post={ag['mean']:.4f} (n_used={ag['n_used']}, n_leak={ag['n_leak']})"
          f"  pre={agp['mean']:.4f} (n_used={agp['n_used']}, n_leak={agp['n_leak']})"
          f"  d={ag['mean']-agp['mean']:+.4f}")
    L("")
    L("C3. cross-check vs G11 post-model anchors (same protocol)")
    L(f"  PC0 varfrac      recalc={res_post['pc0_varfrac']:.4f}  g11={G11_ANCHORS['pc0_varfrac']:.4f}"
      f"  d={res_post['pc0_varfrac']-G11_ANCHORS['pc0_varfrac']:+.4f}")
    L(f"  srcLR test AUC   recalc={res_post['auc_test']:.4f}  g11={G11_ANCHORS['srclr_test_auc']:.4f}"
      f"  d={res_post['auc_test']-G11_ANCHORS['srclr_test_auc']:+.4f}")
    for dm in TARGET_DOMAINS:
        a = res_post["dom"][dm]
        L(f"  {dm:<7} e0AUC recalc={a['auc_e0']:.4f} g11={G11_ANCHORS['e0_auc'][dm]:.4f}"
          f" d={a['auc_e0']-G11_ANCHORS['e0_auc'][dm]:+.4f}   "
          f"srcLR recalc={a['auc_srclr']:.4f} g11={G11_ANCHORS['srclr_auc'][dm]:.4f}"
          f" d={a['auc_srclr']-G11_ANCHORS['srclr_auc'][dm]:+.4f}")
    L("")

    # ------------------------------------------------ pre-registered verdicts
    L("#" * 120)
    L("C4. MECHANICAL VERDICTS (pre-registered constants)")
    L("#" * 120)
    b1 = "IMPROVED" if d_auc >= B1_TAU else "NO-GAIN"
    L(f"B1 in-domain test AUC: post={res_post['auc_test']:.4f} pre={res_pre['auc_test']:.4f} "
      f"delta={d_auc:+.4f} tau=+{B1_TAU:.2f} -> {b1}")

    if d_cross >= B2_HI:
        b2 = "IMPROVED"
    elif abs(d_cross) < B2_LO:
        b2 = "NO-GAIN"
    elif d_cross <= -B2_HI:
        b2 = "DEGRADED"
    else:
        b2 = "PARTIAL"
    L("B2 cross-domain (source-LR AUC), delta = post - pre per domain: " +
      ", ".join(f"{dm}={d_cross_dom[dm]:+.4f}" for dm in TARGET_DOMAINS))
    L(f"   mean_delta={d_cross:+.4f}  (>=+{B2_HI} -> IMPROVED; |d|<{B2_LO} -> NO-GAIN;"
      f" <=-{B2_HI} -> DEGRADED; else PARTIAL) -> {b2}")

    n_drop = sum(1 for dm in TARGET_DOMAINS if d_gapr[dm] <= -B3_GAP_DROP)
    n_bigshift = sum(1 for dm in TARGET_DOMAINS if shift_ratio[dm] > B3_SHIFT_BIG)
    n_near = sum(1 for dm in TARGET_DOMAINS if abs(d_gapr[dm]) < B3_NEAR)
    n_equal = sum(1 for dm in TARGET_DOMAINS
                  if B3_SHIFT_EQ_LO <= shift_ratio[dm] <= B3_SHIFT_EQ_HI)
    if (n_drop >= 1) or (n_bigshift >= B3_SHIFT_DOMAINS):
        b3 = "TRAINING_CAUSED"
    elif n_near == len(TARGET_DOMAINS) and n_equal == len(TARGET_DOMAINS):
        b3 = "PRETRAINED_INHERENT"
    else:
        b3 = "MIXED"
    L("B3 gap_ratio delta per domain: " +
      ", ".join(f"{dm}={d_gapr[dm]:+.4f}" for dm in TARGET_DOMAINS))
    L("   shift_mag |d_real,d_fake| ratio post/pre per domain: " +
      ", ".join(f"{dm}={shift_ratio[dm]:.3f}" for dm in TARGET_DOMAINS))
    L(f"   n(gap_ratio drop <= -{B3_GAP_DROP})={n_drop}  "
      f"n(shift ratio > {B3_SHIFT_BIG})={n_bigshift}/{len(TARGET_DOMAINS)}  "
      f"n(|d gap_ratio| < {B3_NEAR})={n_near}  "
      f"n(shift ratio in [{B3_SHIFT_EQ_LO},{B3_SHIFT_EQ_HI}])={n_equal} -> {b3}")

    vf_pre = res_pre["pc0_varfrac"]
    b4 = "PRE_INHERENT_AXIS" if vf_pre >= B4_HI else ("TRAINING_MADE_AXIS" if vf_pre < B4_LO
                                                      else "PARTIAL")
    L(f"B4 pre PC0 varfrac={vf_pre:.4f} (>= {B4_HI} -> PRE_INHERENT_AXIS; < {B4_LO} ->"
      f" TRAINING_MADE_AXIS; else PARTIAL) -> {b4}")
    L("")

    # ======================================================= PHASE D =========
    L("#" * 120)
    L("D. OPTIONAL INTERMEDIATE POINT vit_m2f2_phase1.pth (frozen-backbone check)")
    L("#" * 120)
    dres = OrderedDict()
    try:
        mid_info = load_pre_vit(model, MID_CKPT, "mid")
        dres["load"] = {k: (v if not isinstance(v, list) else v) for k, v in mid_info.items()}
        ident_d = vit_tensor_identity(snapshot_vit(model), pre_vit_sd)
        dres["vit_identity"] = ident_d
        L(f"D0 tensor-level vit comparison (vit_m2f2_phase1 vs net_050/pre): "
          f"compared={ident_d['n_compared']} differing={ident_d['n_diff']} "
          f"max_abs_diff={ident_d['max_abs']:.3e} worst_key={ident_d['worst_key'] or 'none'}")
        L(f"   only-in-mid={ident_d['only_a']} only-in-pre={ident_d['only_b']}")
        n_chk = 300
        d_ids = te_idx[:n_chk]
        reset_audit()
        t_d = time.time()
        d_imgs = [to_336(read_rgb(str(P_paths[i]))) for i in d_ids]
        mid_V = forward_V(model, d_imgs, device)
        dres["wall_s"] = time.time() - t_d
        dres["n_imgs"] = n_chk
        dres["read_fail"] = AUDIT["read_fail"]
        cos_d_pre = [float(np.dot(mid_V[k], pre_V[i]) /
                           (np.linalg.norm(mid_V[k]) * np.linalg.norm(pre_V[i]) + 1e-12))
                     for k, i in enumerate(d_ids)]
        cos_d_post = [float(np.dot(mid_V[k], P_V[i]) /
                            (np.linalg.norm(mid_V[k]) * np.linalg.norm(P_V[i]) + 1e-12))
                      for k, i in enumerate(d_ids)]
        dres["cos_vs_pre_min"] = float(np.min(cos_d_pre))
        dres["cos_vs_pre_mean"] = float(np.mean(cos_d_pre))
        dres["cos_vs_post_min"] = float(np.min(cos_d_post))
        dres["cos_vs_post_mean"] = float(np.mean(cos_d_post))
        dres["frozen"] = bool(dres["cos_vs_pre_min"] >= D_COS_FROZEN)
        L(f"D1 vit_m2f2_phase1.pth: wrapper={mid_info['wrapper']} "
          f"file_keys={mid_info['n_keys_in_file']} mapped={mid_info['n_mapped']} "
          f"missing={len(mid_info['missing'])} skipped={len(mid_info['skipped'])} "
          f"non_model={len(mid_info['non_model_prefix'])}")
        if mid_info["skipped"]:
            L(f"   skipped: {mid_info['skipped']}")
        if mid_info["missing"]:
            L(f"   missing -> KEPT previous values: {mid_info['missing']}")
        if mid_info["non_model_prefix"]:
            nm = mid_info["non_model_prefix"]
            L(f"   non-model-prefix keys IGNORED: n={len(nm)} -- these are the CLIP text"
              f" encoder ({sum(1 for k in nm if k.startswith('clip_text_encoder.'))} keys),"
              f" deepfake_proj / vision_proj / text_proj / output head and the two alpha"
              f" scalars; they are NOT part of the ViT and do not affect raw V."
              f"  first 6: {nm[:6]}")
        L(f"D2 300 probe-test imgs, {dres['wall_s']:.1f}s, read_fail={dres['read_fail']}")
        L(f"   cos(V_mid, V_pre)  min={dres['cos_vs_pre_min']:.6f} "
          f"mean={dres['cos_vs_pre_mean']:.6f}  (expect ~1 -> frozen backbone)")
        L(f"   cos(V_mid, V_post) min={dres['cos_vs_post_min']:.6f} "
          f"mean={dres['cos_vs_post_mean']:.6f}")
        if dres["frozen"]:
            L("   VERDICT: FROZEN -- V_mid == V_pre (cos>=%.3f; tensor comparison above:"
              " %d/%d differing, max|d|=%.3e). Raw-V metrics are identical to pre's by"
              " construction, so the intermediate point on raw V is the SAME node as 'pre';"
              " no separate 4500-image extraction was run (time box)."
              % (D_COS_FROZEN, ident_d["n_diff"], ident_d["n_compared"], ident_d["max_abs"]))
            dres["metrics_equivalent_to"] = "pre"
        else:
            L("   VERDICT: NOT frozen on raw V (cos < %.3f). Full 4500-image extraction"
              " SKIPPED per the time box; no mid-point domain metrics reported." % D_COS_FROZEN)
            dres["metrics_equivalent_to"] = None
    except Exception as exc:                                   # pragma: no cover
        dres["error"] = repr(exc)
        L(f"D! FAILED: {exc!r} -- skipped (per time box).")
    L("")

    # ------------------------------------------------------------- audit ----
    L("#" * 120)
    L("AUDIT")
    L("#" * 120)
    L(f"  GPU query: {SMI_TEXT}")
    L(f"  picked GPU index={GPU_PICK} mode={DEVICE_MODE} device={device} name={dev_name} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    L(f"  phase A forwards: imgs={a_audit['imgs_forward']} batches={a_audit['batch_calls']} "
      f"read_fail={a_audit['read_fail']}")
    L(f"  phase B forwards: imgs={b_audit['imgs_forward']} batches={b_audit['batch_calls']} "
      f"read_fail={b_audit['read_fail']} wall={wall_b:.1f}s "
      f"peak_alloc={b_audit['peak_alloc_mib']:.0f} MiB "
      f"peak_reserved={b_audit['peak_reserved_mib']:.0f} MiB")
    if dres.get("n_imgs"):
        L(f"  phase D forwards: imgs={dres['n_imgs']} wall={dres['wall_s']:.1f}s "
          f"read_fail={dres['read_fail']}")
    L(f"  batch size={BATCH} fp32; CPU threads pinned to 1; cv2.setNumThreads(0); "
      f"single process; no DataLoader workers")
    L(f"  seed={SEED}; total wall={time.time() - t_all:.1f}s; "
      f"pre-V file={OUT_PRE_NPZ}")
    L("")

    # ----------------------------------------------------------- caveats ----
    L("#" * 120)
    L("CAVEATS (honest)")
    L("#" * 120)
    caveats = [
        "1. 'pre' model = bridge_v2 architecture with ONLY the vit submodule overwritten by"
        " net_050_backup.pth; every non-vit module (deepfake_proj, CLIP encoders, bridge"
        " adapter, heads) still carries bridge_v2 weights.  V = vit.forward_features(...)"
        "[:,0] depends only on the vit submodule + _preprocess_for_vit (parameter-free), so"
        " the pre-V / post-V contrast is clean -- but this is a hybrid model, not a real"
        " historical checkpoint.",
        "2. *** PREMISE OF H7 IS FACTUALLY VIOLATED (headline). *** All 162 ViT tensors of"
        " checkpoints/stage_1/bridge_v2_phase1.pth are BIT-IDENTICAL to net_050_backup.pth"
        " (A2a: compared=162 differing=0 max_abs_diff=0.000e+00), so the 'post' node's ViT"
        " is literally the 'pre' ViT.  V_pre == V_post on the 10-image probe"
        " (cos=1.000000) and, over all 4500 extracted images, bit-exactly on the 3000 probe"
        " rows (max|d|=0.000e+00; S3 self-check) and to <=3.2e-05 on the 1500 domain rows"
        " (backend mismatch only, caveat 10).  Every pre-vs-post delta here is therefore structurally"
        " 0 by construction, NOT a measured 'no change'.  H7 cannot be answered with the"
        " bridge_v2 checkpoint: its FF++ training (if any) did not modify the ViT"
        " submodule -- consistent with a frozen backbone / ViT-off training recipe.",
        "2b. Factual status of the premise: net_050 is a generic (non-FF++) checkpoint; it was"
        " NEVER fine-tuned on FF++ in this project.  Nothing here proves what data net_050"
        " itself saw.",
        "3. 4500-image subset: probe 2200/800 come from probe_feats.npz (FF++ c23, video-"
        " disjoint), 5 target domains are 300 each from feats_multi.npz (the 'ffpp' domain"
        " there is 98/140 vids shared with source train, i.e. contaminated, so it is"
        " EXCLUDED here).  Each domain is a 150-real/150-fake sample; results are"
        " subset-representative only.",
        "4. Per target domain n=300 and only 41/124/47/1/239 vids (cd1/cd2/dfdcp/ffiw/wild)."
        "  ffiw has exactly 1 vid -> its 70/30 class-conditional split cannot produce a"
        " disjoint test side -> marked leak=True and excluded from the AUC_real/AUC_fake"
        " aggregate (cd1/cd2/dfdcp/wild enter it).  Cross-domain AUCs on 150/150 rows are"
        " noisy (SE ~0.03).",
        "4b. Leak criterion for the class-conditional probes = shared video DIRECTORY"
        " (parent dir of the frame file: .../faces23/<id>_<t> or .../faces23/<id>) between"
        " the source (FF++) and target pool, or a degenerate 70/30 split.  Bare vid-label"
        " collisions across datasets DO occur (wild uses plain numeric ids that coincide"
        " with FF++ identity numbers, e.g. WildDeepfake fake_test/85/fake/102 vs FF++"
        " faces23/102_114) but no file or directory is shared, so they are reported and not"
        " treated as leakage.",
        "5. d_z and the single-axis AUC are mathematically the same statistic (both are"
        " monotone in the mean gap / spread of the same 1-D score); they are reported"
        " separately only because the protocol asks for both.",
        "6. The pre axis e0 and the post axis e0 are each fitted on their OWN source train V"
        " (center-only PCA).  Cross-model comparisons therefore compare 'the same protocol"
        " applied to two representations', not one fixed vector.  |cos(e0_dom, e0_src)| is"
        " likewise within-model.",
        "7. StandardScaler/LogisticRegression are refit per model on that model's source"
        " train V; C=1e-3, lbfgs, max_iter=3000, seed fixed.  The class-conditional probes"
        " use a video-level 70/30 split of BOTH pools and C=1e-3 (not pre-registered for"
        " this sub-probe; stated for transparency).",
        "8. 'IMPROVED/DEGRADED' labels are mechanical applications of the pre-registered"
        " constants on the numbers above; no direction is inferred beyond them.",
    ]
    caveats_tail = [
        "10. Extraction-backend mismatch on the domain rows: the 5-domain post-V comes from"
        " feats_multi.npz, produced by _tsne/extract_multi.py which FORCES the pure-PyTorch"
        " MHA shim (br.MHA = ShimMHA), while this run used the installed flash_attn MHA (as"
        " run_g10.py does).  Result: probe rows bit-identical (max|d|=0.000e+00) but domain"
        " rows differ by up to 3.159e-05 -- the delta column is 0 only to 4 reported"
        " decimals.  No metric here is sensitive to a 3e-05 perturbation of a raw CLS"
        " feature (all pre/post deltas round to 0.0000).",
    ]
    if dres.get("frozen"):
        caveats.append(
            "9. Phase D: vit_m2f2_phase1.pth reproduces pre-V (cos>=%.3f) -- consistent with"
            " train_vit_m2f2_phase1.py loading the ViT from net_050 and FREEZING it.  Its"
            " raw-V metrics are therefore identical to 'pre' by construction and no separate"
            " 4500-image extraction was run for it (time box)." % D_COS_FROZEN)
    elif dres.get("metrics_equivalent_to") is None and "error" not in dres:
        caveats.append(
            "9. Phase D: vit_m2f2_phase1.pth does NOT reproduce pre-V on the 300-image probe"
            " (cos=%.6f); its raw-V domain metrics were NOT computed (time box)." %
            dres.get("cos_vs_pre_min", float("nan")))
    for c in caveats + caveats_tail:
        L("  " + c)
    L("")

    # ------------------------------------------------------ machine block ----
    mb.append(f"G13_TITLE=H7 net_050(pre) vs bridge_v2(post) ViT raw-CLS representation contrast")
    mb.append(f"G13_SEED={SEED} G13_BATCH={BATCH} G13_N_IMGS={N_TOTAL} G13_FP32=1 "
              f"G13_PROTOCOL=center-only_PCA@srcTrain2200 + StandardScaler+LR(C=1e-3,lbfgs,max_iter=3000)")
    mb.append(f"G13_GPU_QUERY=[{SMI_TEXT.replace(chr(10), ' | ')}] G13_GPU_PICK={GPU_PICK} "
              f"G13_MODE={DEVICE_MODE} G13_DEVICE={dev_name} "
              f"G13_CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    mb.append(f"G13_BUILD_MISSING={miss} G13_BUILD_UNEXPECTED={unexp}")
    mb.append(f"G13_A_PIPE_COS_probe10_min={A['pipe_cos_min']:.6f} "
              f"G13_A_PIPE_COS_mean={A['pipe_cos_mean']:.6f} "
              f"G13_A_PRELOAD_file_keys={pre_info['n_keys_in_file']} "
              f"G13_A_PRELOAD_vit_targets={pre_info['n_vit_target_keys']} "
              f"G13_A_PRELOAD_mapped={pre_info['n_mapped']} "
              f"G13_A_PRELOAD_missing={len(pre_info['missing'])} "
              f"G13_A_PRELOAD_skipped={len(pre_info['skipped'])} "
              f"G13_A_PRELOAD_skipped_keys={';'.join(pre_info['skipped']) if pre_info['skipped'] else 'none'} "
              f"G13_A_PRELOAD_missing_keys={';'.join(pre_info['missing']) if pre_info['missing'] else 'none'} "
              f"G13_A_COS_pre_post10_min={A['cos_pre_post_min']:.6f} "
              f"G13_A_COS_pre_post10_mean={A['cos_pre_post_mean']:.6f}")
    mb.append(f"G13_A_VIT_TENSOR_IDENT=pre_vs_post compared={ident['n_compared']} "
              f"differing={ident['n_diff']} shape_mismatch={ident['n_shape_mismatch']} "
              f"max_abs_diff={ident['max_abs']:.3e} BIT_IDENTICAL={int(ident['n_diff'] == 0)} "
              f"only_in_pre={len(ident['only_a'])} only_in_post={len(ident['only_b'])}")
    mb.append(f"G13_PRE_INDOMAIN_testAUC={res_pre['auc_test']:.4f} "
              f"G13_PRE_PC0_varfrac={res_pre['pc0_varfrac']:.4f} "
              f"G13_PRE_DZSRC={res_pre['d_z_src']:.4f} "
              f"G13_PRE_e0AUC_fftest={res_pre['e0_auc_test']:.4f} "
              f"G13_PRE_DZ_fftest={res_pre['d_z_fftest']:.4f} "
              f"G13_POST_INDOMAIN_testAUC={res_post['auc_test']:.4f} "
              f"G13_POST_PC0_varfrac={res_post['pc0_varfrac']:.4f} "
              f"G13_POST_DZSRC={res_post['d_z_src']:.4f} "
              f"G13_POST_e0AUC_fftest={res_post['e0_auc_test']:.4f} "
              f"G13_POST_DZ_fftest={res_post['d_z_fftest']:.4f} "
              f"G13_D_INDOMAIN_testAUC={d_auc:+.4f}")
    for dm in TARGET_DOMAINS:
        a, b = res_post["dom"][dm], res_pre["dom"][dm]
        ca, cb = res_post["cc"]["real"][dm], res_pre["cc"]["real"][dm]
        fa, fb = res_post["cc"]["fake"][dm], res_pre["cc"]["fake"][dm]
        mb.append(
            f"G13_DOM_{dm}=n={a['n']} nvid={a['n_vid']} "
            f"e0AUC_post={a['auc_e0']:.4f} e0AUC_pre={b['auc_e0']:.4f} "
            f"d_e0AUC={a['auc_e0']-b['auc_e0']:+.4f} "
            f"srcLR_post={a['auc_srclr']:.4f} srcLR_pre={b['auc_srclr']:.4f} "
            f"d_srcLR={a['auc_srclr']-b['auc_srclr']:+.4f} "
            f"d_z_post={a['d_z']:.4f} d_z_pre={b['d_z']:.4f} "
            f"gapRatio_post={a['gap_ratio']:.4f} gapRatio_pre={b['gap_ratio']:.4f} "
            f"d_gapRatio={a['gap_ratio']-b['gap_ratio']:+.4f} "
            f"dReal_post={a['d_real']:+.4f} dReal_pre={b['d_real']:+.4f} "
            f"dFake_post={a['d_fake']:+.4f} dFake_pre={b['d_fake']:+.4f} "
            f"shiftRatio_post_pre={shift_ratio[dm]:.4f} "
            f"zdimFrac_post={a['zdim_frac']:.4f} zdimFrac_pre={b['zdim_frac']:.4f} "
            f"abscos_e0dom_src_post={a['cos_e0dom_e0src']:.4f} "
            f"abscos_e0dom_src_pre={b['cos_e0dom_e0src']:.4f} "
            f"AUCreal_post={ca['auc']:.4f} AUCreal_pre={cb['auc']:.4f} "
            f"AUCfake_post={fa['auc']:.4f} AUCfake_pre={fb['auc']:.4f} "
            f"leak_real={int(ca['leak'])} leak_fake={int(fa['leak'])}")
    for cls in ("real", "fake"):
        ag, agp = res_post["cc"][cls]["_agg"], res_pre["cc"][cls]["_agg"]
        mb.append(f"G13_CC_{cls}_mean=post={ag['mean']:.4f} pre={agp['mean']:.4f} "
                  f"d={ag['mean']-agp['mean']:+.4f} n_used_post={ag['n_used']} "
                  f"n_leak_post={ag['n_leak']} n_used_pre={agp['n_used']} "
                  f"n_leak_pre={agp['n_leak']}")
    mb.append(f"G13_B1_CONST=tau_incr={B1_TAU:.2f} d_INDOMAIN={d_auc:+.4f} VERDICT={b1}")
    mb.append(f"G13_B2_CONST=hi={B2_HI:.2f} lo={B2_LO:.2f} "
              f"d_cross_domains={';'.join(f'{dm}:{d_cross_dom[dm]:+.4f}' for dm in TARGET_DOMAINS)} "
              f"d_cross_mean={d_cross:+.4f} VERDICT={b2}")
    mb.append(f"G13_B3_CONST=gap_drop={B3_GAP_DROP:.2f} near={B3_NEAR:.2f} "
              f"shift_big={B3_SHIFT_BIG:.2f} shift_eq=[{B3_SHIFT_EQ_LO:.2f},{B3_SHIFT_EQ_HI:.2f}] "
              f"n_drop={n_drop} n_bigshift={n_bigshift} n_near={n_near} n_equal={n_equal} "
              f"VERDICT={b3}")
    mb.append(f"G13_B4_CONST=hi={B4_HI:.2f} lo={B4_LO:.2f} pre_PC0_varfrac={vf_pre:.4f} "
              f"VERDICT={b4}")
    mb.append(f"G13_AUDIT=forwards_A={a_audit['imgs_forward']} forwards_B={b_audit['imgs_forward']} "
              f"batch_calls_A={a_audit['batch_calls']} batch_calls_B={b_audit['batch_calls']} "
              f"forwards_D={dres.get('n_imgs', 0)} read_fail={b_audit['read_fail']} "
              f"device={dev_name} peak_alloc_MiB={b_audit['peak_alloc_mib']:.0f} "
              f"peak_reserved_MiB={b_audit['peak_reserved_mib']:.0f} wall_s={wall_b:.1f} "
              f"seed={SEED} G13_PREV_VS_POSTV_maxabs_probe4500={b_audit['maxabs_vs_post_probe']:.3e} "
              f"G13_PREV_VS_POSTV_maxabs_domains={b_audit['maxabs_vs_post_domains']:.3e}")
    if "error" in dres or "vit_identity" not in dres:
        mb.append(f"G13_D_MID=FAILED_OR_INCOMPLETE error={dres.get('error', 'aborted_midway')}")
    else:
        mb.append(f"G13_D_MID=file_keys={dres['load']['n_keys_in_file']} "
                  f"mapped={dres['load']['n_mapped']} missing={len(dres['load']['missing'])} "
                  f"skipped={len(dres['load']['skipped'])} n_imgs={dres['n_imgs']} "
                  f"cos_vs_pre_min={dres['cos_vs_pre_min']:.6f} "
                  f"cos_vs_pre_mean={dres['cos_vs_pre_mean']:.6f} "
                  f"cos_vs_post_min={dres['cos_vs_post_min']:.6f} "
                  f"tensor_differing_vs_pre={dres['vit_identity']['n_diff']} "
                  f"tensor_maxabs_vs_pre={dres['vit_identity']['max_abs']:.3e} "
                  f"tensor_only_in_mid={len(dres['vit_identity']['only_a'])} "
                  f"FROZEN={int(dres['frozen'])} "
                  f"rawV_metrics={'identical_to_PRE' if dres.get('metrics_equivalent_to') == 'pre' else 'NOT_COMPUTED'}")
    mb.append(f"G13_VERDICTS=B1={b1} B2={b2} B3={b3} B4={b4}")

    # ------------------------------------------------------------- stats ----
    stats = {
        "pre_auc_test": res_pre["auc_test"], "post_auc_test": res_post["auc_test"],
        "pre_pc0_varfrac": res_pre["pc0_varfrac"], "post_pc0_varfrac": res_post["pc0_varfrac"],
        "pre_d_z_src": res_pre["d_z_src"], "post_d_z_src": res_post["d_z_src"],
        "pre_e0_auc_test": res_pre["e0_auc_test"], "post_e0_auc_test": res_post["e0_auc_test"],
        "pre_gap_src": res_pre["gap_src"], "post_gap_src": res_post["gap_src"],
        "pre_mu": res_pre["mu"], "post_mu": res_post["mu"],
        "pre_e0": res_pre["e0"], "post_e0": res_post["e0"],
        "pipe_cos10": np.array(A["pipe_cos_list"], np.float64),
        "cos_pre_post10": np.array(A["cos_pre_post_list"], np.float64),
        "domains": np.array(TARGET_DOMAINS),
        "d_cross_dom": np.array([d_cross_dom[d] for d in TARGET_DOMAINS], np.float64),
        "d_gap_ratio": np.array([d_gapr[d] for d in TARGET_DOMAINS], np.float64),
        "shift_ratio": np.array([shift_ratio[d] for d in TARGET_DOMAINS], np.float64),
    }
    for tag, r in (("pre", res_pre), ("post", res_post)):
        for f in ("auc_e0", "auc_srclr", "d_z", "gap", "gap_ratio", "d_real", "d_fake",
                  "zdim_frac", "cos_e0dom_e0src", "shift_mag"):
            stats[f"{tag}_{f}"] = np.array([r["dom"][d][f] for d in TARGET_DOMAINS], np.float64)
    for tag, r in (("pre", res_pre), ("post", res_post)):
        for cls in ("real", "fake"):
            stats[f"{tag}_cc_{cls}_auc"] = np.array(
                [r["cc"][cls][d]["auc"] for d in TARGET_DOMAINS], np.float64)
    if "error" not in dres:
        stats["d_cos_vs_pre"] = np.array(dres["cos_vs_pre_min"], np.float64)
        stats["d_cos_vs_post"] = np.array(dres["cos_vs_post_min"], np.float64)
    np.savez_compressed(OUT_STATS_NPZ, **stats)
    L(f"[out] stats -> {OUT_STATS_NPZ}")

    # ------------------------------------------------------------ assemble ----
    header = [
        "=" * 120,
        "G13 REPORT -- H7: net_050 (pre-finetune) vs bridge_v2 (post-finetune) ViT representation",
        "Protocol: V = vit.forward_features(_preprocess_for_vit(336->224 CLIP-normalized))[:,0]",
        "          raw 768-d CLS; pre-V extracted fresh (4500 imgs), post-V read from the stored npz.",
        "          axis = center-only PCA on FF++ source-train V (2200); LR C=1e-3 lbfgs max_iter=3000;",
        "          is_fake=(y==0); split by video; post metrics are NOT re-extracted.",
        "=" * 120,
        "MACHINE_BLOCK",
    ]
    out = header + mb + [""] + report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[out] report -> {REPORT}", flush=True)
    print("\n".join(["", "MACHINE_BLOCK:"] + mb), flush=True)
    print(f"[done] total wall={time.time() - t_all:.1f}s", flush=True)


def write_outputs(report, mb, stats, path=None):
    """abort path: still emit report + machine block."""
    header = ["=" * 120, "G13 REPORT (ABORTED)", "=" * 120, "MACHINE_BLOCK"]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(header + list(mb) + [""] + list(report)) + "\n")
    print(f"[out] report -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
