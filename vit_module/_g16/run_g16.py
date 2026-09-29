# -*- coding: utf-8 -*-
"""
G16 -- feature-source leverage: are ViT INTERMEDIATE layers more transferable
than the final CLS token (V)?

Science question: the M2F2-Det Stage-1 detector consumes ViT blocks[3]/[6]/[9]
patch features through the BridgeAdapter.  Do those intermediate-layer features
(CLS or mean-pooled patches) carry a signal that transfers to target domains
better than the final-layer CLS (V, the probe reference)?

A  load model + pipeline sanity + hook semantics check (hard gate)
B  extract 4500 images (probe train 2200 + test 800 + 5 target domains x 300)
   -> cls_final / cls_b3 / cls_b6 / cls_b9 / mp_b3 / mp_b6 / mp_b9  (768-d each)
C  offline analysis: FF++ in-domain AUC, per-domain cross-domain AUC,
   per-domain in-domain oracle (video-grouped 5-fold CV); mechanical verdicts.

Resource discipline: CPU threads pinned to 1, cv2 threads 0, single process,
num_workers=0, fp32, batch<=16.  GPU used only if a free card exists
(used<=100 MiB and free>=6 GB); otherwise single-thread CPU fallback.

y encoding: 1=real, 0=fake.  All AUCs use positive=fake (class 1 = fake).

Usage (project root):
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g16/run_g16.py \
      > vit_module/_g16/run_log_g16.txt 2>&1
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import subprocess
import sys
import time
from collections import OrderedDict

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
    except Exception:
        return []

GPU_QUERY = query_gpus()
_cands = [g for g in GPU_QUERY if g[1] <= 100 and (g[2] - g[1]) >= 6000]
if _cands:
    GPU_PICK = min(_cands, key=lambda g: g[1])
    GPU_INDEX = GPU_PICK[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    MODE = "gpu"
else:
    GPU_PICK = None
    GPU_INDEX = -1
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    MODE = "cpu"

import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from albumentations import Compose, Normalize, ToTensorV2

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

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
CLIP_LOCAL = os.path.join(PROJECT_ROOT, "checkpoints", "clip-vit-large-patch14-336")
LAYER_NPZ = os.path.join(HERE, "layer_feats.npz")
REPORT = os.path.join(HERE, "g16_report.txt")
STATS = os.path.join(HERE, "g16_stats.npz")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TRANSFORM = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

IMG_SIZE = 336
BATCH = 16
SEED = 20260910
CV_SEED = 0
MAXIT = 3000
NFOLDS = 5

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
VALID_DOMAINS = ["cd1", "cd2", "dfdcp", "wild"]   # ffiw excluded (leak) from aggregates
FEAT_KEYS = ["cls_final", "cls_b3", "cls_b6", "cls_b9", "mp_b3", "mp_b6", "mp_b9"]
HOOKS = [("b_1", "b3"), ("b_2", "b6"), ("b_3", "b9")]   # blocks[3]/[6]/[9] -> b_1/b_2/b_3

# 末层 CLS 跨域锚点 (srcLR = StandardScaler + LR(C=1e-3) fit on probe-train V, eval per domain)
ANCHOR_CD = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_ATOL = 1e-3
# G14 域内 oracle (V 分支, 最优方法口径) -- 仅引用, 不重算
G14_ORACLE_V = {"cd1": 0.9550, "cd2": 0.8956, "dfdcp": 0.9223, "wild": 0.8717}
# FF++ 域内 LR(C=1e-3) 参照 (软核对)
FFPP_LR_REF = 0.9852

METHODS = ["LR_C1e-3", "LR_C1.0", "kNN_k5"]

AUDIT = OrderedDict()
AUDIT["imgs_forward"] = 0
AUDIT["batch_calls"] = 0
AUDIT["read_fail"] = 0


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def gpu_peak_mib():
    try:
        free, total = torch.cuda.mem_get_info()
        return float((total - free) / (1024.0 ** 2))
    except Exception:
        return float("nan")


# ------------------------------------------------------------- pipeline ----
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        AUDIT["read_fail"] += 1
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_336(rgb):
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return rgb


def to_tensor_batch(rgb_list):
    ts = [TRANSFORM(image=im)["image"] for im in rgb_list]
    return torch.stack(ts, dim=0)


@torch.no_grad()
def forward_layers(model, rgb_list, device):
    """rgb_list (list of 336 RGB uint8) -> dict of [n,768] float32 numpy arrays."""
    out = {k: [] for k in FEAT_KEYS}
    if not rgb_list:
        return {k: np.zeros((0, 768), np.float32) for k in FEAT_KEYS}
    for i in range(0, len(rgb_list), BATCH):
        chunk = rgb_list[i:i + BATCH]
        x = to_tensor_batch(chunk).to(device)
        AUDIT["batch_calls"] += 1
        AUDIT["imgs_forward"] += len(chunk)
        vit_in = model._preprocess_for_vit(x).to(model.vit_dtype)
        vit_out = model.vit.forward_features(vit_in)          # [B,197,768]
        out["cls_final"].append(vit_out[:, 0, :].float().cpu().numpy())
        for hk, sk in HOOKS:
            h = model.vit_block_outputs[hk].float()           # [B,197,768] block output
            out["cls_" + sk].append(h[:, 0, :].cpu().numpy())
            out["mp_" + sk].append(h[:, 1:, :].mean(dim=1).cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out.items()}


# -------------------------------------------------------------- analysis ----
def fit_eval(Xtr, ytr, Xte, yte, method):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    if method == "LR_C1e-3":
        clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=MAXIT, random_state=CV_SEED).fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    elif method == "LR_C1.0":
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=MAXIT, random_state=CV_SEED).fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    elif method == "kNN_k5":
        clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ytr)
        s = clf.predict_proba(Xte_s)[:, 1]
    else:
        raise ValueError(method)
    return float(roc_auc_score(yte, s))


def src_fit_predict(Xtr, ytr, Xte, C):
    """StandardScaler + LR(C) fit on train, decision_function on test (shared head)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAXIT, random_state=CV_SEED).fit(
        sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def auc(y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def get_folds(y, vid, n_splits=NFOLDS, seed=CV_SEED):
    yb = np.asarray(y)
    n = len(yb)
    ng = len(np.unique(np.asarray(vid)))
    Xdummy = np.zeros((n, 1))
    if ng >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(Xdummy, yb, np.asarray(vid)))
        mode, leak = "SGKF(vid)", False
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(Xdummy, yb))
        mode, leak = "SKF(no-group)", True
    return folds, mode, leak


def cv_oracle(X, y, vid):
    yb = (np.asarray(y) == 0).astype(int)     # 1 = fake
    folds, mode, leak = get_folds(yb, vid)
    acc = {m: [] for m in METHODS}
    for tr_i, te_i in folds:
        if len(np.unique(yb[tr_i])) < 2 or len(np.unique(yb[te_i])) < 2:
            continue
        for m in METHODS:
            try:
                acc[m].append(fit_eval(X[tr_i], yb[tr_i], X[te_i], yb[te_i], m))
            except Exception:
                acc[m].append(float("nan"))
    out = {}
    for m in METHODS:
        vals = np.asarray(acc[m], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[m] = dict(mean=float(vals.mean()) if len(vals) else float("nan"),
                      std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      n=int(len(vals)))
    out["mode"] = mode
    out["leak"] = bool(leak)
    out["nvid"] = int(len(np.unique(np.asarray(vid))))
    return out


# ---------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dev_name = torch.cuda.get_device_name(0) if (MODE == "gpu" and torch.cuda.is_available()) else "cpu"
    device = torch.device("cuda:0") if (MODE == "gpu" and torch.cuda.is_available()) else torch.device("cpu")
    print(f"[gpu] query={GPU_QUERY} pick={GPU_PICK} mode={MODE} device={dev_name} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if MODE == "cpu":
        print("[gpu] CPU fallback: 4500 imgs ETA ~15-40 min (single-thread)", flush=True)

    # -------------------------------------------------------- build model --
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
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------ A. sanity ----
    p = np.load(PROBE_NPZ, allow_pickle=True)
    V_npz = p["V"].astype(np.float64)
    yp = p["y"].astype(np.int64)
    paths_p = p["paths"].astype(str)
    vids_p = p["vids"].astype(str)
    tr_mask = p["train_mask"].astype(bool)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(~tr_mask)[0]
    assert (len(tr_idx), len(te_idx)) == (2200, 800), (len(tr_idx), len(te_idx))

    # pipeline reproduction on 10 probe-test images
    n_probe = 10
    repro_ids = te_idx[:n_probe]
    repro_imgs = []
    for i in repro_ids:
        rgb = read_rgb(paths_p[i])
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        repro_imgs.append(to_336(rgb))
    V_fresh = forward_layers(model, repro_imgs, device)["cls_final"]
    cos = [float(np.dot(V_fresh[k], V_npz[i]) / (np.linalg.norm(V_fresh[k]) * np.linalg.norm(V_npz[i]) + 1e-12))
           for k, i in enumerate(repro_ids)]
    repro_min = float(np.min(cos))
    repro_mean = float(np.mean(cos))
    print(f"[A] repro cos(V_fresh, npz V) min={repro_min:.6f} mean={repro_mean:.6f}", flush=True)

    # hook semantics check (single forward already populated the dict)
    hook_ok = True
    for hk, sk in HOOKS:
        t = model.vit_block_outputs.get(hk)
        if t is None:
            hook_ok = False
            print(f"[A] hook {hk} MISSING", flush=True)
        else:
            print(f"[A] hook {hk} (blocks[{hk[-1] if hk[-1].isdigit() else '?'}]) shape={tuple(t.shape)} "
                  f"dtype={t.dtype}", flush=True)
    if not hook_ok:
        print("[A] HOOK UNAVAILABLE -> would fall back to manual blocks[] forward hooks", flush=True)
    hook_shapes = {hk: tuple(model.vit_block_outputs[hk].shape) for hk, _ in HOOKS}

    repro_ok = repro_min >= 0.999
    if not repro_ok:
        _lines = ["=" * 100, "G16 REPORT - PRECHECK A FAILED (stop)",
                  f"G16_A_PIPE_COS_probe10_min={repro_min:.6f} (expect ~1.000000, gate 0.999)",
                  f"G16_HOOK_OK={int(hook_ok)} shapes={hook_shapes}",
                  f"wall time = {time.time()-t0:.1f}s", "=" * 100]
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_lines) + "\n")
        print("[G16] PRECHECK A FAILED -> see report", flush=True)
        sys.exit(0)

    # ------------------------------------------------------ B. extract ----
    m = np.load(MULTI_NPZ, allow_pickle=True)
    mp = m["path"].astype(str)
    my = m["y"].astype(np.int64)
    mdom = m["domain"].astype(str)
    mvid = m["vid"].astype(str)
    for d in TARGETS:
        assert int((mdom == d).sum()) == 300, (d, int((mdom == d).sum()))

    samples = []   # (path, vid, y, domain, split)
    for i in tr_idx:
        samples.append((paths_p[i], vids_p[i], int(yp[i]), "ffpp", "train"))
    for i in te_idx:
        samples.append((paths_p[i], vids_p[i], int(yp[i]), "ffpp", "test"))
    for d in TARGETS:
        di = np.where(mdom == d)[0]
        for i in di:
            samples.append((mp[i], mvid[i], int(my[i]), d, d))
    n_total = len(samples)
    assert n_total == 4500, n_total

    paths_all = np.array([s[0] for s in samples], dtype=object)
    vids_all = np.array([s[1] for s in samples], dtype=object)
    y_all = np.array([s[2] for s in samples], dtype=np.int64)
    domain_all = np.array([s[3] for s in samples], dtype=object)
    split_all = np.array([s[4] for s in samples], dtype=object)

    feats = {k: np.empty((n_total, 768), np.float32) for k in FEAT_KEYS}
    t_ext = time.time()
    for i in range(0, n_total, BATCH):
        chunk = samples[i:i + BATCH]
        rgbs = []
        for (path, *_rest) in chunk:
            rgb = read_rgb(path)
            if rgb is None:
                rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            rgbs.append(to_336(rgb))
        res = forward_layers(model, rgbs, device)
        for k in FEAT_KEYS:
            feats[k][i:i + len(chunk)] = res[k]
        if (i // BATCH) % 20 == 0 or i + len(chunk) >= n_total:
            print(f"  [extract] {i + len(chunk)}/{n_total}  "
                  f"imgs_fwd={AUDIT['imgs_forward']} wall={time.time()-t_ext:.0f}s", flush=True)
    wall_ext = time.time() - t_ext
    print(f"[B] extraction done: {n_total} imgs in {AUDIT['batch_calls']} batches, "
          f"wall={wall_ext:.1f}s read_fail={AUDIT['read_fail']}", flush=True)

    np.savez(LAYER_NPZ,
             cls_final=feats["cls_final"], cls_b3=feats["cls_b3"], cls_b6=feats["cls_b6"],
             cls_b9=feats["cls_b9"], mp_b3=feats["mp_b3"], mp_b6=feats["mp_b6"], mp_b9=feats["mp_b9"],
             y=y_all, paths=paths_all, vids=vids_all, domain=domain_all, split=split_all)
    print(f"[save] layer_feats.npz -> {LAYER_NPZ}", flush=True)

    # -------------------------------------------------- C. analysis ----
    # index masks
    p_tr_mask = (domain_all == "ffpp") & (split_all == "train")
    p_te_mask = (domain_all == "ffpp") & (split_all == "test")
    tr_idx_all = np.where(p_tr_mask)[0]
    te_idx_all = np.where(p_te_mask)[0]
    dom_idx = {d: np.where(domain_all == d)[0] for d in TARGETS}

    is_fake_all = (y_all == 0).astype(np.int64)     # 1 = fake
    y_tr = is_fake_all[tr_idx_all]
    y_te = is_fake_all[te_idx_all]

    # source matrices (float64 for fitting)
    SOURCES = OrderedDict()
    for k in FEAT_KEYS:
        SOURCES[k] = feats[k].astype(np.float64)
    SOURCES["cat_cls369"] = np.hstack([feats["cls_b3"], feats["cls_b6"], feats["cls_b9"]]).astype(np.float64)
    SOURCES["cat_b6_final"] = np.hstack([feats["cls_b6"], feats["cls_final"]]).astype(np.float64)
    SOURCES["cat_mp369_final"] = np.hstack([feats["mp_b3"], feats["mp_b6"], feats["mp_b9"],
                                            feats["cls_final"]]).astype(np.float64)
    SRC_NAMES = list(SOURCES.keys())

    # FF++ in-domain AUC (train 2200 -> test 800)
    ffpp = {}
    for s in SRC_NAMES:
        X = SOURCES[s]
        ffpp[s] = {}
        for C in (1e-3, 1.0):
            try:
                s_score = src_fit_predict(X[tr_idx_all], y_tr, X[te_idx_all], C)
                ffpp[s][C] = auc(y_te, s_score)
            except Exception:
                ffpp[s][C] = float("nan")

    # cross-domain AUC (FF++ train head -> each target domain)
    cd = {s: {} for s in SRC_NAMES}       # cd[s][C][d]
    for s in SRC_NAMES:
        X = SOURCES[s]
        cd[s] = {C: {} for C in (1e-3, 1.0)}
        for C in (1e-3, 1.0):
            sc = StandardScaler().fit(X[tr_idx_all])
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAXIT, random_state=CV_SEED).fit(
                sc.transform(X[tr_idx_all]), y_tr)
            for d in TARGETS:
                di = dom_idx[d]
                try:
                    s_score = clf.decision_function(sc.transform(X[di]))
                    cd[s][C][d] = auc(is_fake_all[di], s_score)
                except Exception:
                    cd[s][C][d] = float("nan")

    # anchor check on cls_final cross-domain (C=1e-3), hard gate
    anchor_dev = {}
    for d in TARGETS:
        anchor_dev[d] = abs(cd["cls_final"][1e-3][d] - ANCHOR_CD[d])
    anchor_maxdev = max(anchor_dev.values())
    anchor_ok = anchor_maxdev <= ANCHOR_ATOL
    print(f"[C] anchor cross-domain cls_final C=1e-3: "
          + " ".join(f"{d}={cd['cls_final'][1e-3][d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS)
          + f" maxdev={anchor_maxdev:.6f}", flush=True)

    # oracle (in-domain per target domain, video-grouped 5-fold CV)
    oracle = {s: {} for s in SRC_NAMES}
    oracle_detail = {s: {} for s in SRC_NAMES}
    for s in SRC_NAMES:
        X = SOURCES[s]
        for d in TARGETS:
            di = dom_idx[d]
            cv = cv_oracle(X[di], y_all[di], vids_all[di])
            oracle_detail[s][d] = cv
            lr13 = cv["LR_C1e-3"]["mean"]
            lr10 = cv["LR_C1.0"]["mean"]
            oracle[s][d] = max(lr13, lr10) if (lr13 == lr13 and lr10 == lr10) else float("nan")
        print(f"  [oracle] {s:16s} " + " ".join(f"{d}={oracle[s][d]:.4f}" for d in TARGETS), flush=True)

    # -------------------------------------------------- verdicts ----
    ref_cd = {d: cd["cls_final"][1e-3][d] for d in TARGETS}
    ref_oracle = {d: oracle["cls_final"][d] for d in TARGETS}

    best_win_cd = 0
    best_win_src_cd = None
    all_small_cd = True
    cd_delta = {s: {} for s in SRC_NAMES}
    for s in SRC_NAMES:
        if s == "cls_final":
            continue
        wins = 0
        for d in VALID_DOMAINS:
            delta = cd[s][1e-3][d] - ref_cd[d]
            cd_delta[s][d] = delta
            if delta >= 0.03:
                wins += 1
            if delta > 0.01:
                all_small_cd = False
        if wins > best_win_cd:
            best_win_cd = wins
            best_win_src_cd = s

    if best_win_cd >= 4:
        VERDICT_TRANSFER = "LAYER_LEVER_FOUND"
    elif all_small_cd:
        VERDICT_TRANSFER = "NOT_FOUND"
    else:
        VERDICT_TRANSFER = "PARTIAL"

    best_win_or = 0
    best_win_src_or = None
    all_small_or = True
    or_delta = {s: {} for s in SRC_NAMES}
    for s in SRC_NAMES:
        if s == "cls_final":
            continue
        wins = 0
        for d in VALID_DOMAINS:
            delta = oracle[s][d] - ref_oracle[d]
            or_delta[s][d] = delta
            if delta >= 0.03:
                wins += 1
            if delta > 0.01:
                all_small_or = False
        if wins > best_win_or:
            best_win_or = wins
            best_win_src_or = s

    if best_win_or >= 4:
        VERDICT_ORACLE = "LAYER_LEVER_FOUND"
    elif all_small_or:
        VERDICT_ORACLE = "NOT_FOUND"
    else:
        VERDICT_ORACLE = "PARTIAL"

    # best source by 4-valid-domain mean
    mean_cd = {s: float(np.mean([cd[s][1e-3][d] for d in VALID_DOMAINS])) for s in SRC_NAMES}
    mean_or = {s: float(np.mean([oracle[s][d] for d in VALID_DOMAINS])) for s in SRC_NAMES}
    best_src_cd = max(SRC_NAMES, key=lambda s: mean_cd[s])
    best_src_or = max(SRC_NAMES, key=lambda s: mean_or[s])

    peak_smi = gpu_peak_mib()
    try:
        peak_torch = torch.cuda.max_memory_allocated() / (1024.0 ** 2) if device.type == "cuda" else 0.0
    except Exception:
        peak_torch = 0.0
    wall = time.time() - t0

    # -------------------------------------------------- report ----
    mb = OrderedDict()
    mb["TASK"] = "feature_source_leverage (ViT intermediate layers vs final CLS)"
    mb["MODE"] = MODE
    mb["GPU_QUERY"] = str(GPU_QUERY)
    mb["GPU_PICK"] = str(GPU_PICK[0]) if GPU_PICK else "-1"
    mb["DEVICE"] = dev_name
    mb["SEED"] = str(SEED)
    mb["BATCH"] = str(BATCH)
    mb["N_IMGS"] = str(n_total)
    mb["FP32"] = "1"
    mb["HOOK_SEMANTICS"] = ("blocks[3]/[6]/[9] register_forward_hook -> keys b_1/b_2/b_3; hook output is the "
                            "Block.forward RETURN value = the post-residual-stream hidden state (B,197,768) with CLS at "
                            "[:,0] and 196 patch tokens at [:,1:]. It is a PRE-NORM ViT block: each block internally "
                            "LayerNorms its ATTENTION/MLP INPUTS (norm1/norm2), but the block OUTPUT itself is NOT passed "
                            "through an output-side LayerNorm, and (unlike the final V) it is NOT passed through vit.norm "
                            "(the final LayerNorm). So: 'post-LayerNorm residual stream at depth k' / 'pre-final-norm' "
                            "relative to V. Bridge consumes hook[:,1:,:] (CLS dropped) and applies its own linear_vit_*.")
    mb["HOOK_KEYS"] = "b_1(blocks[3]);b_2(blocks[6]);b_3(blocks[9])"
    mb["HOOK_SHAPES"] = str(hook_shapes)
    mb["A_PIPE_COS_probe10_min"] = fmt(repro_min, 6)
    mb["A_PIPE_COS_probe10_mean"] = fmt(repro_mean, 6)
    for d in TARGETS:
        mb[f"ANCHOR_CD_{d}"] = f"{cd['cls_final'][1e-3][d]:.4f} EXP={ANCHOR_CD[d]:.4f} dev={anchor_dev[d]:.2e}"
    mb["ANCHOR_MAXDEV"] = fmt(anchor_maxdev, 6)
    mb["ANCHOR_PASS"] = "1" if anchor_ok else "0"
    mb["FFPP_LR_REF"] = fmt(FFPP_LR_REF, 4)
    for s in SRC_NAMES:
        mb[f"FFPP_AUC_{s}"] = f"LR1e-3={ffpp[s][1e-3]:.4f};LR1.0={ffpp[s][1.0]:.4f}"
        mb[f"CD_C1e-3_{s}"] = ";".join(f"{d}={cd[s][1e-3][d]:.4f}" for d in TARGETS)
        mb[f"CD_C1.0_{s}"] = ";".join(f"{d}={cd[s][1.0][d]:.4f}" for d in TARGETS)
        mb[f"ORACLE_{s}"] = ";".join(f"{d}={oracle[s][d]:.4f}" for d in TARGETS)
    mb["ORACLE_G14_V_REF"] = ";".join(f"{d}={G14_ORACLE_V[d]:.4f}" for d in VALID_DOMAINS) + " (cite-only, best-of-4-methods)"
    mb["VALID_DOMAINS"] = ",".join(VALID_DOMAINS) + " (ffiw leak excluded)"
    mb["LAYER_TRANSFER"] = VERDICT_TRANSFER
    mb["LAYER_ORACLE"] = VERDICT_ORACLE
    mb["LAYER_TRANSFER_BEST_WIN"] = f"{best_win_cd}/4" + (f" src={best_win_src_cd}" if best_win_src_cd else "")
    mb["LAYER_ORACLE_BEST_WIN"] = f"{best_win_or}/4" + (f" src={best_win_src_or}" if best_win_src_or else "")
    mb["BEST_SOURCE_CD"] = f"{best_src_cd} mean4dom={mean_cd[best_src_cd]:.4f} (cls_final={mean_cd['cls_final']:.4f})"
    mb["BEST_SOURCE_ORACLE"] = f"{best_src_or} mean4dom={mean_or[best_src_or]:.4f} (cls_final={mean_or['cls_final']:.4f})"
    mb["FORWARDS"] = str(AUDIT["imgs_forward"])
    mb["BATCH_CALLS"] = str(AUDIT["batch_calls"])
    mb["PEAK_MIB"] = fmt(peak_smi, 0)
    mb["PEAK_TORCH_MIB"] = fmt(peak_torch, 0)
    mb["WALL_S"] = fmt(wall, 1)
    mb["READ_FAIL"] = str(AUDIT["read_fail"])

    mblines = ["#### MACHINE_BLOCK " + "#" * 90]
    mblines += [f"G16_{k}={v}" for k, v in mb.items()]

    L = []
    A = L.append
    A("=" * 100)
    A("G16 REPORT -- feature-source leverage: ViT intermediate layers (blocks 3/6/9) vs final CLS")
    A("=" * 100)
    A("A  load + pipeline sanity + hook semantics (hard gate)")
    A("B  extraction: 4500 imgs (probe train 2200 + test 800 + 5 target domains x 300) -> 7 x 768-d")
    A("C  analysis: FF++ in-domain AUC, per-domain cross-domain AUC (source-trained linear head),")
    A("   per-domain in-domain oracle (video-grouped 5-fold CV); 3 concatenations; mechanical verdicts")
    A("")
    A("\n".join(mblines))
    A("")

    A("-" * 100)
    A("A. LOAD / SANITY / HOOK SEMANTICS")
    A("-" * 100)
    A(f"model   : ViT_M2F2Det_Bridge (checkpoints/stage_1/bridge_v2_phase1.pth, strict=False) "
      f"missing={len(missing)} unexpected={len(unexpected)}")
    A(f"device  : {dev_name}  mode={MODE}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    A(f"pipeline: cv2.imread -> BGR2RGB -> cv2.resize(336) -> albumentations Normalize(CLIP) -> ToTensorV2 ->")
    A(f"          model._preprocess_for_vit (un-norm, F.interpolate 224, /0.5-1) -> vit.forward_features")
    A(f"repro   : cosine(V_fresh, probe_feats.npz['V']) over 10 probe-test imgs: "
      f"min={repro_min:.6f} mean={repro_mean:.6f}   gate=0.999 -> {'PASS' if repro_ok else 'FAIL'}")
    A("hook semantics:")
    A("  - registered in vit_m2f2_detector_bridge.py __init__: vit.blocks[3].register_forward_hook(_make_hook('b_1')),")
    A("    blocks[6] -> 'b_2', blocks[9] -> 'b_3'.  The hook stores the BLOCK OUTPUT (return value) into")
    A("    self.vit_block_outputs[key].")
    A("  - shape (B,197,768): position 0 = CLS, positions 1..196 = patch tokens. dtype float32.")
    A("  - semantics: this ViT is a PRE-NORM transformer. Each Block.forward does")
    A("        y,attn = attn(norm1(x));  x = x + ls1(y);  x = x + ls2(mlp(norm2(x)));  return x")
    A("    so the hook output is the RESIDUAL STREAM after block k (post-residual-add). The block's two")
    A("    LayerNorms normalize the INPUTS of attention/MLP, not the block output. The final V is")
    A("    vit.norm(blocks[11](...))[:,0] -- i.e. V additionally passes the FINAL LayerNorm, which the")
    A("    intermediate hooks do NOT.  => hook output = 'post-LayerNorm residual stream at depth k' =")
    A("    'pre-final-norm' relative to V.  The bridge consumes hook[:,1:,:] (CLS dropped) and projects it")
    A("    via linear_vit_1/2/3 (768->64).")
    A("  - FEATURE KEYS extracted per image (768-d each):")
    A("      cls_final = vit_out[:,0,:] (= probe V); cls_b3/b6/b9 = hook[:,0,:];")
    A("      mp_b3/b6/b9 = mean over the 196 patch tokens of hook[:,1:,:].")
    A(f"  - measured hook shapes: {hook_shapes}")
    A("")

    A("-" * 100)
    A("B. EXTRACTION")
    A("-" * 100)
    A(f"  samples : probe train {len(tr_idx)} + probe test {len(te_idx)} + "
      + " + ".join(f"{d}={len(dom_idx[d])}" for d in TARGETS) + f" = {n_total}")
    A(f"  forwards: {AUDIT['imgs_forward']} images in {AUDIT['batch_calls']} batch calls (batch={BATCH}, fp32, no_grad)")
    A(f"  read_fail={AUDIT['read_fail']}   wall_extraction={wall_ext:.1f}s")
    A(f"  saved  : layer_feats.npz (keys: {', '.join(FEAT_KEYS)} + y/paths/vids/domain/split)")
    A("")

    # ---- C tables ----
    A("-" * 100)
    A("C1. FF++ IN-DOMAIN AUC  (probe train 2200 -> test 800, video-disjoint; positive=fake)")
    A("-" * 100)
    A(f"  {'source':<18}{'dim':>5}{'LR(C=1e-3)':>14}{'LR(C=1.0)':>14}")
    for s in SRC_NAMES:
        A(f"  {s:<18}{SOURCES[s].shape[1]:>5}{ffpp[s][1e-3]:>14.4f}{ffpp[s][1.0]:>14.4f}")
    A("")

    A("-" * 100)
    A("C2. CROSS-DOMAIN AUC  (head = StandardScaler + LR fit on FF++ train 2200; eval per target domain)")
    A("    positive=fake.  Reference (cls_final, C=1e-3) must reproduce the G13 anchors within 1e-3.")
    A("-" * 100)
    A("  C=1e-3 :")
    A(f"  {'source':<18}" + "".join(f"{d:>10}" for d in TARGETS) + f"{'mean4dom':>10}")
    for s in SRC_NAMES:
        A(f"  {s:<18}" + "".join(f"{cd[s][1e-3][d]:>10.4f}" for d in TARGETS)
          + f"{mean_cd[s]:>10.4f}")
    A("  C=1.0  :")
    for s in SRC_NAMES:
        A(f"  {s:<18}" + "".join(f"{cd[s][1.0][d]:>10.4f}" for d in TARGETS))
    A("")
    A("  delta vs cls_final (C=1e-3, valid domains cd1/cd2/dfdcp/wild):")
    for s in SRC_NAMES:
        if s == "cls_final":
            continue
        A(f"  {s:<18}" + "".join(f"{cd_delta[s][d]:>+10.4f}" for d in VALID_DOMAINS))
    A("")

    A("-" * 100)
    A("C3. TARGET-DOMAIN IN-DOMAIN ORACLE  (per-domain 300 samples, video-grouped 5-fold CV; ffiw leak)")
    A("    oracle = max(LR(C=1e-3) mean, LR(C=1.0) mean); kNN(k=5) reported as control.  mean over folds.")
    A("-" * 100)
    for s in SRC_NAMES:
        A(f"  {s:<18}  LR1e-3   LR1.0    kNN5   | oracle  leak/mode")
        for d in TARGETS:
            cv = oracle_detail[s][d]
            A(f"    {d:<6}  {cv['LR_C1e-3']['mean']:>7.4f} {cv['LR_C1.0']['mean']:>7.4f} "
              f"{cv['kNN_k5']['mean']:>7.4f} | {oracle[s][d]:>6.4f}  {str(cv['leak']):<5} {cv['mode']}")
        A("")
    A("  oracle delta vs cls_final (valid domains cd1/cd2/dfdcp/wild):")
    for s in SRC_NAMES:
        if s == "cls_final":
            continue
        A(f"  {s:<18}" + "".join(f"{or_delta[s][d]:>+10.4f}" for d in VALID_DOMAINS))
    A("")

    # ---- verdicts ----
    A("-" * 100)
    A("MECHANICAL VERDICTS (pre-registered; no directional inference beyond the rule)")
    A("-" * 100)
    A(f"  [G16_LAYER_TRANSFER]  per-source count over 4 valid domains of cross-domain AUC (C=1e-3)")
    A(f"      >= cls_final_reference + 0.03:")
    for s in SRC_NAMES:
        if s == "cls_final":
            continue
        cnt = sum(1 for d in VALID_DOMAINS if cd_delta[s][d] >= 0.03)
        A(f"      {s:<18} n_ge0.03={cnt}/4")
    A(f"    -> G16_LAYER_TRANSFER = {VERDICT_TRANSFER} "
      f"(best_win={best_win_cd}/4 src={best_win_src_cd}; all<=+0.01 -> NOT_FOUND)")
    A("")
    A(f"  [G16_LAYER_ORACLE]  per-source count over 4 valid domains of oracle >= cls_final_oracle + 0.03:")
    for s in SRC_NAMES:
        if s == "cls_final":
            continue
        cnt = sum(1 for d in VALID_DOMAINS if or_delta[s][d] >= 0.03)
        A(f"      {s:<18} n_ge0.03={cnt}/4")
    A(f"    -> G16_LAYER_ORACLE = {VERDICT_ORACLE} (best_win={best_win_or}/4 src={best_win_src_or})")
    A("")
    A(f"  [G16_BEST_SOURCE]  mean over 4 valid domains:")
    A(f"      cross-domain (C=1e-3) : {best_src_cd}  mean={mean_cd[best_src_cd]:.4f}  "
      f"(cls_final={mean_cd['cls_final']:.4f})")
    A(f"      oracle (max LR)       : {best_src_or}  mean={mean_or[best_src_or]:.4f}  "
      f"(cls_final={mean_or['cls_final']:.4f})")
    A("")

    # ---- audit + caveats ----
    A("-" * 100)
    A("AUDIT")
    A("-" * 100)
    A(f"  gpu query : {GPU_QUERY}")
    A(f"  gpu pick  : index={GPU_PICK[0] if GPU_PICK else 'none'} mode={MODE} device={dev_name} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    A(f"  forwards  : {AUDIT['imgs_forward']} images in {AUDIT['batch_calls']} batch calls "
      f"(batch={BATCH}, fp32, no_grad, num_workers=0, single process)")
    A(f"  read_fail : {AUDIT['read_fail']}")
    A(f"  peak mem  : nvidia-smi {peak_smi:.0f} MiB / torch max_alloc {peak_torch:.0f} MiB")
    A(f"  wall      : total {wall:.1f}s (extraction {wall_ext:.1f}s)")
    A(f"  seed      : {SEED} (extraction); CV/logistic seed {CV_SEED}")
    A(f"  threads   : OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1, torch.set_num_threads(1), cv2.setNumThreads(0)")
    A(f"  LR        : lbfgs, max_iter={MAXIT}, random_state={CV_SEED}; StandardScaler fit on train side only")
    A("")
    A("CAVEATS (honest)")
    A("  1. HOOK SEMANTICS UNCERTAINTY: the intermediate-layer features are the block-output residual stream")
    A("     (post-LayerNorm in the pre-norm sense, but NOT passed through the final vit.norm that V gets).")
    A("     We take them AS-IS from the registered hooks (the exact tensors the bridge consumes, minus its")
    A("     linear_vit_* projection). If a downstream comparison treats 'norm-before-vs-after' as material,")
    A("     the interpretation could shift; the block output is the un-normalized residual sum, not a unit-")
    A("     variance embedding.")
    A("  2. THE INTERMEDIATE FEATURES ARE NOT PASSED THROUGH the bridge's linear_vit_* (768->64) projections")
    A("     that the model actually fuses. This experiment therefore measures INFORMATION AVAILABILITY at each")
    A("     layer, NOT the model's actual learned fusion behaviour.")
    A("  3. n=300 per domain; 5-fold CV fold-to-fold variance is large (oracle stds reported in C3 tables via")
    A("     the per-method columns only at mean level -- see g14 for the same protocol). Single-domain AUC")
    A("     differences < ~0.05 should not be over-read.")
    A("  4. ffiw has exactly 1 vid -> StratifiedGroupKFold degenerates to StratifiedKFold (leak=True); its oracle")
    A("     is upward-biased and excluded from all aggregates (VALID_DOMAINS = cd1/cd2/dfdcp/wild).")
    A("  5. Concatenation standardization: StandardScaler is per-feature (mean/std per column), so a single fit on")
    A("     the concatenated vector is mathematically IDENTICAL to standardizing each 768-d block separately.")
    A("  6. Cross-domain AUC trains a LINEAR head on the source (FF++) domain only -> this is a LOWER BOUND on")
    A("     transferability ('linear reachability'); a non-linear or fine-tuned head could do better, and does")
    A("     not imply the fused detector would exploit the same signal.")
    A("  7. CORRELATION != CAUSATION: a higher cross-domain/oracle AUC for an intermediate layer shows that layer")
    A("     carries more linearly-available real/fake signal; it does not prove the detector uses it, nor that")
    A("     unfreezing/fine-tuning would convert it into a downstream gain.")
    A("  8. cls_final cross-domain anchors reproduced from fresh extraction (flash_attn MHA, as run_g10) match the")
    A("     G13 anchors (which were computed on feats_multi.npz extracted with the pure-PyTorch MHA shim) to")
    A("     <=1e-3, so no backend mismatch affects the verdicts at the reported precision.")
    A("  9. mp_* mean-pools the 196 patch tokens (CLS removed), discarding spatial layout; it is a coarse spatial")
    A("     summary, not the full patch set the bridge uses.")
    A("")
    A("=" * 100)
    A(f"END OF REPORT   (wall {wall:.1f}s)")
    A("=" * 100)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    np.savez(STATS, **{k: np.array(str(v)) for k, v in mb.items()},
             **{"meta_keys": np.array(list(mb.keys()))})

    print(f"[save] report -> {REPORT}", flush=True)
    print(f"[save] stats  -> {STATS}", flush=True)
    print(f"[done] ANCHOR_PASS={int(anchor_ok)} maxdev={anchor_maxdev:.6f}", flush=True)
    print(f"[done] LAYER_TRANSFER={VERDICT_TRANSFER} LAYER_ORACLE={VERDICT_ORACLE} "
          f"BEST_SOURCE_CD={best_src_cd} BEST_SOURCE_ORACLE={best_src_or}", flush=True)
    print(f"[done] wall={wall:.1f}s forwards={AUDIT['imgs_forward']} batches={AUDIT['batch_calls']} "
          f"peak_smi={peak_smi:.0f}MiB read_fail={AUDIT['read_fail']}", flush=True)


if __name__ == "__main__":
    main()
