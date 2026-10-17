# -*- coding: utf-8 -*-
"""G10 -- in-model attribution experiments on the M2F2-Det Stage-1 detector.

I0  reproduction check of the probe feature pipeline (cosine vs probe_feats.npz)
E0  center-only PCA on probe-train V -> mu, e0 (the E3/G8B axis); z(x) = (V-mu)@e0
I1  transformation invariance (gray / jpeg70 / resize2x / flip / blur + crop90 control)
I2  spatial occlusion attribution, 8x8 patches (ffpp probe-test) + optional input x grad
I3  source-in vs cross-domain attribution (cd1, ffiw) -- occlusion, same 8x8 grid

GPU discipline: CUDA_VISIBLE_DEVICES=2 only, fp32, batch<=16, num_workers=0,
torch.set_num_threads(1), cv2.setNumThreads(0).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import time
import random
from collections import defaultdict, OrderedDict

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
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
CLIP_LOCAL = os.path.join(PROJECT_ROOT, "checkpoints", "clip-vit-large-patch14-336")
REPORT = os.path.join(HERE, "g10_report.txt")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# exactly the transform used by vit_module/_probe/residual_probe.py FeatDataset
TRANSFORM = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

IMG_SIZE = 336
GRID = 8
PATCH = IMG_SIZE // GRID          # 42
BATCH = 16
SEED = 20260910

# ---------------------------------------------------------------- audit ----
AUDIT = OrderedDict()
AUDIT["imgs_forward"] = 0
AUDIT["batch_calls"] = 0
AUDIT["bwd_calls"] = 0
AUDIT["read_fail"] = 0
AUDIT["peaks_mib"] = []


def gpu_peak_mib():
    try:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1024.0 ** 2)
        AUDIT["peaks_mib"].append(float(used))
        return float(used)
    except Exception:
        return float("nan")


# ------------------------------------------------------- image pipeline ----
def read_rgb(path):
    """cv2.imread -> BGR2RGB at native resolution (None on failure)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        AUDIT["read_fail"] += 1
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_variants(rgb):
    """7 low-level variants produced on the native-resolution RGB uint8 image."""
    out = OrderedDict()
    out["orig"] = rgb
    out["gray"] = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        out["jpeg70"] = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        out["jpeg70"] = rgb.copy()
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    out["resize2x"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    out["flip"] = cv2.flip(rgb, 1)
    out["blur"] = cv2.GaussianBlur(rgb, (5, 5), 1.0)
    ch, cw = int(round(h * 0.9)), int(round(w * 0.9))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = rgb[y0:y0 + ch, x0:x0 + cw]
    out["crop90"] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


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


# ------------------------------------------------------------- helpers ----
def lr_fit_score(Xtr, ytr, Xte, C=1e-3, seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def auc(y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def stratified_pick(idx, labels, vids, n_total, seed):
    """Round-robin over video keys within each class, balanced real/fake."""
    rng = random.Random(seed)
    per_class = n_total // 2
    picks = []
    for lab in (1, 0):
        groups = defaultdict(list)
        for i in idx:
            if labels[i] == lab:
                groups[vids[i]].append(i)
        keys = sorted(groups.keys())
        rng.shuffle(keys)
        for k in keys:
            rng.shuffle(groups[k])
        got, r = [], 0
        while len(got) < per_class and keys:
            progressed = False
            for k in keys:
                if len(got) >= per_class:
                    break
                if r < len(groups[k]):
                    got.append(groups[k][r])
                    progressed = True
            r += 1
            if not progressed:
                break
        picks.extend(got)
    rng.shuffle(picks)
    return picks


def center_only_pca(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    # economy SVD; e0 = first right singular vector
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    e0 = Vt[0].astype(np.float64)
    varfrac = float((S[0] ** 2) / (S ** 2).sum())
    return mu.astype(np.float64), e0, varfrac, S


def occ_heatmaps(model, device, ids, labels, paths, mu, e0, tag):
    """8x8 occlusion attribution. Filling = per-image global mean colour."""
    n = len(ids)
    dz = np.zeros((n, GRID * GRID), np.float64)
    z_orig = np.zeros(n, np.float64)
    for k, i in enumerate(ids):
        rgb = read_rgb(paths[i])
        if rgb is None:
            continue
        rgb = to_336(rgb)
        z_orig[k] = (forward_V(model, [rgb], device)[0] - mu) @ e0
        flat = rgb.reshape(-1, 3)
        fill = flat.mean(axis=0).round().astype(np.uint8)
        variants = []
        for p in range(GRID * GRID):
            r, c = divmod(p, GRID)
            v = rgb.copy()
            v[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = fill
            variants.append(v)
        Vs = forward_V(model, variants, device)
        dz[k] = (Vs - mu) @ e0 - z_orig[k]
        if (k + 1) % 10 == 0:
            print(f"  [{tag}] occlusion {k+1}/{n}  imgs_fwd={AUDIT['imgs_forward']}", flush=True)
    return dz, z_orig


def hot_stats(H):
    """H: 8x8 non-negative heatmap -> top-5 blocks, center/edge fractions."""
    H = np.asarray(H, np.float64)
    tot = H.sum()
    order = np.argsort(H.ravel())[::-1]
    top5 = [int(p) for p in order[:5]]
    top5_share = float(H.ravel()[order[:5]].sum() / tot) if tot > 0 else float("nan")
    cen = H[2:6, 2:6].sum()
    cen_frac = float(cen / tot) if tot > 0 else float("nan")
    return top5, top5_share, cen_frac


def fmt_blocks(ps):
    return ",".join(f"({p//GRID},{p%GRID})" for p in ps)


def input_grad_heatmaps(model, device, ids, paths, mu, e0):
    """input x grad for z(x): |grad_x z * x| summed over channels, pooled per 8x8 patch.

    model._preprocess_for_vit uses F.interpolate -> the whole path is differentiable.
    mu/e0 MUST be torch tensors on the same device (numpy would hit the device-mismatch
    TypeError that aborted the optional branch of the first run)."""
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float32, device=device)
    e0_t = torch.as_tensor(np.asarray(e0), dtype=torch.float32, device=device)
    grads = np.full((len(ids), GRID * GRID), np.nan, np.float64)
    for k, i in enumerate(ids):
        rgb = read_rgb(paths[i])
        if rgb is None:
            continue
        x = to_tensor_batch([to_336(rgb)]).to(device)
        x.requires_grad_(True)
        with torch.enable_grad():
            vit_in = model._preprocess_for_vit(x).to(model.vit_dtype)
            out = model.vit.forward_features(vit_in)
            z = ((out[:, 0, :].float() - mu_t) @ e0_t).sum()
            g, = torch.autograd.grad(z, x)
            AUDIT["bwd_calls"] += 1
        AUDIT["imgs_forward"] += 1
        AUDIT["batch_calls"] += 1
        a = (g * x).detach().abs().sum(dim=1)[0].cpu().numpy()      # 336x336
        for p in range(GRID * GRID):
            r, c = divmod(p, GRID)
            grads[k, p] = a[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH].sum()
    return grads


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def parse_report_heatmaps(report_path):
    """Read back the I2 [real]/[fake] mean|dz| heat maps written by the first stage."""
    out = {}
    with open(report_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for key in ("[real]", "[fake]"):
        for j, ln in enumerate(lines):
            if ln.startswith(f"{key} mean|dz| heat map"):
                H = [list(map(float, lines[j + 1 + r].split())) for r in range(GRID)]
                out[key.strip("[]")] = np.array(H)
                break
    if len(out) != 2:
        raise RuntimeError(f"could not parse I2 heat maps from {report_path}")
    return out


# ---------------------------------------------------------------- main ----
def run_ig_stage(report_path, model, device, te_idx, y, vids, paths, src, mu, e0):
    """Cheap follow-up stage: only the optional input x grad for the 40 I2 images.

    Reuses the I2 heat maps already written to the report (parsed back) so no occlusion
    forwards are repeated. Costs 40 forwards + 40 backward calls.
    """
    t_ig = time.time()
    i2_ids = stratified_pick(te_idx, y, vids, 40, seed=SEED + 2)
    lab2 = np.array([src[i] for i in i2_ids])
    grads = input_grad_heatmaps(model, device, i2_ids, paths, mu, e0)
    occ = parse_report_heatmaps(report_path)

    def cls_mean(mat, lab_val):
        return np.nanmean(np.where(lab2[:, None] == lab_val, mat, np.nan), axis=0)

    gm = {c: cls_mean(grads, 1 if c == "real" else 0) for c in ("real", "fake")}
    res = {}
    for c in ("real", "fake"):
        corr = float(np.dot(_norm(occ[c].ravel()), _norm(gm[c])))
        tp, sh, cf = hot_stats(gm[c].reshape(GRID, GRID))
        res[c] = (corr, tp, sh, cf)
        print(f"[ig] {c}: cos(occlusion, inputXgrad)={corr:.3f} top5={fmt_blocks(tp)} "
              f"top5_share={sh:.3f} center16={cf:.4f}")
    summary = (f"inputXgrad=ok cos_occlusion_vs_inputxgrad real={res['real'][0]:.3f} "
               f"fake={res['fake'][0]:.3f} top5_blocks_fake={fmt_blocks(res['fake'][1])} "
               f"top5_share_fake={res['fake'][2]:.4f} center16_fake={res['fake'][3]:.4f} "
               f"center16_real={res['real'][3]:.4f} (n=40, 20/20; z backprop through "
               f"F.interpolate in _preprocess_for_vit)")
    import re
    text = open(report_path, "r", encoding="utf-8").read()
    # whole-line replacement (the "skipped" line carried a trailing TypeError text)
    text = re.sub(r"optional input x grad:.*", "optional input x grad: " + summary, text)
    text = text.replace("G10_I3_CD1_TOP=", f"G10_I2_IG_COS_real={res['real'][0]:.3f} "
                                           f"_fake={res['fake'][0]:.3f} G10_I3_CD1_TOP=")
    text = re.sub(r"backward calls \(input x grad\)\s*:\s*(\d+)",
                  lambda m: "backward calls (input x grad)            : "
                            f"{int(m.group(1)) + AUDIT['bwd_calls']}", text)
    text = re.sub(r"FORWARD_BATCH_CALLS=(\d+)",
                  lambda m: f"FORWARD_BATCH_CALLS={int(m.group(1)) + AUDIT['batch_calls']}", text)
    text = re.sub(r"BWD_CALLS=(\d+)",
                  lambda m: f"BWD_CALLS={int(m.group(1)) + AUDIT['bwd_calls']}", text)
    text = re.sub(r"batch-equivalents \(imgs/16\)\s*:\s*([\d.]+)",
                  lambda m: "batch-equivalents (imgs/16)              : "
                            f"{float(m.group(1)) + AUDIT['imgs_forward'] / 16.0:.1f}", text)
    m = re.search(r"G10_FORWARDS_USED=(\d+)", text)
    if m:
        tot = int(m.group(1)) + AUDIT["imgs_forward"]
        text = text[:m.start(1)] + str(tot) + text[m.end(1):]
    m = re.search(r"forwards \(images through the ViT branch\) : (\d+) in (\d+) batch calls",
                  text)
    if m:
        new = (f"forwards (images through the ViT branch) : "
               f"{int(m.group(1)) + AUDIT['imgs_forward']} in "
               f"{int(m.group(2)) + AUDIT['batch_calls']} batch calls  "
               f"[{m.group(1)}+{AUDIT['imgs_forward']} images; the +{AUDIT['imgs_forward']} "
               f"and the {AUDIT['bwd_calls']} backward calls come from the --stage ig "
               f"inputXgrad follow-up run]")
        text = text[:m.start()] + new + text[m.end():]
    text = text.rstrip() + (f"\n\nNOTE: the optional input x grad of I2 was computed in a separate\n"
                            f"low-cost follow-up run (--stage ig, {time.time()-t_ig:.0f} s) over the same\n"
                            f"deterministic I2 sample; it re-parsed the occlusion heat maps from this\n"
                            f"file instead of re-running the 2560 occlusion forwards, so the budget of\n"
                            f"this follow-up was only {AUDIT['imgs_forward']} forwards + "
                            f"{AUDIT['bwd_calls']} backward calls.\n")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[save] patched -> {report_path}")
    print(f"ig stage audit: fwd_imgs={AUDIT['imgs_forward']} batches={AUDIT['batch_calls']} "
          f"bwd={AUDIT['bwd_calls']} read_fail={AUDIT['read_fail']}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "ig"],
                    help="all = full G10 (I1/I2/I3); ig = only the optional input x grad "
                         "for the I2 images, patch the existing report")
    ap.add_argument("--no-ig", action="store_true", help="skip input x grad inside --stage all")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0")   # physical GPU2 (CUDA_VISIBLE_DEVICES=2)

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
    print(f"[model] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[model] device={dev_name} visible={os.environ['CUDA_VISIBLE_DEVICES']}")

    # ================================================== I0 reproduction ====
    d = np.load(PROBE_NPZ, allow_pickle=True)
    V_npz = d["V"].astype(np.float64)
    Vproj_npz = d["V_proj"].astype(np.float64)
    C_npz = d["C"]
    y = d["y"].astype(np.int64)
    paths = d["paths"]
    vids = d["vids"]
    tr_mask = d["train_mask"]
    src = (y == 0).astype(np.int64)          # 1 = fake (probe target)
    te_idx = np.where(~tr_mask)[0]
    tr_idx = np.where(tr_mask)[0]

    # LR(C=1e-3) anchor on the stored features (no forwards)
    s_lr = lr_fit_score(V_npz[tr_idx], src[tr_idx], V_npz[te_idx], C=1e-3)
    lr_full_auc = auc(src[te_idx], s_lr)
    mu_npz, e0_npz, vf_npz, _ = center_only_pca(V_npz[tr_idx])
    z_npz = (V_npz - mu_npz) @ e0_npz
    if z_npz[tr_idx][src[tr_idx] == 1].mean() < z_npz[tr_idx][src[tr_idx] == 0].mean():
        e0_npz = -e0_npz
        z_npz = -z_npz
    z_full_auc = auc(src[te_idx], z_npz[te_idx])
    print(f"[E0] npz-V PC0 varfrac={vf_npz:.4f}  LR(C=1e-3) test AUC={lr_full_auc:.4f}  "
          f"z-axis test AUC={z_full_auc:.4f}")

    # re-run the pipeline on the first 10 test images
    n_probe = 10
    repro_ids = te_idx[:n_probe]
    repro_imgs = []
    for i in repro_ids:
        rgb = read_rgb(paths[i])
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        repro_imgs.append(to_336(rgb))
    V_fresh = forward_V(model, repro_imgs, device)
    cos = [float(np.dot(V_fresh[k], V_npz[i]) / (np.linalg.norm(V_fresh[k]) * np.linalg.norm(V_npz[i]) + 1e-12))
           for k, i in enumerate(repro_ids)]
    cos_proj = [float(np.dot(V_fresh[k], Vproj_npz[i]) / (np.linalg.norm(V_fresh[k]) * np.linalg.norm(Vproj_npz[i]) + 1e-12))
                for k, i in enumerate(repro_ids)]
    repro_min = float(np.min(cos))
    repro_min_proj = float(np.min(cos_proj))
    # absolute relative L2 error vs the larger-magnitude reference
    rel = [float(np.linalg.norm(V_fresh[k] - V_npz[i]) / (np.linalg.norm(V_npz[i]) + 1e-12))
           for k, i in enumerate(repro_ids)]
    print(f"[I0] repro cos(V_fresh, npz V) min={repro_min:.6f} mean={np.mean(cos):.6f} "
          f"| vs npz V_proj min={repro_min_proj:.6f} | relL2 max={max(rel):.4e}")

    repro_ok = repro_min >= 0.999

    # ============================================== E0 axis on npz V =======
    mu = mu_npz
    e0 = e0_npz
    z_all = (V_npz - mu) @ e0
    if z_all[tr_idx][src[tr_idx] == 1].mean() < z_all[tr_idx][src[tr_idx] == 0].mean():
        e0, z_all = -e0, -z_all
    z_tr, z_te = z_all[tr_idx], z_all[te_idx]
    y_tr, y_te = src[tr_idx], src[te_idx]

    if args.stage == "ig":
        run_ig_stage(REPORT, model, device, te_idx, y, vids, paths, src, mu, e0)
        return

    # ======================================================== I1 ==========
    i1_ids = stratified_pick(te_idx, y, vids, 120, seed=SEED + 1)
    variant_names = ["orig", "gray", "jpeg70", "resize2x", "flip", "blur", "crop90"]
    V_var = {k: [] for k in variant_names}
    i1_lab, i1_vid = [], []
    for c, i in enumerate(i1_ids):
        rgb = read_rgb(paths[i])
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        vs = make_variants(rgb)
        imgs = [to_336(vs[k]) for k in variant_names]
        feats = forward_V(model, imgs, device)
        for k, f in zip(variant_names, feats):
            V_var[k].append(f)
        i1_lab.append(src[i])
        i1_vid.append(vids[i])
        if (c + 1) % 20 == 0:
            print(f"  [I1] {c+1}/{len(i1_ids)}  imgs_fwd={AUDIT['imgs_forward']}", flush=True)
    i1_lab = np.array(i1_lab)
    V_var = {k: np.array(v) for k, v in V_var.items()}
    z_var = {k: (V_var[k] - mu) @ e0 for k in variant_names}

    z_orig_auc = auc(i1_lab, z_var["orig"])
    z_auc = {k: auc(i1_lab, z_var[k]) for k in variant_names}
    # LR(C=1e-3) trained on probe train, applied to the 120 originals/variants
    lr_s = lr_fit_score(V_npz[tr_idx], y_tr, V_var["orig"], C=1e-3)
    lr_auc = {"orig": auc(i1_lab, lr_s)}
    for k in variant_names[1:]:
        lr_auc[k] = auc(i1_lab, lr_fit_score(V_npz[tr_idx], y_tr, V_var[k], C=1e-3))
    delta_z = {k: z_orig_auc - z_auc[k] for k in variant_names[1:]}
    delta_lr = {k: lr_auc["orig"] - lr_auc[k] for k in variant_names[1:]}
    print("[I1] z-AUC:", {k: round(v, 4) for k, v in z_auc.items()})
    print("[I1] LR-AUC:", {k: round(v, 4) for k, v in lr_auc.items()})
    print("[I1] dAUC(z):", {k: round(v * 100, 2) for k, v in delta_z.items()})

    low_level = ["gray", "jpeg70", "resize2x", "flip", "blur"]
    max_pt_z = max(delta_z[k] for k in low_level) * 100.0
    max_pt_lr = max(delta_lr[k] for k in low_level) * 100.0
    if max_pt_z >= 15.0:
        verdict = "AXIS_SENSITIVE_TO_LOWLEVEL"
    elif max_pt_z <= 5.0:
        verdict = "AXIS_NOT_LOWLEVEL"
    else:
        verdict = "MIXED"

    # ======================================================== I2 ==========
    i2_ids = stratified_pick(te_idx, y, vids, 40, seed=SEED + 2)
    dz2, zo2 = occ_heatmaps(model, device, i2_ids, y, paths, mu, e0, "I2")
    lab2 = np.array([src[i] for i in i2_ids])
    dz2_f = np.where(lab2[:, None] == 1, dz2, np.nan)     # real
    dz2_k = np.where(lab2[:, None] == 0, dz2, np.nan)     # fake
    H_real = np.nanmean(np.abs(dz2_f), axis=0).reshape(GRID, GRID)
    H_fake = np.nanmean(np.abs(dz2_k), axis=0).reshape(GRID, GRID)
    top_r, share_r, cen_r = hot_stats(H_real)
    top_f, share_f, cen_f = hot_stats(H_fake)
    sum_r = float(np.nansum(np.abs(dz2_f)))
    sum_f = float(np.nansum(np.abs(dz2_k)))
    signed_r = np.nanmean(dz2_f, axis=0).reshape(GRID, GRID)
    signed_f = np.nanmean(dz2_k, axis=0).reshape(GRID, GRID)
    print(f"[I2] real top={fmt_blocks(top_r)} cen={cen_r:.3f} | fake top={fmt_blocks(top_f)} cen={cen_f:.3f}")

    # ---- optional input x grad on the same 40 images ----
    ig_summary, corr_real, corr_fake = "inputXgrad=not_run", float("nan"), float("nan")
    if not args.no_ig:
        try:
            grads = input_grad_heatmaps(model, device, i2_ids, paths, mu, e0)
            gm_real = np.nanmean(np.where(lab2[:, None] == 1, grads, np.nan), axis=0)
            gm_fake = np.nanmean(np.where(lab2[:, None] == 0, grads, np.nan), axis=0)
            corr_real = float(np.dot(_norm(np.abs(dz2_f).mean(axis=0)), _norm(gm_real)))
            corr_fake = float(np.dot(_norm(np.abs(dz2_k).mean(axis=0)), _norm(gm_fake)))
            top_ig = list(np.argsort(gm_fake)[::-1][:5])
            ig_summary = (f"inputXgrad=ok cos_occ_vs_ig real={corr_real:.3f} fake={corr_fake:.3f} "
                          f"top_blocks_fake={fmt_blocks(top_ig)} "
                          f"center16_fake={gm_fake.reshape(GRID,GRID)[2:6,2:6].sum()/gm_fake.sum():.4f}")
        except Exception as e:                               # pragma: no cover
            ig_summary = f"inputXgrad=failed ({type(e).__name__}: {e})"
    print(f"[I2] {ig_summary}")

    # ======================================================== I3 ==========
    dm = np.load(MULTI_NPZ, allow_pickle=True)
    m_paths, m_y, m_dom, m_vid = dm["path"], dm["y"].astype(np.int64), dm["domain"], dm["vid"]
    m_src = (m_y == 0).astype(np.int64)
    i3 = {}
    for dom, seed_off in (("cd1", 11), ("ffiw", 12)):
        di = np.where(m_dom == dom)[0]
        picks = stratified_pick(di, m_src, m_vid, 40, seed=SEED + seed_off)
        dzd, zod = occ_heatmaps(model, device, picks, m_src, m_paths, mu, e0, f" I3-{dom}")
        i3[dom] = {"dz": dzd, "lab": np.array([m_src[i] for i in picks]), "ids": picks}
    # ffpp source-in reference = I2
    i3["ffpp"] = {"dz": dz2, "lab": lab2, "ids": i2_ids}
    i3_stats = {}
    for dom in ("ffpp", "cd1", "ffiw"):
        H = np.nanmean(np.abs(i3[dom]["dz"]), axis=0).reshape(GRID, GRID)
        tp, sh, cf = hot_stats(H)
        i3_stats[dom] = {"H": H, "top5": tp, "share": sh, "cen": cf}
        print(f"[I3] {dom:5s} top5={fmt_blocks(tp)} share={sh:.3f} center16={cf:.3f}")
    # fake-only breakdown (real/fake split)
    i3_split = {}
    for dom in ("ffpp", "cd1", "ffiw"):
        lab = i3[dom]["lab"]
        dzs = i3[dom]["dz"]
        Hk = np.nanmean(np.where(lab[:, None] == 0, np.abs(dzs), np.nan), axis=0).reshape(GRID, GRID)
        Hr = np.nanmean(np.where(lab[:, None] == 1, np.abs(dzs), np.nan), axis=0).reshape(GRID, GRID)
        i3_split[dom] = {"real": hot_stats(Hr), "fake": hot_stats(Hk)}

    # =================================================== audit / report ====
    peak = gpu_peak_mib()
    try:
        peak_torch = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    except Exception:
        peak_torch = float("nan")
    dt = time.time() - t0

    def f4(x):
        return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

    lines = []
    A = lines.append
    A("=" * 78)
    A("G10  IN-MODEL ATTRIBUTION EXPERIMENT  (M2F2-Det Stage-1 detector, bridge_v2_phase1)")
    A("=" * 78)
    A("")
    A("MACHINE BLOCK")
    A(f"G10_REPRO_COS_MIN={repro_min:.6f} "
      f"G10_I1_ORIG_AUC={f4(z_orig_auc)} "
      f"G10_I1_DELTA_gray={delta_z['gray']*100:.2f} "
      f"_jpeg70={delta_z['jpeg70']*100:.2f} "
      f"_resize2x={delta_z['resize2x']*100:.2f} "
      f"_flip={delta_z['flip']*100:.2f} "
      f"_blur={delta_z['blur']*100:.2f} "
      f"_crop90={delta_z['crop90']*100:.2f} "
      f"G10_I2_TOPPATCH_real=({top_r[0]//GRID},{top_r[0]%GRID}) "
      f"G10_I2_TOPPATCH_fake=({top_f[0]//GRID},{top_f[0]%GRID}) "
      f"G10_I2_CENTER_FRAC={cen_r:.4f}/{cen_f:.4f} "
      f"G10_I3_CD1_TOP={fmt_blocks(i3_stats['cd1']['top5'][:3])} "
      f"G10_I3_FFIW_TOP={fmt_blocks(i3_stats['ffiw']['top5'][:3])} "
      f"G10_FORWARDS_USED={AUDIT['imgs_forward']} "
      f"G10_GPU=2 "
      f"G10_VERDICT={verdict}")
    A(f"G10_EXTRA: repro_cos_vs_npzV={repro_min:.6f} repro_cos_vs_npzVproj={repro_min_proj:.6f} "
      f"repro_relL2_max={max(rel):.3e} PC0_varfrac={vf_npz:.4f} "
      f"LR_C1e-3_fulltest_AUC={lr_full_auc:.4f} z_axis_fulltest_AUC={z_full_auc:.4f} "
      f"dAUC_LR_gray={delta_lr['gray']*100:.2f} _jpeg70={delta_lr['jpeg70']*100:.2f} "
      f"_resize2x={delta_lr['resize2x']*100:.2f} _flip={delta_lr['flip']*100:.2f} "
      f"_blur={delta_lr['blur']*100:.2f} _crop90={delta_lr['crop90']*100:.2f} "
      f"max_lowlevel_dAUC_z={max_pt_z:.2f}pt max_lowlevel_dAUC_LR={max_pt_lr:.2f}pt "
      f"I2_sum_absdz_real={sum_r:.3f} I2_sum_absdz_fake={sum_f:.3f} "
      f"I3_CD1_CENTER={i3_stats['cd1']['cen']:.4f} I3_FFIW_CENTER={i3_stats['ffiw']['cen']:.4f} "
      f"FORWARD_BATCH_CALLS={AUDIT['batch_calls']} BWD_CALLS={AUDIT['bwd_calls']} "
      f"READ_FAIL={AUDIT['read_fail']} GPU_SMI_PEAK_MiB={peak:.0f} "
      f"TORCH_PEAK_MiB={peak_torch:.0f} WALL_S={dt:.1f}")
    A("")
    A("-" * 78)
    A("I0 REPRODUCTION CHECK")
    A("-" * 78)
    A("Pipeline: cv2.imread -> BGR2RGB -> cv2.resize(336) -> albumentations "
      "Normalize(CLIP mean/std) -> ToTensorV2 ->")
    A("          model._preprocess_for_vit (un-norm, F.interpolate 224, /0.5-1) -> "
      "vit.forward_features -> CLS[:,0]")
    A("_preprocess_for_vit uses torch F.interpolate => DIFFERENTIABLE (input x grad is possible).")
    A(f"10 test images (video-disjoint), cosine(V_fresh, probe_feats.npz['V']) : "
      f"min={repro_min:.6f}  mean={np.mean(cos):.6f}")
    A(f"same, vs probe_feats.npz['V_proj']                               : "
      f"min={repro_min_proj:.6f}")
    A(f"relative L2 error vs npz['V'] (max over 10)                      : {max(rel):.3e}")
    A(f"REPRO {'PASS (>=0.999)' if repro_ok else 'FAIL (<0.999)'}; "
      f"the npz 'V' key holds the RAW ViT CLS (768-d, pre-deepfake_proj).")
    A("")
    A("-" * 78)
    A("E0 AXIS  (center-only PCA on probe-train V; E3/G8B convention)")
    A("-" * 78)
    A(f"mu = mean(V[train_mask])   e0 = 1st right singular vector of centered V[train]")
    A(f"PC0 variance fraction = {vf_npz:.4f}   (anchor reported by the probe work: 0.6229)")
    A(f"sign fixed so that mean z(fake,train) > mean z(real,train); z(x) = (V(x) - mu) . e0")
    A(f"z-axis AUC on full test 800 (from npz) = {z_full_auc:.4f}   "
      f"LR(C=1e-3) probe AUC on full test 800 = {lr_full_auc:.4f}")
    A(f"train n={len(tr_idx)} (real {int((y_tr==1).sum())}, fake {int((y_tr==0).sum())})  "
      f"test n={len(te_idx)} (real {int((y_te==1).sum())}, fake {int((y_te==0).sum())}); "
      f"target = fake (y==0)")
    A("")
    A("-" * 78)
    A("I1 TRANSFORMATION INVARIANCE  (n=120 stratified video-disjoint FF++ test; 60 real / 60 fake)")
    A("-" * 78)
    A("variants are applied on the native-resolution RGB uint8 image, then the 336 pipeline")
    A("")
    A(f"{'variant':<10} {'z-AUC':>8} {'dAUC(z)pt':>10} {'LR-AUC':>8} {'dAUC(LR)pt':>11} {'sign':>6}")
    for k in variant_names:
        tag = "control" if k == "crop90" else ("orig" if k == "orig" else "low-level")
        A(f"{k:<10} {z_auc[k]:>8.4f} {delta_z.get(k,0.0)*100:>10.2f} {lr_auc[k]:>8.4f} "
          f"{delta_lr.get(k,0.0)*100:>11.2f} {tag:>6}")
    A("")
    A(f"max |dAUC| over low-level variants: z-axis {max_pt_z:.2f}pt   LR {max_pt_lr:.2f}pt")
    A(f"crop90 control dAUC: z-axis {delta_z['crop90']*100:.2f}pt   LR {delta_lr['crop90']*100:.2f}pt")
    A(f"MECHANICAL VERDICT (rule: any low-level dAUC >= 15pt -> sensitive; all <= 5pt -> not "
      f"low-level; else mixed): {verdict}")
    A("")
    A("-" * 78)
    A("I2 SPATIAL OCCLUSION ATTRIBUTION  (ffpp video-disjoint test, n=40, 20 real / 20 fake)")
    A("-" * 78)
    A("8x8 grid over the 336 image (42x42 px per patch); patch filled with the per-image global")
    A("mean RGB colour; attribution value = z(masked) - z(orig); heatmaps = mean |dz| per class.")
    A("")
    A("[real] mean|dz| heat map (rows 0..7 top->bottom)   sum|dz| = %.3f" % sum_r)
    for r in range(GRID):
        A("   " + " ".join(f"{H_real[r,c]:7.4f}" for c in range(GRID)))
    A("")
    A("[fake] mean|dz| heat map (rows 0..7 top->bottom)   sum|dz| = %.3f" % sum_f)
    for r in range(GRID):
        A("   " + " ".join(f"{H_fake[r,c]:7.4f}" for c in range(GRID)))
    A("")
    A(f"real : top-1 block (r,c) = ({top_r[0]//GRID},{top_r[0]%GRID})  "
      f"top5 = {fmt_blocks(top_r)}  top5 share = {share_r:.4f}  center16/64 share = {cen_r:.4f}")
    A(f"fake : top-1 block (r,c) = ({top_f[0]//GRID},{top_f[0]%GRID})  "
      f"top5 = {fmt_blocks(top_f)}  top5 share = {share_f:.4f}  center16/64 share = {cen_f:.4f}")
    A("center16 = rows 2..5 x cols 2..5 (face/central region); edge = remaining 48 blocks")
    A("")
    A("signed mean dz (diagnostic only, not used for the verdict):")
    A("[real signed]")
    for r in range(GRID):
        A("   " + " ".join(f"{signed_r[r,c]:+7.4f}" for c in range(GRID)))
    A("[fake signed]")
    for r in range(GRID):
        A("   " + " ".join(f"{signed_f[r,c]:+7.4f}" for c in range(GRID)))
    A("")
    A(f"optional input x grad: {ig_summary}")
    A("")
    A("-" * 78)
    A("I3 SOURCE-IN (ffpp) VS CROSS-DOMAIN (cd1, ffiw) ATTRIBUTION")
    A("-" * 78)
    A("same occlusion protocol, n=40 per domain (20 real / 20 fake), 8x8 grid")
    A("")
    A(f"{'domain':<8} {'center16 share':>15} {'top5 share':>11}  top-5 blocks (r,c)")
    for dom in ("ffpp", "cd1", "ffiw"):
        st = i3_stats[dom]
        A(f"{dom:<8} {st['cen']:>15.4f} {st['share']:>11.4f}  {fmt_blocks(st['top5'])}")
    A("")
    A(f"{'domain':<8} {'class':<6} {'center16':>9} {'top1':>8}  top-5 blocks")
    for dom in ("ffpp", "cd1", "ffiw"):
        for cls in ("real", "fake"):
            tp, sh, cf = i3_split[dom][cls]
            A(f"{dom:<8} {cls:<6} {cf:>9.4f} {tp[0]:>3d}({tp[0]//GRID},{tp[0]%GRID})  {fmt_blocks(tp)}")
    A("")
    for dom in ("cd1", "ffiw"):
        A(f"[{dom}] mean|dz| heat map")
        for r in range(GRID):
            A("   " + " ".join(f"{i3_stats[dom]['H'][r,c]:7.4f}" for c in range(GRID)))
        A("")
    A("")
    A("-" * 78)
    A("AUDIT")
    A("-" * 78)
    A(f"GPU                      : physical GPU 2 (CUDA_VISIBLE_DEVICES=2) -> {dev_name}")
    A(f"forwards (images through the ViT branch) : {AUDIT['imgs_forward']} "
      f"in {AUDIT['batch_calls']} batch calls (batch<=16, fp32, no_grad except inputXgrad)")
    A(f"backward calls (input x grad)            : {AUDIT['bwd_calls']}")
    A(f"batch-equivalents (imgs/16)              : {AUDIT['imgs_forward']/16.0:.1f}")
    A(f"image read failures                      : {AUDIT['read_fail']}")
    A(f"GPU memory used, max nvidia-smi sample   : {peak:.0f} MiB "
      f"(torch max_memory_allocated {peak_torch:.0f} MiB)")
    A(f"wall clock                               : {dt:.1f} s")
    A(f"sampling seeds                           : base {SEED} (I1 +1, I2 +2, cd1 +11, ffiw +12)")
    A("")
    A("CAVEATS (mandatory)")
    A("  * subsets only: 120 images for I1 and 40 images per domain for I2/I3, not the full")
    A("    800-sample probe-test; AUCs on n=120 carry wide binomial error (~+/-4-6pt).")
    A("  * occlusion fill is a single human choice (per-image global mean colour); a different")
    A("    fill (noise / blur / black) changes the heat-map values and possibly the top blocks.")
    A("  * transformation parameters are single fixed settings (JPEG q=70, 5x5 sigma=1 blur,")
    A("    1/2 downsample, 90% centre crop); other strengths may shift dAUC.")
    A("  * model is the Stage-1 checkpoint checkpoints/stage_1/bridge_v2_phase1.pth, the same")
    A("    weights the probe analysis used; results do not transfer to later stages.")
    A("  * batch<=16 fp32, no_grad except the 40 input x grad backward passes.")
    A("=" * 78)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[save] report -> {REPORT}")
    print("\n===== SUMMARY =====")
    print(f"repro_cos_min={repro_min:.6f}  PC0_varfrac={vf_npz:.4f}  z-AUC full test={z_full_auc:.4f}")
    print(f"I1 z-AUC orig={z_orig_auc:.4f}  dAUC(pt)=" +
          " ".join(f"{k}:{delta_z[k]*100:.2f}" for k in variant_names[1:]))
    print(f"I2 top real=({top_r[0]//GRID},{top_r[0]%GRID}) cen={cen_r:.3f}  "
          f"fake=({top_f[0]//GRID},{top_f[0]%GRID}) cen={cen_f:.3f}")
    print(f"I3 center: ffpp={i3_stats['ffpp']['cen']:.3f} cd1={i3_stats['cd1']['cen']:.3f} "
          f"ffiw={i3_stats['ffiw']['cen']:.3f}")
    print(f"VERDICT={verdict}  forwards={AUDIT['imgs_forward']} "
          f"batches={AUDIT['batch_calls']}  peak_smi={peak:.0f}MiB")


if __name__ == "__main__":
    main()
