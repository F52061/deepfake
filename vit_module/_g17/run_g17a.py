# -*- coding: utf-8 -*-
"""
G17a -- low-level / frequency-domain descriptor extraction (pure CPU, offline).

Science question (context only, no classifier here): the ViT CLS branch V (768-d)
transfers well cross-domain (mean4 AUC=0.8318) but its discriminative axis is
geometric/structural, NOT low-level pixel-level (5 low-level transforms delta<=3.69pt
vs 17.25pt for geometric crop).  We now extract frequency-domain / low-level
descriptors that are more sensitive to local detail, so a later stage (G17c) can
test whether they carry the class-discriminative signal V is missing.

This script ONLY extracts features + self-check.  It does NOT train any classifier.

Image pipeline (identical to V / g10 / g16):
    cv2.imread(path) -> cv2.cvtColor(BGR2RGB) -> cv2.resize(336,336)
Features are computed on the resized RGB (uint8 0-255) as the main scope, and
(CLIP-normalized tensor) as a sensitivity-check scope.

Rows: 5300 = probe_feats.npz (3000, in file order) + feats_multi.npz (2300, in file order).
y encoding: 1=real, 0=fake.

Resource discipline: single process, OMP/MKL/OPENBLAS/NUMEXPR=4, cv2 threads=2,
concurrent.futures ThreadPoolExecutor max_workers=2.  Zero GPU.
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["BLIS_NUM_THREADS"] = "4"

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from scipy.fft import dctn
cv2.setNumThreads(2)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
FREQ_NPZ = os.path.join(HERE, "freq_feats.npz")
LOG = os.path.join(HERE, "run_log_g17a.txt")

IMG_SIZE = 336
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float64)

SEED = 20260910
N_REF_EDGES = 400          # images pooled for the global SRM residual quantile edges
N_SENS = 300               # images for the CLIP-normalization sensitivity check

# ------------------------------------------------------------------ logging ----
class Logger:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")
    def __call__(self, msg):
        print(msg, flush=True)
        self.fh.write(str(msg) + "\n")
        self.fh.flush()
    def close(self):
        self.fh.close()

log = Logger(LOG)

# ------------------------------------------------------------- SRM kernels ----
# 3x3 high-pass residuals, first/second/third order (5 kernels).
#  - s1_h / s1_v : first-order directional differences (edge detectors).
#  - s2_h / s2_v : second-order directional derivatives.
#  - s3_d        : third-order difference along the main diagonal.
# Each kernel sums to zero (pure high-pass).
SRM_KERNELS = [
    ("s1_h", np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], np.float32)),
    ("s1_v", np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], np.float32)),
    ("s2_h", np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], np.float32)),
    ("s2_v", np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], np.float32)),
    ("s3_d", np.array([[0, 0, 0], [-1, 3, -3], [0, 0, 1]], np.float32)),
]

# ----------------------------------------------------------------- helpers ----
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return rgb


def normalize_clip(rgb_u8):
    """uint8 RGB (H,W,3) -> CLIP-normalized float32 tensor."""
    x = rgb_u8.astype(np.float32) / 255.0
    x = (x - CLIP_MEAN.astype(np.float32)) / CLIP_STD.astype(np.float32)
    return x.astype(np.float32)


def gray_of(rgb_f):
    """float RGB (H,W,3) -> luma float (0.299/0.587/0.114)."""
    return (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1]
            + 0.114 * rgb_f[:, :, 2]).astype(np.float32)


def mom4(x):
    """raw moments about the mean: (mean, var, skew, kurt) on float64.
    kurt is the raw 4th standardized moment (NOT excess; no -3)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = x.mean()
    d = x - m
    v = float((d * d).mean())
    s = math.sqrt(v)
    sk = float((d ** 3).mean()) / (s ** 3) if s > 1e-9 else 0.0
    ku = float((d ** 4).mean()) / (s ** 4) if s > 1e-9 else 0.0
    return float(m), float(v), sk, ku


def sobel_mag(img_f):
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)


def top_frac(mag, q=0.99):
    thr = float(np.quantile(mag, q))
    return float((mag >= thr).mean())


def hist_fixed(x, edges):
    """32-bin histogram of x using FIXED global quantile edges (33 values).
    Robust to duplicate edges (returns exactly 32 bins)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    n = len(edges) - 1
    interior = edges[1:-1]
    idx = np.digitize(x, interior)          # 0..n-1
    counts = np.bincount(idx, minlength=n)[:n]
    s = counts.sum()
    return (counts.astype(np.float32) / s) if s > 0 else np.zeros(n, np.float32)


def zigzag_8():
    idx = []
    n = 8
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                idx.append((i, s - i))
        else:
            for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                idx.append((i, s - i))
    return idx

ZZ = zigzag_8()          # 64 entries, index 0 = DC (0,0)
ZZ_AC = ZZ[1:]           # 63 AC coefficients in zigzag order


def logpolar_profile(logmag, n_rad=32, n_ang=8, oversample=4):
    """Log-polar resampling of the (shifted) log-power spectrum.
    Radial axis is log-spaced (r from 1 to rmax); samples are bilinearly
    interpolated onto a dense (n_rad*over x n_ang*over) log-polar grid, then
    averaged to (n_rad,) radial and (n_ang,) angular profiles."""
    H, W = logmag.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    rmax = min(cy, cx) - 0.5                       # stay inside inscribed circle
    nr = n_rad * oversample
    na = n_ang * oversample
    radii = np.exp(np.linspace(0.0, np.log(rmax), nr))     # log-spaced
    angles = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
    A, R = np.meshgrid(angles, radii)              # (nr, na)
    X = cx + R * np.cos(A)
    Y = cy + R * np.sin(A)
    samp = map_coordinates(logmag, [Y.ravel(), X.ravel()], order=1,
                           mode="constant", cval=0.0).reshape(nr, na)
    radial = samp.mean(axis=1).reshape(n_rad, oversample).mean(axis=1)
    angular = samp.mean(axis=0).reshape(n_ang, oversample).mean(axis=1)
    return radial.astype(np.float32), angular.astype(np.float32)


# ------------------------------------------------------------- core extract ----
def _extract(rgb_f):
    """rgb_f: float32 (336,336,3). Returns dict of float32 feature vectors."""
    H, W = rgb_f.shape[:2]
    gray = gray_of(rgb_f)

    # ---- shared FFT ----
    F = np.fft.fftshift(np.fft.fft2(gray))          # complex128
    mag = np.abs(F)                                 # float64
    logmag = np.log1p(mag).astype(np.float64)

    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    dy = (yy - cy).astype(np.float64)
    dx = (xx - cx).astype(np.float64)
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)                       # [-pi, pi]
    rmax = float(min(cx, cy))                        # 168

    # ---- F1: log-polar radial(32) + angular(8) ----
    radial, angular = logpolar_profile(logmag)
    F1 = np.concatenate([radial, angular])           # 40
    F1_radial = radial                              # 32 (also stored separately)

    # ---- F2: spectrum statistics (4) ----
    p = mag ** 2                                     # power, float64
    fx = dx / W
    fy = dy / H
    f = np.sqrt(fx * fx + fy * fy)
    tot = p[r >= 1.0].sum()                          # exclude DC
    hf25 = float(p[f > 0.25].sum() / tot) if tot > 0 else 0.0
    hf50 = float(p[f > 0.5].sum() / tot) if tot > 0 else 0.0
    r_int = np.round(r).astype(np.int64)
    valid = (r_int >= 1) & (r_int <= int(rmax))
    rv = r_int[valid].ravel()
    lv = logmag[valid].ravel()
    sums = np.bincount(rv, weights=lv, minlength=int(rmax) + 1)[1:int(rmax) + 1]
    cnts = np.bincount(rv, minlength=int(rmax) + 1)[1:int(rmax) + 1]
    ok = cnts > 0
    freqs = np.arange(1, int(rmax) + 1)[ok].astype(np.float64)
    lp = sums[ok] / cnts[ok]
    slope = float(np.polyfit(np.log(freqs), lp, 1)[0]) if len(freqs) >= 2 else 0.0
    p_ex = p[r >= 1.0]
    if p_ex.size > 0 and p_ex.mean() > 0:
        flat = float(np.exp(np.mean(np.log(p_ex + 1e-12))) / p_ex.mean())
    else:
        flat = 0.0
    F2 = np.array([hf25, hf50, slope, flat], np.float32)     # 4

    # ---- F3: SRM residual stats (5 kernels x 4 = 20) + fixed-edge hist ----
    F3 = np.zeros(len(SRM_KERNELS) * 4, np.float32)
    F3_hist = np.zeros(len(SRM_KERNELS) * 32, np.float32)
    for ki, (_, kern) in enumerate(SRM_KERNELS):
        resid = np.concatenate([cv2.filter2D(rgb_f[:, :, c], -1, kern).ravel()
                                for c in range(3)])
        m, v, sk, ku = mom4(resid)
        F3[ki * 4:ki * 4 + 4] = (m, v, sk, ku)
        if EDGES is not None:
            F3_hist[ki * 32:ki * 32 + 32] = hist_fixed(resid, EDGES[ki])

    # ---- F4: color/chroma discontinuity (23) ----
    R, G, B = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    Yc = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Yc) * 0.713
    Cb = (B - Yc) * 0.564
    mag_Cr = sobel_mag(Cr)
    mag_Cb = sobel_mag(Cb)
    mag_RG = sobel_mag(R - G)
    mag_GB = sobel_mag(G - B)
    mag_RB = sobel_mag(R - B)
    mag_Y = sobel_mag(Yc)
    mag_color = np.maximum(np.maximum(sobel_mag(R), sobel_mag(G)), sobel_mag(B))
    f4 = []
    for m_ in (mag_Cr, mag_Cb):
        f4 += [float(m_.mean()), float(m_.var()),
               float(np.quantile(m_, 0.90)), float(np.quantile(m_, 0.99))]
    for m_ in (mag_RG, mag_GB, mag_RB):
        f4 += [float(m_.mean()), float(m_.var())]
    f4 += [float(mag_Y.mean()), float(mag_Y.var())]
    f4 += [top_frac(mag_color, 0.99), top_frac(mag_Y, 0.99)]
    f4 += [float(np.abs(Cr - np.median(Cr)).mean()),
           float(np.abs(Cb - np.median(Cb)).mean())]
    for m_ in (mag_RG, mag_GB, mag_RB):
        f4.append(float(np.quantile(m_, 0.95)))
    F4 = np.array(f4, np.float32)                        # 23

    # ---- F5: 8x8 block DCT statistics (4) ----
    blocks = gray.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    C = dctn(blocks.astype(np.float64), axes=(1, 2), norm="ortho")
    c_flat = C.reshape(-1, 64)
    dc = c_flat[:, 0]
    ac_energy = (c_flat ** 2).sum(axis=1) - dc ** 2
    total_energy = (c_flat ** 2).sum(axis=1)
    ac_ratio = float(ac_energy.sum() / total_energy.sum()) if total_energy.sum() > 0 else 0.0
    low_idx = [ZZ.index(z) for z in ZZ_AC[:10]]
    high_idx = [ZZ.index(z) for z in ZZ_AC[10:]]
    low_e = float((c_flat[:, low_idx] ** 2).sum())
    high_e = float((c_flat[:, high_idx] ** 2).sum())
    low_frac = float(low_e / (low_e + high_e)) if (low_e + high_e) > 0 else 0.0
    per_block = ac_energy / np.maximum(1e-12, total_energy)
    F5 = np.array([ac_ratio, low_frac,
                   float(per_block.mean()), float(per_block.std())], np.float32)  # 4

    return {
        "F1": F1, "F1_radial": F1_radial, "F2": F2,
        "F3": F3, "F3_hist": F3_hist, "F4": F4, "F5": F5,
    }


EDGES = None          # filled by build_reference_edges() before extraction


def build_reference_edges():
    """Pool SRM residuals over N_REF_EDGES images -> per-kernel 33 quantile edges."""
    p = np.load(PROBE_NPZ, allow_pickle=True)
    m = np.load(MULTI_NPZ, allow_pickle=True)
    paths = np.concatenate([p["paths"], m["path"]])
    # first 200 probe + first 200 multi
    idx = np.concatenate([np.arange(0, 200), np.arange(3000, 3200)])
    N = len(idx)
    npx = IMG_SIZE * IMG_SIZE
    pools = [np.empty((N, 3 * npx), np.float32) for _ in SRM_KERNELS]
    for n_, i in enumerate(idx):
        rgb = read_rgb(str(paths[i]))
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        rf = rgb.astype(np.float32)
        for ki, (_, kern) in enumerate(SRM_KERNELS):
            resid = np.concatenate([cv2.filter2D(rf[:, :, c], -1, kern).ravel()
                                    for c in range(3)])
            pools[ki][n_] = resid
    edges = []
    for ki in range(len(SRM_KERNELS)):
        edges.append(np.quantile(pools[ki].ravel(), np.linspace(0.0, 1.0, 33)).astype(np.float64))
    return edges


def work(path):
    rgb = read_rgb(str(path))
    if rgb is None:
        return _extract(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)), False
    return _extract(rgb.astype(np.float32)), True


# -------------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---- load sources ----
    p = np.load(PROBE_NPZ, allow_pickle=True)
    m = np.load(MULTI_NPZ, allow_pickle=True)
    paths_p = p["paths"].astype(str)
    vids_p = p["vids"].astype(str)
    y_p = p["y"].astype(np.int64)
    tr_mask = p["train_mask"].astype(bool)
    paths_m = m["path"].astype(str)
    vids_m = m["vid"].astype(str)
    y_m = m["y"].astype(np.int64)
    dom_m = m["domain"].astype(str)

    n_p = len(paths_p)
    n_m = len(paths_m)
    n_total = n_p + n_m
    assert n_p == 3000 and n_m == 2300 and n_total == 5300, (n_p, n_m)
    assert int(tr_mask.sum()) == 2200

    paths_all = np.concatenate([paths_p, paths_m]).astype(object)
    y_all = np.concatenate([y_p, y_m]).astype(np.int64)
    vids_all = np.concatenate([vids_p, vids_m]).astype(object)
    domain_all = np.concatenate([np.repeat("ffpp_probe", n_p).astype(object),
                                 dom_m.astype(object)])
    split_all = np.concatenate([
        np.where(tr_mask, "train", "test").astype(object),
        dom_m.astype(object),
    ])
    source_all = np.concatenate([np.repeat("probe", n_p).astype(object),
                                 np.repeat("multi", n_m).astype(object)])

    # ---- row-order alignment hard gate (must match G17b exactly) ----
    align_ok = True
    if not (paths_all[:n_p].astype(object) == p["paths"].astype(object)).all():
        align_ok = False
        log("[align] FAIL: probe paths != source probe_feats.npz['paths']")
    if not (paths_all[n_p:].astype(object) == m["path"].astype(object)).all():
        align_ok = False
        log("[align] FAIL: multi paths != source feats_multi.npz['path']")
    if not (y_all == np.concatenate([p["y"], m["y"]])).all():
        align_ok = False
        log("[align] FAIL: y mismatch")
    if not (vids_all[:n_p].astype(object) == p["vids"].astype(object)).all():
        align_ok = False
        log("[align] FAIL: probe vids mismatch")
    if not (vids_all[n_p:].astype(object) == m["vid"].astype(object)).all():
        align_ok = False
        log("[align] FAIL: multi vids mismatch")
    if not (domain_all[:n_p] == "ffpp_probe").all():
        align_ok = False
        log("[align] FAIL: probe domain != ffpp_probe")
    if not align_ok:
        log("[align] ABORT: row-order alignment check failed")
        log.close()
        sys.exit(1)
    log("[align] row order = probe native 3000 + multi native 2300; "
        "paths/y/vids/domain == source npz element-wise PASS")

    log("=" * 90)
    log("G17a -- low-level / frequency-domain descriptor extraction")
    log("=" * 90)
    log(f"n_total={n_total}  (probe={n_p} + multi={n_m})")
    log(f"threads: OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS=4, cv2.setNumThreads(2), "
        f"ThreadPoolExecutor max_workers=2, single process")
    log(f"pipeline: cv2.imread -> BGR2RGB -> cv2.resize(336,336) -> uint8 RGB (0-255)")
    log(f"sensitivity scope: CLIP-normalized tensor (mean/std) -> same _extract")
    log(f"N_REF_EDGES={N_REF_EDGES} N_SENS={N_SENS}")

    # ---- reference quantile edges for F3_hist ----
    t_ref = time.time()
    global EDGES
    EDGES = build_reference_edges()
    log(f"[ref] SRM residual quantile edges built from {N_REF_EDGES} imgs "
        f"in {time.time()-t_ref:.1f}s")

    # ---- main extraction ----
    t_ext = time.time()
    FEAT_KEYS = ["F1", "F1_radial", "F2", "F3", "F3_hist", "F4", "F5"]
    dims = {"F1": 40, "F1_radial": 32, "F2": 4, "F3": 20,
            "F3_hist": 160, "F4": 23, "F5": 4}
    feats = {k: np.empty((n_total, dims[k]), np.float32) for k in FEAT_KEYS}
    read_fail = 0
    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, (fd, ok) in enumerate(ex.map(work, paths_all)):
            for k in FEAT_KEYS:
                feats[k][i] = fd[k]
            if not ok:
                read_fail += 1
            done += 1
            if done % 500 == 0 or done == n_total:
                log(f"  [extract] {done}/{n_total} wall={time.time()-t_ext:.0f}s "
                    f"read_fail={read_fail}")
    wall_ext = time.time() - t_ext
    log(f"[extract] done in {wall_ext:.1f}s  read_fail={read_fail}")

    # ---- sensitivity check (CLIP-normalized vs uint8) ----
    t_sens = time.time()
    sens_p = rng.choice(n_p, size=N_SENS // 2, replace=False)
    sens_m = rng.choice(n_m, size=N_SENS - N_SENS // 2, replace=False) + n_p
    sens_idx = np.sort(np.concatenate([sens_p, sens_m]))
    A = {k: np.empty((len(sens_idx), dims[k]), np.float32) for k in FEAT_KEYS}
    B = {k: np.empty((len(sens_idx), dims[k]), np.float32) for k in FEAT_KEYS}
    for n_, i in enumerate(sens_idx):
        rgb = read_rgb(str(paths_all[i]))
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        a = _extract(rgb.astype(np.float32))
        b = _extract(normalize_clip(rgb))
        for k in FEAT_KEYS:
            A[k][n_] = a[k]
            B[k][n_] = b[k]
    sens_summary = {}
    for k in FEAT_KEYS:
        rs = []
        for j in range(dims[k]):
            x = A[k][:, j].astype(np.float64)
            y = B[k][:, j].astype(np.float64)
            if x.std() > 1e-12 and y.std() > 1e-12:
                rs.append(float(np.corrcoef(x, y)[0, 1]))
            else:
                rs.append(float("nan"))
        rs = np.array(rs)
        rs_f = rs[np.isfinite(rs)]
        sens_summary[k] = (float(rs_f.mean()) if rs_f.size else float("nan"),
                           float(rs_f.min()) if rs_f.size else float("nan"))
        log(f"  [sens] {k:9s} dim={dims[k]:3d} mean|r|={sens_summary[k][0]:.6f} "
            f"min|r|={sens_summary[k][1]:.6f} (n={len(sens_idx)})")
    log(f"[sens] CLIP-normalized vs uint8 correlation done in {time.time()-t_sens:.1f}s")

    # ---- self-check: 5 random-ish rows ----
    check_idx = [0, 1500, 2999, 3000, 5299]
    log("")
    log("-" * 90)
    log("SELF-CHECK: 5 rows (path + first 5 dims of each family)")
    log("-" * 90)
    for i in check_idx:
        log(f"row {i}: source={source_all[i]} domain={domain_all[i]} split={split_all[i]} "
            f"y={y_all[i]}")
        log(f"    path={paths_all[i]}")
        for k in FEAT_KEYS:
            log(f"    {k:9s}[:5]={np.round(feats[k][i][:5], 5)}")

    # ---- save ----
    def _u(arr, w):
        return np.array([str(x) for x in arr], dtype=f"<U{w}")

    w_path = max(int(max(len(str(x)) for x in paths_all)), 1)
    w_vid = max(int(max(len(str(x)) for x in vids_all)), 1)
    save_kwargs = {k: feats[k] for k in FEAT_KEYS}
    save_kwargs.update(
        y=y_all,
        paths=_u(paths_all, w_path),
        vids=_u(vids_all, w_vid),
        domain=_u(domain_all, 12),
        split=_u(split_all, 12),
        source=_u(source_all, 8),
        feat_keys=_u(FEAT_KEYS, 12),
        feat_dims=_u([f"{k}:{dims[k]}" for k in FEAT_KEYS], 16),
    )
    np.savez(FREQ_NPZ, **save_kwargs)
    wall = time.time() - t0
    log(f"[save] {FREQ_NPZ}")

    # ---- reload verification (artifact vs source npz, element-wise) ----
    chk = np.load(FREQ_NPZ, allow_pickle=True)
    pchk = np.load(PROBE_NPZ, allow_pickle=True)
    mchk = np.load(MULTI_NPZ, allow_pickle=True)
    v_ok = True
    if not (chk["paths"][:n_p] == pchk["paths"]).all():
        v_ok = False; log("[verify] FAIL probe paths")
    if not (chk["paths"][n_p:] == mchk["path"]).all():
        v_ok = False; log("[verify] FAIL multi paths")
    if not (chk["y"] == np.concatenate([pchk["y"], mchk["y"]])).all():
        v_ok = False; log("[verify] FAIL y")
    for k in FEAT_KEYS:
        if chk[k].shape[0] != n_total or not np.isfinite(chk[k]).all():
            v_ok = False; log(f"[verify] FAIL {k} shape/finite")
    if not v_ok:
        log("[verify] FAIL: saved artifact does not match source row order")
        log.close()
        sys.exit(1)
    log("[verify] reloaded freq_feats.npz: paths/y == source npz element-wise, "
        f"all {len(FEAT_KEYS)} feature arrays ({n_total} rows) finite -> PASS")

    log(f"[done] wall={wall:.1f}s  read_ok={n_total-read_fail} read_fail={read_fail}")
    log(f"[done] feature families: " + ", ".join(f"{k}={dims[k]}d" for k in FEAT_KEYS))
    log.close()


if __name__ == "__main__":
    main()
