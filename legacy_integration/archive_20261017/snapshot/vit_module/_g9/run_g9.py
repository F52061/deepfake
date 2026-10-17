import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
"""
G9 - 低级统计区域化(分块) 能否解释 V 判别? (H4"局部伪影捷径"变体) + PC0 内容定位
纯 CPU 单线程. 禁止 torch / GPU / 多线程 / 多进程. 一次一个 python 进程.

对照锚点 (固定 LR 协议): StandardScaler(fit train) -> LR(C=1e-3,lbfgs,max_iter=2000)
  V_full_AUC=0.9852  C_full_AUC=0.9108  G8A 整图14统计 stat-only test AUC=0.5580
  G8A adjR2(V-logit ~ 14统计, train)=0.0426

两块区域特征 (都在 224x224 LANCZOS 重采样灰度图上):
  Grid A: 4x4=16块, 每块56x56, 5统计 {grad_mean, lapl_var, hi_en(块FFT), edge_density(grad>25), blockiness(块内8px边界)} -> 80维
  Grid B: 8x8=64块, 每块28x28, 4统计 {grad_mean, lapl_var, hi_en(块FFT), blockiness} -> 256维

判定门 (预注册, 机械): Grid A stat-only test AUC vs G8A 全局 0.5580:
  >=0.75 SUPPORT ; 0.65-0.75 PARTIAL ; <0.65 NOT-SUPPORTED

用法:
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g9/run_g9.py [--limit N] [--selftest M]

缓存: vit_module/_g9/g9_stats.npz, 每 ~300 张增量保存; 已算 chunk 重跑跳过.
报告: vit_module/_g9/g9_report.txt (UTF-8)
"""
import sys, io, time, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_g8"))   # for _lowlevel

try:
    import scipy.ndimage as ndi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from PIL import Image

from _lowlevel import _grad  # shared whole-image gradient (scipy sobel)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
NPZ_PATH   = os.path.join(HERE, "g9_stats.npz")
PROBE_PATH = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
REPORT     = os.path.join(HERE, "g9_report.txt")
CAND_TXT   = os.path.join(HERE, "g9_visual_candidates.txt")
CHUNK      = 300
N_IMG      = 3000
IMG_SIZE   = 224

# Grid A: 4x4 blocks of 56; stats order per block
NA = 4
BA = 56
A_STATS = ["grad_mean", "lapl_var", "hi_en", "edge_density", "blockiness"]  # 5
DA = NA * NA * len(A_STATS)  # 80

# Grid B: 8x8 blocks of 28; stats order per block
NB = 8
BB = 28
B_STATS = ["grad_mean", "lapl_var", "hi_en", "blockiness"]  # 4
DB = NB * NB * len(B_STATS)  # 256

AUC_LOW = 0.75
AUC_PARTIAL = 0.65


# ----------------------------------------------------------------------------
def pearson(x, y):
    """pairwise-complete Pearson r; NaN safe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xv, yv = x[m], y[m]
    if xv.std() == 0 or yv.std() == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def roc_real_pos(feat, yisreal):
    """single-feature ROC AUC on given samples; positive class = real(y==1).
    NaN rows dropped."""
    x = np.asarray(feat, dtype=np.float64)
    lab = np.asarray(yisreal, dtype=np.int64)
    m = np.isfinite(x)
    if m.sum() < 3 or (lab[m] == 1).sum() < 1 or (lab[m] == 0).sum() < 1:
        return float("nan")
    u = np.unique(x[m])
    if u.size < 2:
        return float("nan")
    return float(roc_auc_score(lab[m], x[m]))


def fill_median_train(S, tr):
    """NaN fill by train-column median."""
    S = S.copy()
    for j in range(S.shape[1]):
        col = S[:, j]
        med = np.nanmedian(col[tr])
        if not np.isfinite(med):
            med = 0.0
        nan = ~np.isfinite(col)
        if nan.any():
            col[nan] = med
    return S


def prep_scaled(S_tr_raw, S_te_raw):
    """median-impute(用train) + StandardScaler(fit train). returns scaled tr/te."""
    full = np.vstack([S_tr_raw, S_te_raw]) if S_te_raw is not None else S_tr_raw
    full = fill_median_train(full, np.arange(S_tr_raw.shape[0]))
    S_tr = full[: S_tr_raw.shape[0]]
    S_te = full[S_tr_raw.shape[0]:] if S_te_raw is not None else None
    sc = StandardScaler().fit(S_tr)
    Xtr = sc.transform(S_tr)
    Xte = sc.transform(S_te) if S_te is not None else None
    return Xtr, Xte


def fit_lr(Xtr, ytr, Xte=None, yte=None):
    lr = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=2000).fit(Xtr, ytr)
    out = {"lr": lr, "s_tr": lr.decision_function(Xtr)}
    if Xte is not None:
        out["s_te"] = lr.decision_function(Xte)
        out["auc_te"] = float(roc_auc_score(yte, out["s_te"]))
    return out


def adj_r2(score, n, p):
    if p >= n - 1:
        return float("nan")
    return float(1.0 - (1.0 - score) * (n - 1.0) / (n - p - 1.0))


def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


# ============================================================================
#  Block (region) low-level features -- implemented here for reproducibility.
# ============================================================================
def _block_hi_en(region):
    """hi_en of a block = radial freq power fraction above 0.25*nyq(=0.125c/p).
    Same band definition as _lowlevel._spectral, but FFT over the block."""
    F = np.fft.fft2(region)
    P = np.abs(F) ** 2
    fy = np.fft.fftfreq(region.shape[0])
    fx = np.fft.fftfreq(region.shape[1])
    r = np.hypot.outer(fy, fx)
    rmax = 0.5
    mask = (r > 0) & (r <= rmax)
    ps = P[mask]
    rr = r[mask]
    tot = ps.sum()
    hi = ps[rr > 0.25 * rmax].sum()
    return float(hi / tot) if tot > 0 else 0.0


def _region_blockiness(region):
    """8px 垂直网格边界不连续超额比, 仅在 region 内. 同 _lowlevel._blockiness 语义,
    但 region 作为独立图 (块内边界). 正 = 网格伪影增强."""
    d = np.abs(np.diff(region, axis=1))          # (H, W-1)
    if d.shape[1] == 0:
        return float("nan")
    bc = np.arange(7, region.shape[1] - 1, 8)    # 边界列对
    inter = np.ones(d.shape[1], dtype=bool)
    if bc.size:
        inter[bc] = False
    mb = float(d[:, bc].mean()) if bc.size else float("nan")
    mi = float(d[:, inter].mean())
    if mi > 1e-9 and np.isfinite(mb):
        return float(mb / mi - 1.0)
    return float("nan")


def _block_means(field, n):
    """(H,H)->(n,n) per-block mean for non-overlapping n x n grid."""
    H = field.shape[0]
    bs = H // n
    return field.reshape(n, bs, n, bs).transpose(0, 2, 1, 3).reshape(n, n, bs * bs).mean(axis=2)


def compute_g9_features(path):
    """单张图 -> (XA len80, XB len256, H native, W native).
    Grid A/B 都从同一 224 灰度图及全场 grad_mag/laplace 图上区域聚合."""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    a = np.asarray(img).astype(np.float64)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    mag, _, _ = _grad(gray)                      # whole-image sobel gradient magnitude
    lap = ndi.laplace(gray, mode="reflect") if _HAVE_SCIPY else np.zeros_like(gray)

    def fill_grid(out, n, bs, stats):
        gm = _block_means(mag, n)
        ed = _block_means((mag > 25.0).astype(np.float64), n)
        lm = _block_means(lap, n)
        lv = _block_means(lap * lap, n) - lm * lm
        for bi in range(n):
            r0 = bi * bs
            r1 = r0 + bs
            for bj in range(n):
                c0 = bj * bs
                c1 = c0 + bs
                idx0 = (bi * n + bj) * len(stats)
                reg = gray[r0:r1, c0:c1]
                for si, s in enumerate(stats):
                    if s == "grad_mean":
                        out[idx0 + si] = gm[bi, bj]
                    elif s == "lapl_var":
                        out[idx0 + si] = lv[bi, bj]
                    elif s == "edge_density":
                        out[idx0 + si] = ed[bi, bj]
                    elif s == "hi_en":
                        out[idx0 + si] = _block_hi_en(reg)
                    elif s == "blockiness":
                        out[idx0 + si] = _region_blockiness(reg)

    XA = np.full(DA, np.nan, dtype=np.float64)
    XB = np.full(DB, np.nan, dtype=np.float64)
    fill_grid(XA, NA, BA, A_STATS)
    fill_grid(XB, NB, BB, B_STATS)
    return XA, XB, float(H), float(W)


# ============================================================================
def main():
    # ---------------- argparse-ish (simple, no stdlib argparse needed) --------
    limit = N_IMG
    selftest = None
    argv = sys.argv[1:]
    for i, a_ in enumerate(argv):
        if a_ == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
        if a_ == "--selftest" and i + 1 < len(argv):
            selftest = int(argv[i + 1])

    t_start = time.time()
    # ================= Phase F: feature side (fast, no image I/O) =============
    d = np.load(PROBE_PATH, allow_pickle=True)
    V = np.asarray(d["V"], dtype=np.float64)
    C = np.asarray(d["C"], dtype=np.float64)
    y = np.asarray(d["y"])
    paths = np.asarray(d["paths"])
    tm = np.asarray(d["train_mask"])
    tr = np.where(tm)[0]
    te = np.where(~tm)[0]
    n_tr, n_te = len(tr), len(te)
    ytr, yte = y[tr], y[te]
    assert (n_tr, n_te) == (2200, 800), (n_tr, n_te)

    # center-only PCA on train V
    mu = V[tr].mean(axis=0)
    Xc = V - mu
    Xtr_c = Xc[tr]
    cov = (Xtr_c.T @ Xtr_c) / (n_tr - 1.0)
    w, E = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    E = E[:, order]
    e0 = E[:, 0].copy()
    lam0 = w[0]
    varfrac0 = float(lam0 / cov.trace())
    z_all = Xc @ e0
    z_train = z_all[tr]

    # fixed anchor LR protocol
    scV = StandardScaler().fit(V[tr])
    Vtr_s = scV.transform(V[tr])
    Vte_s = scV.transform(V[te])
    scC = StandardScaler().fit(C[tr])
    Ctr_s = scC.transform(C[tr])
    Cte_s = scC.transform(C[te])

    lrV = fit_lr(Vtr_s, ytr, Vte_s, yte)
    lrC = fit_lr(Ctr_s, ytr, Cte_s, yte)
    SV_train = lrV["s_tr"]
    SC_train = lrC["s_tr"]
    SV_test = lrV["s_te"]
    V_full_auc = lrV["auc_te"]
    C_full_auc = lrC["auc_te"]
    print(f"[prep] V_full_AUC={V_full_auc:.4f}  C_full_AUC={C_full_auc:.4f}  "
          f"PC0 varfrac={varfrac0:.4f}  (t={time.time()-t_start:.1f}s)", flush=True)

    # ================= Phase S: image region stats (cached incremental) =======
    n_chunks = int(np.ceil(N_IMG / CHUNK))
    XA_all = np.full((N_IMG, DA), np.nan, dtype=np.float64)
    XB_all = np.full((N_IMG, DB), np.nan, dtype=np.float64)
    hw_all = np.full((N_IMG, 2), np.nan, dtype=np.float64)
    done = np.zeros(n_chunks, dtype=bool)
    fail_idx = []

    if os.path.exists(NPZ_PATH):
        try:
            ck = np.load(NPZ_PATH, allow_pickle=True)
            if "XA" in ck and ck["XA"].shape[0] == N_IMG:
                XA_all = np.asarray(ck["XA"]).astype(np.float64).copy()
                XB_all = np.asarray(ck["XB"]).astype(np.float64).copy()
                hw_all = np.asarray(ck["hw"]).astype(np.float64).copy()
                done = np.asarray(ck["done"]).astype(bool).copy()
                if "fail_idx" in ck and ck["fail_idx"].size:
                    fail_idx = list(ck["fail_idx"].astype(int))
                print(f"[stats] partial cache loaded: chunks done {int(done.sum())}/{n_chunks}", flush=True)
        except Exception as e:
            print(f"[stats] cache load failed -> recompute: {e!r}", flush=True)
            XA_all = np.full((N_IMG, DA), np.nan)
            XB_all = np.full((N_IMG, DB), np.nan)
            hw_all = np.full((N_IMG, 2), np.nan)
            done = np.zeros(n_chunks, dtype=bool)
            fail_idx = []

    def save_cache():
        np.savez(NPZ_PATH, XA=XA_all, XB=XB_all, hw=hw_all, done=done,
                 fail_idx=np.asarray(fail_idx, dtype=np.int64),
                 fail_paths=np.asarray([str(paths[i]) for i in fail_idx]),
                 Y_=y)

    if selftest is not None:
        # quick smoke test over first selftest images (ignore cache)
        ok = 0
        for i in range(min(selftest, N_IMG)):
            t0 = time.time()
            try:
                xa, xb, h, w = compute_g9_features(str(paths[i]))
                ok += 1
                if i < 3 or i % 5 == 0:
                    print(f"[selftest] img {i}: XA nan={np.isnan(xa).sum()}/{DA} "
                          f"XB nan={np.isnan(xb).sum()}/{DB} H={h:.0f} W={w:.0f} "
                          f"t={time.time()-t0:.3f}s", flush=True)
            except Exception as e:
                print(f"[selftest] FAIL img {i}: {e!r}", flush=True)
        print(f"[selftest] ok {ok}/{min(selftest, N_IMG)}  "
              f"(t={time.time()-t_start:.1f}s)", flush=True)
        return

    img_limit = min(limit, N_IMG)
    for c in range(n_chunks):
        if done[c]:
            continue
        i0 = c * CHUNK
        i1 = min(i0 + CHUNK, img_limit)
        if i0 >= img_limit:
            break
        for i in range(i0, i1):
            try:
                xa, xb, h, w = compute_g9_features(str(paths[i]))
                XA_all[i] = xa
                XB_all[i] = xb
                hw_all[i, 0] = h
                hw_all[i, 1] = w
            except Exception as e:
                fail_idx.append(i)
                if len(fail_idx) <= 10:
                    print(f"[stats] FAIL img {i}: {e!r}", flush=True)
        done[c] = True
        save_cache()
        print(f"[stats] chunk {c+1}/{n_chunks} done (imgs {i0}-{i1})  "
              f"failed_total={len(fail_idx)}  (t={time.time()-t_start:.1f}s)", flush=True)

    if img_limit < N_IMG:
        print(f"[stats] WARNING: limited run, only {img_limit} images processed. Exiting.", flush=True)
        return

    # final save with split keys required by task
    XA_train = XA_all[tr]
    XA_test = XA_all[te]
    XB_train = XB_all[tr]
    XB_test = XB_all[te]
    hw_train = hw_all[tr]
    hw_test = hw_all[te]
    np.savez(NPZ_PATH, XA=XA_all, XB=XB_all, hw=hw_all, done=done,
             fail_idx=np.asarray(fail_idx, dtype=np.int64),
             fail_paths=np.asarray([str(paths[i]) for i in fail_idx]),
             Y_=y,
             Xg4_train=XA_train, Xg4_test=XA_test,
             Xg8_train=XB_train, Xg8_test=XB_test,
             hw_train=hw_train, hw_test=hw_test,
             Y_train=ytr, Y_test=yte,
             Z_train=z_train, SV_train=SV_train, SC_train=SC_train,
             SV_test=SV_test, mu=mu, e0=e0, lam0=lam0, varfrac0=varfrac0)
    print(f"[stats] final cache saved. failed={len(fail_idx)}  (t={time.time()-t_start:.1f}s)", flush=True)

    # ================= Analysis ================================================
    log = []
    def L(x=""):
        log.append(str(x))

    # --- nan audit
    n_nanA = int(np.isnan(XA_all).sum())
    n_nanB = int(np.isnan(XB_all).sum())

    # --- 1. Grid A stat-only LR
    XtrA_s, XteA_s = prep_scaled(XA_train, XA_test)
    lrA = fit_lr(XtrA_s, ytr, XteA_s, yte)
    gridA_auc = lrA["auc_te"]

    # --- 2. Grid B stat-only LR
    XtrB_s, XteB_s = prep_scaled(XB_train, XB_test)
    lrB = fit_lr(XtrB_s, ytr, XteB_s, yte)
    gridB_auc = lrB["auc_te"]

    # --- 3. adjR2 s_V ~ GridA/GridB (train)
    A_tr_imp = fill_median_train(XA_train, np.arange(n_tr))
    B_tr_imp = fill_median_train(XB_train, np.arange(n_tr))
    olsA = LinearRegression().fit(A_tr_imp, SV_train)
    r2A = olsA.score(A_tr_imp, SV_train)
    adjA = adj_r2(r2A, n_tr, DA)
    olsB = LinearRegression().fit(B_tr_imp, SV_train)
    r2B = olsB.score(B_tr_imp, SV_train)
    adjB = adj_r2(r2B, n_tr, DB)

    # --- grid block coordinate helpers
    def block_of_A(fcol):
        # fcol = feature column index in Grid A (0..79)
        b = fcol // len(A_STATS)
        si = fcol % len(A_STATS)
        return b // NA, b % NA, A_STATS[si], b

    def block_of_B(fcol):
        b = fcol // len(B_STATS)
        si = fcol % len(B_STATS)
        return b // NB, b % NB, B_STATS[si], b

    # --- 4. per-block single-feature AUC on train (real=positive), Grid A
    aA_auc = np.full((DA,), np.nan)
    aB_auc = np.full((DB,), np.nan)
    for j in range(DA):
        aA_auc[j] = roc_real_pos(XA_train[:, j], ytr)
    for j in range(DB):
        aB_auc[j] = roc_real_pos(XB_train[:, j], ytr)

    # --- 5. |Pearson| each Grid A feature vs z_train / s_V
    rA_z = np.full(DA, np.nan)
    rA_v = np.full(DA, np.nan)
    for j in range(DA):
        rA_z[j] = pearson(XA_train[:, j], z_train)
        rA_v[j] = pearson(XA_train[:, j], SV_train)
    # argmax locations
    i_z = int(np.nanargmax(np.abs(rA_z)))
    i_v = int(np.nanargmax(np.abs(rA_v)))
    z_loc = block_of_A(i_z)
    v_loc = block_of_A(i_v)

    # --- signal metric for ranking (max over auc-separation, |r_sV|, |r_z|)
    # auc-separation normalized to [0,1] as 2*|auc-0.5|
    sep = np.abs(aA_auc - 0.5) * 2.0
    sig = np.where(np.isnan(sep), 0.0, sep)
    sig = np.maximum(sig, np.where(np.isnan(rA_v), 0.0, np.abs(rA_v)))
    sig = np.maximum(sig, np.where(np.isnan(rA_z), 0.0, np.abs(rA_z)))
    top3_idx = np.argsort(sig)[::-1][:3].tolist()
    top3 = [(block_of_A(j), aA_auc[j], rA_z[j], rA_v[j], sig[j]) for j in top3_idx]

    # top overall block (for machine block): max |auc-0.5| among Grid A
    tb = int(np.nanargmax(np.abs(aA_auc - 0.5)))
    tbb = block_of_A(tb)
    topblock_auc = aA_auc[tb]

    # --- 6. native-size / scale confound audit
    h_all = hw_all[:, 0]
    w_all = hw_all[:, 1]
    hw = h_all * w_all  # face-crop area proxy (face scale proxy)
    # only train subset for confound audit (consistent with feature correlations)
    h_tr = hw_all[tr, 0]
    w_tr = hw_all[tr, 1]
    hw_tr = h_tr * w_tr

    # stats for the native-distribution line
    hmin, hmed, hmax = np.nanmin(h_all), np.nanmedian(h_all), np.nanmax(h_all)
    wmin, wmed, wmax = np.nanmin(w_all), np.nanmedian(w_all), np.nanmax(w_all)

    conf_rows = []  # (desc, bi, bj, stat, r_h, r_w, r_hw, flag)
    for j, (blk, aucj, rzj, rvj, sg) in zip(top3_idx, top3):
        bi, bj, stat, b = blk
        feat = XA_train[:, j]
        rh = pearson(feat, h_tr)
        rw = pearson(feat, w_tr)
        rhw = pearson(feat, hw_tr)
        flagged = any(abs(r) > 0.5 for r in (rh, rw, rhw) if np.isfinite(r))
        conf_rows.append((bi, bj, stat, rh, rw, rhw, flagged))

    # --- 7. V wrong fakes per-block diff (optional)
    # LR on full V, decision_function on all fakes (train+test); wrong = logit>0
    fake_mask = y == 0
    sV_all = np.empty(N_IMG, dtype=np.float64)
    Vfull_s_all = scV.transform(V)   # StandardScaler fit on train applied to all
    sV_all[:] = lrV["lr"].decision_function(Vfull_s_all)
    fake_wrong = fake_mask & (sV_all > 0)
    fake_corr = fake_mask & ~(sV_all > 0)
    n_wrong = int(fake_wrong.sum())
    n_corr = int(fake_corr.sum())
    wrong_diff_rows = []
    if n_wrong >= 2 and n_corr >= 2:
        # per Grid A feature: standardized diff wrong-vs-correct fakes
        wmean = np.full(DA, np.nan)
        cmean = np.full(DA, np.nan)
        wstd = np.full(DA, np.nan)
        for j in range(DA):
            fw = XA_all[fake_wrong, j]
            fc = XA_all[fake_corr, j]
            fw = fw[np.isfinite(fw)]
            fc = fc[np.isfinite(fc)]
            if len(fw) >= 2 and len(fc) >= 2:
                wmean[j] = fw.mean()
                cmean[j] = fc.mean()
                sp = np.sqrt(((len(fw) - 1) * fw.std(ddof=1) ** 2 +
                              (len(fc) - 1) * fc.std(ddof=1) ** 2) / (len(fw) + len(fc) - 2))
                wstd[j] = sp
        d = (wmean - cmean) / wstd
        o = np.argsort(-np.abs(d))[:10]
        # also real mean for interpretation
        real_idx_all = np.where(y == 1)[0]
        for j in o:
            bi, bj, stat, b = block_of_A(int(j))
            rmean = np.nanmean(XA_all[real_idx_all, int(j)])
            wrong_diff_rows.append((bi, bj, stat, cmean[int(j)], wmean[int(j)], rmean, d[int(j)]))

    # --------------------------------------------------------------------------
    # VERDICT (pre-registered, mechanical)
    if gridA_auc >= AUC_LOW:
        verdict = "SUPPORT"
    elif gridA_auc >= AUC_PARTIAL:
        verdict = "PARTIAL"
    else:
        verdict = "NOT-SUPPORTED"

    # ---- write visual-candidate file for step 8 (only fakes+reals of interest)
    with io.open(CAND_TXT, "w", encoding="utf-8") as f:
        def p(i):
            return str(paths[i])
        # z extremes among fakes
        zf = z_all.copy()
        zf[~fake_mask] = np.nan
        zf_te = zf.copy()
        zf_te[tr] = np.nan
        zf_tr = zf.copy()
        zf_tr[te] = np.nan
        order_tr = np.argsort(zf_tr)
        order_te = np.argsort(zf_te)
        f.write("# G9 visual inspection candidates (step 8)\n")
        f.write(f"# V wrong fakes n={n_wrong} correct fakes n={n_corr}\n")
        shown = []
        def add(label, idx):
            if idx in shown:
                return
            shown.append(idx)
            f.write(f"{label}\t{idx}\t{p(idx)}\n")
        # z_train max/min fakes (correct finite filtering)
        cand_tr = [int(i) for i in order_tr if np.isfinite(zf_tr[i])]
        if cand_tr:
            add("z_train_min_fake", cand_tr[0])
            add("z_train_max_fake", cand_tr[-1])
        cand_te = [int(i) for i in order_te if np.isfinite(zf_te[i])]
        if cand_te:
            add("z_test_max_fake", cand_te[-1])
            add("z_test_min_fake", cand_te[0])
        # V-wrong fakes: pick by |logit| closest to 0 (most ambiguous) & highest logit
        wrong_idx = np.where(fake_wrong)[0]
        if len(wrong_idx):
            o = wrong_idx[np.argsort(sV_all[wrong_idx])]
            for k in [0, len(o) // 2, len(o) - 1]:
                if k < len(o):
                    add("V_wrong_fake", int(o[k]))
        # correct fakes with most real-like high logit below 0
        corr_idx = np.where(fake_corr)[0]
        if len(corr_idx):
            o = corr_idx[np.argsort(-sV_all[corr_idx])]
            add("V_correct_fake_highlogit", int(o[0]))
        # extremes of sV real side? add min real logit (most fake-like real)
        real_mask = y == 1
        ri = np.where(real_mask)[0]
        o = ri[np.argsort(sV_all[ri])]
        add("real_most_fakelike", int(o[0]))
        add("real_most_reallike", int(o[-1]))
    print(f"[candidates] wrote {CAND_TXT}", flush=True)

    # ================= build report ===========================================
    L("=" * 92)
    L("G9 REPORT - 低级统计区域化能否解释 V 判别 (H4局部伪影变体) + 定位 PC0 内容  (纯 CPU, 单线程)")
    L("=" * 92)

    L("MACHINE_BLOCK")
    L(f"G9_GRID_A_AUC={gridA_auc:.4f} G9_GRID_B_AUC={gridB_auc:.4f} "
      f"G9_ADJR2_V_GRIDA={adjA:.4f} G9_ADJR2_V_GRIDB={adjB:.4f} "
      f"G9_V_FULL_AUC={V_full_auc:.4f} G9_C_FULL_AUC={C_full_auc:.4f} "
      f"G9_TOPBLOCK=({tbb[0]},{tbb[1]},{tbb[2]},{topblock_auc:.4f}) "
      f"G9_VERDICT={verdict}")
    L("")
    L(f"Verification anchors: V_full_AUC = {V_full_auc:.4f} (expect 0.9852), "
      f"C_full_AUC = {C_full_auc:.4f} (expect 0.9108)")

    L("")
    L("G0. 特征侧/区域特征说明:")
    L(f"  PC0 varfrac (train V) = {varfrac0:.4f}; e0=E[:,0]; z_train=(V_train-mu)@e0.")
    L(f"  s_V/s_C = train in-sample decision_function of anchor LR (StandardScaler(fit train) + LR(C=1e-3,lbfgs,2000)).")
    L(f"  Grid A: {NA}x{NA} 每块 {BA}px, 5统计 {A_STATS} -> {DA} 维/图; "
      f"Grid B: {NB}x{NB} 每块 {BB}px, 4统计 {B_STATS} -> {DB} 维/图. 全部在 224x224 LANCZOS 灰度图. "
      f"hi_en = 块FFT径向>0.25*nyq 能量占比; blockiness = 块内 8px 垂直网格超额.")
    L(f"  读图失败数: {len(fail_idx)}  (期望0)   NaN 元素数: A={n_nanA}  B={n_nanB}")
    if len(fail_idx):
        L(f"  失败样本 index/path 前20: {[(i, str(paths[i])) for i in fail_idx[:20]]}")
    L(f"  native h,w 分布(全部{n_te+n_tr}张): h min/med/max = {hmin:.0f}/{hmed:.0f}/{hmax:.0f}, "
      f"w min/med/max = {wmin:.0f}/{wmed:.0f}/{wmax:.0f}")

    # ---- 1 & 2
    L("")
    L(f"G1. Grid A 全 {DA} 维 stat-only test AUC (主门数字) = {gridA_auc:.4f}")
    L(f"    对照: G8A 整图 14 统计 stat-only test AUC = 0.5580 ; V_full = {V_full_auc:.4f}")
    L("")
    L(f"G2. Grid B 全 {DB} 维 stat-only test AUC = {gridB_auc:.4f}  (交叉验证/精定位参考)")

    # ---- 3
    L("")
    L(f"G3. OLS (LinearRegression, intercept) on train: s_V ~ GridA/GridB :")
    L(f"  adjR2(s_V ~ GridA({DA})) = {adjA:.4f}   (R2={r2A:.4f}, n={n_tr}, p={DA})")
    L(f"  adjR2(s_V ~ GridB({DB})) = {adjB:.4f}   (R2={r2B:.4f}, n={n_tr}, p={DB})")
    L(f"  对照: G8A adjR2(s_V ~ 整图14统计) = 0.0426")

    # ---- 4 per-block AUC tables
    L("")
    L("G4. Grid A 逐块判别定位. 每格 = 该块 5 统计中单特征 fake-vs-real AUC (train, real=positive) 的最大者 "
      "及其统计名; 数值 = real-positive AUC. |auc-0.5| 越大越判别.")
    # build per-cell: max |auc-0.5| stat and its auc
    cell_txt = [["" for _ in range(NA)] for _ in range(NA)]
    for bi in range(NA):
        for bj in range(NA):
            best = None
            for si, s in enumerate(A_STATS):
                j = (bi * NA + bj) * len(A_STATS) + si
                a = aA_auc[j]
                if np.isfinite(a):
                    if best is None or abs(a - 0.5) > abs(best[0] - 0.5):
                        best = (a, s)
            if best:
                cell_txt[bi][bj] = f"{best[0]:.3f}({best[1][:7]})"
            else:
                cell_txt[bi][bj] = "nan"
    for bi in range(NA):
        L("  row(top->bottom) " + " | ".join(cell_txt[bi]))

    for stat in ["hi_en", "grad_mean"]:
        L("")
        L(f"  Grid A 逐块 {stat} AUC (train, real=positive):")
        rows = []
        si = A_STATS.index(stat)
        for bi in range(NA):
            rr = []
            for bj in range(NA):
                j = (bi * NA + bj) * len(A_STATS) + si
                rr.append(f"{aA_auc[j]:.3f}")
            rows.append("  " + " | ".join(rr))
        L("\n".join(rows))

    # also overall top block/stat by |auc-0.5|
    L("")
    L(f"  Grid A 最强单块单统计 (|auc-0.5| max, train): 块({tbb[0]},{tbb[1]}) stat={tbb[2]} "
      f"real-positive-AUC={topblock_auc:.4f} (若<0.5 => fake 该统计更大)")

    # ---- 5 pearson argmax
    L("")
    L(f"G5. Grid A 每块特征 vs z_train / s_V 的 |Pearson| 最大者定位 (train):")
    L(f"  vs z_train: argmax 块({z_loc[0]},{z_loc[1]}) stat={z_loc[2]}  r={rA_z[i_z]:.4f}  (feature col {i_z})")
    L(f"  vs s_V:     argmax 块({v_loc[0]},{v_loc[1]}) stat={v_loc[2]}  r={rA_v[i_v]:.4f}  (feature col {i_v})")

    # ---- 6 confound audit
    L("")
    L("G6. 对齐/尺度混淆审计. 信号最强 top-3 判别块特征 (rank by max(2*|auc-0.5|,|r_z|,|r_sV|), train) "
      "与 native h / w / 人脸尺度代理(h*w) 的 Pearson:")
    L("  块(bi,bj) | stat        |  r(h)    r(w)    r(h*w)  | |r|>0.5 混淆?")
    for (bi, bj, stat, rh, rw, rhw, flag) in conf_rows:
        L(f"  ({bi},{bj})     | {stat:<11} | {fmt(rh,3):>8} {fmt(rw,3):>8} {fmt(rhw,3):>8} | {'YES-混淆' if flag else 'no'}")
    L("  判据: 若任一 |r|>0.5 -> '区域信号可能与对齐/尺度伪影混淆' (标注 YES-混淆)")

    # ---- 7 wrong-fake analysis
    L("")
    L(f"G7. (可选) V 判错 fakes 的 Grid A 逐块差. V判错 = 全V LR(logit>0) 把 fake 判成 real.")
    L(f"  train+test 合并 fakes: n_fake={int(fake_mask.sum())}, 判错={n_wrong}, 判对={n_corr}")
    if wrong_diff_rows:
        L("  按 |标准化差| 前 10: (cmean=V判对fake均值, wmean=V判错fake均值, rmean=real均值)")
        L("  块(bi,bj) | stat        | cmean      wmean      rmean      | z-diff(w-c)")
        for (bi, bj, stat, cmean, wmean, rmean, dz) in wrong_diff_rows:
            L(f"  ({bi},{bj})     | {stat:<11} | {fmt(cmean,4):>10} {fmt(wmean,4):>10} "
              f"{fmt(rmean,4):>10} | {fmt(dz,3):>9}")
        top_j = int(np.nanargmax(np.abs(d)))  # from loop var
        t_j = block_of_A(top_j)
        L(f"  判错 fake 与判对 fake 差异最大块: ({t_j[0]},{t_j[1]}) stat={t_j[2]}. "
          f"若 wmean 更靠近 rmean => '判错 fakes 在该块像 real'.")
    else:
        L("  (判错或判对样本不足, 跳过逐块差)")

    # ---- verdict
    L("")
    L("VERDICT logic (预注册, 机械):")
    L(f"  Grid A stat-only test AUC = {gridA_auc:.4f} ; gates: >=0.75 SUPPORT, 0.65-0.75 PARTIAL, <0.65 NOT-SUPPORTED")
    L(f"  对照 G8A 整图全局 stat-only test AUC = 0.5580")
    L(f"  G9_VERDICT = {verdict}")

    L("")
    L("Caveats (honest):")
    L("  * 判定基于 Grid A (80维) stat-only test AUC (video-disjoint out-of-sample). "
      "单块单特征 AUC 与 Pearson 均在 train 上(in-sample 定位诊断), 乐观, 只用于定位信号在哪, 不作泛化声明.")
    L("  * 原生分辨率 ~70-920px 宽 crop 全部重采样到 224; 上/下采样会平滑/改变高频, "
      "分块高/中频统计 (hi_en/grad) 与 native h,w 尺度耦合 (见 G6 混淆审计), 相关 != 因果.")
    L("  * 分块边界与对齐 (face crop 内人脸位置/尺度) 耦合是主要混淆源; 无原始人脸框, 用 crop h*w 作尺度代理.")
    L("  * blockiness 为块内 (region 独立) 8px 垂直网格超额比; 每块相位固定于块原点, "
      "且不含跨块最外边界, 与整图 _blockiness 略有差异; 该统计全局信号本就极弱.")
    L("  * hi_en 逐块 FFT 无窗, 块边界泄漏 + 小尺寸(28px)频率分辨率低; 结果保守解读.")
    L("  * in-sample R^2 / adjR^2 (G3) 与 in-sample Pearson (G5/G6) 乐观, 是上界.")
    L("  * G7 用 0.5(logit>0) 阈值定义 'V判错'; 若判错样本少则均值噪声大.")
    L("  * G8 (可选) 定性目检: Read 工具对 FF++ 标准 RGB PNG 返回 [Unsupported Image] (本会话视觉不可用);")
    L("    转存标准 RGB PNG 后仍不可显示 -> 视觉不可用, 跳过 qualitative. 候选索引见 g9_visual_candidates.txt.")
    L("")
    L(f"wall time = {time.time()-t_start:.1f}s")

    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\n[report] written -> {REPORT}")


if __name__ == "__main__":
    main()
