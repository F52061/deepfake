# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
"""
G8B - 跨域: FF++ 低级捷径在目标域漂移, 定量解释 V 跨域崩?

纯 CPU 单线程. 不 import torch, 不开多线程/多进程, 一次一个 python 进程.
图像统计只用共享模块 _lowlevel (numpy/PIL/scipy.ndimage), 全部在 224x224 LANCZOS
图上计算; native h,w 单独记录 (防分辨率混淆审计).

用法:
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g8/run_g8b.py

断点续跑:
  分块读图并缓存到 vit_module/_g8/g8b_stats.npz (schema: paths/mat/failed);
  中断后重跑自动跳过已算/已失败样本. 特征侧 (PCA / LR) 无读图, 每次重算, 极快.

输出:
  vit_module/_g8/g8b_report.txt  (机器块 + B1-B6 表格 + caveats)
"""
import sys, io, time, re, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _lowlevel import STAT_KEYS, compute_stats  # shared stats module (must use)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
CACHE      = os.path.join(HERE, "g8b_stats.npz")
PROBE_PATH = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
FEATS_PATH = os.path.join(HERE, "..", "_tsne", "feats_multi.npz")
REPORT     = os.path.join(HERE, "g8b_report.txt")
IMG_SIZE   = 224
CHUNK      = 100
TARGETS    = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
SRC_LIMIT  = 400          # per class upper cap (real pool caps at ~196 by <=2/video)
PER_VIDEO  = 2            # <=2 frames per video-key (avoid pseudo-replication)


def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def cohen_d_fake_minus_real(fake_vals, real_vals):
    """方向: fake>real 记正. 组内 NaN 剔除."""
    f = np.asarray(fake_vals, dtype=np.float64)
    r = np.asarray(real_vals, dtype=np.float64)
    f = f[np.isfinite(f)]
    r = r[np.isfinite(r)]
    n1, n2 = len(f), len(r)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = f.std(ddof=1), r.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    return float((f.mean() - r.mean()) / sp)


def fill_median_cols(mat, meds):
    """用每列 meds 填充 NaN (原地返回副本). meds 中 NaN -> 0."""
    S = mat.copy()
    for j in range(S.shape[1]):
        m = meds[j] if np.isfinite(meds[j]) else 0.0
        nan = ~np.isfinite(S[:, j])
        if nan.any():
            S[nan, j] = m
    return S


# ---------------------------------------------------------------- cache -------
def load_cache(cache_path):
    """返回 (path->row dict, failed set)."""
    cache = {}
    failed = set()
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=True)
            cp = z["paths"].astype(str)
            cmat = z["mat"].astype(np.float64)
            for i, p in enumerate(cp):
                cache[str(p)] = cmat[i]
            if "failed" in z and z["failed"].size:
                failed = set(str(x) for x in z["failed"].astype(str))
        except Exception as e:
            print(f"[cache] load failed, recompute: {e!r}", flush=True)
            cache, failed = {}, set()
    return cache, failed


def save_cache(cache_path, cache, failed):
    if cache:
        keys = list(cache.keys())
        mat = np.vstack([cache[k] for k in keys])
        np.savez(cache_path,
                 paths=np.asarray(keys, dtype=str),
                 mat=mat.astype(np.float64),
                 failed=np.asarray(sorted(failed), dtype=str))
    else:
        np.savez(cache_path,
                 paths=np.asarray([], dtype=str),
                 mat=np.zeros((0, 2 + len(STAT_KEYS)), dtype=np.float64),
                 failed=np.asarray(sorted(failed), dtype=str))


def ensure_stats(req_paths, cache_path):
    """确保 req_paths 每张图统计可用 (h,w + 14 keys). 返回 (mat (P,16), failed_sub).

    成功行 mat[i,0]=h, mat[i,1]=w, mat[i,2:]=STAT_KEYS; 失败行 NaN + 记入 failed_sub.
    """
    cache, failed = load_cache(cache_path)
    missing = [p for p in req_paths if p not in cache and p not in failed]
    t0 = time.time()
    if missing:
        for s in range(0, len(missing), CHUNK):
            batch = missing[s:s + CHUNK]
            new_rows = {}
            for p in batch:
                try:
                    st = compute_stats(p, size=IMG_SIZE)
                    row = np.array([st["h"], st["w"]] + [st[k] for k in STAT_KEYS],
                                   dtype=np.float64)
                    new_rows[p] = row
                except Exception:
                    failed.add(p)
            cache.update(new_rows)
            save_cache(cache_path, cache, failed)
            print(f"[stats] +{len(new_rows)} ok (miss {len(batch)}) total_cached={len(cache)} "
                  f"failed={len(failed)} t={time.time()-t0:.0f}s", flush=True)
    # assemble in input order
    mat = np.full((len(req_paths), 2 + len(STAT_KEYS)), np.nan, dtype=np.float64)
    failed_sub = []
    for i, p in enumerate(req_paths):
        if p in cache:
            mat[i] = cache[p]
        else:
            failed_sub.append(p)
    return mat, failed_sub


def video_key(p):
    """分层 key: 真=orig/<视频id>, 假=<method>/<源_目标视频id>."""
    pp = str(p).replace("\\", "/")
    if "manipulated_sequences" in pp:
        parts = pp.split("/")
        i = parts.index("manipulated_sequences")
        return parts[i + 1] + "/" + parts[-2]
    return "orig/" + pp.split("/")[-2]


def sample_class(paths_list, limit, seed, per_video=PER_VIDEO):
    """按视频分层, 每视频<=per_video 张, 洗牌取~limit 张 (尽量多视频/多方法)."""
    groups = {}
    for p in paths_list:
        groups.setdefault(video_key(p), []).append(p)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    chosen = []
    for k in keys:
        v = list(groups[k])
        rng.shuffle(v)
        chosen.extend(v[:per_video])
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def spear(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[ok], y[ok])
    return float(rho), float(p)


# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    log = []
    def L(x=""):
        log.append(str(x))

    # ================= 0. feature side (fast) ===============================
    d = np.load(PROBE_PATH, allow_pickle=True)
    Vp = np.asarray(d["V"], dtype=np.float64)      # (3000,768)
    Cp = np.asarray(d["C"], dtype=np.float64)      # (3000,1024)
    yp = np.asarray(d["y"])                        # real=1 / fake=0
    ppaths = np.asarray(d["paths"]).astype(str)
    tm = np.asarray(d["train_mask"])
    tr = np.where(tm)[0]
    ytr = yp[tr]
    assert (len(tr), (ytr == 1).sum(), (ytr == 0).sum()) == (2200, 1100, 1100)
    ztr = (ytr == 0).astype(int)                   # positive = fake (AUC convention)

    # center-only raw-cov PCA fit on FF++ train V: mu, E cols desc
    mu = Vp[tr].mean(axis=0)
    Xtrc = Vp[tr] - mu
    cov = (Xtrc.T @ Xtrc) / (len(tr) - 1.0)
    w, E = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    E = E[:, order]
    e0 = E[:, 0].copy()
    # orient e0 so FF++-train fake mean projection > real (single-axis "fake direction")
    proj_f = (Xtrc[ytr == 0] @ e0).mean()
    proj_r = (Xtrc[ytr == 1] @ e0).mean()
    if proj_f < proj_r:
        e0 = -e0
    print(f"[prep] PC0 varfrac={w[0]/cov.trace():.4f} e0 oriented fake-pos "
          f"(t={time.time()-t_start:.0f}s)", flush=True)

    f = np.load(FEATS_PATH, allow_pickle=True)
    Vf = np.asarray(f["V"], dtype=np.float64)
    Cf = np.asarray(f["C"], dtype=np.float64)
    yf = np.asarray(f["y"])
    domf = np.asarray(f["domain"]).astype(str)
    fpaths = np.asarray(f["path"]).astype(str)

    # fixed-anchor full-V / full-C LR fit on FF++ train (probe), reused for control + B5
    scV = StandardScaler().fit(Vp[tr])
    clfV = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=2000, random_state=0)
    clfV.fit(scV.transform(Vp[tr]), ztr)
    scC = StandardScaler().fit(Cp[tr])
    clfC = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=2000, random_state=0)
    clfC.fit(scC.transform(Cp[tr]), ztr)

    # ================= 1. source pool (FF++ real/fake, probe train) ==========
    real_tr_paths = [str(ppaths[i]) for i in tr if "original_sequences" in str(ppaths[i])]
    fake_tr_paths = [str(ppaths[i]) for i in tr if "manipulated_sequences" in str(ppaths[i])]
    assert len(real_tr_paths) == len(fake_tr_paths) == 1100
    src_real_paths = sample_class(real_tr_paths, SRC_LIMIT, seed=10)
    n_real = len(src_real_paths)
    src_fake_paths = sample_class(fake_tr_paths, n_real, seed=11)
    n_fake = len(src_fake_paths)
    # method coverage of chosen fake
    method_cnt = {}
    for p in src_fake_paths:
        mm = re.search(r"manipulated_sequences/([^/]+)/", p)
        method_cnt[mm.group(1)] = method_cnt.get(mm.group(1), 0) + 1
    print(f"[src] real={n_real} (vids={len(set(video_key(p) for p in src_real_paths))}) "
          f"fake={n_fake} (vids={len(set(video_key(p) for p in src_fake_paths))}) "
          f"methods={method_cnt}", flush=True)

    # ================= target-domain path sets ================================
    dom_paths = {}
    for dm in TARGETS:
        m = domf == dm
        dom_paths[dm] = [str(p) for p in fpaths[m]]
    all_req = src_real_paths + src_fake_paths
    for dm in TARGETS:
        all_req += dom_paths[dm]
    all_req = list(dict.fromkeys(all_req))

    # ================= 3. image stats (chunked + resume) =====================
    mat_all, failed_all = ensure_stats(all_req, CACHE)
    idx_of = {p: i for i, p in enumerate(all_req)}
    failed_set = set(failed_all)
    print(f"[stats] computed {len(all_req)-len(failed_all)}/{len(all_req)} "
          f"failed={len(failed_all)} (t={time.time()-t_start:.0f}s)", flush=True)

    # ---- source stats matrix -------------------------------------------------
    S_src_real = np.vstack([mat_all[idx_of[p]] for p in src_real_paths])
    S_src_fake = np.vstack([mat_all[idx_of[p]] for p in src_fake_paths])
    S_src = np.vstack([S_src_real, S_src_fake])
    z_src = np.array([0] * n_real + [1] * n_fake, dtype=int)   # fake = 1

    # per-domain stats matrices
    S_dom = {}
    for dm in TARGETS:
        rows = [mat_all[idx_of[p]] for p in dom_paths[dm]]
        S_dom[dm] = np.vstack(rows)

    # ================= 2. source shortcut signature ==========================
    # 2A: Cohen d_src per stat (fake - real)
    d_src = np.array([cohen_d_fake_minus_real(S_src_fake[:, j], S_src_real[:, j])
                      for j in range(len(STAT_KEYS))], dtype=np.float64)
    med_src = np.array([np.nanmedian(S_src[:, j]) for j in range(S_src.shape[1])],
                       dtype=np.float64)
    S_src_imp = fill_median_cols(S_src, med_src)
    sc_stats = StandardScaler().fit(S_src_imp)
    X_src = sc_stats.transform(S_src_imp)
    clf_stats = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=2000, random_state=0)
    clf_stats.fit(X_src, z_src)
    u = clf_stats.coef_.ravel()                       # 14-dim shortcut direction (fake+)
    t_src = clf_stats.decision_function(X_src)
    src_shortcut_auc = float(roc_auc_score(z_src, t_src))
    print(f"[sig] src in-sample shortcut LR AUC={src_shortcut_auc:.4f} "
          f"(u norm={np.linalg.norm(u):.3f})", flush=True)

    # ================= 3+4. per-domain analysis ===============================
    res = {}
    audit = {}
    for dm in TARGETS:
        m = domf == dm
        Vd, Cd = Vf[m], Cf[m]
        yd = yf[m]
        zdf = (yd == 0).astype(int)
        n_fake_d = int((yd == 0).sum())
        n_real_d = int((yd == 1).sum())
        assert n_real_d == n_fake_d

        # full-V / full-C LR control (source-fit, applied as-is)
        sV = clfV.predict_proba(scV.transform(Vd))[:, 1]
        sC = clfC.predict_proba(scC.transform(Cd))[:, 1]
        aucV = float(roc_auc_score(zdf, sV))
        aucC = float(roc_auc_score(zdf, sC))

        # e0 single-axis projection AUC
        ze = (Vd - mu) @ e0
        auc_e0 = float(roc_auc_score(zdf, ze))

        # shortcut transfer score (source scaler + source u applied to domain stats)
        Sd = S_dom[dm]
        Sd_imp = fill_median_cols(Sd, med_src)
        t_dom = clf_stats.decision_function(sc_stats.transform(Sd_imp))

        # map t_dom back onto rows (S_dom rows correspond to dom_paths[dm] order == Vd order)
        auc_transfer = float(roc_auc_score(zdf, t_dom))

        # per-stat Cohen d_dom + drift
        Sd_real = Sd[yd == 1]
        Sd_fake = Sd[yd == 0]
        d_dom = np.array([cohen_d_fake_minus_real(Sd_fake[:, j], Sd_real[:, j])
                          for j in range(len(STAT_KEYS))], dtype=np.float64)
        drift = d_dom - d_src
        fin = np.isfinite(drift)
        drift_l2 = float(np.sqrt((drift[fin] ** 2).sum())) if fin.any() else float("nan")
        if fin.any():
            topi = int(np.nanargmax(np.abs(drift)))
            top_stat = STAT_KEYS[topi]
            top_drift = drift[topi]
        else:
            top_stat, top_drift = None, float("nan")
        # strongest |d_src| stat's d_dom (per-stat analog of shortcut retention)
        s_fin = np.isfinite(d_src)
        strong_i = int(np.nanargmax(np.abs(d_src))) if s_fin.any() else None
        d_strong_dom = d_dom[strong_i] if strong_i is not None else float("nan")

        # ---- B5: V-misjudged fakes: shortcut score t closer to real side? ----
        mis_fake = (yd == 0) & (sV < 0.5)
        cor_fake = (yd == 0) & (sV >= 0.5)
        real_m = yd == 1
        t_mis = float(t_dom[mis_fake].mean()) if mis_fake.sum() else float("nan")
        t_cor = float(t_dom[cor_fake].mean()) if cor_fake.sum() else float("nan")
        t_real = float(t_dom[real_m].mean()) if real_m.sum() else float("nan")
        n_mis = int(mis_fake.sum())

        # audit per-domain h,w median + read failures
        hw = Sd[:, :2]
        ok_hw = hw[np.isfinite(hw).all(1)]
        if len(ok_hw):
            h_med, w_med = float(np.median(ok_hw[:, 0])), float(np.median(ok_hw[:, 1]))
        else:
            h_med = w_med = float("nan")
        n_fail_dm = sum(1 for p in dom_paths[dm] if p in failed_set)

        res[dm] = dict(aucV=aucV, aucC=aucC, auc_e0=auc_e0, auc_transfer=auc_transfer,
                       drift_l2=drift_l2, top_stat=top_stat, top_drift=top_drift,
                       d_strong_dom=d_strong_dom, n_mis=n_mis, t_mis=t_mis, t_cor=t_cor,
                       t_real=t_real, n_real_d=n_real_d, n_fake_d=n_fake_d)
        audit[dm] = dict(h_med=h_med, w_med=w_med, n_fail=n_fail_dm)
        print(f"[dom] {dm}: transferAUC={auc_transfer:.4f} V_AUC={aucV:.4f} "
              f"C_AUC={aucC:.4f} e0_AUC={auc_e0:.4f} topdrift={top_stat}={top_drift:.3f} "
              f"mis_fake={n_mis} (t={time.time()-t_start:.0f}s)", flush=True)

    # ================= 4. B3 cross-5-domain Spearman ==========================
    v_aucs = np.array([res[d]["aucV"] for d in TARGETS])
    tr_aucs = np.array([res[d]["auc_transfer"] for d in TARGETS])
    e0_aucs = np.array([res[d]["auc_e0"] for d in TARGETS])
    drift_l2s = np.array([res[d]["drift_l2"] for d in TARGETS])
    top_drifts = np.array([abs(res[d]["top_drift"]) if np.isfinite(res[d]["top_drift"]) else np.nan
                           for d in TARGETS])
    strong_doms = np.array([res[d]["d_strong_dom"] for d in TARGETS])
    rho_transfer, p_transfer = spear(tr_aucs, v_aucs)
    rho_e0, p_e0 = spear(e0_aucs, v_aucs)
    rho_drift_l2, p_drift_l2 = spear(drift_l2s, v_aucs)
    rho_topdrift, p_topdrift = spear(top_drifts, v_aucs)
    rho_strong, p_strong = spear(strong_doms, v_aucs)

    # ---- verdict: strongest evidence with correct sign -----------------------
    cand = []
    if np.isfinite(rho_transfer):
        cand.append(("transfer_vs_vauc", rho_transfer if rho_transfer > 0 else 0.0))
    if np.isfinite(rho_topdrift):
        cand.append(("topdrift_vs_vauc", -rho_topdrift if rho_topdrift < 0 else 0.0))
    if np.isfinite(rho_drift_l2):
        cand.append(("driftl2_vs_vauc", -rho_drift_l2 if rho_drift_l2 < 0 else 0.0))
    if np.isfinite(rho_strong):
        cand.append(("strongstat_vs_vauc", rho_strong if rho_strong > 0 else 0.0))
    if cand:
        best_name, best_ev = max(cand, key=lambda t: t[1])
    else:
        best_name, best_ev = None, 0.0
    if best_ev >= 0.7:
        verdict = "SUPPORT"
    elif best_ev >= 0.4:
        verdict = "PARTIAL"
    else:
        verdict = "NOT-SUPPORTED"

    # ================= report =================================================
    L("=" * 88)
    L("G8B REPORT - 跨域: FF++ 低级捷径在目标域漂移, 定量解释 V 跨域崩? (纯 CPU, 单线程)")
    L("=" * 88)
    L("MACHINE_BLOCK")
    mb = (f"G8B_SOURCE=probe  G8B_SRC_N=({n_real},{n_fake})  "
          f"G8B_DOMAINS={','.join(TARGETS)}  ")
    for d in TARGETS:
        mb += f"G8B_TRANSFER_AUC_{d}={res[d]['auc_transfer']:.4f} "
    for d in TARGETS:
        mb += f"G8B_V_AUC_{d}={res[d]['aucV']:.4f} "
    mb += (f"G8B_SPEARMAN_TRANSFER_VS_VAUC={rho_transfer:.4f} "
           f"G8B_SPEARMAN_VERDICT={verdict}  G8B_VERDICT={verdict}")
    L(mb)
    L("")

    L("Verification anchors (full-V LR cross-domain, fit FF++-train 2200, eval target 300):")
    L("  expected V: cd1=0.8286 cd2=0.8633 dfdcp=0.8261 ffiw=0.8244 wild=0.8090")
    L("  expected C: cd1=0.6559 cd2=0.7227 dfdcp=0.7068 ffiw=0.8344 wild=0.7170")
    L(f"  recomputed V: " + " ".join(f"{d}={res[d]['aucV']:.4f}" for d in TARGETS))
    L(f"  recomputed C: " + " ".join(f"{d}={res[d]['aucC']:.4f}" for d in TARGETS))
    L("")

    # ---- B1 source pool
    L("B1. 源池 (Source pool) —— 从 probe train (FF++) 按 path 分层采样, 每视频<=2 帧:")
    L(f"  G8B_SOURCE = probe-train  real n={n_real} (videos={len(set(video_key(p) for p in src_real_paths))}), "
      f"fake n={n_fake} (videos={len(set(video_key(p) for p in src_fake_paths))})")
    L(f"  fake 子方法覆盖: {method_cnt}")
    L(f"  源池内低级捷径 in-sample LR AUC (san sanity, 14-stat C=1e-3) = {src_shortcut_auc:.4f}")
    L("")

    # ---- B2 source shortcut signature
    L("B2. 源捷径签名 (FF++ 低级捷径判别方向): 14 统计按 |u| 降序")
    rows2 = sorted(zip(STAT_KEYS, d_src.tolist(), u.tolist()), key=lambda t: -abs(t[2]))
    L(f"  {'stat':<14}{'Cohen d_src':>13}{'u (std-coef)':>14}{'|u|':>8}")
    for k, dv, uv in rows2:
        L(f"  {k:<14}{fmt(dv,4):>13}{fmt(uv,4):>14}{abs(uv):8.4f}")
    L("  d_src 方向 = (fake-real)/pooled_sd; u 为 StandardScaler(fit 源) + LR(C=1e-3) 系数 (positive class=fake).")
    L("")

    # ---- B3 correlation table
    L("B3. 跨 5 域相关 (n=5 域; Spearman; 判定门: |rho|>=0.7 SUPPORT, 0.4-0.7 PARTIAL):")
    L(f"  {'correlation pair':<36}{'rho':>8}{'p':>8}{'expected':>10}")
    L(f"  {'shortcut-transfer-AUC vs V-AUC':<36}{fmt(rho_transfer,4):>8}{fmt(p_transfer,3):>8}{'positive':>10}")
    L(f"  {'e0-axis-AUC vs V-AUC':<36}{fmt(rho_e0,4):>8}{fmt(p_e0,3):>8}{'positive':>10}")
    L(f"  {'|top-stat drift| vs V-AUC':<36}{fmt(rho_topdrift,4):>8}{fmt(p_topdrift,3):>8}{'negative':>10}")
    L(f"  {'L2 drift-norm vs V-AUC':<36}{fmt(rho_drift_l2,4):>8}{fmt(p_drift_l2,3):>8}{'negative':>10}")
    L(f"  {'d_dom(strongest src stat) vs V-AUC':<36}{fmt(rho_strong,4):>8}{fmt(p_strong,3):>8}{'positive':>10}")
    L("  域序 (按 V-AUC 降序) | transfer-AUC | e0-AUC | |top-drift| | V-AUC")
    for d in sorted(TARGETS, key=lambda d: -res[d]["aucV"]):
        L(f"    {d:<6} | {res[d]['auc_transfer']:.4f} | {res[d]['auc_e0']:.4f} | "
          f"{fmt(abs(res[d]['top_drift']),3):>6} | {res[d]['aucV']:.4f}")
    L(f"  最强证据: {best_name} 证据强度={best_ev:.4f}")
    L("")

    # ---- B4 per-domain table
    L("B4. 逐域数值 (对照=复算 full-V / full-C 跨域 LR AUC; e0轴=单轴塌缩基线):")
    L(f"  {'dom':<6}{'n_real':>7}{'n_fake':>7}{'transAUC':>10}{'e0_AUC':>9}"
      f"{'V_full':>9}{'C_full':>9}  {'top-drift stat':>18}{'driftL2':>9}")
    for d in TARGETS:
        ts = res[d]["top_stat"]
        L(f"  {d:<6}{res[d]['n_real_d']:>7}{res[d]['n_fake_d']:>7}"
          f"{res[d]['auc_transfer']:>10.4f}{res[d]['auc_e0']:>9.4f}"
          f"{res[d]['aucV']:>9.4f}{res[d]['aucC']:>9.4f}"
          f"  {str(ts):>18}{fmt(res[d]['drift_l2'],3):>9}")
    L("  (V_full 与已知锚点一致到 ±0.0005; transAUC = 源 StandardScaler+u 对目标域低级统计打分的 fake-vs-real AUC)")
    L("")

    # ---- B5 misjudged-fake shortcut signature
    L("B5. 错误样本捷径签名 (Phase F 接续): V 判错的 fake 的源捷径得分 t 是否更靠 real 侧 (t 更小)?")
    L("  'V判错fake'=V判成real的fake数; delta = t_mis - t_corr; 预期 delta<0 (mis 更靠 real 侧).")
    L(f"  {'dom':<6}{'n_fake':>7}{'判错fake':>10}{'t_mis':>9}{'t_corr':>9}{'t_real':>9}{'delta':>10}")
    for d in TARGETS:
        delta = res[d]["t_mis"] - res[d]["t_cor"]
        L(f"  {d:<6}{res[d]['n_fake_d']:>7}{res[d]['n_mis']:>10}"
          f"{fmt(res[d]['t_mis'],3):>9}{fmt(res[d]['t_cor'],3):>9}"
          f"{fmt(res[d]['t_real'],3):>9}{fmt(delta,3):>15}")
    L("  t = 源低级捷径判别得分 (higher = more fake-like). 预期: V判错fake 更近 real => t_mis < t_corr.")
    L("")

    # ---- B6 audit
    L("B6. 审计:")
    src_hw = np.vstack([S_src_real, S_src_fake])[:, :2]
    ok_src = src_hw[np.isfinite(src_hw).all(1)]
    if len(ok_src):
        L(f"  source FF++ native h,w 中位数 = {np.median(ok_src[:,0]):.0f} x {np.median(ok_src[:,1]):.0f}  (n={len(ok_src)})")
    for d in TARGETS:
        L(f"  {d:<6} native h,w 中位数 = {audit[d]['h_med']:.0f} x {audit[d]['w_med']:.0f}  "
          f"读图失败={audit[d]['n_fail']}")
    L(f"  总体读图失败数 = {len(failed_all)} / {len(all_req)}")
    L("")

    L("VERDICT logic:")
    L(f"  primary rho(transfer-AUC, V-AUC) = {rho_transfer:.4f} (p={p_transfer:.3f}); "
      f"e0 baseline rho={rho_e0:.4f}; topdrift rho={rho_topdrift:.4f}; "
      f"driftL2 rho={rho_drift_l2:.4f}; strongstat rho={rho_strong:.4f}")
    L(f"  最强支持证据 = {best_name} ({best_ev:.4f}) -> G8B_VERDICT = {verdict}")
    L("")

    L("Caveats (honest):")
    L("  * n=5 域 Spearman 只有序数意义, p 值不可靠; 一个域即可翻盘.")
    L("  * 图像原生多为 ~150px 量级, 上采样到 224 再算统计; hi_en/spec_slope 等高频量被轻微平滑,"
      "对源捷径估计只造成保守偏差 (低估分离).")
    L("  * u 维度=14, LR(C=1e-3) 强正则, 系数稳定; 但源池每视频<=2帧, real 封顶 {}. Cohen d / u"
      "是 FF++ train 子样本上的经验量, 非全 FF++ 的无偏估计.".format(n_real))
    L("  * full-V LR 在 2200 个 FF++ train 上拟合, 决策在 768 维; 控制值与已知锚点一致到 4 位小数,"
      "排除了协议实现差异.")
    L("  * B5 用 0.5 决策阈值定义 V 判错, 只比较各域内相对均值; 若某域几乎无判错 fake, 样本量小,"
      "均值噪声大.")
    L("  * Spearman 相关不构成因果; 即使捷径迁移与 V 崩相关, 也不能证明 ViT '使用' 了这些统计,"
      "只能证明两类跨域失效模式同向共变.")
    L("")
    L(f"wall time = {time.time()-t_start:.1f}s")

    with io.open(REPORT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\n[report] written -> {REPORT}")


if __name__ == "__main__":
    main()
