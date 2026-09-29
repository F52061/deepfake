import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
"""
G8A - FF++ 域内归因: V 判别方向是不是低级统计捷径?

纯 CPU 单线程. 不 import torch, 不开多线程/多进程, 一次一个 python 进程.
图像统计只用共享模块 _lowlevel (numpy/PIL/scipy.ndimage), 全部在 224x224 LANCZOS
图上计算; native h,w 单独记录.

用法:
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g8/run_g8a.py

断点续跑:
  分块(每 300 张)读图并把统计缓存写入 vit_module/_g8/g8a_stats.npz;
  进程中断后重跑会自动跳过已算完的块. 特征侧(prep)无读图, 每次重算, 极快.

输出:
  vit_module/_g8/g8a_report.txt  (机器块 + A1-A6 表格 + caveats)
"""
import sys, io, time, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _lowlevel import STAT_KEYS, compute_stats  # shared stats module (must use)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
NPZ_PATH   = os.path.join(HERE, "g8a_stats.npz")
PROBE_PATH = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
REPORT     = os.path.join(HERE, "g8a_report.txt")
CHUNK      = 300
N_IMG      = 3000
IMG_SIZE   = 224


def pearson(x, y):
    """pairwise-complete Pearson r; NaN 安全."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xv, yv = x[m], y[m]
    if xv.std() == 0 or yv.std() == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def cohen_d_fake_minus_real(stats_fake, stats_real):
    """方向: fake>real 记正. 组内 NaN 剔除."""
    f = stats_fake[np.isfinite(stats_fake)]
    r = stats_real[np.isfinite(stats_real)]
    n1, n2 = len(f), len(r)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = f.std(ddof=1), r.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    return float((f.mean() - r.mean()) / sp)


def fill_median_train(S, tr):
    """用 train 行各列中位数填充 NaN; 返回副本."""
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
    """median-impute(用train) + StandardScaler(fit train). 返回 scaled tr/te."""
    tr_idx = np.arange(S_tr_raw.shape[0])
    # fill using train medians on concatenated
    full = np.vstack([S_tr_raw, S_te_raw]) if S_te_raw is not None else S_tr_raw
    full = fill_median_train(full, np.arange(len(S_tr_raw)))
    S_tr = full[: len(S_tr_raw)]
    S_te = full[len(S_tr_raw):] if S_te_raw is not None else None
    sc = StandardScaler().fit(S_tr)
    Xtr = sc.transform(S_tr)
    Xte = sc.transform(S_te) if S_te is not None else None
    return Xtr, Xte


def fit_lr(Xtr, ytr, Xte, yte=None):
    lr = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=2000).fit(Xtr, ytr)
    out = {"lr": lr}
    out["s_tr"] = lr.decision_function(Xtr)
    if Xte is not None:
        out["s_te"] = lr.decision_function(Xte)
        out["auc_te"] = float(roc_auc_score(yte, out["s_te"]))
    return out


def adj_r2(score, n, p):
    """score = sklearn R^2 (in-sample, intercept). adj = 1-(1-R2)*(n-1)/(n-p-1)."""
    return float(1.0 - (1.0 - score) * (n - 1.0) / (n - p - 1.0))


def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    # ================= Phase F: feature-side prep (fast, no image I/O) =======
    d = np.load(PROBE_PATH, allow_pickle=True)
    V  = np.asarray(d["V"], dtype=np.float64)    # (3000,768)
    C  = np.asarray(d["C"], dtype=np.float64)    # (3000,1024)
    y  = np.asarray(d["y"])                      # real=1 / fake=0
    paths = np.asarray(d["paths"])
    tm = np.asarray(d["train_mask"])             # True = train
    tr = np.where(tm)[0]
    te = np.where(~tm)[0]
    n_tr, n_te = len(tr), len(te)
    ytr, yte = y[tr], y[te]
    assert (n_tr, n_te) == (2200, 800), (n_tr, n_te)

    # --- center-only PCA fit on train V:  mu, E (columns=右奇异向量, 特征值降序) ---
    mu = V[tr].mean(axis=0)
    Xc = V - mu                       # (3000,768) centered
    Xtr_c = Xc[tr]                    # train centered
    cov = (Xtr_c.T @ Xtr_c) / (n_tr - 1.0)
    w, E = np.linalg.eigh(cov)        # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    E = E[:, order]                   # (768,768), cols desc
    e0 = E[:, 0].copy()
    lam0 = w[0]
    z_all = Xc @ e0                   # (3000,) PC0 projection
    z_train = z_all[tr]

    # rank-1 reconstruction residual norm per sample (RES definition, unused in gates)
    recon1 = mu + np.outer(z_all, e0)
    res_all = np.linalg.norm(V - recon1, axis=1)
    RES_train = res_all[tr]
    RES_test = res_all[te]

    # --- fixed anchor probe protocol: StandardScaler(fit train) + LR(C=1e-3,lbfgs)
    scV = StandardScaler().fit(V[tr]); Vtr_s = scV.transform(V[tr]); Vte_s = scV.transform(V[te])
    scC = StandardScaler().fit(C[tr]); Ctr_s = scC.transform(C[tr]); Cte_s = scC.transform(C[te])

    lrV = fit_lr(Vtr_s, ytr, Vte_s, yte)
    lrC = fit_lr(Ctr_s, ytr, Cte_s, yte)
    SV_train = lrV["s_tr"]; SC_train = lrC["s_tr"]
    V_full_auc = lrV["auc_te"]; C_full_auc = lrC["auc_te"]
    print(f"[prep] V_full_AUC={V_full_auc:.4f}  C_full_AUC={C_full_auc:.4f}  "
          f"PC0 varfrac={lam0/cov.trace():.4f}  (t={time.time()-t_start:.1f}s)", flush=True)

    # ================= Phase S: image stats (slow, chunked + resume) ===========
    K = len(STAT_KEYS)
    n_chunks = int(np.ceil(N_IMG / CHUNK))
    S_all = np.full((N_IMG, K), np.nan, dtype=np.float64)
    hw_all = np.full((N_IMG, 2), np.nan, dtype=np.float64)
    done = np.zeros(n_chunks, dtype=bool)
    fail_idx = []

    if os.path.exists(NPZ_PATH):
        try:
            ck = np.load(NPZ_PATH, allow_pickle=True)
            if "S_all" in ck:
                S_all = ck["S_all"].astype(np.float64).copy()
                hw_all = ck["hw_all"].astype(np.float64).copy()
                done = ck["done"].astype(bool).copy()
                if "fail_idx" in ck and ck["fail_idx"].size:
                    fail_idx = list(ck["fail_idx"].astype(int))
                print(f"[stats] loaded partial cache: chunks done {int(done.sum())}/{n_chunks}", flush=True)
        except Exception as e:
            print(f"[stats] cache load failed, recompute from scratch: {e!r}", flush=True)
            S_all = np.full((N_IMG, K), np.nan)
            hw_all = np.full((N_IMG, 2), np.nan)
            done = np.zeros(n_chunks, dtype=bool)
            fail_idx = []

    def save_cache():
        np.savez(NPZ_PATH, S_all=S_all, hw_all=hw_all, done=done,
                 fail_idx=np.asarray(fail_idx, dtype=np.int64),
                 # feature-side (for reproducibility / downstream readback)
                 Y_train=ytr, Y_test=yte, Z_train=z_train,
                 SV_train=SV_train, SC_train=SC_train,
                 RES_train=RES_train, RES_test=RES_test)

    for c in range(n_chunks):
        if done[c]:
            continue
        i0 = c * CHUNK
        i1 = min(i0 + CHUNK, N_IMG)
        for i in range(i0, i1):
            p = str(paths[i])
            try:
                st = compute_stats(p, size=IMG_SIZE)
                for j, k in enumerate(STAT_KEYS):
                    S_all[i, j] = st[k]
                hw_all[i, 0] = st["h"]
                hw_all[i, 1] = st["w"]
            except Exception as e:
                fail_idx.append(i)
        done[c] = True
        save_cache()
        print(f"[stats] chunk {c+1}/{n_chunks} done (imgs {i0}-{i1})  "
              f"failed_total={len(fail_idx)}  (t={time.time()-t_start:.1f}s)", flush=True)

    if fail_idx:
        print("[stats] FAILED:", len(fail_idx))
        for fi in fail_idx[:10]:
            print("   ", fi, paths[fi])
    else:
        print("[stats] all images OK, 0 failed.", flush=True)

    # ---- split into train/test views ------------------------------------------
    S_train = S_all[tr]
    S_test = S_all[te]
    hw_train = hw_all[tr]
    hw_test = hw_all[te]
    Y_train = ytr
    Y_test = yte
    # final cache with all schema keys
    np.savez(NPZ_PATH, S_all=S_all, hw_all=hw_all, done=done,
             fail_idx=np.asarray(fail_idx, dtype=np.int64),
             S_train=S_train, S_test=S_test, Y_train=Y_train, Y_test=Y_test,
             Z_train=z_train, SV_train=SV_train, SC_train=SC_train,
             RES_train=RES_train, RES_test=RES_test,
             hw_train=hw_train, hw_test=hw_test)
    print(f"[stats] final cache saved. (t={time.time()-t_start:.1f}s)", flush=True)

    # ================= Analysis =================================================
    log = []
    def L(x=""):
        log.append(str(x))

    machine = {}

    # ---------- A1 : real/fake 逐统计分离 (train), Cohen d ----------------------
    fake_tr = ytr == 0
    real_tr = ytr == 1
    a1 = []
    for j, k in enumerate(STAT_KEYS):
        d_val = cohen_d_fake_minus_real(S_train[fake_tr, j], S_train[real_tr, j])
        mf = np.nanmean(S_train[fake_tr, j])
        mr = np.nanmean(S_train[real_tr, j])
        a1.append((k, d_val, mf, mr))
    a1.sort(key=lambda t: -abs(t[1]))
    top_d_stats = [t[0] for t in a1][:5]

    # ---------- A2 : 每统计 vs z_train(PC0) / s_V(in-sample logit), train -------
    a2 = []
    for j, k in enumerate(STAT_KEYS):
        r_z = pearson(S_train[:, j], z_train)
        r_v = pearson(S_train[:, j], SV_train)
        a2.append((k, r_z, r_v))
    a2_by_sv = sorted(a2, key=lambda t: -abs(t[2]))
    top3_sv = [t[0] for t in a2_by_sv[:3]]
    a2_by_z = sorted(a2, key=lambda t: -abs(t[1]))

    # ---------- A3 : 仅低级统计量的判别力 ---------------------------------------
    Xtr_s, Xte_s = prep_scaled(S_train, S_test)
    stat_lr = fit_lr(Xtr_s, ytr, Xte_s, yte)
    statonly_auc = stat_lr["auc_te"]

    # top-3 (by |r with s_V|) LR
    jj = [STAT_KEYS.index(k) for k in top3_sv]
    Xt3_s, Xe3_s = prep_scaled(S_train[:, jj], S_test[:, jj])
    stat_top3 = fit_lr(Xt3_s, ytr, Xe3_s, yte)
    statonly_top3_auc = stat_top3["auc_te"]

    # 14 个单统计各自 AUC
    single = []
    for j, k in enumerate(STAT_KEYS):
        X1_s, X1e_s = prep_scaled(S_train[:, j:j + 1], S_test[:, j:j + 1])
        r = fit_lr(X1_s, ytr, X1e_s, yte)
        single.append((k, r["auc_te"]))
    single.sort(key=lambda t: -t[1])

    # ---------- A4 : OLS s_V/s_C ~ stats (train), adjusted R2 -------------------
    S_tr_imp = fill_median_train(S_train, np.arange(n_tr))
    ols_v = LinearRegression().fit(S_tr_imp, SV_train)
    r2_v = ols_v.score(S_tr_imp, SV_train)
    adj_v = adj_r2(r2_v, n_tr, K)
    ols_c = LinearRegression().fit(S_tr_imp, SC_train)
    r2_c = ols_c.score(S_tr_imp, SC_train)
    adj_c = adj_r2(r2_c, n_tr, K)

    # ---------- A5 : fake 子方法拆解 (train) -------------------------------------
    method_of = {}
    for i in range(len(paths)):
        m = re.search(r"manipulated_sequences/([^/]+)/", str(paths[i]))
        method_of[i] = m.group(1) if m else None
    real_means = {k: np.nanmean(S_train[real_tr, j]) for j, k in enumerate(STAT_KEYS)}
    groups = {}
    for i in tr:
        if y[i] == 0:
            mm = method_of[int(i)]
            groups.setdefault(mm, []).append(int(i))
    a5 = []
    for mm in sorted(groups):
        idx = np.asarray(groups[mm])
        cnt = len(idx)
        gmeans = {k: np.nanmean(S_all[idx, j]) for j, k in enumerate(STAT_KEYS)}
        a5.append((mm, cnt, gmeans))

    # ---------- A6 : 审计 ---------------------------------------------------------
    corr14 = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            corr14[i, j] = pearson(S_train[:, i], S_train[:, j])
    hw_all_flat_ok = hw_all[np.isfinite(hw_all).all(1)]
    if len(hw_all_flat_ok):
        h_med, w_med = np.median(hw_all_flat_ok, axis=0)
        h_min, w_min = hw_all_flat_ok.min(axis=0)
        h_max, w_max = hw_all_flat_ok.max(axis=0)
    else:
        h_med = w_med = h_min = w_min = h_max = w_max = float("nan")

    n_single_ge80 = sum(1 for _, a in single if a >= 0.80)
    top3_suffices = (statonly_auc - statonly_top3_auc) <= 0.01

    # ---------- 判定 --------------------------------------------------------------
    support = (statonly_auc >= 0.93) and (adj_v >= 0.5) and top3_suffices
    notsup = (statonly_auc < 0.80) and (adj_v < 0.2)
    if support:
        verdict = "SUPPORT"
    elif notsup:
        verdict = "NOT-SUPPORTED"
    else:
        verdict = "PARTIAL"

    machine.update(
        V_full_auc=V_full_auc, C_full_auc=C_full_auc, statonly_auc=statonly_auc,
        statonly_top3_auc=statonly_top3_auc, adj_v=adj_v, adj_c=adj_c, verdict=verdict)

    # ================= Report ====================================================
    L("=" * 86)
    L("G8A REPORT - FF++ 域内归因: V 判别方向是不是低级统计捷径?  (纯 CPU, 单线程)")
    L("=" * 86)
    L("MACHINE_BLOCK")
    L(f"G8A_V_full_AUC={V_full_auc:.4f} G8A_C_full_AUC={C_full_auc:.4f} "
      f"G8A_STATONLY_AUC={statonly_auc:.4f} G8A_STATONLY_TOP3_AUC={statonly_top3_auc:.4f} "
      f"G8A_ADJR2_V={adj_v:.4f} G8A_ADJR2_C={adj_c:.4f} G8A_VERDICT={verdict}")
    L("")
    L(f"Verification anchors: V_full_AUC = {V_full_auc:.4f} (expect 0.9852), "
      f"C_full_AUC = {C_full_auc:.4f} (expect 0.9108)")

    # ---- A0/A1
    L("")
    L(f"A0. 特征侧: center-only PCA on train V. mu (768d), E cols desc. "
      f"PC0 explains varfrac={lam0/cov.trace():.4f}. z_train=(V_train-mu)@e0.")
    L("    s_V / s_C = train in-sample decision_function of anchor LR; "
      "RES_train/test = per-sample rank-1(PC0) reconstruction residual norm (|V-(mu+z e0)|), unused by gates.")
    L("")
    L("A1. real/fake 分离 (train; n_real=1100, n_fake=1100). Cohen d 方向 = (fake-real)/pooled_sd, 正=fake更大:")
    L(f"  {'stat':<14}{'d(fake-real)':>13}{'mean_fake':>12}{'mean_real':>12}")
    for k, dv, mf, mr in a1:
        L(f"  {k:<14}{fmt(dv,4):>13}{fmt(mf,4):>12}{fmt(mr,4):>12}")
    L(f"  top |d| stats: {[t[0] for t in a1[:5]]}")

    # ---- A2
    L("")
    L("A2. 每统计 与 z_train(PC0投影) 及 s_V(in-sample logit) 的 Pearson r (train), 按|r_v|排序:")
    L(f"  {'stat':<14}{'r(z_train)':>12}{'r(s_V)':>12}")
    for k, rz, rv in a2_by_sv:
        L(f"  {k:<14}{fmt(rz,4):>12}{fmt(rv,4):>12}")
    L(f"  top3 by |r(s_V)|: {top3_sv}")
    L(f"  top3 by |r(z_train)|: {[t[0] for t in a2_by_z[:3]]}")

    # ---- A3
    L("")
    L("A3. 仅低级统计判别力 (StandardScaler fit train -> LR(C=1e-3,lbfgs); test AUC):")
    L(f"  stat-only(all 14) AUC       = {statonly_auc:.4f}   [对照: V_full={V_full_auc:.4f}, C_full={C_full_auc:.4f}]")
    L(f"  stat-only(top3 by |r_sV|)   = {statonly_top3_auc:.4f}   top3={top3_sv}")
    L(f"  delta(all14 - top3)         = {statonly_auc - statonly_top3_auc:.4f}   (<=0.01 => <=3 stats suffice: {top3_suffices})")
    L("  14 个统计各自单统计 AUC (desc):")
    for k, a in single:
        L(f"    {k:<14}{a:.4f}")
    L(f"  #stats single-AUC >= 0.80 : {n_single_ge80}")

    # ---- A4
    L("")
    L("A4. OLS (LinearRegression, intercept) on train:  s_V ~ 14stats,  s_C ~ 14stats:")
    L(f"  adjR2(s_V ~ stats) = {adj_v:.4f}   (R2={r2_v:.4f}, n={n_tr}, p={K})")
    L(f"  adjR2(s_C ~ stats) = {adj_c:.4f}   (R2={r2_c:.4f})")
    L("  解读(机械): V高C低 => V判别可被低级量解释而C不能 (V捷径/C弥散语义候选)")

    # ---- A5
    L("")
    L("A5. fake 子方法拆解 (train). 显示 top|d| 统计均值 per method + real 均值:")
    disp = top_d_stats
    L("  method / count | " + " | ".join(f"{k}" for k in disp))
    for mm, cnt, gm in a5:
        L(f"  {mm:<15} n={cnt:<4} | " + " | ".join(f"{gm.get(k, float('nan')):.2f}" for k in disp))
    L(f"  real            n={int(real_tr.sum()):<4} | " + " | ".join(f"{real_means[k]:.2f}" for k in disp))
    L(f"  (train fake counts total={int(fake_tr.sum())})")

    # ---- A6
    L("")
    L("A6. 审计:")
    L("  14 统计 train 相关矩阵 (Pearson):")
    L("        " + "".join(f"{k[:6]:>7}" for k in STAT_KEYS))
    for i, k in enumerate(STAT_KEYS):
        L(f"  {k[:6]:>7}" + "".join(f"{corr14[i, j]:7.2f}" for j in range(K)))
    L(f"  native h,w 分布(全部{n_te + n_tr}张): h min/med/max = {h_min:.0f}/{h_med:.0f}/{h_max:.0f}, "
      f"w min/med/max = {w_min:.0f}/{w_med:.0f}/{w_max:.0f}")
    L(f"  读图失败数: {len(fail_idx)}")
    if len(fail_idx):
        L(f"  失败样本: {[int(x) for x in fail_idx[:20]]}")

    # ---- verdict
    L("")
    L("VERDICT logic:")
    L(f"  stat-only test AUC={statonly_auc:.4f} ; adjR2_V={adj_v:.4f} ; <=3stats_suffice={top3_suffices}")
    L(f"  G8A_VERDICT = {verdict}")

    L("")
    L("Caveats (honest):")
    L(f"  * 图像原生分辨率实测 h/w min/med/max = 70/167/920 / 70/161/834 px; 统一重采样到 224 "
      "做 LANCZOS (多数轻微上采样, 少数明显下采样). hi_en / spec_slope 等频域/锐度统计在重采样后高频被轻微平滑/衰减, "
      "单统计判别力可能被低估.")
    L("  * A2/A4 用 in-sample decision function / in-sample OLS: R^2/adjusted-R^2 乐观, 是上界而非"
      "泛化值; A3 的 test AUC 是真正的 out-of-video 泛化.")
    L("  * A2 的 Pearson r 为相关, 不构成因果; V logit 与低级统计高度相关不能证明 ViT 只用了这些量,"
      "只能证明这些量足以线性复现其判别轴. 相关!=因果.")
    L("  * PC0/e0 的符号由 SVD/eigh 约定决定, 无物理意义; r(z) 符号相应任意, |r| 才有意义.")
    L("  * RES_train/RES_test 定义 = rank-1 PC0 重建残差范数(任务缓存 schema 列名但无分析步骤使用).")
    L("")
    L(f"wall time = {time.time()-t_start:.1f}s")

    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\n[report] written -> {REPORT}")


if __name__ == "__main__":
    main()
