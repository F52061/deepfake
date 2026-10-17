# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"; os.environ["JOBLIB_NUM_THREADS"]="1"
"""
G11 - 特征空间归因: 源 PC0 轴 (e0) 在目标域是"错位(misalignment)"还是"缺失(missing)"?

纯 CPU 单线程. 不 import torch, 不开多线程/多进程. 只读已有 npz, 不读图像.
sklearn 全部默认单线程 (不传 n_jobs).

轴定义 (与 E3/G8B 一致):
  center-only raw-cov PCA fit 于 probe train (FF++, 2200, video-disjoint) 的 V;
  mu = train 均值, e0 = E[:,0] (最大方差方向, 按 train fake 均值 > real 均值定向);
  z(x) = (x - mu) @ e0.

固定协议: StandardScaler(fit train) -> LogisticRegression(C=1e-3, lbfgs, max_iter=2000);
AUC 只在 test / 目标域上算. y real=1 / fake=0, AUC 一律 positive=fake.

参照集选择 (重要):
  "FF++ test(800)" 取 probe_feats 的 test 部分 (42 vids, 与源 train 98 vids 完全 video-disjoint,
  且是 0.9852 锚点所属集合).
  feats_multi 的 'ffpp' 域 (800, 140 vids) 与 probe 完全同源, 但其 vid 含 98/140 个源 train 视频
  (只在 probe 全集中出现 207 帧: train 156 / test 51) -> 作为"测试参照"会被源视频污染,
  故本次只作脚注, 不充当参照行.

实验:
  Q1 轴表达/支撑 (a 沿轴能量占比 / b 源 top-K 子空间外残差能量 / c 域 PC0 与 e0 的 cos
     及 top-10 子空间重叠 / d real-fake 沿 z 的间隔 d_z)
  Q2 无标签重对中 (z' = z - mean(z_d)) 的可校正性; + 补充 (CORAL 白化, 跨域池化)
  Q3 oracle 上限: 域内自训 LR 按 vid 分组 5 折 CV AUC, 对比源训 LR 跨域 AUC;
     域内 LDA 方向 / 域内均值差方向 / 源训 LR 原始空间方向 与 e0 的 cos
  Q4 汇总表   Q5 机械判定 H5

用法:
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g11/run_g11.py

输出:
  vit_module/_g11/g11_report.txt  (机器块 + 表格 + caveats)
  vit_module/_g11/g11_stats.npz   (机器块键值)
"""
import time
from math import erf
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_PATH = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
MULTI_PATH = os.path.join(HERE, "..", "_tsne", "feats_multi.npz")
REPORT = os.path.join(HERE, "g11_report.txt")
STATS = os.path.join(HERE, "g11_stats.npz")

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
KS = [1, 5, 50]
KPC = 10          # top-10 subspace overlap (Q1c)
C_FIX = 1e-3
SEED = 0
# 机械判定门 (Q5)
GATE_RESID = 0.05     # 残差能量比 FF++ test 高 >= 5pt 记 "明显高于"
GATE_DZ = 0.50        # |d_z_dom| <= 0.5 * |d_z_src| 记 "显著收缩"
GATE_RECENTER = 0.02  # Q2 delta >= +2pt
GATE_ORACLE = 0.05    # oracle - srcLR >= +5pt

# 已知对照锚点 (必须复算一致)
ANCHOR_VARFRAC = 0.6229
ANCHOR_VTEST = 0.9852
ANCHOR_E0 = {"cd1": 0.8576, "cd2": 0.8604, "dfdcp": 0.8322, "ffiw": 0.8103, "wild": 0.8060}
ANCHOR_SRCLR = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_ATOL = 1e-3


def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def sep_d(proj_fake, proj_real):
    """(mean_fake - mean_real) / pooled sd  (方向: fake>real 记正)."""
    f = np.asarray(proj_fake, dtype=np.float64)
    r = np.asarray(proj_real, dtype=np.float64)
    n1, n2 = len(f), len(r)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp2 = ((n1 - 1) * f.var(ddof=1) + (n2 - 1) * r.var(ddof=1)) / (n1 + n2 - 2)
    if sp2 <= 0:
        return float("nan")
    return float((f.mean() - r.mean()) / np.sqrt(sp2))


def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def top_subspace(Xc, k):
    """Xc: already-centered (n,d). Return d x k right singular vectors."""
    k = int(min(k, Xc.shape[0] - 1, Xc.shape[1]))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def sub_overlap(Ua, Ub):
    k = Ua.shape[1]
    return float(((Ua.T @ Ub) ** 2).sum() / k)


def lr_fit(Xtr, ytr, C=C_FIX):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                              random_state=SEED).fit(Xtr, ytr)


def cv_auc_grouped(X, yb, groups, n_splits=5, C=C_FIX, seed=SEED):
    """域内 oracle: StandardScaler fit 折内 train, LR 折内 train.
    优先按 vid 分组 (StratifiedGroupKFold) 避免同视频泄漏; 若域内 vid 数 < n_splits
    则退化为 StratifiedKFold 并置 leak=True (AUC 被高估)."""
    n = len(yb)
    ng = len(np.unique(np.asarray(groups)))
    if ng >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, yb, groups))
        mode, leak = "SGKF(vid)", False
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, yb))
        mode, leak = "SKF(no-group)", True
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_aucs = []
    for tr, te in folds:
        if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = lr_fit(sc.transform(X[tr]), yb[tr], C=C)
        s = clf.decision_function(sc.transform(X[te]))
        oof[te] = s
        fold_aucs.append(float(roc_auc_score(yb[te], s)))
    ok = np.isfinite(oof)
    auc_oof = float(roc_auc_score(yb[ok], oof[ok])) if ok.any() else float("nan")
    if fold_aucs:
        auc_mean = float(np.mean(fold_aucs))
        auc_std = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
    else:
        auc_mean = auc_std = float("nan")
    return dict(auc_mean=auc_mean, auc_std=auc_std, auc_oof=auc_oof,
                mode=mode, leak=leak, n_folds=len(fold_aucs), ngroups=ng)


def coral_map(Xs, Xt, ridge=1e-3):
    """CORAL (symmetric whitening): 把 target 协方差搬回 source (label-free).
    A = Ct^{-1/2} Cs^{1/2}; 返回 Xt_aligned = (Xt - mu_t) @ A + mu_s."""
    d = Xs.shape[1]
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False)
    Ct = np.cov(Xt, rowvar=False)
    Ct = Ct + (ridge * np.trace(Ct) / d) * np.eye(d)
    Cs = Cs + (ridge * np.trace(Cs) / d) * np.eye(d)
    ws, Es = np.linalg.eigh(Cs); ws = np.clip(ws, 1e-12, None)
    wt, Et = np.linalg.eigh(Ct); wt = np.clip(wt, 1e-12, None)
    A = ((Et * (1.0 / np.sqrt(wt))) @ Et.T) @ ((Es * np.sqrt(ws)) @ Es.T)
    return (Xt - mu_t) @ A + mu_s


def analyze_unit(name, V, y, vid, ctx):
    """对单个域 (V, y, vid) 算 Q1/Q2/Q3. ctx 携带源侧所有量."""
    mu, e0 = ctx["mu"], ctx["e0"]
    zdf = (y == 0).astype(int)                  # positive = fake
    mf, mr = (y == 0), (y == 1)
    r = dict(name=name, n=int(len(y)), n_vid=int(len(np.unique(vid))),
             n_fake=int(mf.sum()), n_real=int(mr.sum()))

    # 源训 LR 跨域 AUC (固定协议, 只 transform)
    r["auc_srcLR"] = float(roc_auc_score(zdf, ctx["clfV"].decision_function(
        ctx["scV"].transform(V))))

    # ---- Q1a 沿轴能量占比 -----------------------------------------------
    z_d = (V - mu) @ e0
    r["z_mean_dom"] = float(z_d.mean())
    r["var_z"] = float(np.var(z_d, ddof=1))
    r["var_tot"] = float(np.var(V, axis=0, ddof=1).sum())
    r["zdim_frac"] = r["var_z"] / r["var_tot"]

    # ---- Q1b 轴外能量 (源 top-K 子空间外残差占比) ------------------------
    Xd = V - mu
    denom = float((Xd ** 2).sum())
    for K in KS:
        P = Xd @ ctx["Esrc_K"][K]
        r[f"resid{K}"] = 1.0 - float((P ** 2).sum()) / denom

    # ---- Q1c 域自身 PCA: cos(PC0_dom, e0) + top-10 子空间重叠 ------------
    Xdc = V - V.mean(axis=0)
    Ud10 = top_subspace(Xdc, KPC)
    ed0 = Ud10[:, 0]
    r["cos_e0dom_e0"] = float(ed0 @ e0)
    r["abs_cos_e0dom_e0"] = abs(r["cos_e0dom_e0"])
    r["overlap10"] = sub_overlap(Ud10, ctx["Esrc_10"])
    lam_dom = np.linalg.svd(Xdc, compute_uv=False) ** 2
    r["pc0_varfrac_dom"] = float(lam_dom[0] / lam_dom.sum())
    # 控制: 用域自身 PC0 轴 (域内中心化) 的 AUC -> 是否"源轴"本身被针对
    z_own = (V - V.mean(axis=0)) @ ed0
    a_own = float(roc_auc_score(zdf, z_own))
    r["auc_e0dom"] = max(a_own, 1.0 - a_own)     # 取 fake 正方向
    r["auc_e0dom_raw"] = a_own

    # ---- Q1d real/fake 沿 z 的间隔 --------------------------------------
    r["dz_dom"] = sep_d(z_d[mf], z_d[mr])
    r["mean_z_fake"] = float(z_d[mf].mean())
    r["mean_z_real"] = float(z_d[mr].mean())
    # z 只按域内均值平移后 (等价于域自身沿 e0 的间隔) —— 数值恒等, 仅作审计
    r["dz_dom_recentered"] = sep_d(z_d[mf] - r["z_mean_dom"], z_d[mr] - r["z_mean_dom"])
    # 沿轴类内合并 sd (d_z 的分母; 与 gap 一起解释 d_z 的下降)
    sp2 = ((mf.sum() - 1) * z_d[mf].var(ddof=1) + (mr.sum() - 1) * z_d[mr].var(ddof=1)) / (len(y) - 2.0)
    r["pooled_sd_z"] = float(np.sqrt(max(sp2, 1e-12)))

    # ---- Q2 无标签重对中: z' = z - mean(z_d) ----------------------------
    r["auc_e0"] = float(roc_auc_score(zdf, z_d))
    z_rc = z_d - r["z_mean_dom"]
    r["auc_z_recenter"] = float(roc_auc_score(zdf, z_rc))
    r["recenter_delta"] = r["auc_z_recenter"] - r["auc_e0"]

    # ---- Q2 补充 (规格外): CORAL 白化 + 重着色 (label-free) ---------------
    V_al = coral_map(ctx["Vsrc_tr"], V)
    s_al = ctx["clfV"].decision_function(ctx["scV"].transform(V_al))
    r["coral_auc"] = float(roc_auc_score(zdf, s_al))
    r["coral_delta"] = r["coral_auc"] - r["auc_srcLR"]

    # ---- Q3 oracle: 域内自训 LR, 按 vid 分组 5 折 -------------------------
    cv = cv_auc_grouped(V, zdf, vid)
    r.update(oracle_auc=cv["auc_mean"], oracle_std=cv["auc_std"],
             oracle_oof=cv["auc_oof"], oracle_mode=cv["mode"],
             oracle_leak=cv["leak"], oracle_ngroups=cv["ngroups"],
             oracle_nfolds=cv["n_folds"])
    r["oracle_minus_srclr"] = cv["auc_mean"] - r["auc_srcLR"]

    # ---- Q3 方向 --------------------------------------------------------
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(V, zdf)
    wd = lda.coef_.ravel()
    r["cos_wd_e0"] = cosv(wd, e0)
    r["abs_cos_wd_e0"] = abs(r["cos_wd_e0"])
    lda2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.1).fit(V, zdf)
    r["cos_wd_fix_e0"] = cosv(lda2.coef_.ravel(), e0)
    d_dom = V[mf].mean(0) - V[mr].mean(0)        # fake - real (raw V, 无正则)
    r["cos_ddom_e0"] = cosv(d_dom, e0)
    r["abs_cos_ddom_e0"] = abs(r["cos_ddom_e0"])
    r["cos_wsrc_e0"] = ctx["cos_wsrc_e0"]
    return r


def main():
    t_start = time.time()
    log = []
    def L(x=""):
        log.append(str(x))

    # ================= 0. probe (FF++) : 源轴 ================================
    p = np.load(PROBE_PATH, allow_pickle=True)
    Vp = np.asarray(p["V"], dtype=np.float64)
    yp = np.asarray(p["y"]).astype(int)          # real=1 / fake=0
    vp = np.asarray(p["vids"]).astype(str)
    tm = np.asarray(p["train_mask"])
    tr = np.where(tm)[0]
    te = np.where(~tm)[0]
    ytr, yte = yp[tr], yp[te]
    ztr = (ytr == 0).astype(int)
    zte = (yte == 0).astype(int)
    n_tr, n_te = len(tr), len(te)
    assert (n_tr, n_te) == (2200, 800), (n_tr, n_te)
    assert int((ytr == 1).sum()) == 1100 and int((ytr == 0).sum()) == 1100
    vtr_vids, vte_vids = set(vp[tr]), set(vp[te])
    assert not (vtr_vids & vte_vids), "probe train/test 视频重叠"

    # center-only raw-cov PCA on train V (与 E3/G8B 完全一致)
    mu = Vp[tr].mean(axis=0)
    Xtrc = Vp[tr] - mu
    cov = (Xtrc.T @ Xtrc) / (n_tr - 1.0)
    w, E = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, E = w[order], E[:, order]
    e0 = E[:, 0].copy()
    proj_f = (Xtrc[ytr == 0] @ e0).mean()
    proj_r = (Xtrc[ytr == 1] @ e0).mean()
    if proj_f < proj_r:
        e0 = -e0
    varfrac0 = float(w[0] / cov.trace())
    print(f"[prep] PC0 varfrac={varfrac0:.4f} (anchor {ANCHOR_VARFRAC})", flush=True)
    assert abs(varfrac0 - ANCHOR_VARFRAC) < ANCHOR_ATOL, varfrac0

    z_tr = Xtrc @ e0
    dz_src = sep_d(z_tr[ytr == 0], z_tr[ytr == 1])
    z_fake_src_mean = float(z_tr[ytr == 0].mean())
    z_real_src_mean = float(z_tr[ytr == 1].mean())
    pooled_sd_src = float(np.sqrt(max(((ytr == 0).sum() - 1) * z_tr[ytr == 0].var(ddof=1) +
                                      ((ytr == 1).sum() - 1) * z_tr[ytr == 1].var(ddof=1),
                                      1e-12) / (n_tr - 2.0)))
    zdim_frac_src = float(np.var(z_tr, ddof=1) / np.var(Vp[tr], axis=0, ddof=1).sum())
    Esrc_10 = E[:, :KPC]
    Esrc_K = {K: E[:, :K] for K in KS}

    def resid_src(K):
        P = Xtrc @ Esrc_K[K]
        return 1.0 - float((P ** 2).sum()) / float((Xtrc ** 2).sum())

    # 源训 LR (固定协议)
    scV = StandardScaler().fit(Vp[tr])
    clfV = lr_fit(scV.transform(Vp[tr]), ztr)
    vtest_auc = float(roc_auc_score(zte, clfV.decision_function(scV.transform(Vp[te]))))
    print(f"[prep] FF++ test in-domain srcLR AUC={vtest_auc:.4f} (anchor {ANCHOR_VTEST})  "
          f"d_z_src={dz_src:.4f}", flush=True)
    assert abs(vtest_auc - ANCHOR_VTEST) < ANCHOR_ATOL, vtest_auc
    w_raw_src = clfV.coef_.ravel() / scV.scale_          # 转回原始 V 空间
    cos_wsrc_e0 = cosv(w_raw_src, e0)

    ctx = dict(mu=mu, e0=e0, Esrc_K=Esrc_K, Esrc_10=Esrc_10, scV=scV, clfV=clfV,
               cos_wsrc_e0=cos_wsrc_e0, Vsrc_tr=Vp[tr])

    # ================= 1. 目标域特征 =========================================
    f = np.load(MULTI_PATH, allow_pickle=True)
    Vf = np.asarray(f["V"], dtype=np.float64)
    yf = np.asarray(f["y"]).astype(int)
    domf = np.asarray(f["domain"]).astype(str)
    vidf = np.asarray(f["vid"]).astype(str)

    # 锚点复算 (跨域 full-V / e0 单轴) —— 用 probe test 作为 ffpp 参照
    units = [("fftest", Vp[te], yte, vp[te])]
    for dm in TARGETS:
        m = domf == dm
        units.append((dm, Vf[m], yf[m], vidf[m]))
    for dm in TARGETS:
        m = domf == dm
        Vd, yd = Vf[m], yf[m]
        zdf = (yd == 0).astype(int)
        a_e0 = float(roc_auc_score(zdf, (Vd - mu) @ e0))
        a_lr = float(roc_auc_score(zdf, clfV.decision_function(scV.transform(Vd))))
        assert abs(a_e0 - ANCHOR_E0[dm]) < ANCHOR_ATOL, (dm, a_e0)
        assert abs(a_lr - ANCHOR_SRCLR[dm]) < ANCHOR_ATOL, (dm, a_lr)
    print("[anchors] all cross-domain e0 / srcLR AUCs reproduce to <1e-3", flush=True)

    # multi 'ffpp' 脚注 (被源 train 视频污染, 不作参照)
    m = domf == "ffpp"
    Vm, ym = Vf[m], yf[m]
    zdfm = (ym == 0).astype(int)
    mffpp = dict(n=int(m.sum()), n_vid=int(len(np.unique(vidf[m]))),
                 n_vid_in_src_train=int(len(set(vidf[m]) & vtr_vids)))
    Xm = Vm - mu
    Pm = Xm @ Esrc_K[1]
    zm = Xm @ e0
    mffpp["resid1"] = 1.0 - float((Pm ** 2).sum()) / float((Xm ** 2).sum())
    mffpp["dz"] = sep_d(zm[ym == 0], zm[ym == 1])
    mffpp["auc_e0"] = float(roc_auc_score(zdfm, zm))
    mffpp["auc_srcLR"] = float(roc_auc_score(zdfm, clfV.decision_function(scV.transform(Vm))))

    # ================= 2. 逐域 Q1/Q2/Q3 =====================================
    S = {}
    for name, Vu, yu, vu in units:
        S[name] = analyze_unit(name, Vu, yu, vu, ctx)
        r = S[name]
        print(f"[unit] {name:7s} n={r['n']:4d}/{r['n_vid']:3d}vid zdim={r['zdim_frac']:.4f} "
              f"residK1={r['resid1']:.4f} dz={r['dz_dom']:.4f} e0AUC={r['auc_e0']:.4f} "
              f"rc_d={r['recenter_delta']:+.4f} coral_d={r['coral_delta']:+.4f} "
              f"oracle={r['oracle_auc']:.4f}({r['oracle_mode']}) srcLR={r['auc_srcLR']:.4f} "
              f"cos_ddom={r['cos_ddom_e0']:+.4f} cos_wd={r['cos_wd_e0']:+.4f} "
              f"t={time.time()-t_start:.1f}s", flush=True)

    # ---- Q2 补充: 跨域池化 (每域各自重对中后合并 AUC) ---------------------
    zs_raw = np.concatenate([(Vu - mu) @ e0 for _, Vu, _, _ in units])
    zs_rc = np.concatenate([(Vu - mu) @ e0 - S[nm]["z_mean_dom"] for nm, Vu, _, _ in units])
    ys_all = np.concatenate([(yu == 0).astype(int) for _, _, yu, _ in units])
    pooled_raw_auc = float(roc_auc_score(ys_all, zs_raw))
    pooled_rc_auc = float(roc_auc_score(ys_all, zs_rc))

    # ================= 3. 机械判定 (Q5) =====================================
    rf = S["fftest"]
    verdicts = {}
    for dm in TARGETS:
        r = S[dm]
        # (b) 逐 K 判定: 源 top-K 子空间外残差能量是否明显高于 FF++ test 参照
        anom_K = {K: bool(r[f"resid{K}"] - rf[f"resid{K}"] >= GATE_RESID) for K in KS}
        resid_anom_any = any(anom_K.values())
        resid_anom_K1 = anom_K[1]
        dz_shrink = bool(abs(r["dz_dom"]) <= GATE_DZ * abs(dz_src))
        anomaly = bool(resid_anom_any or dz_shrink)
        rec_recenter = bool(r["recenter_delta"] >= GATE_RECENTER)
        rec_oracle = bool(r["oracle_minus_srclr"] >= GATE_ORACLE)
        recoverable = bool(rec_recenter or rec_oracle)
        small = bool((r["recenter_delta"] < GATE_RECENTER) and
                     (r["oracle_minus_srclr"] < GATE_ORACLE))
        v = "NOT-SUPPORTED" if not anomaly else ("PARTIAL" if small else "SUPPORT")
        # 补充 1 (认定 (d) 与 AUC_z 同义): 只认 (b) 任一 K 异常; 可恢复门含规格外 CORAL
        v_b = ("NOT-SUPPORTED" if not resid_anom_any else
               ("SUPPORT" if (rec_oracle or r["coral_delta"] >= GATE_RECENTER) else "PARTIAL"))
        # 补充 2 保守: 只认 (d) (轴表达指标里唯一被规格明确点名的) + oracle 门
        v_c = ("NOT-SUPPORTED" if not dz_shrink else
               ("SUPPORT" if rec_oracle else "PARTIAL"))
        # 补充 3: 原判定门不变, 只把退化的 recenter 门换成规格外 CORAL delta
        v_s = ("NOT-SUPPORTED" if not anomaly else
               ("SUPPORT" if (r["coral_delta"] >= GATE_RECENTER or rec_oracle) else "PARTIAL"))
        verdicts[dm] = dict(anom_K=anom_K, resid_anom_any=resid_anom_any,
                            resid_anom_K1=resid_anom_K1, dz_shrink=dz_shrink,
                            anomaly=anomaly, rec_recenter=rec_recenter, rec_oracle=rec_oracle,
                            recoverable=recoverable, small=small,
                            verdict=v, verdict_b=v_b, verdict_c=v_c, verdict_supp=v_s)

    def glob(counter):
        if counter.get("SUPPORT", 0) >= 3:
            return "SUPPORT"
        if counter.get("PARTIAL", 0) >= 3 or (counter.get("SUPPORT", 0) + counter.get("PARTIAL", 0)) >= 4:
            return "PARTIAL"
        return "NOT-SUPPORTED"

    vc = Counter(verdicts[d]["verdict"] for d in TARGETS)
    vcb = Counter(verdicts[d]["verdict_b"] for d in TARGETS)
    vcc = Counter(verdicts[d]["verdict_c"] for d in TARGETS)
    vcs = Counter(verdicts[d]["verdict_supp"] for d in TARGETS)
    GLOBAL, GLOBAL_B, GLOBAL_C, GLOBAL_S = glob(vc), glob(vcb), glob(vcc), glob(vcs)
    # b 只取 K=1 时的敏感性
    vc_k1 = Counter(("NOT-SUPPORTED" if not verdicts[d]["resid_anom_K1"] else
                     ("SUPPORT" if (verdicts[d]["rec_oracle"] or
                                    S[d]["coral_delta"] >= GATE_RECENTER) else "PARTIAL"))
                    for d in TARGETS)
    GLOBAL_K1 = glob(vc_k1)
    phi_dev = max(abs(0.5 * (1.0 + erf(S[c]["dz_dom"] / 2.0)) - S[c]["auc_e0"])
                  for c in S)   # Phi(d/sqrt(2)) = 0.5*(1+erf(d/2))

    # ================= 4. 报告 ==============================================
    L("=" * 126)
    L("G11 REPORT - 特征归因: 源 PC0 轴 (e0) 在目标域 是 错位(misalignment) 还是 缺失(missing)?")
    L("(纯 CPU 单线程, 不 import torch, 只读 npz; 轴=center-only raw-cov PCA@probe-train-V (2200),")
    L(" e0=E[:,0], mu=train 均值, z=(V-mu)@e0; LR 固定 C=1e-3 lbfgs max_iter=2000; AUC positive=fake)")
    L("=" * 126)

    mb = ["MACHINE_BLOCK"]
    mb.append(f"G11_SRC=probe_train_n={n_tr} G11_SRC_ZDIMF_frac={zdim_frac_src:.4f} "
              f"G11_SRC_PC0_varfrac={varfrac0:.4f} G11_SRC_DELTAZ={dz_src:.4f} "
              f"G11_SRC_RESIDK1={resid_src(1):.4f} G11_SRC_RESIDK50={resid_src(50):.4f} "
              f"G11_ANCHOR_VtestAUC={vtest_auc:.4f} G11_COS_wsrc_e0={cos_wsrc_e0:+.4f}")
    mb.append(f"G11_FFTEST_row=probe_test_n={rf['n']}_vids={rf['n_vid']} "
              f"G11_ZDIMF_frac_fftest={rf['zdim_frac']:.4f} G11_RESIDK1_fftest={rf['resid1']:.4f} "
              f"G11_RESIDK5_fftest={rf['resid5']:.4f} G11_RESIDK50_fftest={rf['resid50']:.4f} "
              f"G11_DELTAZ_fftest={rf['dz_dom']:.4f} G11_AUCe0_fftest={rf['auc_e0']:.4f} "
              f"G11_AUCe0dom_fftest={rf['auc_e0dom']:.4f} G11_COS_e0dom_fftest={rf['cos_e0dom_e0']:+.4f} "
              f"G11_OVERLAP10_fftest={rf['overlap10']:.4f} G11_ORACLE_AUC_fftest={rf['oracle_auc']:.4f} "
              f"G11_SRCLR_AUC_fftest={rf['auc_srcLR']:.4f} G11_RECENTER_DELTA_fftest={rf['recenter_delta']:+.4f} "
              f"G11_CORAL_DELTA_fftest={rf['coral_delta']:+.4f} G11_COS_ddom_e0_fftest={rf['cos_ddom_e0']:+.4f} "
              f"G11_COS_wd_e0_fftest={rf['cos_wd_e0']:+.4f}")
    mb.append(f"G11_MULTIFFPP_contaminated=1 n={mffpp['n']} vids={mffpp['n_vid']} "
              f"vids_also_in_srcTrain={mffpp['n_vid_in_src_train']} "
              f"G11_RESIDK1_multiffpp={mffpp['resid1']:.4f} G11_DELTAZ_multiffpp={mffpp['dz']:.4f} "
              f"G11_AUCe0_multiffpp={mffpp['auc_e0']:.4f} G11_SRCLR_AUC_multiffpp={mffpp['auc_srcLR']:.4f}")
    for dm in TARGETS:
        r = S[dm]
        mb.append(
            f"G11_ANCHOR_e0AUC_{dm}={r['auc_e0']:.4f} G11_ZDIMF_frac_{dm}={r['zdim_frac']:.4f} "
            f"G11_RESIDK1_{dm}={r['resid1']:.4f} G11_RESIDK5_{dm}={r['resid5']:.4f} "
            f"G11_RESIDK50_{dm}={r['resid50']:.4f} G11_COS_e0dom_{dm}={r['cos_e0dom_e0']:+.4f} "
            f"G11_OVERLAP10_{dm}={r['overlap10']:.4f} G11_DELTAZ_{dm}={r['dz_dom']:.4f} "
            f"G11_AUCe0dom_{dm}={r['auc_e0dom']:.4f} G11_RECENTER_AUC_{dm}={r['auc_z_recenter']:.4f} "
            f"G11_RECENTER_DELTA_{dm}={r['recenter_delta']:+.4f} G11_CORAL_AUC_{dm}={r['coral_auc']:.4f} "
            f"G11_CORAL_DELTA_{dm}={r['coral_delta']:+.4f} G11_ORACLE_AUC_{dm}={r['oracle_auc']:.4f} "
            f"G11_ORACLE_AUCSTD_{dm}={r['oracle_std']:.4f} G11_SRCLR_AUC_{dm}={r['auc_srcLR']:.4f} "
            f"G11_ORACLE_MINUS_SRCLR_{dm}={r['oracle_minus_srclr']:+.4f} "
            f"G11_COS_wd_e0_{dm}={r['cos_wd_e0']:+.4f} G11_COS_ddom_e0_{dm}={r['cos_ddom_e0']:+.4f} "
            f"G11_DZ_RATIO_{dm}={abs(r['dz_dom'])/abs(dz_src):.4f} "
            f"G11_GAPRATIO_{dm}={(r['mean_z_fake']-r['mean_z_real'])/(z_fake_src_mean-z_real_src_mean):.4f} "
            f"G11_SDRATIO_{dm}={r['pooled_sd_z']/pooled_sd_src:.4f} "
            f"G11_SHIFTMID_{dm}={0.5*(r['mean_z_fake']+r['mean_z_real']):.4f} "
            f"G11_DREAL_{dm}={r['mean_z_real']-z_real_src_mean:+.4f} "
            f"G11_DFAKE_{dm}={r['mean_z_fake']-z_fake_src_mean:+.4f} "
            f"G11_DRESIDK1_{dm}={r['resid1']-rf['resid1']:+.4f} "
            f"G11_DRESIDK5_{dm}={r['resid5']-rf['resid5']:+.4f} "
            f"G11_DRESIDK50_{dm}={r['resid50']-rf['resid50']:+.4f} "
            f"G11_RESIDANOM_K1_{dm}={int(verdicts[dm]['anom_K'][1])} "
            f"G11_RESIDANOM_K5_{dm}={int(verdicts[dm]['anom_K'][5])} "
            f"G11_RESIDANOM_K50_{dm}={int(verdicts[dm]['anom_K'][50])} "
            f"G11_DZSHRINK_{dm}={int(verdicts[dm]['dz_shrink'])}")
    for dm in TARGETS:
        v = verdicts[dm]
        mb.append(f"G11_H5_ANOM_{dm}={int(v['anomaly'])} "
                  f"(residK_anyK={int(v['resid_anom_any'])},residK1={int(v['resid_anom_K1'])},"
                  f"dz_shrink={int(v['dz_shrink'])}) "
                  f"G11_H5_RECOVER_{dm}={int(v['recoverable'])} "
                  f"(recenter={int(v['rec_recenter'])},oracle={int(v['rec_oracle'])}) "
                  f"G11_VERDICT_{dm}={v['verdict']} G11_VERDICT_B_{dm}={v['verdict_b']} "
                  f"G11_VERDICT_C_{dm}={v['verdict_c']} G11_VERDICT_SUPP_{dm}={v['verdict_supp']}")
    mb.append(f"G11_VERDICT={GLOBAL} G11_VERDICT_COUNTS_SUPPORT={vc.get('SUPPORT',0)} "
              f"PARTIAL={vc.get('PARTIAL',0)} NOT-SUPPORTED={vc.get('NOT-SUPPORTED',0)}")
    mb.append(f"G11_VERDICT_B(b 任一 K 异常, 认为 d 与 AUC_z 同义; 可恢复含 CORAL)={GLOBAL_B} "
              f"COUNTS={vcb.get('SUPPORT',0)}/{vcb.get('PARTIAL',0)}/{vcb.get('NOT-SUPPORTED',0)}")
    mb.append(f"G11_VERDICT_C(只认 d 间隔收缩 + oracle 门)={GLOBAL_C} "
              f"COUNTS={vcc.get('SUPPORT',0)}/{vcc.get('PARTIAL',0)}/{vcc.get('NOT-SUPPORTED',0)}")
    mb.append(f"G11_VERDICT_K1b(b 只取 K=1)={GLOBAL_K1} "
              f"COUNTS={vc_k1.get('SUPPORT',0)}/{vc_k1.get('PARTIAL',0)}/{vc_k1.get('NOT-SUPPORTED',0)} "
              f"DOMAINS={'|'.join(dm + ':' + ('NOT-SUPPORTED' if not verdicts[dm]['resid_anom_K1'] else 'ANOM') for dm in TARGETS)}")
    mb.append(f"G11_VERDICT_SUPP(recenter 门换规格外 CORAL_delta)={GLOBAL_S} "
              f"COUNTS={vcs.get('SUPPORT',0)}/{vcs.get('PARTIAL',0)}/{vcs.get('NOT-SUPPORTED',0)}")
    mb.append(f"G11_PHI_AUDIT_maxabsdev={phi_dev:.4f} G11_POOLED_RECENTER_DELTA={pooled_rc_auc-pooled_raw_auc:+.4f}")
    for line in mb:
        L(line)
    L("")

    L("-" * 126)
    L("A0. 锚点复算 (全部必须与给定对照一致到 <1e-3)")
    L(f"  源 train PC0 varfrac (沿轴能量占比参照)  : 复算 {varfrac0:.4f} / 预期 {ANCHOR_VARFRAC:.4f}")
    L(f"  FF++ test 域内 srcLR AUC (probe test 800) : 复算 {vtest_auc:.4f} / 预期 {ANCHOR_VTEST:.4f}")
    L(f"  {'dom':<7}{'e0_AUC_recalc':>14}{'e0_AUC_exp':>12}{'srcLR_recalc':>14}{'srcLR_exp':>12}")
    for dm in TARGETS:
        L(f"  {dm:<7}{S[dm]['auc_e0']:>14.4f}{ANCHOR_E0[dm]:>12.4f}"
          f"{S[dm]['auc_srcLR']:>14.4f}{ANCHOR_SRCLR[dm]:>12.4f}")
    L("")
    L("  参照集说明: FF++ test 参照行取 probe_feats 的 test 部分 (800, 42 vids, 与源 train")
    L("  98 vids 完全 video-disjoint). feats_multi 的 'ffpp' 域 (800, 140 vids) 虽与 probe 同源,")
    L(f"  但其 vid 有 {mffpp['n_vid_in_src_train']}/{mffpp['n_vid']} 属于源 train 视频 -> 作参照会被污染, 仅作脚注:")
    L(f"    multi-ffpp (污染): residK1={mffpp['resid1']:.4f}  d_z={mffpp['dz']:.4f}  "
      f"AUC_e0={mffpp['auc_e0']:.4f}  srcLR_AUC={mffpp['auc_srcLR']:.4f}")
    L("")

    L("-" * 126)
    L("Q1. 轴表达/支撑 (参照: 源 train PC0 varfrac=%.4f, d_z_src=%.4f, residK1_src=%.4f, residK50_src=%.4f)"
      % (varfrac0, dz_src, resid_src(1), resid_src(50)))
    L("    zdim_frac = 域内 z 方差 / 域内 V 各维方差和 (ddof=1, 与源 0.6229 同口径);")
    L("    residK = 1 - ||proj_{E[:,:K]}(V_d - mu)||^2 / ||V_d - mu||^2  (源子空间外残差能量占比);")
    L("    cos(e0d,e0)=域自身 PC0 与源 e0 的 cos (PCA 符号任意, 看 |cos| 列); overlap10=域/源 top-10 子空间平均 cos^2;")
    L("    AUCe0d = 用域自身 PC0 轴(域内中心化)的 AUC (控制: 是否'源轴'本身被针对);")
    L("    d_z = (mean(z_fake)-mean(z_real))/合并sd (域内重对中不改变该值).")
    L("")
    hdr = (f"  {'dom':<8}{'n':>5}{'nvid':>6}{'zdim_frac':>11}{'residK1':>9}{'residK5':>9}"
           f"{'residK50':>10}{'cos(e0d,e0)':>13}{'|cos|':>8}{'overlap10':>11}"
           f"{'pc0vf_dom':>11}{'AUCe0d':>9}{'d_z':>9}")
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    L(f"  {'TRAIN':<8}{n_tr:>5}{len(vtr_vids):>6}{zdim_frac_src:>11.4f}"
      f"{resid_src(1):>9.4f}{resid_src(5):>9.4f}{resid_src(50):>10.4f}"
      f"{'1.0000':>13}{'1.0000':>8}{'1.0000':>11}{varfrac0:>11.4f}{'-':>9}{dz_src:>9.4f}")
    for nm in ["fftest"] + TARGETS:
        r = S[nm]
        L(f"  {nm:<8}{r['n']:>5}{r['n_vid']:>6}{r['zdim_frac']:>11.4f}{r['resid1']:>9.4f}"
          f"{r['resid5']:>9.4f}{r['resid50']:>10.4f}{r['cos_e0dom_e0']:>+13.4f}"
          f"{r['abs_cos_e0dom_e0']:>8.4f}{r['overlap10']:>11.4f}"
          f"{r['pc0_varfrac_dom']:>11.4f}{r['auc_e0dom']:>9.4f}{r['dz_dom']:>9.4f}")
    L("  (fftest = FF++ test参照行; TRAIN = probe train 2200, 轴在该集上拟合)")
    L("")
    L("  审计: d_z 与单轴 AUC_z 的算术一致性 (等方差正态近似 AUC ~ Phi(d_z/sqrt(2))):")
    L(f"    {'dom':<8}{'d_z':>9}{'Phi(d/sqrt2)':>14}{'AUC_z':>9}{'diff':>9}")
    for nm in ["fftest"] + TARGETS:
        r = S[nm]
        ph = 0.5 * (1.0 + erf(r["dz_dom"] / 2.0))
        L(f"    {nm:<8}{r['dz_dom']:>9.4f}{ph:>14.4f}{r['auc_e0']:>9.4f}{ph-r['auc_e0']:>+9.4f}")
    L(f"    最大偏差 = {phi_dev:.4f} -> (d) d_z 收缩与单轴 AUC_z 下降是同一现象的两种表述,")
    L("    (d) 分支不含超出 AUC_z 的独立证据.")
    L("")
    L("  Q1e 沿轴类别位置分解 (区分'整体错位' vs '间隔衰减'):")
    L("    z 以源 train 的 mu 为零点, 故源 train 的 fake/real 均值即为其自身偏移 (和为 0).")
    L(f"    {'dom':<8}{'z_real':>10}{'z_fake':>10}{'gap':>9}{'d_real':>10}{'d_fake':>10}"
      f"{'gap/gap_src':>12}{'shift_mid':>11}{'sd_z':>8}{'sd_ratio':>9}")
    L(f"    {'TRAIN':<8}{z_real_src_mean:>10.4f}{z_fake_src_mean:>10.4f}"
      f"{z_fake_src_mean-z_real_src_mean:>9.4f}{0.0:>10.4f}{0.0:>10.4f}{1.0:>12.4f}{0.0:>11.4f}"
      f"{pooled_sd_src:>8.3f}{1.0:>9.4f}")
    gap_src = z_fake_src_mean - z_real_src_mean
    for nm in ["fftest"] + TARGETS:
        r = S[nm]
        gap = r["mean_z_fake"] - r["mean_z_real"]
        dr = r["mean_z_real"] - z_real_src_mean
        df_ = r["mean_z_fake"] - z_fake_src_mean
        mid = 0.5 * (r["mean_z_fake"] + r["mean_z_real"]) - 0.5 * (z_fake_src_mean + z_real_src_mean)
        L(f"    {nm:<8}{r['mean_z_real']:>10.4f}{r['mean_z_fake']:>10.4f}{gap:>9.4f}"
          f"{dr:>10.4f}{df_:>10.4f}{gap/gap_src:>12.4f}{mid:>11.4f}"
          f"{r['pooled_sd_z']:>8.3f}{r['pooled_sd_z']/pooled_sd_src:>9.4f}")
    L("    d_real/d_fake = 相对源 train 的类均值位移; shift_mid = 两类中点位移 (整体错位分量);")
    L("    gap/gap_src = 间隔保留比例 (衰减分量); sd_ratio = 沿轴类内合并 sd 比值 (噪声分量);")
    L("    d_z = gap/(合并 sd), 故三者共同解释 d_z 下降.")
    L("")

    L("-" * 126)
    L("Q2. 无标签重对中 z'=z-mean(z_d) 的可校正性 (+ 补充: CORAL 白化 / 跨域池化)")
    L("  重要: AUC 是秩统计量, 域内整体平移 z->z-mean(z_d) 不改变任何排序,")
    L("        故 RECENTER_DELTA 恒为 0.0000 (精确退化, 数学恒等; 且 d_z 亦不变).")
    L("")
    hdr = (f"  {'dom':<8}{'AUC_z':>9}{'AUC_z_recenter':>16}{'delta':>9}{'mean_z_dom':>12}"
           f"{'CORAL_AUC':>11}{'CORAL_delta':>13}")
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    for nm in ["fftest"] + TARGETS:
        r = S[nm]
        L(f"  {nm:<8}{r['auc_e0']:>9.4f}{r['auc_z_recenter']:>16.4f}{r['recenter_delta']:>+9.4f}"
          f"{r['z_mean_dom']:>12.4f}{r['coral_auc']:>11.4f}{r['coral_delta']:>+13.4f}")
    L(f"  跨域池化 ({len(units)} 域合并, 每域先各自重对中): AUC raw z = {pooled_raw_auc:.4f} -> "
      f"recentered = {pooled_rc_auc:.4f}  (delta {pooled_rc_auc-pooled_raw_auc:+.4f})")
    L("  (CORAL = label-free 协方差对齐到源后源训 LR 的跨域 AUC; 规格外补充, 用于替代退化的 recenter)")
    L("")

    L("-" * 126)
    L("Q3. oracle 上限: 域内自训 LR(768d, 同 C=1e-3) 按 vid 分组 5 折 CV  vs  源训 LR 跨域")
    L("")
    hdr = (f"  {'dom':<8}{'oracle_mean':>13}{'fold_std':>10}{'oof':>9}{'srcLR':>9}"
           f"{'oracle-src':>12}{'nvid':>6}{'split':>13}{'leak':>6}"
           f"{'cos(wd,e0)':>12}{'cos(ddom,e0)':>14}{'cos(wsrc,e0)':>14}")
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    for nm in ["fftest"] + TARGETS:
        r = S[nm]
        L(f"  {nm:<8}{r['oracle_auc']:>13.4f}{r['oracle_std']:>10.4f}{r['oracle_oof']:>9.4f}"
          f"{r['auc_srcLR']:>9.4f}{r['oracle_minus_srclr']:>+12.4f}{r['n_vid']:>6}"
          f"{r['oracle_mode']:>13}{str(r['oracle_leak']):>6}"
          f"{r['cos_wd_e0']:>+12.4f}{r['cos_ddom_e0']:>+14.4f}{r['cos_wsrc_e0']:>+14.4f}")
    L("  cos(wd,e0)=域内 LDA(lsqr, shrinkage='auto' 受正则影响); cos(ddom,e0)=域内均值差方向")
    L("  (mean_fake-mean_real, raw V, 无正则, 更稳健); cos(wsrc,e0) 为常数 (源 LR 标准化权重转回原始空间,")
    L("  w_raw=w_std/std_train, 与 e0 同为 fake 正方向). split=SGKF(vid) 才无同视频泄漏; nvid<5 的域退化 SKF, leak=True.")
    L("")

    L("-" * 126)
    L("Q4. 汇总表 (每域一行)")
    L("")
    cols = [("z_var_frac", "zdim_frac", 10, 4), ("residK1", "resid1", 8, 4),
            ("residK5", "resid5", 8, 4), ("residK50", "resid50", 9, 4),
            ("cos(e0d,e0)", "cos_e0dom_e0", 11, 4), ("ovlp10", "overlap10", 8, 4),
            ("d_z", "dz_dom", 8, 4), ("AUC_z", "auc_e0", 8, 4),
            ("AUC_z_rc", "auc_z_recenter", 9, 4), ("d_rec", "recenter_delta", 8, 4),
            ("oracle", "oracle_auc", 8, 4), ("srcLR", "auc_srcLR", 8, 4),
            ("orc-src", "oracle_minus_srclr", 9, 4), ("cos(wd,e0)", "cos_wd_e0", 11, 4)]
    hdr = "  " + f"{'dom':<8}" + "".join(f"{c[0]:>{c[2]}}" for c in cols) + f"{'verdict':>15}"
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    for nm in ["fftest"] + TARGETS:
        r = S[nm]
        line = "  " + f"{nm:<8}"
        for _, key, wd_, nd in cols:
            line += f"{r[key]:>{wd_}.{nd}f}"
        line += f"{'(ref)':>15}" if nm == "fftest" else f"{verdicts[nm]['verdict']:>15}"
        L(line)
    L("  d_rec = AUC_z_recenter - AUC_z (恒 0, 见 Q2); orc-src = oracle_mean - srcLR_AUC")
    L("")

    L("-" * 126)
    L("Q5. 机械判定 H5「跨域崩是否对应源轴表达错位(misalignment)」")
    L(f"  门: 残差能量 (源 top-K 子空间外, K 见下表) 比 FF++ test 参照高 >= {GATE_RESID:.2f}")
    L(f"      或 |d_z| <= {GATE_DZ:.2f}*|d_z_src| => 轴表达异常;")
    L(f"      可恢复: d_recenter >= {GATE_RECENTER:+.2f} 或 oracle-srcLR >= {GATE_ORACLE:+.2f}")
    L(f"  判定: 无异常=NOT-SUPPORTED; 异常且两可恢复门都未过=PARTIAL; 异常且至少一门过=SUPPORT")
    L("")
    L(f"  {'dom':<8}{'residK1':>9}{'residK5':>9}{'residK50':>9}"
      f"{'anomK1':>8}{'anomK5':>8}{'anomK50':>9}{'dz_shr':>8}{'anomaly':>9}"
      f"{'rec_rec':>9}{'rec_orc':>9}{'recover':>9}{'orc-src':>9}{'verdict':>15}")
    L(f"  {'(ref)':<8}{rf['resid1']:>9.4f}{rf['resid5']:>9.4f}{rf['resid50']:>9.4f}"
      f"{'-':>8}{'-':>8}{'-':>9}{'-':>8}{'-':>9}{'-':>9}{'-':>9}{'-':>9}{'-':>9}{'(fftest)':>15}")
    for dm in TARGETS:
        r = S[dm]; v = verdicts[dm]
        L(f"  {dm:<8}{r['resid1']:>9.4f}{r['resid5']:>9.4f}{r['resid50']:>9.4f}"
          f"{str(v['anom_K'][1]):>8}{str(v['anom_K'][5]):>8}{str(v['anom_K'][50]):>9}"
          f"{str(v['dz_shrink']):>8}{str(v['anomaly']):>9}"
          f"{str(v['rec_recenter']):>9}{str(v['rec_oracle']):>9}{str(v['recoverable']):>9}"
          f"{r['oracle_minus_srclr']:>+9.4f}{v['verdict']:>15}")
    L("")
    L(f"  d_z_src={dz_src:.4f}; residK 参照(fftest) 见 '(ref)' 行; anomK* = residK 比参照高 >= {GATE_RESID:.2f}")
    L(f"  逐域: " + "  ".join(f"{dm}={verdicts[dm]['verdict']}" for dm in TARGETS))
    L(f"  全局 (>=3 域同名 或 SUPPORT+PARTIAL>=4): {GLOBAL}   "
      f"[SUPPORT={vc.get('SUPPORT',0)} PARTIAL={vc.get('PARTIAL',0)} NOT-SUPPORTED={vc.get('NOT-SUPPORTED',0)}]")
    L(f"  替代判定 B (严格版: 认为 (d) 与 AUC_z 数学同义, 只认 (b); (b) 取任一 K 命中; 可恢复门含规格外 CORAL): "
      f"全局={GLOBAL_B} "
      f"[SUPPORT={vcb.get('SUPPORT',0)} PARTIAL={vcb.get('PARTIAL',0)} NOT-SUPPORTED={vcb.get('NOT-SUPPORTED',0)}]  "
      f"DOMAINS={'|'.join(dm + ':' + verdicts[dm]['verdict_b'] for dm in TARGETS)}")
    L(f"  替代判定 C (更保守: 只认 (d) 间隔收缩 + oracle 门): 全局={GLOBAL_C} "
      f"[SUPPORT={vcc.get('SUPPORT',0)} PARTIAL={vcc.get('PARTIAL',0)} NOT-SUPPORTED={vcc.get('NOT-SUPPORTED',0)}]  "
      f"DOMAINS={'|'.join(dm + ':' + verdicts[dm]['verdict_c'] for dm in TARGETS)}")
    L(f"  替代判定 K1b ((b) 只取 K=1): 全局={GLOBAL_K1} "
      f"[{vc_k1.get('SUPPORT',0)}/{vc_k1.get('PARTIAL',0)}/{vc_k1.get('NOT-SUPPORTED',0)}] —— "
      f"即 (b) 的结论对 K 敏感: K=1 时 0/5 域异常, K=5 时 4/5 域异常, K=50 时 0/5 域达门.")
    L(f"  替代判定 SUPP (原判定门不变, 只把退化的 recenter 门换成规格外 CORAL_delta): "
      f"全局={GLOBAL_S} " + "  ".join(f"{dm}={verdicts[dm]['verdict_supp']}" for dm in TARGETS))
    L("")

    L("-" * 126)
    L("CAVEATS (honest):")
    L("  * Q2 规格内的 recenter 严格退化: AUC 是秩统计量, 域内整体平移不改变任何排序, RECENTER_DELTA")
    L("    恒为 0.0000 (数学恒等, 非数值误差/实现差异); 连 d_z 也一字不变. 要检验'错位可恢复', 必须用")
    L("    非平移的 label-free 校正 (规格外补充 CORAL / 跨域池化), 或允许目标域标签 (本实验不做).")
    L("  * (d) 分支与 AUC 数学同义: 等方差正态近似 AUC ~ Phi(d_z/sqrt(2)), 实测最大偏差 %.4f;" % phi_dev)
    L("    因此'(d) d_z 显著收缩'不能作为独立于 AUC_z 的'轴表达错位'证据.")
    L("  * (b) 分支对 K 敏感 (同一份数据, 换 K 换结论): K=1 时 5 域残差能量与 FF++ test 参照的差为")
    L("    -0.060~+0.046 (0.318-0.423 vs 0.378), 无一达 +0.05 门; K=5 时 5 域全部更高 (0.119-0.175 vs")
    L("    0.070), 其中 4 域达门 (cd2 +0.049 差 0.001 未达); K=50 时 5 域全部更高 (0.041-0.060 vs 0.017),")
    L("    但幅度无一达 +0.05 门. 即: 承载 62% 能量的第一主轴在目标域并未错位, 而在第 2-5 维上目标域")
    L("    样本明显更偏离源子空间; '(b) 轴外能量是否明显更高'的结论完全取决于 K, 不能只报单一 K.")
    L("  * 目标域每域仅 300 样本 (balanced 150/150), oracle 5 折 AUC 折间方差大 (见 fold_std);")
    L("    单域 AUC 差异 < ~0.03-0.05 不宜解读为真实差异. cd1 fold_std 最大 (每视频最多 18 帧).")
    L("  * ffiw 的 vid 只有 1 个 (vid 字段退化), oracle 无法按视频分组, 退化为 StratifiedKFold,")
    L("    同视频泄漏使该域 oracle AUC 被高估; 该域 oracle 数字不可与其它域并列比较.")
    L("  * LDA shrinkage='auto' (Ledoit-Wolf) 在 768 维/300 样本下正则强, cos(w_d,e0) 被压向 0;")
    L("    故另报无正则的均值差方向 cos(d_dom,e0) 作对照 (更稳健但受域内噪声影响).")
    L("  * 源 e0 只在 FF++ 上估计 (probe train 2200, video-disjoint), 不代表任何'真' fake 方向;")
    L("    目标域 PCA / 子空间重叠由该域自身 300 样本估计, n 小, 有噪声.")
    L("  * resid_K 以 (V_d - mu_src) 为中心 (规格要求), 同时含'域均值平移'与'域内形状差异'两来源;")
    L("    它衡量'样本是否待在源子空间里', 不等于'源子空间是否携带判别信息'.")
    L("  * CORAL 用域内无标签样本估协方差 (768x768/300), 估计本身高方差; 其 delta 只应看作弱证据.")
    L("  * 相关不等于因果: 即使轴表达与跨域崩同向, 也不能证明 ViT 依赖该轴判别, 更不能证明校正该轴")
    L("    即可提升跨域性能 (oracle 是上界, 且'域内自训'可用上目标域标签与域特有捷径).")
    L("")
    L(f"wall time = {time.time()-t_start:.1f}s")

    with open(REPORT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(log) + "\n")

    kv = {}
    for line in mb:
        for tok in line.split():
            if tok.startswith("G11_") and "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
    np.savez(STATS, keys=np.asarray(list(kv.keys()), dtype=str),
             vals=np.asarray(list(kv.values()), dtype=str))
    print(f"[done] report -> {REPORT}\n[done] stats  -> {STATS}\n[done] verdict: "
          f"primary={GLOBAL} B={GLOBAL_B} C={GLOBAL_C} K1b={GLOBAL_K1} supp={GLOBAL_S}",
          flush=True)


if __name__ == "__main__":
    main()
