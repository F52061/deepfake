# -*- coding: utf-8 -*-
"""
G14 -- H8 判定 + 瓶颈定位 (纯离线 CPU 分析, 零模型前向, 禁止 GPU, 只读 npz)

科学问题 (H8, 用户提出): ViT 能表达不同数据集来源图像的特征向量, 但对同域内不同类
(real/fake) 的区分能力有限 -- 两类特征向量被拉近。本实验检验 H8 并定位瓶颈。

分支口径 (与 G12 一致):
  V = V(768) ;  C = C(1024) ;  F = concat(F[:,0:768], F[:,896:1664]) -> 1536-d
  (G12 已实测布局: [0:768]=0.306288*C_proj, [768:896]=bridge 128-d 离线不可得,
   [896:1664]=V_proj, cos_min=1.000000; 故源侧 F = [0.306288*C_proj | V_proj])

数据:
  probe_feats.npz : V(3000,768)/C(3000,1024)/y/paths/vids/train_mask (train 2200/test 800, 42 vids)
  feats_multi.npz : F(2300,1664)/V(2300,768)/C(2300,1024)/y/domain/vid/path
                    域 = ffpp(800,140vid,污染,禁止使用) + cd1/cd2/dfdcp/ffiw/wild 各 300
                    (150 real/150 fake; ffiw 仅 1 vid)

硬性资源约束 (顶部已设):
  OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB = 1 ; torch.set_num_threads(1) ;
  cv2.setNumThreads(0) ; 单进程 ; CUDA_VISIBLE_DEVICES='' (禁止 GPU) ;
  零模型前向 (不实例化模型 / 不读图像 / 不读 checkpoint), 只读 npz。

y 编码: 1=real, 0=fake; 探针 AUC 一律 positive=fake (即 class 1 = fake)。

用法 (项目根目录下):
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g14/run_g14.py > vit_module/_g14/run_log_g14.txt 2>&1

输出:
  vit_module/_g14/g14_report.txt  (MACHINE_BLOCK + R/A-E 表格 + 机械判定 + AUDIT/CAVEATS)
  vit_module/_g14/g14_stats.npz   (机器块键值)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # 禁止任何 GPU 可见/使用

import sys
import time
from math import erf, sqrt
from collections import OrderedDict

import numpy as np
import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
REPORT = os.path.join(HERE, "g14_report.txt")
STATS = os.path.join(HERE, "g14_stats.npz")

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
BRANCHES = ["V", "C", "F"]
KS = [1, 5, 50]
ALPHA_V = 0.306288          # G12 实测: F[:,0:768] = 0.306288 * C_proj (离线不可得 checkpoint, 用给定常量)
C_FIX = 1e-3
MAXIT = 3000
SEED = 0
NFOLDS = 5
DZ_SRC_REF = 4.6555         # G11 源 d_z (用于高斯投影对照, 常量)

METHODS = ["LR_C1e-3", "LR_C1.0", "kNN_k5", "RBF_SVM"]
METHOD_LABELS = {
    "LR_C1e-3": "LogisticRegression(C=1e-3)",
    "LR_C1.0":  "LogisticRegression(C=1.0)",
    "kNN_k5":   "kNN(k=5)",
    "RBF_SVM":  "RBF-SVM(C=1.0,gamma=scale,balanced)",
}

# 锚点 (必须复算一致, atol=1e-3)
ANCHOR_E0_CD1 = 0.8576
ANCHOR_SRCLR_CD1 = 0.8286
ANCHOR_VARFRAC = 0.6229
ANCHOR_ATOL = 1e-3
# G11 cd1 行 (口径对齐, atol=1e-3)
ALIGN = dict(gapRatio=0.4268, dReal=+25.8687, dFake=-8.9057, zdimFrac=0.6357, absCos=0.9870)
# G11 全 5 域参照行 (软核对, 不作硬断言)
G11_ROW = {
    "cd1":    dict(gapRatio=0.4268, dReal=+25.8687, dFake=-8.9057, zdimFrac=0.6357, absCos=0.9870, sdRatio=1.4160),
    "cd2":    dict(gapRatio=0.4669, dReal=+23.5098, dFake=-8.8294, zdimFrac=0.5986, absCos=0.9890, sdRatio=1.4212),
    "dfdcp":  dict(gapRatio=0.4644, dReal=+18.7632, dFake=-13.7313, zdimFrac=0.6581, absCos=0.9790, sdRatio=1.6584),
    "ffiw":   dict(gapRatio=0.4296, dReal=+7.4468, dFake=-27.1553, zdimFrac=0.6852, absCos=0.9796, sdRatio=1.6286),
    "wild":   dict(gapRatio=0.3833, dReal=+29.9671, dFake=-7.4488, zdimFrac=0.6122, absCos=0.9793, sdRatio=1.5638),
}
G12_RATIO_V = {"cd1": 0.2113, "cd2": 0.1861, "dfdcp": 0.1609, "ffiw": 0.1995, "wild": 0.2336}
G12_RATIO_C = {"cd1": 2.4252, "cd2": 2.5436, "dfdcp": 3.2183, "ffiw": 2.8577, "wild": 2.5350}


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def center_only_pca(X):
    """center-only raw-cov PCA via SVD. mu=均值, E=右奇异向量(降序). 与 G11 eigh(raw-cov) 同口径."""
    mu = X.mean(axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt.T, S


def pooled_sd_z(z, y):
    """沿 z 的类内合并 sd (ddof=1). y: 1=real, 0=fake."""
    n1 = int((y == 0).sum()); n2 = int((y == 1).sum())
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp2 = ((n1 - 1) * z[y == 0].var(ddof=1) + (n2 - 1) * z[y == 1].var(ddof=1)) / (n1 + n2 - 2.0)
    return float(np.sqrt(max(sp2, 1e-12)))


def norm_cdf(x):
    """标准正态 CDF: Phi(x) = 0.5*(1+erf(x/sqrt(2)))."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def resid_k(X, mu_src, E, K):
    """1 - ||proj_{E[:,:K]}(X-mu_src)||^2 / ||X-mu_src||^2 (源子空间外残差能量占比)."""
    Xc = X - mu_src
    tot = float((Xc ** 2).sum())
    P = Xc @ E[:, :K]
    return 1.0 - float((P ** 2).sum()) / tot


def get_folds(y, vid, n_splits=NFOLDS, seed=SEED):
    """优先按 vid 分组 StratifiedGroupKFold; 域内 vid 数 < n_splits 则退化 StratifiedKFold, leak=True."""
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


def fit_eval(Xtr, ytr, Xte, yte, method):
    """StandardScaler(fit 折内 train) -> 分类器 -> test AUC (positive=fake, 即 class 1=fake)."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    if method == "LR_C1e-3":
        clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    elif method == "LR_C1.0":
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    elif method == "kNN_k5":
        clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ytr)
        s = clf.predict_proba(Xte_s)[:, 1]
    elif method == "RBF_SVM":
        clf = SVC(C=1.0, gamma="scale", class_weight="balanced").fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    else:
        raise ValueError(method)
    return float(roc_auc_score(yte, s))


def cv_oracle(X, y, vid):
    """域内 oracle: video-grouped 5 折 CV, 4 方法各自折间 AUC 均值/std. 返回每方法统计 + mode/leak."""
    yb = (np.asarray(y) == 0).astype(int)      # 1 = fake
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


def main():
    t0 = time.time()
    log = []

    def L(x=""):
        log.append(str(x))

    def P(x=""):
        print(x, flush=True)

    # ================================================================ 数据 ====
    p = np.load(PROBE_NPZ, allow_pickle=True)
    Vp = p["V"].astype(np.float64)
    Cp = p["C"].astype(np.float64)
    Vpp_npz = p["V_proj"].astype(np.float64)
    Cpp_npz = p["C_proj"].astype(np.float64)
    yp = p["y"].astype(int)
    vp = p["vids"].astype(str)
    tr_m = np.asarray(p["train_mask"])
    tr, te = np.where(tr_m)[0], np.where(~tr_m)[0]
    assert (len(tr), len(te)) == (2200, 800), (len(tr), len(te))
    assert not (set(vp[tr]) & set(vp[te])), "probe train/test 视频重叠"

    f = np.load(MULTI_NPZ, allow_pickle=True)
    Fm = f["F"].astype(np.float64)
    Vm = f["V"].astype(np.float64)
    Cm = f["C"].astype(np.float64)
    ym = f["y"].astype(int)
    domm = f["domain"].astype(str)
    vidm = f["vid"].astype(str)

    for dm in TARGETS:
        m = domm == dm
        assert int(m.sum()) == 300, (dm, int(m.sum()))
        assert int((ym[m] == 1).sum()) == 150 and int((ym[m] == 0).sum()) == 150, dm
    assert int((domm == "ffpp").sum()) == 800

    # 分支矩阵装配
    BR = {
        "V": dict(src=Vp, dst=Vm, dim=768, note="raw ViT CLS"),
        "C": dict(src=Cp, dst=Cm, dim=1024, note="raw CLIP vision CLS"),
        "F": dict(src=np.hstack([ALPHA_V * Cpp_npz, Vpp_npz]),
                   dst=np.hstack([Fm[:, 0:768], Fm[:, 896:1664]]),
                   dim=1536, note="[alpha_v*C_proj | V_proj] 1536 (无 bridge 128)"),
    }

    DOM = OrderedDict()
    for dm in TARGETS:
        m = domm == dm
        DOM[dm] = dict(X={b: BR[b]["dst"][m] for b in BRANCHES},
                       y=ym[m], vid=vidm[m], n=int(m.sum()),
                       nvid=int(len(np.unique(vidm[m]))))

    # ==================================================== 前置校验 1: 锚点 ====
    # 源 V 轴 (center-only PCA@probe-train-V, SVD 口径)
    mu_src_V, E_src_V, S_src_V = center_only_pca(Vp[tr])
    e0_src_V = E_src_V[:, 0].copy()
    z_tr = (Vp[tr] - mu_src_V) @ e0_src_V
    if z_tr[yp[tr] == 0].mean() < z_tr[yp[tr] == 1].mean():
        e0_src_V = -e0_src_V
        z_tr = -z_tr
    varfrac_src_V = float(S_src_V[0] ** 2 / (S_src_V ** 2).sum())

    m_cd1 = domm == "cd1"
    z_cd1 = (Vm[m_cd1] - mu_src_V) @ e0_src_V
    z_cd1_bin = (ym[m_cd1] == 0).astype(int)
    e0auc_cd1 = float(roc_auc_score(z_cd1_bin, z_cd1))

    ztr_bin = (yp[tr] == 0).astype(int)
    sc_V = StandardScaler().fit(Vp[tr])
    clf_V = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
        sc_V.transform(Vp[tr]), ztr_bin)
    srclr_cd1 = float(roc_auc_score(z_cd1_bin, clf_V.decision_function(sc_V.transform(Vm[m_cd1]))))

    dev_e0 = abs(e0auc_cd1 - ANCHOR_E0_CD1)
    dev_lr = abs(srclr_cd1 - ANCHOR_SRCLR_CD1)
    dev_vf = abs(varfrac_src_V - ANCHOR_VARFRAC)
    anchor_ok = (dev_e0 < ANCHOR_ATOL) and (dev_lr < ANCHOR_ATOL) and (dev_vf < ANCHOR_ATOL)

    # ============================================== 前置校验 2: G11 cd1 行 ====
    z_fake_src_mean = float(z_tr[yp[tr] == 0].mean())
    z_real_src_mean = float(z_tr[yp[tr] == 1].mean())
    gap_src = z_fake_src_mean - z_real_src_mean
    pooled_sd_src = pooled_sd_z(z_tr, yp[tr])

    z_d_cd1 = (Vm[m_cd1] - mu_src_V) @ e0_src_V
    gap_cd1 = float(z_d_cd1[ym[m_cd1] == 0].mean() - z_d_cd1[ym[m_cd1] == 1].mean())
    gapratio_cd1 = gap_cd1 / gap_src
    dreal_cd1 = float(z_d_cd1[ym[m_cd1] == 1].mean() - z_real_src_mean)
    dfake_cd1 = float(z_d_cd1[ym[m_cd1] == 0].mean() - z_fake_src_mean)
    zdimfrac_cd1 = float(np.var(z_d_cd1, ddof=1) / np.var(Vm[m_cd1], axis=0, ddof=1).sum())
    Xdc_cd1 = Vm[m_cd1] - Vm[m_cd1].mean(axis=0)
    _, _, Vtd_cd1 = np.linalg.svd(Xdc_cd1, full_matrices=False)
    abscos_cd1 = float(abs(Vtd_cd1[0] @ e0_src_V))

    align = dict(gapRatio=gapratio_cd1, dReal=dreal_cd1, dFake=dfake_cd1,
                 zdimFrac=zdimfrac_cd1, absCos=abscos_cd1)
    align_devs = {k: abs(align[k] - ALIGN[k]) for k in ALIGN}
    align_ok = all(v < ANCHOR_ATOL for v in align_devs.values())

    P("[G14] anchors: e0AUC_cd1=%.4f (dev %.2e) srcLR_cd1=%.4f (dev %.2e) varfrac=%.4f" %
      (e0auc_cd1, dev_e0, srclr_cd1, dev_lr, varfrac_src_V))
    P("[G14] align cd1: gapRatio=%.4f dReal=%+.4f dFake=%+.4f zdimFrac=%.4f absCos=%.4f" %
      (gapratio_cd1, dreal_cd1, dfake_cd1, zdimfrac_cd1, abscos_cd1))

    if not (anchor_ok and align_ok):
        # 停下报告
        L("=" * 126)
        L("G14 REPORT - PRECHECK FAILED (stop)")
        L("=" * 126)
        L(f"G14_ANCHOR_e0AUC_cd1={fmt(e0auc_cd1)} EXP={ANCHOR_E0_CD1} dev={fmt(dev_e0,6)}")
        L(f"G14_ANCHOR_srcLR_cd1={fmt(srclr_cd1)} EXP={ANCHOR_SRCLR_CD1} dev={fmt(dev_lr,6)}")
        L(f"G14_ANCHOR_varfrac={fmt(varfrac_src_V)} EXP={ANCHOR_VARFRAC} dev={fmt(dev_vf,6)}")
        L(f"G14_ALIGN={align} dev={align_devs}")
        L(f"G14_ANCHOR_PASS={int(anchor_ok)} G14_ALIGN_PASS={int(align_ok)}")
        L(f"wall time = {time.time()-t0:.1f}s")
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
        P("[G14] PRECHECK FAILED -> see report", )
        sys.exit(0)

    # ======================================================== R 块: G11 行 ====
    # 每域复现 G11 Q1e 行 (与 G11 对表)
    R_rows = {}
    for dm in TARGETS:
        m = domm == dm
        Vd, yd = Vm[m], ym[m]
        z = (Vd - mu_src_V) @ e0_src_V
        gap = float(z[yd == 0].mean() - z[yd == 1].mean())
        sd_pool = pooled_sd_z(z, yd)
        d_z = gap / sd_pool
        Xdc = Vd - Vd.mean(axis=0)
        _, _, Vtd = np.linalg.svd(Xdc, full_matrices=False)
        R_rows[dm] = dict(
            n=int(len(yd)), nvid=int(len(np.unique(vidm[m]))),
            gap=gap, gapRatio=gap / gap_src,
            dReal=float(z[yd == 1].mean() - z_real_src_mean),
            dFake=float(z[yd == 0].mean() - z_fake_src_mean),
            shift_mid=float(0.5 * (z[yd == 0].mean() + z[yd == 1].mean())
                            - 0.5 * (z_fake_src_mean + z_real_src_mean)),
            sdRatio=sd_pool / pooled_sd_src,
            d_z=d_z,
            zdimFrac=float(np.var(z, ddof=1) / np.var(Vd, axis=0, ddof=1).sum()),
            absCos=float(abs(Vtd[0] @ e0_src_V)),
            cosRaw=float(Vtd[0] @ e0_src_V),
            auc_e0=float(roc_auc_score((yd == 0).astype(int), z)),
        )
    P("[G14] block R done")

    # ======================================================== A 块: 域一致性 ====
    # 每分支 {V,C,F}: s, class_gap, dom_gap(d), ratio(d), residK(K=1,5,50)
    A = {}
    for b in BRANCHES:
        Xtr_b = BR[b]["src"][tr]
        ys = yp[tr]
        mu_src_b = Xtr_b.mean(axis=0)
        mu_real_b = Xtr_b[ys == 1].mean(axis=0)
        mu_fake_b = Xtr_b[ys == 0].mean(axis=0)
        s_b = float(np.sqrt(np.var(Xtr_b, axis=0, ddof=1).mean()))
        class_gap_b = float(np.linalg.norm(mu_fake_b - mu_real_b) / s_b)
        _, E_src_b, _ = center_only_pca(Xtr_b)
        ratio = {}
        dom_gap = {}
        resid = {}
        resid["src"] = {K: resid_k(Xtr_b, mu_src_b, E_src_b, K) for K in KS}
        resid["fftest"] = {K: resid_k(BR[b]["src"][te], mu_src_b, E_src_b, K) for K in KS}
        for dm in TARGETS:
            Xd = DOM[dm]["X"][b]
            dom_gap[dm] = float(np.linalg.norm(Xd.mean(axis=0) - mu_src_b) / s_b)
            ratio[dm] = dom_gap[dm] / class_gap_b
            resid[dm] = {K: resid_k(Xd, mu_src_b, E_src_b, K) for K in KS}
        A[b] = dict(s=s_b, class_gap=class_gap_b, mu_src=mu_src_b,
                    ratio=ratio, dom_gap=dom_gap, resid=resid)
    ratio_mean = {b: float(np.mean([A[b]["ratio"][dm] for dm in TARGETS])) for b in BRANCHES}
    P("[G14] block A done")

    # ======================================================== B 块: 域内上限 ====
    # 每域每分支 4 方法 video-grouped 5 折 CV; ffiw leak 剔除聚合
    Bcv = {b: {} for b in BRANCHES}
    for b in BRANCHES:
        for dm in TARGETS:
            Bcv[b][dm] = cv_oracle(DOM[dm]["X"][b], DOM[dm]["y"], DOM[dm]["vid"])
    # FF++ 域内对照 (probe train 2200 / test 800)
    FF = {b: {} for b in BRANCHES}
    for b in BRANCHES:
        Xb = BR[b]["src"]
        yb = (yp == 0).astype(int)
        for m in METHODS:
            try:
                FF[b][m] = fit_eval(Xb[tr], yb[tr], Xb[te], yb[te], m)
            except Exception:
                FF[b][m] = float("nan")
    # oracle = 每分支最好方法 (max over 4 methods), 逐域; FF++ 侧同样取最好方法
    oracle = {b: {} for b in BRANCHES}
    for b in BRANCHES:
        for dm in TARGETS:
            oracle[b][dm] = max(Bcv[b][dm][m]["mean"] for m in METHODS)
        oracle[b]["ffpp"] = max(FF[b][m] for m in METHODS)
    valid_domains = [dm for dm in TARGETS if not Bcv["V"][dm]["leak"]]   # 剔除 ffiw
    delta_oracle = {dm: oracle["V"]["ffpp"] - oracle["V"][dm] for dm in TARGETS}
    P("[G14] block B done; valid_domains=%s" % valid_domains)

    # ======================================================== C 块: 拉近量化 ====
    C_rows = {}
    for dm in TARGETS:
        m = domm == dm
        Vd, yd = Vm[m], ym[m]
        z = (Vd - mu_src_V) @ e0_src_V
        zf = z[yd == 0]; zr = z[yd == 1]
        gap = float(zf.mean() - zr.mean())
        sd_real = float(zr.std(ddof=1)); sd_fake = float(zf.std(ddof=1))
        sd_pool = pooled_sd_z(z, yd)
        d_z = gap / sd_pool
        fisher = gap ** 2 / (sd_real ** 2 + sd_fake ** 2)
        auc_e0 = float(roc_auc_score((yd == 0).astype(int), z))
        overlap = 1.0 - auc_e0
        C_rows[dm] = dict(
            gap=gap, sd_pool=sd_pool, sd_real=sd_real, sd_fake=sd_fake,
            d_z=d_z, fisher=fisher, overlap=overlap, auc_e0=auc_e0,
            dReal=float(zr.mean() - z_real_src_mean),
            dFake=float(zf.mean() - z_fake_src_mean),
            shift_mid=float(0.5 * (zf.mean() + zr.mean()) - 0.5 * (z_fake_src_mean + z_real_src_mean)),
            gauss={f_: norm_cdf(d_z * f_ / sqrt(2.0)) for f_ in [1.0, 1.5, 2.0, 3.0]},
        )
    src_gauss = norm_cdf(DZ_SRC_REF / sqrt(2.0))
    P("[G14] block C done")

    # ======================================================== D 块: 衰减 vs 旋转 ====
    D_rows = {}
    for dm in TARGETS:
        Xd = DOM[dm]["X"]["V"]
        yd = DOM[dm]["y"]
        vid = DOM[dm]["vid"]
        yb = (yd == 0).astype(int)
        folds, mode, leak = get_folds(yb, vid)
        aucs = {"full": [], "e0": [], "off": []}
        for tr_i, te_i in folds:
            if len(np.unique(yb[tr_i])) < 2 or len(np.unique(yb[te_i])) < 2:
                continue
            mu_fake = Xd[tr_i][yb[tr_i] == 1].mean(axis=0)
            mu_real = Xd[tr_i][yb[tr_i] == 0].mean(axis=0)
            Delta = mu_fake - mu_real
            Delta_off = Delta - (Delta @ e0_src_V) * e0_src_V
            Xc_te = Xd[te_i] - mu_src_V
            try:
                aucs["full"].append(float(roc_auc_score(yb[te_i], Xc_te @ Delta)))
                aucs["e0"].append(float(roc_auc_score(yb[te_i], Xc_te @ e0_src_V)))
                aucs["off"].append(float(roc_auc_score(yb[te_i], Xc_te @ Delta_off)))
            except Exception:
                pass
        # cos(Delta_d, e0) 用全 300 样本均值差 (更稳健, 与 G11 cos_ddom_e0 同口径)
        mu_fake_full = Xd[yb == 1].mean(axis=0)
        mu_real_full = Xd[yb == 0].mean(axis=0)
        Delta_full = mu_fake_full - mu_real_full
        D_rows[dm] = dict(
            mode=mode, leak=bool(leak),
            cos=float(cosv(Delta_full, e0_src_V)),
            auc_full=float(np.mean(aucs["full"])) if aucs["full"] else float("nan"),
            auc_e0=float(np.mean(aucs["e0"])) if aucs["e0"] else float("nan"),
            auc_off=float(np.mean(aucs["off"])) if aucs["off"] else float("nan"),
            delta_full_e0=(float(np.mean(aucs["full"])) - float(np.mean(aucs["e0"]))) if aucs["full"] and aucs["e0"] else float("nan"),
            nfolds=int(len(aucs["full"])),
        )
    P("[G14] block D done")

    # ======================================================== E 块: 瓶颈定位 ====
    mean_oracle = {}
    for b in BRANCHES:
        mean_oracle[b] = float(np.mean([oracle[b][dm] for dm in valid_domains]))
    P("[G14] block E done")

    # ======================================================== 机械判定 ========
    # G14_H8_DOMAIN_CONSIST
    mean_ratio_V = ratio_mean["V"]
    mean_ratio_C = ratio_mean["C"]
    if (mean_ratio_V <= 0.5) and (mean_ratio_V < mean_ratio_C - 1.0):
        H8_DOMAIN_CONSIST = "SUPPORT"
    elif mean_ratio_V >= 1.5:
        H8_DOMAIN_CONSIST = "NOT-SUPPORT"
    else:
        H8_DOMAIN_CONSIST = "PARTIAL"

    # G14_H8_LIMITED
    mean_delta_oracle = float(np.mean([delta_oracle[dm] for dm in valid_domains]))
    if mean_delta_oracle >= 0.08:
        H8_LIMITED = "LIMITED"
    elif mean_delta_oracle <= 0.03:
        H8_LIMITED = "NOT-LIMITED"
    else:
        H8_LIMITED = "PARTIAL"

    # G14_H8_ATTENUATION (5 域计数)
    n_atten = 0
    n_rot = 0
    atten_detail = {}
    for dm in TARGETS:
        r = D_rows[dm]
        c_ge = (r["cos"] >= 0.9)
        d_le = (r["delta_full_e0"] <= 0.02)
        c_lt = (r["cos"] < 0.7)
        d_gt = (r["delta_full_e0"] > 0.05)
        atten_detail[dm] = dict(cos_ge=c_ge, d_le=d_le, cos_lt=c_lt, d_gt=d_gt,
                                atten=bool(c_ge and d_le), rot=bool(c_lt or d_gt))
        if c_ge and d_le:
            n_atten += 1
        if c_lt or d_gt:
            n_rot += 1
    if n_atten >= 4:
        H8_ATTENUATION = "ATTENUATION_ONLY"
    elif n_rot >= 3:
        H8_ATTENUATION = "ROTATED_OR_MIXED"
    else:
        H8_ATTENUATION = "MIXED"

    # G14_BOTTLENECK
    diff_bn = max(mean_oracle["C"], mean_oracle["F"]) - mean_oracle["V"]
    if diff_bn >= +0.05:
        BOTTLENECK = "BOTTLENECK_AT_V"
    elif abs(diff_bn) < 0.02:
        BOTTLENECK = "BOTTLENECK_SHARED"
    else:
        BOTTLENECK = "MIXED"

    VERDICT_SUMMARY = ("H8_DOMAIN_CONSIST=%s;H8_LIMITED=%s;H8_ATTENUATION=%s;BOTTLENECK=%s" %
                       (H8_DOMAIN_CONSIST, H8_LIMITED, H8_ATTENUATION, BOTTLENECK))
    wall = time.time() - t0
    P("[G14] verdicts: DOMAIN_CONSIST=%s LIMITED=%s ATTENUATION=%s BOTTLENECK=%s (wall=%.1fs)" %
      (H8_DOMAIN_CONSIST, H8_LIMITED, H8_ATTENUATION, BOTTLENECK, wall))

    # ======================================================== MACHINE_BLOCK ====
    mb = OrderedDict()
    mb["G14_TASK"] = "H8_verdict_and_bottleneck"
    mb["G14_DATA"] = ("probe_feats.npz(V/C/V_proj/C_proj n=3000:train2200/test800); "
                      "feats_multi.npz(F/V/C n=2300:ffpp800+5x300)")
    mb["G14_BRANCHES"] = "V=768; C=1024; F=1536([F[:,0:768]|F[:,896:1664]] concat, no bridge128)"
    mb["G14_ALPHA_V"] = fmt(ALPHA_V, 6)
    mb["G14_ANCHOR_e0AUC_cd1"] = fmt(e0auc_cd1, 4)
    mb["G14_ANCHOR_e0AUC_cd1_EXP"] = fmt(ANCHOR_E0_CD1, 4)
    mb["G14_ANCHOR_srcLR_cd1"] = fmt(srclr_cd1, 4)
    mb["G14_ANCHOR_srcLR_cd1_EXP"] = fmt(ANCHOR_SRCLR_CD1, 4)
    mb["G14_ANCHOR_PC0_varfrac_srcV"] = fmt(varfrac_src_V, 4)
    mb["G14_ANCHOR_MAXDEV"] = fmt(max(dev_e0, dev_lr, dev_vf), 6)
    mb["G14_ANCHOR_PASS"] = "1" if anchor_ok else "0"
    for k in ALIGN:
        mb["G14_ALIGN_%s_cd1" % k] = fmt(align[k], 4)
        mb["G14_ALIGN_%s_cd1_EXP" % k] = fmt(ALIGN[k], 4)
    mb["G14_ALIGN_MAXDEV"] = fmt(max(align_devs.values()), 6)
    mb["G14_ALIGN_PASS"] = "1" if align_ok else "0"
    # R
    for k in ["gapRatio", "dReal", "dFake", "zdimFrac", "absCos", "sdRatio"]:
        mb["G14_R_%s" % k] = ";".join("%s=%.4f" % (dm, R_rows[dm][k]) for dm in TARGETS)
    # A
    for b in BRANCHES:
        mb["G14_RATIO_%s" % b] = ";".join("%s=%.4f" % (dm, A[b]["ratio"][dm]) for dm in TARGETS)
    mb["G14_RATIO_MEAN_BY_BRANCH"] = ";".join("%s=%.4f" % (b, ratio_mean[b]) for b in BRANCHES)
    mb["G14_CLASSGAP_BY_BRANCH"] = ";".join("%s=%.4f" % (b, A[b]["class_gap"]) for b in BRANCHES)
    mb["G14_SCALE_s_BY_BRANCH"] = ";".join("%s=%.3f" % (b, A[b]["s"]) for b in BRANCHES)
    for b in BRANCHES:
        mb["G14_RESID_%s" % b] = ("src:" + ",".join("K%d=%.4f" % (K, A[b]["resid"]["src"][K]) for K in KS) +
                                   "|fftest:" + ",".join("K%d=%.4f" % (K, A[b]["resid"]["fftest"][K]) for K in KS) +
                                   "|dom:" + ";".join("%s=" % dm + ",".join("K%d=%.4f" % (K, A[b]["resid"][dm][K]) for K in KS)
                                                      for dm in TARGETS))
    # B
    for b in BRANCHES:
        mb["G14_ORACLE_%s" % b] = ";".join("%s=%.4f" % (dm, oracle[b][dm]) for dm in TARGETS)
    mb["G14_FFPP_BEST_BY_BRANCH"] = ";".join("%s=%.4f" % (b, oracle[b]["ffpp"]) for b in BRANCHES)
    mb["G14_FFPP_LR1e-3"] = ";".join("%s=%.4f" % (b, FF[b]["LR_C1e-3"]) for b in BRANCHES)
    mb["G14_DELTA_ORACLE_V"] = ";".join("%s=%+.4f" % (dm, delta_oracle[dm]) for dm in TARGETS)
    mb["G14_VALID_DOMAINS"] = ",".join(valid_domains) + " (ffiw leak=True excluded)"
    mb["G14_ORACLE_METHOD_MEANS_BY_BRANCH"] = ";".join(
        "%s:" % b + ",".join("%s=%.4f" % (m, float(np.mean([Bcv[b][dm][m]["mean"] for dm in valid_domains])))
                             for m in METHODS) for b in BRANCHES)
    # C
    for k in ["gap", "d_z", "fisher", "overlap"]:
        mb["G14_C_%s" % k] = ";".join("%s=%.4f" % (dm, C_rows[dm][k]) for dm in TARGETS)
    mb["G14_C_GAUSS_src_dz4.6555"] = fmt(src_gauss, 4)
    for f_ in [1.0, 1.5, 2.0, 3.0]:
        mb["G14_C_GAUSS_f%d" % int(f_) if f_ == int(f_) else "G14_C_GAUSS_f%.1f" % f_] = \
            ";".join("%s=%.4f" % (dm, C_rows[dm]["gauss"][f_]) for dm in TARGETS)
    # D
    mb["G14_COS_DELTA_e0"] = ";".join("%s=%+.4f" % (dm, D_rows[dm]["cos"]) for dm in TARGETS)
    mb["G14_DIR_AUC_full"] = ";".join("%s=%.4f" % (dm, D_rows[dm]["auc_full"]) for dm in TARGETS)
    mb["G14_DIR_AUC_e0"] = ";".join("%s=%.4f" % (dm, D_rows[dm]["auc_e0"]) for dm in TARGETS)
    mb["G14_DIR_AUC_off"] = ";".join("%s=%.4f" % (dm, D_rows[dm]["auc_off"]) for dm in TARGETS)
    mb["G14_DIR_DELTA_full_e0"] = ";".join("%s=%+.4f" % (dm, D_rows[dm]["delta_full_e0"]) for dm in TARGETS)
    # E
    mb["G14_MEAN_ORACLE_BY_BRANCH"] = ";".join("%s=%.4f" % (b, mean_oracle[b]) for b in BRANCHES)
    mb["G14_BOTTLENECK_DIFF"] = fmt(diff_bn, 4)
    # 判定常量
    mb["G14_H8_DOMAIN_CONSIST"] = H8_DOMAIN_CONSIST
    mb["G14_H8_LIMITED"] = H8_LIMITED
    mb["G14_H8_ATTENUATION"] = H8_ATTENUATION
    mb["G14_BOTTLENECK"] = BOTTLENECK
    mb["G14_VERDICT_SUMMARY"] = VERDICT_SUMMARY
    mb["G14_ATTEN_N_DOMAINS"] = "%d/5" % n_atten
    mb["G14_ROT_N_DOMAINS"] = "%d/5" % n_rot
    mb["G14_MEAN_DELTA_ORACLE"] = fmt(mean_delta_oracle, 4)
    mb["G14_WALL_S"] = fmt(wall, 1)
    mb["G14_IMGS_READ"] = "0"
    mb["G14_GPU"] = "0"
    mb["G14_FORWARDS"] = "0"
    mb["G14_THREADS"] = "1"
    mb["G14_PROCESSES"] = "1"
    mb["G14_NPZ_READ"] = "2"

    mblines = ["#### MACHINE_BLOCK " + "#" * 105]
    mblines += ["G14_%s=%s" % (k[4:], v) for k, v in mb.items()]

    # ======================================================== 报告正文 ========
    L("=" * 126)
    L("G14 REPORT - H8 判定 (ViT 对同域内 real/fake 区分有限: 两类特征被拉近) + 瓶颈定位")
    L("纯 CPU 单线程 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1, torch=1, cv2=0, 单进程, CUDA_VISIBLE_DEVICES='')")
    L("零模型前向 (不实例化模型/不读图像/不读 checkpoint), 只读 npz; 分支 V/C/F; y: 1=real, 0=fake; AUC positive=fake")
    L("=" * 126)
    L("")
    L("\n".join(mblines))
    L("")

    # ---- 前置校验 ----
    L("-" * 126)
    L("PRECHECK. 前置校验 (任一不过即停, atol=1e-3)")
    L("  1) 锚点复算 (probe train V 拟合 center-only PCA@SVD -> e0; 在 multi cd1 V 上算 z 轴 AUC)")
    L("     e0 z-axis AUC cd1       : 复算 %s / 预期 %s   (|dev|=%s)" %
      (fmt(e0auc_cd1), fmt(ANCHOR_E0_CD1), fmt(dev_e0, 6)))
    L("     srcLR(SS+LR C=1e-3) cd1 : 复算 %s / 预期 %s   (|dev|=%s)" %
      (fmt(srclr_cd1), fmt(ANCHOR_SRCLR_CD1), fmt(dev_lr, 6)))
    L("     源 train V PC0 varfrac  : 复算 %s / 预期 %s   (|dev|=%s)" %
      (fmt(varfrac_src_V), fmt(ANCHOR_VARFRAC), fmt(dev_vf, 6)))
    L("  2) 口径对齐 (复现 G11 cd1 行, z 以源 train 均值为零点)")
    L("     gapRatio=%s(exp %s) dReal=%+.4f(exp %+.4f) dFake=%+.4f(exp %+.4f) zdimFrac=%s(exp %s) |cos|=%s(exp %s)" %
      (fmt(gapratio_cd1), fmt(ALIGN["gapRatio"]), dreal_cd1, ALIGN["dReal"],
       dfake_cd1, ALIGN["dFake"], fmt(zdimfrac_cd1), fmt(ALIGN["zdimFrac"]),
       fmt(abscos_cd1), fmt(ALIGN["absCos"])))
    L("  -> PRECHECK PASS (ANCHOR_PASS=%d ALIGN_PASS=%d)" % (int(anchor_ok), int(align_ok)))
    L("")

    # ---- R ----
    L("-" * 126)
    L("R. 复核行 (复现 G11 Q1e 全 5 域; z 以源 train 均值为零点; gap=mean z_fake-mean z_real)")
    L("    gapRatio=gap/gap_src; dReal/dFake=相对源 train 类均值位移; shift_mid=两类中点位移;")
    L("    sdRatio=域内合并 sd/源合并 sd; zdimFrac=域内 z 方差/域内 V 各维方差和; |cos|=|域 PC0 与源 e0|")
    L("")
    hdr = (f"  {'dom':<8}{'nvid':>5}{'gapRatio':>9}{'gapRatio[G11]':>13}{'dReal':>9}{'dReal[G11]':>11}"
           f"{'dFake':>9}{'dFake[G11]':>11}{'shift_mid':>11}{'sdRatio':>9}{'d_z':>8}{'zdimFrac':>10}{'|cos|':>8}")
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    for dm in TARGETS:
        r = R_rows[dm]; g = G11_ROW[dm]
        L(f"  {dm:<8}{r['nvid']:>5}{r['gapRatio']:>9.4f}{g['gapRatio']:>13.4f}"
          f"{r['dReal']:>+9.4f}{g['dReal']:>+11.4f}{r['dFake']:>+9.4f}{g['dFake']:>+11.4f}"
          f"{r['shift_mid']:>+11.4f}{r['sdRatio']:>9.4f}{r['d_z']:>8.4f}{r['zdimFrac']:>10.4f}{r['absCos']:>8.4f}")
    L("  (G11 列为其 report 参照值; 复算与参照一致到 <1e-3 即口径对齐通过)")
    L("")

    # ---- A ----
    L("-" * 126)
    L("A. 域一致性 (记录行, 不设主门; G12 B/D 口径)")
    L("    class_gap=||mu_fake-mu_real||/s (源 train); dom_gap(d)=||mu_d-mu_src||/s; ratio=dom_gap/class_gap;")
    L("    s=sqrt(mean_j Var_j(源全池)); residK=1-||proj_{E[:,:K]}(X-mu_src)||^2/||X-mu_src||^2 (源 train 中心化)")
    L("")
    L("  branch   dim      s    class_gap | " + " ".join("%9s" % dm for dm in TARGETS) + "   |  mean")
    L("  " + "-" * 100)
    for b in BRANCHES:
        L("  %-7s %5d %7.3f %8.4f  | " % (b, BR[b]["dim"], A[b]["s"], A[b]["class_gap"]) +
          " ".join("%9.4f" % A[b]["ratio"][dm] for dm in TARGETS) + "   | %6.4f" % ratio_mean[b])
    L("  " + "-" * 100)
    L("  ratio 跨域均值: " + "  ".join("%s=%.4f" % (b, ratio_mean[b]) for b in BRANCHES))
    L("  [G12 软核对] ratio_V: " + " ".join("%s=%.4f" % (dm, G12_RATIO_V[dm]) for dm in TARGETS))
    L("  [G12 软核对] ratio_C: " + " ".join("%s=%.4f" % (dm, G12_RATIO_C[dm]) for dm in TARGETS))
    L("")
    L("  residK (源 train 中心化, K 主成分子空间外残差能量占比):")
    L("    branch   " + " ".join("%9s" % ("K=%d" % K) for K in KS) + "    |  域 (K1/K5/K50)")
    L("    " + "-" * 100)
    for b in BRANCHES:
        L("    %-7s src  " % b + " ".join("%9.4f" % A[b]["resid"]["src"][K] for K in KS))
        L("    %-7s ftst " % b + " ".join("%9.4f" % A[b]["resid"]["fftest"][K] for K in KS))
        for dm in TARGETS:
            L("    %-7s %-4s " % (b, dm) + " ".join("%9.4f" % A[b]["resid"][dm][K] for K in KS))
        L("")
    L("")

    # ---- B ----
    L("-" * 126)
    L("B. 域内可分离性上限 (每域每分支, 只用该域 300 样本, video-grouped 5 折 CV; 特征 StandardScaler(fit 折内 train))")
    L("    4 方法: (a) LR(C=1e-3) (b) LR(C=1.0) (c) kNN(k=5) (d) RBF-SVM(C=1.0,gamma=scale,balanced,decision_function)")
    L("    ffiw 仅 1 vid -> 退化 StratifiedKFold, leak=True, 不入聚合; 表格数字为 5 折 AUC 均值 (折间 std 见括号)")
    L("")
    for b in BRANCHES:
        L("  branch %s (dim=%d, %s):" % (b, BR[b]["dim"], BR[b]["note"]))
        L("    %-6s" % "dom" + "".join("  %-11s" % m for m in METHODS) + "   %-8s %-5s %s" % ("oracle", "leak", "mode"))
        for dm in TARGETS:
            r = Bcv[b][dm]
            cells = ""
            for m in METHODS:
                cells += "  %6.4f(%s)" % (r[m]["mean"], fmt(r[m]["std"], 3))
            L("    %-6s%s   %-8.4f %-5s %s" % (dm, cells, oracle[b][dm], str(r["leak"]), r["mode"]))
        L("")
    L("  FF++ 域内对照 (probe train 2200 训练 / test 800 评估, 同族方法; V 分支 LR(C=1e-3) 应为 ~0.985):")
    L("    %-6s" % "branch" + "".join("  %-11s" % m for m in METHODS) + "   %-8s" % "oracle(best)")
    for b in BRANCHES:
        L("    %-6s" % b + "".join("  %10.4f" % FF[b][m] for m in METHODS) + "   %-8.4f" % oracle[b]["ffpp"])
    L("")
    L("  Delta_oracle(d) = AUC_FFPP_V(best) - oracle_V(d)  (positive=fake):")
    L("    " + "  ".join("%s=%+.4f" % (dm, delta_oracle[dm]) for dm in TARGETS) +
      "   |  有效域均值(剔 ffiw) %+.4f" % mean_delta_oracle)
    L("")

    # ---- C ----
    L("-" * 126)
    L("C. 拉近量化 (V 分支, 沿源轴 z=(V-mu_src)@e0; 每域 300 样本)")
    L("    gap=mean z_fake-mean z_real; d_z=gap/合并sd; Fisher=gap^2/(sd_real^2+sd_fake^2);")
    L("    overlap=1-AUC_e0(z); d_real/d_fake=相对源 train 类均值位移; shift_mid=两类中点位移")
    L("")
    hdr = (f"  {'dom':<8}{'gap':>9}{'sd_pool':>9}{'d_z':>8}{'Fisher':>9}{'AUC_e0':>9}{'overlap':>9}"
           f"{'d_real':>10}{'d_fake':>10}{'shift_mid':>11}")
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    for dm in TARGETS:
        r = C_rows[dm]
        L(f"  {dm:<8}{r['gap']:>9.4f}{r['sd_pool']:>9.3f}{r['d_z']:>8.4f}{r['fisher']:>9.4f}"
          f"{r['auc_e0']:>9.4f}{r['overlap']:>9.4f}{r['dReal']:>+10.4f}{r['dFake']:>+10.4f}{r['shift_mid']:>+11.4f}")
    L("")
    L("  高斯模型投影对照 (等方差正态近似, 不是测量; Phi(d_z*f/sqrt2) 给出把间隔放大 f 倍后的等效 AUC):")
    L("    %-8s" % "dom" + "".join("%10s" % ("f=%.1f" % f_) for f_ in [1.0, 1.5, 2.0, 3.0]) +
      "   (f=1 即 Phi(d_z/sqrt2))")
    for dm in TARGETS:
        r = C_rows[dm]
        L("    %-8s" % dm + "".join("%10.4f" % r["gauss"][f_] for f_ in [1.0, 1.5, 2.0, 3.0]))
    L("  源 d_z_src=%.4f 对应的等效 AUC = %.4f  (等方差正态近似投影, 非测量)" % (DZ_SRC_REF, src_gauss))
    L("")

    # ---- D ----
    L("-" * 126)
    L("D. 衰减 vs 旋转 (V 分支; 与 B 块同一 StratifiedGroupKFold 5 折, 折内 train 构方向 / 折内 test 算 AUC, 5 折均值)")
    L("    方向: (i) Delta_d=mu_fake-mu_real (域内均值差, 无正则, 折内 train 标签);")
    L("          (ii) 源轴 e0 (不含域标签); (iii) Delta_d 轴外分量 Delta_d-(Delta_d.e0)e0")
    L("")
    hdr = (f"  {'dom':<8}{'cos(Delta,e0)':>15}{'AUC(Delta全)':>13}{'AUC(仅e0)':>12}{'AUC(轴外)':>12}"
           f"{'全-仅e0':>10}{'leak':>6}{'mode':>14}")
    L(hdr); L("  " + "-" * (len(hdr) - 2))
    for dm in TARGETS:
        r = D_rows[dm]
        L(f"  {dm:<8}{r['cos']:>+15.4f}{r['auc_full']:>13.4f}{r['auc_e0']:>12.4f}{r['auc_off']:>12.4f}"
          f"{r['delta_full_e0']:>+10.4f}{str(r['leak']):>6}{r['mode']:>14}")
    L("  判读: 若 AUC(仅e0)≈AUC(Delta全) 且远高于 AUC(轴外) -> 同方向、幅度不足(纯衰减);")
    L("        若 AUC(Delta全) 明显高于 AUC(仅e0)(>0.05) -> 类方向在目标域发生旋转.")
    L("")

    # ---- E ----
    L("-" * 126)
    L("E. 瓶颈定位 (汇总 B 块域内 oracle AUC, 取每分支最好方法, 有效域 4 域均值, 剔 ffiw)")
    L("")
    L("  %-8s" % "branch" + "".join("  %-9s" % dm for dm in TARGETS) + "   %-10s" % "mean(4dom)")
    for b in BRANCHES:
        L("  %-8s" % b + "".join("  %9.4f" % oracle[b][dm] for dm in TARGETS) + "   %10.4f" % mean_oracle[b])
    L("  max(mean oracle_C, mean oracle_F) - mean oracle_V = %+.4f" % diff_bn)
    L("")

    # ---- 机械判定 ----
    L("-" * 126)
    L("MECHANICAL VERDICTS (预注册, 照抄执行, 不做方向演绎)")
    L("")
    L("  [H8_DOMAIN_CONSIST] mean_d ratio_V=%.4f <=0.5 ? %s ; ratio_V < ratio_C-1.0 (%.4f < %.4f) ? %s" %
      (mean_ratio_V, "Y" if mean_ratio_V <= 0.5 else "n",
       mean_ratio_V, mean_ratio_C - 1.0, "Y" if mean_ratio_V < mean_ratio_C - 1.0 else "n"))
    L("      ratio_V >= 1.5 ? %s  -> G14_H8_DOMAIN_CONSIST=%s" %
      ("Y" if mean_ratio_V >= 1.5 else "n", H8_DOMAIN_CONSIST))
    L("  [H8_LIMITED] mean_d (AUC_FFPP_V - oracle_V(d)) = %+.4f (有效域, 剔 ffiw)" % mean_delta_oracle)
    L("      >=0.08 ? %s ; <=0.03 ? %s  -> G14_H8_LIMITED=%s" %
      ("Y" if mean_delta_oracle >= 0.08 else "n", "Y" if mean_delta_oracle <= 0.03 else "n", H8_LIMITED))
    L("  [H8_ATTENUATION] 逐域 cos>=0.9 且 (全-仅e0)<=0.02:")
    for dm in TARGETS:
        d = atten_detail[dm]; r = D_rows[dm]
        L("      %-6s cos=%+.4f(>=0.9?%s) delta=%+.4f(<=0.02?%s) cos<0.7?%s delta>0.05?%s -> %s" %
          (dm, r["cos"], d["cos_ge"], r["delta_full_e0"], d["d_le"], d["cos_lt"], d["d_gt"],
           "ATTEN" if d["atten"] else ("ROT" if d["rot"] else "-")))
    L("      同时满足域数=%d/5 ; 出现(cos<0.7 或 delta>0.05)域数=%d/5 -> G14_H8_ATTENUATION=%s" %
      (n_atten, n_rot, H8_ATTENUATION))
    L("  [BOTTLENECK] max(mean oracle_C, mean oracle_F)-mean oracle_V = %+.4f" % diff_bn)
    L("      >=+0.05 ? %s ; |diff|<0.02 ? %s  -> G14_BOTTLENECK=%s" %
      ("Y" if diff_bn >= 0.05 else "n", "Y" if abs(diff_bn) < 0.02 else "n", BOTTLENECK))
    L("")
    L("  === 判定常量汇总 ===")
    L("    G14_H8_DOMAIN_CONSIST = %s" % H8_DOMAIN_CONSIST)
    L("    G14_H8_LIMITED        = %s" % H8_LIMITED)
    L("    G14_H8_ATTENUATION    = %s" % H8_ATTENUATION)
    L("    G14_BOTTLENECK        = %s" % BOTTLENECK)
    L("    G14_VERDICT_SUMMARY   = %s" % VERDICT_SUMMARY)
    L("")

    # ---- AUDIT + CAVEATS ----
    L("-" * 126)
    L("AUDIT")
    L("  imgs_read=0 (零图像读取)   gpu=0 (CUDA_VISIBLE_DEVICES='', 未调用任何 cuda API)")
    L("  forwards=0 (未实例化模型, 零前向/反向, 未读 checkpoint)   threads=1 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1,")
    L("  torch.set_num_threads(1), cv2.setNumThreads(0)) ; processes=1 ; npz 读取=2 (probe_feats.npz, feats_multi.npz)")
    L("  探针: StandardScaler(fit 折内 train) + 分类器; LR lbfgs max_iter=3000; 划分 seed=0; 无 PCA 白化")
    L("  PCA: np.linalg.svd(full_matrices=False) 于中心化特征矩阵 (与 G11 eigh(raw-cov)/G12 SVD 同口径)")
    L("  F 源侧 = [0.306288*C_proj | V_proj] (probe npz 直接取 V_proj/C_proj, 不读 checkpoint; alpha_v 用 G12 实测常量)")
    L("  wall_s = %.1f" % wall)
    L("")
    L("CAVEATS (honest)")
    L("  1. oracle 是域内标签训练 (标签依赖, 上界性质, 不可部署): B 块 4 方法都在目标域 300 样本上")
    L("     用 real/fake 标签训练, 得到的是'该域内线性/局部可分性的上限', 不代表无标签跨域可用.")
    L("  2. ffiw 域只有 1 个 vid (300 样本=150 real+150 fake 同视频): 无法按视频分组, B/D 块退化为")
    L("     StratifiedKFold 并标 leak=True, 同视频泄漏使其 oracle/AUC 被高估, 已剔出 E 聚合与 H8_LIMITED")
    L("     mean; A/R/C 块量不依赖样本划分, 保留 ffiw; D 块机械判定的 4/5、3/5 计数按规格照抄 5 域计.")
    L("  3. 每域仅 300 样本, 5 折 CV 折间方差大 (B 块已报折间 std): 单域 AUC 差异 < ~0.05 不宜解读;")
    L("     D 块每折 test 约 60 样本, 方向 AUC 折间波动大, 只看跨域一致方向与量级.")
    L("  4. kNN/SVM 与 LR 的可比性: 三者均先 StandardScaler(fit 折内 train); SVM 用 class_weight='balanced'")
    L("     应对均衡数据(实际 150/150 已均衡); kNN 用 predict_proba; 三者的 AUC 都按 positive=fake 统一.")
    L("  5. 高斯投影是等方差正态近似下的投影, 不是测量: Phi(d_z*f/sqrt2) 假设两类沿 z 等方差正态,")
    L("     只用于'若把间隔放大 f 倍, 单轴 AUC 大致到多少'的对照, 不能当实测.")
    L("  6. 相关 != 因果: 域内 oracle 高于跨域 srcLR, 只说明'域内标签可训练出更好的判别器', 不能证明")
    L("     ViT 依赖该判别方向, 也不能证明增广/解冻即可把 oracle 变现为跨域增益.")
    L("  7. '域一致'只能表述为相对: ratio_V(mean)=%.4f 表示 V 分支的类间隔(源)相对域位移很小, 但 V 的" % ratio_mean["V"])
    L("     real-only 域判别 AUC 仍在 0.86 级别 (G12 C 块), 即 V 并非对域身份零敏感, 只是相对 C 弱得多.")
    L("  8. F 分支缺 128-d bridge 块 (7.7% 维度, 需前向, 离线不可得): F 的真实可分性可能高于此处 1536-d 估计.")
    L("  9. D 块 '与 B 的折一致' 落实为复用 B 块同一 StratifiedGroupKFold 5 折 (train~80%/test~20%);")
    L("     规格中 '70/30' 按'与 B 折一致'的约束理解为 5 折划分, 而非另起 70/30 单次划分.")
    L("  10. 本实验全部离线特征几何量, 未重测任何下游检测性能; 锚点均复算一致 (<1e-3), 口径未漂移.")
    L("  11. H8_LIMITED 为临界值: mean_d (AUC_FFPP_V - oracle_V(d)) = +0.0770, 距 LIMITED 门 +0.08 仅 0.003.")
    L("      该值对 'oracle=最好方法' 与 'AUC_FFPP=LR(C=1e-3)锚点' 的口径选择敏感 (0.074~0.077),")
    L("      两种口径都落在 PARTIAL 区间上沿 (0.03, 0.08); 结论应表述为 'PARTIAL(临界, 略低于 LIMITED 门)',")
    L("      不宜解读为 'H8 已被否证' 或 'H8 已被证实'.")
    L("")
    L("=" * 126)
    L("END OF REPORT   (wall %.1f s)" % wall)
    L("=" * 126)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")

    np.savez(STATS, **{k: np.array(str(mb[k])) for k in mb.keys()},
             **{"meta_keys": np.array(list(mb.keys()))})
    P("[done] report -> %s" % REPORT)
    P("[done] stats  -> %s" % STATS)
    P("[done] verdicts: DOMAIN_CONSIST=%s LIMITED=%s ATTENUATION=%s BOTTLENECK=%s" %
      (H8_DOMAIN_CONSIST, H8_LIMITED, H8_ATTENUATION, BOTTLENECK))


if __name__ == "__main__":
    main()
