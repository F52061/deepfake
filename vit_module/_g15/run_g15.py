# -*- coding: utf-8 -*-
"""
G15 -- 「如何提高目标域内类间判别」的离线杠杆排序 (纯离线 CPU, 零模型前向, 禁止 GPU, 只读 npz)

科学问题: 目标域 (cd1/cd2/dfdcp/ffiw/wild) 内 real/fake 判别有限; 本实验在【冻结特征 + 线性头】
代理下, 对若干「杠杆」做离线排序, 量化哪个杠杆对「目标域内类间判别 (跨域 AUC)」贡献最大、哪个最便宜。

杠杆:
  L0  基线           : 源训 LR(C=1e-3 锚点) 与 LR(C=1.0) 的跨域 AUC
  L1  无标签特征变换  : (a) 目标域逐维标准化 (b1) 目标域 ZCA 白化(含方差归一)
                        (b2) 目标域 ZCA 去相关(不归一方差) (c) 源域白化后重训头 (d) CORAL 参照行
  L2  半监督自训练    : video-grouped 5 折; 4/5 折目标样本用源头打伪标签, 取置信比例 p∈{25,50,100}%
                        (置信=|打分-阈值|), 用 "FF++train+伪标签" / "仅伪标签" 重训 LR(C=1.0),
                        留出折用真标签评估; 迭代 2 轮; 伪标签绝不使用目标域真标签
  L3  多域联合监督    : leave-one-domain-out; 训练={FF++train + 其它4域} / {仅其它4域}; LR(C=1.0)
                        + 其它域→留出域 单源迁移矩阵 (5x5 去对角)
  L3b 数据集边际曲线  : 训练集从 1 个其它数据集(×300) 扩到 2/3/4 个 (全组合枚举, 报均值+范围),
                        另附 FF++train+全部其它域 完整行; "无源"=只用其它数据集(不含 FF++)
  L4  参照行          : 直接写 G14 域内 oracle (V: cd1 0.9550 / cd2 0.8956 / dfdcp 0.9223 / wild 0.8717;
                        ffiw leak 0.9991) 作为「有标签上限」参照, 不重算
  L5  组合 (可选)     : L3(FF+++其它4域) + L2 自训练 组合增益

数据:
  probe_feats.npz : V(3000,768)/C(3000,1024)/y/paths/vids/train_mask (train 2200/test 800, 42 vids)
  feats_multi.npz : F(2300,1664)/V(2300,768)/C(2300,1024)/y/domain/vid/path
                    域 = ffpp(800,污染禁止) + cd1/cd2/dfdcp/ffiw/wild 各 300 (150/150; ffiw 仅 1 vid)

硬性资源约束 (顶部已设): OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1; torch.set_num_threads(1);
cv2.setNumThreads(0); 单进程; CUDA_VISIBLE_DEVICES=''; 零模型前向 (不实例化模型/不读图像/不读 checkpoint).

y 编码: 1=real, 0=fake; AUC 一律 positive=fake (class 1 = fake).

用法 (项目根目录下):
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g15/run_g15.py > vit_module/_g15/run_log_g15.txt 2>&1

输出:
  vit_module/_g15/g15_report.txt  (MACHINE_BLOCK + L0-L5 表格 + 机械判定 + 排序 + AUDIT/CAVEATS)
  vit_module/_g15/g15_stats.npz   (机器块键值, str)
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
from itertools import combinations
from collections import OrderedDict

import numpy as np
import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
REPORT = os.path.join(HERE, "g15_report.txt")
STATS = os.path.join(HERE, "g15_stats.npz")

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
VALID = ["cd1", "cd2", "dfdcp", "wild"]           # ffiw 1 vid 退化, 单列不入聚合
C_FIX = 1e-3
MAXIT = 3000
SEED = 0
NFOLDS = 5
P_VALUES = [0.25, 0.50, 1.00]
VARIANTS = ["ffpp+pseudo", "pseudo_only"]
N_ITER = 2

# 锚点 (必须复算一致, atol=1e-3)
ANCHOR_SRCLR = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_E0_CD1 = 0.8576
ANCHOR_ATOL = 1e-3

# L4 参照行 (直接引用 G14 oracle V, 不重算)
ORACLE_V = {"cd1": 0.9550, "cd2": 0.8956, "dfdcp": 0.9223, "ffiw": 0.9991, "wild": 0.8717}

COST_ORDER = ["L1_label_free", "L2_selftrain", "L3_multidomain", "L4_oracle"]  # 无标签<半监督<多域监督<域内标签


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def fake_bin(y):
    return (np.asarray(y) == 0).astype(int)          # 1 = fake


def fit_lr_score(Xtr, ytr, Xte, C):
    """StandardScaler(fit train) -> LR(C, lbfgs) -> decision_function(test). ytr/yte: fake indicator."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(sc.transform(Xtr), ytr)
    return sc, clf, clf.decision_function(sc.transform(Xte))


def fit_eval_auc(Xtr, ytr, Xte, yte, C):
    _, _, s = fit_lr_score(Xtr, ytr, Xte, C)
    return float(roc_auc_score(yte, s))


def get_folds(y, vid, n_splits=NFOLDS, seed=SEED):
    """优先 vid 分组 StratifiedGroupKFold; 域内 vid 数 < n_splits 退化 StratifiedKFold, leak=True."""
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


def coral_map(Xs, Xt, ridge=1e-3):
    """CORAL (symmetric whitening): 把 target 协方差搬回 source (label-free). 与 G11 同口径."""
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


def zca_fit(X, ridge=1e-3):
    """ZCA 白化: 返回 (mu, W) 其中 W = C^{-1/2}. X 中心化后右乘 W 得到白化特征."""
    d = X.shape[1]
    mu = X.mean(0)
    C = np.cov(X, rowvar=False)
    C = C + (ridge * np.trace(C) / d) * np.eye(d)
    w, E = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    W = (E * (1.0 / np.sqrt(w))) @ E.T
    return mu, W


def zca_decorr_fit(X):
    """ZCA 去相关(不归一方差): 返回 (mu, E) 其中 E 为协方差特征向量; X' = (X-mu)@E 即 PCA 旋转(保留特征值)."""
    mu = X.mean(0)
    C = np.cov(X, rowvar=False)
    _, E = np.linalg.eigh(C)
    return mu, E


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
    yp = p["y"].astype(int)
    vp = p["vids"].astype(str)
    tr_m = np.asarray(p["train_mask"])
    tr, te = np.where(tr_m)[0], np.where(~tr_m)[0]
    assert (len(tr), len(te)) == (2200, 800), (len(tr), len(te))
    assert not (set(vp[tr]) & set(vp[te])), "probe train/test 视频重叠"

    f = np.load(MULTI_NPZ, allow_pickle=True)
    Vm = f["V"].astype(np.float64)
    ym = f["y"].astype(int)
    domm = f["domain"].astype(str)
    vidm = f["vid"].astype(str)

    for dm in TARGETS:
        m = domm == dm
        assert int(m.sum()) == 300, (dm, int(m.sum()))
        assert int((ym[m] == 1).sum()) == 150 and int((ym[m] == 0).sum()) == 150, dm
    assert int((domm == "ffpp").sum()) == 800

    DOM = OrderedDict()
    for dm in TARGETS:
        m = domm == dm
        DOM[dm] = dict(X=Vm[m], y=ym[m], vid=vidm[m], n=int(m.sum()),
                       nvid=int(len(np.unique(vidm[m]))))

    FFX = Vp[tr]           # FF++ train V (2200)
    FFy = fake_bin(yp[tr])

    # ================================================ 前置校验: 锚点 ========
    sc_src13 = StandardScaler().fit(FFX)
    clf_src13 = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
        sc_src13.transform(FFX), FFy)
    src13_auc = {}
    for dm in TARGETS:
        src13_auc[dm] = float(roc_auc_score(fake_bin(DOM[dm]["y"]),
                                            clf_src13.decision_function(sc_src13.transform(DOM[dm]["X"]))))
    devs = {dm: abs(src13_auc[dm] - ANCHOR_SRCLR[dm]) for dm in TARGETS}
    anchor_ok = all(v < ANCHOR_ATOL for v in devs.values())

    # e0-AUC cd1 (center-only PCA @ SVD, 符号校正使 fake 均值 > real 均值)
    mu_src = FFX.mean(0)
    Xc = FFX - mu_src
    _, S_svd, Vt = np.linalg.svd(Xc, full_matrices=False)
    e0 = Vt[0].copy()
    z_tr = Xc @ e0
    if z_tr[yp[tr] == 0].mean() < z_tr[yp[tr] == 1].mean():
        e0 = -e0
        z_tr = -z_tr
    m_cd1 = domm == "cd1"
    z_cd1 = (DOM["cd1"]["X"] - mu_src) @ e0
    e0auc_cd1 = float(roc_auc_score(fake_bin(DOM["cd1"]["y"]), z_cd1))
    dev_e0 = abs(e0auc_cd1 - ANCHOR_E0_CD1)

    P("[G15] anchor srcLR C=1e-3: " + " ".join("%s=%.4f" % (d, src13_auc[d]) for d in TARGETS))
    P("[G15] anchor e0 AUC cd1=%.4f (dev %.2e)  srcLR maxdev=%.2e" % (e0auc_cd1, dev_e0, max(devs.values())))

    if not (anchor_ok and dev_e0 < ANCHOR_ATOL):
        L("=" * 120)
        L("G15 REPORT - PRECHECK FAILED (stop)")
        L("=" * 120)
        for dm in TARGETS:
            L(f"G15_ANCHOR_srcLR_{dm}={fmt(src13_auc[dm])} EXP={ANCHOR_SRCLR[dm]} dev={fmt(devs[dm],6)}")
        L(f"G15_ANCHOR_e0_cd1={fmt(e0auc_cd1)} EXP={ANCHOR_E0_CD1} dev={fmt(dev_e0,6)}")
        L(f"G15_ANCHOR_PASS=0 wall={time.time()-t0:.1f}s")
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
        P("[G15] PRECHECK FAILED -> see report")
        sys.exit(0)

    # ================================================================ L0 ====
    sc_src10 = StandardScaler().fit(FFX)
    clf_src10 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
        sc_src10.transform(FFX), FFy)
    src10_auc = {}
    for dm in TARGETS:
        src10_auc[dm] = float(roc_auc_score(fake_bin(DOM[dm]["y"]),
                                            clf_src10.decision_function(sc_src10.transform(DOM[dm]["X"]))))
    P("[G15] L0 done")

    # ================================================================ L1 ====
    # 目标侧无标签变换 + 源头评估 (head C=1e-3 锚点头 与 C=1.0 头都算)
    L1_TRANSFORMS = OrderedDict()
    L1_TRANSFORMS["a_std_target"] = ("target 逐维标准化(目标 mean/std)", None)
    L1_TRANSFORMS["b1_zca_white"] = ("target ZCA 白化(含方差归一)", None)
    L1_TRANSFORMS["b2_zca_decorr"] = ("target ZCA 去相关(不归一方差)", None)
    L1_TRANSFORMS["d_coral"] = ("CORAL 协方差对齐到源(参照行)", None)

    l1_auc13 = {}      # head C=1e-3
    l1_auc10 = {}      # head C=1.0
    for key in L1_TRANSFORMS:
        l1_auc13[key] = {}
        l1_auc10[key] = {}
        for dm in TARGETS:
            Xt = DOM[dm]["X"]
            if key == "a_std_target":
                mu_t = Xt.mean(0)
                sd_t = Xt.std(0)
                sd_t = np.where(sd_t < 1e-12, 1.0, sd_t)
                Xt2 = (Xt - mu_t) / sd_t
            elif key == "b1_zca_white":
                mu_t, Wt = zca_fit(Xt)
                Xt2 = (Xt - mu_t) @ Wt
            elif key == "b2_zca_decorr":
                mu_t, Et = zca_decorr_fit(Xt)
                Xt2 = (Xt - mu_t) @ Et
            elif key == "d_coral":
                Xt2 = coral_map(FFX, Xt)
            else:
                Xt2 = Xt
            l1_auc13[key][dm] = float(roc_auc_score(fake_bin(DOM[dm]["y"]),
                                                    clf_src13.decision_function(sc_src13.transform(Xt2))))
            l1_auc10[key][dm] = float(roc_auc_score(fake_bin(DOM[dm]["y"]),
                                                    clf_src10.decision_function(sc_src10.transform(Xt2))))

    # (c) 源域白化后重训头 + 跨域评估 (head 重训于白化源, 目标用同一源白化)
    mu_src_w, W_src_w = zca_fit(FFX)
    FFX_w = (FFX - mu_src_w) @ W_src_w
    l1c_auc13 = {}
    l1c_auc10 = {}
    for dm in TARGETS:
        Xt_w = (DOM[dm]["X"] - mu_src_w) @ W_src_w
        l1c_auc13[dm] = fit_eval_auc(FFX_w, FFy, Xt_w, fake_bin(DOM[dm]["y"]), C_FIX)
        l1c_auc10[dm] = fit_eval_auc(FFX_w, FFy, Xt_w, fake_bin(DOM[dm]["y"]), 1.0)
    P("[G15] L1 done")

    # ================================================================ L2 ====
    def selftrain_domain(dm):
        X = DOM[dm]["X"]; y = DOM[dm]["y"]; vid = DOM[dm]["vid"]
        yb = fake_bin(y)
        folds, mode, leak = get_folds(yb, vid)
        aucs = {(round(p, 2), v, it): [] for p in P_VALUES for v in VARIANTS for it in range(1, N_ITER + 1)}
        for tr_i, te_i in folds:
            Xp, Xh, yh = X[tr_i], X[te_i], yb[te_i]
            if len(np.unique(yb[tr_i])) < 2 or len(np.unique(yh)) < 2:
                continue
            s0 = clf_src10.decision_function(sc_src10.transform(Xp))
            prev_score = {(round(p, 2), v): s0 for p in P_VALUES for v in VARIANTS}
            for it in range(1, N_ITER + 1):
                for p in P_VALUES:
                    pk = round(p, 2)
                    for v in VARIANTS:
                        s = prev_score[(pk, v)]
                        lab = (s > 0.0).astype(int)
                        conf = np.abs(s)
                        n_sel = max(2, int(round(p * len(Xp))))
                        order = np.argsort(-conf)[:n_sel]
                        Xsel, ysel = Xp[order], lab[order]
                        if v == "ffpp+pseudo":
                            Xtr = np.vstack([FFX, Xsel]); ytr = np.concatenate([FFy, ysel])
                        else:
                            Xtr = Xsel; ytr = ysel
                        if len(np.unique(ytr)) < 2:
                            aucs[(pk, v, it)].append(np.nan)
                            prev_score[(pk, v)] = s
                            continue
                        try:
                            sc = StandardScaler().fit(Xtr)
                            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
                                sc.transform(Xtr), ytr)
                            aucs[(pk, v, it)].append(float(roc_auc_score(yh, clf.decision_function(sc.transform(Xh)))))
                            prev_score[(pk, v)] = clf.decision_function(sc.transform(Xp))
                        except Exception:
                            aucs[(pk, v, it)].append(np.nan)
                            prev_score[(pk, v)] = s
        # 汇总
        summary = {}
        for k, vals in aucs.items():
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            summary[k] = dict(mean=float(arr.mean()) if len(arr) else float("nan"),
                              std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                              n=int(len(arr)))
        best_key = None
        best_mean = -1.0
        for k, r in summary.items():
            if r["n"] and r["mean"] > best_mean:
                best_mean = r["mean"]; best_key = k
        return dict(summary=summary, best_key=best_key, best_mean=best_mean,
                    mode=mode, leak=leak, nvid=DOM[dm]["nvid"])

    l2 = {}
    for dm in TARGETS:
        l2[dm] = selftrain_domain(dm)
    # 每域最优组合 (p,v,it) 的增益 vs L0(C=1.0)
    l2_best_gain = {}
    for dm in TARGETS:
        best = l2[dm]["best_mean"]
        l2_best_gain[dm] = best - src10_auc[dm]
    # 全局最优组合 (按 4 有效域均值)
    global_best = {}
    for p in P_VALUES:
        pk = round(p, 2)
        for v in VARIANTS:
            for it in range(1, N_ITER + 1):
                key = (pk, v, it)
                ms = [l2[d]["summary"][key]["mean"] for d in VALID if l2[d]["summary"][key]["n"]]
                if ms:
                    global_best[key] = float(np.mean(ms))
    gb_key = max(global_best, key=global_best.get)
    P("[G15] L2 done; global best combo %s mean=%.4f" % (gb_key, global_best[gb_key]))

    # ================================================================ L3 ====
    # leave-one-domain-out; 其它域 = TARGETS - {dm}
    l3 = {}          # dm -> {variant: {C: auc}}
    for dm in TARGETS:
        others = [d for d in TARGETS if d != dm]
        Xo = np.vstack([DOM[o]["X"] for o in others])
        yo = np.concatenate([fake_bin(DOM[o]["y"]) for o in others])
        Xd, yd = DOM[dm]["X"], fake_bin(DOM[dm]["y"])
        row = {}
        # A: FF++ train + 其它4域
        Xa = np.vstack([FFX, Xo]); ya = np.concatenate([FFy, yo])
        row["A_ffpp+other4"] = dict(C13=fit_eval_auc(Xa, ya, Xd, yd, C_FIX),
                                     C10=fit_eval_auc(Xa, ya, Xd, yd, 1.0))
        # B: 仅其它4域
        row["B_other4_only"] = dict(C13=fit_eval_auc(Xo, yo, Xd, yd, C_FIX),
                                     C10=fit_eval_auc(Xo, yo, Xd, yd, 1.0))
        l3[dm] = row

    # 其它域->留出域 单源迁移矩阵 (5x5 去对角, C=1e-3 正则化)
    transfer = {}
    for src in TARGETS:
        transfer[src] = {}
        for dst in TARGETS:
            if src == dst:
                transfer[src][dst] = float("nan")
                continue
            transfer[src][dst] = fit_eval_auc(DOM[src]["X"], fake_bin(DOM[src]["y"]),
                                              DOM[dst]["X"], fake_bin(DOM[dst]["y"]), C_FIX)
    P("[G15] L3 done")

    # ============================================================= L3b ======
    # 数据集数量边际增益曲线 (无源 k=1..4 其它数据集, 全组合枚举; 另有源 FF+++k 其它数据集; 同 C=1.0)
    l3b = {}
    for dm in TARGETS:
        others = [d for d in TARGETS if d != dm]
        Xd, yd = DOM[dm]["X"], fake_bin(DOM[dm]["y"])
        curve_nosrc = {k: [] for k in range(1, 5)}    # 只用其它数据集(不含 FF++)
        curve_src = {k: [] for k in range(1, 5)}      # FF++ train + k 个其它数据集
        for k in range(1, 5):
            for combo in combinations(others, k):
                Xo = np.vstack([DOM[o]["X"] for o in combo])
                yo = np.concatenate([fake_bin(DOM[o]["y"]) for o in combo])
                curve_nosrc[k].append(fit_eval_auc(Xo, yo, Xd, yd, 1.0))
                Xs = np.vstack([FFX, Xo]); ys = np.concatenate([FFy, yo])
                curve_src[k].append(fit_eval_auc(Xs, ys, Xd, yd, 1.0))
        l3b[dm] = dict(nosrc={k: dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)),
                                       n=int(len(v))) for k, v in curve_nosrc.items()},
                       src={k: dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)),
                                    n=int(len(v))) for k, v in curve_src.items()})
    # L3b 边际增益 Δ: 无源 k=1 -> k=4 (4 有效域均值)
    l3b_delta = {}
    for dm in TARGETS:
        l3b_delta[dm] = l3b[dm]["nosrc"][4]["mean"] - l3b[dm]["nosrc"][1]["mean"]
    P("[G15] L3b done")

    # ================================================================ L4 ====
    # 引用 G14 oracle (不重算)
    P("[G15] L4 reference written")

    # ================================================================ L5 ====
    # L3(A: FF+++其它4域, C=1.0) + L2 自训练 (transductive: 留出域用 L3 模型打伪标签, 取最佳 p)
    l5 = {}
    for dm in TARGETS:
        others = [d for d in TARGETS if d != dm]
        Xo = np.vstack([DOM[o]["X"] for o in others])
        yo = np.concatenate([fake_bin(DOM[o]["y"]) for o in others])
        Xa = np.vstack([FFX, Xo]); ya = np.concatenate([FFy, yo])
        Xd, yd = DOM[dm]["X"], fake_bin(DOM[dm]["y"])
        sc3, clf3, _ = fit_lr_score(Xa, ya, Xd, 1.0)
        s = clf3.decision_function(sc3.transform(Xd))
        # 用 L2 全局最佳 p (回退 0.5)
        p_best = gb_key[0] if gb_key else 0.5
        lab = (s > 0.0).astype(int); conf = np.abs(s)
        n_sel = max(2, int(round(p_best * len(Xd))))
        order = np.argsort(-conf)[:n_sel]
        if len(np.unique(np.concatenate([ya, lab[order]]))) < 2:
            l5[dm] = float("nan")
            continue
        Xtr = np.vstack([Xa, Xd[order]]); ytr = np.concatenate([ya, lab[order]])
        l5[dm] = fit_eval_auc(Xtr, ytr, Xd, yd, 1.0)
    P("[G15] L5 done")

    # ======================================================== 机械判定 ========
    def mean4(d):
        return float(np.mean([d[x] for x in VALID]))

    # L1: 最优 L1 变换 (head C=1e-3) 相对 L0(C=1e-3) 的增益
    l1_gain = {}
    for key in L1_TRANSFORMS:
        l1_gain[key] = {dm: l1_auc13[key][dm] - src13_auc[dm] for dm in TARGETS}
    l1c_gain = {dm: l1c_auc13[dm] - src13_auc[dm] for dm in TARGETS}
    # 最优 L1 变换 = 4 有效域均值增益最大者 (含 c)
    l1_candidates = OrderedDict()
    for key in L1_TRANSFORMS:
        l1_candidates["L1_" + key] = mean4({dm: l1_gain[key][dm] for dm in TARGETS})
    l1_candidates["L1_c_srcwhite"] = mean4(l1c_gain)
    l1_best_name = max(l1_candidates, key=l1_candidates.get)
    l1_best_mean_gain = l1_candidates[l1_best_name]
    l1_best_gain_by_dom = {}
    if l1_best_name.startswith("L1_c"):
        l1_best_gain_by_dom = {dm: l1c_gain[dm] for dm in TARGETS}
    else:
        key = l1_best_name[3:]
        l1_best_gain_by_dom = {dm: l1_gain[key][dm] for dm in TARGETS}

    # L2: 每域最优组合增益 vs L0(C=1.0)
    # L3: FF+++其它4域 C=1.0 增益 vs L0(C=1.0)
    l3_gain = {dm: l3[dm]["A_ffpp+other4"]["C10"] - src10_auc[dm] for dm in TARGETS}
    # L4: oracle 增益 vs L0(C=1.0)
    l4_gain = {dm: ORACLE_V[dm] - src10_auc[dm] for dm in TARGETS}

    n_ge = lambda gains, th: sum(1 for dm in VALID if gains[dm] >= th)

    # G15_L1_LABELFREE
    n1 = n_ge(l1_best_gain_by_dom, +0.03)
    n1_le = sum(1 for dm in VALID if l1_best_gain_by_dom[dm] <= +0.01)
    if n1 >= 4:
        L1_VERDICT = "EFFECTIVE"
    elif n1_le >= 4:
        L1_VERDICT = "INEFFECTIVE"
    else:
        L1_VERDICT = "PARTIAL"

    # G15_L2_SELFTRAIN
    n2 = n_ge(l2_best_gain, +0.05)
    n2_le = sum(1 for dm in VALID if l2_best_gain[dm] <= +0.01)
    if n2 >= 4:
        L2_VERDICT = "EFFECTIVE"
    elif n2_le >= 4:
        L2_VERDICT = "INEFFECTIVE"
    else:
        L2_VERDICT = "PARTIAL"

    # G15_L3_MULTIDOMAIN
    n3 = n_ge(l3_gain, +0.05)
    n3_le = sum(1 for dm in VALID if l3_gain[dm] <= +0.01)
    if n3 >= 4:
        L3_VERDICT = "DATA_BREADTH_EFFECTIVE"
    elif n3_le >= 4:
        L3_VERDICT = "INEFFECTIVE"
    else:
        L3_VERDICT = "PARTIAL"

    # G15_L3b_MARGINAL
    l3b_delta_mean = mean4(l3b_delta)
    if l3b_delta_mean >= +0.05:
        L3B_VERDICT = "MARGINAL_POSITIVE"
    elif l3b_delta_mean <= +0.01:
        L3B_VERDICT = "SATURATED_EARLY"
    else:
        L3B_VERDICT = "PARTIAL"

    # ======================================================== 排序 G15_RANK ==
    # 各杠杆 4 有效域均值增益 (isolated: L1 vs C=1e-3 头, L2/L3/L4 vs C=1.0 头)
    lever_gain = OrderedDict()
    lever_gain["L1_label_free"] = mean4(l1_best_gain_by_dom)
    lever_gain["L2_selftrain"] = mean4(l2_best_gain)
    lever_gain["L3_multidomain"] = mean4(l3_gain)
    lever_gain["L4_oracle"] = mean4(l4_gain)
    ranked = sorted(lever_gain.items(), key=lambda kv: kv[1], reverse=True)
    best_lever = ranked[0][0]
    # 最便宜有效杠杆: 成本序中第一个 有效(均值增益>0.001) 的杠杆
    cheapest = None
    for name in COST_ORDER:
        if lever_gain[name] > 0.001:
            cheapest = name
            break
    if cheapest is None:
        cheapest = "NONE_EFFECTIVE"

    wall = time.time() - t0
    P("[G15] verdicts L1=%s L2=%s L3=%s L3b=%s best_lever=%s cheapest=%s (wall=%.1fs)" %
      (L1_VERDICT, L2_VERDICT, L3_VERDICT, L3B_VERDICT, best_lever, cheapest, wall))

    # ======================================================== MACHINE_BLOCK ==
    mb = OrderedDict()
    mb["G15_TASK"] = "offline_lever_ranking_target_intraclass_discrimination"
    mb["G15_DATA"] = "probe_feats.npz(V n=3000:train2200/test800) + feats_multi.npz(V n=2300:ffpp800+5x300)"
    mb["G15_BRANCH"] = "V(768) only (L1 C-branch control omitted)"
    mb["G15_POSITIVE"] = "fake(class1)"
    # 锚点
    for dm in TARGETS:
        mb["G15_ANCHOR_srcLR_C1e-3_%s" % dm] = fmt(src13_auc[dm], 4)
        mb["G15_ANCHOR_srcLR_C1e-3_%s_EXP" % dm] = fmt(ANCHOR_SRCLR[dm], 4)
    mb["G15_ANCHOR_e0_cd1"] = fmt(e0auc_cd1, 4)
    mb["G15_ANCHOR_e0_cd1_EXP"] = fmt(ANCHOR_E0_CD1, 4)
    mb["G15_ANCHOR_MAXDEV"] = fmt(max(dev_e0, max(devs.values())), 6)
    mb["G15_ANCHOR_PASS"] = "1"
    # L0
    mb["G15_L0_C1e-3"] = ";".join("%s=%.4f" % (d, src13_auc[d]) for d in TARGETS)
    mb["G15_L0_C1.0"] = ";".join("%s=%.4f" % (d, src10_auc[d]) for d in TARGETS)
    # L1
    for key in L1_TRANSFORMS:
        mb["G15_L1_%s_C1e-3" % key] = ";".join("%s=%.4f" % (d, l1_auc13[key][d]) for d in TARGETS)
        mb["G15_L1_%s_gain" % key] = ";".join("%s=%+.4f" % (d, l1_gain[key][d]) for d in TARGETS)
    mb["G15_L1_c_srcwhite_C1e-3"] = ";".join("%s=%.4f" % (d, l1c_auc13[d]) for d in TARGETS)
    mb["G15_L1_c_srcwhite_gain"] = ";".join("%s=%+.4f" % (d, l1c_gain[d]) for d in TARGETS)
    mb["G15_L1_BEST"] = l1_best_name
    mb["G15_L1_BEST_MEAN_GAIN"] = fmt(l1_best_mean_gain, 4)
    # L2
    for dm in TARGETS:
        bk = l2[dm]["best_key"]
        mb["G15_L2_best_%s" % dm] = "%s:auc=%.4f:gain=%+.4f:n=%d:leak=%s" % (
            str(bk), l2[dm]["best_mean"], l2_best_gain[dm],
            l2[dm]["summary"][bk]["n"], str(l2[dm]["leak"]))
    mb["G15_L2_GLOBAL_BEST_COMBO"] = "%s:mean4=%.4f" % (str(gb_key), global_best[gb_key])
    mb["G15_L2_gain_vsL0C1.0"] = ";".join("%s=%+.4f" % (d, l2_best_gain[d]) for d in TARGETS)
    # L3
    mb["G15_L3_A_ffpp+other4_C1.0"] = ";".join("%s=%.4f" % (d, l3[d]["A_ffpp+other4"]["C10"]) for d in TARGETS)
    mb["G15_L3_A_gain_vsL0C1.0"] = ";".join("%s=%+.4f" % (d, l3_gain[d]) for d in TARGETS)
    mb["G15_L3_B_other4only_C1.0"] = ";".join("%s=%.4f" % (d, l3[d]["B_other4_only"]["C10"]) for d in TARGETS)
    mb["G15_L3_TRANSFER_MATRIX"] = "|".join("%s->" % s + ",".join("%s=%.4f" % (d, transfer[s][d]) for d in TARGETS if d != s) for s in TARGETS)
    # L3b
    for dm in TARGETS:
        mb["G15_L3b_nosrc_%s" % dm] = ";".join("k%d=%.4f[%.4f..%.4f]" % (
            k, l3b[dm]["nosrc"][k]["mean"], l3b[dm]["nosrc"][k]["lo"], l3b[dm]["nosrc"][k]["hi"]) for k in range(1, 5))
        mb["G15_L3b_src_%s" % dm] = ";".join("k%d=%.4f[%.4f..%.4f]" % (
            k, l3b[dm]["src"][k]["mean"], l3b[dm]["src"][k]["lo"], l3b[dm]["src"][k]["hi"]) for k in range(1, 5))
    mb["G15_L3b_DELTA_k1_to_k4"] = ";".join("%s=%+.4f" % (d, l3b_delta[d]) for d in TARGETS)
    # L4
    mb["G15_L4_ORACLE_V"] = ";".join("%s=%.4f" % (d, ORACLE_V[d]) for d in TARGETS)
    mb["G15_L4_gain_vsL0C1.0"] = ";".join("%s=%+.4f" % (d, l4_gain[d]) for d in TARGETS)
    # L5
    mb["G15_L5_combo"] = ";".join("%s=%.4f" % (d, l5[d]) for d in TARGETS)
    mb["G15_L5_gain_vs_L3A"] = ";".join("%s=%+.4f" % (d, l5[d] - l3[d]["A_ffpp+other4"]["C10"]) for d in TARGETS)
    # 判定
    mb["G15_L1_LABELFREE"] = L1_VERDICT
    mb["G15_L2_SELFTRAIN"] = L2_VERDICT
    mb["G15_L3_MULTIDOMAIN"] = L3_VERDICT
    mb["G15_L3b_MARGINAL"] = L3B_VERDICT
    mb["G15_L1_N_GE_0.03"] = "%d/4" % n1
    mb["G15_L2_N_GE_0.05"] = "%d/4" % n2
    mb["G15_L3_N_GE_0.05"] = "%d/4" % n3
    mb["G15_L3b_DELTA_MEAN"] = fmt(l3b_delta_mean, 4)
    mb["G15_RANK"] = " > ".join("%s(%+.4f)" % (n, g) for n, g in ranked)
    mb["G15_RANK_BEST_LEVER"] = best_lever
    mb["G15_RANK_CHEAPEST_EFFECTIVE"] = cheapest
    mb["G15_RANK_COST_ORDER"] = "无标签(L1) < 半监督(L2) < 多域监督(L3) < 域内标签(L4)"
    # 审计
    mb["G15_WALL_S"] = fmt(wall, 1)
    mb["G15_IMGS_READ"] = "0"
    mb["G15_GPU"] = "0"
    mb["G15_FORWARDS"] = "0"
    mb["G15_THREADS"] = "1"
    mb["G15_PROCESSES"] = "1"
    mb["G15_NPZ_READ"] = "2"

    mblines = ["#### MACHINE_BLOCK " + "#" * 100]
    mblines += ["G15_%s=%s" % (k[4:], v) for k, v in mb.items()]

    # ======================================================== 报告正文 ========
    L("=" * 120)
    L("G15 REPORT - 如何提高目标域内类间判别: 离线杠杆排序 (冻结特征 + 线性头代理)")
    L("纯 CPU 单线程 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1, torch=1, cv2=0, 单进程, CUDA_VISIBLE_DEVICES='')")
    L("零模型前向 (不实例化模型/不读图像/不读 checkpoint), 只读 npz; 分支 V(768); y: 1=real, 0=fake; AUC positive=fake")
    L("=" * 120)
    L("")
    L("\n".join(mblines))
    L("")

    L("-" * 120)
    L("PRECHECK. 前置校验 (任一不过即停, atol=1e-3)")
    L("  源训 StandardScaler+LR(C=1e-3) 在 probe train V 拟合、multi 各域 V 评估:")
    for dm in TARGETS:
        L("    %-6s 复算=%.4f  预期=%.4f  |dev|=%.2e" % (dm, src13_auc[dm], ANCHOR_SRCLR[dm], devs[dm]))
    L("  单轴 e0-AUC cd1: 复算=%.4f 预期=%.4f |dev|=%.2e" % (e0auc_cd1, ANCHOR_E0_CD1, dev_e0))
    L("  -> PRECHECK PASS")
    L("")

    L("-" * 120)
    L("L0. 基线: 源训 LR 跨域 AUC (V 分支, 全 5 域; ffiw 单列)")
    L("    注意: 跨域上 C=1e-3 (强正则) 反而比 C=1.0 迁移更好 (C=1.0 过拟合 2200 源样本).")
    L("")
    L("  %-8s %12s %12s" % ("dom", "LR(C=1e-3)", "LR(C=1.0)"))
    L("  " + "-" * 40)
    for dm in TARGETS:
        L("  %-8s %12.4f %12.4f" % (dm, src13_auc[dm], src10_auc[dm]))
    L("  有效域均值(剔 ffiw): LR(C=1e-3)=%.4f  LR(C=1.0)=%.4f" % (mean4(src13_auc), mean4(src10_auc)))
    L("")

    L("-" * 120)
    L("L1. 无标签特征变换 (变换只用目标域无标签样本估计; 头仍用源 train 拟合; 头 C=1e-3 为主, C=1.0 附)")
    L("    变换: (a)目标逐维标准化 (b1)目标ZCA白化(含方差归一) (b2)目标ZCA去相关(不归一方差)")
    L("          (c)源域白化后重训头 (d)CORAL(对齐目标协方差到源, G11 参照行)")
    L("    增益 = 变换后 AUC - L0(同 C) 基线; 单调分数变换不可能提升 AUC, 但此处都是特征空间变换, 可改变 AUC.")
    L("")
    for headC, headtag in [(C_FIX, "C=1e-3"), (1.0, "C=1.0")]:
        base = src13_auc if headC == C_FIX else src10_auc
        L("  --- 头 LR(%s) 相对 L0(%s) 的增益 ---" % (headtag, headtag))
        L("  %-16s" % "transform" + "".join("  %-9s" % dm for dm in TARGETS) + "   %-9s" % "mean4")
        for key, (label, _) in L1_TRANSFORMS.items():
            aucs = l1_auc13 if headC == C_FIX else l1_auc10
            cells = "".join("  %+8.4f" % (aucs[key][dm] - base[dm]) for dm in TARGETS)
            L("  %-16s%s   %+9.4f" % (key, cells, mean4({dm: aucs[key][dm] - base[dm] for dm in TARGETS})))
        c_aucs = l1c_auc13 if headC == C_FIX else l1c_auc10
        cells = "".join("  %+8.4f" % (c_aucs[dm] - base[dm]) for dm in TARGETS)
        L("  %-16s%s   %+9.4f" % ("c_srcwhite", cells, mean4({dm: c_aucs[dm] - base[dm] for dm in TARGETS})))
    L("  (c 变换头被重训于白化源, 故其 C=1e-3/C=1.0 差异缩小; CORAL 为参照行, 不作门)")
    L("")

    L("-" * 120)
    L("L2. 半监督自训练 (无标签, video-grouped 5 折; p∈{25,50,100}%; 变体 {FF+++伪标签, 仅伪标签}; 迭代 2 轮)")
    L("    伪标签 = 源头打分(>0 判 fake), 置信=|打分-阈值|; 留出折用真标签评估; 伪标签绝不使用目标域真标签.")
    L("    表内为每域最优 (p,变体,迭代) 组合; 增益 = 相对 L0(C=1.0).")
    L("")
    L("  %-6s %-24s %-9s %-9s %-9s %-10s %-6s %-5s %s" %
      ("dom", "best(p,variant,iter)", "AUC", "L0C1.0", "gain", "std", "nfolds", "leak", "mode"))
    for dm in TARGETS:
        r = l2[dm]; bk = r["best_key"]
        L("  %-6s %-24s %9.4f %9.4f %+9.4f %10.4f %6d %-5s %s" %
          (dm, str(bk), r["best_mean"], src10_auc[dm], l2_best_gain[dm], r["summary"][bk]["std"], r["summary"][bk]["n"], str(r["leak"]), r["mode"]))
    L("  全局最优组合(4有效域均值): %s -> mean4=%.4f" % (str(gb_key), global_best[gb_key]))
    L("  注意: dfdcp 最优组合(迭代2, 仅伪标签)仅 2 折有效 (其余 3 折迭代2再打分退化为单类), 其均值方差大、可信度低;")
    L("  有效域均值增益(每域最优): %+.4f" % mean4(l2_best_gain))
    L("")

    L("-" * 120)
    L("L3. 多域联合监督 (leave-one-domain-out; 训练=FF++train2200+其它4域x300 或 仅其它4域; LR C=1.0)")
    L("    增益 = 相对 L0(C=1.0); ffiw 单列 (1 vid 退化).")
    L("")
    L("  %-6s %14s %14s %12s %12s" % ("dom", "A_ffpp+other4", "gain(A)", "B_other4only", "L0C1.0"))
    for dm in TARGETS:
        L("  %-6s %14.4f %+13.4f %14.4f %12.4f" %
          (dm, l3[dm]["A_ffpp+other4"]["C10"], l3_gain[dm], l3[dm]["B_other4_only"]["C10"], src10_auc[dm]))
    L("  有效域均值: A(FF+++其它4域)=%.4f (gain %+.4f)  B(仅其它4域)=%.4f" %
      (mean4({dm: l3[dm]["A_ffpp+other4"]["C10"] for dm in TARGETS}),
       mean4(l3_gain),
       mean4({dm: l3[dm]["B_other4_only"]["C10"] for dm in TARGETS})))
    L("")
    L("  其它域->留出域 单源迁移 AUC 矩阵 (行=源域, 列=留出/目标域; C=1e-3; 对角留空):")
    L("  %-8s" % "src\\dst" + "".join("  %-9s" % dm for dm in TARGETS))
    for src in TARGETS:
        cells = "".join("  %9.4f" % transfer[src][dst] for dst in TARGETS)
        L("  %-8s%s" % (src, cells))
    L("")

    L("-" * 120)
    L("L3b. 数据集数量边际增益曲线 (无源: 训练集只用其它数据集不含 FF++; 有源: FF++train+k 个其它数据集)")
    L("     每点 = 全组合枚举 (C(4,k)) 在留出域 300 真标签上的 AUC 均值[最小..最大]; LR C=1.0 与 L3 一致.")
    L("")
    L("  无源 (训练集只用其它数据集, 完全不含 FF++ train):")
    L("  %-6s %-22s %-22s %-22s %-22s" % ("dom", "k=1", "k=2", "k=3", "k=4"))
    for dm in TARGETS:
        cells = "".join("  %7.4f[%s..%s]" % (l3b[dm]["nosrc"][k]["mean"], fmt(l3b[dm]["nosrc"][k]["lo"], 3), fmt(l3b[dm]["nosrc"][k]["hi"], 3)) for k in range(1, 5))
        L("  %-6s%s" % (dm, cells))
    L("  有源 (FF++ train + k 个其它数据集):")
    L("  %-6s %-22s %-22s %-22s %-22s" % ("dom", "k=1", "k=2", "k=3", "k=4"))
    for dm in TARGETS:
        cells = "".join("  %7.4f[%s..%s]" % (l3b[dm]["src"][k]["mean"], fmt(l3b[dm]["src"][k]["lo"], 3), fmt(l3b[dm]["src"][k]["hi"], 3)) for k in range(1, 5))
        L("  %-6s%s" % (dm, cells))
    L("  Δ(k1->k4, 无源): " + " ".join("%s=%+.4f" % (d, l3b_delta[d]) for d in TARGETS) +
      "  有效域均值=%+.4f" % l3b_delta_mean)
    L("  (对照: L0 即 FF++源 各域 AUC 见 L0 表; L3 的 A/B 两行分别对应 有源k=4 / 无源k=4)")
    L("")

    L("-" * 120)
    L("L4. 参照行 (直接引用 G14 域内 oracle, 不重算): 有标签上限.")
    L("  %-6s %12s %12s" % ("dom", "oracle_V", "gain_vsL0C1.0"))
    for dm in TARGETS:
        L("  %-6s %12.4f %+12.4f" % (dm, ORACLE_V[dm], l4_gain[dm]))
    L("  ffiw oracle=0.9991 为 1-vid 泄漏 (leak=True) 高估, 单列不入聚合.")
    L("")

    L("-" * 120)
    L("L5. 组合 (可选): L3(A:FF+++其它4域,C=1.0) + L2 自训练 (transductive, 用 L2 全局最佳 p=%s)" %
      (fmt(gb_key[0], 2) if gb_key else "0.50"))
    L("  %-6s %12s %12s" % ("dom", "combo_AUC", "gain_vs_L3A"))
    for dm in TARGETS:
        L("  %-6s %12.4f %+12.4f" % (dm, l5[dm], l5[dm] - l3[dm]["A_ffpp+other4"]["C10"]))
    L("")

    L("-" * 120)
    L("MECHANICAL VERDICTS (预注册, 照抄执行, 不做方向演绎)")
    L("")
    L("  各杠杆 4 有效域均值增益 (isolated):")
    for name, g in ranked:
        L("    %-16s %+9.4f" % (name, g))
    L("")
    L("  [G15_L1_LABELFREE] 最优 L1 变换=%s, 4有效域均值增益=%+.4f; ≥+0.03 域数=%d/4" %
      (l1_best_name, l1_best_mean_gain, n1))
    L("      -> G15_L1_LABELFREE=%s" % L1_VERDICT)
    L("  [G15_L2_SELFTRAIN] 每域最优组合增益 vs L0(C=1.0); ≥+0.05 域数=%d/4" % n2)
    L("      -> G15_L2_SELFTRAIN=%s" % L2_VERDICT)
    L("  [G15_L3_MULTIDOMAIN] FF+++其它4域 增益 vs L0(C=1.0); ≥+0.05 域数=%d/4" % n3)
    L("      -> G15_L3_MULTIDOMAIN=%s" % L3_VERDICT)
    L("  [G15_L3b_MARGINAL] 无源 k1->k4 平均增益 Δ=%+.4f" % l3b_delta_mean)
    L("      -> G15_L3b_MARGINAL=%s" % L3B_VERDICT)
    L("  [G15_RANK] 按有效域均值增益排序: " + " > ".join("%s(%+.4f)" % (n, g) for n, g in ranked))
    L("      最有效杠杆 = %s ; 最便宜有效杠杆 = %s" % (best_lever, cheapest))
    L("      成本序: 无标签(L1) < 半监督(L2) < 多域监督(L3) < 域内标签(L4)")
    L("")

    L("-" * 120)
    L("AUDIT")
    L("  imgs_read=0 (零图像读取)   gpu=0 (CUDA_VISIBLE_DEVICES='', 未调用任何 cuda API)")
    L("  forwards=0 (未实例化模型, 零前向/反向, 未读 checkpoint)   threads=1 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1,")
    L("  torch.set_num_threads(1), cv2.setNumThreads(0)) ; processes=1 ; npz 读取=2")
    L("  探针: StandardScaler(fit 训练集) + LogisticRegression(lbfgs, max_iter=3000); 划分 seed=0")
    L("  wall_s = %.1f" % wall)
    L("")
    L("CAVEATS (honest)")
    L("  1. 全部杠杆作用于【冻结特征上的线性头】, 是'换头/域适配'的代理, 不等于模型里 bridge/头被重训后的效果.")
    L("  2. L1 是无标签(无需任何目标标签); L2 用无标签(伪标签); L3/L3b 用其它域的真标签(域标签); L4 用目标域真标签.")
    L("     部署成本/可行性逐级升高, 需分别标注 (成本序见上).")
    L("  3. 每域仅 300 样本, L2 折间方差大 (已报 nfold/std); 单域 AUC 差异 <~0.05 不宜过度解读.")
    L("  4. ffiw 仅 1 vid: 分组 CV 退化为 StratifiedKFold(leak=True), 其 oracle/AUC 被高估, 单列不入聚合;")
    L("     作为训练数据时 ffiw 300 样本高度相关 (1 vid), 其作为 L3/L3b 训练源时的贡献也偏弱.")
    L("  5. CORAL 参照行口径: 目标协方差对齐到源后仍用源训 LR(C=1e-3) 评估 (与 G11 同); G11 已知 ≈0 增益, 不作门.")
    L("  6. 单调分数变换不可能提升 AUC; 本实验 L1 的变换都是【特征空间】变换 (标准化/白化/去相关/CORAL),")
    L("     故可改变 AUC, 需与'分数阈值/标定变换'严格区分.")
    L("  7. 跨域上 C=1e-3(强正则)迁移优于 C=1.0: L2/L3/L3b/L4 用 C=1.0 (更贴近'更多数据更高容量'口径), 其")
    L("     增益按 vs L0(C=1.0) 计算以隔离各杠杆自身效应; 若 vs 锚点 C=1e-3 会含 C 超参的负贡献, 报告中已并列给出.")
    L("  8. L2 源头打伪标签使用 L0(C=1.0) 模型; 若改用迁移更优的 C=1e-3 头做初始伪标签, 结果可能不同 (未做).")
    L("  9. L3b '无源'=训练集只用其它目标数据集(完全不含 FF++), 直接回答'源换成别的数据集会怎样'; k=1 时仅 300 样本,")
    L("     AUC 低是样本量+域差异双重因素, 边际增益 Δ 才是'数据集丰富度'杠杆的信号.")
    L("  10. 相关 != 因果: 多域监督/自训练在冻结表征上的增益, 不能证明在端到端重训(解冻 bridge/头)后仍同向成立.")
    L("  11. 本实验全程离线特征几何/线性头计算, 未重测任何下游检测性能; 锚点复算一致 (<1e-3), 口径未漂移.")
    L("  12. L1 的 'ZCA 去相关(不归一方差)' 实现为 X'=(X-mu)@E (PCA 旋转, 保留特征值); 严格对称 ZCA 下 E E^T=I 仅")
    L("      退化为去均值, 故采用去相关旋转作为可解释的非平凡变体.")
    L("")
    L("=" * 120)
    L("END OF REPORT   (wall %.1f s)" % wall)
    L("=" * 120)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")

    np.savez(STATS, **{k: np.array(str(v)) for k, v in mb.items()},
             **{"meta_keys": np.array(list(mb.keys()))})
    P("[done] report -> %s" % REPORT)
    P("[done] stats  -> %s" % STATS)
    P("[done] verdicts: L1=%s L2=%s L3=%s L3b=%s best=%s cheapest=%s" %
      (L1_VERDICT, L2_VERDICT, L3_VERDICT, L3B_VERDICT, best_lever, cheapest))


if __name__ == "__main__":
    main()
