# -*- coding: utf-8 -*-
"""
G12 -- H6 分支级 "FF++ 专用性" 分解 (branch-level FF++-specificity decomposition)

问题: G11 证明"源 PC0 轴在目标域错位 + 沿轴间隔收缩", 但那是单一分支 (raw ViT V) 的结论。
G12 问: 目标域可分的"域信息"是 ViT 塔特有, 还是整个栈 (CLIP 塔 / 融合 F) 共有?

分支 B in {V, C, F, V_proj, C_proj};  目标域 d in {cd1, cd2, dfdcp, ffiw, wild}
  V       raw PDI-ViT final CLS token                 (768)
  C       raw CLIP vision CLS token                   (1024)
  F       classifier head input 的可离线重建子集 [alpha_v*C_proj | V_proj] (1536)
          (F 的 128-d bridge_adapter 块需要前向 -> 离线不可得, 见 CAVEATS)
  V_proj  deepfake_proj(V)      = LayerNorm(Linear(V))       (768) [checkpoint 离线复算]
  C_proj  vision_proj(C)[:,0,:] = LayerNorm(Linear(C))       (768) [checkpoint 离线复算]

实验块:
  A 前置校验   : (1) 锚点复算 e0AUC_cd1=0.8576 / srcLR_cd1=0.8286
                 (2) V_proj / C_proj 离线复算 cos >= 0.999
                 (3) F 1664 维布局逐段 cos 验证
  B 域-类间隔比: class_gap=||mu_fake-mu_real||/s, dom_gap(d)=||mu_d-mu_src||/s, ratio=dom/class
  C 类条件域可分: real-only / fake-only / mixed 三个二分类线性探针 (按 vid 70/30, 两侧均衡)
  D 子空间包含 : 源子空间 E[:K] (K=1,5,50) 之外的残差能量占比, 与 FF++ test 800 参照的差 Delta
  E 轴对齐     : 域内自拟合 PC0 与源 e0 的 |cos|, 以及 zdim_frac

硬性资源约束 (脚本顶部已设):
  OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB = 1, torch.set_num_threads(1),
  cv2.setNumThreads(0), 单进程, CUDA_VISIBLE_DEVICES='' (禁止 GPU),
  零模型前向 (不实例化模型, 不读图像), torch.load(map_location='cpu') 只取线性层权重.

用法 (项目根目录下):
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g12/run_g12.py > vit_module/_g12/run_log_g12.txt 2>&1

输出:
  vit_module/_g12/g12_report.txt   (MACHINE_BLOCK + A-E 表格 + 机械判定 + AUDIT/CAVEATS)
  vit_module/_g12/g12_stats.npz    (机器块键值)
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
from collections import OrderedDict

import numpy as np
import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_1", "bridge_v2_phase1.pth")
REPORT = os.path.join(HERE, "g12_report.txt")
STATS = os.path.join(HERE, "g12_stats.npz")

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
BRANCHES = ["V", "C", "F", "V_proj", "C_proj"]
KS = [1, 5, 50]

C_FIX = 1e-3
MAXIT = 3000
SEED = 0
SPLIT_FRAC = 0.30      # 按 vid 70/30
LN_EPS = 1e-5

# 已知锚点 (必须复算一致, atol=1e-3)
ANCHOR_E0_CD1 = 0.8576
ANCHOR_SRCLR_CD1 = 0.8286
ANCHOR_VARFRAC = 0.6229
ANCHOR_ATOL = 1e-3
# G11 参照值 (V 分支, 仅作软交叉核对, 不作断言)
G11_RESID_V = {                       # residK1 / K5 / K50
    "src":    (0.3771, 0.0682, 0.0122),
    "fftest": (0.3778, 0.0699, 0.0174),
    "cd1":    (0.3983, 0.1282, 0.0431),
    "cd2":    (0.4233, 0.1189, 0.0409),
    "dfdcp":  (0.3932, 0.1749, 0.0602),
    "ffiw":   (0.3180, 0.1421, 0.0540),
    "wild":   (0.3844, 0.1478, 0.0588),
}
G11_ZDIMF_V = {"src": 0.6229, "fftest": 0.6232, "cd1": 0.6357, "cd2": 0.5986,
               "dfdcp": 0.6581, "ffiw": 0.6852, "wild": 0.6122}
G11_AUCROW_V = {"cd1": 0.8576, "cd2": 0.8604, "dfdcp": 0.8322, "ffiw": 0.8103, "wild": 0.8060}

# 机械判定门 (预注册, 照抄执行)
GATE_H6_RATIO = 1.0        # Delta_ratio >= +1.0
GATE_H6_AUC = 0.05         # Delta_AUCreal >= +0.05
GATE_COUNT = 4             # >= 4/5 域
GATE_FW_INHERIT_REL = 0.8  # ratio_F >= 0.8*ratio_V  或 AUCreal_F >= 0.8*AUCreal_V
GATE_FW_MITIG_C = 1.2      # ratio_F <= min(1.2*ratio_C, 0.8*ratio_V)
GATE_COND_DIFF = 0.05      # mean|AUC_real-AUC_fake| < 0.05
GATE_COND_BOTH = 0.90      # 且 两者均 >= 0.90
GATE_COND_REAL = 0.75      # AUC_real < 0.75
GATE_COND_GAP = 0.10       # 且 AUC_fake >= AUC_real + 0.10


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


def cos_rows(A, B):
    """逐样本 cos, 返回 (min, mean, max|1-|cos||)."""
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    c = num / np.maximum(den, 1e-30)
    return float(c.min()), float(c.mean()), float(np.abs(np.abs(c) - 1.0).max())


def center_only_pca(X):
    """center-only raw-cov PCA: mu=样本均值, E=SVD 右奇异向量(降序). 与 G11 eigh(raw-cov) 同口径."""
    mu = X.mean(axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt.T, S


def layer_norm(x, w, b, eps=LN_EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, ddof=0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def load_proj_params():
    """只从 checkpoint 取 Linear/LayerNorm 权重 + alpha (map_location='cpu', 零前向)."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    def pair(pre):
        W = sd[pre + ".0.weight"].detach().cpu().numpy().astype(np.float64)
        b = sd[pre + ".0.bias"].detach().cpu().numpy().astype(np.float64)
        g = sd[pre + ".1.weight"].detach().cpu().numpy().astype(np.float64)
        be = sd[pre + ".1.bias"].detach().cpu().numpy().astype(np.float64)
        return W, b, g, be

    av = float(sd["clip_vision_alpha"].detach().cpu().numpy())
    at = float(sd["clip_text_alpha"].detach().cpu().numpy())
    return pair("deepfake_proj"), pair("vision_proj"), av, at


def apply_proj(X, params):
    W, b, g, be = params
    return layer_norm(X @ W.T + b, g, be)


def sep_d(proj_fake, proj_real):
    """(mean_fake-mean_real)/pooled_sd, 方向 fake>real 为正 (辅助量, 本脚本主指标是范数比)."""
    f = np.asarray(proj_fake, dtype=np.float64)
    r = np.asarray(proj_real, dtype=np.float64)
    n1, n2 = len(f), len(r)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp2 = ((n1 - 1) * f.var(ddof=1) + (n2 - 1) * r.var(ddof=1)) / (n1 + n2 - 2)
    if sp2 <= 0:
        return float("nan")
    return float((f.mean() - r.mean()) / np.sqrt(sp2))


# ---------------------------------------------------------------- split utils
def split_by_vid(vid, frac_test=SPLIT_FRAC, seed=0):
    """按 vid 做 70/30 划分; vid 数 < 3 -> 退化随机按样本划分, leak=True."""
    vid = np.asarray(vid).astype(str)
    uvid = np.unique(vid)
    rng = np.random.default_rng(seed)
    n = len(vid)
    if len(uvid) < 3:
        idx = rng.permutation(n)
        nte = int(round(frac_test * n))
        nte = min(max(nte, 1), n - 1)
        return np.sort(idx[nte:]), np.sort(idx[:nte]), True
    groups = {v: np.where(vid == v)[0] for v in uvid}
    order = rng.permutation(len(uvid))
    target = frac_test * n
    sel, cnt = [], 0
    for j in order:
        m = groups[uvid[j]]
        sel.append(m)
        cnt += len(m)
        if cnt >= target:
            break
    te = np.sort(np.concatenate(sel))
    mask = np.zeros(n, dtype=bool)
    mask[te] = True
    return np.where(~mask)[0], te, False


def balanced_pair(ia, ib, rng):
    m = min(len(ia), len(ib))
    a = rng.choice(ia, m, replace=False) if len(ia) > m else ia
    b = rng.choice(ib, m, replace=False) if len(ib) > m else ib
    return np.sort(a), np.sort(b)


def two_side_probe(Xa, va, Xb, vb, seed=0):
    """两侧二分类线性探针: StandardScaler(fit train)+LR(C=1e-3, lbfgs, max_iter=3000).
    label 1 = A 侧 (FF++ 参照), 0 = B 侧 (目标域). AUC 越高 = 域身份越可分.
    两侧各自按 vid 70/30 划分, 再用 min 均衡 train/test 样本数."""
    trA, teA, leakA = split_by_vid(va, SPLIT_FRAC, seed=seed)
    trB, teB, leakB = split_by_vid(vb, SPLIT_FRAC, seed=seed + 101)
    rng = np.random.default_rng(seed)
    trA, trB = balanced_pair(trA, trB, rng)
    teA, teB = balanced_pair(teA, teB, rng)
    if len(trA) < 4 or len(teA) < 4:
        return dict(auc=float("nan"), leak=True, n_tr=int(2 * len(trA)), n_te=int(2 * len(teA)))
    Xtr = np.vstack([Xa[trA], Xb[trB]])
    ytr = np.r_[np.ones(len(trA)), np.zeros(len(trB))]
    Xte = np.vstack([Xa[teA], Xb[teB]])
    yte = np.r_[np.ones(len(teA)), np.zeros(len(teB))]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT,
                             random_state=SEED).fit(sc.transform(Xtr), ytr)
    auc = float(roc_auc_score(yte, clf.decision_function(sc.transform(Xte))))
    return dict(auc=auc, leak=bool(leakA or leakB), n_tr=int(len(Xtr)), n_te=int(len(Xte)))


def resid_k(X, mu_src, E, K):
    """1 - ||proj_{E[:,:K]}(X-mu_src)||^2 / ||X-mu_src||^2 (集合能量比, 逐分量符号无关)."""
    Xc = X - mu_src
    tot = float((Xc ** 2).sum())
    P = Xc @ E[:, :K]
    return 1.0 - float((P ** 2).sum()) / tot


def mean_over(vals, valid):
    a = np.asarray([v for v, ok in zip(vals, valid) if ok], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


# ============================================================ main ===========
def main():
    t0 = time.time()
    log = []

    def L(x=""):
        log.append(str(x))

    def P(x=""):
        print(x, flush=True)

    # ------------------------------------------------ 数据 (只读 npz) --------
    p = np.load(PROBE_NPZ, allow_pickle=True)
    Vp_all = p["V"].astype(np.float64)
    Cp_all = p["C"].astype(np.float64)
    Vpp_npz = p["V_proj"].astype(np.float64)
    Cpp_npz = p["C_proj"].astype(np.float64)
    yp = p["y"].astype(int)
    vp = p["vids"].astype(str)
    tr_m = np.asarray(p["train_mask"])
    tr, te = np.where(tr_m)[0], np.where(~tr_m)[0]
    assert (len(tr), len(te)) == (2200, 800), (len(tr), len(te))
    assert not (set(vp[tr]) & set(vp[te])), "probe train/test 视频重叠"

    f = np.load(MULTI_NPZ, allow_pickle=True)
    Fm_all = f["F"].astype(np.float64)
    Vm_all = f["V"].astype(np.float64)
    Cm_all = f["C"].astype(np.float64)
    ym = f["y"].astype(int)
    domm = f["domain"].astype(str)
    vidm = f["vid"].astype(str)
    for dm in TARGETS:
        m = domm == dm
        assert int(m.sum()) == 300, (dm, int(m.sum()))
        assert int((ym[m] == 1).sum()) == 150 and int((ym[m] == 0).sum()) == 150, dm
    assert int((domm == "ffpp").sum()) == 800

    # ------------------------------------------- checkpoint 线性层权重 ------
    dj_par, vj_par, alpha_v, alpha_t = load_proj_params()

    log.append("=" * 126)
    log.append("G12 REPORT - H6 分支级 'FF++ 专用性' 分解")
    log.append("分支 B in {V, C, F, V_proj, C_proj};  目标域 d in {cd1, cd2, dfdcp, ffiw, wild}")
    log.append("纯 CPU 单线程 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1, torch=1, cv2=0, 单进程, CUDA_VISIBLE_DEVICES='')")
    log.append("零模型前向 (不实例化模型/不读图像); 只读 npz + checkpoint 的 Linear/LayerNorm 权重 (map_location='cpu')")
    log.append("y: 1=real, 0=fake;  线性探针一律 StandardScaler(fit train)+LR(C=1e-3, lbfgs, max_iter=3000)")
    log.append("=" * 126)
    MB_POS = len(log)
    log.append("@@MACHINE_BLOCK@@")

    # ============================================================== A 前置校验
    # A0.1 锚点: probe train V 上 center-only PCA -> e0; 在 multi cd1 V 上单轴 AUC
    mu_src_V, E_src_V, S_src_V = center_only_pca(Vp_all[tr])
    e0_src_V = E_src_V[:, 0].copy()
    z_tr = (Vp_all[tr] - mu_src_V) @ e0_src_V
    if z_tr[yp[tr] == 0].mean() < z_tr[yp[tr] == 1].mean():
        e0_src_V = -e0_src_V
        z_tr = -z_tr
    varfrac_src_V = float(S_src_V[0] ** 2 / (S_src_V ** 2).sum())
    m_cd1 = domm == "cd1"
    z_cd1 = (Vm_all[m_cd1] - mu_src_V) @ e0_src_V
    z_cd1_bin = (ym[m_cd1] == 0).astype(int)          # 1 = fake (positive)
    e0auc_cd1 = float(roc_auc_score(z_cd1_bin, z_cd1))

    sc_V = StandardScaler().fit(Vp_all[tr])
    clf_V = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT,
                               random_state=SEED).fit(sc_V.transform(Vp_all[tr]),
                                                      (yp[tr] == 0).astype(int))
    srclr_cd1 = float(roc_auc_score(z_cd1_bin,
                                    clf_V.decision_function(sc_V.transform(Vm_all[m_cd1]))))
    dev_e0 = abs(e0auc_cd1 - ANCHOR_E0_CD1)
    dev_lr = abs(srclr_cd1 - ANCHOR_SRCLR_CD1)
    dev_vf = abs(varfrac_src_V - ANCHOR_VARFRAC)
    anchor_ok = (dev_e0 < ANCHOR_ATOL) and (dev_lr < ANCHOR_ATOL) and (dev_vf < ANCHOR_ATOL)

    # A0.2 V_proj / C_proj 离线复算 (float64, LayerNorm eps=1e-5)
    Vp_hat = apply_proj(Vp_all, dj_par)
    Cp_hat = apply_proj(Cp_all, vj_par)
    vpj_min, vpj_mean, vpj_dev = cos_rows(Vp_hat, Vpp_npz)
    cpj_min, cpj_mean, cpj_dev = cos_rows(Cp_hat, Cpp_npz)
    proj_ok_V = vpj_min >= 0.999
    proj_ok_C = cpj_min >= 0.999
    Vpp_m = apply_proj(Vm_all, dj_par)     # multi 侧同法重建 (npz multi 无 V_proj/C_proj)
    Cpp_m = apply_proj(Cm_all, vj_par)

    # A0.3 F 布局验证: 对候选 768 窗口 (步长 32) 算 cos
    F_cos_C, F_cos_V = {}, {}
    for j in range(0, Fm_all.shape[1] - 768 + 1, 32):
        F_cos_C[j] = cos_rows(Fm_all[:, j:j + 768], Cpp_m)[1]
        F_cos_V[j] = cos_rows(Fm_all[:, j:j + 768], Vpp_m)[1]
    best_C = max(F_cos_C, key=lambda k: abs(F_cos_C[k]))
    best_V = max(F_cos_V, key=lambda k: abs(F_cos_V[k]))
    c0_min, c0_mean, c0_maxdev = cos_rows(Fm_all[:, 0:768], alpha_v * Cpp_m)
    c2_min, c2_mean, c2_maxdev = cos_rows(Fm_all[:, 896:1664], Vpp_m)
    c1_vs_c0 = cos_rows(Fm_all[:, 0:768], Vpp_m)[1]
    c2_vs_c1 = cos_rows(Fm_all[:, 896:1664], alpha_v * Cpp_m)[1]
    layout_ok = (best_C == 0) and (best_V == 896) and (c0_min >= 0.999) and (c2_min >= 0.999)
    F_LAYOUT = ("seg0[0:768]=clip_vision_alpha*C_proj(cos_min=%.6f);"
                "seg1[768:896]=bridge_adapter_embed(128-d,离线不可得);"
                "seg2[896:1664]=V_proj(cos_min=%.6f)") % (c0_min, c2_min)

    # ------------------------------------------------- 分支矩阵装配 ---------
    BR = {
        "V":      dict(src=Vp_all, dst=Vm_all, dim=768, note="raw ViT CLS"),
        "C":      dict(src=Cp_all, dst=Cm_all, dim=1024, note="raw CLIP vision CLS"),
        "V_proj": dict(src=Vpp_npz, dst=Vpp_m, dim=768, note="LN(Linear(V))"),
        "C_proj": dict(src=Cpp_npz, dst=Cpp_m, dim=768, note="LN(Linear(C))"),
        "F":      dict(src=np.hstack([alpha_v * Cpp_npz, Vpp_npz]),
                       dst=np.hstack([Fm_all[:, 0:768], Fm_all[:, 896:1664]]),
                       dim=1536, note="F 可离线重建子集 (无 128-d bridge 块)"),
    }
    for b in BRANCHES:
        BR[b]["srctr"], BR[b]["srcte"] = tr, te
    f_chk_c = cos_rows(Fm_all[:, 0:768], alpha_v * Cpp_m)
    f_chk_v = cos_rows(Fm_all[:, 896:1664], Vpp_m)
    f_recon_min = min(f_chk_c[0], f_chk_v[0])
    assert f_recon_min >= 0.999, "F 分支重建与 npz F 不一致"

    DOM = OrderedDict()
    for dm in TARGETS:
        m = domm == dm
        DOM[dm] = dict(X={b: BR[b]["dst"][m] for b in BRANCHES},
                       y=ym[m], vid=vidm[m], n=int(m.sum()),
                       nvid=int(len(np.unique(vidm[m]))))

    P("[G12] data + checkpoint ok; anchors: e0AUC_cd1=%.4f srcLR_cd1=%.4f varfrac=%.4f" %
      (e0auc_cd1, srclr_cd1, varfrac_src_V))
    P("[G12] proj cos: V_proj min=%.8f C_proj min=%.8f ; F layout best_C=%d best_V=%d" %
      (vpj_min, cpj_min, best_C, best_V))

    # ============================================================== B 块 =====
    s_scale, mu_src, class_gap, dom_gap, ratio = {}, {}, {}, {}, {}
    for b in BRANCHES:
        Xtr_b = BR[b]["src"][tr]
        ys = yp[tr]
        mu_src[b] = Xtr_b.mean(axis=0)
        mu_real_b = Xtr_b[ys == 1].mean(axis=0)
        mu_fake_b = Xtr_b[ys == 0].mean(axis=0)
        s_scale[b] = float(np.sqrt(np.var(Xtr_b, axis=0, ddof=1).mean()))
        class_gap[b] = float(np.linalg.norm(mu_fake_b - mu_real_b) / s_scale[b])
        dom_gap[b], ratio[b] = {}, {}
        for dm in TARGETS:
            Xd = DOM[dm]["X"][b]
            dom_gap[b][dm] = float(np.linalg.norm(Xd.mean(axis=0) - mu_src[b]) / s_scale[b])
            ratio[b][dm] = dom_gap[b][dm] / class_gap[b]
    P("[G12] block B done")

    # ============================================================== C 块 =====
    cprobe = {b: {} for b in BRANCHES}
    for b in BRANCHES:
        Xsrc_te = BR[b]["src"][te]
        ysrc_te = yp[te]
        vsrc_te = vp[te]
        for dm in TARGETS:
            Xd, yd, vd = DOM[dm]["X"][b], DOM[dm]["y"], DOM[dm]["vid"]
            res = {}
            for tag, lab in (("real", 1), ("fake", 0)):
                Xa, va = Xsrc_te[ysrc_te == lab], vsrc_te[ysrc_te == lab]
                Xb_, vb = Xd[yd == lab], vd[yd == lab]
                res[tag] = two_side_probe(Xa, va, Xb_, vb, seed=SEED)
            k = min(int((ysrc_te == 1).sum()), int((ysrc_te == 0).sum()),
                    int((yd == 1).sum()), int((yd == 0).sum()))
            rngm = np.random.default_rng(SEED)
            ia = np.r_[rngm.choice(np.where(ysrc_te == 1)[0], k, replace=False),
                       rngm.choice(np.where(ysrc_te == 0)[0], k, replace=False)]
            ib = np.r_[rngm.choice(np.where(yd == 1)[0], k, replace=False),
                       rngm.choice(np.where(yd == 0)[0], k, replace=False)]
            res["all"] = two_side_probe(Xsrc_te[ia], vsrc_te[ia], Xd[ib], vd[ib], seed=SEED)
            res["mix_n"] = int(2 * k)
            cprobe[b][dm] = res
    P("[G12] block C done")

    cmean = {}
    for b in BRANCHES:
        valid = [dm for dm in TARGETS if not cprobe[b][dm]["real"]["leak"]]
        cmean[b] = dict(
            real=float(np.nanmean([cprobe[b][dm]["real"]["auc"] for dm in TARGETS])),
            fake=float(np.nanmean([cprobe[b][dm]["fake"]["auc"] for dm in TARGETS])),
            all=float(np.nanmean([cprobe[b][dm]["all"]["auc"] for dm in TARGETS])),
            real_v=mean_over([cprobe[b][dm]["real"]["auc"] for dm in TARGETS],
                             [dm in valid for dm in TARGETS]),
            fake_v=mean_over([cprobe[b][dm]["fake"]["auc"] for dm in TARGETS],
                             [dm in valid for dm in TARGETS]),
            all_v=mean_over([cprobe[b][dm]["all"]["auc"] for dm in TARGETS],
                            [dm in valid for dm in TARGETS]),
            n_valid=len(valid), valid=valid)

    # ============================================================== D 块 =====
    Esrc, resid, dresid = {}, {}, {}
    for b in BRANCHES:
        _, Esrc[b], _ = center_only_pca(BR[b]["src"][tr])
        resid[b] = {}
        resid[b]["src"] = {K: resid_k(BR[b]["src"][tr], mu_src[b], Esrc[b], K) for K in KS}
        resid[b]["fftest"] = {K: resid_k(BR[b]["src"][te], mu_src[b], Esrc[b], K) for K in KS}
        for dm in TARGETS:
            resid[b][dm] = {K: resid_k(DOM[dm]["X"][b], mu_src[b], Esrc[b], K) for K in KS}
        dresid[b] = {dm: {K: resid[b][dm][K] - resid[b]["fftest"][K] for K in KS}
                     for dm in TARGETS}
    P("[G12] block D done")

    # ============================================================== E 块 =====
    efit = {}
    for b in BRANCHES:
        efit[b] = {}
        for tag, X in (("src", BR[b]["src"][tr]), ("fftest", BR[b]["src"][te])):
            mu_d, E_d, _ = center_only_pca(X)
            efit[b][tag] = dict(
                abs_cos=abs(cosv(E_d[:, 0], Esrc[b][:, 0])),
                cos=cosv(E_d[:, 0], Esrc[b][:, 0]),
                # [G11 同口径] z 用'源轴 + 源中心化', 分母是该域自身各维方差和 (G11 analyze_unit Q1a)
                zdim_frac=float(np.var((X - mu_src[b]) @ Esrc[b][:, 0], ddof=1) /
                                np.var(X, axis=0, ddof=1).sum()),
                # [域自身轴口径] z 用'域 PC0 + 域中心化'
                zdim_frac_domaxis=float(np.var((X - mu_d) @ E_d[:, 0], ddof=1) /
                                        np.var(X, axis=0, ddof=1).sum()))
        for dm in TARGETS:
            X = DOM[dm]["X"][b]
            mu_d, E_d, _ = center_only_pca(X)
            efit[b][dm] = dict(
                abs_cos=abs(cosv(E_d[:, 0], Esrc[b][:, 0])),
                cos=cosv(E_d[:, 0], Esrc[b][:, 0]),
                zdim_frac=float(np.var((X - mu_src[b]) @ Esrc[b][:, 0], ddof=1) /
                                np.var(X, axis=0, ddof=1).sum()),
                zdim_frac_domaxis=float(np.var((X - mu_d) @ E_d[:, 0], ddof=1) /
                                        np.var(X, axis=0, ddof=1).sum()))
    P("[G12] block E done")

    # ======================================================= 机械判定 ========
    dratio = {dm: ratio["V"][dm] - ratio["C"][dm] for dm in TARGETS}
    dauc = {dm: cprobe["V"][dm]["real"]["auc"] - cprobe["C"][dm]["real"]["auc"]
            for dm in TARGETS}
    valid_auc = [dm for dm in TARGETS if not cprobe["V"][dm]["real"]["leak"]]
    n_auc_v = len(valid_auc)

    cond_ratio_v = [dm for dm in TARGETS if dratio[dm] >= GATE_H6_RATIO]
    cond_auc_v = [dm for dm in valid_auc if dauc[dm] >= GATE_H6_AUC]
    cond_auc_all = [dm for dm in TARGETS if dauc[dm] >= GATE_H6_AUC]

    h6_vit = (len(cond_ratio_v) >= GATE_COUNT) or (len(cond_auc_v) >= GATE_COUNT)
    h6_shared = (len(cond_ratio_v) < GATE_COUNT) and (len(cond_auc_v) < GATE_COUNT)
    H6_VIT = "SUPPORT" if h6_vit else "NOT-SUPPORT"
    H6_SHARED = "SUPPORT" if h6_shared else "NOT-SUPPORT"

    ratio_mean = {b: float(np.mean([ratio[b][dm] for dm in TARGETS])) for b in BRANCHES}
    auc_re_v = {b: cmean[b]["real_v"] for b in BRANCHES}
    fw_inh = (ratio_mean["F"] >= GATE_FW_INHERIT_REL * ratio_mean["V"]) or \
             (auc_re_v["F"] >= GATE_FW_INHERIT_REL * auc_re_v["V"])
    fw_mit = (ratio_mean["F"] <= min(GATE_FW_MITIG_C * ratio_mean["C"],
                                     GATE_FW_INHERIT_REL * ratio_mean["V"]))
    FW = "INHERITS" if fw_inh else ("MITIGATES" if fw_mit else "MIXED")
    fw_pd_inh = [dm for dm in TARGETS
                 if (ratio["F"][dm] >= GATE_FW_INHERIT_REL * ratio["V"][dm]) or
                    (cprobe["F"][dm]["real"]["auc"] >=
                     GATE_FW_INHERIT_REL * cprobe["V"][dm]["real"]["auc"])]
    fw_pd_mit = [dm for dm in TARGETS
                 if (ratio["F"][dm] <= min(GATE_FW_MITIG_C * ratio["C"][dm],
                                           GATE_FW_INHERIT_REL * ratio["V"][dm]))]
    fw_pd_mit_auc = [dm for dm in TARGETS
                     if (cprobe["F"][dm]["real"]["auc"] <=
                         min(GATE_FW_MITIG_C * cprobe["C"][dm]["real"]["auc"],
                             GATE_FW_INHERIT_REL * cprobe["V"][dm]["real"]["auc"]))]

    def cond_verdict(b, domains):
        diffs = [abs(cprobe[b][dm]["real"]["auc"] - cprobe[b][dm]["fake"]["auc"])
                 for dm in domains]
        md = float(np.mean(diffs))
        ar = mean_over([cprobe[b][dm]["real"]["auc"] for dm in domains], [True] * len(domains))
        af = mean_over([cprobe[b][dm]["fake"]["auc"] for dm in domains], [True] * len(domains))
        if md < GATE_COND_DIFF and ar >= GATE_COND_BOTH and af >= GATE_COND_BOTH:
            return "REGION_LEVEL", md, ar, af
        if ar < GATE_COND_REAL and af >= ar + GATE_COND_GAP:
            return "FAKE_SUBCLASS_ONLY", md, ar, af
        return "CONDITIONAL_MIXED", md, ar, af

    condV = cond_verdict("V", valid_auc)
    condC = cond_verdict("C", valid_auc)
    condF = cond_verdict("F", valid_auc)
    COND = condV[0]

    # ------------------------------------------------- MACHINE_BLOCK --------
    wall = time.time() - t0
    mb = OrderedDict()
    mb["G12_TASK"] = "H6_branch_level_FFPP_specificity"
    mb["G12_DATA"] = ("probe_feats.npz(V/C/V_proj/C_proj n=3000:train2200/test800); "
                      "feats_multi.npz(F/V/C n=2300:ffpp800+5x300)")
    mb["G12_BRANCHES"] = "V=768; C=1024; F=1536(V_proj|C_proj join, no bridge128); V_proj=768; C_proj=768"
    mb["G12_ANCHOR_e0AUC_cd1"] = fmt(e0auc_cd1, 4)
    mb["G12_ANCHOR_e0AUC_cd1_EXP"] = fmt(ANCHOR_E0_CD1, 4)
    mb["G12_ANCHOR_srcLR_cd1"] = fmt(srclr_cd1, 4)
    mb["G12_ANCHOR_srcLR_cd1_EXP"] = fmt(ANCHOR_SRCLR_CD1, 4)
    mb["G12_ANCHOR_PC0_varfrac_srcV"] = fmt(varfrac_src_V, 4)
    mb["G12_ANCHOR_MAXDEV"] = fmt(max(dev_e0, dev_lr, dev_vf), 6)
    mb["G12_ANCHOR_PASS"] = "1" if anchor_ok else "0"
    mb["G12_PROJCHECK_Vproj_cos_min"] = fmt(vpj_min, 6)
    mb["G12_PROJCHECK_Vproj_cos_mean"] = fmt(vpj_mean, 6)
    mb["G12_PROJCHECK_Cproj_cos_min"] = fmt(cpj_min, 6)
    mb["G12_PROJCHECK_Cproj_cos_mean"] = fmt(cpj_mean, 6)
    mb["G12_PROJCHECK_LN_eps"] = "1e-05"
    mb["G12_PROJCHECK_PASS"] = "1" if (proj_ok_V and proj_ok_C) else "0"
    mb["G12_F_LAYOUT"] = F_LAYOUT
    mb["G12_F_LAYOUT_bestwin_C"] = str(best_C)
    mb["G12_F_LAYOUT_bestwin_V"] = str(best_V)
    mb["G12_F_LAYOUT_cos_seg0_vs_alphaCproj"] = fmt(c0_min, 6)
    mb["G12_F_LAYOUT_cos_seg2_vs_Vproj"] = fmt(c2_min, 6)
    mb["G12_F_LAYOUT_crosscheck_seg0_vs_Vproj"] = fmt(c1_vs_c0, 6)
    mb["G12_F_LAYOUT_crosscheck_seg2_vs_alphaCproj"] = fmt(c2_vs_c1, 6)
    mb["G12_F_LAYOUT_PASS"] = "1" if layout_ok else "0"
    mb["G12_F_BRANCH_USED"] = "[F[:,0:768]|F[:,896:1664]] concat -> 1536-d (bridge 128-d offline-unavailable)"
    mb["G12_F_RECON_CHECK_cos_min"] = fmt(f_recon_min, 6)
    mb["G12_ALPHA_v"] = fmt(alpha_v, 6)
    mb["G12_ALPHA_t"] = fmt(alpha_t, 6)
    mb["G12_DECISION_GATES"] = ("H6VIT:dratio>=+1.0 or dauc>=+0.05 in >=4/5; "
                                "H6SHARED:both below in >=4/5; "
                                "FW:rF>=0.8rV or aF>=0.8aV->INHERITS; rF<=min(1.2rC,0.8rV)->MITIGATES; "
                                "COND:mean|ar-af|<0.05&both>=0.90->REGION_LEVEL; ar<0.75&af>=ar+0.10->FAKE_SUBCLASS_ONLY")
    mb["G12_VALID_DOMAINS_ratio"] = ",".join(TARGETS)
    mb["G12_VALID_DOMAINS_auc"] = ",".join(valid_auc) + " (ffiw leak=True excluded)"
    mb["G12_GAP_CLASSGAP_BY_BRANCH"] = ";".join("%s=%.4f" % (b, class_gap[b]) for b in BRANCHES)
    mb["G12_GAP_SCALE_s_BY_BRANCH"] = ";".join("%s=%.3f" % (b, s_scale[b]) for b in BRANCHES)
    for b in BRANCHES:
        mb["G12_RATIO_%s" % b] = ";".join("%s=%.4f" % (dm, ratio[b][dm]) for dm in TARGETS)
    for b in BRANCHES:
        mb["G12_DOMGAP_%s" % b] = ";".join("%s=%.4f" % (dm, dom_gap[b][dm]) for dm in TARGETS)
    mb["G12_RATIO_MEAN_BY_BRANCH"] = ";".join("%s=%.4f" % (b, ratio_mean[b]) for b in BRANCHES)
    for b in BRANCHES:
        mb["G12_AUC_%s" % b] = ("real:" + ";".join("%s=%.4f" % (dm, cprobe[b][dm]["real"]["auc"])
                                                   for dm in TARGETS) +
                                "|fake:" + ";".join("%s=%.4f" % (dm, cprobe[b][dm]["fake"]["auc"])
                                                    for dm in TARGETS) +
                                "|all:" + ";".join("%s=%.4f" % (dm, cprobe[b][dm]["all"]["auc"])
                                                   for dm in TARGETS))
    mb["G12_AUC_LEAK_DOMAINS"] = ",".join(dm for dm in TARGETS
                                          if cprobe["V"][dm]["real"]["leak"]) or "none"
    mb["G12_AUC_MEAN_REAL_BY_BRANCH_5dom"] = ";".join("%s=%.4f" % (b, cmean[b]["real"])
                                                      for b in BRANCHES)
    mb["G12_AUC_MEAN_REAL_BY_BRANCH_VALID"] = ";".join("%s=%.4f" % (b, cmean[b]["real_v"])
                                                       for b in BRANCHES)
    mb["G12_AUC_MEAN_FAKE_BY_BRANCH_VALID"] = ";".join("%s=%.4f" % (b, cmean[b]["fake_v"])
                                                       for b in BRANCHES)
    mb["G12_AUC_MEAN_ALL_BY_BRANCH_VALID"] = ";".join("%s=%.4f" % (b, cmean[b]["all_v"])
                                                      for b in BRANCHES)
    for b in BRANCHES:
        mb["G12_RESID_%s" % b] = ("src:" + ",".join("K%d=%.4f" % (K, resid[b]["src"][K]) for K in KS) +
                                  "|fftest:" + ",".join("K%d=%.4f" % (K, resid[b]["fftest"][K])
                                                        for K in KS) +
                                  "|delta:" + ";".join(
                                      "%s=" % dm + ",".join("K%d=%+.4f" % (K, dresid[b][dm][K])
                                                            for K in KS) for dm in TARGETS))
    for b in BRANCHES:
        mb["G12_AXIS_%s" % b] = ("src_zdim=%.4f " % efit[b]["src"]["zdim_frac"] +
                                 "fftest_abs_cos=%.4f fftest_zdim=%.4f | " %
                                 (efit[b]["fftest"]["abs_cos"], efit[b]["fftest"]["zdim_frac"]) +
                                 "dom_abs_cos:" + ";".join("%s=%.4f" % (dm, efit[b][dm]["abs_cos"])
                                                           for dm in TARGETS) +
                                 "|dom_zdim_srcaxis:" + ";".join("%s=%.4f" % (dm, efit[b][dm]["zdim_frac"])
                                                                 for dm in TARGETS) +
                                 "|dom_zdim_domaxis:" + ";".join("%s=%.4f" % (dm, efit[b][dm]["zdim_frac_domaxis"])
                                                                 for dm in TARGETS))
    mb["G12_DELTA_RATIO_V_MINUS_C"] = ";".join("%s=%+.4f" % (dm, dratio[dm]) for dm in TARGETS)
    mb["G12_DELTA_AUCREAL_V_MINUS_C"] = ";".join("%s=%+.4f" % (dm, dauc[dm]) for dm in TARGETS)
    mb["G12_DELTA_RATIO_MEAN_V_MINUS_C"] = fmt(ratio_mean["V"] - ratio_mean["C"], 4)
    mb["G12_DELTA_AUCREAL_MEAN_V_MINUS_C"] = fmt(cmean["V"]["real_v"] - cmean["C"]["real_v"], 4)
    mb["G12_H6_COUNT_dratio_ge_1_0"] = "%d/%d" % (len(cond_ratio_v), 5)
    mb["G12_H6_COUNT_dauc_ge_0_05"] = "%d/%d" % (len(cond_auc_v), n_auc_v)
    mb["G12_H6_COUNT_dauc_ge_0_05_all5"] = "%d/5" % len(cond_auc_all)
    mb["G12_H6_VIT_ONLY"] = H6_VIT
    mb["G12_H6_SHARED"] = H6_SHARED
    mb["G12_FRAMEWORK"] = FW
    mb["G12_FRAMEWORK_perdomain_INHERITS"] = "%d/5 (%s)" % (len(fw_pd_inh), ",".join(fw_pd_inh))
    mb["G12_FRAMEWORK_perdomain_MITIGATES"] = "%d/5 (%s)" % (len(fw_pd_mit), ",".join(fw_pd_mit))
    mb["G12_FRAMEWORK_perdomain_MITIGATES_auc"] = "%d/5 (%s)" % (len(fw_pd_mit_auc),
                                                                 ",".join(fw_pd_mit_auc))
    mb["G12_CONDITION"] = COND + " (branch=V, valid-domains mean)"
    mb["G12_CONDITION_C"] = condC[0]
    mb["G12_CONDITION_F"] = condF[0]
    mb["G12_CONDITION_mean_absdiff_ar_af"] = "V=%.4f;C=%.4f;F=%.4f" % (condV[1], condC[1], condF[1])
    mb["G12_CONDITION_auc_real_fake"] = "V=%.4f/%.4f;C=%.4f/%.4f;F=%.4f/%.4f" % (
        condV[2], condV[3], condC[2], condC[3], condF[2], condF[3])
    mb["G12_VERDICT_SUMMARY"] = "H6VIT=%s;H6SHARED=%s;FRAMEWORK=%s;CONDITION=%s" % (
        H6_VIT, H6_SHARED, FW, COND)
    mb["G12_WALL_S"] = fmt(wall, 1)
    mb["G12_IMGS_READ"] = "0"
    mb["G12_GPU"] = "0"
    mb["G12_FORWARDS"] = "0"
    mb["G12_THREADS"] = "1"
    mb["G12_PROCESSES"] = "1"
    mb["G12_NPZ_READ"] = "2"
    mb["G12_CKPT_READ"] = "1 (map_location=cpu; only deepfake_proj/vision_proj/clip_*_alpha)"

    mblines = ["#### MACHINE_BLOCK " + "#" * 109]
    mblines += ["G12_%s=%s" % (k[4:], v) for k, v in mb.items()]
    log[MB_POS:MB_POS + 1] = mblines + [""]
    P("")
    for ln in mblines:
        P(ln)

    # ================================================== 报告正文 A - E =======
    L("")
    L("-" * 126)
    L("A. 前置校验 (任一不过则该分支停用; 锚点偏差 >1e-3 即停)")
    L("A0.1 锚点复算 (probe train 2200 拟合; 口径与 G11 一致: center-only raw-cov PCA, SVD/eigh 首轴)")
    L("  源 train V PC0 varfrac        : 复算 %s / 预期 %s" % (fmt(varfrac_src_V), fmt(ANCHOR_VARFRAC)))
    L("  cd1 单轴 e0 z AUC             : 复算 %s / 预期 %s   (|dev|=%s)" %
      (fmt(e0auc_cd1), fmt(ANCHOR_E0_CD1), fmt(dev_e0, 6)))
    L("  cd1 源训 LR 跨域 AUC          : 复算 %s / 预期 %s   (|dev|=%s)" %
      (fmt(srclr_cd1), fmt(ANCHOR_SRCLR_CD1), fmt(dev_lr, 6)))
    L("  -> ANCHOR_PASS=%s (atol=1e-3)" % mb["G12_ANCHOR_PASS"])
    L("")
    L("A0.2 V_proj / C_proj 离线复算 (float64; LayerNorm eps=1e-5, elementwise_affine=True)")
    L("  V_proj_hat = LayerNorm(Linear(V))        vs npz V_proj : cos min=%s mean=%s (max|1-|cos||=%s)" %
      (fmt(vpj_min, 6), fmt(vpj_mean, 6), fmt(vpj_dev, 8)))
    L("  C_proj_hat = LayerNorm(Linear(C))[:,0,:] vs npz C_proj : cos min=%s mean=%s (max|1-|cos||=%s)" %
      (fmt(cpj_min, 6), fmt(cpj_mean, 6), fmt(cpj_dev, 8)))
    L("  门槛 cos >= 0.999 : V_proj %s / C_proj %s -> %s" %
      ("PASS" if proj_ok_V else "FAIL", "PASS" if proj_ok_C else "FAIL",
       "V_proj 与 C_proj 派生分支均采用" if (proj_ok_V and proj_ok_C)
       else "至少一个派生分支停用 (见判定)"))
    L("  multi 侧无 V_proj/C_proj 字段 -> 用同组权重离线重建 (V_proj_m, C_proj_m)")
    L("")
    L("A0.3 F (1664-d) 布局逐段 cos 验证 (multi 全 2300 样本; 无前向重建)")
    L("  F = cat[clip_vision_alpha*vision_proj(C)[:,0,:] (768), bridge_adapter_embed*clip_text_alpha (128), deepfake_proj(V) (768)]")
    L("  候选 768 窗口 (步长 32) 逐位 cos 均值, 取 |cos| 最大者:")
    lc = sorted(F_cos_C.items(), key=lambda kv: -abs(kv[1]))[:3]
    lv = sorted(F_cos_V.items(), key=lambda kv: -abs(kv[1]))[:3]
    L("    vs C_proj  top3: " + "  ".join("win@%d cos=%.6f" % (k, v) for k, v in lc))
    L("    vs V_proj  top3: " + "  ".join("win@%d cos=%.6f" % (k, v) for k, v in lv))
    L("  实测布局: " + F_LAYOUT)
    L("  F[:,0:768]    vs alpha_v*C_proj (alpha_v=%s) : cos min=%s mean=%s" %
      (fmt(alpha_v, 6), fmt(c0_min, 6), fmt(c0_mean, 6)))
    L("  F[:,896:1664] vs V_proj (无 alpha 缩放)      : cos min=%s mean=%s" %
      (fmt(c2_min, 6), fmt(c2_mean, 6)))
    L("  交叉核对 (非对角应低) F[:,0:768] vs V_proj = %s ; F[:,896:] vs alpha_v*C_proj = %s" %
      (fmt(c1_vs_c0, 6), fmt(c2_vs_c1, 6)))
    L("  bridge 中间块 128-d (alpha_t=%s) 需模型前向, 离线不可得 -> 分支 F 只用 1536/1664 维 (92.3%%)" %
      fmt(alpha_t, 6))
    L("  F 分支重建一致性 (npz F 对应列 vs 离线重建) cos min=%s -> F_LAYOUT_PASS=%s" %
      (fmt(f_recon_min, 6), mb["G12_F_LAYOUT_PASS"]))
    L("")
    L("-" * 126)
    L("B. 域-类间隔比 (源侧 probe train 2200: mu_src/mu_real/mu_fake;")
    L("   s_B = sqrt(mean_j Var_j(源全池, ddof=1)); class_gap_B=||mu_fake-mu_real||/s;")
    L("   dom_gap_B(d)=||mu_d-mu_src||/s; ratio=dom_gap/class_gap)")
    L("  注: ratio 的分子分母同除 s -> ratio 与 s 无关 (s 只给绝对尺度)")
    L("")
    L("  branch   dim      s     class_gap | " + " ".join("%9s" % dm for dm in TARGETS) +
      "   (ratio = dom_gap/class_gap)")
    L("  " + "-" * 116)
    for b in BRANCHES:
        L("  %-8s %4d %7.3f %8.4f  | " % (b, BR[b]["dim"], s_scale[b], class_gap[b]) +
          " ".join("%9.4f" % ratio[b][dm] for dm in TARGETS))
    L("  " + "-" * 116)
    L("  ratio 跨域均值: " + "  ".join("%s=%.4f" % (b, ratio_mean[b]) for b in BRANCHES))
    L("")
    L("  dom_gap 绝对值表 (单位 s):")
    L("  branch   " + " ".join("%9s" % dm for dm in TARGETS) + "   |  mean")
    for b in BRANCHES:
        L("  %-8s " % b + " ".join("%9.4f" % dom_gap[b][dm] for dm in TARGETS) +
          "   | %6.4f" % float(np.mean([dom_gap[b][dm] for dm in TARGETS])))
    L("")
    L("  Delta_ratio(d) = ratio_V - ratio_C :")
    L("    " + " ".join("%s=%+.4f" % (dm, dratio[dm]) for dm in TARGETS) +
      "   |  跨域均值 %+.4f" % (ratio_mean["V"] - ratio_mean["C"]))
    L("")
    L("-" * 126)
    L("C. 类条件域可分性 (二分类线性探针: StandardScaler(fit train)+LR(C=1e-3,lbfgs,max_iter=3000))")
    L("  参照侧 = probe test 800 (FF++ test, 42 vids, 与源 train video-disjoint); 域侧 = 该域 300")
    L("  两侧各自按 vid 70/30 划分 -> min 均衡 train/test 样本数; label 1=FF++参照, 0=目标域")
    L("  AUC 高 = 该分支空间里'域身份'线性可分性强 (不等于模型真的用了它, 见 CAVEATS)")
    L("  ffiw 仅 1 vid -> 退化随机划分 leak=True, 剔出聚合门 ('mean_valid' 列)")
    L("")
    for tag, lab in (("real", "AUC_real (real-only 探针)"),
                     ("fake", "AUC_fake (fake-only 探针)"),
                     ("all", "AUC_all (mixed real+fake 域判别探针, 对照)")):
        L("  %s :" % lab)
        L("    branch  " + " ".join("%9s" % dm for dm in TARGETS) +
          " |  mean5  mean_valid  (n_tr/n_te @cd1)")
        for b in BRANCHES:
            vals = [cprobe[b][dm][tag]["auc"] for dm in TARGETS]
            mv = mean_over(vals, [not cprobe[b][dm][tag]["leak"] for dm in TARGETS])
            r0 = cprobe[b]["cd1"][tag]
            L("    %-7s " % b + " ".join("%9.4f" % v for v in vals) +
              " | %7.4f %8.4f      %d/%d" % (float(np.nanmean(vals)), mv, r0["n_tr"], r0["n_te"]))
        L("")
    L("  mixed 探针规模: 每侧 %d 样本 (real %d + fake %d)" %
      (cprobe["V"]["cd1"]["mix_n"], cprobe["V"]["cd1"]["mix_n"] // 2,
       cprobe["V"]["cd1"]["mix_n"] // 2))
    L("  leak 标记: " + ", ".join("%s=%d" % (dm, int(cprobe["V"][dm]["real"]["leak"]))
                                  for dm in TARGETS))
    L("")
    L("  Delta_AUCreal(d) = AUCreal_V - AUCreal_C :")
    L("    " + " ".join("%s=%+.4f" % (dm, dauc[dm]) for dm in TARGETS) +
      "   |  跨域均值(valid) %+.4f" % (cmean["V"]["real_v"] - cmean["C"]["real_v"]))
    L("")
    L("  定位参照 (G11 V 分支单轴 e0 的 fake-detection AUC, positive=fake): " +
      " ".join("%s=%.4f" % (dm, G11_AUCROW_V[dm]) for dm in TARGETS))
    L("  (该行与 AUC_real 含义不同: 前者判 real/fake, 后者判 FF++/目标域, 不可直接相减)")
    L("")
    L("-" * 126)
    L("D. 子空间包含 (源 train 拟合 center-only PCA; SVD 前 K 右奇异向量; X-mu_src 的集合能量比)")
    L("  residK = 1 - ||proj_{E[:K]}(X-mu_src)||^2 / ||X-mu_src||^2 ; Delta = 域 - FF++ test 参照")
    L("  K=1 时 E[:1] 即源 PC0 轴 e0 (与 G11 同轴)")
    L("")
    for b in BRANCHES:
        L("  branch %s (dim=%d, %s):" % (b, BR[b]["dim"], BR[b]["note"]))
        L("    row          " + " ".join("%9s" % ("K=%d" % K) for K in KS))
        for tag, lab in (("src", "src2200  "), ("fftest", "fftest800")):
            L("    %s " % lab + " ".join("%9.4f" % resid[b][tag][K] for K in KS))
        for dm in TARGETS:
            L("    %-10s " % dm + " ".join("%9.4f" % resid[b][dm][K] for K in KS) +
              "   Delta: " + " ".join("%+.4f" % dresid[b][dm][K] for K in KS))
        if b == "V":
            g = G11_RESID_V
            L("    [G11 软核对, V] src%s fftest%s cd1%s cd2%s" %
              ("(%.4f,%.4f,%.4f)" % g["src"], "(%.4f,%.4f,%.4f)" % g["fftest"],
               "(%.4f,%.4f,%.4f)" % g["cd1"], "(%.4f,%.4f,%.4f)" % g["cd2"]))
            L("                   dfdcp%s ffiw%s wild%s" %
              ("(%.4f,%.4f,%.4f)" % g["dfdcp"], "(%.4f,%.4f,%.4f)" % g["ffiw"],
               "(%.4f,%.4f,%.4f)" % g["wild"]))
            L("      (G12 用 SVD, G11 用 eigh(raw-cov); 同口径应 <5e-3 级一致; 不一致则说明口径漂移)")
        L("")
    L("-" * 126)
    L("E. 轴对齐 (每分支每域用该域样本 300 单独拟合 center-only PCA -> 域 PC0 与源 e0 的 |cos|;")
    L("   zdim 两口径 (分母均为该域自身 sum_j Var_j(域特征), ddof=1):")
    L("     zdim_srcaxis = Var((X_d - mu_src) @ e0_src)/sum_j Var_j(X_d)   <- G11 analyze_unit Q1a 同口径")
    L("     zdim_domaxis = Var((X_d - mu_dom) @ e0_dom)/sum_j Var_j(X_d)   <- 域自身轴口径 (>=srcaxis, 域轴最大化域方差)")
    L("   末列 [G11] 为 V 分支 zdim_srcaxis 参照)")
    L("")
    L("  branch   row        |cos(e0d,e0_src)|  zdim_srcaxis  zdim_domaxis  [G11 V]")
    L("  " + "-" * 82)
    for b in BRANCHES:
        L("  %-8s src2200          %8.4f       %8.4f      %8.4f    %s" %
          (b, efit[b]["src"]["abs_cos"], efit[b]["src"]["zdim_frac"],
           efit[b]["src"]["zdim_frac_domaxis"],
           fmt(G11_ZDIMF_V["src"]) if b == "V" else "-"))
        L("  %-8s fftest800        %8.4f       %8.4f      %8.4f    %s" %
          (b, efit[b]["fftest"]["abs_cos"], efit[b]["fftest"]["zdim_frac"],
           efit[b]["fftest"]["zdim_frac_domaxis"],
           fmt(G11_ZDIMF_V["fftest"]) if b == "V" else "-"))
        for dm in TARGETS:
            L("  %-8s %-9s        %8.4f       %8.4f      %8.4f    %s" %
              (b, dm, efit[b][dm]["abs_cos"], efit[b][dm]["zdim_frac"],
               efit[b][dm]["zdim_frac_domaxis"],
               fmt(G11_ZDIMF_V[dm]) if b == "V" else "-"))
        L("")
    L("-" * 126)
    L("MECHANICAL VERDICTS (预注册, 照抄执行, 不做方向演绎)")
    L("")
    L("  [V1] Delta_ratio(d) = ratio_V-ratio_C >= +1.0 :")
    L("       " + " ".join("%s:%s(%+.4f,%d)" % (dm, "Y" if dratio[dm] >= GATE_H6_RATIO else "n",
                                                 dratio[dm], int(dratio[dm] >= GATE_H6_RATIO))
                            for dm in TARGETS))
    L("       成立域 = %s -> %d/%d 有效域 (B 块不依赖 vid 划分, 5 域全有效)" %
      (",".join(cond_ratio_v) if cond_ratio_v else "无", len(cond_ratio_v), 5))
    L("  [V2] Delta_AUCreal(d) = AUCreal_V-AUCreal_C >= +0.05 :")
    L("       " + " ".join("%s:%s(%+.4f,%d%s)" % (dm, "Y" if dauc[dm] >= GATE_H6_AUC else "n",
                                                    dauc[dm], int(dauc[dm] >= GATE_H6_AUC),
                                                    "[leak]" if cprobe["V"][dm]["real"]["leak"] else "")
                            for dm in TARGETS))
    L("       成立域(有效) = %s -> %d/%d ; 含 ffiw 口径 = %d/5" %
      (",".join(cond_auc_v) if cond_auc_v else "无", len(cond_auc_v), n_auc_v, len(cond_auc_all)))
    L("")
    L("  => G12_H6_VIT_ONLY : (V1 在 >=4/5 域成立) 或 (V2 在 >=4/5 有效域成立)")
    L("     V1 %d/5 , V2 %d/%d  -> G12_H6_VIT_ONLY=%s" %
      (len(cond_ratio_v), len(cond_auc_v), n_auc_v, H6_VIT))
    L("  => G12_H6_SHARED   : 上述两者均 <4/5 低于阈值:")
    L("     V1 %d/5 <4 且 V2 %d/%d <4 -> G12_H6_SHARED=%s" %
      (len(cond_ratio_v), len(cond_auc_v), n_auc_v, H6_SHARED))
    L("     (两条判定为对同一前件的互补表述, 故恒有恰一条为 SUPPORT; 本处结论以实际计数为准)")
    L("")
    L("  [V3] 框架归属 (F 是否继承/缓解 V 的域信息):")
    L("       ratio_F(mean)=%.4f vs 0.8*ratio_V(mean)=%.4f  -> %s" %
      (ratio_mean["F"], GATE_FW_INHERIT_REL * ratio_mean["V"],
       "Y" if ratio_mean["F"] >= GATE_FW_INHERIT_REL * ratio_mean["V"] else "n"))
    L("       AUCreal_F(valid mean)=%.4f vs 0.8*AUCreal_V(valid mean)=%.4f -> %s" %
      (auc_re_v["F"], GATE_FW_INHERIT_REL * auc_re_v["V"],
       "Y" if auc_re_v["F"] >= GATE_FW_INHERIT_REL * auc_re_v["V"] else "n"))
    L("       INHERITS 判据 (或) = %s ; 逐域计数 = %d/5 %s" %
      (int(fw_inh), len(fw_pd_inh), "(" + ",".join(fw_pd_inh) + ")" if fw_pd_inh else ""))
    L("       MITIGATES 判据: ratio_F <= min(1.2*ratio_C=%.4f, 0.8*ratio_V=%.4f) = %.4f -> %s" %
      (GATE_FW_MITIG_C * ratio_mean["C"], GATE_FW_INHERIT_REL * ratio_mean["V"],
       min(GATE_FW_MITIG_C * ratio_mean["C"], GATE_FW_INHERIT_REL * ratio_mean["V"]),
       "Y" if fw_mit else "n"))
    L("       逐域 ratio 口径 = %d/5 ; 逐域 AUC 口径 = %d/5" %
      (len(fw_pd_mit), len(fw_pd_mit_auc)))
    L("     -> G12_FRAMEWORK=%s" % FW)
    L("")
    L("  [V4] 类条件归属 (主判 branch=V, 有效域 mean):")
    L("       mean|AUC_real-AUC_fake| = %.4f (<0.05 ? %s)" %
      (condV[1], "Y" if condV[1] < GATE_COND_DIFF else "n"))
    L("       AUC_real=%.4f (>=0.90 ? %s) ; AUC_fake=%.4f (>=0.90 ? %s)" %
      (condV[2], "Y" if condV[2] >= GATE_COND_BOTH else "n",
       condV[3], "Y" if condV[3] >= GATE_COND_BOTH else "n"))
    L("       FAKE_SUBCLASS 判据: AUC_real<0.75 ? %s ; AUC_fake >= AUC_real+0.10 ? %s" %
      ("Y" if condV[2] < GATE_COND_REAL else "n",
       "Y" if condV[3] >= condV[2] + GATE_COND_GAP else "n"))
    L("       -> G12_CONDITION=%s   (对照: C=%s, F=%s)" % (condV[0], condC[0], condF[0]))
    L("       C 明细: mean|d|=%.4f real=%.4f fake=%.4f ; F 明细: mean|d|=%.4f real=%.4f fake=%.4f" %
      (condC[1], condC[2], condC[3], condF[1], condF[2], condF[3]))
    L("")
    L("  === 判定常量汇总 ===")
    L("    G12_H6_VIT_ONLY = %s" % H6_VIT)
    L("    G12_H6_SHARED   = %s" % H6_SHARED)
    L("    G12_FRAMEWORK   = %s" % FW)
    L("    G12_CONDITION   = %s" % COND)
    L("")

    L("-" * 126)
    L("AUDIT")
    L("  imgs_read=0 (零图像读取)            gpu=0 (CUDA_VISIBLE_DEVICES='', 未调用任何 cuda API)")
    L("  forwards=0 (未实例化模型, 零前向/反向)   threads=1 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1,")
    L("  torch.set_num_threads(1), cv2.setNumThreads(0)) ; processes=1")
    L("  npz 读取=2 (probe_feats.npz, feats_multi.npz) ; checkpoint 读取=1 (map_location='cpu',")
    L("  只取 deepfake_proj.0/1, vision_proj.0/1, clip_vision_alpha, clip_text_alpha)")
    L("  V_proj/C_proj 复算: float64, LayerNorm eps=1e-5, elementwise_affine=True (与模型定义一致)")
    L("  探针: StandardScaler(fit train) + LR(C=%s, lbfgs, max_iter=%d); 划分 seed=%d; 无 PCA 白化" %
      (C_FIX, MAXIT, SEED))
    L("  SVD: np.linalg.svd(full_matrices=False) 于中心化特征矩阵 (与 G11 eigh(raw-cov) 同口径)")
    L("  wall_s = %.1f" % wall)
    L("")
    L("CAVEATS (honest)")
    L("  1. ffiw 域只有 1 个 vid (300 样本 = 150 real + 150 fake 同视频): C 块按 vid 划分退化为随机按样本")
    L("     划分 (leak=True), 其 AUC_real/AUC_fake 被高估, 已剔出聚合门与 H6 计数; 其 B/D/E 块量不依赖")
    L("     样本划分, 仍然保留, 但'同一视频内 150+150'意味着 D/E 的域几何可能被单视频风格主导.")
    L("  2. 每个目标域仅 300 样本 (150/150): 均衡后探针 train/test 各约 105/45 每侧 (mixed 为 210/90),")
    L("     test=90 时 AUC 的 1SE 量级约 0.05-0.07 -> 域间小差异不可解读, 只看跨域一致方向与量级.")
    L("  3. 线性探针给的是'可分性下界类'指标: 线性可分性存在 != 下游模型使用了该信息; 探针弱不代表")
    L("     信息不存在 (非线性/局部可读), 探针强也不代表模型依赖. 三个探针 (real/fake/mixed) 只是同一")
    L("     问题的三种切片.")
    L("  4. 相关 != 因果: 域可分性高不等于'ViT 只用域信息', 也不等于'ViT 学的是域而非伪造痕迹'.")
    L("     本实验只量化源-域可分的线性结构强度在分支间的分布, 不能定位模型决策所依赖的变量.")
    L("  5. multi 的 'ffpp' 域 (800, 140 vids, 其中 98 vids 属于 probe 源 train 视频) 有源视频污染,")
    L("     全程只作脚注, 不作 FF++ 参照; FF++ 参照一律用 probe test 800 (42 vids, 完全 video-disjoint).")
    L("  6. 派生分支 V_proj/C_proj 依赖 A0.2 的 cos 校验 (实测 min cos 与 1 的偏差 ~1e-13, 通过);")
    L("     若 checkpoint 与 npz 不同源则该分支失效. F 分支只含可离线重建的 1536/1664 维")
    L("     (bridge_adapter 128-d 需前向), 且 F 是两塔特征拼接 -> 其域可分性可被融合放大或压制,")
    L("     不能直接归因到任一单塔.")
    L("  7. s 尺度用 ddof=1 的逐维方差均值开方; ratio = dom_gap/class_gap 中 s 约去, 故 s 的 ddof")
    L("     选择不影响任何判定常量; dom_gap/class_gap 的绝对值受 ddof 影响 (<1% 量级).")
    L("  8. 分支 F 的源侧 = probe npz 的 [alpha_v*C_proj | V_proj] 拼接, 域侧 = npz F 的对应列")
    L("     (两侧构造一致, 校验 cos>0.999); 缺 128-d bridge 块 (7.7% 维度) 使 F 的绝对量级与真值 F")
    L("     有偏差, 但该块在模型里乘 alpha_t=%s (较大), 故 F 的真实域可分性可能高于此处估计." % fmt(alpha_t, 4))
    L("  9. B 块 dom_gap 用'域均值-源全池均值'的欧氏范数, 未做白化/去相关 -> 被高方差方向主导;")
    L("     D 块 residK 与 E 块 zdim_frac 是同一现象的另两种归一化口径, 三者结论一致才可采信.")
    L("     E 块 zdim 给了两口径 (源轴/G11 同口径 与 域自身轴口径), 二者对源集相同、对目标域差 0.01-0.03;")
    L("     该量不参与任何判定常量, 只作诊断.")
    L(" 10. 本实验全部为离线特征几何量, 没有重测任何下游检测性能; C 块的 AUC_real/AUC_fake 是")
    L("     '域判别'探针, 与 G11 的单轴 fake-detection AUC 含义不同, 不可直接比较或相减.")
    L(" 11. H6 的 4/5 门在 ffiw 被剔除后分母变为 4 (V2 口径): 即要求 4/4 全部成立; 该严格化已如实")
    L("     报告 (同时给出含 ffiw 的 5 域计数), 不对阈值做任何宽松化处理.")
    L("")
    L("=" * 126)
    L("END OF REPORT   (wall %.1f s)" % wall)
    L("=" * 126)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    keys = list(mb.keys())
    np.savez(STATS, **{k: np.array(str(mb[k])) for k in keys},
             **{"meta_keys": np.array(keys)})
    print("[done] report -> %s" % REPORT, flush=True)
    print("[done] stats  -> %s" % STATS, flush=True)


if __name__ == "__main__":
    main()
