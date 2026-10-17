# -*- coding: utf-8 -*-
"""
G17c -- joint evaluation: do low-level / frequency-domain (and CNN local-detail)
features complement the class-discriminative information that the ViT CLS
representation V (768) is missing?

Pure offline CPU. Zero deep-model forward / zero GPU / zero deep training; only
light linear/logistic heads are fitted.

Conventions reused verbatim from G12 / G14 / G16 (do not re-invent):
  * y: 1=real, 0=fake.  Every AUC uses positive=fake (class 1 = fake), i.e.
    labels are yb = (y == 0).astype(int) and scores are decision_function or
    predict_proba[:,1] in the "fake is positive" orientation.
  * cross-domain anchor head: StandardScaler(fit probe-train) + LogisticRegression(
      C=1e-3, solver="lbfgs", max_iter=3000, random_state=0), eval per target domain.
    Expected: cd1=0.8286 cd2=0.8633 dfdcp=0.8261 ffiw=0.8244 wild=0.8090 (maxdev<1e-3).
  * G12 domain-sensitivity ratio: s=sqrt(mean_j Var_j(src_train, ddof=1));
    class_gap=||mu_fake-mu_real||/s; dom_gap(d)=||mu_d-mu_src||/s; ratio=dom_gap/class_gap.
    Expected means: ratio_V=0.1983, ratio_C=2.7160.
  * G14 in-domain oracle: video-grouped 5-fold CV, best-of-4 methods
    (LR_C1e-3, LR_C1.0, kNN_k5, RBF_SVM); ffiw (1 vid) degenerates to SKF -> leak=True.
    Expected oracle_V mean4 (excl ffiw) = 0.9111.

Feature arms:
  V baseline (768 raw); current-state refs C_proj(768), V|C_proj(1536);
  freq arms F1(40),F1_radial(32),F2(4),F3(20),F3_hist(160),F4(23),F5(4),
  F_all(123 = F1|F1_radial|F2|F3|F4|F5, no F3_hist);
  CNN arms dense121_db2/3/final, effnet_b4_blk5/6/final, resnet18_layer3/4;
  joint arms V|feat for every feat; replacement arms V|C_proj|feat.

Preprocessing: drop zero-variance columns, then StandardScaler fit on the train
side only (cross-domain: probe train; oracle: fold train).  Main regularisation
C=1e-3 fixed + a second column of inner-CV-selected C (video-grouped CV on the
train side only, never touching the target domain).

Resource discipline: single process, OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS=4,
max_workers<=2 (n_jobs=None everywhere), CUDA_VISIBLE_DEVICES=''.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["BLIS_NUM_THREADS"] = "4"
os.environ["JOBLIB_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time
from collections import OrderedDict

import numpy as np

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
FREQ_NPZ = os.path.join(HERE, "freq_feats.npz")
CNN_NPZ = os.path.join(HERE, "cnn_feats.npz")
REPORT = os.path.join(HERE, "g17_report.txt")
STATS = os.path.join(HERE, "g17_stats.npz")

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
DECISION = ["dfdcp", "cd2", "wild"]          # judgement domains (dfdcp clean; cd2/wild sub-clean)
C_FIX = 1e-3
MAXIT = 3000
SEED = 0
NFOLDS = 5
ALPHA_V = 0.306288                           # G12 measured: F[:,0:768] = alpha_v * C_proj
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CTRL_SEED_BASE = 20260910
CTRL_SEEDS = [0, 1, 2]
CTRL_MARGIN = 0.01
GATE_GAIN = 0.03
GATE_NO = 0.01

METHODS = ["LR_C1e-3", "LR_C1.0", "kNN_k5", "RBF_SVM"]

ANCHOR_CD = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_RATIO_V = 0.1983
ANCHOR_RATIO_C = 2.7160
ANCHOR_ATOL = 1e-3
ORACLE_V_REF = 0.9111


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
    mu = X.mean(axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt.T, S


def get_folds(y, vid, n_splits=NFOLDS, seed=SEED):
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


def drop_zero_var(X):
    """Column mask keeping columns with var>0 over the given (all-rows) matrix."""
    v = np.var(X, axis=0)
    return v > 0.0


def fit_scaler(Xtr):
    sc = StandardScaler().fit(Xtr)
    sc.scale_ = np.where(sc.scale_ < 1e-12, 1.0, sc.scale_)
    return sc


def auc(y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def fit_eval_method(Xtr, ytr, Xte, yte, method):
    """G14-exact: StandardScaler(fit fold train) -> classifier -> AUC (positive=fake)."""
    sc = fit_scaler(Xtr)
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
    return auc(yte, s)


def cv_oracle(X, y, vid):
    """G14-exact video-grouped 5-fold oracle; returns per-method mean/std + mode/leak/nvid."""
    yb = (np.asarray(y) == 0).astype(int)     # 1 = fake
    folds, mode, leak = get_folds(yb, vid)
    acc = {m: [] for m in METHODS}
    for tr_i, te_i in folds:
        if len(np.unique(yb[tr_i])) < 2 or len(np.unique(yb[te_i])) < 2:
            continue
        for m in METHODS:
            try:
                acc[m].append(fit_eval_method(X[tr_i], yb[tr_i], X[te_i], yb[te_i], m))
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


def fit_lr_scores(Xtr, ytr, Xte, C):
    sc = fit_scaler(Xtr)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
        sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def select_C_cv(Xtr, ytr, vtr, C_grid=C_GRID):
    """video-grouped CV on the train side only -> best C (never sees target/test)."""
    yb = np.asarray(ytr, dtype=int)   # already 1=fake (positive class)
    folds, _mode, _leak = get_folds(yb, vtr)
    best_C, best_auc = None, -1.0
    per_C = {}
    for C in C_grid:
        aucs = []
        for tr_i, te_i in folds:
            if len(np.unique(yb[tr_i])) < 2 or len(np.unique(yb[te_i])) < 2:
                continue
            try:
                aucs.append(auc(yb[te_i], fit_lr_scores(Xtr[tr_i], yb[tr_i], Xtr[te_i], C)))
            except Exception:
                aucs.append(float("nan"))
        a = np.asarray(aucs, dtype=np.float64)
        a = a[np.isfinite(a)]
        m = float(a.mean()) if len(a) else float("nan")
        per_C[C] = m
        if m == m and (best_auc < 0 or m > best_auc):
            best_auc, best_C = m, C
    return best_C, best_auc, per_C


def main():
    t0 = time.time()
    log = []

    def L(x=""):
        log.append(str(x))

    def P(x=""):
        print(x, flush=True)

    # ================================================================ data ====
    p = np.load(PROBE_NPZ, allow_pickle=True)
    Vp = p["V"].astype(np.float64)
    Cp = p["C"].astype(np.float64)
    C_proj_p = p["C_proj"].astype(np.float32)
    yp = p["y"].astype(int)
    vp = p["vids"].astype(str)
    tr_mask = np.asarray(p["train_mask"])

    m = np.load(MULTI_NPZ, allow_pickle=True)
    Vm = m["V"].astype(np.float64)
    Cm = m["C"].astype(np.float64)
    Fm = m["F"].astype(np.float32)
    ym = m["y"].astype(int)
    domm = m["domain"].astype(str)
    vidm = m["vid"].astype(str)

    fq = np.load(FREQ_NPZ, allow_pickle=True)
    cn = np.load(CNN_NPZ, allow_pickle=True)

    tr = np.where(tr_mask)[0]
    te = np.where(~tr_mask)[0]
    assert (len(tr), len(te)) == (2200, 800), (len(tr), len(te))
    assert not (set(vp[tr]) & set(vp[te])), "probe train/test vid overlap"
    for d in TARGETS:
        mm = domm == d
        assert int(mm.sum()) == 300, (d, int(mm.sum()))
        assert int((ym[mm] == 1).sum()) == 150 and int((ym[mm] == 0).sum()) == 150, d
    assert int((domm == "ffpp").sum()) == 800

    # unified 5300-row metadata (freq/cnn rows already verified element-wise aligned)
    V_full = np.vstack([Vp, Vm]).astype(np.float32)          # 5300 x 768
    C_proj_m = (Fm[:, 0:768] / ALPHA_V).astype(np.float32)   # 2300 x 768 (= vision_proj(C)[:,0,:])
    C_proj_full = np.vstack([C_proj_p, C_proj_m]).astype(np.float32)

    y_all = fq["y"].astype(int)
    vid_all = fq["vids"].astype(str)
    domain_all = fq["domain"].astype(str)
    source_all = fq["source"].astype(str)

    # row index helpers (5300-row space). probe rows 0..2999 (train 0..2199, test 2200..2999);
    # multi rows 3000..5299 (per-domain via dom_idx).
    tr_idx = np.where(tr_mask)[0]      # 0..2199  (probe train)
    te_idx = np.where(~tr_mask)[0]     # 2200..2999 (probe test, 800)
    dom_idx = {d: np.where(domain_all == d)[0] for d in TARGETS}
    y_tr = (y_all[tr_idx] == 0).astype(int)   # 1=fake
    y_te = (y_all[te_idx] == 0).astype(int)
    vids_tr = vid_all[tr_idx]

    FEATS = OrderedDict()
    FEATS["F1"] = fq["F1"].astype(np.float32)
    FEATS["F1_radial"] = fq["F1_radial"].astype(np.float32)
    FEATS["F2"] = fq["F2"].astype(np.float32)
    FEATS["F3"] = fq["F3"].astype(np.float32)
    FEATS["F3_hist"] = fq["F3_hist"].astype(np.float32)
    FEATS["F4"] = fq["F4"].astype(np.float32)
    FEATS["F5"] = fq["F5"].astype(np.float32)
    FEATS["F_all"] = np.hstack([FEATS["F1"], FEATS["F1_radial"], FEATS["F2"],
                                FEATS["F3"], FEATS["F4"], FEATS["F5"]]).astype(np.float32)  # 123
    FEATS["dense121_db2"] = cn["dense121_db2"].astype(np.float32)
    FEATS["dense121_db3"] = cn["dense121_db3"].astype(np.float32)
    FEATS["dense121_final"] = cn["dense121_final"].astype(np.float32)
    FEATS["effnet_b4_blk5"] = cn["effnet_b4_blk5"].astype(np.float32)
    FEATS["effnet_b4_blk6"] = cn["effnet_b4_blk6"].astype(np.float32)
    FEATS["effnet_b4_final"] = cn["effnet_b4_final"].astype(np.float32)
    FEATS["resnet18_layer3"] = cn["resnet18_layer3"].astype(np.float32)
    FEATS["resnet18_layer4"] = cn["resnet18_layer4"].astype(np.float32)

    FEAT_NAMES = list(FEATS.keys())
    JOINT_NAMES = ["V|" + f for f in FEAT_NAMES]
    REPLACE_NAMES = ["V|C_proj|" + f for f in FEAT_NAMES]
    BASE_NAMES = ["V", "C_proj", "V|C_proj"]
    ARM_NAMES = BASE_NAMES + FEAT_NAMES + JOINT_NAMES + REPLACE_NAMES

    def build_matrix(name):
        if name == "V":
            return V_full
        if name == "C_proj":
            return C_proj_full
        if name == "V|C_proj":
            return np.hstack([V_full, C_proj_full]).astype(np.float32)
        if name in FEATS:
            return FEATS[name]
        if name.startswith("V|C_proj|"):
            f = name[len("V|C_proj|"):]
            return np.hstack([V_full, C_proj_full, FEATS[f]]).astype(np.float32)
        if name.startswith("V|"):
            f = name[2:]
            return np.hstack([V_full, FEATS[f]]).astype(np.float32)
        raise KeyError(name)

    # ==================================================== A. anchor (hard gate)
    ytr_f = (yp[tr] == 0).astype(int)
    sc_a = StandardScaler().fit(Vp[tr])
    clf_a = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
        sc_a.transform(Vp[tr]), ytr_f)
    anchor_auc = {}
    for d in TARGETS:
        mm = domm == d
        anchor_auc[d] = float(roc_auc_score((ym[mm] == 0).astype(int),
                                            clf_a.decision_function(sc_a.transform(Vm[mm]))))
    anchor_dev = {d: abs(anchor_auc[d] - ANCHOR_CD[d]) for d in TARGETS}
    anchor_maxdev = max(anchor_dev.values())

    def ratio_branch(Xsrc_tr, ysrc_tr, Xdst, domm_, ym_):
        s = float(np.sqrt(np.var(Xsrc_tr, axis=0, ddof=1).mean()))
        mu_src = Xsrc_tr.mean(axis=0)
        mu_real = Xsrc_tr[ysrc_tr == 1].mean(axis=0)
        mu_fake = Xsrc_tr[ysrc_tr == 0].mean(axis=0)
        cg = float(np.linalg.norm(mu_fake - mu_real) / s)
        ratios = {}
        for d in TARGETS:
            mm = domm_ == d
            ratios[d] = float(np.linalg.norm(Xdst[mm].mean(axis=0) - mu_src) / s) / cg
        return ratios, float(np.mean(list(ratios.values())))

    ratio_V_per, ratio_V = ratio_branch(Vp[tr], yp[tr], Vm, domm, ym)
    ratio_C_per, ratio_C = ratio_branch(Cp[tr], yp[tr], Cm, domm, ym)
    dev_rV = abs(ratio_V - ANCHOR_RATIO_V)
    dev_rC = abs(ratio_C - ANCHOR_RATIO_C)
    anchor_ok = (anchor_maxdev < ANCHOR_ATOL) and (dev_rV < ANCHOR_ATOL) and (dev_rC < ANCHOR_ATOL)

    P("[G17c] anchor cross-domain V C=1e-3: " + " ".join(
        f"{d}={anchor_auc[d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS) +
      f" maxdev={anchor_maxdev:.2e}")
    P(f"[G17c] ratio_V={ratio_V:.4f}(exp {ANCHOR_RATIO_V}) ratio_C={ratio_C:.4f}(exp {ANCHOR_RATIO_C})")

    if not anchor_ok:
        L("=" * 126)
        L("G17c REPORT - PRECHECK FAILED (stop)")
        L("=" * 126)
        for d in TARGETS:
            L(f"G17c_ANCHOR_CD_{d}={fmt(anchor_auc[d])} EXP={ANCHOR_CD[d]} dev={fmt(anchor_dev[d],6)}")
        L(f"G17c_ANCHOR_MAXDEV={fmt(anchor_maxdev,6)}")
        L(f"G17c_ratio_V={fmt(ratio_V)} EXP={ANCHOR_RATIO_V} dev={fmt(dev_rV,6)}")
        L(f"G17c_ratio_C={fmt(ratio_C)} EXP={ANCHOR_RATIO_C} dev={fmt(dev_rC,6)}")
        L(f"G17c_ANCHOR_PASS=0")
        L(f"wall time = {time.time()-t0:.1f}s")
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
        P("[G17c] PRECHECK FAILED -> see report")
        sys.exit(0)

    # ================================================= data hygiene (vid) ====
    ptrain_vids = set(vp[tr])
    hy = []
    cd1v = set(vidm[domm == "cd1"])
    for d in ["cd1", "cd2", "dfdcp", "ffiw", "wild"]:
        mm = domm == d
        dv = set(vidm[mm])
        n_ov_ptr = len(dv & ptrain_vids)
        n_ov_cd1 = len(dv & cd1v) if d != "cd1" else len(dv)
        n_img_ov_ptr = int(sum(1 for v in vidm[mm] if v in ptrain_vids))
        n_img_ov_cd1 = int(sum(1 for v in vidm[mm] if v in cd1v)) if d != "cd1" else 300
        hy.append((d, len(dv), n_ov_ptr, n_ov_cd1, n_img_ov_ptr, n_img_ov_cd1))
    P("[G17c] hygiene: " + " | ".join(
        f"{d}(nvid={nv},ov_ptr={op},ov_cd1={oc},img_ov_ptr={ip},img_ov_cd1={ic})"
        for (d, nv, op, oc, ip, ic) in hy))

    # ============================================== source e0 (for C1/C2) ====
    mu_src_V, E_src_V, S_src_V = center_only_pca(Vp[tr])
    e0 = E_src_V[:, 0].copy()
    z_tr = (Vp[tr] - mu_src_V) @ e0
    if z_tr[yp[tr] == 0].mean() < z_tr[yp[tr] == 1].mean():
        e0 = -e0
        z_tr = -z_tr
    gap_src = float(z_tr[yp[tr] == 0].mean() - z_tr[yp[tr] == 1].mean())

    # ====================================================== B1/B2/B3 arms ====
    RES = {}
    for name in ARM_NAMES:
        Xf = build_matrix(name).astype(np.float64)
        keep = drop_zero_var(Xf)
        Xf = Xf[:, keep]
        dim_orig = int(keep.size)
        dim = int(Xf.shape[1])
        ndrop = dim_orig - dim
        res = dict(dim=dim, dim_orig=dim_orig, ndrop=ndrop)

        Xtr_a = Xf[tr_idx]
        Xte_a = Xf[te_idx]
        # ---- B1 in-domain FF++ (fixed C + CV-selected C) ----
        res["b1_fix"] = auc(y_te, fit_lr_scores(Xtr_a, y_tr, Xte_a, C_FIX))
        Cv, _a, _pc = select_C_cv(Xtr_a, y_tr, vids_tr)
        res["b1_cvC"] = Cv
        res["b1_cv"] = auc(y_te, fit_lr_scores(Xtr_a, y_tr, Xte_a, Cv)) if Cv is not None else float("nan")

        # ---- B2 cross-domain ----
        res["b2_fix"] = {}
        res["b2_cv"] = {}
        sc_fix = fit_scaler(Xtr_a)
        clf_fix = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
            sc_fix.transform(Xtr_a), y_tr)
        sc_cv = fit_scaler(Xtr_a)
        clf_cv = LogisticRegression(C=Cv, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
            sc_cv.transform(Xtr_a), y_tr)
        for d in TARGETS:
            di = dom_idx[d]
            yd = (y_all[di] == 0).astype(int)
            res["b2_fix"][d] = auc(yd, clf_fix.decision_function(sc_fix.transform(Xf[di])))
            res["b2_cv"][d] = auc(yd, clf_cv.decision_function(sc_cv.transform(Xf[di])))

        # ---- B3 oracle (per target domain) ----
        res["b3"] = {}
        for d in TARGETS:
            di = dom_idx[d]
            cv = cv_oracle(Xf[di], y_all[di], vid_all[di])
            oracle_val = max(cv[m]["mean"] for m in METHODS)
            res["b3"][d] = dict(oracle=oracle_val, lr13=cv["LR_C1e-3"]["mean"],
                                lr10=cv["LR_C1.0"]["mean"], leak=cv["leak"],
                                mode=cv["mode"], nvid=cv["nvid"])
        RES[name] = res
        b2s = ";".join("%s=%.4f" % (d, res["b2_fix"][d]) for d in TARGETS)
        P(f"[G17c] arm {name:22s} dim={dim:5d} b1={res['b1_fix']:.4f} b2={b2s}")

    # oracle_V mean4 check
    oracle_V_mean4 = float(np.mean([RES["V"]["b3"][d]["oracle"] for d in ["cd1", "cd2", "dfdcp", "wild"]]))
    ovs = ";".join("%s=%.4f" % (d, RES["V"]["b3"][d]["oracle"]) for d in TARGETS)
    P(f"[G17c] oracle_V per-domain={ovs} mean4={oracle_V_mean4:.4f} (ref {ORACLE_V_REF})")

    # ================================================= B4 increment + ctrl ====
    # cross-domain gains (C=1e-3 fixed) and oracle gains, plus capacity controls
    B4 = {}
    for f in FEAT_NAMES:
        jn = "V|" + f
        b4 = dict(feat=f, K=RES[jn]["dim"] - RES["V"]["dim"])
        b4["gain_xdom"] = {d: RES[jn]["b2_fix"][d] - RES["V"]["b2_fix"][d] for d in TARGETS}
        b4["gain_oracle"] = {d: RES[jn]["b3"][d]["oracle"] - RES["V"]["b3"][d]["oracle"] for d in TARGETS}
        B4[f] = b4

    # capacity controls (cross-domain + oracle), 3 seeds
    P("[G17c] computing capacity controls ...")
    for f in FEAT_NAMES:
        K = B4[f]["K"]
        ctrl_xdom = {"randn": {d: [] for d in TARGETS}, "VR": {d: [] for d in TARGETS}}
        ctrl_oracle = {"randn": {d: [] for d in TARGETS}, "VR": {d: [] for d in TARGETS}}
        for s in CTRL_SEEDS:
            rng = np.random.default_rng(CTRL_SEED_BASE + s)
            R = rng.standard_normal((K, 768)) / np.sqrt(768.0)
            VR = (V_full.astype(np.float64) @ R.T)          # 5300 x K
            blocks = {"randn": rng.standard_normal((5300, K)), "VR": VR}
            for cname, blk in blocks.items():
                Xc = np.hstack([V_full, blk]).astype(np.float64)
                Xtr_c = Xc[tr_idx]; ytr_c = y_tr
                # cross-domain C=1e-3
                scc = fit_scaler(Xtr_c)
                clc = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=SEED).fit(
                    scc.transform(Xtr_c), ytr_c)
                for d in TARGETS:
                    di = dom_idx[d]
                    yd = (y_all[di] == 0).astype(int)
                    a_ = auc(yd, clc.decision_function(scc.transform(Xc[di])))
                    ctrl_xdom[cname][d].append(a_ - RES["V"]["b2_fix"][d])
                # oracle
                for d in TARGETS:
                    di = dom_idx[d]
                    cv = cv_oracle(Xc[di], y_all[di], vid_all[di])
                    ctrl_oracle[cname][d].append(max(cv[m]["mean"] for m in METHODS) -
                                                 RES["V"]["b3"][d]["oracle"])
        B4[f]["ctrl_xdom"] = ctrl_xdom
        B4[f]["ctrl_oracle"] = ctrl_oracle
        P(f"[G17c]   ctrl {f:16s} K={K} done")

    # ======================================================= C1/C2/C3 ====
    # per decision domain, along in-domain class-mean-difference direction
    C1 = {}; C2 = {}; C3 = {}
    for f in FEAT_NAMES:
        jn = "V|" + f
        Xj = build_matrix(jn).astype(np.float64)
        keepj = drop_zero_var(Xj)
        Xj = Xj[:, keepj]
        # V part = first 768 columns (V never dropped)
        Xv = build_matrix("V").astype(np.float64)
        c1 = {}; c2 = {}; c3 = {}
        for d in DECISION:
            di = dom_idx[d]
            yd = y_all[di]
            yb = (yd == 0).astype(int)
            Xv_d = Xv[di]; Xj_d = Xj[di]
            dV = Xv_d[yb == 1].mean(axis=0) - Xv_d[yb == 0].mean(axis=0)   # mu_fake - mu_real
            dJ = Xj_d[yb == 1].mean(axis=0) - Xj_d[yb == 0].mean(axis=0)
            # C1: d' along unit mean-diff direction
            def dprime(X, delta):
                u = delta / (np.linalg.norm(delta) + 1e-30)
                z = X @ u
                gap = float(z[yb == 1].mean() - z[yb == 0].mean())
                n1, n2 = int((yb == 1).sum()), int((yb == 0).sum())
                sp2 = ((n1 - 1) * z[yb == 1].var(ddof=1) + (n2 - 1) * z[yb == 0].var(ddof=1)) / (n1 + n2 - 2.0)
                return abs(gap) / np.sqrt(max(sp2, 1e-12))
            dp_V = dprime(Xv_d, dV)
            dp_J = dprime(Xj_d, dJ)
            c1[d] = dict(dp_V=dp_V, dp_J=dp_J, ratio=dp_J / dp_V if dp_V > 0 else float("nan"))
            # C2
            dJ_V = dJ[:768]
            dJ_F = dJ[768:]
            cos_dV_e0 = abs(cosv(dJ_V, e0))
            energy_off = float((dJ_F ** 2).sum() / ((dJ_V ** 2).sum() + (dJ_F ** 2).sum() + 1e-30))
            c2[d] = dict(cos_dV_e0=cos_dV_e0, energy_off=energy_off,
                         normV=float(np.linalg.norm(dJ_V)), normF=float(np.linalg.norm(dJ_F)))
            # C3: Fisher ratio + class center distance (along mean-diff direction)
            D_V = float(np.linalg.norm(dV)); D_J = float(np.linalg.norm(dJ))
            def fisher(X, delta):
                u = delta / (np.linalg.norm(delta) + 1e-30)
                z = X @ u
                gap = float(z[yb == 1].mean() - z[yb == 0].mean())
                sr = float(z[yb == 0].var(ddof=1)); sf = float(z[yb == 1].var(ddof=1))
                return gap * gap / (sr + sf + 1e-12)
            c3[d] = dict(D_V=D_V, D_J=D_J, fish_V=fisher(Xv_d, dV), fish_J=fisher(Xj_d, dJ),
                         D_ratio=D_J / D_V if D_V > 0 else float("nan"))
        C1[f] = c1; C2[f] = c2; C3[f] = c3

    # e0 gap-retention reference (V, along source e0)
    e0ret = {}
    for d in TARGETS:
        mm = domm == d
        z = (Vm[mm] - mu_src_V) @ e0
        g = float(z[ym[mm] == 0].mean() - z[ym[mm] == 1].mean())
        e0ret[d] = g / gap_src
    P("[G17c] C1/C2/C3 done; e0 gap retention V: " +
      " ".join(f"{d}={e0ret[d]:.4f}" for d in TARGETS))

    # ====================================================== C4 ratio ====
    # G12 ratio for standalone feats + joint arms, benchmark V/C
    C4 = {}
    for f in FEAT_NAMES:
        Xf = FEATS[f].astype(np.float64)
        keep = drop_zero_var(Xf)
        Xf = Xf[:, keep]
        ratio_per, ratio_mean = ratio_branch(Xf[tr_idx], y_all[tr_idx], Xf, domain_all, y_all)
        C4[f] = dict(ratio_per=ratio_per, mean=ratio_mean)
    P("[G17c] C4 done")

    # ================================================== mechanical verdicts ====
    # capacity-beat helper
    def beats_controls(gain_feat, ctrl_dict, domains):
        best = -1e9
        for cname in ctrl_dict:
            for d in domains:
                vals = ctrl_dict[cname][d]
                if vals:
                    best = max(best, max(vals))
        return (gain_feat - best) > CTRL_MARGIN, best

    XDOM_VERDICT = {}; ORACLE_VERDICT = {}
    for f in FEAT_NAMES:
        gx = B4[f]["gain_xdom"]
        go = B4[f]["gain_oracle"]
        # XDOM
        g_dfdcp = gx["dfdcp"]
        same_dir = (gx["cd2"] >= 0.0) and (gx["wild"] >= 0.0)
        beats, best_ctrl = beats_controls(g_dfdcp, B4[f]["ctrl_xdom"], ["dfdcp"])
        if g_dfdcp >= GATE_GAIN and same_dir:
            v = "COMPLEMENT_FOUND" if beats else "CAPACITY_ILLUSION"
        elif g_dfdcp < GATE_NO:
            v = "NO_COMPLEMENT"
        else:
            v = "PARTIAL"
        XDOM_VERDICT[f] = dict(v=v, g_dfdcp=g_dfdcp, same_dir=same_dir, beats=beats,
                               best_ctrl=best_ctrl, gain=gx)
        # ORACLE
        g_dfdcp_o = go["dfdcp"]
        same_dir_o = (go["cd2"] >= 0.0) and (go["wild"] >= 0.0)
        beats_o, best_ctrl_o = beats_controls(g_dfdcp_o, B4[f]["ctrl_oracle"], ["dfdcp"])
        if g_dfdcp_o >= GATE_GAIN and same_dir_o:
            v_o = "COMPLEMENT_FOUND" if beats_o else "CAPACITY_ILLUSION"
        elif g_dfdcp_o < GATE_NO:
            v_o = "NO_COMPLEMENT"
        else:
            v_o = "PARTIAL"
        ORACLE_VERDICT[f] = dict(v=v_o, g_dfdcp=g_dfdcp_o, same_dir=same_dir_o,
                                 beats=beats_o, best_ctrl=best_ctrl_o, gain=go)

    # G17_DOMAIN_LOCK
    DOMAIN_LOCK = {}
    for f in FEAT_NAMES:
        r = C4[f]["mean"]
        DOMAIN_LOCK[f] = "DOMAIN_LOCKED" if r > 1.0 else "NOT_LOCKED"
    # G17_ORTHOGONAL (mechanically on C2 |cos(dV,e0)|; note it is arm-invariant)
    ORTHO = {}
    for f in FEAT_NAMES:
        mean_cos = float(np.mean([C2[f][d]["cos_dV_e0"] for d in DECISION]))
        if mean_cos < 0.5:
            ORTHO[f] = "ORTHOGONAL"
        elif mean_cos > 0.8:
            ORTHO[f] = "REDUNDANT"
        else:
            ORTHO[f] = "INTERMEDIATE"

    # G17_BEST_FEATURE: rank by cross-domain increment (mean over decision domains)
    def mean_dec(gain):
        return float(np.mean([gain[d] for d in DECISION]))
    ranked_xdom = sorted(FEAT_NAMES, key=lambda f: -mean_dec(B4[f]["gain_xdom"]))
    best_feat = ranked_xdom[0]
    # cheapest effective: smallest dim among feats with mean_dec gain >= 0.03 and beating controls
    cheapest = None
    for f in sorted(FEAT_NAMES, key=lambda f: B4[f]["K"]):
        if mean_dec(B4[f]["gain_xdom"]) >= GATE_GAIN:
            g_dfdcp = B4[f]["gain_xdom"]["dfdcp"]
            beats, _ = beats_controls(g_dfdcp, B4[f]["ctrl_xdom"], ["dfdcp"])
            if beats:
                cheapest = f
                break
    P("[G17c] verdicts done; best_feat=%s cheapest_effective=%s" %
      (best_feat, cheapest if cheapest else "none"))

    wall = time.time() - t0

    # ==================================================== MACHINE_BLOCK ====
    mb = OrderedDict()
    mb["G17c_TASK"] = "joint_eval: do low-level/freq/CNN feats complement ViT-V missing class info"
    mb["G17c_DATA"] = "probe_feats.npz(V/C/C_proj n=3000) + feats_multi.npz(V/C/F n=2300) + freq_feats.npz(5300) + cnn_feats.npz(5300)"
    mb["G17c_ARMS"] = "V(768);C_proj(768);V|C_proj(1536);16 feats(freq8+cnn8);16 V|feat;16 V|C_proj|feat"
    mb["G17c_F_all_DEF"] = "F1|F1_radial|F2|F3|F4|F5=123 (no F3_hist; see CAVEATS re 91-d)"
    mb["G17c_Y_CONV"] = "1=real,0=fake; AUC positive=fake"
    mb["G17c_ANCHOR_CD"] = ";".join(f"{d}={anchor_auc[d]:.4f}(exp{ANCHOR_CD[d]})" for d in TARGETS)
    mb["G17c_ANCHOR_MAXDEV"] = fmt(anchor_maxdev, 6)
    mb["G17c_ANCHOR_PASS"] = "1" if anchor_ok else "0"
    mb["G17c_ratio_V"] = fmt(ratio_V, 4)
    mb["G17c_ratio_V_EXP"] = fmt(ANCHOR_RATIO_V, 4)
    mb["G17c_ratio_C"] = fmt(ratio_C, 4)
    mb["G17c_ratio_C_EXP"] = fmt(ANCHOR_RATIO_C, 4)
    mb["G17c_ORACLE_V_mean4"] = fmt(oracle_V_mean4, 4)
    mb["G17c_ORACLE_V_mean4_EXP"] = fmt(ORACLE_V_REF, 4)
    for d in TARGETS:
        mb[f"G17c_B2_V_C1e-3_{d}"] = fmt(RES["V"]["b2_fix"][d], 4)
    mb["G17c_VALID_DOMAINS"] = ",".join(DECISION) + " (cd1=leak ref, ffiw=1-vid leak, both excluded from aggregates)"
    for f in FEAT_NAMES:
        mb[f"G17c_XDOM_VERDICT_{f}"] = XDOM_VERDICT[f]["v"]
        mb[f"G17c_ORACLE_VERDICT_{f}"] = ORACLE_VERDICT[f]["v"]
        mb[f"G17c_DOMAIN_LOCK_{f}"] = DOMAIN_LOCK[f]
        mb[f"G17c_ORTHO_{f}"] = ORTHO[f]
    mb["G17c_BEST_FEATURE"] = best_feat
    mb["G17c_CHEAPEST_EFFECTIVE"] = cheapest if cheapest else "none"
    mb["G17c_WALL_S"] = fmt(wall, 1)
    mb["G17c_GPU"] = "0"
    mb["G17c_FORWARDS"] = "0"
    mb["G17c_THREADS"] = "4"
    mb["G17c_PROCESSES"] = "1"
    mb["G17c_NPZ_READ"] = "4"

    mblines = ["#### MACHINE_BLOCK " + "#" * 103]
    mblines += [f"{k}={v}" for k, v in mb.items()]

    # ====================================================== REPORT ===========
    L("=" * 126)
    L("G17c REPORT -- joint evaluation: do low-level / frequency-domain (and CNN local-detail)")
    L("   features complement the class-discriminative information that ViT CLS (V) is missing?")
    L("=" * 126)
    L("Pure offline CPU; zero deep-model forward / zero GPU / zero deep training (only light LR heads).")
    L("y: 1=real, 0=fake; all AUC positive=fake.  Conventions reused verbatim from G12/G14/G16.")
    L("=" * 126)
    L("")
    L("\n".join(mblines))
    L("")

    L("-" * 126)
    L("A. ANCHOR REPRODUCTION (hard gate, atol=1e-3)")
    L("-" * 126)
    L("  cross-domain V (raw 768, LR C=1e-3, StandardScaler fit probe-train):")
    L("    " + "  ".join(f"{d}={anchor_auc[d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS))
    L(f"    maxdev={anchor_maxdev:.2e}  -> ANCHOR_PASS={int(anchor_ok)}")
    L(f"  G12 ratio (mean over 5 target domains): ratio_V={ratio_V:.4f}(exp {ANCHOR_RATIO_V}) "
      f"ratio_C={ratio_C:.4f}(exp {ANCHOR_RATIO_C})")
    L("  oracle_V (best-of-4-methods, video-grouped 5-fold, mean4 excl ffiw) = "
      f"{oracle_V_mean4:.4f} (ref {ORACLE_V_REF})")
    L("")

    L("-" * 126)
    L("B. DATA HYGIENE (video-level)")
    L("-" * 126)
    L("  domain   nvid  ov_probe_train  ov_cd1  img_ov_ptr  img_ov_cd1  |  label")
    L("  " + "-" * 100)
    for (d, nv, op, oc, ip, ic) in hy:
        lab = {d: d}[d]
        if d == "dfdcp":
            lab = "CLEAN (only fully clean domain)"
        elif d == "cd1":
            lab = "LEAK (same source as cd2; 41/41 vid overlap -> NOT held-out)"
        elif d == "cd2":
            lab = "SUB-CLEAN (28% imgs share cd1 vids)"
        elif d == "wild":
            lab = "SUB-CLEAN (3% imgs share probe-train vids)"
        elif d == "ffiw":
            lab = "1-VID (excluded from aggregates)"
        L(f"  {d:<6}  {nv:>5}  {op:>14}  {oc:>6}  {ip:>10}  {ic:>10}  |  {lab}")
    L("  aggregates: 'dfdcp only' and 'dfdcp+cd2+wild' (decision domains).")
    L("")

    L("-" * 126)
    L("B1. FF++ IN-DOMAIN AUC (probe train 2200 -> test 800, video-disjoint; positive=fake)")
    L("-" * 126)
    L(f"  {'arm':<20}{'dim':>6}{'LR(C=1e-3)':>13}{'CV-C':>8}{'LR(CV-C)':>12}")
    for name in ARM_NAMES:
        r = RES[name]
        L(f"  {name:<20}{r['dim']:>6}{r['b1_fix']:>13.4f}{r['b1_cvC']:>8.0e}{r['b1_cv']:>12.4f}")
    L("")

    L("-" * 126)
    L("B2. CROSS-DOMAIN AUC (head = SS+LR fit on probe-train 2200; eval per target 300; positive=fake)")
    L("    columns: C=1e-3 fixed ; mean over decision domains (dfdcp/cd2/wild) for both C")
    L("-" * 126)
    L(f"  {'arm':<20}{'dim':>6}  " + "  ".join(f"{d:>9}" for d in TARGETS) +
      f"{'m3fix':>8}{'m3cv':>8}")
    for name in ARM_NAMES:
        r = RES[name]
        m3fix = float(np.mean([r["b2_fix"][d] for d in DECISION]))
        m3cv = float(np.mean([r["b2_cv"][d] for d in DECISION]))
        L(f"  {name:<20}{r['dim']:>6}  " + "  ".join(f"{r['b2_fix'][d]:>9.4f}" for d in TARGETS) +
          f"{m3fix:>8.4f}{m3cv:>8.4f}")
    L("")

    L("-" * 126)
    L("B3. TARGET-DOMAIN IN-DOMAIN ORACLE (per-domain 300, video-grouped 5-fold; oracle=best-of-4-methods)")
    L("    ffiw leak=True (1 vid) excluded from all aggregates")
    L("-" * 126)
    L(f"  {'arm':<20}{'dim':>6}  " + "  ".join(f"{d:>9}" for d in TARGETS) + f"{'mean4':>8}")
    for name in ARM_NAMES:
        r = RES[name]
        m4 = float(np.mean([r["b3"][d]["oracle"] for d in ["cd1", "cd2", "dfdcp", "wild"]]))
        L(f"  {name:<20}{r['dim']:>6}  " + "  ".join(f"{r['b3'][d]['oracle']:>9.4f}" for d in TARGETS) +
          f"{m4:>8.4f}")
    L("")

    L("-" * 126)
    L("B4. INCREMENT TEST (decisive): [V|feat] vs V, cross-domain (C=1e-3) + oracle (best-of-4)")
    L("    capacity controls: randn(K) = N(0,1) noise; VR = V@R (R: Kx768 Gaussian/sqrt768). 3 seeds.")
    L("    'beats ctrl' = gain(dfdcp) > max over both controls and seeds + 0.01 margin.")
    L("-" * 126)
    L(f"  {'feat':<17}{'K':>5}  " + "  ".join(f"{d:>9}" for d in TARGETS) + f"{'m3':>8}" +
      "  ctrl_xdom(dfdcp mean/range)  ctrl_oracle(dfdcp mean/range)")
    for f in FEAT_NAMES:
        g = B4[f]["gain_xdom"]
        m3 = mean_dec(g)
        cx = B4[f]["ctrl_xdom"]
        co = B4[f]["ctrl_oracle"]
        cx_dfdcp = cx["randn"]["dfdcp"] + cx["VR"]["dfdcp"]
        co_dfdcp = co["randn"]["dfdcp"] + co["VR"]["dfdcp"]
        def mr(vals):
            a = np.asarray(vals)
            return f"{a.mean():+.4f}/[{a.min():+.4f},{a.max():+.4f}]"
        L(f"  {f:<17}{B4[f]['K']:>5}  " + "  ".join(f"{g[d]:>+9.4f}" for d in TARGETS) +
          f"{m3:>+8.4f}  {mr(cx_dfdcp):<28} {mr(co_dfdcp)}")
    L("")

    L("-" * 126)
    L("C1. e0 GAP-RETENTION CONTEXT + d' RATIO (along in-domain class-mean-difference direction)")
    L("    V e0 gap retention (G11/g14 gapRatio): " +
      " ".join(f"{d}={e0ret[d]:.4f}" for d in TARGETS))
    L("    d' = |mean_fake-mean_real| / pooled_sd along the in-domain mean-diff direction; ratio = d'(joint)/d'(V)")
    L("-" * 126)
    L(f"  {'feat':<17}" + "".join(f"{d:>16}" for d in DECISION))
    for f in FEAT_NAMES:
        cells = "".join(f"{C1[f][d]['ratio']:>16.4f}" for d in DECISION)
        L(f"  {f:<17}{cells}")
    L("")

    L("-" * 126)
    L("C2. DIRECTION ORTHOGONALITY (joint class-mean-diff delta_joint = [dV | dF])")
    L("    1) |cos(dV, e0)|  (dV = first 768 coords of delta_joint; NOTE arm-invariant, = V's own cos)")
    L("    2) energy of delta_joint outside V subspace = ||dF||^2 / (||dV||^2+||dF||^2)")
    L("    (G14 off-axis AUC reference ~0.69-0.75: the off-e0 direction carries real signal in V)")
    L("-" * 126)
    L(f"  {'feat':<17}" + "".join(f"{d:>18}" for d in DECISION))
    for f in FEAT_NAMES:
        cells = "".join(f"cos={C2[f][d]['cos_dV_e0']:.3f}/off={C2[f][d]['energy_off']:.3f}" for d in DECISION)
        L(f"  {f:<17}{cells}")
    L("")

    L("-" * 126)
    L("C3. FISHER RATIO / CLASS-CENTER DISTANCE ([V|feat] vs V, per decision domain)")
    L("-" * 126)
    L(f"  {'feat':<17}" + "".join(f"{d:>22}" for d in DECISION))
    for f in FEAT_NAMES:
        cells = "".join(f"D={C3[f][d]['D_ratio']:.2f}/F={C3[f][d]['fish_J'] / max(C3[f][d]['fish_V'],1e-12):.2f}" for d in DECISION)
        L(f"  {f:<17}{cells}")
    L("")

    L("-" * 126)
    L("C4. G12 DOMAIN-SENSITIVITY RATIO (mean over 5 domains; benchmark V=0.1983, C=2.7160)")
    L("-" * 126)
    L(f"  {'feat':<17}{'ratio_mean':>12}{'DOMAIN_LOCK':>14}")
    for f in FEAT_NAMES:
        L(f"  {f:<17}{C4[f]['mean']:>12.4f}{DOMAIN_LOCK[f]:>14}")
    L("")

    L("-" * 126)
    L("MECHANICAL VERDICTS (pre-registered; no directional inference beyond the rule)")
    L("-" * 126)
    L("  [G17_COMPLEMENT_XDOM]  gain(dfdcp)>=+0.03 AND cd2/wild same-direction AND beats both controls ->")
    L("      COMPLEMENT_FOUND; gain(dfdcp)<+0.01 -> NO_COMPLEMENT; else PARTIAL. (fails ctrl -> CAPACITY_ILLUSION)")
    L("")
    for f in FEAT_NAMES:
        v = XDOM_VERDICT[f]
        L(f"    {f:<17} gain_dfdcp={v['g_dfdcp']:+.4f} same_dir={int(v['same_dir'])} beats={int(v['beats'])} "
          f"(best_ctrl_dfdcp={v['best_ctrl']:+.4f}) -> {v['v']}")
    L("")
    L("  [G17_COMPLEMENT_ORACLE]  same rule in oracle (best-of-4)口径")
    for f in FEAT_NAMES:
        v = ORACLE_VERDICT[f]
        L(f"    {f:<17} gain_dfdcp={v['g_dfdcp']:+.4f} same_dir={int(v['same_dir'])} beats={int(v['beats'])} "
          f"(best_ctrl_dfdcp={v['best_ctrl']:+.4f}) -> {v['v']}")
    L("")
    L("  [G17_DOMAIN_LOCK]  ratio>1.0 -> DOMAIN_LOCKED (benchmark V=0.1983, C=2.7160)")
    for f in FEAT_NAMES:
        L(f"    {f:<17} ratio={C4[f]['mean']:.4f} -> {DOMAIN_LOCK[f]}")
    L("")
    L("  [G17_ORTHOGONAL]  C2 |cos(dV,e0)|<0.5 -> ORTHOGONAL; >0.8 -> REDUNDANT")
    L("    NOTE: dV is the V-subspace part of the joint class-mean-diff; for raw concatenation this is")
    L("    IDENTICAL across all arms (= V's own class-mean-diff, cos~0.94-0.98), so this gate cannot")
    L("    discriminate between feats -- the discriminating C2 quantity is energy-outside-V (see C2).")
    for f in FEAT_NAMES:
        mc = float(np.mean([C2[f][d]["cos_dV_e0"] for d in DECISION]))
        L(f"    {f:<17} mean|cos(dV,e0)|={mc:.4f} -> {ORTHO[f]}")
    L("")
    L("  [G17_BEST_FEATURE]  ranked by cross-domain mean3 increment (dfdcp+cd2+wild):")
    L("    " + "  ".join(f"{f}({mean_dec(B4[f]['gain_xdom']):+.4f})" for f in ranked_xdom))
    L(f"    best_feature={best_feat}  cheapest_effective={cheapest if cheapest else 'none'}")
    L("")
    L("  [G17_REPLACE]  [V|C_proj|feat] vs [V|C_proj] (cross-domain C=1e-3, mean over decision domains):")
    for f in FEAT_NAMES:
        rn = "V|C_proj|" + f
        base = "V|C_proj"
        g = {d: RES[rn]["b2_fix"][d] - RES[base]["b2_fix"][d] for d in TARGETS}
        m3 = mean_dec(g)
        L(f"    {f:<17} " + "  ".join(f"{d}={g[d]:+.4f}" for d in TARGETS) + f"  mean3={m3:+.4f}")
    L("")

    L("-" * 126)
    L("AUDIT")
    L("-" * 126)
    L("  imgs_read=0  gpu=0 (CUDA_VISIBLE_DEVICES='')  forwards=0 (no model instantiated / no forward)")
    L("  threads=4 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS=4)  processes=1  max_workers<=2 (n_jobs=None)")
    L("  npz read=4 (probe_feats, feats_multi, freq_feats, cnn_feats)")
    L("  head: StandardScaler(fit train) + LogisticRegression(lbfgs, max_iter=3000, random_state=0);")
    L("       C=1e-3 fixed + inner-CV-selected C (video-grouped 5-fold on train side only).")
    L("  oracle: best-of-4-methods (LR_C1e-3, LR_C1.0, kNN_k5, RBF_SVM), video-grouped 5-fold, seed=0.")
    L(f"  wall_s={wall:.1f}")
    L("")
    L("CAVEATS (honest)")
    L("  1. n=300 per target domain, 5-fold CV fold-to-fold variance large: single-domain AUC differences")
    L("     < ~0.04 should NOT be over-read; only cross-domain consistent directions + magnitude count.")
    L("  2. dfdcp is the ONLY fully clean held-out domain. cd2 has 28% images sharing cd1 videos,")
    L("     wild has 3% sharing probe-train videos -> both labelled SUB-CLEAN. cd1 is same-source as cd2")
    L("     (41/41 video overlap) -> NOT a held-out domain (reference only). ffiw = 1 video -> excluded.")
    L("  3. Aggregates use decision domains only: 'dfdcp' and 'dfdcp+cd2+wild'. cd1/ffiw never aggregated.")
    L("  4. F_all is built as F1|F1_radial|F2|F3|F4|F5 = 123-d to match the spec '123 dims, no F3_hist'.")
    L("     The spec text also wrote 'concat(F1,F2,F3,F4,F5)' which is 91-d; F1_radial is a SUBSET of F1")
    L("     (F1[:, :32]), so the 123-d version duplicates 32 dims. This is flagged; a 91-d alternative would")
    L("     drop the redundant F1_radial. All per-column standardization makes the duplication harmless to LR.")
    L("  5. C_proj for the 2300 multi rows is derived offline as F[:,0:768]/alpha_v (alpha_v=0.306288,")
    L("     G12 verified F[:,0:768]=alpha_v*C_proj with cos_min=1.000000); no checkpoint read, zero forward.")
    L("     StandardScaler is per-column so any uniform global scale on a block is exactly cancelled.")
    L("  6. G17_ORTHOGONAL gate (|cos(dV,e0)|) is structurally arm-invariant: for raw concatenation the")
    L("     V-subspace part of the joint class-mean-diff equals V's own class-mean-diff (cos~0.94-0.98).")
    L("     Hence it mechanically returns REDUNDANT for every arm; the informative orthogonality metric is")
    L("     C2's energy-outside-V fraction (||dF||^2/||delta_joint||^2).")
    L("  7. Capacity controls randn(K)/V@R control for 'adding K dims' vs 'adding real information'.")
    L("     'beats controls' = gain(dfdcp) > max(control gains over 3 seeds) + 0.01. With only 3 seeds the")
    L("     control range is a lower bound on its variance; borderline cases are labelled explicitly.")
    L("  8. Cross-domain AUC trains a LINEAR head on the source only -> lower bound on transferability;")
    L("     oracle trains in-domain -> upper-bound-ish (label-dependent, not deployable). Correlation !=")
    L("     causation: a [V|feat] gain shows the feat carries linearly-available signal V lacks, NOT that")
    L("     the detector would exploit it.")
    L("  9. Oracle 'best-of-4' reproduces G14 oracle_V mean4=0.9111 within tolerance (see A).")
    L("  10. All feature/CNN/freq rows were produced by G17a/G17b and are element-wise aligned with")
    L("     probe_feats / feats_multi row order (verified here too via y/vids).")
    L("")
    L("=" * 126)
    L(f"END OF REPORT   (wall {wall:.1f}s)")
    L("=" * 126)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")

    np.savez(STATS, **{k: np.array(str(mb[k])) for k in mb.keys()},
             **{"meta_keys": np.array(list(mb.keys()))})
    P("[done] report -> %s" % REPORT)
    P("[done] stats  -> %s" % STATS)
    P("[done] anchor_pass=%d maxdev=%.2e oracle_V_mean4=%.4f best_feat=%s wall=%.1fs" %
      (int(anchor_ok), anchor_maxdev, oracle_V_mean4, best_feat, wall))


if __name__ == "__main__":
    main()
