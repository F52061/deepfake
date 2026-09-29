# -*- coding: utf-8 -*-
"""
G18c -- decisive increment test: does a DenseNet121 SPECIALIZED for forgery
detection add discriminative power to the ViT CLS representation V (768) that V
lacks, i.e. does [V | dense121_specialized] beat V cross-domain?

Protocol / code path reused VERBATIM from G17c (vit_module/_g17/run_g17c.py),
section "B4. INCREMENT TEST", restricted to the G18 arms:
  * same StandardScaler (fit on the train side only, scale<1e-12 -> 1.0)
  * same LogisticRegression(C=1e-3 fixed, solver='lbfgs', max_iter=3000,
    random_state=seed) cross-domain head fit on probe-train (2200 rows)
  * same C grid [1e-4,1e-3,1e-2,1e-1,1.0] + video-grouped 5-fold inner-CV C
    selection on the train side only (used for the B1 context column only --
    the B4 increment itself is the fixed C=1e-3 column, exactly as in G17c)
  * same capacity-control construction:
        randn:  N(0,1) (5300 x K)
        VR:     V_full @ R.T , R = rng.standard_normal((K,768))/sqrt(768)
    with rng = np.random.default_rng(CTRL_SEED_BASE + s)
  * same target domains cd1/cd2/dfdcp/ffiw/wild, 300 rows each
  * same oracle (G14-exact video-grouped 5-fold, best-of-4 methods)口径

Arms:
  ImageNet_dense121_final          (G17, unspecialized ImageNet reference)
  df_ffpp_{final,db3,db2}          (G18a, FF++-specialized)
  df_lodo_{dfdcp,cd2,wild}_final   (G18b, multi-domain LODO-specialized)

LODO contamination discipline: for a df_lodo_X arm, only the held-out domain X
column is a valid transfer measurement; every other target-domain column is
training-set memorization -> marked CONTAMINATED and excluded from every gate
and aggregate.  df_ffpp_* / ImageNet_* never saw any target domain -> valid
everywhere.

Resource discipline (user requirement): pure CPU, OMP/MKL/OPENBLAS/NUMEXPR/
VECLIB/BLIS=4, torch.set_num_threads(4), cv2.setNumThreads(0), num_workers=0,
CUDA_VISIBLE_DEVICES='' (no GPU is used at all, none is claimed).
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
    import cv2
    cv2.setNumThreads(0)
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False
try:
    import torch
    torch.set_num_threads(4)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VIT = os.path.join(PROJECT_ROOT, "vit_module")
PROBE_NPZ = os.path.join(VIT, "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(VIT, "_tsne", "feats_multi.npz")
FREQ_NPZ = os.path.join(VIT, "_g17", "freq_feats.npz")          # metadata cross-check only
CNN_NPZ = os.path.join(VIT, "_g17", "cnn_feats.npz")
FFPP_NPZ = os.path.join(HERE, "df_ffpp_feats.npz")
LODO_NPZ = {d: os.path.join(HERE, "df_lodo_%s_feats.npz" % d) for d in ["dfdcp", "cd2", "wild"]}
REPORT = os.path.join(HERE, "g18c_report.txt")
STATS = os.path.join(HERE, "g18c_stats.npz")
RUNLOG = os.path.join(HERE, "run_log_g18c.txt")

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
DECISION = ["dfdcp", "cd2", "wild"]                 # G17c judgement domains
VALID4 = ["cd1", "cd2", "dfdcp", "wild"]            # G18c: all target domains with >=1 video
EXCLUDED = ["ffiw"]                                 # 1 video -> never aggregated

C_FIX = 1e-3
MAXIT = 3000
NFOLDS = 5
ALPHA_V = 0.306288
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

SEEDS = [0, 1, 2, 3, 4]             # G17c used [0,1,2]; extended prefix to 5
G17C_SEEDS = [0, 1, 2]              # exact G17c control subset
CTRL_SEED_BASE = 20260910
CTRL_MARGIN = 0.01
GATE_SPEC = 0.03
GATE_LODO = 0.03
GATE_DELOCK = 0.70
BOOT_B = 2000
BOOT_SEED = 20260910

METHODS = ["LR_C1e-3", "LR_C1.0", "kNN_k5", "RBF_SVM"]

ANCHOR_CD = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_RATIO_V = 0.1983
ANCHOR_RATIO_C = 2.7160
ANCHOR_ATOL = 1e-3
ORACLE_V_REF = 0.9111

# arm name -> (source npz tag, key inside that npz)
ARM_SRC = OrderedDict([
    ("ImageNet_dense121_final", ("cnn", "dense121_final")),
    ("df_ffpp_final",           ("ffpp", "df_ffpp_final")),
    ("df_ffpp_db3",             ("ffpp", "df_ffpp_db3")),
    ("df_ffpp_db2",             ("ffpp", "df_ffpp_db2")),
    ("df_lodo_dfdcp_final",     ("lodo_dfdcp", "dense121_final")),
    ("df_lodo_cd2_final",       ("lodo_cd2", "dense121_final")),
    ("df_lodo_wild_final",      ("lodo_wild", "dense121_final")),
])
ARM_NAMES = list(ARM_SRC.keys())

# LODO contamination discipline: which target-domain columns are a valid
# transfer measurement for this arm.  None listed == valid on all TARGETS.
ARM_VALID_DOMAINS = {
    "ImageNet_dense121_final": None,
    "df_ffpp_final": None,
    "df_ffpp_db3": None,
    "df_ffpp_db2": None,
    "df_lodo_dfdcp_final": ["dfdcp"],
    "df_lodo_cd2_final": ["cd2", "cd1"],     # cd1 flagged as cd1-cd2 family
    "df_lodo_wild_final": ["wild"],
}
FAMILY_FLAG = {("df_lodo_cd2_final", "cd1"): "cd1_SUBSET_OF_cd2_FAMILY"}


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def auc(y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def fit_scaler(Xtr):
    sc = StandardScaler().fit(Xtr)
    sc.scale_ = np.where(sc.scale_ < 1e-12, 1.0, sc.scale_)
    return sc


def drop_zero_var(X):
    return np.var(X, axis=0) > 0.0


def get_folds(y, vid, n_splits=NFOLDS, seed=0):
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


def fit_eval_method(Xtr, ytr, Xte, yte, method):
    sc = fit_scaler(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    if method == "LR_C1e-3":
        clf = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=MAXIT, random_state=0).fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    elif method == "LR_C1.0":
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=MAXIT, random_state=0).fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    elif method == "kNN_k5":
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=None).fit(Xtr_s, ytr)
        s = clf.predict_proba(Xte_s)[:, 1]
    elif method == "RBF_SVM":
        clf = SVC(C=1.0, gamma="scale", class_weight="balanced").fit(Xtr_s, ytr)
        s = clf.decision_function(Xte_s)
    else:
        raise ValueError(method)
    return auc(yte, s)


def cv_oracle(X, y, vid):
    """G14/G17c-exact video-grouped 5-fold best-of-4 oracle."""
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
    out["oracle"] = float(max(out[m]["mean"] for m in METHODS))
    return out


def fit_lr_scores(Xtr, ytr, Xte, C, seed=0):
    sc = fit_scaler(Xtr)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAXIT, random_state=seed).fit(
        sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def select_C_cv(Xtr, ytr, vtr, C_grid=C_GRID):
    yb = np.asarray(ytr, dtype=int)
    folds, _mode, _leak = get_folds(yb, vtr)
    best_C, best_auc, per_C = None, -1.0, {}
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


def paired_bootstrap_delta(y, s_joint, s_base, B=BOOT_B, seed=BOOT_SEED):
    """AUXILIARY (not in G17c): paired row bootstrap of dAUC = AUC(joint)-AUC(base)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()
    n = len(y)
    d = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            d[b] = np.nan
            continue
        d[b] = roc_auc_score(yb, s_joint[idx]) - roc_auc_score(yb, s_base[idx])
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(mean=float("nan"), std=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    return dict(mean=float(d.mean()), std=float(d.std(ddof=1)),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), n=int(len(d)))


def main():
    t0 = time.time()
    log = []

    def L(x=""):
        log.append(str(x))

    def P(x=""):
        print(x, flush=True)

    # ======================================================== 0. DATA LOAD ====
    p = np.load(PROBE_NPZ, allow_pickle=True)
    Vp = p["V"].astype(np.float64)
    Cp = p["C"].astype(np.float64)
    C_proj_p = p["C_proj"].astype(np.float32)
    yp = p["y"].astype(int)
    vp = np.asarray(p["vids"], dtype=str)
    ppath = np.asarray(p["paths"], dtype=str)
    tr_mask = np.asarray(p["train_mask"])
    assert Vp.shape == (3000, 768) and Cp.shape == (3000, 1024), "probe shapes"

    m = np.load(MULTI_NPZ, allow_pickle=True)
    Vm = m["V"].astype(np.float64)
    Cm = m["C"].astype(np.float64)
    Fm = m["F"].astype(np.float32)
    ym = m["y"].astype(int)
    domm = np.asarray(m["domain"], dtype=str)
    vidm = np.asarray(m["vid"], dtype=str)
    mpath = np.asarray(m["path"], dtype=str)

    cn = np.load(CNN_NPZ, allow_pickle=True)
    ff = np.load(FFPP_NPZ, allow_pickle=True)
    lo = {d: np.load(LODO_NPZ[d], allow_pickle=True) for d in LODO_NPZ}
    fq = np.load(FREQ_NPZ, allow_pickle=True)

    # ======================================= 1. ALIGNMENT ASSERTIONS (loud) ==
    align = OrderedDict()
    cn_paths = np.asarray(cn["paths"], dtype=str)
    cn_y = np.asarray(cn["y"])
    cn_vid = np.asarray(cn["vids"], dtype=str)
    cn_dom = np.asarray(cn["domain"], dtype=str)
    errs = []

    def chk(label, cond, detail=""):
        align[label] = bool(cond)
        if not cond:
            errs.append("%s %s" % (label, detail))
        return cond

    chk("cnn_n_rows", cn_paths.shape[0] == 5300, str(cn_paths.shape))
    chk("ffpp_paths_byte_equal_g17", np.array_equal(cn_paths, np.asarray(ff["paths"], dtype=str)))
    chk("ffpp_y_equal", np.array_equal(cn_y, np.asarray(ff["y"])))
    chk("ffpp_vid_equal", np.array_equal(cn_vid, np.asarray(ff["vids"], dtype=str)))
    chk("ffpp_dom_equal", np.array_equal(cn_dom, np.asarray(ff["domain"], dtype=str)))
    for d in LODO_NPZ:
        chk("lodo_%s_paths_byte_equal_g17" % d,
            np.array_equal(cn_paths, np.asarray(lo[d]["paths"], dtype=str)))
        chk("lodo_%s_y_equal" % d, np.array_equal(cn_y, np.asarray(lo[d]["y"])))
        chk("lodo_%s_vid_equal" % d, np.array_equal(cn_vid, np.asarray(lo[d]["vids"], dtype=str)))
        chk("lodo_%s_dom_equal" % d, np.array_equal(cn_dom, np.asarray(lo[d]["domain"], dtype=str)))

    # 5300-row space = vertical stack of probe (0..2999) then multi (3000..5299)
    chk("y_stack_probe_multi", np.array_equal(cn_y, np.concatenate([yp, ym])))
    chk("vid_stack_probe_multi",
        np.array_equal(cn_vid, np.concatenate([vp, vidm])))
    chk("path_stack_probe_multi",
        np.array_equal(cn_paths, np.concatenate([ppath, mpath])))
    chk("domain_probe_block", set(np.unique(cn_dom[:3000])) == {"ffpp_probe"},
        str(np.unique(cn_dom[:3000])))
    chk("domain_multi_block_equals_feats_multi", np.array_equal(cn_dom[3000:], domm))
    # freq_feats.npz metadata (the G17c source of domain_all / vid_all / y_all)
    chk("freq_y_equal_cnn", np.array_equal(np.asarray(fq["y"]), cn_y))
    chk("freq_vid_equal_cnn", np.array_equal(np.asarray(fq["vids"], dtype=str), cn_vid))
    chk("freq_dom_equal_cnn", np.array_equal(np.asarray(fq["domain"], dtype=str), cn_dom))

    if errs:
        P("[G18c] ALIGNMENT FAILED:")
        for e in errs:
            P("   " + e)
        with open(RUNLOG, "w", encoding="utf-8") as fh:
            fh.write("G18c ABORT -- alignment failures:\n" + "\n".join(errs) + "\n")
        sys.exit(2)
    P("[g18c] alignment OK (%d checks): all arms' paths/y/vids/domain byte-equal to _g17/cnn_feats.npz,"
      " which stacks probe(0:3000)+multi(3000:5300)" % len(align))

    tr = np.where(tr_mask)[0]
    te = np.where(~tr_mask)[0]
    assert (len(tr), len(te)) == (2200, 800), (len(tr), len(te))
    assert not (set(vp[tr]) & set(vp[te])), "probe train/test vid overlap"
    for d in TARGETS:
        mm = domm == d
        assert int(mm.sum()) == 300, (d, int(mm.sum()))
        assert int((ym[mm] == 1).sum()) == 150 and int((ym[mm] == 0).sum()) == 150, d
    assert int((domm == "ffpp").sum()) == 800

    # ---------------------------------------------- unified 5300-row space --
    V_full = np.vstack([Vp, Vm]).astype(np.float32)
    y_all = cn_y.astype(int)
    vid_all = cn_vid
    domain_all = cn_dom
    tr_idx = np.where(tr_mask)[0]        # 0..2199 probe train
    te_idx = np.where(~tr_mask)[0]       # 2200..2999 probe test
    dom_idx = {d: np.where(domain_all == d)[0] for d in TARGETS}
    y_tr = (y_all[tr_idx] == 0).astype(int)     # 1=fake
    y_te = (y_all[te_idx] == 0).astype(int)
    vids_tr = vid_all[tr_idx]

    # ------------------------------------------------------------- features --
    SRC = {
        "cnn": cn,
        "ffpp": ff,
        "lodo_dfdcp": lo["dfdcp"],
        "lodo_cd2": lo["cd2"],
        "lodo_wild": lo["wild"],
    }

    def feat_of(arm):
        tag, key = ARM_SRC[arm]
        a = SRC[tag][key]
        assert a.shape[0] == 5300, (arm, a.shape)
        return a.astype(np.float32)

    def valid_domains(arm):
        v = ARM_VALID_DOMAINS[arm]
        return list(TARGETS) if v is None else list(v)

    def is_valid(arm, d):
        return d in valid_domains(arm)

    # ============================================ 2. ANCHOR (hard gate) =====
    ytr_f = (yp[tr] == 0).astype(int)
    sc_a = StandardScaler().fit(Vp[tr])
    clf_a = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=0).fit(
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

    C_proj_m = (Fm[:, 0:768] / ALPHA_V).astype(np.float32)
    C_proj_full = np.vstack([C_proj_p, C_proj_m]).astype(np.float32)
    ratio_V_per, ratio_V = ratio_branch(Vp[tr], yp[tr], Vm, domm, ym)
    ratio_C_per, ratio_C = ratio_branch(Cp[tr], yp[tr], Cm, domm, ym)
    dev_rV = abs(ratio_V - ANCHOR_RATIO_V)
    dev_rC = abs(ratio_C - ANCHOR_RATIO_C)
    anchor_ok = (anchor_maxdev < ANCHOR_ATOL) and (dev_rV < ANCHOR_ATOL) and (dev_rC < ANCHOR_ATOL)

    P("[g18c] anchor V C=1e-3: " + " ".join(f"{d}={anchor_auc[d]:.4f}" for d in TARGETS) +
      f" maxdev={anchor_maxdev:.2e}  ratio_V={ratio_V:.4f}  ratio_C={ratio_C:.4f}")

    if not anchor_ok:
        L("G18c REPORT -- PRECHECK FAILED (anchor reproduction, stop)")
        L(f"ANCHOR_MAXDEV={anchor_maxdev:.6f}")
        L(f"ratio_V={ratio_V:.6f} exp {ANCHOR_RATIO_V}  ratio_C={ratio_C:.6f} exp {ANCHOR_RATIO_C}")
        L("ANCHOR_PASS=0")
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
        P("[g18c] PRECHECK FAILED -> report")
        sys.exit(0)

    # ============================================== 3. CAPACITY CONTROLS =====
    # The control matrices depend ONLY on K (and the seed base), so they are
    # built once per unique K exactly as G17c builds them per arm.
    KS = sorted({feat_of(a).shape[1] for a in ARM_NAMES})
    P("[g18c] unique K = %s ; building capacity controls over seeds %s ..." % (KS, SEEDS))
    CTRL = {}
    for K in KS:
        cx = {"randn": {d: [] for d in TARGETS}, "VR": {d: [] for d in TARGETS}}
        co = {"randn": {d: [] for d in TARGETS}, "VR": {d: [] for d in TARGETS}}
        for s in SEEDS:
            rng = np.random.default_rng(CTRL_SEED_BASE + s)
            R = rng.standard_normal((K, 768)) / np.sqrt(768.0)
            blk = {"randn": rng.standard_normal((5300, K)),
                   "VR": (V_full.astype(np.float64) @ R.T)}
            for cname in ["randn", "VR"]:
                Xc = np.hstack([V_full, blk[cname]]).astype(np.float64)
                Xtr_c = Xc[tr_idx]
                scc = fit_scaler(Xtr_c)
                clc = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT,
                                         random_state=0).fit(scc.transform(Xtr_c), y_tr)
                for d in TARGETS:
                    di = dom_idx[d]
                    yd = (y_all[di] == 0).astype(int)
                    cx[cname][d].append(auc(yd, clc.decision_function(scc.transform(Xc[di]))))
                for d in TARGETS:
                    di = dom_idx[d]
                    o = cv_oracle(Xc[di], y_all[di], vid_all[di])["oracle"]
                    co[cname][d].append(o)
                del Xc, blk[cname]
            P("[g18c]   ctrl K=%d seed=%d done (%.0fs)" % (K, s, time.time() - t0))
        CTRL[K] = dict(xdom=cx, oracle=co)

    # ======================================================== 4. ARMS =======
    RES = {}
    ARMS_ALL = ["V"] + ARM_NAMES
    for name in ARMS_ALL:
        Xf = V_full.astype(np.float64) if name == "V" else np.hstack(
            [V_full, feat_of(name)]).astype(np.float64)
        keep = drop_zero_var(Xf)
        Xf = Xf[:, keep]
        res = dict(dim=int(Xf.shape[1]), dim_orig=int(keep.size), ndrop=int(keep.size - Xf.shape[1]))
        Xtr_a, Xte_a = Xf[tr_idx], Xf[te_idx]

        # B1 in-domain FF++ (context; C fixed + inner-CV C on the train side)
        res["b1_fix"] = auc(y_te, fit_lr_scores(Xtr_a, y_tr, Xte_a, C_FIX))
        Cv, _a, _pc = select_C_cv(Xtr_a, y_tr, vids_tr)
        res["b1_cvC"] = Cv
        res["b1_cv"] = auc(y_te, fit_lr_scores(Xtr_a, y_tr, Xte_a, Cv)) if Cv is not None else float("nan")

        # B2 cross-domain C=1e-3 (the B4 increment口径), per seed
        res["b2_fix"] = {}
        res["b2_fix_seeds"] = {}
        for s in SEEDS:
            scc = fit_scaler(Xtr_a)
            clc = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT,
                                     random_state=s).fit(scc.transform(Xtr_a), y_tr)
            for d in TARGETS:
                di = dom_idx[d]
                yd = (y_all[di] == 0).astype(int)
                sc_d = clc.decision_function(scc.transform(Xf[di]))
                res["b2_fix_seeds"].setdefault(d, []).append(auc(yd, sc_d))
                if s == 0:
                    res["b2_fix"][d] = float(res["b2_fix_seeds"][d][0])
                    res.setdefault("scores", {})[d] = sc_d
        res["b2_fix_seed_std"] = {d: float(np.std(res["b2_fix_seeds"][d], ddof=1)) for d in TARGETS}

        # B3 oracle (context + gain_oracle); skip feat-alone oracle for cost
        if name == "V" or name in ARM_NAMES:
            res["b3"] = {}
            for d in TARGETS:
                di = dom_idx[d]
                cv = cv_oracle(Xf[di], y_all[di], vid_all[di])
                res["b3"][d] = dict(oracle=cv["oracle"], leak=cv["leak"], mode=cv["mode"],
                                    nvid=cv["nvid"],
                                    lr13=cv["LR_C1e-3"]["mean"], lr10=cv["LR_C1.0"]["mean"])
        RES[name] = res
        P("[g18c] arm %-24s dim=%5d b1=%.4f  b2(C1e-3)=" % (name, res["dim"], res["b1_fix"]) +
          " ".join("%s=%.4f" % (d, res["b2_fix"][d]) for d in TARGETS))

    # feat-alone cross-domain AUC (context only, no oracle)
    FEATONLY = {}
    for name in ARM_NAMES:
        F = feat_of(name).astype(np.float64)
        keep = drop_zero_var(F)
        F = F[:, keep]
        scf = fit_scaler(F[tr_idx])
        clf = LogisticRegression(C=C_FIX, solver="lbfgs", max_iter=MAXIT, random_state=0).fit(
            scf.transform(F[tr_idx]), y_tr)
        FEATONLY[name] = {d: auc((y_all[dom_idx[d]] == 0).astype(int),
                                 clf.decision_function(scf.transform(F[dom_idx[d]])))
                          for d in TARGETS}

    oracle_V_mean4 = float(np.mean([RES["V"]["b3"][d]["oracle"] for d in ["cd1", "cd2", "dfdcp", "wild"]]))
    P("[g18c] oracle_V mean4=%.4f (ref %.4f)" % (oracle_V_mean4, ORACLE_V_REF))

    # ========================================================== 5. B4 =======
    B4 = {}
    for name in ARM_NAMES:
        K = RES[name]["dim"] - RES["V"]["dim"]
        b4 = dict(K=K, feat=name)
        b4["auc_joint"] = {d: RES[name]["b2_fix"][d] for d in TARGETS}
        b4["auc_base"] = {d: RES["V"]["b2_fix"][d] for d in TARGETS}
        b4["gain_xdom"] = {d: RES[name]["b2_fix"][d] - RES["V"]["b2_fix"][d] for d in TARGETS}
        b4["gain_xdom_seeds"] = {d: [RES[name]["b2_fix_seeds"][d][i] - RES["V"]["b2_fix_seeds"][d][i]
                                     for i in range(len(SEEDS))] for d in TARGETS}
        b4["gain_oracle"] = {d: RES[name]["b3"][d]["oracle"] - RES["V"]["b3"][d]["oracle"] for d in TARGETS}
        b4["valid"] = {d: is_valid(name, d) for d in TARGETS}
        # bootstrap std of the cross-domain increment (paired, target rows only)
        b4["boot"] = {}
        for d in TARGETS:
            di = dom_idx[d]
            yd = (y_all[di] == 0).astype(int)
            b4["boot"][d] = paired_bootstrap_delta(yd, RES[name]["scores"][d], RES["V"]["scores"][d])
        B4[name] = b4

    # controls attached per arm (identical for equal K -- verified below)
    for name in ARM_NAMES:
        K = B4[name]["K"]
        c = CTRL[K]
        cx = {cn_: {d: [v - RES["V"]["b2_fix"][d] for v in c["xdom"][cn_][d]] for d in TARGETS}
              for cn_ in ["randn", "VR"]}
        co = {cn_: {d: [v - RES["V"]["b3"][d]["oracle"] for v in c["oracle"][cn_][d]] for d in TARGETS}
              for cn_ in ["randn", "VR"]}
        B4[name]["ctrl_xdom"] = cx
        B4[name]["ctrl_oracle"] = co
        B4[name]["ctrl_xdom_g17c3"] = {cn_: {d: cx[cn_][d][:len(G17C_SEEDS)] for d in TARGETS}
                                       for cn_ in ["randn", "VR"]}
        B4[name]["ctrl_oracle_g17c3"] = {cn_: {d: co[cn_][d][:len(G17C_SEEDS)] for d in TARGETS}
                                         for cn_ in ["randn", "VR"]}

    # ============================================== 6. DELOCK RATIOS =======
    RATIO = {}
    for name in ARM_NAMES:
        F = feat_of(name).astype(np.float64)
        keep = drop_zero_var(F)
        F = F[:, keep]
        per, mean = ratio_branch(F[tr_idx], y_all[tr_idx], F, domain_all, y_all)
        RATIO[name] = dict(per=per, mean=mean)
    g18a_mean = RATIO["df_ffpp_final"]["mean"]
    g18b_folds = [RATIO["df_lodo_dfdcp_final"]["mean"], RATIO["df_lodo_cd2_final"]["mean"],
                  RATIO["df_lodo_wild_final"]["mean"]]
    g18b_fold_mean = float(np.mean(g18b_folds))
    delock_ratio = g18b_fold_mean / g18a_mean if g18a_mean != 0 else float("nan")
    P("[g18c] ratio: g18a(df_ffpp_final)=%.4f  g18b folds=%s mean=%.4f  ratio=%.4f (gate<=%.2f)" %
      (g18a_mean, ["%.4f" % x for x in g18b_folds], g18b_fold_mean, delock_ratio, GATE_DELOCK))

    # ================================================== 7. GATES ===========
    def ctrl_max(ctrl_dict, d, seeds_n=None):
        best = -1e9
        for cn_ in ctrl_dict:
            v = ctrl_dict[cn_][d]
            if seeds_n is not None:
                v = v[:seeds_n]
            if v:
                best = max(best, max(v))
        return float(best)

    # ---- G18_INCREMENT: per arm over its own VALID4 domains ----------------
    INC = {}
    for name in ARM_NAMES:
        vd = valid_domains(name)
        vd4 = [d for d in VALID4 if d in vd]
        vd3 = [d for d in DECISION if d in vd]
        per = {}
        for d in vd4:
            g = B4[name]["gain_xdom"][d]
            cm5 = ctrl_max(B4[name]["ctrl_xdom"], d)
            cm3 = ctrl_max(B4[name]["ctrl_xdom_g17c3"], d)
            r5 = ctrl_max(B4[name]["ctrl_xdom"], d, seeds_n=None)
            per[d] = dict(gain=g, ctrl_max5=cm5, ctrl_max3=cm3,
                          passes5=bool(g > cm5 + CTRL_MARGIN),
                          passes3=bool(g > cm3 + CTRL_MARGIN),
                          positive=bool(g > 0.0),
                          clean=bool(g > 0.0 and g > cm5 + CTRL_MARGIN))
        n_pass5 = sum(1 for d in per if per[d]["clean"])
        n_pass3 = sum(1 for d in per if per[d]["positive"] and per[d]["passes3"])
        if len(vd4) >= 4:
            verdict = "COMPLEMENT_EXISTS" if n_pass5 >= 3 else "NO_COMPLEMENT"
            verdict3 = "COMPLEMENT_EXISTS" if n_pass3 >= 3 else "NO_COMPLEMENT"
        else:
            verdict = "N/A_LT4_VALID(%d/%d)" % (n_pass5, len(vd4))
            verdict3 = "N/A_LT4_VALID(%d/%d)" % (n_pass3, len(vd4))
        # variant restricted to the clean decision domains (dfdcp/cd2/wild)
        per3 = {d: per[d] for d in vd3}
        n_pass3d = sum(1 for d in per3 if per3[d]["clean"])
        if len(per3) >= 3:
            verdict_dec = "COMPLEMENT_EXISTS" if n_pass3d >= 3 else "NO_COMPLEMENT"
        else:
            verdict_dec = "N/A_LT3_DECISION(%d/%d)" % (n_pass3d, len(per3))
        INC[name] = dict(per=per, n_pass5=n_pass5, n_pass3=n_pass3, n_valid4=len(vd4),
                         verdict=verdict, verdict_g17c3=verdict3, verdict_decision=verdict_dec,
                         n_pass_decision=n_pass3d, per_decision=per3)

    # ---- G18_SPECIALIZATION_EFFECT -----------------------------------------
    def mean_over(arm, domains):
        return float(np.mean([B4[arm]["gain_xdom"][d] for d in domains]))

    shared4 = [d for d in VALID4 if is_valid("df_ffpp_final", d) and is_valid("ImageNet_dense121_final", d)]
    shared3 = [d for d in DECISION if is_valid("df_ffpp_final", d) and is_valid("ImageNet_dense121_final", d)]
    spec4 = mean_over("df_ffpp_final", shared4) - mean_over("ImageNet_dense121_final", shared4)
    spec3 = mean_over("df_ffpp_final", shared3) - mean_over("ImageNet_dense121_final", shared3)
    spec_verdict4 = "SPECIALIZATION_HELPS" if spec4 >= GATE_SPEC else "NO_EFFECT"
    spec_verdict3 = "SPECIALIZATION_HELPS" if spec3 >= GATE_SPEC else "NO_EFFECT"

    # ---- G18_LODO_EFFECT ---------------------------------------------------
    LODO = {}
    for d in ["dfdcp", "cd2", "wild"]:
        arm = "df_lodo_%s_final" % d
        g_lodo = B4[arm]["gain_xdom"][d]
        g_ffpp = B4["df_ffpp_final"]["gain_xdom"][d]
        diff = g_lodo - g_ffpp
        LODO[d] = dict(arm=arm, gain_lodo=g_lodo, gain_ffpp=g_ffpp, diff=diff,
                       verdict="LODO_HELPS" if diff >= GATE_LODO else "NO_EFFECT",
                       ctrl_max5=ctrl_max(B4[arm]["ctrl_xdom"], d),
                       auc_lodo=B4[arm]["auc_joint"][d])
    n_lodo_help = sum(1 for d in LODO if LODO[d]["verdict"] == "LODO_HELPS")
    lodo_verdict = "LODO_HELPS" if n_lodo_help == 3 else "NO_EFFECT"

    # ---- G18_DELOCK --------------------------------------------------------
    delock_verdict = "DELOCK_EFFECTIVE" if delock_ratio <= GATE_DELOCK else "NO_EFFECT"

    # =================================================== 8. REPORT =========
    wall = time.time() - t0
    mb = OrderedDict()
    mb["G18c_TASK"] = "decisive increment: does a forgery-specialized DenseNet121 add cross-domain signal V lacks"
    mb["G18c_ARMS"] = "V(768); " + "; ".join("%s(%d)" % (a, RES[a]["dim"] - 768) for a in ARM_NAMES)
    mb["G18c_ARMS_SRC"] = "; ".join("%s<-%s:%s" % (a, ARM_SRC[a][0], ARM_SRC[a][1]) for a in ARM_NAMES)
    mb["G18c_PROTOCOL"] = "G17c B4 increment, verbatim: SS(fit probe-train)+LR(C=1e-3,lbfgs,max_iter=3000); [V|feat] vs [V]"
    mb["G18c_SEEDS"] = "head random_state seeds=%s (arm dAUC seed-invariant: lbfgs deterministic); control seeds=%s" % (SEEDS, SEEDS)
    mb["G18c_CTRL"] = "randn(K)~N(0,1); VR=V@R.T, R=normal(K,768)/sqrt(768), rng=default_rng(20260910+s); max over both"
    mb["G18c_Y_CONV"] = "1=real,0=fake; AUC positive=fake"
    mb["G18c_ALIGNMENT"] = "ALL_PASS n_checks=%d (paths byte-equal g17<->g18a<->g18b; y/vids/domain byte-equal; probe+multi stack)" % len(align)
    mb["G18c_ANCHOR_CD"] = ";".join("%s=%.4f(exp %.4f)" % (d, anchor_auc[d], ANCHOR_CD[d]) for d in TARGETS)
    mb["G18c_ANCHOR_MAXDEV"] = "%.6f" % anchor_maxdev
    mb["G18c_ANCHOR_PASS"] = "1"
    mb["G18c_ratio_V"] = "%.4f" % ratio_V
    mb["G18c_ratio_C"] = "%.4f" % ratio_C
    mb["G18c_ORACLE_V_mean4"] = "%.4f" % oracle_V_mean4
    mb["G18c_VALID4"] = ",".join(VALID4)
    mb["G18c_DECISION3"] = ",".join(DECISION)
    mb["G18c_LODO_DISCIPLINE"] = "; ".join(
        "%s->valid=%s" % (a, "ALL" if ARM_VALID_DOMAINS[a] is None else "+".join(ARM_VALID_DOMAINS[a]))
        for a in ARM_NAMES)
    for d in TARGETS:
        mb["G18c_BASE_V_AUC_%s" % d] = "%.4f" % RES["V"]["b2_fix"][d]
    for a in ARM_NAMES:
        for d in TARGETS:
            tag = "" if is_valid(a, d) else "_CONTAMINATED"
            mb["G18c_dAUC_%s_%s%s" % (a, d, tag)] = "%+.4f" % B4[a]["gain_xdom"][d]
    for a in ARM_NAMES:
        mb["G18c_G18_INCREMENT_%s" % a] = INC[a]["verdict"]
    mb["G18c_G18_SPECIALIZATION_EFFECT"] = spec_verdict4
    mb["G18c_G18_SPECIALIZATION_DELTA4"] = "%+.4f" % spec4
    mb["G18c_G18_SPECIALIZATION_DELTA3"] = "%+.4f" % spec3
    for d in LODO:
        mb["G18c_G18_LODO_EFFECT_%s" % d] = LODO[d]["verdict"]
        mb["G18c_G18_LODO_DIFF_%s" % d] = "%+.4f" % LODO[d]["diff"]
    mb["G18c_G18_LODO_EFFECT"] = lodo_verdict
    mb["G18c_ratio_G18a_df_ffpp_final"] = "%.4f" % g18a_mean
    mb["G18c_ratio_G18b_fold_mean"] = "%.4f" % g18b_fold_mean
    mb["G18c_DELOCK_ratio_over_g18a"] = "%.4f" % delock_ratio
    mb["G18c_G18_DELOCK"] = delock_verdict
    mb["G18c_WALL_S"] = "%.1f" % wall
    mb["G18c_GPU"] = "0"
    mb["G18c_FORWARDS"] = "0"
    mb["G18c_THREADS"] = "4"
    mb["G18c_PROCESSES"] = "1"
    mb["G18c_NUM_WORKERS"] = "0"
    mb["G18c_TORCH_SET_NUM_THREADS"] = "4" if HAVE_TORCH else "torch_unavailable"
    mb["G18c_CV2_NUM_THREADS"] = "0" if HAVE_CV2 else "cv2_unavailable"
    mb["G18c_NPZ_READ"] = str(len(set(list(SRC.keys()))) + 3)

    mblines = ["#### MACHINE_BLOCK " + "#" * 103] + ["%s=%s" % (k, v) for k, v in mb.items()]

    L("=" * 126)
    L("G18c REPORT -- decisive test: does a forgery-SPECIALIZED DenseNet121 add cross-domain")
    L("   discriminative power that the ViT CLS representation V (768) lacks?")
    L("=" * 126)
    L("Pure offline CPU; zero deep-model forward / zero GPU / zero deep training (only light LR heads).")
    L("Protocol = G17c section B4, verbatim (same scaler, same LR, same C grid, same controls, same targets).")
    L("y: 1=real, 0=fake; all AUC positive=fake.  Baseline everywhere = V alone (768-d); arm = [V|feat].")
    L("=" * 126)
    L("")
    L("\n".join(mblines))
    L("")

    L("-" * 126)
    L("A. ALIGNMENT + ANCHOR (hard gate)")
    L("-" * 126)
    L("  alignment checks: %d, all PASS" % len(align))
    L("   - df_ffpp/db2/db3, df_lodo_{dfdcp,cd2,wild} 'paths' byte-equal to _g17/cnn_feats.npz: True")
    L("   - y / vids / domain byte-equal across all 5 arm files and to freq_feats.npz: True")
    L("   - cnn rows == vstack(probe_feats[0:3000], feats_multi[3000:5300]) (y/vids/paths): True")
    L("  anchor cross-domain V C=1e-3: " + " ".join(
        f"{d}={anchor_auc[d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS) +
      f"  maxdev={anchor_maxdev:.2e} -> ANCHOR_PASS=1")
    L(f"  G12 ratio: V={ratio_V:.4f}(exp {ANCHOR_RATIO_V})  C={ratio_C:.4f}(exp {ANCHOR_RATIO_C})")
    L(f"  oracle_V (best-of-4, mean4 excl ffiw) = {oracle_V_mean4:.4f} (ref {ORACLE_V_REF})")
    L("")
    L("  hygiene (from G17c, unchanged):  dfdcp=CLEAN | cd2=SUB-CLEAN (28% imgs share cd1 vids) |")
    L("  wild=SUB-CLEAN (3% share probe-train vids) | cd1=NOT-held-out reference (41/41 vid overlap with cd2) |")
    L("  ffiw=1 video -> excluded from all aggregates.")
    L("")

    L("-" * 126)
    L("B. BASELINE: V ALONE, cross-domain C=1e-3 (probe-train 2200 -> target 300) + oracle")
    L("-" * 126)
    L(f"  {'domain':<8}{'V_auc_xdom':>12}{'V_oracle':>10}{'n_vid':>7}")
    for d in TARGETS:
        L(f"  {d:<8}{RES['V']['b2_fix'][d]:>12.4f}{RES['V']['b3'][d]['oracle']:>10.4f}"
          f"{RES['V']['b3'][d]['nvid']:>7}")
    L(f"  V seed-std over head seeds {SEEDS}: " +
      " ".join("%s=%.1e" % (d, RES["V"]["b2_fix_seed_std"][d]) for d in TARGETS))
    L("")

    L("-" * 126)
    L("C. ARM AUCs.  [V|feat] cross-domain C=1e-3 per target domain; CONTAMINATED = df_lodo_X arm")
    L("   on a domain that was in ITS OWN training set (memorization, excluded from every gate/aggregate)")
    L("-" * 126)
    L(f"  {'arm':<24}{'dim':>6}  " + "  ".join(f"{d:>9}" for d in TARGETS) + f"{'meanV4':>9}{'oracle_dfdcp':>14}")
    for a in ARM_NAMES:
        cells = []
        for d in TARGETS:
            s = f"{RES[a]['b2_fix'][d]:.4f}"
            cells.append(s if is_valid(a, d) else s + "*")
        mv4 = float(np.mean([RES[a]["b2_fix"][d] for d in VALID4]))
        L(f"  {a:<24}{RES[a]['dim']:>6}  " + "  ".join(f"{c:>9}" for c in cells) +
          f"{mv4:>9.4f}{RES[a]['b3']['dfdcp']['oracle']:>14.4f}")
    L(f"  {'V':<24}{RES['V']['dim']:>6}  " + "  ".join(f"{RES['V']['b2_fix'][d]:>9.4f}" for d in TARGETS) +
      f"{np.mean([RES['V']['b2_fix'][d] for d in VALID4]):>9.4f}{RES['V']['b3']['dfdcp']['oracle']:>14.4f}")
    L("  (* = CONTAMINATED for that arm: the domain was in the arm's training set)")
    L("")
    L("  feat-alone cross-domain AUC (no V), C=1e-3 (context: does the specialized detector transfer at all?):")
    L(f"  {'feat':<24}  " + "  ".join(f"{d:>9}" for d in TARGETS))
    for a in ARM_NAMES:
        cells = []
        for d in TARGETS:
            s = f"{FEATONLY[a][d]:.4f}"
            cells.append(s if is_valid(a, d) else s + "*")
        L(f"  {a:<24}  " + "  ".join(f"{c:>9}" for c in cells))
    L("")

    L("-" * 126)
    L("D. B4 INCREMENT (decisive).  dAUC = AUC([V|feat]) - AUC([V]), cross-domain C=1e-3, positive=fake")
    L(f"   head seeds={SEEDS} (lbfgs is deterministic -> dAUC is seed-invariant; per-seed list shows that)")
    L("   std_boot = AUXILIARY paired row-bootstrap std over the 300 target rows (B=2000, "
      "seed 20260910) -- NOT part of the G17c protocol, variance context only")
    L("-" * 126)
    L(f"  {'arm':<24}{'K':>5}  " + "".join(f"{d:>22}" for d in TARGETS))
    for a in ARM_NAMES:
        cells = []
        for d in TARGETS:
            g = B4[a]["gain_xdom"][d]
            bt = B4[a]["boot"][d]
            txt = f"{g:+.4f}+-{bt['std']:.4f}"
            if not is_valid(a, d):
                txt = "CONTAM"
            cells.append(f"{txt:>22}")
        L(f"  {a:<24}{B4[a]['K']:>5}  " + "".join(cells))
    L("")
    L("  per-seed dAUC (SEEDS=%s) -- identical across seeds by construction:" % SEEDS)
    for a in ARM_NAMES:
        L(f"    {a:<24} " + "  ".join(
            "%s:[%s]" % (d, ",".join("%+.4f" % v for v in B4[a]["gain_xdom_seeds"][d]))
            for d in TARGETS if is_valid(a, d)))
    L("")
    L("  V|feat vs feat-alone vs V (context), valid domains only:")
    for a in ARM_NAMES:
        for d in valid_domains(a):
            L(f"    {a:<24} {d:<6} V={RES['V']['b2_fix'][d]:.4f}  feat={FEATONLY[a][d]:.4f}  "
              f"[V|feat]={RES[a]['b2_fix'][d]:.4f}  dAUC={B4[a]['gain_xdom'][d]:+.4f}")
    L("")

    L("-" * 126)
    L("E. CAPACITY CONTROLS side by side (same K, 5 seeds each).  ctrl dAUC = AUC([V|ctrl]) - AUC([V])")
    L("-" * 126)
    L(f"  {'arm':<24}{'K':>5}  {'control':<7}{'seed':>6}  " + "".join(f"{d:>10}" for d in TARGETS))
    for a in ARM_NAMES:
        K = B4[a]["K"]
        for cn_ in ["randn", "VR"]:
            for i, s in enumerate(SEEDS):
                vals = [B4[a]["ctrl_xdom"][cn_][d][i] for d in TARGETS]
                L(f"  {a:<24}{K:>5}  {cn_:<7}{s:>6}  " + "".join(f"{v:>+10.4f}" for v in vals))
        L(f"  {a:<24}{K:>5}  {'MAX5':<7}{'':>6}  " +
          "".join(f"{ctrl_max(B4[a]['ctrl_xdom'], d):>+10.4f}" for d in TARGETS))
        L(f"  {a:<24}{K:>5}  {'MAX3':<7}{'':>6}  " +
          "".join(f"{ctrl_max(B4[a]['ctrl_xdom_g17c3'], d):>+10.4f}" for d in TARGETS))
    L("")
    L("  oracle-side controls (best-of-4) MAX5:")
    L(f"  {'arm':<24}{'K':>5}  " + "".join(f"{d:>10}" for d in TARGETS))
    for a in ARM_NAMES:
        L(f"  {a:<24}{B4[a]['K']:>5}  " +
          "".join(f"{ctrl_max(B4[a]['ctrl_oracle'], d):>+10.4f}" for d in TARGETS))
    L("")

    L("-" * 126)
    L("F. ORACLE INCREMENT (context; best-of-4 in-domain, video-grouped 5-fold)")
    L("-" * 126)
    L(f"  {'arm':<24}{'K':>5}  " + "".join(f"{d:>22}" for d in TARGETS))
    for a in ARM_NAMES:
        cells = []
        for d in TARGETS:
            g = B4[a]["gain_oracle"][d]
            cells.append(f"{g:+.4f}" if is_valid(a, d) else "CONTAM")
        L(f"  {a:<24}{B4[a]['K']:>5}  " + "".join(f"{c:>22}" for c in cells))
    L("")

    L("-" * 126)
    L("G. MECHANICAL GATES (pre-registered)")
    L("-" * 126)
    L("  [G18_INCREMENT] dAUC>0 AND > max(both controls, 5 seeds)+0.01 in >=3/4 valid domains")
    L("    (valid domains of an arm = its non-contaminated target domains, max 4 = cd1,cd2,dfdcp,wild)")
    for a in ARM_NAMES:
        r = INC[a]
        L(f"    {a:<24} valid4={r['n_valid4']} pass5={r['n_pass5']} -> {r['verdict']}"
          f"   [G17c-3seed variant: pass={r['n_pass3']} -> {r['verdict_g17c3']}]"
          f"   [decision3 dfdcp/cd2/wild: pass={r['n_pass_decision']} -> {r['verdict_decision']}]")
        for d in valid_domains(a):
            if d in r["per"]:
                q = r["per"][d]
                L(f"        {d:<6} dAUC={q['gain']:+.4f} ctrl_max5={q['ctrl_max5']:+.4f} "
                  f"ctrl_max3={q['ctrl_max3']:+.4f} pass5={int(q['passes5'])} pass3={int(q['passes3'])}")
    L("")
    L(f"  [G18_SPECIALIZATION_EFFECT] mean dAUC(df_ffpp_final) - mean dAUC(ImageNet_dense121_final) >= +0.03")
    L(f"    shared valid domains {shared4}: " +
      " ".join("%s(%+.4f vs %+.4f)" % (d, B4['df_ffpp_final']['gain_xdom'][d],
                                        B4['ImageNet_dense121_final']['gain_xdom'][d]) for d in shared4))
    L(f"    mean dAUC df_ffpp_final={mean_over('df_ffpp_final', shared4):+.4f}  "
      f"ImageNet_dense121_final={mean_over('ImageNet_dense121_final', shared4):+.4f}  "
      f"delta={spec4:+.4f} -> {spec_verdict4}")
    L(f"    [decision3 variant: delta={spec3:+.4f} -> {spec_verdict3}]")
    L("")
    L("  [G18_LODO_EFFECT] dAUC(df_lodo_X_final) - dAUC(df_ffpp_final) >= +0.03 on held-out X")
    for d in LODO:
        q = LODO[d]
        L(f"    {d:<6} dAUC(lodo)={q['gain_lodo']:+.4f}  dAUC(df_ffpp)={q['gain_ffpp']:+.4f}  "
          f"diff={q['diff']:+.4f}  ctrl_max5={q['ctrl_max5']:+.4f} -> {q['verdict']}")
    L(f"    overall: {n_lodo_help}/3 held-out domains -> {lodo_verdict}")
    L("")
    L("  [G18_DELOCK] ratio(G18b fold mean) <= 0.70 * ratio(G18a mean)")
    L(f"    ratio(df_ffpp_final)  G18a mean            = {g18a_mean:.4f}"
      + ("   [RECOMPUTED here]" if True else ""))
    L(f"    ratio(df_lodo_dfdcp_final)  G18b fold     = {RATIO['df_lodo_dfdcp_final']['mean']:.4f}")
    L(f"    ratio(df_lodo_cd2_final)    G18b fold     = {RATIO['df_lodo_cd2_final']['mean']:.4f}")
    L(f"    ratio(df_lodo_wild_final)   G18b fold     = {RATIO['df_lodo_wild_final']['mean']:.4f}")
    L(f"    G18b fold mean = {g18b_fold_mean:.4f} ; G18a mean = {g18a_mean:.4f} ; "
      f"ratio/G18a = {delock_ratio:.4f} ; 0.70*G18a = {GATE_DELOCK*g18a_mean:.4f} -> {delock_verdict}")
    L("    recomputed here (not reused from the parent):  "
      f"g18a={g18a_mean:.4f} [parent 0.2598]  "
      f"dfdcp={RATIO['df_lodo_dfdcp_final']['mean']:.4f} [parent 0.6713]  "
      f"cd2={RATIO['df_lodo_cd2_final']['mean']:.4f} [parent 0.2824]  "
      f"wild={RATIO['df_lodo_wild_final']['mean']:.4f} [parent 0.3375]")
    L("")
    L("  per-domain ratio (feat-only, source=probe-train 2200, mean over 5 target domains):")
    L(f"    {'arm':<24}" + "".join(f"{d:>9}" for d in TARGETS) + f"{'mean':>9}")
    for a in ARM_NAMES:
        L(f"    {a:<24}" + "".join(f"{RATIO[a]['per'][d]:>9.4f}" for d in TARGETS) +
          f"{RATIO[a]['mean']:>9.4f}")
    L("")

    L("-" * 126)
    L("H. DEVIATIONS FROM G17c")
    L("-" * 126)
    L("  1. ARMS: only the 7 G18 arms + V baseline are evaluated (G17c evaluated 16 freq/CNN feats).")
    L("     The code path for each arm is byte-identical (same functions, same constants).")
    L("  2. SEEDS: G17c used control seeds [0,1,2]; here the control seed list is extended to")
    L("     [0,1,2,3,4] (>=5 as required) and the same list is used as the LR random_state.  Because")
    L("     lbfgs ignores random_state the arm dAUC is EXACTLY seed-invariant (std=0, per-seed list")
    L("     printed).  The G17c 3-seed control maximum is reported alongside (ctrl_max3) and the gate")
    L("     verdict under the G17c-exact 3-seed controls is printed as a variant.  The 5-seed max is")
    L("     the stricter (larger) bound, so the primary verdict uses it.")
    L("  3. Control matrices depend only on K, so they are built once per unique K (K in {512,1024})")
    L("     instead of per arm.  Same rng, same seed base, same construction -> identical numbers.")
    L("  4. ADDED (not in G17c): paired row-bootstrap std of dAUC on the target rows (B=2000) to give a")
    L("     real uncertainty scale, since the deterministic protocol yields std=0 over head seeds.")
    L("  5. ADDED (not in G17c): feat-alone cross-domain AUC, per-seed dAUC listing, G18a/G18b ratio")
    L("     recomputation.  None of these change the pre-registered quantities.")
    L("  6. ffiw is never aggregated (1 video).  VALID4 = cd1,cd2,dfdcp,wild is used for the")
    L("     G18_INCREMENT '3/4 valid domains' rule; the decision-domain variant (dfdcp,cd2,wild) is")
    L("     reported too.  cd1 is a valid transfer target for the probe-trained head (0 video overlap")
    L("     with probe-train) but is the same source family as cd2, so it is flagged not dropped.")
    L("")

    L("-" * 126)
    L("I. AUDIT")
    L("-" * 126)
    L("  imgs_read=0  gpu=0  forwards=0  (no deep model instantiated, no forward, no GPU claimed)")
    L(f"  threads=4 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS=4)  torch.set_num_threads(4)="
      f"{HAVE_TORCH}  cv2.setNumThreads(0)={HAVE_CV2}  processes=1  num_workers=0")
    L("  head: StandardScaler(fit train) + LogisticRegression(lbfgs, max_iter=3000, random_state=seed);")
    L("       C=1e-3 fixed (B2/B4) + inner-CV-selected C on the train side only (B1 context).")
    L("  oracle: best-of-4-methods (LR_C1e-3, LR_C1.0, kNN_k5, RBF_SVM), video-grouped 5-fold, seed=0.")
    L(f"  wall_s={wall:.1f}")
    L("")
    L("CAVEATS")
    L("  1. n=300 per target domain -> single-domain AUC differences < ~0.04 must not be over-read;")
    L("     the bootstrap std column quantifies exactly this noise for dAUC.")
    L("  2. dfdcp is the only fully clean held-out domain; cd2 (28% imgs share cd1 vids) and wild (3%)")
    L("     are SUB-CLEAN; cd1 is cd2's source family; ffiw is 1 video.")
    L("  3. A [V|feat] gain shows the feat carries LINEARLY available signal V lacks when both are")
    L("     standardized and fed to one LR -- it is not proof the deployed detector would use it.")
    L("  4. The df_lodo_X arms' non-X columns are training-set memorization (AUC ~0.99) and are")
    L("     marked CONTAMINATED; they are excluded from all means and gates.")
    L("")
    L("=" * 126)
    L(f"END OF REPORT   (wall {wall:.1f}s)")
    L("=" * 126)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")

    np.savez(STATS,
             **{("mb_" + k): np.array(str(v)) for k, v in mb.items()},
             **{("align_" + k): np.array(bool(v)) for k, v in align.items()},
             **{("gain_" + a + "_" + d): np.array(B4[a]["gain_xdom"][d]) for a in ARM_NAMES for d in TARGETS},
             **{("valid_" + a + "_" + d): np.array(bool(B4[a]["valid"][d])) for a in ARM_NAMES for d in TARGETS},
             **{("ctrl_xdom_" + a + "_" + cn_ + "_" + d): np.array(B4[a]["ctrl_xdom"][cn_][d])
                for a in ARM_NAMES for cn_ in ["randn", "VR"] for d in TARGETS},
             **{("ctrl_oracle_" + a + "_" + cn_ + "_" + d): np.array(B4[a]["ctrl_oracle"][cn_][d])
                for a in ARM_NAMES for cn_ in ["randn", "VR"] for d in TARGETS},
             **{("ratio_" + a + "_" + d): np.array(RATIO[a]["per"][d]) for a in ARM_NAMES for d in TARGETS},
             **{("boot_" + a + "_" + d): np.array([B4[a]["boot"][d]["mean"], B4[a]["boot"][d]["std"],
                                                   B4[a]["boot"][d]["lo"], B4[a]["boot"][d]["hi"]])
                for a in ARM_NAMES for d in TARGETS},
             meta_keys=np.array(list(mb.keys())))

    with open(RUNLOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(mblines) + "\n")
        fh.write("\nGATE VERDICTS\n")
        fh.write("G18_INCREMENT: " + "; ".join("%s=%s" % (a, INC[a]["verdict"]) for a in ARM_NAMES) + "\n")
        fh.write("G18_SPECIALIZATION_EFFECT: %s (delta4=%+.4f delta3=%+.4f)\n" % (spec_verdict4, spec4, spec3))
        fh.write("G18_LODO_EFFECT: %s (%d/3)\n" % (lodo_verdict, n_lodo_help))
        fh.write("G18_DELOCK: %s (ratio=%.4f vs 0.70*%.4f=%.4f)\n" %
                 (delock_verdict, delock_ratio, g18a_mean, GATE_DELOCK * g18a_mean))
        fh.write("wall=%.1fs\n" % wall)

    P("[done] report -> %s" % REPORT)
    P("[done] stats  -> %s" % STATS)
    P("[GATES] INC=%s | SPEC=%s(%+.4f) | LODO=%s(%d/3) | DELOCK=%s(%.4f)" %
      (";".join(INC[a]["verdict"].replace("COMPLEMENT_EXISTS", "COMP").replace("NO_COMPLEMENT", "NOCOMP")
                for a in ARM_NAMES), spec_verdict4, spec4, lodo_verdict, n_lodo_help,
       delock_verdict, delock_ratio))
    P("[done] wall=%.1fs" % wall)


if __name__ == "__main__":
    main()
