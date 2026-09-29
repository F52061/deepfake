# -*- coding: utf-8 -*-
"""Block G3: sample-level bootstrap CI for key AUC numbers (no new training).

Protocol (task block 4):
  * All LR probes are fit ONCE on probe_feats.npz train_mask (n=2200).
  * Settings: V-only (raw V 768), C-only (raw C 1024), concat LR (np.c_[V,C] 1792).
  * Evaluation: (i) within-domain = probe npz test 800 rows;
                (ii) cross-domain = feats_multi 5 target domains x 300 rows each
                (ffpp 800-row block in feats_multi excluded per Phase-A protocol).
  * bootstrap: with-replacement over evaluation SAMPLES, B=2000, fixed seed=0,
                95% CI = 2.5/97.5 percentiles of roc_auc.
  * CAVEAT (reported verbatim): sample-level bootstrap; within-video frame
    correlation makes these CIs optimistic.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = r"E:/Cross-domain_authentication_verification/Next_work/M2F2_Det-main-hyy"
PNPZ = os.path.join(BASE, "vit_module", "_probe", "probe_feats.npz")
MNPZ = os.path.join(BASE, "vit_module", "_tsne", "feats_multi.npz")
OUT  = os.path.join(BASE, "vit_module", "_head", "head_report.txt")

B = 2000
BS_SEED = 0
LR_SEED = 0
LR_C = 1e-3   # E0/E3d/E4 probe protocol (matches published anchors; residual_probe C=1.0 differs)
TARGET_DOMAINS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]  # ffpp excluded

L = []
def log(s=""):
    L.append(str(s))
    print(s, flush=True)

def fit_once(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    clf = LogisticRegression(C=LR_C, max_iter=3000, solver="lbfgs", random_state=LR_SEED)
    clf.fit(Xs, ytr)
    return sc, clf

def score(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]

def auc_ci(y, s):
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(BS_SEED)
    n = len(y)
    idx = np.arange(n)
    vals = []
    for _ in range(B):
        ii = rng.choice(idx, size=n, replace=True)
        if len(np.unique(y[ii])) < 2:
            continue
        try:
            vals.append(roc_auc_score(y[ii], s[ii]))
        except ValueError:
            pass
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, lo, hi

def main():
    log("=" * 80)
    log("BLOCK G3 - bootstrap CI for key AUC numbers (sample-level, no new training)")
    log("=" * 80)

    pd = np.load(PNPZ)
    V = pd["V"]; C = pd["C"]; y = pd["y"]; tm = pd["train_mask"]
    tr, te = tm, ~tm
    Vtr, Vte = V[tr], V[te]
    Ctr, Cte = C[tr], C[te]
    ytr, yte = y[tr], y[te]
    log(f"[G3] probe npz: train n={Vtr.shape[0]} test n={Vte.shape[0]} "
        f"y1={int((yte==1).sum())} y0={int((yte==0).sum())}")

    md = np.load(MNPZ)
    dom = md["domain"]
    log(f"[G3] feats_multi: total={len(dom)} domains in file={list(md['domains'])}")
    for dn in md["domains"]:
        m = dom == dn
        log(f"    domain {str(dn):<6} n={int(m.sum())} y1={int((md['y'][m]==1).sum())} y0={int((md['y'][m]==0).sum())}")

    # build feature matrices for the 5 target domains
    domV, domC, domy = {}, {}, {}
    for dn in TARGET_DOMAINS:
        m = dom == dn
        domV[dn] = md["V"][m]
        domC[dn] = md["C"][m]
        domy[dn] = md["y"][m]

    def show(label, yt, st, n):
        auc, lo, hi = auc_ci(yt, st)
        log(f"    {label:<46}{n:>5}  AUC={auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (width={hi-lo:.4f})")

    # ---------------- IN-DOMAIN ----------------
    log("\n[G3.1] IN-DOMAIN: probe npz test 800 (fit on train_mask 2200)")
    # V-only
    sc, clf = fit_once(Vtr, ytr)
    sv = score(sc, clf, Vte)
    show("V-only-LR", yte, sv, len(yte))
    # C-only
    sc, clf = fit_once(Ctr, ytr)
    sc_ = score(sc, clf, Cte)
    show("C-only-LR", yte, sc_, len(yte))
    # concat
    Xtr = np.c_[Vtr, Ctr]; Xte = np.c_[Vte, Cte]
    sc, clf = fit_once(Xtr, ytr)
    sconcat = score(sc, clf, Xte)
    show("concat-LR (np.c_[V,C])", yte, sconcat, len(yte))

    # ---------------- CROSS-DOMAIN ----------------
    log("\n[G3.2] CROSS-DOMAIN: feats_multi 5 target domains x 300 (fit on probe train 2200)")
    results = {"V-only": {}, "C-only": {}, "concat": {}}
    # fit each LR probe once on probe train; reuse across domains
    scV, clfV = fit_once(Vtr, ytr)
    scC, clfC = fit_once(Ctr, ytr)
    scX, clfX = fit_once(np.c_[Vtr, Ctr], ytr)
    for dn in TARGET_DOMAINS:
        n = len(domy[dn])
        s = score(scV, clfV, domV[dn])
        auc, lo, hi = auc_ci(domy[dn], s)
        results["V-only"][dn] = (auc, lo, hi)
        s = score(scC, clfC, domC[dn])
        auc, lo, hi = auc_ci(domy[dn], s)
        results["C-only"][dn] = (auc, lo, hi)
        s = score(scX, clfX, np.c_[domV[dn], domC[dn]])
        auc, lo, hi = auc_ci(domy[dn], s)
        results["concat"][dn] = (auc, lo, hi)

    hdr = f"    {'setting':<10}" + "".join(f"{dn:>26}" for dn in TARGET_DOMAINS)
    log(hdr)
    for setting in ("V-only", "C-only", "concat"):
        row = f"    {setting:<10}"
        for dn in TARGET_DOMAINS:
            auc, lo, hi = results[setting][dn]
            row += f"  {auc:.4f}[{lo:.4f},{hi:.4f}]"
        log(row)

    # compact per-domain CI widths for concat
    log("\n[G3.3] concat-LR cross-domain AUC + CI width (the E4 key new number)")
    for dn in TARGET_DOMAINS:
        auc, lo, hi = results["concat"][dn]
        log(f"    concat {dn:<6} AUC={auc:.4f} CI[{lo:.4f},{hi:.4f}] width={hi-lo:.4f}")

    log("\n[G3] SEED declaration: LR = StandardScaler + LogisticRegression(C=1e-3, lbfgs, "
        "max_iter=3000, random_state=0) fit once on probe train_mask 2200 (E0/E3d/E4 protocol); "
        "bootstrap RNG=default_rng(0); B=2000; CI = 2.5/97.5 percentile of 2000 resampled roc_auc.")
    log("[G3] CAVEAT: sample-level bootstrap; within-video frame correlation makes these "
        "confidence intervals optimistic (anti-conservative).")
    log("[g3_done]")

if __name__ == "__main__":
    main()
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
