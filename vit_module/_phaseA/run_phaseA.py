# -*- coding: utf-8 -*-
"""
Phase A  : E0 / E1 / E2   (single fixed protocol; CPU-limited)
Source   : FF++ train = probe_feats.npz rows[train_mask]  (2200, video-clean)
Targets  : cd1 cd2 dfdcp ffiw wild  = feats_multi.npz      (300 each, balanced)
Protocol : standardize with source-mean/std ONLY (never re-fit on target);
           L2-logistic C=1e-3, lbfgs, single thread.
y        : 1=real, 0=fake  (AUC label-invariant)
Feature channels: V (768 raw PDI-ViT CLS), C (1024 raw CLIP CLS),
                  V_proj/C_proj (768 head-input projections, FF++ only)
"""
import os, sys, json
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

HERE  = os.path.dirname(os.path.abspath(__file__))
PROBE = np.load(os.path.join(HERE, "..", "_probe",  "probe_feats.npz"),  allow_pickle=True)
MULTI = np.load(os.path.join(HERE, "..", "_tsne",   "feats_multi.npz"),  allow_pickle=True)
OUT   = os.path.join(HERE, "phaseA_report.txt")
_rep  = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); _rep.append(s)

SEED = 0
DOMS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
TGT  = {d: MULTI["domain"] == d for d in DOMS}

def auc_fit_predict(Xtr, ytr, Xte, yte, C=1e-3):
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), clf, sc

def probe_split(chX, ch_name):
    """video-split in-domain AUC on probe itself."""
    tr = PROBE["train_mask"].astype(bool); te = ~tr
    return auc_fit_predict(chX[tr], PROBE["y"][tr], chX[te], PROBE["y"][te]), ch_name

# ====================================================================
log("="*78); log("PHASE A  —  E0: in-domain fusion case discrimination (FF++ video split)")
log("="*78)
y  = PROBE["y"]; tr = PROBE["train_mask"].astype(bool); te = ~tr

res = {}
for X, nm in [(PROBE["V_proj"], "V_proj"), (PROBE["C_proj"], "C_proj"),
              (np.hstack([PROBE["V_proj"], PROBE["C_proj"]]), "concat(V_proj,C_proj)"),
              (np.hstack([PROBE["V"], PROBE["C"]]), "concat(V,C)_raw")]:
    (a, clf, sc), nm2 = probe_split(X, nm)
    res[nm] = a
    log(f"  video-split AUC  {nm:24s} = {a:.4f}")
log(f"  (V_proj concat C_proj = 1536-d head-input analog; V,C raw concat = 1792-d)")

# MLP in-domain (capacity control: can a nonlinear head on concat beat V?)
for X, nm in [(PROBE["V_proj"], "MLP_V_proj"),
              (np.hstack([PROBE["V_proj"], PROBE["C_proj"]]), "MLP_concat(V_proj,C_proj)")]:
    scc = StandardScaler().fit(X[tr]); Xt = scc.transform(X[tr]); Xe = scc.transform(X[te])
    m = MLPClassifier(hidden_layer_sizes=(256,), activation="relu", alpha=1e-4,
                      max_iter=400, early_stopping=True, n_iter_no_change=12,
                      validation_fraction=0.15, random_state=SEED)
    m.fit(Xt, y[tr])
    res[nm] = roc_auc_score(y[te], m.predict_proba(Xe)[:, 1])
    log(f"  video-split AUC  {nm:24s} = {res[nm]:.4f}")
log(f"  CASE: linear concat vs linear V_proj   -> {'II (fusion helps in-domain; transfer overfit)' if res['concat(V_proj,C_proj)']>res['V_proj']+1e-3 else 'I (linear fusion adds ~nothing in-domain)'}")
log(f"  CASE: MLP concat vs MLP V_proj         -> {'capacity exists (nonlinear fusion could exploit complementarity)' if res['MLP_concat(V_proj,C_proj)']>res['MLP_V_proj']+1e-3 else 'even nonlinear fusion adds ~nothing in-domain'}")

# ====================================================================
log(""); log("="*78); log("PHASE A  —  E1: cross-domain transfer + label-free predictors")
log("="*78)
# ---- transfer probe (single protocol) per channel V / C ----
Xsrc = {"V": PROBE["V"][tr], "C": PROBE["C"][tr]}
ysrc = y[tr]
transfer = {ch: {} for ch in ["V", "C"]}
weights  = {}
for ch in ["V", "C"]:
    _, clf, sc = auc_fit_predict(Xsrc[ch], ysrc, Xsrc[ch], ysrc)  # refit to grab clf
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])                # unit decision axis (std space)
    weights[ch] = (clf, sc, w)
    for d in DOMS:
        Xt = MULTI[ch][TGT[d]]; yt = MULTI["y"][TGT[d]]
        a, _, _ = auc_fit_predict(Xsrc[ch], ysrc, Xt, yt)
        transfer[ch][d] = a
        log(f"  transfer AUC   {ch} -> {d:6s} = {a:.4f}")

def spearman(xs, ys):
    rx = {v:i for i,v in enumerate(sorted(xs))}; ry = {v:i for i,v in enumerate(sorted(ys))}
    n = len(xs)
    if n < 3: return float("nan")
    d = np.mean([(rx[x]-ry[y])**2 for x,y in zip(xs,ys)])
    return 1 - 6*d/(n*(n*n-1))

# ---- label-free predictors per domain (source = probe train; target features unlabeled) ----
def topk_overlap(Xsrc, Xt, k):
    """aligned top-k principal-subspace cosine overlap in FEATURE space.
    Uses right singular vectors (d x k) so different sample counts are comparable."""
    Xc = Xsrc - Xsrc.mean(0)
    Us = np.linalg.svd(Xc, full_matrices=False)[2].T[:, :k]   # d x k (PCA loadings)
    Xc2 = Xt - Xt.mean(0)
    Ut = np.linalg.svd(Xc2, full_matrices=False)[2].T[:, :k]  # d x k
    return np.sum((Us.T @ Ut)**2) / k

preds = {d: {} for d in DOMS}
for ch in ["V", "C"]:
    Xs = Xsrc[ch]
    for d in DOMS:
        Xt = MULTI[ch][TGT[d]]
        for k in [5, 10, 20, 50]:
            preds[d][f"ovl_{ch}_{k}"] = topk_overlap(Xs, Xt, k)
        # decision-axis alignment (target variance fraction along source w)
        clf, sc, w = weights[ch]
        Z = sc.transform(Xt)
        Ct = np.cov(Z, rowvar=False)
        preds[d][f"align_{ch}"] = float(w @ Ct @ w) / float(np.trace(Ct))
        # oracle ceiling: |cos(source w, target LDA dir)|  (USES target labels -> upper bound)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, MULTI["y"][TGT[d]])
        ul = lda.coef_[0]; ul = ul / (np.linalg.norm(ul)+1e-12)
        preds[d][f"oracle_{ch}"] = abs(float(w @ ul))

log("")
log("  per-domain predictors (label-free except oracle) + measured transfer AUC(V):")
hdr = f"  {'dom':6s} {'AUC_V':>7s} {'AUC_C':>7s} | " + " ".join(f"{kk:>10s}" for kk in ["ovl_V_10","ovl_V_20","ovl_C_10","ovl_C_20","align_V","align_C","oracle_V","oracle_C"])
log(hdr); log("  " + "-"*(len(hdr)-2))
for d in DOMS:
    row = f"  {d:6s} {transfer['V'][d]:7.4f} {transfer['C'][d]:7.4f} | "
    for kk in ["ovl_V_10","ovl_V_20","ovl_C_10","ovl_C_20","align_V","align_C","oracle_V","oracle_C"]:
        row += f"{preds[d][kk]:10.3f} "
    log(row)

log("")
log("  Spearman rho(predictor, transfer AUC) across 5 target domains:")
aucV = [transfer["V"][d] for d in DOMS]
aucC = [transfer["C"][d] for d in DOMS]
for kk in sorted(preds[DOMS[0]].keys()):
    xs = [preds[d][kk] for d in DOMS]
    log(f"    {kk:22s} vs AUC_V rho={spearman(xs, aucV):+.3f}   vs AUC_C rho={spearman(xs, aucC):+.3f}")

# ====================================================================
log(""); log("="*78); log("PHASE A  —  E2: regime search  (is there a far-regime where C leads?)")
log("="*78)
# Mahalanobis in V effective subspace (feature-space PCs) of SOURCE train manifold
Xc = Xsrc["V"] - Xsrc["V"].mean(0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
U = Vt.T                    # d x d right singular vectors (feature directions)
mu = Xsrc["V"].mean(0)
for k in [5, 10]:
    Uk = U[:, :k]; lam = (S[:k]**2)/(len(Xsrc["V"])-1); lam[lam<1e-12]=1e-12
    dist = {}
    for d in DOMS:
        z = (MULTI["V"][TGT[d]] - mu) @ Uk
        dist[d] = np.sum(z*z / lam, axis=1)
    log(f"\n  distance scale: Mahalanobis^2 in source-V top-{k} subspace")
    # per-domain median split (near/far within each domain), then pool halves
    half_all = {q: {ch: {"s": [], "y": []} for ch in ["V","C"]} for q in ["near","far"]}
    for d in DOMS:
        gidx = np.nonzero(TGT[d])[0]
        med = np.median(dist[d])
        for q, sel300 in [("near", dist[d] <= med), ("far", dist[d] > med)]:
            sel = gidx[sel300]
            for ch in ["V", "C"]:
                clf, sc, w = weights[ch]
                s = clf.decision_function(sc.transform(MULTI[ch][sel]))
                half_all[q][ch]["s"] += list(s); half_all[q][ch]["y"] += list(MULTI["y"][sel])
    half_auc = {}
    log(f"  within-domain near/far halves, pooled across 5 domains (n={len(half_all['near']['V']['s'])} each):")
    for q in ["near","far"]:
        row = f"    {q:5s}"
        for ch in ["V","C"]:
            s = np.array(half_all[q][ch]["s"]); yy = np.array(half_all[q][ch]["y"])
            half_auc[(q,ch)] = roc_auc_score(yy, s)
            row += f"  AUC({ch})={half_auc[(q,ch)]:.4f}"
        row += f"   C-V={half_auc[(q,'C')]-half_auc[(q,'V')]:+.4f}"
        log(row)
    # per-domain near/far deltas
    log(f"  per-domain (k={k}) near/far AUC(V), AUC(C):")
    for d in DOMS:
        gidx = np.nonzero(TGT[d])[0]
        med = np.median(dist[d])
        out = [f"    {d:6s}"]
        for q, sel300 in [("n", dist[d] <= med), ("f", dist[d] > med)]:
            sel = gidx[sel300]
            a = {}
            for ch in ["V","C"]:
                clf, sc, w = weights[ch]
                a[ch] = roc_auc_score(MULTI["y"][sel], clf.decision_function(sc.transform(MULTI[ch][sel])))
            out.append(f"{q}:V={a['V']:.3f} C={a['C']:.3f}")
        log(" ".join(out))

# global-tercile pooled (absolute farness across all targets, distance from the SAME source manifold)
Uk_g = U[:, :10]; lam_g = (S[:10]**2)/(len(Xsrc["V"])-1); lam_g[lam_g<1e-12]=1e-12
# per-domain distance arrays + index offsets
dchunks = []; off = 0
for d in DOMS:
    z = (MULTI["V"][TGT[d]] - mu) @ Uk_g
    dd = np.sum(z*z/lam_g, axis=1)
    dchunks.append((d, dd, np.nonzero(TGT[d])[0]))
gdd  = np.concatenate([c[1] for c in dchunks])
t1, t2 = np.quantile(gdd, [1/3, 2/3])
log("")
log(f"  global-tercile pooled (absolute source-V distance, all 5 targets n={len(gdd)}):")
for q, lo, hi in [("near", -np.inf, t1), ("mid", t1, t2), ("far", t2, np.inf)]:
    scores = {ch: [] for ch in ["V", "C"]}; yy = []
    for d, dd, idx in dchunks:
        sel = idx[(dd > lo) & (dd <= hi)]
        if len(sel) == 0: continue
        for ch in ["V", "C"]:
            clf, sc, w = weights[ch]
            scores[ch] += list(clf.decision_function(sc.transform(MULTI[ch][sel])))
        yy += list(MULTI["y"][sel])
    out = f"    {q:5s} n={len(yy):4d}"
    for ch in ["V", "C"]:
        out += f"   AUC({ch})={roc_auc_score(yy, scores[ch]):.4f}"
    out += f"   C-V={roc_auc_score(yy, scores['C'])-roc_auc_score(yy, scores['V']):+.4f}"
    log(out)
log("")
log("  NOTE: E2 verdict (regime exists iff a distance-defined subset has AUC(C) > AUC(V)).")
log("  Also report whether AUC(V) itself degrades with distance (V-fragility).")

# save json + report
import pickle
with open(os.path.join(HERE, "phaseA_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_rep))
summ = {"E0_in_domain": {k: float(v) for k, v in res.items()},
        "transfer_V": transfer["V"], "transfer_C": transfer["C"],
        "preds": {d: {k: float(v) for k, v in preds[d].items()} for d in DOMS}}
with open(os.path.join(HERE, "phaseA_summary.json"), "w") as f:
    json.dump(summ, f, indent=1)
log("\nsaved: phaseA_report.txt, phaseA_summary.json")
