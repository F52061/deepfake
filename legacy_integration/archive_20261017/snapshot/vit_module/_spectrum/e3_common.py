# -*- coding: utf-8 -*-
"""
E3 common helpers  (variance-spectrum attribution; CPU-limited, no GPU, no forward)
Shared probe protocol (identical to Phase A):
    StandardScaler().fit(train) -> LogisticRegression(C=1e-3, max_iter=3000, lbfgs)
PCA convention (feature-space loadings = RIGHT singular vectors):
    E = np.linalg.svd(Xc, full_matrices=False)[2].T   # d x d, columns descending variance
    projection of (n x d) data onto top-k directions = (X-mu) @ E[:, :k]
y: real=1, fake=0 (AUC is label-invariant under swap of positives).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE   = os.path.dirname(os.path.abspath(__file__))
PROBE  = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
MULTI  = os.path.join(HERE, "..", "_tsne", "feats_multi.npz")
REPORT = os.path.join(HERE, "spectrum_report.txt")
CURVES = os.path.join(HERE, "spectrum_curves.npz")
SEED   = 0
DOMS   = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]


def rng():
    return np.random.default_rng(SEED)


def load_probe():
    return np.load(PROBE, allow_pickle=True)


def load_multi():
    return np.load(MULTI, allow_pickle=True)


def probe_auc(Xtr, ytr, Xte, yte, C=1e-3, max_iter=3000):
    """Standard probe on raw features X (already projected if any). Returns AUC."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def fit_probe(Xtr, ytr, C=1e-3, max_iter=3000):
    """Fit standard probe, return (clf, scaler)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc


def score_auc(clf, sc, Xte, yte):
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def pca_basis(Xtr):
    """Center-only (raw covariance) PCA fit on train.
    Returns mu (d,), E (d x d, columns by DESCENDING eigenvalue), lam (d,)."""
    mu = np.asarray(Xtr, dtype=np.float64).mean(axis=0)
    Xc = np.asarray(Xtr, dtype=np.float64) - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    E = Vt.T
    n = Xc.shape[0]
    lam = (S ** 2) / (n - 1.0)
    return mu, E, lam


def project(X, mu, E, cols):
    """(X - mu) @ E[:, cols]  -> (n x len(cols)) float64."""
    X = np.asarray(X, dtype=np.float64)
    return (X - mu) @ E[:, cols]


def fisher_dir(z, y, eps=1e-12):
    """Per-direction Fisher = (mean diff)^2 / within-class pooled var."""
    y = np.asarray(y)
    yb = (y == 1)
    m1 = z[yb].mean(); m0 = z[~yb].mean()
    v1 = z[yb].var();  v0 = z[~yb].var()
    return float((m1 - m0) ** 2 / ((v1 + v0) / 2.0 + eps))


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    n = len(xs)
    if n < 3:
        return float("nan")
    d = float(np.mean((rx - ry) ** 2))
    return float(1 - 6 * d / (n * (n * n - 1)))


def topk_tailk_curve(Xtr, ytr, Xte, yte, Ks, mode="raw"):
    """Truncation causal curve (E3b) -- PROJECTS onto the PCA basis (variance-ordered).
    mode='raw' : raw-covariance PCA (center only) on Xtr.
    mode='corr': StandardScaler(train) then PCA (correlation PCA).
    top-K := (X-mu) @ E[:, :K]   (keep K LARGEST-variance directions)
    tail-K:= (X-mu) @ E[:, K:]   (drop K largest, keep the rest)
    Returns dict: ('top',K)->AUC, ('tail',K)->AUC."""
    Xtr = np.asarray(Xtr, dtype=np.float64); Xte = np.asarray(Xte, dtype=np.float64)
    if mode == "raw":
        mu, E, lam = pca_basis(Xtr)
        Xtr_c = Xtr - mu; Xte_c = Xte - mu
    else:
        sc0 = StandardScaler().fit(Xtr)
        Xtr_c = sc0.transform(Xtr); Xte_c = sc0.transform(Xte)
        mu2, E, lam = pca_basis(Xtr_c)
        Xtr_c = Xtr_c - mu2; Xte_c = Xte_c - mu2
    d = E.shape[0]
    res = {}
    for K in Ks:
        if 1 <= K <= d:
            res[("top", K)] = probe_auc(Xtr_c @ E[:, :K], ytr, Xte_c @ E[:, :K], yte)
        if 0 <= K < d:
            res[("tail", K)] = probe_auc(Xtr_c @ E[:, K:], ytr, Xte_c @ E[:, K:], yte)
    res["_meta"] = dict(d=d)
    return res


def cum_disc_dims(F, lam, fractions=(0.10, 0.50, 0.90)):
    """Directions (by descending F) needed to reach q of total F, and energy carried.
    Returns list of dicts per fraction: n_dims, energy_frac."""
    idx = np.argsort(F)[::-1]
    cumF = np.cumsum(F[idx])
    totF = cumF[-1] if cumF[-1] > 0 else 1.0
    totE = lam.sum() if lam.sum() > 0 else 1.0
    out = []
    for q in fractions:
        target = q * totF
        n = int(np.searchsorted(cumF, target) + 1)
        n = min(n, len(F))
        en = float(lam[idx[:n]].sum() / totE)
        out.append(dict(fraction=q, n_dims=int(n), energy_frac=float(en)))
    return out


def bin_spec(F, lam, nbins=16):
    """Split the d variance-ordered directions into ~nbins equal-size bins.
    Returns list of (bin_no, rank_lo, rank_hi, mean_loglam, mean_F, energy_frac, n)."""
    d = len(F)
    edges = np.linspace(0, d, nbins + 1).astype(int)
    out = []
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        Fb = F[lo:hi]; lb = lam[lo:hi]
        out.append(dict(bin=b + 1, rank_lo=lo, rank_hi=hi - 1, n=int(hi - lo),
                        mean_loglam=float(np.mean(np.log10(lb + 1e-30))),
                        mean_F=float(np.mean(Fb)),
                        max_F=float(Fb.max()),
                        energy_frac=float(lb.sum() / (lam.sum() + 1e-30))))
    return out


def write_report(text):
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def reset_report(header):
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(header + "\n")
    try:
        if os.path.exists(CURVES):
            os.remove(CURVES)
    except OSError:
        pass


def save_curves(payload):
    """Merge payload dict into spectrum_curves.npz."""
    existing = {}
    if os.path.exists(CURVES):
        try:
            with np.load(CURVES, allow_pickle=True) as d:
                for k in d.files:
                    existing[k] = d[k]
        except Exception:
            existing = {}
    existing.update(payload)
    np.savez(CURVES, **existing)
