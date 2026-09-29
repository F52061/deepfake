# -*- coding: utf-8 -*-
"""
E3e - model (probe) weight spectral attribution.
Probe weight w lives in the SCALED (StandardScaler) space; decompose onto the PCA basis
of the scaled train features => weight-energy per variance-rank g_k = (w . e_k)^2.
Compare with the per-direction Fisher spectrum F_k computed in the SAME scaled space.
Runs on V_proj (head-input 768-d, the actual concat head's ViT half) and V (raw) as reference.
CPU-limited, no GPU.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import e3_common as C

SEP = "=" * 92
L = [SEP, "E3e - probe-weight spectral attribution (scaled-space PCA basis)", SEP]

p = C.load_probe()
y = p["y"]; tr = p["train_mask"].astype(bool); te = ~tr
ytr, yte = y[tr], y[te]

def analyze(ch, nbins=16):
    X = p[ch].astype(np.float64)
    Xtr, Xte = X[tr], X[te]
    clf, sc = C.fit_probe(Xtr, ytr)
    auc = C.score_auc(clf, sc, Xte, yte)
    w = clf.coef_[0]; w = w / (np.linalg.norm(w) + 1e-12)
    # PCA of SCALED train (coordinate space of w)
    Zs = sc.transform(Xtr)
    mu_s, E_s, lam_s = C.pca_basis(Zs)   # E_s columns = scaled-space variance directions (desc)
    g = (w @ E_s) ** 2                     # weight energy per PC (sums to ~1)
    # Fisher spectrum in same scaled space
    proj = (Zs - mu_s) @ E_s
    F = np.array([C.fisher_dir(proj[:, k], ytr) for k in range(X.shape[1])])
    # band summaries: fraction of weight energy in variance-ordered bands
    d = X.shape[1]
    edges = np.linspace(0, d, nbins + 1).astype(int)
    gb = np.zeros(nbins); Fb = np.zeros(nbins)
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        gb[b] = g[lo:hi].sum(); Fb[b] = F[lo:hi].sum()
    return dict(auc=auc, g=g, F=F, lam_s=lam_s, gb=gb, Fb=Fb, d=d, edges=edges)

for ch in ["V_proj", "V"]:
    r = analyze(ch)
    L.append(f"--- channel {ch} (dim={r['d']})  full-AUC={r['auc']:.4f} (same probe protocol)")
    L.append(f"    variance-rank of max weight-energy PC : {int(np.argmax(r['g']))}   "
             f"variance-rank of max Fisher PC: {int(np.argmax(r['F']))}")
    L.append(f"    Spearman(log10 lam_s, weight-energy g) = {C.spearman(np.log10(r['lam_s']+1e-30), r['g']):+.4f}   "
             f"(+1 => model leans on HIGH-variance dirs, -1 => low-variance dirs)")
    L.append(f"    cumulative weight energy in top-1/5/10/50/200 var ranks = "
             f"{r['g'][:1].sum():.4f}/{r['g'][:5].sum():.4f}/{r['g'][:10].sum():.4f}/"
             f"{r['g'][:50].sum():.4f}/{r['g'][:200].sum():.4f}")
    L.append(f"    weight-energy by {r['edges'].shape[0]-1} variance-ordered bands (rank0=largest var): "
             + " ".join(f"{100*x:.1f}%" for x in r["gb"]))
    L.append("")
    C.save_curves({f"E3e_{ch}_g": r["g"], f"E3e_{ch}_F": r["F"], f"E3e_{ch}_lam": r["lam_s"]})

C.write_report("\n".join(L))
print("\n".join(L)); print("\n[e3_e] done.")
