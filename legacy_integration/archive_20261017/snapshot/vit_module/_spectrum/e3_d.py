# -*- coding: utf-8 -*-
"""
E3d - which variance band transfers cross-domain?
Basis: FF++ train (probe npz, video split) raw-cov PCA per channel (V: 768, C: 1024).
Bands: full | topK(K=50) | tailK(K=50).  Probe fit on FF++ train band features,
evaluated per target domain (feats_multi.npz: cd1/cd2/dfdcp/ffiw/wild, 300 each).
Protocol identical to Phase A transfer (C=1e-3). CPU-limited, no GPU.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import e3_common as C

SEP = "=" * 92
K0 = 50
L = [SEP, f"E3d - cross-domain band transfer (bands: full | top{K0} | tail{K0}; probe C=1e-3)", SEP]

p = C.load_probe(); m = C.load_multi()
y = p["y"]; tr = p["train_mask"].astype(bool); te = ~tr
ytr = y[tr]

table = {}
for ch in ["V", "C"]:
    X = p[ch].astype(np.float64)
    mu, E, lam = C.pca_basis(X[tr])
    Xtr_full = X[tr]
    Xtr_top  = (X[tr] - mu) @ E[:, :K0]
    Xtr_tail = (X[tr] - mu) @ E[:, K0:]
    Xte_full = (X[te] - mu)
    Xte_top  = (X[te] - mu) @ E[:, :K0]
    Xte_tail = (X[te] - mu) @ E[:, K0:]
    # FF++ in-domain band anchors (video-clean test)
    a_f  = C.probe_auc(Xtr_full, ytr, Xte_full, y[te])
    a_t  = C.probe_auc(Xtr_top,  ytr, Xte_top,  y[te])
    a_tl = C.probe_auc(Xtr_tail, ytr, Xte_tail, y[te])
    L.append(f"--- channel {ch}  FF++ in-domain (video split): full={a_f:.4f} top{K0}={a_t:.4f} tail{K0}={a_tl:.4f}")
    row = f"  {'domain':>6s}  {'full':>7s}  {'top'+str(K0):>8s}  {'tail'+str(K0):>8s}  {'tail-full':>10s}"
    L.append(row); L.append("  " + "-" * (len(row) - 2))
    chres = {}
    for d in C.DOMS:
        g = m["domain"] == d
        Xd = m[ch][g].astype(np.float64)
        # center with SOURCE mu, project with SOURCE basis, scale NOT refit (only LR scaler on train inside probe_auc)
        Xd_full = Xd
        Xd_top  = (Xd - mu) @ E[:, :K0]
        Xd_tail = (Xd - mu) @ E[:, K0:]
        r = {"full": C.probe_auc(Xtr_full, ytr, Xd_full, m["y"][g]),
             "top":  C.probe_auc(Xtr_top,  ytr, Xd_top,  m["y"][g]),
             "tail": C.probe_auc(Xtr_tail, ytr, Xd_tail, m["y"][g])}
        chres[d] = r
        L.append(f"  {d:>6s}  {r['full']:7.4f}  {r['top']:8.4f}  {r['tail']:8.4f}  {r['tail']-r['full']:+10.4f}")
    table[ch] = chres
    L.append("")

L.append("Reading: tail-full > 0 means keeping only the LOW-variance tail (drop top-50 var dirs) transfers "
         "better than full features; tail-full < 0 means the top variance directions carry the transferable signal.")
C.write_report("\n".join(L))

curves = {}
for ch in ["V", "C"]:
    curves["E3d_" + ch + "_full"] = np.array([table[ch][d]["full"] for d in C.DOMS])
    curves["E3d_" + ch + "_top"] = np.array([table[ch][d]["top"] for d in C.DOMS])
    curves["E3d_" + ch + "_tail"] = np.array([table[ch][d]["tail"] for d in C.DOMS])
curves["E3d_domains"] = np.array(C.DOMS)
C.save_curves(curves)
print("\n".join(L)); print("\n[e3_d] done.")
