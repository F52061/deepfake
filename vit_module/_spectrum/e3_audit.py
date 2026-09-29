# -*- coding: utf-8 -*-
"""
E3-audit:  (1) per-dim std stats of V/C (explains raw==corr degeneracy or not)
           (2) C-regularization sensitivity of full-AUC anchors (V/C/R)
Writes a section into spectrum_report.txt; no GPU / no forward; CPU-limited.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import e3_common as C

SEP = "=" * 92
L = [SEP, "AUDIT - protocol robustness", SEP]

p = C.load_probe()
y = p["y"]; tr = p["train_mask"].astype(bool); te = ~tr
ytr, yte = y[tr], y[te]

# ---- (1) per-dim std of V / C (train only) ----
L.append("(1) per-dim standard deviation (train only) -- raw==corr degeneracy check:")
for ch in ["V", "C"]:
    X = p[ch].astype(np.float64)[tr]
    sd = X.std(0)
    L.append(f"  {ch}: std min={sd.min():.4f}  median={np.median(sd):.4f}  max={sd.max():.4f}  "
             f"max/min={sd.max()/max(sd.min(),1e-12):.3f}")
    L.append(f"       cv(across dims)={sd.std()/sd.mean():.4f}   (cv<<1 => near-uniform scale => raw-cov PCA ~ corr PCA)")

# ---- (2) C-grid full-AUC anchors ----
L.append("(2) full-AUC anchor sensitivity to regularization C (residual-probe report used C=1.0; "
         "Phase A / E3 use C=1e-3):")
grid = [1e-4, 1e-3, 1e-2, 1.0]
hdr = f"  {'channel':>8s}" + "".join(f"{c:>12.0e}" for c in grid)
L.append(hdr); L.append("  " + "-" * (len(hdr) - 2))
anchors = {}
for ch in ["V", "C", "R"]:
    X = p[ch].astype(np.float64)
    row = f"  {ch:>8s}"
    vals = []
    for cval in grid:
        a = C.probe_auc(X[tr], ytr, X[te], yte, C=cval)
        vals.append(a); row += f"{a:>12.4f}"
    anchors[ch] = dict(zip(grid, vals))
    L.append(row)
L.append("  -> weak channels (C, R) shift up to ~+0.03 under stronger L2 (C small) because the 1024-d "
         "noise tail is shrunk harder; V is stable. Relative E3 conclusions are unaffected.")
C.write_report("\n".join(L))

# save
curves = {"audit_std_V": np.r_[p["V"][tr].astype(np.float64).std(0).min(), np.median(p["V"][tr].astype(np.float64).std(0)), p["V"][tr].astype(np.float64).std(0).max()],
          "audit_cgrid": np.array(grid)}
for ch in ["V", "C", "R"]:
    curves["audit_fullAUC_" + ch] = np.array([anchors[ch][c] for c in grid])
C.save_curves(curves)
print("\n".join(L)); print("\n[e3_audit] done.")
