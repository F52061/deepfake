# -*- coding: utf-8 -*-
"""G20-pre: the G16 concat variants that were NEVER tested -- low-lock-only concat.
Tests the prediction of the concat-drag law: cat(b9,final) should be the BEST concat.
CPU-only, reads _g16/layer_feats.npz. Formula from _g12/run_g12.py block B (ratio)
and C-block probe (StandardScaler+LR C=1e-3, fit on FF++ train 2200).
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","JOBLIB_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "..", "_g16", "layer_feats.npz"), allow_pickle=True)
y = np.asarray(d["y"]).reshape(-1); dom = np.asarray(d["domain"]).astype(str)
tr = (np.asarray(d["split"]).astype(str) == "train"); te = ~tr
TARGETS = ["cd1","cd2","dfdcp","ffiw","wild"]
V4 = ["cd1","cd2","dfdcp","wild"]

F = {k: np.asarray(d[k], dtype=np.float64) for k in d.files
     if k.startswith(("cls_","mp_"))}
F["cat_b9_final"]    = np.concatenate([F["cls_b9"], F["cls_final"]], axis=1)   # NEW
F["cat_mp9_final"]   = np.concatenate([F["mp_b9"],  F["cls_final"]], axis=1)   # NEW
F["cat_b6_final"]    = np.concatenate([F["cls_b6"], F["cls_final"]], axis=1)   # G16 ref
F["cat_cls369"]      = np.concatenate([F["cls_b3"], F["cls_b6"], F["cls_b9"]], axis=1)  # G16 ref

def ratio(X):
    Xtr = X[tr]; ys = y[tr]
    mu_src = Xtr.mean(0)
    s = float(np.sqrt(np.var(Xtr, axis=0, ddof=1).mean()))
    cg = float(np.linalg.norm(Xtr[ys==0].mean(0) - Xtr[ys==1].mean(0)) / s)
    rt = {dm: float(np.linalg.norm(X[dom==dm].mean(0) - mu_src)/s)/cg for dm in TARGETS}
    return s, cg, float(np.mean(list(rt.values()))), rt

def cd_auc(X, C=1e-3):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
    out = {}
    for dm in TARGETS:
        m = (dom == dm)
        out[dm] = float(roc_auc_score(y[m], clf.decision_function(sc.transform(X[m]))))
    return out

ORDER = ["cls_final","cat_b9_final","cat_mp9_final","cat_b6_final","cat_cls369"]
print("="*118)
print("G20-pre  LOW-LOCK-ONLY CONCAT -- the G16 variants that were never run")
print("="*118)
print(f"{'feature':<16}{'dim':>6}{'s':>9}{'class_gap':>11}{'ratio':>9}  |  " +
      "  ".join(f"{t:>7}" for t in V4) + f"{'mean4':>8}")
rows = {}
for k in ORDER:
    s, cg, rtall, rt = ratio(F[k])
    a = cd_auc(F[k])
    m4 = float(np.mean([a[t] for t in V4]))
    rows[k] = (s, cg, rtall, rt, a, m4)
    print(f"{k:<16}{F[k].shape[1]:>6}{s:>9.4f}{cg:>11.4f}{rtall:>9.4f}  |  " +
          "  ".join(f"{a[t]:>7.4f}" for t in V4) + f"{m4:>8.4f}")

base = rows["cls_final"][5]
print("\ndelta mean4 vs cls_final (C=1e-3):")
for k in ORDER[1:]:
    d4 = rows[k][5] - base
    per = "  ".join(f"{rows[k][4][t]-rows['cls_final'][4][t]:+.4f}" for t in V4)
    flag = "  <== BEST" if rows[k][5] == max(rows[x][5] for x in ORDER) else ""
    print(f"  {k:<16} mean4 {d4:+.4f}   per-domain: {per}{flag}")

best = max(ORDER, key=lambda k: rows[k][5])
print(f"\nBEST_CONCAT = {best}  mean4={rows[best][5]:.4f}  "
      f"(G16 best was cat_b6_final 0.8429)")
pred = rows["cat_b9_final"][5] >= rows["cat_b6_final"][5]
print(f"PREDICTION (concat-drag law: low-lock-only concat is best) -> "
      f"{'SUPPORTED' if pred else 'REFUTED'}")
np.savez(os.path.join(HERE, "ratio_lowlock_stats.npz"),
         names=np.array(ORDER),
         vals=np.array([[rows[k][0],rows[k][1],rows[k][2],rows[k][5]]+
                        [rows[k][4][t] for t in V4] for k in ORDER]))
print("[saved] ratio_lowlock_stats.npz")
