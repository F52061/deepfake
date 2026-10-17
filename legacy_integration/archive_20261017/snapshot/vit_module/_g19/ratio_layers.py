# -*- coding: utf-8 -*-
"""G19d-pre: E6 domain-lock ratio for G16 layer features + concat variants.
CPU-only, reads existing _g16/layer_feats.npz. No GPU, no training.
Formula copied verbatim from _g12/run_g12.py block B (L392-405).
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","JOBLIB_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
np.set_printoptions(precision=4, suppress=True)

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ  = os.path.join(HERE, "..", "_g16", "layer_feats.npz")
TARGETS = ["cd1","cd2","dfdcp","ffiw","wild"]

d = np.load(NPZ, allow_pickle=True)
print("[keys]", sorted({k for k in d.files}))
for k in ("y","domain","split","vids","paths"):
    if k in d.files:
        a = d[k]
        u = np.unique(a)
        print(f"  {k}: shape={a.shape} dtype={a.dtype} uniq{'(%d)'%len(u) if len(u)<30 else ''}={u[:30]}")

y_all = np.asarray(d["y"]).reshape(-1)
dom   = np.asarray(d["domain"]).astype(str).reshape(-1)
spl   = np.asarray(d["split"]).astype(str).reshape(-1) if "split" in d.files else None
FEAT_KEYS = [k for k in ("cls_final","cls_b3","cls_b6","cls_b9",
                         "mp_b3","mp_b6","mp_b9") if k in d.files]
print("[dim]", {k: d[k].shape for k in FEAT_KEYS})

# --- source train rows = FF++ probe-train (2200) ---
tr = (spl == "train")
print(f"[mask] split==train n={tr.sum()}; y uniq in train={np.unique(y_all[tr])}")
src_mu_rows = tr

def ratio_block(X, name):
    Xtr = X[src_mu_rows]
    ys  = y_all[src_mu_rows]
    mu_src  = Xtr.mean(axis=0)
    mu_real = Xtr[ys == 1].mean(axis=0)
    mu_fake = Xtr[ys == 0].mean(axis=0)
    s = float(np.sqrt(np.var(Xtr, axis=0, ddof=1).mean()))
    cg = float(np.linalg.norm(mu_fake - mu_real) / s)
    dg, rt = {}, {}
    for dm in TARGETS:
        m = (dom == dm)
        if m.sum() == 0: dg[dm]=np.nan; rt[dm]=np.nan; continue
        dg[dm] = float(np.linalg.norm(X[m].mean(axis=0) - mu_src) / s)
        rt[dm] = dg[dm] / cg
    return dict(name=name, s=s, cg=cg, dg=dg, rt=rt,
                rt_all=float(np.nanmean([rt[x] for x in TARGETS])),
                rt_v4=float(np.nanmean([rt[x] for x in ["cd1","cd2","dfdcp","wild"]])))

blocks = {}
for k in FEAT_KEYS:
    blocks[k] = np.asarray(d[k], dtype=np.float64)

# concat variants tested in G16 (C-block)
cats = {
    "cat_cls369":     ["cls_b3","cls_b6","cls_b9"],
    "cat_b6_final":   ["cls_b6","cls_final"],
    "cat_mp369_final":["mp_b3","mp_b6","mp_b9","cls_final"],
}
for cn, ks in cats.items():
    blocks[cn] = np.concatenate([blocks[k] for k in ks], axis=1)

ORDER = ["cls_b3","cls_b6","cls_b9","cls_final","mp_b3","mp_b6","mp_b9",
         "cat_cls369","cat_b6_final","cat_mp369_final"]

print("\n" + "="*112)
print("E6 DOMAIN-LOCK RATIO (G12 formula, verbatim)   s=RMS within-feature std of FF++ train")
print("="*112)
print(f"{'feature':<18}{'dim':>6}{'s':>10}{'class_gap':>11}{'ratio_all5':>12}{'ratio_v4':>10}   per-domain ratio (cd1/cd2/dfdcp/ffiw/wild)")
res = {}
for k in ORDER:
    if k not in blocks: continue
    r = blocks[k].shape[1]
    o = ratio_block(blocks[k], k)
    res[k] = o
    pd_ = "/".join(f"{o['rt'][x]:.3f}" for x in TARGETS)
    print(f"{k:<18}{r:>6}{o['s']:>10.4f}{o['cg']:>11.4f}{o['rt_all']:>12.4f}{o['rt_v4']:>10.4f}   {pd_}")
    if k == "cls_final":
        ok = (abs(o['s'] - 1.5097) < 5e-4 and abs(o['cg'] - 40.2122) < 5e-3
              and abs(o['rt_all'] - 0.1983) < 5e-4)
        print(f"{'  ^ANCHOR G12 ratio_V':<18}{'':>6}  {'PASS' if ok else 'FAIL'} vs G12 recorded "
              f"(s=1.5097 cg=40.2122 ratio=0.1983)")

np.savez(os.path.join(HERE, "ratio_layers_stats.npz"),
         **{f"{k}__{fld}": np.array([blocks[k].shape[1], res[k]['s'], res[k]['cg'],
                                    res[k]['rt_all'], res[k]['rt_v4']]
                                   + [res[k]['rt'][x] for x in TARGETS])
            for k in ORDER if k in blocks
            for fld in ["dim", "s", "class_gap", "ratio_all5", "ratio_v4"] + TARGETS})
print("\n[saved] ratio_layers_stats.npz")
