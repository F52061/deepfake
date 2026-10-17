# -*- coding: utf-8 -*-
"""Phase A step 0: inspect the two feature npz files for schema/definition consistency."""
import os, sys
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
import numpy as np

def describe(tag, d):
    print(f"\n=== {tag} ===")
    for k in d.files:
        a = d[k]
        if a.ndim == 0:
            print(f"  {k:12s} scalar = {a.item()}")
        elif np.issubdtype(a.dtype, np.number):
            print(f"  {k:12s} shape={a.shape} dtype={a.dtype}  min={np.nanmin(a):.3f} max={np.nanmax(a):.3f} mean={np.nanmean(a):.3f}")
        else:
            print(f"  {k:12s} shape={a.shape} dtype={a.dtype}  (non-numeric; uniq0={a.flat[0] if a.size else None})")

HERE = os.path.dirname(os.path.abspath(__file__))
probe = np.load(os.path.join(HERE, "..", "_probe", "probe_feats.npz"), allow_pickle=True)
multi = np.load(os.path.join(HERE, "..", "_tsne", "feats_multi.npz"), allow_pickle=True)

describe("probe_feats.npz (FF++ video split)", probe)
describe("feats_multi.npz (cross-domain)", multi)

# domain counts & label balance in multi
if "domain" in multi.files:
    dom = multi["domain"]
    y = multi["y"]
    print("\n=== feats_multi domain composition ===")
    for dname in np.unique(dom):
        m = dom == dname
        print(f"  {str(dname):10s} n={m.sum()}  real(y=1)={int((y[m]==1).sum())} fake(y=0)={int((y[m]==0).sum())}")

# probe split sanity
if "train_mask" in probe.files and "vids" in probe.files:
    tr = probe["train_mask"].astype(bool)
    te = ~tr
    y = probe["y"]
    print("\n=== probe video-split sanity ===")
    print(f"  train n={int(tr.sum())} test n={int(te.sum())}")
    print(f"  train real/fake = {int((y[tr]==1).sum())}/{int((y[tr]==0).sum())}   test real/fake = {int((y[te]==1).sum())}/{int((y[te]==0).sum())}")
    vt = set(probe["vids"][tr]); ve = set(probe["vids"][te])
    print(f"  train/test video overlap = {len(vt & ve)}  (should be 0)")

# Feature cross-file consistency: is multi's V/C the same feature def as probe's V/C?
# Compare a small shared check via dims only here; deeper check (mean corr of matched paths) in main script.
for key in ["V","C"]:
    if key in probe.files and key in multi.files:
        print(f"\n  dims: probe[{key}]={probe[key].shape[1]}  multi[{key}]={multi[key].shape[1]}")

# is multi['F'] the concat head-input? print dims
if "F" in multi.files:
    print(f"  multi['F'] shape={multi['F'].shape}  (if 1536 -> concat(V_proj,C_proj) likely)")
print("\nprobe V_proj/C_proj dims:", probe["V_proj"].shape[1], probe["C_proj"].shape[1])
