# -*- coding: utf-8 -*-
"""E4 fusion-head experiment -- data inspection (standalone, CPU-limited)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
MULTI = os.path.join(HERE, "..", "_tsne", "feats_multi.npz")

def log(*a):
    print(" ".join(str(x) for x in a), flush=True)

P = np.load(PROBE, allow_pickle=True)
M = np.load(MULTI, allow_pickle=True)

log("=== PROBE npz: %s ===" % PROBE)
for k in P.files:
    a = P[k]
    log("  key %-12s shape=%s dtype=%s" % (k, getattr(a, "shape", None), getattr(a, "dtype", None)))
tr = P["train_mask"].astype(bool)
log("  train_mask: True=%d False=%d (total %d)" % (tr.sum(), (~tr).sum(), len(tr)))
y = P["y"]
log("  y: total=%d  y==1: train=%d test=%d ; y==0: train=%d test=%d"
    % (len(y), (y[tr] == 1).sum(), (y[~tr] == 1).sum(), (y[tr] == 0).sum(), (y[~tr] == 0).sum()))
for nm, m in [("train", tr), ("test", ~tr)]:
    v = P["vids"][m]
    u, c = np.unique(v, return_counts=True)
    log("  vids[%s]: n=%d unique=%d  duplicate-video groups=%d  max samples per vid=%d  unique==n -> %s"
        % (nm, len(v), len(u), int((c > 1).sum()), c.max(), len(u) == len(v)))
# overlap check train/test videos
vtr = set(P["vids"][tr].tolist()); vte = set(P["vids"][~tr].tolist())
log("  train/test vid overlap: %d (0 => video-disjoint split)" % len(vtr & vte))

log("=== MULTI npz: %s ===" % MULTI)
for k in M.files:
    a = M[k]
    log("  key %-12s shape=%s dtype=%s" % (k, getattr(a, "shape", None), getattr(a, "dtype", None)))
doms = np.unique(M["domain"])
log("  domains: %s (n=%d)" % (list(doms), len(doms)))
for d in doms:
    sel = M["domain"] == d
    yy = M["y"][sel]
    vv = M["vid"][sel] if "vid" in M.files else None
    log("    %-6s n=%d  y==1:%d y==0:%d  unique vid=%d" % (d, sel.sum(), (yy == 1).sum(), (yy == 0).sum(),
        len(np.unique(vv)) if vv is not None else -1))
# probe V vs multi V: same pipeline sanity (dims + rough mean/std scale)
log("  V dims: probe=%s multi=%s ; C dims: probe=%s multi=%s ; F dims: multi=%s"
    % (P["V"].shape[1], M["V"].shape[1], P["C"].shape[1], M["C"].shape[1], M["F"].shape[1]))
log("[inspect] done.")
