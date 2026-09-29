# -*- coding: utf-8 -*-
"""
Phase F - selective prediction & channel routing experiment (CPU-only, frozen features).
  F1: oracle-upper-bound failure prediction (Q1) + selective abstention (Q2)
  F2: channel routing (Q3) + abstention baselines (MSP / energy)

Protocol (task sheet verbatim):
  * base heads fit ONCE on the FULL FF++-train subset (probe npz train_mask, n=2200):
      V head: StandardScaler(fit train V) + LogisticRegression(C=1e-3, lbfgs, max_iter=3000)
      C head: same on train C.
      score = decision_function; proba pV=sigma(sV), pC=sigma(sC).
  * 800 test and the 5 target domains (300 each) are used ONLY for final evaluation;
    never in fit / scaling / hyperparam choice.
  * AUROC & accuracy all computed on those evaluation sets.
  * E2 source-distance re-implemented verbatim from vit_module/_fusion/fusion_e4.py
    (which is itself the verified replica of Phase A E2, k=10): Mahalanobis^2 in the
    top-k(=10) PCA subspace of the FULL FF++-train V manifold.
  * CPU-limited: thread env vars set before numpy import; torch threads=1; no GPU.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
torch.set_num_threads(1)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE   = os.path.dirname(os.path.abspath(__file__))
PROBE  = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
MULTI  = os.path.join(HERE, "..", "_tsne", "feats_multi.npz")
REPORT = os.path.join(HERE, "selective_report.txt")
DOMS   = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
COVS   = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
FS     = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
T0 = time.time()
_rep = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _rep.append(s)

def f64(a): return np.ascontiguousarray(a, dtype=np.float64)

def sig(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def safe_auc(ytrue, score):
    yb = np.asarray(ytrue).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    return float(roc_auc_score(yb, score))

# =====================================================================
# data
# =====================================================================
P = np.load(PROBE, allow_pickle=True)
M = np.load(MULTI, allow_pickle=True)
y = P["y"]; tr = P["train_mask"].astype(bool); te = ~tr
g = {d: np.nonzero(M["domain"] == d)[0] for d in DOMS}
all_rows = np.concatenate([g[d] for d in DOMS])   # pooled 1500, DOMS order
assert tr.sum() == 2200 and te.sum() == 800
log("=" * 100)
log("PHASE F  selective / channel-routing  (started %s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 100)
log("[0] data: probe n=%d (train=%d/test=%d), y1/y0 train=%d/%d test=%d/%d"
    % (len(y), tr.sum(), te.sum(), (y[tr] == 1).sum(), (y[tr] == 0).sum(),
       (y[te] == 1).sum(), (y[te] == 0).sum()))
for d in DOMS:
    log("    multi %-5s n=%d y1=%d y0=%d" % (d, len(g[d]), (M["y"][g[d]] == 1).sum(), (M["y"][g[d]] == 0).sum()))
log("    feats_multi ffpp block excluded from all scoring; only the 5 target domains used.")

# =====================================================================
# module 0: base heads (fit once on FULL FF++ train) + anchor consistency
# =====================================================================
scV = StandardScaler().fit(f64(P["V"][tr]))
clfV = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000).fit(scV.transform(f64(P["V"][tr])), y[tr])
scC = StandardScaler().fit(f64(P["C"][tr]))
clfC = LogisticRegression(C=1e-3, solver="lbfgs", max_iter=3000).fit(scC.transform(f64(P["C"][tr])), y[tr])
sV_tr_te = clfV.decision_function(scV.transform(f64(P["V"][te])))
sC_tr_te = clfC.decision_function(scC.transform(f64(P["C"][te])))
auc_V_te = safe_auc(y[te], sV_tr_te)
auc_C_te = safe_auc(y[te], sC_tr_te)
log("")
log("[sel-base] V head (C=1e-3, full-2200 fit): in-domain 800-test AUC = %.4f (expect ~0.9852)" % auc_V_te)
log("[sel-base] C head (C=1e-3, full-2200 fit): in-domain 800-test AUC = %.4f" % auc_C_te)

V_anchor_quoted = {"cd1": 0.831, "cd2": 0.864, "dfdcp": 0.827, "ffiw": 0.826, "wild": 0.808}
C_anchor_quoted = {"cd1": 0.656, "cd2": 0.723, "dfdcp": 0.707, "ffiw": 0.834, "wild": 0.717}
V_auc_d = {}; C_auc_d = {}
for d in DOMS:
    V_auc_d[d] = safe_auc(M["y"][g[d]], clfV.decision_function(scV.transform(f64(M["V"][g[d]]))))
    C_auc_d[d] = safe_auc(M["y"][g[d]], clfC.decision_function(scC.transform(f64(M["C"][g[d]]))))
log("[sel-base] V five-domain AUC (computed):  " + " ".join("%s=%.4f" % (d, V_auc_d[d]) for d in DOMS))
log("[sel-base] V five-domain AUC (quoted ~):   " + " ".join("%s=%.3f" % (d, V_anchor_quoted[d]) for d in DOMS))
log("[sel-base] C five-domain AUC (computed):  " + " ".join("%s=%.4f" % (d, C_auc_d[d]) for d in DOMS))
log("[sel-base] C five-domain AUC (quoted ~):   " + " ".join("%s=%.3f" % (d, C_anchor_quoted[d]) for d in DOMS))

# anchor consistency check (stop if >0.005 away from quoted anchors)
max_diff = 0.0
for d in DOMS:
    max_diff = max(max_diff, abs(V_auc_d[d] - V_anchor_quoted[d]), abs(C_auc_d[d] - C_anchor_quoted[d]))
max_diff = max(max_diff, abs(auc_V_te - 0.9852))
log("[sel-base] max |computed - quoted-anchor| = %.4f  (threshold 0.005)" % max_diff)
if max_diff > 0.005:
    log("[sel-base] ANCHOR MISMATCH > 0.005 -> stopping for protocol check (difference reported, no fix applied).")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_rep) + "\n")
    sys.exit(1)
log("[sel-base] anchor consistency OK (diff <= 0.005).")
log("[sel-base] done.")

# =====================================================================
# E2 source-distance manifold (fit on FULL FF++-train V, k=10) -- verbatim from fusion_e4.py
# =====================================================================
Vfull = f64(P["V"][tr])
n_src = Vfull.shape[0]
muV = Vfull.mean(0)
_, S, Vt = np.linalg.svd(Vfull - muV, full_matrices=False)
K = 10
Uk = Vt.T[:, :K]
lamk = (S[:K] ** 2) / (n_src - 1.0); lamk[lamk < 1e-12] = 1e-12
def src_dist(X):
    z = (X - muV) @ Uk
    return np.sum(z * z / lamk, axis=1)
e0 = Vt[0]  # first right singular vector of raw-cov PCA of train V (used by tau_spec)
log("[sel-dist] E2 source-distance manifold fit on FF++-train V n=%d k=%d (muV/e0/Uk/lamk frozen)." % (n_src, K))

# =====================================================================
# evaluation-set builder
# =====================================================================
E0_idx = np.nonzero(te)[0]
def build_eval(name, kind, idx):
    if kind == "probe":
        Vr, Cr, yr = f64(P["V"][idx]), f64(P["C"][idx]), y[idx]
    else:
        Vr, Cr, yr = f64(M["V"][idx]), f64(M["C"][idx]), M["y"][idx]
    sV = clfV.decision_function(scV.transform(Vr))
    sC = clfC.decision_function(scC.transform(Cr))
    pV, pC = sig(sV), sig(sC)
    errV = (pV >= 0.5) != (yr == 1)
    errC = (pC >= 0.5) != (yr == 1)
    ev = dict(name=name, n=len(yr), V=Vr, C=Cr, y=yr, sV=sV, sC=sC, pV=pV, pC=pC, errV=errV, errC=errC)
    # trust signals (single-signal, no training)
    ev["dist"] = src_dist(Vr)                       # tau_dist
    ev["spec"] = np.abs((Vr - muV) @ e0)            # tau_spec (axis support)
    ev["disc"] = np.abs(pV - pC)                    # tau_disc
    ev["ent"]  = pV * (1.0 - pV)                    # tau_ent
    ev["msp"]  = 1.0 - np.maximum(pV, 1.0 - pV)     # tau_msp (MSP complement)
    return ev

evals = [build_eval("E0_800", "probe", E0_idx)]
for d in DOMS:
    evals.append(build_eval(d, "multi", g[d]))
evals.append(build_eval("all5_1500", "multi", all_rows))

log("[sel-evals] built %d evaluation sets:" % len(evals))
for e in evals:
    log("    %-9s n=%d y1=%d y0=%d  errV=%.4f errC=%.4f"
        % (e["name"], e["n"], (e["y"] == 1).sum(), (e["y"] == 0).sum(), e["errV"].mean(), e["errC"].mean()))
log("[sel-evals] done.")

SIGNALS = ["dist", "disc", "spec", "ent", "msp"]
SIG_LABEL = {"dist": "tau_dist(srcdist)", "disc": "tau_disc(|pV-pC|)", "spec": "tau_spec(axis-support)",
             "ent": "tau_ent(pV(1-pV))", "msp": "tau_msp(1-max(pV,1-pV))"}
SIG_DIR_EXPECT = {"dist": "+", "disc": "+", "spec": "-", "ent": "+", "msp": "+"}

def table_lines(title, header, rows):
    out = [""]
    colw = [max(len(str(c)) for c in col) + 2 for col in zip(*([header] + rows))]
    out.append(title)
    out.append("  " + "-" * (sum(colw) + 4))
    out.append("  " + "".join(("%-" + str(w) + "s") % str(c) for c, w in zip(header, colw)))
    for r in rows:
        out.append("  " + "".join(("%-" + str(w) + "s") % str(c) for c, w in zip(r, colw)))
    return out

# =====================================================================
# F1 - Q1  failure prediction AUROC
# =====================================================================
def fmt_auc_cell(raw):
    if raw != raw:
        return "  n/a  "
    val = max(raw, 1.0 - raw)
    sgn = "+" if raw >= 0.5 else "-"
    return "%s%.4f" % (sgn, val)

log("")
log("=" * 100)
log("F1 - Q1  failure prediction: does a trust signal predict errV=1 (V wrong at pV>=0.5 rule)?")
log("=" * 100)
log("  errV per sample = (pV>=0.5) != (y==1).  AUROC(errV, tau): raw>0.5 => high tau => V likely wrong.")
log("  Reported cell = max(raw,1-raw) with sign = direction (+ high->error | - low->error).")

# raw table (for reference / appendix)
hdr = ["signal"] + [e["name"] for e in evals]
raw_rows = []
raw_auc = {s: {e["name"]: float("nan") for e in evals} for s in SIGNALS}
for s in SIGNALS:
    row = [SIG_LABEL[s]]
    for e in evals:
        raw_auc[s][e["name"]] = safe_auc(e["errV"], e[s])
        row.append("%.4f" % raw_auc[s][e["name"]] if raw_auc[s][e["name"]] == raw_auc[s][e["name"]] else "  n/a ")
    raw_rows.append(row)
for ln in table_lines("TABLE Q1-A  raw-direction AUROC  AUROC(errV, tau)  (raw values; sign below)",
                      hdr, raw_rows):
    log(ln)

# aligned table (main result)
main_rows = []
dir_rows = []
for s in SIGNALS:
    mrow = [SIG_LABEL[s]]
    drow = ["dir " + s]
    for e in evals:
        mrow.append(fmt_auc_cell(raw_auc[s][e["name"]]))
        raw = raw_auc[s][e["name"]]
        drow.append("+" if (raw == raw and raw >= 0.5) else "-")
    main_rows.append(mrow)
    dir_rows.append(drow)
for ln in table_lines("TABLE Q1-B  failure-prediction AUROC (aligned; value=max(raw,1-raw), cell sign = direction)",
                      hdr, main_rows):
    log(ln)
log("  direction key: + = high signal value -> V error-prone ; - = low signal value -> V error-prone")
log("  (tau_spec expected '-':  failure prediction negatively correlated with axis support.)")

# training-free ensemble (3 direction-aligned signals, rank-equal weight), 800 test ONLY
e0ev = evals[0]
ens = np.zeros(e0ev["n"])
for s in ["dist", "ent", "spec"]:
    aligned = e0ev[s].copy()
    raw = raw_auc[s]["E0_800"]
    if raw == raw and raw < 0.5:
        aligned = -aligned
    ranks = np.empty(e0ev["n"], dtype=np.int64)
    ranks[np.argsort(aligned, kind="mergesort")] = np.arange(e0ev["n"])
    ens += ranks
ens_auc = safe_auc(e0ev["errV"], ens)
log("")
log("[sel-Q1-ensemble] training-free ensemble = rank(tau_dist)+rank(tau_ent)+rank(aligned tau_spec), equal weight")
log("  E0_800 AUROC = %.4f   (rank alignment computed on the 800-test distribution -> reference magnitude only," % ens_auc)
log("  NOT used as the cross-domain conclusion; cross-domain main table = single-signal AUROCs above.)")
log("[sel-Q1] done.")

# =====================================================================
# F1 - Q2  selective abstention risk@coverage (drop largest-tau first)
# =====================================================================
def risk_curve(ev, tau, covs):
    n = ev["n"]
    out = {}
    order = np.argsort(tau, kind="mergesort")          # ascending tau
    for c in covs:
        nk = int(round(c * n))
        keep = order[:nk] if nk < n else np.arange(n)
        out[c] = float(ev["errV"][keep].mean())
    return out

log("")
log("=" * 100)
log("F1 - Q2  selective abstention risk@coverage  (V error rate among RETAINED samples; drop highest-tau first)")
log("=" * 100)
log("  coverage = kept fraction;  risk = mean(errV) over kept samples;  coverage 1.0 = no-abstention baseline.")
log("  retained sample count is IDENTICAL across abstention signals at the same coverage (same n, same rule);")
retained_note = {}
for c in COVS:
    retained_note[c] = {e["name"]: int(round(c * e["n"])) for e in evals}
log("  retained counts per set:  " + " ".join("%s@%g=%d" % (e["name"], c, retained_note[c][e["name"]])
    for c in COVS for e in evals if e["name"] == "E0_800"))

def q2_table(sigkey, tag):
    hdr2 = ["coverage"] + [e["name"] for e in evals]
    rows = []
    for c in COVS:
        row = ["%.2f" % c]
        for e in evals:
            rc = risk_curve(e, e[sigkey], COVS)
            row.append("%.4f" % rc[c])
        rows.append(row)
    lines = table_lines("TABLE Q2-%s  risk@coverage, abstention signal = %s" % (tag, SIG_LABEL[sigkey]), hdr2, rows)
    for ln in lines:
        log(ln)

q2_table("dist", "a")
q2_table("msp",  "b")
log("[sel-Q2] done.")

# =====================================================================
# F2 - Q3  channel routing (V unreliable -> C)
# =====================================================================
def route_acc(ev, tau, f):
    n = ev["n"]
    nroute = int(round(f * n))
    if nroute <= 0:
        dec = (ev["pV"] >= 0.5)
    elif nroute >= n:
        dec = (ev["pC"] >= 0.5)
    else:
        order = np.argsort(tau, kind="mergesort")       # ascending
        route = order[-nroute:]                          # top f (highest tau) -> C
        dec = (ev["pV"] >= 0.5).copy()
        dec[route] = (ev["pC"] >= 0.5)[route]
    return float((dec == (ev["y"] == 1)).mean())

log("")
log("=" * 100)
log("F2 - Q3  channel routing  V-unreliable -> C  (score=pC, decision pC>=0.5)")
log("=" * 100)

for e in evals:
    if e["name"] in ("E0_800", "all5_1500"):
        log("")
        log("--- routing summary on %s (n=%d) ---" % (e["name"], e["n"]))
        errV = float(e["errV"].mean()); errC = float(e["errC"].mean())
        save = (e["errV"] & ~e["errC"])
        save_cnt = int(save.sum()); save_frac = float(save_cnt) / e["n"]
        log("  errV (V rule pV>=0.5)          = %.4f  -> accuracy(V)          = %.4f" % (errV, 1.0 - errV))
        log("  errC (C rule pC>=0.5)          = %.4f  -> accuracy(C)          = %.4f" % (errC, 1.0 - errC))
        log("  saveable  = (errV & ~errC) count= %d  frac= %.4f" % (save_cnt, save_frac))
        oracle_acc = 1.0 - (errV - save_frac)
        log("  oracle-switch ceiling accuracy  = %.4f   (switch only V-wrong&C-right to C;  = accuracy(V)+save_frac)"
            % oracle_acc)
        # distribution compare: saveable vs all
        for sigkey in ["dist", "disc"]:
            mu_all = float(e[sigkey].mean())
            mu_sav = float(e[sigkey][save].mean()) if save_cnt else float("nan")
            log("  saveable  %-6s mean=%.4f  | all mean=%.4f" % (sigkey, mu_sav, mu_all))
        # full route curve for both families printed below

hdr3 = ["f"] + ["E0_800", "all5_1500", "ffiw", "cd1", "wild"]
route_family = {}
for fam, sigkey in [("tau_dist", "dist"), ("tau_msp", "msp")]:
    rows = []
    for f in FS:
        row = ["%.2f" % f]
        for ename, dname in [("E0_800", "E0_800"), ("all5_1500", "all5_1500"),
                             ("ffiw", "ffiw"), ("cd1", "cd1"), ("wild", "wild")]:
            ev = next(x for x in evals if x["name"] == ename)
            row.append("%.4f" % route_acc(ev, ev[sigkey], f))
        rows.append(row)
    route_family[fam] = rows
    for ln in table_lines("TABLE Q3  accuracy(f) routing top-f by %s to C  (V decision replaced by C for routed)" % fam,
                          hdr3, rows):
        log(ln)

log("[sel-Q3] done.")

# =====================================================================
# appendix
# =====================================================================
log("")
log("=" * 100)
log("APPENDIX")
log("=" * 100)
log("  total wall time = %.1f s" % (time.time() - T0))
log("  CPU-only run; env OMP/MKL/OPENBLAS/NUMEXPR/VECLIB threads=1; torch threads=1; no GPU/no .cuda().")
log("  Data: probe_feats.npz (V 768, C 1024, y, vids, train_mask; train=2200/test=800, video-disjoint);")
log("        feats_multi.npz (V/C/F, y, domain; 5 target domains x300 balanced 150/150;")
log("        the ffpp 800-row block in feats_multi.npz excluded from all scoring).")
log("  Base heads fit once on the FULL 2200 FF++ train only; scalers & LR (C=1e-3, lbfgs, max_iter=3000);")
log("  score=decision_function, p=sigma(score).  All AUROC/accuracy on 800-test / target domains only.")
log("  E2 source-distance re-implemented verbatim from vit_module/_fusion/fusion_e4.py (verified Phase-A E2 replica,")
log("  k=10; manifold = FULL FF++-train V n=2200; Mahalanobis^2 in top-10 PCA subspace).")
log("  Q2 'drop largest tau first'; Q3 'route largest tau first'; retained/routed counts = round(fraction*n).")
log("[sel_done].")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(_rep) + "\n")
print("saved report -> %s" % REPORT, flush=True)
