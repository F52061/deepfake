# -*- coding: utf-8 -*-
"""
E4 fusion-head comparison experiment (CPU-limited; no GPU, no forward of anything
beyond the small heads defined here).  Fixed protocol (task sheet, verbatim):
  * setA = concat[V raw 768, C raw 1024] (1792-d);  setB = concat[V_proj 768, C_proj 768] (1536-d)
  * H0 = LogisticRegression(C=1e-3, lbfgs, max_iter=3000) on standardized concat
  * H1 = MLP 256 / H2 = 2-token 1-layer Transformer d=128 / H3 = 2-token 2-layer Transformer d=256
  * scalers fit on Xtr (FF++ train subset) only; inner-val = ~10% videos (seed0-fixed, video-disjoint)
  * 3 seeds {0,1,2} vary init + shuffle only; early stop on val AUC (patience 8, max 40 ep), restore best
  * AUC (y=1 real). 800-test / 5x300 target / ffiw halves / pooled terciles scored once per model.
Distance split = faithful re-implementation of Phase A E2 (vit_module/_phaseA/run_phaseA.py, E2 block):
  Mahalanobis^2 of each target row in the top-k PCA subspace of the FULL FF++-train V manifold
  (rows probe V[train_mask], n=2200; mu = mean; U = right singular vectors; lam = S^2/(n-1)).
  k=10 used for BOTH sub-tables (E2's global-tercile block used k=10; per-domain halves in E2 were
  printed for k in {5,10} -- choice noted in report).
"""
import os, sys, time, copy
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE   = os.path.dirname(os.path.abspath(__file__))
PROBE  = os.path.join(HERE, "..", "_probe", "probe_feats.npz")
MULTI  = os.path.join(HERE, "..", "_tsne", "feats_multi.npz")
REPORT = os.path.join(HERE, "fusion_report.txt")

DOMS  = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
SEEDS = [0, 1, 2]
T0 = time.time()
_rep = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _rep.append(s)

def f32(a): return np.ascontiguousarray(a, dtype=np.float32)
def f64(a): return np.ascontiguousarray(a, dtype=np.float64)

log("=" * 90)
log("E4 FUSION-HEAD COMPARISON  (started %s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 90)

# =====================================================================
# 0. data check (mirrors inspect_data.py)
# =====================================================================
P = np.load(PROBE, allow_pickle=True)
M = np.load(MULTI, allow_pickle=True)
y = P["y"]; tr = P["train_mask"].astype(bool); te = ~tr

log("")
log("[0] DATA CHECK")
log("  probe_feats.npz keys: " + ", ".join(P.files))
for k in ["V", "C", "V_proj", "C_proj"]:
    log("    %-7s shape=%s dtype=%s" % (k, P[k].shape, P[k].dtype))
log("  train_mask True=%d False=%d ; y==1 train/test=%d/%d ; y==0 train/test=%d/%d (balanced)"
    % (tr.sum(), te.sum(), (y[tr] == 1).sum(), (y[te] == 1).sum(), (y[tr] == 0).sum(), (y[te] == 0).sum()))
for nm, m in [("train", tr), ("test", te)]:
    v = P["vids"][m]; u, c = np.unique(v, return_counts=True)
    log("  probe vids[%s]: n=%d unique_videos=%d -> multi-frame per video (max %d samples/group)"
        % (nm, len(v), len(u), c.max()))
log("  train/test video overlap = %d (0 => video-disjoint)" % len(set(P["vids"][tr].tolist()) & set(P["vids"][te].tolist())))
log("  feats_multi.npz keys: " + ", ".join(M.files))
log("  multi domains found in file: %s" % list(np.unique(M["domain"])))
for d in np.unique(M["domain"]):
    sel = M["domain"] == d
    log("    %-6s n=%d y1=%d y0=%d" % (d, sel.sum(), (M["y"][sel] == 1).sum(), (M["y"][sel] == 0).sum()))
log("  NOTE: file additionally holds an 'ffpp' 800-row block; per Phase-A/E2 protocol only the 5")
log("        declared target domains (300 each) are used; ffpp rows excluded from all scoring/pooling.")

# =====================================================================
# 1. fixed video-grouped inner-val split (seed 0) inside the 2200 train samples
# =====================================================================
def video_val_split(train_idx, vids, seed=0, target=190):
    groups = {}
    for i in train_idx:
        groups.setdefault(str(vids[i]), []).append(int(i))
    keys = sorted(groups.keys())
    perm = np.random.default_rng(seed).permutation(len(keys))
    val = []
    for p in perm:
        if len(val) >= target:
            break
        val += groups[keys[p]]
    m = np.zeros(len(vids), dtype=bool)
    m[np.sort(np.array(val))] = True
    return m

valm = video_val_split(np.nonzero(tr)[0], P["vids"], seed=0, target=190)
trm  = tr & ~valm
Xtr0 = np.nonzero(trm)[0]; Xval0 = np.nonzero(valm)[0]; Xte0 = np.nonzero(te)[0]
n_vtr  = len(set(P["vids"][Xtr0].tolist()))
n_vval = len(set(P["vids"][Xval0].tolist()))
n_vte  = len(set(P["vids"][Xte0].tolist()))
log("")
log("  fixed split (rng(0), video groups): Xtr n=%d (%d videos) | Xval n=%d (%d videos) | Xte n=%d (%d videos)"
    % (len(Xtr0), n_vtr, len(Xval0), n_vval, len(Xte0), n_vte))
log("  Xval y1/y0 = %d/%d ; Xtr y1/y0 = %d/%d" % ((y[Xval0] == 1).sum(), (y[Xval0] == 0).sum(),
                                                  (y[Xtr0] == 1).sum(), (y[Xtr0] == 0).sum()))

# =====================================================================
# 2. feature builders
# =====================================================================
def feat_probe(name):
    if name == "setA":   return f64(np.hstack([P["V"], P["C"]]))
    if name == "setB":   return f64(np.hstack([P["V_proj"], P["C_proj"]]))
    if name == "V":      return f64(P["V"])
    if name == "C":      return f64(P["C"])
    if name == "V_proj": return f64(P["V_proj"])
    if name == "C_proj": return f64(P["C_proj"])
    raise ValueError(name)

def feat_multi(name):
    if name == "setA": return f64(np.hstack([M["V"], M["C"]]))
    if name == "V":    return f64(M["V"])
    if name == "C":    return f64(M["C"])
    raise ValueError(name)

# =====================================================================
# 3. head models
# =====================================================================
class TwoTok(nn.Module):
    """per-channel projection -> 2 tokens -> N x TransformerEncoderLayer -> mean -> LN -> Linear(->1)"""
    def __init__(self, dV, dC, d=128, dff=256, nlayers=1):
        super().__init__()
        self.dV = dV
        self.linV = nn.Linear(dV, d)
        self.linC = nn.Linear(dC, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=dff,
                                       dropout=0.1, batch_first=True, activation="relu")
            for _ in range(nlayers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
    def forward(self, x):
        tv = self.linV(x[:, :self.dV])
        tc = self.linC(x[:, self.dV:])
        toks = torch.stack([tv, tc], dim=1)          # [B, 2, d]
        for L in self.layers:
            toks = L(toks)
        m = toks.mean(dim=1)
        m = self.ln(m)
        return self.out(m)

def make_head(kind, d_in, dV, seed):
    torch.manual_seed(seed)
    if kind == "H1":
        return nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    if kind == "H2":
        return TwoTok(dV, d_in - dV, d=128, dff=256, nlayers=1)
    if kind == "H3":
        return TwoTok(dV, d_in - dV, d=256, dff=512, nlayers=2)
    raise ValueError(kind)

def train_nn(model, Xs, ys, Xvs, yvs, seed, batch=128, lr=1e-3, wd=1e-4,
             max_epoch=40, patience=8):
    """Xs/Xvs = standardized float64 arrays. Returns (model_with_best_weights, best_epoch, best_val_auc)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(f32(Xs)); yt = torch.from_numpy(f32(ys.astype(np.float64)))
    Xv = torch.from_numpy(f32(Xvs))
    n = Xt.shape[0]
    best_auc = -1.0; best_sd = None; best_ep = 0; bad = 0
    for ep in range(1, max_epoch + 1):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, batch):
            b = order[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xt[b]).squeeze(1), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xv).squeeze(1)).numpy()
        a = float(roc_auc_score(yvs, p))
        if a > best_auc + 1e-9:
            best_auc, best_ep, bad = a, ep, 0
            best_sd = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, best_ep, best_auc

def nn_scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(f32(X))).squeeze(1)).numpy()

def lr_scores(clf, X):
    return clf.predict_proba(X)[:, 1]

# =====================================================================
# 4. E2 source-distance (faithful re-implementation, k=10)  -> globals for scoring
# =====================================================================
log("")
log("[distance] Phase-A E2 source distance, re-implemented verbatim (k=10; manifold = FULL FF++-train V n=%d)"
    % tr.sum())
Vfull = f64(P["V"][tr])
n_src = Vfull.shape[0]
muV = Vfull.mean(0)
_, S, Vt = np.linalg.svd(Vfull - muV, full_matrices=False)
U = Vt.T
K = 10
Uk = U[:, :K]
lamk = (S[:K] ** 2) / (n_src - 1.0); lamk[lamk < 1e-12] = 1e-12
def src_dist(X):
    z = (X - muV) @ Uk
    return np.sum(z * z / lamk, axis=1)

g = {d: np.nonzero(M["domain"] == d)[0] for d in DOMS}
allrows = np.concatenate([g[d] for d in DOMS])          # pooled 1500 rows, DOMS order
dists = {d: src_dist(f64(M["V"][g[d]])) for d in DOMS}
gdd = np.concatenate([dists[d] for d in DOMS])
t1, t2 = np.quantile(gdd, [1.0 / 3, 2.0 / 3])
terc_p = {"near": np.nonzero(gdd <= t1)[0],
          "mid":  np.nonzero((gdd > t1) & (gdd <= t2))[0],
          "far":  np.nonzero(gdd > t2)[0]}
terc_m = {q: allrows[idx] for q, idx in terc_p.items()}  # multi-array row indices per tercile
log("  tercile thresholds t1=%.4e t2=%.4e ; bin sizes near/mid/far = %d/%d/%d"
    % (t1, t2, len(terc_p["near"]), len(terc_p["mid"]), len(terc_p["far"])))
medf = np.median(dists["ffiw"])
ffiw_near = dists["ffiw"] <= medf
ffiw_far  = dists["ffiw"] > medf
log("  ffiw dist median=%.4e ; near n=%d far n=%d" % (medf, ffiw_near.sum(), ffiw_far.sum()))

# =====================================================================
# 5. per-(head,set) run + evaluation
# =====================================================================
def run_head(kind, fname, dV_side):
    t0 = time.time()
    Xp = feat_probe(fname)                       # 3000 x d
    Xtr_f, Xval_f, Xte_f = Xp[Xtr0], Xp[Xval0], Xp[Xte0]
    ytr_f, yval_f, yte = y[Xtr0], y[Xval0], y[Xte0]
    sc = StandardScaler().fit(Xtr_f)
    Xtr_s = sc.transform(Xtr_f)
    Xval_s = sc.transform(Xval_f)
    Xte_s = sc.transform(Xte_f)

    if kind == "H0":
        clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs")
        clf.fit(Xtr_s, ytr_f)
        mods = [clf]
        scoref = lr_scores
        single = True
    else:
        mods, scoref, single = [], nn_scores, False
        for seed in SEEDS:
            model = make_head(kind, Xtr_s.shape[1], dV_side, seed)
            model, _, _ = train_nn(model, Xtr_s, ytr_f, Xval_s, yval_f, seed)
            mods.append(model)

    vals = {"te800": []}
    for m in mods:
        vals["te800"].append(float(roc_auc_score(yte, scoref(m, Xte_s))))

    # cross-domain / distance-split scoring exists only for setA
    # (feats_multi has no V_proj/C_proj channels -> setB not scored there)
    if fname == "setA":
        Xm = feat_multi("setA")                  # 2300 x 1792 (raw rows)
        Xm_sc = {d: sc.transform(Xm[g[d]]) for d in DOMS}
        Xm_all = sc.transform(Xm[allrows])       # pooled 1500 rows in DOMS order
        ffiw_sc = Xm_sc["ffiw"]
        yffiw = M["y"][g["ffiw"]]
        for d in DOMS:
            vals[d] = []
        vals["ffiw_near"] = []; vals["ffiw_far"] = []
        vals["t_near"] = []; vals["t_mid"] = []; vals["t_far"] = []
        for m in mods:
            for d in DOMS:
                vals[d].append(float(roc_auc_score(M["y"][g[d]], scoref(m, Xm_sc[d]))))
            vals["ffiw_near"].append(float(roc_auc_score(yffiw[ffiw_near], scoref(m, ffiw_sc[ffiw_near]))))
            vals["ffiw_far"].append(float(roc_auc_score(yffiw[ffiw_far], scoref(m, ffiw_sc[ffiw_far]))))
            for q in ["near", "mid", "far"]:
                vals["t_" + q].append(float(roc_auc_score(M["y"][terc_m[q]], scoref(m, Xm_all[terc_p[q]]))))
    res = {"vals": vals, "single": single, "wall": time.time() - t0}
    return res

def meanstd(vlist):
    v = np.array(vlist, dtype=np.float64)
    return (float(v.mean()), float(v.std())) if len(v) > 1 else (float(v[0]), None)

def fmt_ms(m, s):
    return "%.4f" % m if s is None else "%.4f+-%.4f" % (m, s)

def fmt_cell(name, vlist, notes):
    m, s = meanstd(vlist)
    if s is not None and s > 0.01:
        notes.append("  std>0.01 %-24s raw=[%s]" % (name, ", ".join("%.4f" % x for x in vlist)))
    return fmt_ms(m, s)

def table(title, rows):
    """rows[0] = header; column widths auto; left-aligned."""
    colw = [max(len(str(c)) for c in col) + 2 for col in zip(*rows)]
    log("")
    log(title)
    log("  " + "-" * (sum(colw) + 4))
    for r in rows:
        log("  " + "".join(("%-" + str(w) + "s") % str(c) for c, w in zip(r, colw)))

def fit_ref_on_Xtr(ch):
    """single-channel LR probe fit on Xtr (same protocol as heads); returns per-eval-point AUCs.
    Multi (cross-domain/distance) scoring only exists for channels present in feats_multi (V, C)."""
    Xp = feat_probe(ch)
    sc = StandardScaler().fit(Xp[Xtr0])
    clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs").fit(sc.transform(Xp[Xtr0]), y[Xtr0])
    out = {"te800": float(roc_auc_score(y[Xte0], lr_scores(clf, sc.transform(Xp[Xte0]))))}
    if ch in ("V", "C"):
        Xm = feat_multi(ch)
        for d in DOMS:
            out[d] = float(roc_auc_score(M["y"][g[d]], lr_scores(clf, sc.transform(Xm[g[d]]))))
        out["ffiw_near"] = float(roc_auc_score(M["y"][g["ffiw"][ffiw_near]], lr_scores(clf, sc.transform(Xm[g["ffiw"][ffiw_near]]))))
        out["ffiw_far"]  = float(roc_auc_score(M["y"][g["ffiw"][ffiw_far]],  lr_scores(clf, sc.transform(Xm[g["ffiw"][ffiw_far]]))))
        for q in ["near", "mid", "far"]:
            rows = terc_m[q]
            out["t_" + q] = float(roc_auc_score(M["y"][rows], lr_scores(clf, sc.transform(Xm[rows]))))
    return out

def fit_anchor_fulltrain(ch):
    """LR probe fit on FULL 2200 FF++ train (E0/E1/E3d replication; consistency check only)."""
    Xp = feat_probe(ch)
    sc = StandardScaler().fit(Xp[tr])
    clf = LogisticRegression(C=1e-3, max_iter=3000, solver="lbfgs").fit(sc.transform(Xp[tr]), y[tr])
    Xm = feat_multi(ch)
    out = {"te800": float(roc_auc_score(y[te], lr_scores(clf, sc.transform(Xp[te]))))}
    for d in DOMS:
        out[d] = float(roc_auc_score(M["y"][g[d]], lr_scores(clf, sc.transform(Xm[g[d]]))))
    out["ffiw_near"] = float(roc_auc_score(M["y"][g["ffiw"][ffiw_near]], lr_scores(clf, sc.transform(Xm[g["ffiw"][ffiw_near]]))))
    out["ffiw_far"]  = float(roc_auc_score(M["y"][g["ffiw"][ffiw_far]],  lr_scores(clf, sc.transform(Xm[g["ffiw"][ffiw_far]]))))
    for q in ["near", "mid", "far"]:
        rows = terc_m[q]
        out["t_" + q] = float(roc_auc_score(M["y"][rows], lr_scores(clf, sc.transform(Xm[rows]))))
    return out

# =====================================================================
# 6. run phases
# =====================================================================
notes = []
log("")
log("RUN 1/3  setA (concat V raw + C raw, 1792-d): H0 H1 H2 H3 x seeds ...")
tA = time.time()
A = {}
for kind in ["H0", "H1", "H2", "H3"]:
    A[kind] = run_head(kind, "setA", 768)
    log("  setA %s done in %.1fs (t=%.0fs)" % (kind, A[kind]["wall"], time.time() - T0))
log("  setA total %.1fs" % (time.time() - tA))

log("RUN 2/3  setB (concat V_proj + C_proj, 1536-d): H0 H1 H2 H3 x seeds ...")
tB = time.time()
B = {}
for kind in ["H0", "H1", "H2", "H3"]:
    B[kind] = run_head(kind, "setB", 768)
    log("  setB %s done in %.1fs (t=%.0fs)" % (kind, B[kind]["wall"], time.time() - T0))
log("  setB total %.1fs" % (time.time() - tB))

log("RUN 3/3  reference rows (Xtr-fit single channels) + full-train anchor replication ...")
refA = {"V": fit_ref_on_Xtr("V"), "C": fit_ref_on_Xtr("C")}
refB = {"V_proj": fit_ref_on_Xtr("V_proj"), "C_proj": fit_ref_on_Xtr("C_proj")}
anch = {"V": fit_anchor_fulltrain("V"), "C": fit_anchor_fulltrain("C")}
log("  done (t=%.0fs)" % (time.time() - T0))

# =====================================================================
# TABLE 1 : setA in-domain
# =====================================================================
rows = [["head", "mean+-std AUC (3 seeds)"]]
for kind in ["H0", "H1", "H2", "H3"]:
    rows.append([kind, fmt_cell("setA te800 " + kind, A[kind]["vals"]["te800"], notes)])
rows.append(["V-only-LR (Xtr fit)", fmt_ms(*meanstd([refA["V"]["te800"]]))])
rows.append(["C-only-LR (Xtr fit)", fmt_ms(*meanstd([refA["C"]["te800"]]))])
table("TABLE 1  setA in-domain AUC on FF++ 800-test (models fit on Xtr n=%d)" % len(Xtr0), rows)
log("  ref rows = Xtr-fit recomputes (same protocol as heads). Full-train anchors (E3d/E0 replication):")
log("  V-only = %.4f (published 0.9852) | C-only = %.4f (published 0.9108)" % (anch["V"]["te800"], anch["C"]["te800"]))

# =====================================================================
# TABLE 2 : setA cross-domain
# =====================================================================
vonly = np.array([refA["V"][d] for d in DOMS])
rows = [["head"] + DOMS + ["mean", "Delta_vs_V"]]
for kind in ["H0", "H1", "H2", "H3"]:
    arr = np.array([meanstd(A[kind]["vals"][d])[0] for d in DOMS])
    rows.append([kind] + ["%.4f" % v for v in arr] + ["%.4f" % arr.mean(), "%+.4f" % (arr.mean() - vonly.mean())])
rows.append(["V-only-LR"] + ["%.4f" % refA["V"][d] for d in DOMS] + ["%.4f" % vonly.mean(), "+0.0000"])
carr = np.array([refA["C"][d] for d in DOMS])
rows.append(["C-only-LR"] + ["%.4f" % v for v in carr] + ["%.4f" % carr.mean(), "%+.4f" % (carr.mean() - vonly.mean())])
table("TABLE 2  setA cross-domain transfer AUC, per target domain n=300 (models fit on Xtr)", rows)
log("  Delta_vs_V = mean over 5 domains of (head_d - V-only-LR_d); V-only-LR = Xtr-fit single-channel row above.")
log("  E3d/E1 anchors (full-train fit) -- replication below:")
log("    V  full-train replication: cd1=%.4f cd2=%.4f dfdcp=%.4f ffiw=%.4f wild=%.4f | in-dom=%.4f"
    % (anch["V"]["cd1"], anch["V"]["cd2"], anch["V"]["dfdcp"], anch["V"]["ffiw"], anch["V"]["wild"], anch["V"]["te800"]))
log("    V  E3d published full row:  cd1=0.8286 cd2=0.8633 dfdcp=0.8261 ffiw=0.8244 wild=0.8090 | in-dom=0.9852")
log("    C  full-train replication: cd1=%.4f cd2=%.4f dfdcp=%.4f ffiw=%.4f wild=%.4f | in-dom=%.4f"
    % (anch["C"]["cd1"], anch["C"]["cd2"], anch["C"]["dfdcp"], anch["C"]["ffiw"], anch["C"]["wild"], anch["C"]["te800"]))
log("    C  E3d published full row:  cd1=0.6559 cd2=0.7227 dfdcp=0.7068 ffiw=0.8344 wild=0.7170 | in-dom=0.9108")

# =====================================================================
# TABLE 3 : setB in-domain
# =====================================================================
rows = [["head", "mean+-std AUC (3 seeds)"]]
for kind in ["H0", "H1", "H2", "H3"]:
    rows.append([kind, fmt_cell("setB te800 " + kind, B[kind]["vals"]["te800"], notes)])
rows.append(["V_proj-only-LR (Xtr fit)", fmt_ms(*meanstd([refB["V_proj"]["te800"]]))])
rows.append(["C_proj-only-LR (Xtr fit)", fmt_ms(*meanstd([refB["C_proj"]["te800"]]))])
table("TABLE 3  setB in-domain AUC on FF++ 800-test (concat V_proj+C_proj, 1536-d)", rows)
log("  setB cross-domain: feats_multi.npz has only V/C/F channels (no V_proj/C_proj) -> NOT MEASURED.")
log("  E0/E3e published full-train in-domain anchors for the setB channels: V_proj=0.9850  C_proj=0.8528")
log("  (reference rows above are Xtr-fit recomputes under the same protocol as the heads)")

# =====================================================================
# TABLE 4 : FFIW near/far halves (setA)
# =====================================================================
rows = [["head", "near-half AUC", "far-half AUC"]]
for kind in ["H0", "H1", "H2", "H3"]:
    rows.append([kind, fmt_cell("setA ffiw_near " + kind, A[kind]["vals"]["ffiw_near"], notes),
                 fmt_cell("setA ffiw_far " + kind, A[kind]["vals"]["ffiw_far"], notes)])
rows.append(["V-only-LR", fmt_ms(*meanstd([refA["V"]["ffiw_near"]])), fmt_ms(*meanstd([refA["V"]["ffiw_far"]]))])
rows.append(["C-only-LR", fmt_ms(*meanstd([refA["C"]["ffiw_near"]])), fmt_ms(*meanstd([refA["C"]["ffiw_far"]]))])
table("TABLE 4  setA FFIW-domain halves by source-distance median (E2 k=10; n=150 each)", rows)
log("  E2 (k=10) published ffiw halves, full-train V probe: near=0.863 far=0.784 ; our full-train V replication:"
    " near=%.4f far=%.4f" % (anch["V"]["ffiw_near"], anch["V"]["ffiw_far"]))

# =====================================================================
# TABLE 5 : pooled 5-domain terciles (setA)
# =====================================================================
rows = [["head", "near (n=%d)" % len(terc_p["near"]), "mid (n=%d)" % len(terc_p["mid"]),
         "far (n=%d)" % len(terc_p["far"])]]
for kind in ["H0", "H1", "H2", "H3"]:
    rows.append([kind] + [fmt_cell("setA terc " + q + " " + kind, A[kind]["vals"]["t_" + q], notes)
                          for q in ["near", "mid", "far"]])
rows.append(["V-only-LR"] + [fmt_ms(*meanstd([refA["V"]["t_" + q]])) for q in ["near", "mid", "far"]])
rows.append(["C-only-LR"] + [fmt_ms(*meanstd([refA["C"]["t_" + q]])) for q in ["near", "mid", "far"]])
table("TABLE 5  setA pooled 5-domain terciles by absolute source distance (E2 k=10)", rows)
log("  E2 published global terciles, full-train V probe: near=0.9211 mid=0.7586 far=0.7199 ; our replication:"
    " near=%.4f mid=%.4f far=%.4f" % (anch["V"]["t_near"], anch["V"]["t_mid"], anch["V"]["t_far"]))

# =====================================================================
# 7. NOTES
# =====================================================================
log("")
log("NOTES / APPENDIX")
log("  total wall time = %.1fs" % (time.time() - T0))
log("  per-head wall (setA+setB train+eval): %s" % ", ".join(
    "H%d=%.1fs" % (i, A["H%d" % i]["wall"] + B["H%d" % i]["wall"]) for i in range(4)))
log("  splits: Xtr n=%d (%d videos) | Xval n=%d (%d videos) | Xte n=%d (%d videos); val drawn video-grouped, rng(0)"
    % (len(Xtr0), n_vtr, len(Xval0), n_vval, len(Xte0), n_vte))
log("  setA d=1792 ; setB d=1536 ; every scaler/LR fit on Xtr only; NN: batch 128, AdamW lr=1e-3 wd=1e-4,")
log("       BCEWithLogits, max 40 epochs, early stop patience 8 on Xval AUC, best-epoch weights restored.")
log("  H0 = linear concat anchor (equivalent E0/E3d probe protocol, single value).")
log("  E2 distance-k note: run_phaseA.py prints per-domain median halves for k in {5,10} and computes the")
log("       global terciles with k=10 only; here k=10 is used for BOTH table 4 and table 5.")
log("  seed variance flags (std>0.01 over the 3 seeds):")
for n in notes:
    log(n)
if not notes:
    log("    (none)")
log("  file layout: this report reproduces stdout 1:1; setB cross-domain rows absent by design (no proj")
log("       features in feats_multi.npz); ffpp block (800 rows) inside feats_multi.npz excluded from all")
log("       scoring (Phase-A protocol declares only the 5 x 300 target domains).")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(_rep) + "\n")
log("")
log("saved report -> %s" % REPORT)
log("[e4_fusion] done.")
