# -*- coding: utf-8 -*-
"""
E3a  - discriminative spectrum (raw-cov PCA, primary)          [feature: V]
E3b  - truncation causal test (topK vs tailK), main+sensitivity
E3b-null - label-shuffle + spectrum-matched random-direction nulls
E3c  - three-channel spectral comparison (V, C, R)

Only reads pre-existing feature npz; no GPU / no model forward.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import e3_common as C

SEP = "=" * 92
SUB = "-" * 92


def spectrum_report(Xtr, ytr, nbins=16):
    """Run E3a single-channel discrimination-spectrum. Returns dict of arrays/scalars."""
    mu, E, lam = C.pca_basis(Xtr)
    d = Xtr.shape[1]
    proj = (Xtr - mu) @ E
    F = np.array([C.fisher_dir(proj[:, k], ytr) for k in range(d)])
    corr = np.array([np.corrcoef(proj[:, k], ytr)[0, 1] for k in range(d)])
    cumE = np.cumsum(lam) / lam.sum()
    sp_F = C.spearman(np.log10(lam + 1e-30), F)
    sp_abs = C.spearman(np.log10(lam + 1e-30), np.abs(corr))
    disc = C.cum_disc_dims(F, lam)
    bins = C.bin_spec(F, lam, nbins=nbins)
    topF_ranks = np.argsort(F)[::-1][:10].tolist()
    return dict(mu=mu, E=E, lam=lam, loglam=np.log10(lam + 1e-30), F=F, corr=corr,
                cumE=cumE, sp_F=sp_F, sp_abs=sp_abs, disc=disc, bins=bins,
                topF_ranks=topF_ranks, d=d)


def trunc_table(chX, tr, te, ytr, yte, Ks, modes=("raw", "corr")):
    """E3b truncation for one channel (chX = full 3000-row matrix; tr/te masks)."""
    out = {}
    Xtr = chX[tr]; Xte = chX[te]
    full_auc = C.probe_auc(Xtr, ytr, Xte, yte)
    out["full"] = full_auc
    for mode in modes:
        res = C.topk_tailk_curve(Xtr, ytr, Xte, yte, Ks, mode=mode)
        out[mode] = res
    return out


def fmt_table(lines, header, rows):
    lines.append(header)
    lines.append("-" * len(header))
    lines.extend(rows)


def main():
    p = C.load_probe()
    y = p["y"]; tr = p["train_mask"].astype(bool); te = ~tr
    ytr, yte = y[tr], y[te]

    C.reset_report("E3 variance-spectrum attribution report (M2F2-Det hyy, only pre-extracted features)\n"
                   "Protocol: StandardScaler(train)->LogisticRegression(C=1e-3,lbfgs).  y: real=1/fake=0.\n"
                   "PCA = center-only raw-cov PCA unless marked 'corr' (= StandardScaler(train) then PCA).\n"
                   f"probe split: train n={int(tr.sum())}, test n={int(te.sum())}, video-disjoint.\n")
    L = []
    L.append(SEP)
    L.append("E3a  - V discriminative spectrum  (raw-cov PCA on train V, 768 dirs)")
    L.append(SEP)

    V = p["V"].astype(np.float64)
    sa = spectrum_report(V[tr], ytr)
    full_auc = C.probe_auc(V[tr], ytr, V[te], yte)
    L.append(f"full-feature video-split AUC(V) = {full_auc:.4f}")
    L.append(f"top-10 Fisher variance-ranks : {sa['topF_ranks']}")
    L.append(f"variance rank of max-F dir   : {int(np.argmax(sa['F']))}   (rank 0 = LARGEST eigenvalue)")
    L.append(f"Spearman(log10 lambda, F)    : {sa['sp_F']:+.4f}")
    L.append(f"Spearman(log10 lambda,|corr|): {sa['sp_abs']:+.4f}")
    L.append("cumulative discrimination (directions sorted by F desc):")
    L.append(f"  {'frac':>6s} {'n_dims':>7s} {'energy_frac':>12s}")
    for d in sa["disc"]:
        L.append(f"  {d['fraction']:>6.0%} {d['n_dims']:>7d} {d['energy_frac']:>12.4f}")
    L.append("energy thresholds: n_dims for 10/50/90% cumulative energy = "
             f"{int(np.searchsorted(sa['cumE'],0.10)+1)}/{int(np.searchsorted(sa['cumE'],0.50)+1)}/"
             f"{int(np.searchsorted(sa['cumE'],0.90)+1)}")
    L.append("binned spectrum (variance order -> 16 bins, rank0 = largest var):")
    hdr = f"  {'bin':>3s} {'ranks':>10s} {'mean_log10lam':>14s} {'mean_F':>10s} {'max_F':>10s} {'energy%':>8s}"
    L.append(hdr); L.append("  " + "-" * (len(hdr) - 2))
    for b in sa["bins"]:
        L.append(f"  {b['bin']:>3d} {str((b['rank_lo'],b['rank_hi'])):>10s} {b['mean_loglam']:>14.3f} "
                 f"{b['mean_F']:>10.4f} {b['max_F']:>10.4f} {100*b['energy_frac']:>7.2f}%")
    L.append("")

    # ---------- E3b on V: main + sensitivity ----------
    L.append(SEP); L.append("E3b  - V truncation causal test (raw-cov PCA) + corr-PCA sensitivity")
    L.append(SEP)
    Ks = [1, 2, 5, 10, 20, 50, 100, 200]
    trunc = trunc_table(V, tr, te, ytr, yte, Ks)
    L.append(f"full AUC(V) = {trunc['full']:.4f}")
    for mode in ["raw", "corr"]:
        res = trunc[mode]
        L.append(f"mode={mode}:  K -> AUC_topK | AUC_tailK")
        row = []
        for K in Ks:
            t = res.get(("top", K), float("nan")); tl = res.get(("tail", K), float("nan"))
            row.append(f"K={K:<4d} topK={t:.4f} tailK={tl:.4f}")
        L.append("  " + "; ".join(row))

    # ---------- E3b-null ----------
    L.append(""); L.append(SEP); L.append("E3b-null  - V controls (K=50)")
    L.append(SEP)
    K0 = 50
    mu, E, lam = C.pca_basis(V[tr])
    Ztop_tr = (V[tr] - mu) @ E[:, :K0];  Ztop_te = (V[te] - mu) @ E[:, :K0]
    Ztl_tr  = (V[tr] - mu) @ E[:, K0:];  Ztl_te  = (V[te] - mu) @ E[:, K0:]
    # null-1 label shuffle, averaged over NSEEDS draws (mean +- std)
    NSEEDS = 30
    top_aucs = []; tail_aucs = []
    for s in range(NSEEDS):
        rng = np.random.default_rng(s)
        ysh = ytr.copy(); rng.shuffle(ysh)
        top_aucs.append(C.probe_auc(Ztop_tr, ysh, Ztop_te, yte))
        tail_aucs.append(C.probe_auc(Ztl_tr, ysh, Ztl_te, yte))
    a_top = float(np.mean(top_aucs)); sd_top = float(np.std(top_aucs))
    a_tail = float(np.mean(tail_aucs)); sd_tail = float(np.std(tail_aucs))
    L.append(f"[null-1 label-shuffle train y, {NSEEDS} seeds, PROJECTED top/tail K={K0}]  "
             f"topK AUC={a_top:.4f}+-{sd_top:.4f}   tailK AUC={a_tail:.4f}+-{sd_tail:.4f}   (expect ~0.5)")
    # null-2 Gaussian surrogate with the tail's own per-direction std (no label coupling)
    m = V.shape[1] - K0
    sd_real = Ztl_tr.std(0)
    rng = C.rng()
    gtr = rng.standard_normal((Ztl_tr.shape[0], m)) * sd_real
    gte = rng.standard_normal((Ztl_te.shape[0], m)) * sd_real
    a_null2 = C.probe_auc(gtr, ytr, gte, yte)
    L.append(f"[null-2 Gaussian surrogate, tail per-dir std]  tailK n_dir={m} AUC={a_null2:.4f}  (expect ~0.5)")
    # null-2a random-orth rotation of REAL data (rotation-invariance sanity)
    Q, _ = np.linalg.qr(rng.normal(size=(V.shape[1], V.shape[1])))
    Qm = Q[:, :m]
    Zp_tr = (V[tr] - mu) @ Qm; Zp_te = (V[te] - mu) @ Qm
    a_null2a = C.probe_auc(Zp_tr, ytr, Zp_te, yte)
    L.append(f"[null-2a random-orth projection of REAL V, {m}-d] AUC={a_null2a:.4f}  "
             f"(rotation-invariance: a random {m}/{V.shape[1]}-d subspace keeps ~all signal -> only the top "
             f"~K0 dirs were load-bearing, the tail itself is signal-free; not a label null)")
    L.append("")

    # ---------- E3c three-channel ----------
    L.append(SEP); L.append("E3c - three-channel spectral comparison (V / C / R, probe npz)")
    L.append(SEP)
    Kc = [5, 20, 50, 100]
    res3c = {}
    for ch in ["V", "C", "R"]:
        X = p[ch].astype(np.float64)
        sp = spectrum_report(X[tr], ytr, nbins=16)
        full = C.probe_auc(X[tr], ytr, X[te], yte)
        trc = trunc_table(X, tr, te, ytr, yte, Kc, modes=("raw",))
        res3c[ch] = dict(spectrum=sp, full=full, trunc=trc)
        L.append(f"--- channel {ch} (dim={X.shape[1]})  full AUC={full:.4f}")
        L.append(f"    top-6 Fisher variance-ranks: {sp['topF_ranks'][:6]}")
        L.append(f"    max-F variance rank: {int(np.argmax(sp['F']))}   Spearman(loglam,F)={sp['sp_F']:+.4f}   Spearman(loglam,|corr|)={sp['sp_abs']:+.4f}")
        cum = sp["disc"]
        L.append(f"    cumulative F: 10%->{cum[0]['n_dims']}dir/{cum[0]['energy_frac']:.3f}E  50%->{cum[1]['n_dims']}dir/{cum[1]['energy_frac']:.3f}E  "
                 f"90%->{cum[2]['n_dims']}dir/{cum[2]['energy_frac']:.3f}E")
        rr = trc["raw"]
        row = " ".join(f"K={K} topK={rr.get(('top',K),float('nan')):.4f} tailK={rr.get(('tail',K),float('nan')):.4f}" for K in Kc)
        L.append("    trunc(raw): " + row)
        L.append("")

    # CKA / overlap V-C and angle V-R (same 768 space)
    L.append("--- V vs C discriminant-band overlap (same test samples)")
    # top-F eigenvector per channel
    topdir = {}
    for ch in ["V", "C", "R"]:
        sp = res3c[ch]["spectrum"]
        k = int(np.argmax(sp["F"]))
        topdir[ch] = sp["E"][:, k]
    def _score(X, sp, tr, te):
        k = int(np.argmax(sp["F"]))
        return ((X[te] - sp["mu"]) @ sp["E"][:, k])
    sV = _score(p["V"].astype(np.float64), res3c["V"]["spectrum"], tr, te)
    sC = _score(p["C"].astype(np.float64), res3c["C"]["spectrum"], tr, te)
    rVC = np.corrcoef(sV, sC)[0, 1]
    L.append(f"    corr(V top-F proj, C top-F proj) on test = {rVC:+.4f}")
    # linear CKA on test
    def cka(A, B):
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
        K = A @ A.T; Lm = B @ B.T
        return float((K * Lm).sum() / np.sqrt((K * K).sum() * (Lm * Lm).sum()))
    ckaVC = cka(p["V"][te].astype(np.float64), p["C"][te].astype(np.float64))
    L.append(f"    linear CKA(V,C) on test = {ckaVC:.4f}   (Phase A reported ~0.14)")
    # probe logit corr V vs C on test
    clfV, scV = C.fit_probe(p["V"][tr].astype(np.float64), ytr)
    clfC, scC = C.fit_probe(p["C"][tr].astype(np.float64), ytr)
    logitV = clfV.decision_function(scV.transform(p["V"][te].astype(np.float64)))
    logitC = clfC.decision_function(scC.transform(p["C"][te].astype(np.float64)))
    L.append(f"    corr(V-probe logit, C-probe logit) on test = {np.corrcoef(logitV, logitC)[0,1]:+.4f}")
    # V vs R same-space angle
    cosVR = float(np.abs(np.dot(topdir["V"], topdir["R"])))
    L.append(f"    V vs R top-F eigenvector |cos angle| (same 768-d space) = {cosVR:.4f}")
    # R discriminant variance-order vs V (are ranks comparable? both self-normalized)
    L.append("    NOTE: 'lower variance order' = larger rank index within each channel's own spectrum.")
    L.append("")

    C.write_report("\n".join(L))

    # save curves
    curves = {}
    for ch in ["V", "C", "R"]:
        sp = res3c[ch]["spectrum"]
        curves[ch + "_loglam"] = sp["loglam"]
        curves[ch + "_F"] = sp["F"]
        curves[ch + "_corr"] = sp["corr"]
        curves[ch + "_cumE"] = sp["cumE"]
        curves[ch + "_topF_ranks"] = np.array(sp["topF_ranks"])
    # V full E3b K curves
    for mode in ["raw", "corr"]:
        curves["E3b_V_" + mode + "_topK"] = np.array([trunc[mode].get(("top", K), np.nan) for K in Ks])
        curves["E3b_V_" + mode + "_tailK"] = np.array([trunc[mode].get(("tail", K), np.nan) for K in Ks])
    curves["E3b_Ks"] = np.array(Ks)
    for ch in ["V", "C", "R"]:
        curves["E3c_" + ch + "_topK"] = np.array([res3c[ch]["trunc"]["raw"].get(("top", K), np.nan) for K in Kc])
        curves["E3c_" + ch + "_tailK"] = np.array([res3c[ch]["trunc"]["raw"].get(("tail", K), np.nan) for K in Kc])
    curves["E3c_Ks"] = np.array(Kc)
    curves["null1_top"] = a_top; curves["null1_tail"] = a_tail; curves["null2"] = a_null2
    C.save_curves(curves)

    print("\n".join(L))
    print("\n[e3_abc] done.")


if __name__ == "__main__":
    main()
