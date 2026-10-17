# -*- coding: utf-8 -*-
"""Block G5: real bridge_v2 stage-1 classification-head weight attribution.

Read-only: loads the stage-1 checkpoint head weights (2,1664) and the probe
feature npz PCA bases. No GPU, no forward pass, pure linear algebra on tensors.
Output written to _head/head_report.txt and printed to stdout.
"""
import os, sys, gc
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
torch.set_num_threads(1)
from scipy.stats import spearmanr

BASE = r"E:/Cross-domain_authentication_verification/Next_work/M2F2_Det-main-hyy"
CKPT = os.path.join(BASE, "checkpoints", "stage_1", "bridge_v2_phase1.pth")
NPZ  = os.path.join(BASE, "vit_module", "_probe", "probe_feats.npz")
OUT  = os.path.join(BASE, "vit_module", "_head", "head_report.txt")

L = []  # report lines
def log(s=""):
    L.append(str(s))
    print(s, flush=True)

def load_checkpoint_head():
    log(f"[G5] loading checkpoint: {CKPT}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck.items()}
    del ck; gc.collect()
    log(f"[G5] state_dict entries: {len(sd)}")

    # locate keys of interest
    cand_keys = [k for k in sd.keys() if any(t in k for t in ("output", "head", "classifier", "alpha"))]
    log("[G5] keys matching output/head/classifier/alpha:")
    for k in sorted(cand_keys):
        v = sd[k]
        log(f"    {k}  shape={tuple(v.shape)} dtype={v.dtype}")
    # all linear weights with out-dim == 2
    log("[G5] all weight keys with shape[0]==2:")
    for k in sorted(sd.keys()):
        v = sd[k]
        if getattr(v, "ndim", 0) == 2 and v.shape[0] == 2:
            log(f"    {k}  shape={tuple(v.shape)}")
    return sd

def main():
    log("=" * 78)
    log("BLOCK G5 - REAL bridge_v2 head attribution (spectral / contribution)")
    log("=" * 78)

    sd = load_checkpoint_head()

    # ---- resolve head weight ----
    wkeys = [k for k in sd if "weight" in k and getattr(sd[k], "ndim", 0) == 2 and sd[k].shape[0] == 2]
    # prefer exact "output.weight"
    if "output.weight" in sd:
        hkey = "output.weight"
    elif wkeys:
        hkey = wkeys[0]
    else:
        hkey = None
    if hkey is None:
        log("[G5] ERROR: no (2,*)-shaped linear weight found for head.")
        return
    W = sd[hkey].float().detach().numpy()
    b = sd["output.bias"].float().detach().numpy() if "output.bias" in sd else None
    D = W.shape[1]
    log(f"\n[A] located head key = '{hkey}'  weight shape = {W.shape}")
    log(f"[A] head.bias shape = {tuple(b.shape) if b is not None else None}")
    log(f"[A] (2,1664) holds: {D == 1664}   (actual F dim = {D})")

    av = sd.get("clip_vision_alpha", None)
    tv = sd.get("clip_text_alpha", None)
    if av is not None: log(f"[A] clip_vision_alpha = {av.float().item():.6f}  shape={tuple(av.shape)}")
    if tv is not None: log(f"[A] clip_text_alpha   = {tv.float().item():.6f}  shape={tuple(tv.shape)}")
    del sd; gc.collect()

    if D != 1664:
        log(f"[G5] WARNING: F dim != 1664, use actual D={D}; slicing below assumes blocks "
            f"(0:{D//2-64})? not applied. ABORT slicing.")
        # still try to infer: two 768 and one 128 if D==1664 else generic
        if D == 1664:
            pass
        else:
            log("[G5] cannot slice unknown layout -> stop.")
            return

    # ---- three-block slicing [C_proj | bridge | V_proj] ----
    slices = {"A_Cproj(0:768)": (0, 768), "bridge(768:896)": (768, 896), "B_Vproj(896:1664)": (896, 1664)}
    names = list(slices.keys())
    Wf = W.astype(np.float64)
    delta = Wf[1] - Wf[0]  # 1664

    Wtot2 = (Wf ** 2).sum()
    d2tot = (delta ** 2).sum()
    d1tot = np.abs(delta).sum()

    log("\n[B] three-block weight share table")
    hdr = f"{'block':<22}{'cols':<12}{'L2w_share':>12}{'L2d_share':>12}{'L1d_share':>12}"
    log(hdr); log("-" * len(hdr))
    shares = {}
    for nm, (a, c) in slices.items():
        sl = Wf[:, a:c]
        dl = delta[a:c]
        l2w = (sl ** 2).sum() / Wtot2
        l2d = (dl ** 2).sum() / d2tot
        l1d = np.abs(dl).sum() / d1tot
        shares[nm] = (l2w, l2d, l1d)
        log(f"{nm:<22}[{a}:{c}){l2w*100:>11.3f}%{l2d*100:>11.3f}%{l1d*100:>11.3f}%")

    # ---- PCA bases from probe npz train rows ----
    log("\n[G5] loading probe npz for PCA bases ...")
    d = np.load(NPZ)
    Vp = d["V_proj"]; Cp = d["C_proj"]; tm = d["train_mask"]
    Vtr = Vp[tm].astype(np.float64); Ctr = Cp[tm].astype(np.float64)
    log(f"[G5] probe npz V_proj{tm.shape} train n={Vtr.shape[0]}  C_proj train n={Ctr.shape[0]}")

    def pca_basis(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        # svd full right basis
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        n = Xc.shape[0]
        lam = (s * s) / (n - 1)
        E = Vt.T  # columns = right singular vectors, descending
        return E, lam, s

    E_V, lam_V, sV = pca_basis(Vtr)
    E_C, lam_C, sC = pca_basis(Ctr)
    log(f"[G5] V_proj train top singular value = {sV[0]:.3e} ; C_proj train top singular = {sC[0]:.3e}")

    dA = delta[0:768]   # block A = C side weight slice
    dB = delta[896:1664]  # block B = V side weight slice

    def expand_stats(wvec, E, lam):
        g = E.T @ wvec            # coefficient in PCA basis (eigen-desc order)
        eg = g ** 2
        tot = eg.sum()
        top5 = eg[:5].sum() / tot
        top50 = eg[:50].sum() / tot
        corr = spearmanr(np.log10(np.maximum(lam, 1e-30)), np.abs(g)).correlation
        return top5, top50, corr, g

    log("\n[C] block A / block B expansions under E_C (C_proj basis) and E_V (V_proj basis)")
    rowhdr = f"{'case':<26}{'basis':<14}{'top5_energy':>12}{'top50_energy':>13}{'spearman':>10}"
    log(rowhdr); log("-" * len(rowhdr))
    results = {}
    for wlab, wv in (("blockA_Δ(0:768)", dA), ("blockB_Δ(896:1664)", dB)):
        for blab, E, lam in (("E_C(C_proj)", E_C, lam_C), ("E_V(V_proj)", E_V, lam_V)):
            top5, top50, corr, g = expand_stats(wv, E, lam)
            key = (wlab, blab)
            results[key] = (top5, top50, corr)
            log(f"{wlab:<26}{blab:<14}{top5*100:>11.3f}%{top50*100:>12.3f}%{corr:>10.4f}")

    # eigenvalue band info (context)
    def band(lam):
        tot = lam.sum()
        c5 = lam[:5].sum() / tot
        c50 = lam[:50].sum() / tot
        pr = (lam.sum() ** 2) / (lam ** 2).sum()
        return c5, c50, pr
    for nm, lam in (("C_proj", lam_C), ("V_proj", lam_V)):
        c5, c50, pr = band(lam)
        log(f"[C] {nm} feature variance band: top5_eig={c5*100:.2f}% top50_eig={c50*100:.2f}% participation_ratio={pr:.2f}")

    # ---- mechanical identification rule ----
    # rule (task): V-side probe weight concentrated (top5>~91%); C-side diffuse.
    # a real-head block whose expansion is *very* concentrated under E_V basis and
    # diffuse under E_C basis => that block corresponds to V.
    log("\n[C] mechanical A/B <-> V/C judgement (by task rule: concentrated in E_V AND diffuse in E_C => V-side)")
    for wlab in ("blockA_Δ(0:768)", "blockB_Δ(896:1664)"):
        t5v, _, _ = results[(wlab, "E_V(V_proj)")]
        t5c, _, _ = results[(wlab, "E_C(C_proj)")]
        log(f"    {wlab}: top5 in E_V={t5v*100:.2f}%  top5 in E_C={t5c*100:.2f}%")

    # Also report the wiring ground-truth for cross-check
    log("\n[C] wiring ground-truth (from vit_m2f2_detector_bridge.py forward): features = "
        "cat([clip_vision_cls(768), bridge(128), vit_features(768)])")
    log("    => block 0:768 feeds clip_vision_cls (= probe C_proj side, CLIP)")
    log("    => block 896:1664 feeds vit_features   (= probe V_proj side, ViT)")

    # ---- Section D: bridge 128 ----
    dl = delta[768:896]
    log("\n[D] bridge 128 block")
    log(f"    bridge Δ-slice L2 share (norm^2): {shares['bridge(768:896)'][1]*100:.4f}%")
    log(f"    bridge Δ-slice |comp| share:       {shares['bridge(768:896)'][2]*100:.4f}%")
    log(f"    bridge Δ-slice L2 norm (raw):      {np.sqrt((dl**2).sum()):.6f}   "
        f"delta total L2 norm: {np.sqrt(d2tot):.6f}")

    log("\n[head_done]")

if __name__ == "__main__":
    main()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
