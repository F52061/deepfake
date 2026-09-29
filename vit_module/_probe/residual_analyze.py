# -*- coding: utf-8 -*-
"""CPU-only analysis of vit_module/_probe/probe_feats.npz.

Re-runs all probes / CKA / effective-rank metrics and writes the text report,
without needing the GPU extraction step again. Mirrors the analysis tail of
residual_probe.py.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_NUM_THREADS'] = '1'

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'probe_feats.npz')


def l2_logistic_auc(Xtr, ytr, Xte, yte, seed=0, C=1.0):
    ytr = np.asarray(ytr).ravel(); yte = np.asarray(yte).ravel()
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float('nan'), None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, s), s


def bootstrap_auc(y, s, n_iter=2000, seed=0):
    y = np.asarray(y).ravel(); s = np.asarray(s).ravel()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); aucs = []
    for _ in range(n_iter):
        ii = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[ii])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[ii], s[ii]))
        except ValueError:
            pass
    if not aucs:
        return (float('nan'), float('nan'))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))


def effective_ranks(X, energy=0.9):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s * s; tot = s2.sum()
    if tot <= 0 or s[0] <= 0:
        return (float('nan'), float('nan'), int(0))
    pr = float((s.sum() ** 2) / tot)
    cum = np.cumsum(s2) / tot
    n90 = int(np.searchsorted(cum, energy) + 1)
    nnum = int((s > s[0] * 1e-6).sum())
    return pr, n90, nnum


def main():
    d = np.load(NPZ, allow_pickle=True)
    C = d['C'].astype(np.float64)          # raw CLIP CLS 1024
    V = d['V'].astype(np.float64)          # raw ViT CLS 768
    R = d['R'].astype(np.float64)          # train-fitted residual 768
    Cp = d['C_proj'].astype(np.float64)    # projected head-input 768
    Vp = d['V_proj'].astype(np.float64)
    y = d['y'].astype(np.int64)            # 1=real, 0=fake
    trmask = d['train_mask'].astype(bool)
    paths = d['paths']

    Ctr, Cte = C[trmask], C[~trmask]
    Vtr, Vte = V[trmask], V[~trmask]
    Rtr, Rte = R[trmask], R[~trmask]
    Cp_tr, Cp_te = Cp[trmask], Cp[~trmask]
    Vp_tr, Vp_te = Vp[trmask], Vp[~trmask]
    ytr, yte = y[trmask], y[~trmask]

    ztr = (ytr == 0).astype(np.int64)      # probe target: fake=1
    zte = (yte == 0).astype(np.int64)

    # sanity: stored R should equal a fresh Ridge residual (train-fit) almost exactly
    sc = StandardScaler().fit(Ctr)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Ctr), Vtr)
    Rtr2 = Vtr - ridge.predict(sc.transform(Ctr))
    Rte2 = Vte - ridge.predict(sc.transform(Cte))
    corr = float(np.corrcoef(np.concatenate([Rtr2, Rte2]).ravel(),
                             np.concatenate([Rtr, Rte]).ravel())[0, 1])
    print(f'[sanity] corr(stored R, recomputed R) = {corr:.6f}')

    auc_C, s_C = l2_logistic_auc(Ctr, ztr, Cte, zte)
    auc_V, s_V = l2_logistic_auc(Vtr, ztr, Vte, zte)
    auc_R, s_R = l2_logistic_auc(Rtr, ztr, Rte, zte)
    ci_C = bootstrap_auc(zte, s_C)
    ci_V = bootstrap_auc(zte, s_V)
    ci_R = bootstrap_auc(zte, s_R)

    # sensitivity on projected head-input pair
    auc_Cp, _ = l2_logistic_auc(Cp_tr, ztr, Cp_te, zte)
    auc_Vp, _ = l2_logistic_auc(Vp_tr, ztr, Vp_te, zte)
    scp = StandardScaler().fit(Cp_tr)
    ridge_p = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scp.transform(Cp_tr), Vp_tr)
    Rp_tr = Vp_tr - ridge_p.predict(scp.transform(Cp_tr))
    Rp_te = Vp_te - ridge_p.predict(scp.transform(Cp_te))
    auc_Rp, s_Rp = l2_logistic_auc(Rp_tr, ztr, Rp_te, zte)
    ci_Rp = bootstrap_auc(zte, s_Rp)

    C_all = np.concatenate([Ctr, Cte], axis=0)
    V_all = np.concatenate([Vtr, Vte], axis=0)
    R_all = np.concatenate([Rtr, Rte], axis=0)
    cka_all = linear_cka(V_all, C_all)
    cka_tr = linear_cka(Vtr, Ctr)
    erV = effective_ranks(V_all)
    erC = effective_ranks(C_all)
    erR = effective_ranks(R_all)

    # decision
    if auc_R >= 0.60 and ci_R[0] > 0.50:
        decision = 'COMPLEMENTARY (residual carries fake-discriminative info)'
        reason = (f'probe(R) AUC={auc_R:.4f} (95% CI {ci_R[0]:.4f}-{ci_R[1]:.4f}) clearly > 0.5: '
                  f'the ViT branch adds info not linearly predictable from CLIP.')
    elif (abs(auc_R - 0.5) < 0.03) or (ci_R[0] <= 0.50 and auc_R < 0.60):
        decision = 'REDUNDANT (no reliable residual signal)'
        reason = (f'probe(R) AUC={auc_R:.4f} (95% CI {ci_R[0]:.4f}-{ci_R[1]:.4f}) ~ 0.5: '
                  f'ViT residual is linearly redundant w.r.t. CLIP.')
    else:
        decision = 'WEAK / INCONCLUSIVE'
        reason = f'probe(R) AUC={auc_R:.4f} (95% CI {ci_R[0]:.4f}-{ci_R[1]:.4f}).'

    lines = []
    lines.append('=' * 78)
    lines.append('R-TEST / RESIDUAL PROBE REPORT  (M2F2-Det Stage-1 detector)')
    lines.append('=' * 78)
    lines.append('')
    lines.append(f'npz input         : {NPZ}')
    lines.append(f'total samples     : {len(y)}  (train {int(trmask.sum())}, test {int((~trmask).sum())})')
    lines.append(f'labels (1=real/0=fake): real={int((y==1).sum())}, fake={int((y==0).sum())}')
    lines.append('')
    lines.append('Feature definitions')
    lines.append(f'  C : raw CLIP vision tower CLS  (hidden_states[-2], position 0), {C.shape[1]}-d, BEFORE vision_proj')
    lines.append(f'  V : raw ViT (PDI-initialized) final CLS token after final LayerNorm, {V.shape[1]}-d, BEFORE deepfake_proj')
    lines.append('  R : V - Ridge_pred(V|C); Ridge fit on probe-train, applied to all')
    lines.append('  (sensitivity: C_proj / V_proj = the projected 768-d head-input pair)')
    lines.append('')
    lines.append('AUC (L2-logistic probe; target fake=1; CI = 2.5/97.5 bootstrap on test)')
    lines.append(f'  probe(C)      AUC = {auc_C:.4f}   95% CI [{ci_C[0]:.4f}, {ci_C[1]:.4f}]')
    lines.append(f'  probe(V)      AUC = {auc_V:.4f}   95% CI [{ci_V[0]:.4f}, {ci_V[1]:.4f}]')
    lines.append(f'  probe(R)      AUC = {auc_R:.4f}   95% CI [{ci_R[0]:.4f}, {ci_R[1]:.4f}]   <-- CORE')
    lines.append('')
    lines.append('Sensitivity (head-input projected 768-d features)')
    lines.append(f'  probe(C_proj) AUC = {auc_Cp:.4f}   probe(V_proj) AUC = {auc_Vp:.4f}   '
                 f'probe(R_proj) AUC = {auc_Rp:.4f}   95% CI [{ci_Rp[0]:.4f}, {ci_Rp[1]:.4f}]')
    lines.append('')
    lines.append(f'Linear CKA (V, C) : all-data = {cka_all:.4f}    probe-train = {cka_tr:.4f}')
    lines.append('Effective rank (column-centered SVD; pooled all data):')
    lines.append('  format = (participation_ratio, n_dims_90%_energy, numerical_rank)')
    lines.append(f'  V : {erV}')
    lines.append(f'  C : {erC}')
    lines.append(f'  R : {erR}')
    lines.append('')
    lines.append('Notes:')
    lines.append('  * video-level (source identity) 70/30 split done in residual_probe.py before extraction.')
    lines.append('  * linear CKA on centered Gram matrices; effective rank on column-centered data matrix '
                 'SVD; participation ratio = sum(s)^2/sum(s^2); numerical_rank = #s > 1e-6*s_max.')
    lines.append(f'  * sanity: corr(stored R, freshly refit R) = {corr:.6f}')
    lines.append('')
    lines.append(f'DECISION : {decision}')
    lines.append(f'REASON  : {reason}')
    lines.append('=' * 78)
    txt = '\n'.join(lines)
    print(txt)

    out = os.path.join(HERE, 'residual_probe_report.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f'[save] report -> {out}')

    print('\n===== SUMMARY =====')
    print(f'probe(C)={auc_C:.4f}  probe(V)={auc_V:.4f}  probe(R)={auc_R:.4f} (CI {ci_R[0]:.4f}-{ci_R[1]:.4f})')
    print(f'CKA(V,C)={cka_all:.4f}  effrank V={erV} C={erC} R={erR}')
    print(f'DECISION: {decision}')


if __name__ == '__main__':
    main()
