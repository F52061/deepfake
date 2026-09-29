# -*- coding: utf-8 -*-
"""
Task #3 — numeric report only (PNGs already produced by tsne_visualize.py).

Writes vit_module/_tsne/tsne_report.txt with:
  * per-domain real/fake linear-probe CV AUC on V / C / F
  * per-domain Fisher discriminant ratio on V (LDA-style, reference)
  * cross-domain V-space overlap matrix.

Why not strict CKA here: CKA is defined between two representations of the
SAME set of inputs. Cross-domain feature sets come from different images and
have unequal sizes, so a strict linear CKA is not well defined. We therefore
report a well-defined surrogate: the normalized aligned-subspace overlap of the
top-k principal components of each domain's centered V matrix,
    overlap(A,B) = ||U_A^T U_B||_F^2 / k,   U = top-k right singular vectors.
1 = identical dominant subspace, 0 = orthogonal. This answers "do these domains
overlap in V space". (Same formula, restricted to top-k PCs, is the standard
linear-CKA value between the two k-d subspaces.)
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'feats_multi.npz')
SEED = 0
DOMAIN_ORDER = ['ffpp', 'cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']
DOMAIN_LABEL = {
    'ffpp': 'FF++(test)', 'cd1': 'Celeb-DF v1', 'cd2': 'Celeb-DF v2',
    'dfdcp': 'DFDC-preview', 'ffiw': 'FFIW', 'wild': 'WildDeepfake',
}
KPC = 20  # principal components for subspace overlap


def cv_auc(X, y, n_splits=5, seed=0):
    yb = (y == 0).astype(int)
    if len(np.unique(yb)) < 2:
        return float('nan')
    ns = min(n_splits, int(np.bincount(yb).min()))
    if ns < 2:
        return float('nan')
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    oof = np.zeros(len(yb))
    for tr, te in skf.split(X, yb):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=seed)
        clf.fit(sc.transform(X[tr]), yb[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    try:
        return float(roc_auc_score(yb, oof))
    except ValueError:
        return float('nan')


def fisher_ratio(X, y, reg=1e-3):
    yb = (y == 0).astype(int)
    m0 = X[yb == 0].mean(0); m1 = X[yb == 1].mean(0)
    d = m1 - m0
    S0 = np.cov(X[yb == 0], rowvar=False)
    S1 = np.cov(X[yb == 1], rowvar=False)
    if S0.ndim == 0 or S1.ndim == 0:
        return float('nan')
    Sw = (S0 + S1) / 2 + reg * np.eye(X.shape[1])
    try:
        w = np.linalg.pinv(Sw) @ d
    except np.linalg.LinAlgError:
        return float('nan')
    num = float((w @ d) ** 2)
    den = float(w @ Sw @ w)
    return float(num / den) if den > 0 else float('nan')


def top_subspace(X, k):
    Xc = X - X.mean(0, keepdims=True)
    k = min(int(k), Xc.shape[0] - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T  # d x k


def subspace_overlap(A, B, k=KPC):
    Ua = top_subspace(A, k)
    Ub = top_subspace(B, k)
    kk = Ua.shape[1]
    return float(((Ua.T @ Ub) ** 2).sum() / kk)


def main():
    d = np.load(NPZ, allow_pickle=True)
    V = d['V']; C = d['C']; F = d['F']
    y = d['y'].astype(int)
    dom = d['domain']
    domains = [dm for dm in DOMAIN_ORDER if dm in set(dom.tolist())]

    rows = []
    rows.append('=' * 92)
    rows.append('T-SNE CROSS-DOMAIN REPORT  (M2F2-Det Stage-1 detector)  [numeric]')
    rows.append('=' * 92)
    rows.append(f'npz     : {NPZ}')
    rows.append(f'total n : {len(y)}   real={int((y==1).sum())}  fake={int((y==0).sum())}')
    rows.append(f'PNGs    : tsne_V_bylabel.png / tsne_V_bydomain.png / '
                f'tsne_C_bydomain.png / tsne_F_bydomain.png  (in vit_module/_tsne/)')
    rows.append(f't-SNE   : PCA->50 then TSNE(perplexity=30, seed=0) on V / C / F')
    rows.append('')
    rows.append('Per-domain real/fake separability (5-fold CV linear probe on that domain only; '
                'fake=positive)')
    hdr = f'{"domain":<10s} {"n":>5s} {"AUC_V":>8s} {"AUC_C":>8s} {"AUC_F":>8s} {"FisherV":>12s}'
    rows.append(hdr); rows.append('-' * len(hdr))
    auc_rows = {}
    for dm in domains:
        m = dom == dm
        n = int(m.sum())
        aV = cv_auc(V[m].astype(np.float64), y[m])
        aC = cv_auc(C[m].astype(np.float64), y[m])
        aF = cv_auc(F[m].astype(np.float64), y[m])
        fV = fisher_ratio(V[m].astype(np.float64), y[m])
        auc_rows[dm] = (aV, aC, aF)
        rows.append(f'{dm:<10s} {n:>5d} {aV:>8.4f} {aC:>8.4f} {aF:>8.4f} {fV:>12.4f}')
    rows.append('')
    rows.append('Cross-domain V-space overlap matrix  (aligned top-20-PC subspace; 1=identical '
                'dominant subspace, 0=orthogonal)')
    hdr = f'{"":>8s}' + ''.join(f'{DOMAIN_LABEL[dm][:8]:>10s}' for dm in domains)
    rows.append(hdr)
    mtx = np.zeros((len(domains), len(domains)))
    for i, da in enumerate(domains):
        m_a = dom == da
        line = f'{da:>8s}'
        for j, db in enumerate(domains):
            if i == j:
                v = 1.0
            else:
                m_b = dom == db
                v = subspace_overlap(V[m_a].astype(np.float64), V[m_b].astype(np.float64))
            mtx[i, j] = v
            line += f'{v:>10.3f}'
        rows.append(line)
    rows.append('')
    off = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            off.append((domains[i], domains[j], mtx[i, j]))
    off.sort(key=lambda t: t[2], reverse=True)
    rows.append('Domain-pair V overlap ranking (high -> low):')
    for a, b, v in off:
        rows.append(f'  {a:>5s} vs {b:<5s}  overlap={v:.3f}')
    rows.append('')
    rows.append('Method notes:')
    rows.append('  * AUC_V/C/F: 5-fold CV L2-logistic (features standardized inside each fold).')
    rows.append('  * FisherV: LDA-style generalized Fisher ratio, in-sample, shrinkage 1e-3; '
                'larger = better within-domain real/fake separation.')
    rows.append('  * Overlap: strict linear CKA needs paired same-set samples, so cross-domain '
                '(unpaired, unequal n) uses the aligned top-20 principal-subspace cosine '
                'overlap = ||U_A^T U_B||_F^2 / 20. This is the linear-CKA value of the two '
                '20-d subspaces.')
    rows.append('  * y convention: real=1, fake=0 (as in dataset txt).')
    rows.append('=' * 92)
    txt = '\n'.join(rows)
    print(txt)
    with open(os.path.join(HERE, 'tsne_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('[save] report ->', os.path.join(HERE, 'tsne_report.txt'))

    print('\n===== SUMMARY =====')
    for dm in domains:
        print(f'  {dm:>5s}: AUC_V={auc_rows[dm][0]:.4f}  AUC_C={auc_rows[dm][1]:.4f}  AUC_F={auc_rows[dm][2]:.4f}')
    np.set_printoptions(precision=3, suppress=True)
    print('V overlap matrix:'); print(mtx)


if __name__ == '__main__':
    main()
