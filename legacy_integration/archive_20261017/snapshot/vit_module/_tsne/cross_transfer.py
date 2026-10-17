# -*- coding: utf-8 -*-
"""Cross-domain transfer probe: train L2-logistic on FF++ features, test AUC on
each other domain for V / C / F. Appends a section to tsne_report.txt."""
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
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'feats_multi.npz')
REPORT = os.path.join(HERE, 'tsne_report.txt')
TARGETS = ['cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']


def probe_auc(Xtr, ytr, Xte, yte, seed=0):
    ytr_b = (ytr == 0).astype(int)
    yte_b = (yte == 0).astype(int)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=seed)
    clf.fit(sc.transform(Xtr), ytr_b)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    try:
        return float(roc_auc_score(yte_b, s))
    except ValueError:
        return float('nan')


def main():
    d = np.load(NPZ, allow_pickle=True)
    V = d['V'].astype(np.float64)
    C = d['C'].astype(np.float64)
    F = d['F'].astype(np.float64)
    y = d['y'].astype(int)
    dom = d['domain']

    mff = dom == 'ffpp'
    lines = []
    lines.append('')
    lines.append('=' * 92)
    lines.append('CROSS-DOMAIN TRANSFER  (linear probe TRAINED on FF++ only, tested on each domain)')
    lines.append('=' * 92)
    lines.append(f'{"target":>6s} {"AUC_V":>9s} {"AUC_C":>9s} {"AUC_F":>9s}')
    lines.append('-' * 40)
    res = {}
    for t in TARGETS:
        mt = dom == t
        aV = probe_auc(V[mff], y[mff], V[mt], y[mt])
        aC = probe_auc(C[mff], y[mff], C[mt], y[mt])
        aF = probe_auc(F[mff], y[mff], F[mt], y[mt])
        res[t] = (aV, aC, aF)
        lines.append(f'{t:>6s} {aV:>9.4f} {aC:>9.4f} {aF:>9.4f}')
    lines.append('')
    lines.append('FF++ source in-domain reference AUC (5-fold): V=0.9969 C=0.9982 F=0.9957')
    lines.append('Interpretation: if target AUC << source in-domain AUC -> cross-domain drop;')
    lines.append('a higher AUC_F vs AUC_V on a target = the fused feature generalises better.')
    lines.append('=' * 92)
    txt = '\n'.join(lines)
    print(txt)
    with open(REPORT, 'a', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('[appended]', REPORT)

    print('\n===== TRANSFER SUMMARY =====')
    for t in TARGETS:
        print(f'  {t:>5s}: V={res[t][0]:.4f}  C={res[t][1]:.4f}  F={res[t][2]:.4f}')


if __name__ == '__main__':
    main()
