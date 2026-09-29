# -*- coding: utf-8 -*-
"""
Task #3 — CPU-only t-SNE visualisation + numeric report.

Loads vit_module/_tsne/feats_multi.npz (F/V/C/y/domain/vid/path) and:

  1. t-SNE (PCA -> 50, perplexity 30, random_state fixed) on V, C, F.
     PNGs: V by label, V by domain, C by domain, F by domain.
  2. Numeric report to vit_module/_tsne/tsne_report.txt:
       - per-domain real/fake linear-probe cross-val AUC for V / C / F
       - per-domain Fisher discriminant ratio on V (LDA-style, reference)
       - cross-domain linear CKA(V) matrix (domain pairwise)

Single-threaded (OMP/MKL/... =1), matplotlib Agg backend.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'feats_multi.npz')
SEED = 0
PERP = 30

DOMAIN_ORDER = ['ffpp', 'cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']
DOMAIN_LABEL = {
    'ffpp': 'FF++(test)', 'cd1': 'Celeb-DF v1', 'cd2': 'Celeb-DF v2',
    'dfdcp': 'DFDC-preview', 'ffiw': 'FFIW', 'wild': 'WildDeepfake',
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cv_auc(X, y, n_splits=5, seed=0):
    """Per-domain real/fake separability via 5-fold linear-probe AUC."""
    yb = (y == 0).astype(int)          # fake-positive target (real=1, fake=0)
    if len(np.unique(yb)) < 2:
        return float('nan')
    n_splits = min(n_splits, int(np.bincount(yb).min()))
    if n_splits < 2:
        return float('nan')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
    """LDA-style generalized Fisher ratio on V: (w.d)^2 / (w^T Sw w), in-sample."""
    yb = (y == 0).astype(int)
    m0 = X[yb == 0].mean(0); m1 = X[yb == 1].mean(0)
    d = m1 - m0
    S0 = np.cov(X[yb == 0], rowvar=False)
    S1 = np.cov(X[yb == 1], rowvar=False)
    if S0.ndim == 0 or S1.ndim == 0:
        return float('nan')
    Sw = (S0 + S1) / 2 + reg * np.eye(X.shape[1])
    # pinv (LDA direction)
    try:
        w = np.linalg.pinv(Sw) @ d
    except np.linalg.LinAlgError:
        return float('nan')
    num = float((w @ d) ** 2)
    den = float(w @ Sw @ w)
    return float(num / den) if den > 0 else float('nan')


def cka_ab(A, B):
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    m = A.T @ B
    num = float((m * m).sum())
    da = float(((A.T @ A) ** 2).sum())
    db = float(((B.T @ B) ** 2).sum())
    if da <= 0 or db <= 0:
        return float('nan')
    return num / math.sqrt(da * db)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def scatter_embed(ax, emb, c, cmap, title, alpha=0.6, s=8):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap=cmap, s=s, alpha=alpha,
                    edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    return sc


def add_legend_categorical(ax, sc, categories, cmap, title):
    import matplotlib.patches as mpatches
    norm = plt.Normalize(vmin=0, vmax=len(categories) - 1)
    handles = [mpatches.Patch(color=cmap(norm(i)), label=categories[i])
               for i in range(len(categories))]
    leg = ax.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.9,
                    title=title, title_fontsize=8, markerscale=1.5)
    leg.get_frame().set_linewidth(0.5)


def tsne_embed(X, n_components=2):
    X50 = PCA(n_components=50, random_state=SEED).fit_transform(X)
    return TSNE(n_components=n_components, perplexity=PERP, random_state=SEED,
                init='pca', learning_rate='auto', max_iter=1000).fit_transform(X50)


# ---------------------------------------------------------------------------
def main():
    d = np.load(NPZ, allow_pickle=True)
    V = d['V']; C = d['C']; F = d['F']
    y = d['y'].astype(int)
    dom = d['domain']
    paths = d['path'] if 'path' in d else None

    domains = [dm for dm in DOMAIN_ORDER if dm in set(dom.tolist())]
    present = sorted(set(dom.tolist()))
    print('[load]', NPZ)
    print(f'[load] domains present: {present}')
    print(f'[load] shapes F{F.shape} V{V.shape} C{C.shape}, n={len(y)}, '
          f'real={int((y==1).sum())}, fake={int((y==0).sum())}')

    # --- t-SNE embeddings (only 3 fits: V reused for by-label & by-domain) ---
    print('[tsne] embedding V ...')
    eV = tsne_embed(V.astype(np.float64))
    print('[tsne] embedding C ...')
    eC = tsne_embed(C.astype(np.float64))
    print('[tsne] embedding F ...')
    eF = tsne_embed(F.astype(np.float64))

    dom_colors = plt.get_cmap('tab10')
    dom_lookup = {dm: i for i, dm in enumerate(domains)}

    # (a) V by real/fake
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ylab = np.where(y == 1, 'real', 'fake')
    col = np.where(y == 1, 0.0, 1.0)
    sc = scatter_embed(ax, eV, col, 'coolwarm', 'V (PDI-ViT CLS) — real/fake')
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color='#3b4cc0', label='real'),
               mpatches.Patch(color='#c0392b', label='fake')]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'tsne_V_bylabel.png'), dpi=150)
    plt.close(fig)

    # (b) V by domain
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    dom_idx = np.array([dom_lookup[x] for x in dom])
    sc = scatter_embed(ax, eV, dom_idx, dom_colors, 'V (PDI-ViT CLS) — by domain')
    add_legend_categorical(ax, sc, [DOMAIN_LABEL[dm] for dm in domains], dom_colors, 'domain')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'tsne_V_bydomain.png'), dpi=150)
    plt.close(fig)

    # (c) C by domain
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = scatter_embed(ax, eC, dom_idx, dom_colors, 'C (CLIP CLS) — by domain')
    add_legend_categorical(ax, sc, [DOMAIN_LABEL[dm] for dm in domains], dom_colors, 'domain')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'tsne_C_bydomain.png'), dpi=150)
    plt.close(fig)

    # (d) F by domain
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = scatter_embed(ax, eF, dom_idx, dom_colors, 'F (classifier input, 1664-d) — by domain')
    add_legend_categorical(ax, sc, [DOMAIN_LABEL[dm] for dm in domains], dom_colors, 'domain')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'tsne_F_bydomain.png'), dpi=150)
    plt.close(fig)
    print('[save] PNGs written')

    # ------------------------------------------------------------------ report
    print('[metric] per-domain AUC ...')
    rows = []
    rows.append('=' * 90)
    rows.append('T-SNE CROSS-DOMAIN REPORT  (M2F2-Det Stage-1 detector)')
    rows.append('=' * 90)
    rows.append(f'npz     : {NPZ}')
    rows.append(f'total n : {len(y)}   real={int((y==1).sum())}  fake={int((y==0).sum())}')
    rows.append(f't-SNE   : PCA->50 then TSNE(perplexity={PERP}, seed={SEED}); t-SNE on V/C/F')
    rows.append('')
    rows.append('Per-domain real/fake separability (linear-probe 5-fold CV AUC on that '
                'domain only; fake=positive)')
    hdr = f'{"domain":<10s} {"n":>5s} {"AUC_V":>8s} {"AUC_C":>8s} {"AUC_F":>8s} {"FisherV":>10s}'
    rows.append(hdr); rows.append('-' * len(hdr))
    auc_rows = {}
    for dm in domains:
        m = dom == dm
        n = int(m.sum())
        aucV = cv_auc(V[m].astype(np.float64), y[m])
        aucC = cv_auc(C[m].astype(np.float64), y[m])
        aucF = cv_auc(F[m].astype(np.float64), y[m])
        fV = fisher_ratio(V[m].astype(np.float64), y[m])
        auc_rows[dm] = (aucV, aucC, aucF)
        rows.append(f'{dm:<10s} {n:>5d} {aucV:>8.4f} {aucC:>8.4f} {aucF:>8.4f} {fV:>10.4f}')
    rows.append('')
    rows.append('Cross-domain linear CKA(V) matrix (domain-pair overlap in ViT feature space)')
    mtx = np.zeros((len(domains), len(domains)))
    hdr = f'{"":>8s}' + ''.join(f'{DOMAIN_LABEL[dm][:8]:>10s}' for dm in domains)
    rows.append(hdr)
    for i, da in enumerate(domains):
        m_a = dom == da
        line = f'{da:>8s}'
        for j, db in enumerate(domains):
            m_b = dom == db
            if i == j:
                cka = 1.0
            else:
                cka = cka_ab(V[m_a].astype(np.float64), V[m_b].astype(np.float64))
            mtx[i, j] = cka
            line += f'{cka:>10.3f}'
        rows.append(line)
    rows.append('')
    # off-diagonal summary
    off = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            off.append((domains[i], domains[j], mtx[i, j]))
    off_sorted = sorted(off, key=lambda t: t[2])
    rows.append('Domain-pair CKA(V) ranking (low -> high overlap):')
    for a, b, v in off_sorted:
        rows.append(f'  {a:>5s} vs {b:<5s}  CKA={v:.3f}')
    rows.append('')
    rows.append('Notes:')
    rows.append('  * AUC_V/C/F: 5-fold CV L2-logistic (features standardized inside fold) on the '
                'domain\'s own sampled frames.')
    rows.append('  * FisherV: LDA-style generalized Fisher ratio (in-sample, shrinkage 1e-3) on V; '
                'larger = better real/fake separation.')
    rows.append('  * CKA: linear CKA of column-centered V between domains; 1=identical subspace.')
    rows.append('  * y convention: real=1, fake=0 (as in dataset txt).')
    rows.append('=' * 90)
    txt = '\n'.join(rows)
    print(txt)
    with open(os.path.join(HERE, 'tsne_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('[save] report ->', os.path.join(HERE, 'tsne_report.txt'))

    print('\n===== SUMMARY =====')
    for dm in domains:
        print(f'  {dm:>5s}: AUC_V={auc_rows[dm][0]:.4f}  AUC_C={auc_rows[dm][1]:.4f}  AUC_F={auc_rows[dm][2]:.4f}')
    np.set_printoptions(precision=3, suppress=True)
    print('CKA(V):')
    print(mtx)


if __name__ == '__main__':
    main()
