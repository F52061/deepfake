# -*- coding: utf-8 -*-
"""Recompute t-SNE embeddings (V/C/F) from feats_multi.npz, save to tsne_emb.npz,
and print cluster-structure metrics so the PNG patterns can be summarised
quantitatively (silhouette of domain/class clusters + centroid separation)."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['JOBLIB_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'feats_multi.npz')
OUT = os.path.join(HERE, 'tsne_emb.npz')
SEED = 0
DOMAIN_ORDER = ['ffpp', 'cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']


def tsne_embed(X):
    X50 = PCA(n_components=50, random_state=SEED).fit_transform(X.astype(np.float64))
    return TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca',
                learning_rate='auto', max_iter=1000).fit_transform(X50)


def centroid_table(emb, labels):
    out = {}
    for lab in np.unique(labels):
        out[lab] = emb[labels == lab].mean(0)
    return out


def mean_centroid_dist(ct):
    ks = list(ct.keys())
    ds = [np.linalg.norm(ct[ks[i]] - ct[ks[j]])
          for i in range(len(ks)) for j in range(i + 1, len(ks))]
    return float(np.mean(ds)), float(np.max(ds))


def main():
    d = np.load(NPZ, allow_pickle=True)
    V = d['V']; C = d['C']; F = d['F']
    y = d['y'].astype(int)
    dom = d['domain']
    domains = [dm for dm in DOMAIN_ORDER if dm in set(dom.tolist())]

    ev = tsne_embed(V)
    ec = tsne_embed(C)
    ef = tsne_embed(F)
    np.savez_compressed(OUT, eV=ev, eC=ec, eF=ef, y=y, domain=dom)

    ylab = (y == 1).astype(int)  # 1=real,0=fake
    print(f'{"embed":>5s} {"sil_domain":>10s} {"sil_real/fake":>13s} '
          f'{"meanCentDom":>12s} {"maxCentDom":>11s}')
    for name, e in [('V', ev), ('C', ec), ('F', ef)]:
        sil_d = silhouette_score(e, dom, sample_size=2000, random_state=SEED)
        sil_c = silhouette_score(e, ylab, sample_size=2000, random_state=SEED)
        ct = centroid_table(e, dom)
        mc, mx = mean_centroid_dist(ct)
        print(f'{name:>5s} {sil_d:>10.4f} {sil_c:>13.4f} {mc:>12.4f} {mx:>11.4f}')

    # also per-domain real/fake silhouette in V embedding (domain-local separation)
    print('\nPer-domain real/fake silhouette in V t-SNE:')
    for dm in domains:
        m = dom == dm
        if len(np.unique(ylab[m])) > 1:
            s = silhouette_score(ev[m], ylab[m])
        else:
            s = float('nan')
        print(f'  {dm:>5s}: {s:.4f}')
    print('\n[emb saved]', OUT)


if __name__ == '__main__':
    main()
