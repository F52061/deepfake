# -*- coding: utf-8 -*-
"""
Phase-2 STEP 0 smoke test v2 (minutes-level) -- after the freeze fix.

Checks:
  1. GPU gate on physical card 1
  2. all three groups build + train a (200-image) epoch + produce AUCs
  3. REAL micro-batch size printed on the first batch (expect 30 -> ruling A)
  4. ms/img roughly compatible with phase-1 (long calibration follows)
  5. FREEZE FIX: randbridge trainable tensors == ONLY `output` (2 tensors)
  6. full 800-row split=="test" set has BOTH classes
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import phase2_common as P  # noqa: E402
import g22_common as C  # noqa: E402

import numpy as np
import torch

LOG = os.path.join(_HERE, 'phase2_smoke_log.txt')
_lines = []


def log(m=''):
    print(m, flush=True)
    _lines.append(str(m))


ok, info = C.gate_on_gpu(C.GPU_INDEX, 100)
log('=' * 78)
log(f'[GPU] BEFORE SMOKE -- {info}')
log(f'[GPU] gate (physical card {C.GPU_INDEX} <= 100 MiB): {"PASS" if ok else "FAIL"}')
if not ok:
    log('[ABORT] card occupied -- NOT switching cards. Reporting.')
    open(LOG, 'w', encoding='utf-8').write('\n'.join(_lines) + '\n')
    sys.exit(3)
C.pin_gpu(C.GPU_INDEX)
log('=' * 78)
DEV = 'cuda:0'

# ---------------------------------------------------------- eval rows -------
rows = C.load_eval_rows()
masks = C.eval_groups(rows)

# explicit check demanded by the coordinator
ti = np.where(masks['ffpp_test'])[0]
cnt = dict(zip(*[x.tolist() for x in np.unique(rows['y'][ti], return_counts=True)]))
log(f'[CHECK] FULL split=="test" set: n={len(ti)}  y counts={cnt}  '
    f'-> both classes? {len(np.unique(rows["y"][ti])) == 2}')
assert len(np.unique(rows['y'][ti])) == 2, 'test set is single-class!'

# subset: 100 fake (y=0) + 100 real (y=1) from test, + 100 each cross domain
sub = np.zeros(len(rows['y']), dtype=bool)
i0 = ti[rows['y'][ti] == 0][:100]
i1 = ti[rows['y'][ti] == 1][:100]
sub[i0] = True
sub[i1] = True
for d in ['cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']:
    sub[np.where(masks[d])[0][:100]] = True
rows_sub = {k: rows[k][sub] for k in ['paths', 'y', 'domain', 'split']}
log(f'[smoke] eval subset = {int(sub.sum())} rows '
    f'(100 fake + 100 real from test + 100 x 5 cross domains); '
    f'fixes the phase-1 smoke nan')

_, IMG_TO_TENSOR, _, _ = C.get_pipeline()
ds = C.RowDataset(rows_sub['paths'], rows_sub['y'], IMG_TO_TENSOR, size=336)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                     num_workers=0)

# ---------------------------------------------------------- smoke train -----
from vit_module.train_bridge_phase1 import BalancedBatchSampler, balanced_collate

SMOKE_TXT = os.path.join(_HERE, '_smoke_train_200.txt')
summary = {}
for grp in P.GROUPS:
    log('')
    log('=' * 78)
    log(f'SMOKE GROUP = {grp}')
    log('=' * 78)
    bseed = P.BRIDGE_INIT_SEED if grp == 'randbridge' else None
    t_build = time.time()
    model, info = P.build_model_variant(grp, DEV, bridge_init_seed=bseed or 7777,
                                        verbose=False)
    log(f'  [build] {time.time()-t_build:.1f}s  fix_a={info["fix_a_applied"]} '
        f'bridge_init_seed={info["bridge_init_seed"]}')

    tr = P.trainable_names(model)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nb_t = sum(1 for n in tr if P.is_bridge_key(n))
    no_t = sum(1 for n in tr if n.startswith('output'))
    other = [n for n in tr if not P.is_bridge_key(n) and not n.startswith('output')]
    log(f'  [trainable] {len(tr)} tensors / {ntr/1e6:.4f}M params '
        f'(bridge={nb_t}, output={no_t}, other={len(other)})')
    log(f'  [trainable] NAMES: {tr}')
    if other:
        log(f'  [trainable] *** UNEXPECTED NON-BRIDGE/NON-OUTPUT: {other}')
    if grp == 'randbridge':
        verdict = 'PASS' if (nb_t == 0 and no_t == 2 and len(tr) == 2) else 'FAIL'
        log(f'  [FREEZE-FIX CHECK] randbridge trainable must be ONLY output '
            f'(2 tensors) -> {verdict}')

    opt, pg = P.make_optimizer(model, grp)
    log(f'  [optim] ' + ', '.join(f'{len(g["params"])}t@lr={g["lr"]}' for g in pg))

    sampler = BalancedBatchSampler(SMOKE_TXT, batch_size=P.BATCH_SIZE_ARG)
    log(f'  [sampler] --batch-size={P.BATCH_SIZE_ARG} -> '
        f'per_class={sampler.per_class} effective_batch={sampler.effective_batch} '
        f'batches={len(sampler)}')

    st = P.train_one_epoch(model, opt, sampler, DEV, balanced_collate,
                           logger=log, log_every=0)
    log(f'  [epoch] batches={st["batches"]} images={st["images"]} '
        f'micro={st["micro_batch_sizes"]} wall={st["wall_s"]:.2f}s '
        f'{st["ms_per_img"]:.1f} ms/img  loss={st["loss"]:.4f} acc={st["acc"]:.2f}%')

    sc = P.score_all(model, loader, DEV, tag=grp, log_every=0)
    auc = P.auc_table(rows_sub, sc)
    log(f'  [eval-subset] ' + '  '.join(
        f'{k}={auc[k]:.4f}' for k in ['ffpp_test', 'cd1', 'cd2', 'dfdcp',
                                      'ffiw', 'wild', 'mean4dom']))
    summary[grp] = {'st': st, 'auc': auc, 'n_train': len(tr),
                    'bridge_t': nb_t, 'out_t': no_t, 'other': other}
    del model, opt
    torch.cuda.empty_cache()

# ---------------------------------------------------------- verdict ---------
log('')
log('=' * 78)
log('SMOKE v2 VERDICT')
log('=' * 78)
allms = set()
for v in summary.values():
    allms |= set(v['st']['micro_batch_sizes'])
log(f'  real micro-batch sizes = {sorted(allms)}  (ruling A accepts 30; '
    f'user hard constraint batch<=32 satisfied: {max(allms) <= 32})')
for g, v in summary.items():
    log(f'  {g:11s} trainable_tensors={v["n_train"]:3d} '
        f'(bridge={v["bridge_t"]}, output={v["out_t"]}) '
        f'loss={v["st"]["loss"]:.4f} mean4dom(sub)={v["auc"]["mean4dom"]:.4f}')
log(f'  randbridge freeze-fix: bridge tensors = '
    f'{summary["randbridge"]["bridge_t"]} (must be 0) -> '
    f'{"PASS" if summary["randbridge"]["bridge_t"] == 0 else "FAIL"}')
log(f'  fix vs nofix bridge tensor counts equal (must be): '
    f'{summary["fix"]["bridge_t"]} vs {summary["nofix"]["bridge_t"]} -> '
    f'{"PASS" if summary["fix"]["bridge_t"] == summary["nofix"]["bridge_t"] else "FAIL"}')
log(f'  GPU peak mem = {torch.cuda.max_memory_allocated()/2**20:.0f} MiB')
log(f'  [GPU] AFTER SMOKE -- ' +
    '; '.join(f'gpu{i}={u}MiB' for i, u, t, ut in C.query_gpus()))

open(LOG, 'w', encoding='utf-8').write('\n'.join(_lines) + '\n')
print('[done] smoke log ->', LOG)
