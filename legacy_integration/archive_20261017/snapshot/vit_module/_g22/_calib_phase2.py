# -*- coding: utf-8 -*-
"""
Phase-2 long-throughput calibration (authorised 2026-09-18).

Runs ONE full real training epoch over 2100 images (420/class) for EACH group,
using the exact production path (BalancedBatchSampler -> balanced_collate ->
fwd -> CE -> bwd -> clip_grad_norm_ -> AdamW.step), with torch.cuda.synchronize()
so the timing is honest.  No half-epoch hook (we are timing, not training).

Outputs the definitive ms/img per group and the extrapolated 9-run wall time
INCLUDING the new half-epoch + end-of-run evaluations.

NO long run is started here.
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

LOG = os.path.join(_HERE, 'phase2_calib_log.txt')
_lines = []


def log(m=''):
    print(m, flush=True)
    _lines.append(str(m))


ok, info = C.gate_on_gpu(C.GPU_INDEX, 100)
log('=' * 78)
log(f'[GPU] BEFORE CALIB -- {info}')
log(f'[GPU] gate (physical card {C.GPU_INDEX} <= 100 MiB): {"PASS" if ok else "FAIL"}')
if not ok:
    log('[ABORT] card occupied -- NOT switching cards. Reporting.')
    open(LOG, 'w', encoding='utf-8').write('\n'.join(_lines) + '\n')
    sys.exit(3)
C.pin_gpu(C.GPU_INDEX)
log('=' * 78)
DEV = 'cuda:0'

# ------------------------------------------------- calibration train list ---
N_PER_CLASS = 420                         # 420//6 = 70 batches x 30 = 2100 imgs
CAL_TXT = os.path.join(_HERE, '_calib_train_2100.txt')
raw = []
with open(C.TRAIN_TXT, 'r') as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            raw.append((p[0], int(p[1])))
CLASSES = ['original_sequences', 'Deepfakes', 'Face2Face', 'FaceSwap',
           'NeuralTextures']
sel = []
for cname in CLASSES:
    got = [(p, l) for p, l in raw if cname in p and os.path.exists(p)][:N_PER_CLASS]
    log(f'  calib pool {cname:20s} {len(got)}')
    sel += got
with open(CAL_TXT, 'w') as f:
    for p, l in sel:
        f.write(f'{p} {l}\n')

from vit_module.train_bridge_phase1 import BalancedBatchSampler, balanced_collate
probe = BalancedBatchSampler(CAL_TXT, batch_size=P.BATCH_SIZE_ARG)
log(f'[calib] --batch-size={P.BATCH_SIZE_ARG} -> per_class={probe.per_class} '
    f'effective_batch={probe.effective_batch} batches={len(probe)} '
    f'=> {len(probe)*probe.effective_batch} images per calibration epoch')

# ------------------------------------------------------------ per group -----
res = {}
for grp in P.GROUPS:
    log('')
    log('=' * 78)
    log(f'CALIB GROUP = {grp}')
    log('=' * 78)
    bseed = P.BRIDGE_INIT_SEED if grp == 'randbridge' else None
    t_b = time.time()
    model, info = P.build_model_variant(grp, DEV, bridge_init_seed=bseed or 7777,
                                        verbose=False)
    t_build = time.time() - t_b
    tr = P.trainable_names(model)
    log(f'  [build] {t_build:.1f}s  fix_a={info["fix_a_applied"]} '
        f'bridge_seed={info["bridge_init_seed"]}  trainable={len(tr)} tensors')
    opt, pg = P.make_optimizer(model, grp)
    log(f'  [optim] ' + ', '.join(f'{len(g["params"])}t@lr={g["lr"]}' for g in pg))

    sampler = BalancedBatchSampler(CAL_TXT, batch_size=P.BATCH_SIZE_ARG)
    st = P.train_one_epoch(model, opt, sampler, DEV, balanced_collate,
                           logger=log, log_every=50)
    st['build_s'] = t_build
    res[grp] = st
    log(f'  [CALIB] {grp}: {st["images"]} imgs, {st["batches"]} batches, '
        f'wall={st["wall_s"]:.2f}s (train-only {st["wall_training_only_s"]:.2f}s)')
    log(f'  [CALIB] {grp}: micro={st["micro_batch_sizes"]}  '
        f'{st["ms_per_img"]:.2f} ms/img   {st["img_per_s"]:.2f} img/s')
    log(f'  [CALIB] {grp}: GPU peak so far '
        f'{torch.cuda.max_memory_allocated()/2**20:.0f} MiB')
    del model, opt
    torch.cuda.empty_cache()

# ------------------------------------------------------------ ETA -----------
log('')
log('=' * 78)
log('ETA EXTRAPOLATION')
log('=' * 78)
N_EPOCH = 107700
EVAL_IMG = 4500
# measured in phase-1 (same eval chain, batch 32, num_workers=0): 4500 imgs took
# 334.4 s (r0) and 304.0 s (fixed) -> use the slower 334.4 s = 74.3 ms/img
MS_EVAL = 74.3
eval_h = EVAL_IMG * MS_EVAL / 1000 / 3600

rows_out = {}
per_run = {}
for grp, st in res.items():
    ms = st['ms_per_img']
    epoch_h = N_EPOCH * ms / 1000 / 3600
    # per run: epoch + 50% eval + 100% eval + build + 2 small checkpoint saves
    run_h = epoch_h + 2 * eval_h + st['build_s'] / 3600 + 2 * 5 / 3600
    per_run[grp] = run_h
    rows_out[grp] = (ms, epoch_h, run_h)
    log(f'  {grp:11s} {ms:6.2f} ms/img -> epoch {epoch_h:.3f} h -> '
        f'per run (epoch + 2 evals + build + saves) {run_h:.3f} h')

mean_ms = float(np.mean([v[0] for v in rows_out.values()]))
mean_run = float(np.mean(list(per_run.values())))
log('')
log(f'  eval cost: {EVAL_IMG} imgs @ {MS_EVAL} ms/img = {eval_h*60:.1f} min each, '
    f'2 per run (50% + 100%)')
log(f'  MEAN over groups: {mean_ms:.2f} ms/img, {mean_run:.3f} h/run')
log(f'  >>> 9 runs  = {9*mean_run:.2f} h')
log(f'  >>> worst case (max ms/img) = '
    f'{9*max(v[2] for v in rows_out.values()):.2f} h')
log(f'  >>> best case  (min ms/img) = '
    f'{9*min(v[2] for v in rows_out.values()):.2f} h')
log(f'  user-approved budget = 21.2 h -> '
    f'{"FITS" if 9*mean_run <= 21.2 else "EXCEEDS by %.2f h" % (9*mean_run-21.2)}')
log('')
log(f'  epoch share  = {mean_run and (np.mean([v[1] for v in rows_out.values()])/mean_run*100):.1f}%')
log(f'  GPU peak mem = {torch.cuda.max_memory_allocated()/2**20:.0f} MiB')
log(f'  [GPU] AFTER CALIB -- ' +
    '; '.join(f'gpu{i}={u}MiB' for i, u, t, ut in C.query_gpus()))

import json
json.dump({'ms_per_img': {k: v[0] for k, v in rows_out.items()},
           'epoch_h': {k: v[1] for k, v in rows_out.items()},
           'run_h': {k: v[2] for k, v in rows_out.items()},
           'mean_ms': mean_ms, 'mean_run_h': mean_run,
           'total_9runs_h': 9 * mean_run,
           'eval_h_each': eval_h,
           'total_9runs_worst_h': 9 * max(v[2] for v in rows_out.values()),
           'total_9runs_best_h': 9 * min(v[2] for v in rows_out.values()),
           'calib_images': int(res[P.GROUPS[0]]['images']),
           'micro_batch': int(res[P.GROUPS[0]]['micro_batch_median']),
           },
          open(os.path.join(_HERE, 'phase2_calib.json'), 'w'), indent=2)

open(LOG, 'w', encoding='utf-8').write('\n'.join(_lines) + '\n')
print('[done] calib log ->', LOG)
