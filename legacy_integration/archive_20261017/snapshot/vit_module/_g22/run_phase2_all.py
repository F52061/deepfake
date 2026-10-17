# -*- coding: utf-8 -*-
"""
G22 PHASE-2 long run -- 3 groups x 3 seeds = 9 runs, INTERLEAVED.

Order (deliberately interleaved, not grouped -- see coordinator note
2026-09-19: grouping would bind machine thermal state to the group label and
could manufacture the unexplained "fix is 6% slower" artefact):

    (fix,42) (randbridge,42) (nofix,42)
    (fix,43) (randbridge,43) (nofix,43)
    (fix,44) (randbridge,44) (nofix,44)

* appends one line to phase2_results.txt IMMEDIATELY after each run finishes
* ANY run failure -> log traceback, append FAILED line, STOP the whole script
* saves ONLY the trainable-parameter state_dict (never the full 2 GB model)
* GPU: physical card 1 only; aborts if it is occupied

Usage (detached):
  nohup <python> vit_module/_g22/run_phase2_all.py >> vit_module/_g22/phase2_log.txt 2>&1 &
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import sys
import gc
import time
import json
import datetime
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import phase2_common as P  # noqa: E402
import g22_common as C  # noqa: E402

import numpy as np
import torch

DEV = 'cuda:0'          # == physical GPU 1 (CUDA_VISIBLE_DEVICES)
N_EPOCH = 107700

# ---- DRY RUN: exercises the ENTIRE control flow (all 9 interleaved runs,
# ---- both evals, both saves, the results append, the summary) on a tiny
# ---- 180-image epoch + 250-row eval.  Writes to *_dryrun files only.
DRYRUN = os.environ.get('G22_DRYRUN', '0') == '1'
SUF = '_dryrun' if DRYRUN else ''
RESULTS = os.path.join(_HERE, f'phase2_results{SUF}.txt')
TRAIN_TXT = os.path.join(_HERE, '_smoke_train_200.txt') if DRYRUN else C.TRAIN_TXT


def log(m=''):
    print(f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")}] {m}', flush=True)


def append_line(line):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, 'a', encoding='utf-8') as f:
        if new:
            f.write('# group / seed / FFPP_test / cd1 / cd2 / dfdcp / ffiw / '
                    'wild / mean4dom / half_mean4dom / wall\n')
        f.write(line + '\n')
        f.flush()
        os.fsync(f.fileno())


# ===========================================================================
log('=' * 78)
log('G22 PHASE-2 LONG RUN START')
log('=' * 78)
log(f'  pid={os.getpid()}  cwd={os.getcwd()}')
log(f'  order = INTERLEAVED: ' + ' '.join(
    f'({g},{s})' for s in P.SEEDS for g in P.GROUPS))
log(f'  seeds={P.SEEDS}  randbridge bridge seeds={P.BRIDGE_INIT_SEEDS}')
log(f'  batch_size_arg={P.BATCH_SIZE_ARG} (real micro-batch 30)  '
    f'lr bridge={P.LR_BRIDGE} output={P.LR_OUTPUT}  wd={P.WEIGHT_DECAY}')
log(f'  eval = g16/layer_feats.npz: split=="test"(800) + cd1/cd2/dfdcp/ffiw/wild(300 ea)')
log(f'  train txt = {TRAIN_TXT}')
log(f'  DRYRUN = {DRYRUN}')
log(f'  results -> {RESULTS}')

# ------------------------------------------------------------- GPU gate ----
ok, info = C.gate_on_gpu(C.GPU_INDEX, 100)
log(f'[GPU] BEFORE -- {info}')
log(f'[GPU] gate (physical card {C.GPU_INDEX} <= 100 MiB): '
    f'{"PASS" if ok else "FAIL"}')
if not ok:
    log('[ABORT] card occupied -- NOT switching cards. Exiting without running.')
    append_line('ABORT / gpu_occupied / ' + info)
    sys.exit(3)
C.pin_gpu(C.GPU_INDEX)
log(f'[GPU] pinned CUDA_VISIBLE_DEVICES={C.GPU_INDEX}  '
    f'visible={torch.cuda.device_count()}  {torch.cuda.get_device_name(0)}')
log('=' * 78)

# --------------------------------------------------- shared eval machinery --
rows = C.load_eval_rows()
if DRYRUN:
    m_all = C.eval_groups(rows)
    keep = np.zeros(len(rows['y']), dtype=bool)
    ti = np.where(m_all['ffpp_test'])[0]
    keep[ti[rows['y'][ti] == 0][:50]] = True
    keep[ti[rows['y'][ti] == 1][:50]] = True
    for d in ['cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']:
        keep[np.where(m_all[d])[0][:30]] = True
    rows = {k: rows[k][keep] for k in ['paths', 'y', 'domain', 'split']}
masks = C.eval_groups(rows)
_, IMG_TO_TENSOR, _, _ = C.get_pipeline()
eval_ds = C.RowDataset(rows['paths'], rows['y'], IMG_TO_TENSOR, size=336)
eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=32, shuffle=False,
                                          num_workers=0)
log(f'[eval] {len(rows["paths"])} rows ready; '
    f'missing_files={sum(1 for p in rows["paths"] if not os.path.exists(p))}')
log(f'[split==test] y counts = '
    f'{dict(zip(*[x.tolist() for x in np.unique(rows["y"][masks["ffpp_test"]], return_counts=True)]))}')

from vit_module.train_bridge_phase1 import BalancedBatchSampler, balanced_collate

t_start_all = time.time()
run_meta = []

try:
    for si, seed in enumerate(P.SEEDS):
        for grp in P.GROUPS:
            tag = f'{grp}_s{seed}'
            log('')
            log('#' * 78)
            log(f'### RUN  {tag}   (group={grp}, seed={seed})')
            log('#' * 78)
            t_run = time.time()

            # ---- seeds (training randomness) ------------------------------
            torch.manual_seed(seed)
            np.random.seed(seed)
            bseed = P.BRIDGE_INIT_SEEDS[si] if grp == 'randbridge' else 7777

            t_b = time.time()
            model, minfo = P.build_model_variant(grp, DEV, bridge_init_seed=bseed,
                                                 verbose=True)
            t_build = time.time() - t_b
            tr = P.trainable_names(model)
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f'  [build] {t_build:.1f}s  fix_a={minfo["fix_a_applied"]} '
                f'bridge_init_seed={minfo["bridge_init_seed"]}')
            log(f'  [trainable] {len(tr)} tensors / {n_tr/1e6:.4f}M')

            # re-seed AFTER construction so training RNG is per-run deterministic
            torch.manual_seed(seed)
            np.random.seed(seed)

            opt, pg = P.make_optimizer(model, grp)
            log(f'  [optim] ' + ', '.join(f'{len(g["params"])}t@lr={g["lr"]}'
                                          for g in pg))

            sampler = BalancedBatchSampler(TRAIN_TXT, batch_size=P.BATCH_SIZE_ARG)
            log(f'  [sampler] per_class={sampler.per_class} '
                f'effective_batch={sampler.effective_batch} '
                f'batches/epoch={len(sampler)} '
                f'images/epoch={len(sampler)*sampler.effective_batch}')

            half_res = {}

            def half_hook(step, n_total, _model=model, _tag=tag, _seed=seed,
                          _grp=grp, _res=half_res):
                log(f'  [50%] reached at step {step}/{n_total} -- evaluating '
                    f'{len(eval_ds)} images ...')
                sc = P.score_all(_model, eval_loader, DEV, tag=_tag, log_every=40)
                a = P.auc_table(rows, sc)
                _res['auc'] = a
                p, n = P.save_trainable(
                    _model, os.path.join(_HERE, f'phase2_ckpt_{_tag}_50{SUF}.pth'),
                    _grp, _seed, '50')
                log(f'  [50%] ' + '  '.join(
                    f'{k}={a[k]:.4f}' for k in ['ffpp_test', 'cd1', 'cd2',
                                                'dfdcp', 'ffiw', 'wild',
                                                'mean4dom']))
                log(f'  [50%] saved {n} tensors -> {p}')

            st = P.train_one_epoch(model, opt, sampler, DEV, balanced_collate,
                                   logger=log, log_every=100, half_hook=half_hook)
            log(f'  [epoch] batches={st["batches"]} images={st["images"]} '
                f'micro={st["micro_batch_sizes"]} wall={st["wall_s"]:.1f}s '
                f'(hook {st["hook_s"]:.1f}s) {st["ms_per_img"]:.2f} ms/img')
            log(f'  [epoch] final loss={st["loss"]:.4f} train_acc={st["acc"]:.2f}%')

            # ---- 100% eval ----------------------------------------------
            log(f'  [100%] evaluating {len(eval_ds)} images ...')
            sc = P.score_all(model, eval_loader, DEV, tag=tag, log_every=40)
            auc = P.auc_table(rows, sc)
            p100, n100 = P.save_trainable(
                model, os.path.join(_HERE, f'phase2_ckpt_{tag}_100{SUF}.pth'),
                grp, seed, '100')

            wall_h = (time.time() - t_run) / 3600
            half_m4 = half_res.get('auc', {}).get('mean4dom', float('nan'))
            log(f'  [100%] ' + '  '.join(
                f'{k}={auc[k]:.4f}' for k in ['ffpp_test', 'cd1', 'cd2', 'dfdcp',
                                              'ffiw', 'wild', 'mean4dom']))
            log(f'  [DONE] {tag}  wall={wall_h:.3f}h  half_mean4dom={half_m4:.4f}')

            append_line(
                f'{grp} / {seed} / {auc["ffpp_test"]:.4f} / {auc["cd1"]:.4f} / '
                f'{auc["cd2"]:.4f} / {auc["dfdcp"]:.4f} / {auc["ffiw"]:.4f} / '
                f'{auc["wild"]:.4f} / {auc["mean4dom"]:.4f} / {half_m4:.4f} / '
                f'{wall_h:.3f}h')

            run_meta.append({'group': grp, 'seed': seed, 'tag': tag,
                             'auc': auc, 'half': half_res.get('auc'),
                             'wall_h': wall_h, 'train': st,
                             'n_tensors': len(tr)})
            json.dump(run_meta,
                      open(os.path.join(_HERE, f'phase2_runs{SUF}.json'), 'w'),
                      indent=2, default=str)

            del model, opt, sampler
            gc.collect()
            torch.cuda.empty_cache()

except Exception:
    tb = traceback.format_exc()
    log('!!! RUN FAILED -- STOPPING THE WHOLE SCRIPT (no skip-and-continue) !!!')
    log(tb)
    append_line(f'FAILED / see phase2_log.txt / {tb.splitlines()[-1][:200]}')
    sys.exit(1)

# ------------------------------------------------------------- summary ------
tot_h = (time.time() - t_start_all) / 3600
log('')
log('=' * 78)
log(f'ALL 9 RUNS COMPLETE in {tot_h:.2f} h')
log('=' * 78)
for g in P.GROUPS:
    a = [m['auc']['mean4dom'] for m in run_meta if m['group'] == g]
    h = [m['half']['mean4dom'] if m['half'] else float('nan')
         for m in run_meta if m['group'] == g]
    log(f'  {g:11s} mean4dom = {np.mean(a):.4f} +- {np.std(a):.4f}  '
        f'(runs {[round(x,4) for x in a]})   half={[round(x,4) for x in h]}')
log(f'  peak GPU mem = {torch.cuda.max_memory_allocated()/2**20:.0f} MiB')
log(f'  [GPU] AFTER -- ' + '; '.join(f'gpu{i}={u}MiB' for i, u, t, ut in C.query_gpus()))
append_line(f'# ALL DONE total={tot_h:.2f}h')
