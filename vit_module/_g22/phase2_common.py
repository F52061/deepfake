# -*- coding: utf-8 -*-
"""
Phase-2 common: group configuration / training loop / eval chain.

Groups (only difference vs `fix`):
  fix        : FIX-A applied, bridge trainable
  randbridge : FIX-A applied, bridge FROZEN at random init, only `output` trained
  nofix      : FIX-A NOT applied (LayerNorm(1) kept), bridge trainable

Reuses the phase-1 evaluation chain verbatim (g22_common / _g16 layer_feats.npz).
"""

import os

# CPU pinning must precede numpy/torch import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import g22_common as C  # noqa: E402

# ---- fixed spec values ----------------------------------------------------
SEEDS = [42, 43, 44]
# randbridge uses a DIFFERENT random bridge per run (ruling 2026-09-18): sharing
# one bridge would sample only a single draw and understate the control's
# variance -> risk of a false-positive "fix is better".
BRIDGE_INIT_SEEDS = [7777, 7778, 7779]
BRIDGE_INIT_SEED = BRIDGE_INIT_SEEDS[0]   # kept for single-run use
LR_BRIDGE = 5e-4                 # stage-1 design: bridge group lr = args.lr
LR_OUTPUT = 2.5e-4               # stage-1 design: proj group lr = args.lr * 0.5
WEIGHT_DECAY = 1e-4
BATCH_SIZE_ARG = 32              # passed to BalancedBatchSampler
GROUPS = ['fix', 'randbridge', 'nofix']

BRIDGE_PREFIXES = ('linear_vit_1', 'linear_vit_2', 'linear_vit_3',
                   'linear_vit_lst', 'clip_reduction', 'bridge_adapter.',
                   'bridge_adapter_proj.')


def is_bridge_key(k):
    return k.startswith(BRIDGE_PREFIXES)


def bridge_modules(model):
    return [model.clip_reduction, *model.linear_vit_lst,
            *model.bridge_adapter, model.bridge_adapter_proj]


# --------------------------------------------------------------------------
def build_model_variant(group, device, bridge_init_seed=BRIDGE_INIT_SEED,
                        verbose=True):
    """Build the model with the group-specific bridge treatment.

    Returns (model, info_dict).
    """
    info = {'group': group}
    shipped = torch.load(C.SHIPPED_CKPT, map_location='cpu')

    if group == 'randbridge':
        # 1) reproducible random init of the bridge, captured at construction
        torch.manual_seed(bridge_init_seed)
        model = C.build_model(device=device, load_shipped=False, verbose=verbose)
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()
                if is_bridge_key(k)}
        n_snap = len(snap)
        # 2) now load the shipped (trained) checkpoint ...
        m, u = model.load_state_dict(shipped, strict=False)
        assert len(m) == 0 and len(u) == 0, (m[:5], u[:5])
        # 3) ... then overwrite the bridge back to the random snapshot
        with torch.no_grad():
            for k, v in snap.items():
                model.state_dict()[k].copy_(v)
        info['bridge_init_seed'] = bridge_init_seed
        info['randbridge_snapshot_keys'] = n_snap
        if verbose:
            print(f'[randbridge] bridge re-initialised randomly with seed '
                  f'{bridge_init_seed} ({n_snap} tensors), rest of model = shipped ckpt')
    else:
        model = C.build_model(device=device, load_shipped=True, verbose=verbose)
        info['bridge_init_seed'] = None

    # ---- freeze EVERYTHING, then re-enable per spec -----------------------
    for p in model.parameters():
        p.requires_grad = False

    if group in ('fix', 'randbridge'):
        prev, dw, db = C.apply_fix_a(model, verbose=verbose)
        info['fix_a_applied'] = True
        info['fix_a_weight_maxdiff'] = dw
        info['fix_a_bias_maxdiff'] = db
        info['_prev'] = prev
    else:
        info['fix_a_applied'] = False

    # FIX (2026-09-18): apply_fix_a() builds a brand-new nn.Linear, which comes
    # with requires_grad=True by default and therefore ESCAPED the freeze above.
    # It leaked 2 trainable bridge tensors (bridge_adapter_proj.1.weight/.bias).
    # Re-freeze everything after the swap, unconditionally.
    for p in model.parameters():
        p.requires_grad = False

    # ---- trainable sets ---------------------------------------------------
    if group in ('fix', 'nofix'):
        for mod in bridge_modules(model):
            for p in mod.parameters():
                p.requires_grad = True
    # randbridge: bridge stays frozen (otherwise identical to fix)

    for p in model.output.parameters():
        p.requires_grad = True

    model.train()
    return model, info


def make_optimizer(model, group):
    """stage-1 param-group design, with the frozen modules excluded."""
    param_groups = []
    if group in ('fix', 'nofix'):
        brid = [p for mod in bridge_modules(model)
                for p in mod.parameters() if p.requires_grad]
        param_groups.append({'params': brid, 'lr': LR_BRIDGE})
    out = [p for p in model.output.parameters() if p.requires_grad]
    param_groups.append({'params': out, 'lr': LR_OUTPUT})
    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return opt, param_groups


def trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def trainable_state_dict(model):
    """Only the trainable parameters (a few tens of MB, never the full 2 GB)."""
    names = set(trainable_names(model))
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in names}


def save_trainable(model, path, group, seed, tag):
    sd = trainable_state_dict(model)
    torch.save({'group': group, 'seed': seed, 'tag': tag,
                'n_tensors': len(sd), 'state_dict': sd}, path)
    return path, len(sd)


# --------------------------------------------------------------------------
def train_one_epoch(model, optimizer, sampler, device, collate, logger=print,
                    log_every=200, max_batches=None, label_smoothing=0.0,
                    half_hook=None):
    """One pass over BalancedBatchSampler; micro-batch size is measured and
    printed on the first batch.

    half_hook(step, n_total) is called once when the epoch crosses 50%;
    the caller is responsible for switching back to train() mode (we do it here).
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    seen_micro = []
    t0 = time.time()
    t_load = 0.0
    t_compute = 0.0
    nb = 0
    n_total = len(sampler) if hasattr(sampler, '__len__') else None
    half_done = False
    t_hook = 0.0
    for step, batch_data in enumerate(sampler):
        if max_batches is not None and step >= max_batches:
            break
        ta = time.time()
        images, labels = collate(batch_data)
        tb = time.time()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tc = time.time()

        seen_micro.append(int(images.shape[0]))
        if step == 0:
            logger(f'  [MICRO-BATCH] first batch real micro-batch size = '
                   f'{images.shape[0]}   (sampler.effective_batch = '
                   f'{getattr(sampler, "effective_batch", "n/a")}, '
                   f'sampler.per_class = {getattr(sampler, "per_class", "n/a")})')
        t_load += tb - ta
        t_compute += tc - tb
        total_loss += float(loss.item())
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        nb += 1
        if log_every and (step + 1) % log_every == 0:
            el = time.time() - t0
            logger(f'  step {step+1:6d}  loss={total_loss/nb:.4f}  '
                   f'acc={100.*correct/max(total,1):.2f}%  {el:.0f}s  '
                   f'{total/max(el,1e-9):.1f} img/s')

        # --- 50% checkpoint hook (must not perturb our own timing) ---------
        if (half_hook is not None and not half_done and n_total
                and (step + 1) >= n_total // 2):
            half_done = True
            th = time.time()
            half_hook(step + 1, n_total)
            model.train()
            t_hook += time.time() - th

    if str(device).startswith('cuda'):
        torch.cuda.synchronize()      # make the timing honest
    wall = time.time() - t0
    wall_training_only = wall - t_hook
    return {
        'batches': nb, 'images': total,
        'micro_batch_sizes': sorted(set(seen_micro)),
        'micro_batch_median': float(np.median(seen_micro)) if seen_micro else 0,
        'loss': total_loss / max(nb, 1),
        'acc': 100. * correct / max(total, 1),
        'wall_s': wall,
        'wall_training_only_s': wall_training_only,
        'hook_s': t_hook,
        'load_s': t_load, 'compute_s': t_compute,
        'ms_per_img': wall / max(total, 1) * 1000,
        'ms_per_img_training_only': wall_training_only / max(total, 1) * 1000,
        'img_per_s': total / max(wall, 1e-9),
    }


# --------------------------------------------------------------------------
def make_eval_loader(rows, groups_mask=None, device='cpu'):
    _, IMG_TO_TENSOR, _, _ = C.get_pipeline()
    ds = C.RowDataset(rows['paths'], rows['y'], IMG_TO_TENSOR, size=336)
    return ds, torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                           num_workers=0)


@torch.no_grad()
def score_all(model, loader, device, tag='', log_every=30):
    model.eval()
    scores = np.zeros(len(loader.dataset), dtype=np.float64)
    pos = 0
    t0 = time.time()
    for bi, (x, y) in enumerate(loader):
        x = x.to(device)
        o = model(x)
        s = (o[:, 1] - o[:, 0]).float().cpu().numpy()
        scores[pos:pos + len(s)] = s
        pos += len(s)
        if log_every and (bi + 1) % log_every == 0:
            print(f'    [{tag}] {pos}/{len(scores)}  {time.time()-t0:.0f}s', flush=True)
    return scores


def auc_table(rows, scores):
    masks = C.eval_groups(rows)
    res = {}
    for g, m in masks.items():
        res[g] = C.auc_safe(rows['y'][m], scores[m])
    res['mean4dom'] = float(np.mean([res['cd1'], res['cd2'], res['dfdcp'],
                                     res['wild']]))
    return res


RESULTS_TXT = os.path.join(_HERE, 'phase2_results.txt')
HEADER = ('group\tseed\tffpp_test\tcd1\tcd2\tdfdcp\tffiw\twild\t'
          'mean4dom\twall_s\tnotes')


def append_result(line):
    new = not os.path.exists(RESULTS_TXT)
    with open(RESULTS_TXT, 'a', encoding='utf-8') as f:
        if new:
            f.write(HEADER + '\n')
        f.write(line.rstrip('\n') + '\n')
