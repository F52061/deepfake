# -*- coding: utf-8 -*-
"""
G22 -- Phase-1 verification gate for ViT_M2F2Det_Bridge.

Task 2 : prove the FIX-A hypothesis (dead bridge) + prove the fix revives grads
Task 3 : R0 baseline AUC with the shipped checkpoint (unmodified vs fixed)
Task 4 : real throughput  -> hours/epoch extrapolation for 107,700 images

Usage (project root):
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g22/run_g22.py \
      > vit_module/_g22/run_log_g22.txt 2>&1
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import g22_common as C  # noqa: E402  (sets thread env vars before numpy/torch)

# ---------------------------------------------------------------- GPU gate --
_gpu_ok, _gpu_info = C.gate_on_gpu(C.GPU_INDEX, max_used_mib=100)
print('=' * 78)
print(f'[GPU] BEFORE RUN -- {_gpu_info}')
print(f'[GPU] gate (physical index {C.GPU_INDEX} must be <=100 MiB): {"PASS" if _gpu_ok else "FAIL"}')
if not _gpu_ok:
    print('[GPU] ABORT -- card occupied, refusing to switch cards. Reported above.')
    sys.exit(3)
C.pin_gpu(C.GPU_INDEX)
print('=' * 78)

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

DEV = 'cuda:0'   # == physical GPU 1 (CUDA_VISIBLE_DEVICES)
print(f'[GPU] torch sees {torch.cuda.device_count()} device(s): '
      f'{torch.cuda.get_device_name(0)}  (physical index pinned to {C.GPU_INDEX})')

OUT_JSON = os.path.join(_HERE, 'phase1_stats.json')
OUT_NPZ = os.path.join(_HERE, 'phase1_stats.npz')
REPORT = os.path.join(_HERE, 'phase1_report.txt')

S = {}          # machine-readable results
lines = []      # report lines


def log(msg=''):
    print(msg, flush=True)
    lines.append(str(msg))


def mem_mib():
    return torch.cuda.memory_allocated() / 2 ** 20, torch.cuda.max_memory_allocated() / 2 ** 20


# ===========================================================================
# Setup: pipeline + model
# ===========================================================================
_FFPPDataset, IMG_TO_TENSOR, CLIP_MEAN, CLIP_STD = C.get_pipeline()
log('=' * 78)
log('G22 PHASE-1 VERIFICATION')
log('=' * 78)
log(f'[preprocess] EXACTLY the training-script pipeline:')
log(f'  cv2.imread(IMREAD_COLOR) -> cvtColor BGR2RGB -> cv2.resize((336,336))')
log(f'  -> albumentations Compose([Normalize(mean={CLIP_MEAN}, std={CLIP_STD}), ToTensorV2()])')
log(f'  (this is FFPPDataset.__getitem__ / train_bridge_phase1.IMG_TO_TENSOR verbatim)')
log(f'[preprocess] ViT branch then gets: denormalize->[0,1] -> bilinear 224 -> (x-0.5)/0.5  '
    f'(model._preprocess_for_vit)')

t0 = time.time()
model = C.build_model(device=DEV, verbose=True)
log(f'[build] ViT ckpt={C.VIT_CKPT}')
log(f'[build] CLIP={C.CLIP_LOCAL}')
log(f'[build] shipped={C.SHIPPED_CKPT}')
log(f'[build] wall={time.time()-t0:.1f}s  gpu_mem={mem_mib()[0]:.0f} MiB')
model.eval()

wrap = model.bridge_adapter_proj
log(f'[model] bridge_adapter_proj.bridge_adapter_proj = {wrap.bridge_adapter_proj}')
_ln = wrap.bridge_adapter_proj[2]
log(f'[model] LayerNorm(1) eps={_ln.eps} weight={_ln.weight.detach().cpu().numpy().tolist()} '
    f'bias={_ln.bias.detach().cpu().numpy().tolist()}')
S['old_ln_weight'] = _ln.weight.detach().cpu().numpy().tolist()
S['old_ln_bias'] = _ln.bias.detach().cpu().numpy().tolist()


# ===========================================================================
# Helper: one forward + backward on a fixed batch, record grad norms
# ===========================================================================
def first_param(mod):
    for n, p in mod.named_parameters():
        return n, p
    return None, None


def probe_list(m):
    n0, p0 = first_param(m.bridge_adapter[0])
    return [
        ('linear_vit_lst[0].weight', m.linear_vit_lst[0].weight),
        ('linear_vit_lst[2].weight', m.linear_vit_lst[2].weight),
        ('clip_reduction.weight', m.clip_reduction.weight),
        (f'bridge_adapter[0].{n0}', p0),
        ('bridge_adapter_proj.bridge_adapter_reduction[0].weight',
         m.bridge_adapter_proj.bridge_adapter_reduction[0].weight),
        ('clip_text_alpha', m.clip_text_alpha),
        # --- extra diagnostics (not required, but cheap and informative) ---
        ('bridge_adapter_proj.bridge_adapter_proj.1.weight(pre-LN Linear)',
         m.bridge_adapter_proj.bridge_adapter_proj[1].weight),
        ('bridge_adapter_proj.bridge_adapter_reduction.1.weight(LN tail)',
         m.bridge_adapter_proj.bridge_adapter_reduction[1].weight),
        ('output.weight', m.output.weight),
        ('deepfake_proj[0].weight', m.deepfake_proj[0].weight),
        ('vision_proj[0].weight', m.vision_proj[0].weight),
        ('text_proj[0].weight', m.text_proj[0].weight),
        ('clip_vision_alpha', m.clip_vision_alpha),
        ('clip_text_encoder.prompt_tokens', m.clip_text_encoder.prompt_tokens),
        ('vit.blocks[0].attn.qkv.weight (frozen)',
         m.vit.blocks[0].attn.qkv.weight),
    ]


def run_fwd_bwd(m, imgs, targets):
    m.train()
    m.zero_grad(set_to_none=True)
    out = m(imgs)
    loss = F.cross_entropy(out, targets)
    loss.backward()
    return float(loss.item()), out.detach()


# ------------------------------------------------------------------ batch ---
raw = []
with open(C.TRAIN_TXT, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            raw.append((parts[0], int(parts[1])))
real = [(p, l) for p, l in raw if l == 1 and os.path.exists(p)]
fake = [(p, l) for p, l in raw if l == 0 and os.path.exists(p)]
log(f'[data] train split {C.TRAIN_TXT}: {len(raw)} lines, '
    f'real(exist)={len(real)} fake(exist)={len(fake)}')

B8 = [p for p, _ in real[:8]]
imgs8 = torch.stack([C.load_image_tensor(p, IMG_TO_TENSOR) for p in B8]).to(DEV)
# label convention: y=1 REAL, y=0 FAKE ; model out index0=real, index1=fake
tgt8 = torch.zeros(len(B8), dtype=torch.long, device=DEV)   # 8 REAL images
log(f'[task2] batch = 8 REAL FF++ training images, target=0 (real) for all, '
    f'shape={tuple(imgs8.shape)}')

log('')
log('=' * 78)
log('TASK 2 -- prove the bug, prove the fix')
log('=' * 78)

# ---------------------------------------------------- 2(a) unmodified ------
loss_a, out_a = run_fwd_bwd(model, imgs8, tgt8)
log(f'[2a] UNMODIFIED model: loss={loss_a:.6f}')
grads_a = {}
for name, p in probe_list(model):
    g = C.grad_norm(p)
    grads_a[name] = g
    log(f'      grad_norm  {name:58s} = {g}')

# ---------------------------------------------------- 2(c) unmodified ------
def capture_embed(m, paths):
    cap = {}
    h = m.bridge_adapter_proj.register_forward_hook(
        lambda mod, i, o: cap.__setitem__('e', o.detach()))
    m.eval()
    with torch.no_grad():
        xs = torch.stack([C.load_image_tensor(p, IMG_TO_TENSOR) for p in paths]).to(DEV)
        m(xs)
    h.remove()
    emb = cap['e'] * m.clip_text_alpha.detach()
    return emb.float().cpu().numpy()


FOUR = [p for p, _ in real[:2]] + [p for p, _ in fake[:2]]
emb_before = capture_embed(model, FOUR)


def row_spread(e):
    n = e.shape[0]
    mx = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            mx = max(mx, float(np.abs(e[i] - e[j]).max()))
    return mx


spread_before = row_spread(emb_before)
log('')
log(f'[2c] UNMODIFIED clip_adapt_embed (after clip_text_alpha), 4 images:')
for i in range(4):
    log(f'      row{i} [0:8] = {np.round(emb_before[i][:8], 8).tolist()}  '
        f'norm={np.linalg.norm(emb_before[i]):.8f}')
log(f'[2c] UNMODIFIED max |row_i - row_j| over all 6 pairs = {spread_before:.3e}  '
    f'(bitwise identical? {spread_before == 0.0})')
S['embed_before'] = emb_before
S['spread_before'] = spread_before

# =================================================== TASK 3 (R0) ============
log('')
log('=' * 78)
log('TASK 3 -- R0 baseline AUC (unmodified shipped checkpoint)')
log('=' * 78)

rows = C.load_eval_rows()
groups = C.eval_groups(rows)
missing_paths = [p for p in rows['paths'] if not os.path.exists(p)]
log(f'[task3] rows={len(rows["paths"])}  missing_image_files={len(missing_paths)}')
if missing_paths:
    log('  MISSING (first 20):')
    for p in missing_paths[:20]:
        log('    ' + p)
S['n_rows'] = int(len(rows['paths']))
S['n_missing_paths'] = int(len(missing_paths))

ds = C.RowDataset(rows['paths'], rows['y'], IMG_TO_TENSOR, size=336)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)


@torch.no_grad()
def score_all(m, tag=''):
    m.eval()
    scores = np.zeros(len(ds), dtype=np.float64)
    t0 = time.time()
    pos = 0
    for bi, (x, y) in enumerate(loader):
        x = x.to(DEV, non_blocking=False)
        o = m(x)
        s = (o[:, 1] - o[:, 0]).float().cpu().numpy()
        scores[pos:pos + len(s)] = s
        pos += len(s)
        if (bi + 1) % 30 == 0:
            el = time.time() - t0
            print(f'    [{tag}] {pos}/{len(ds)}  {el:.1f}s  '
                  f'({pos/max(el,1e-9):.1f} img/s)', flush=True)
    log(f'[task3] {tag}: scored {pos} images in {time.time()-t0:.1f}s')
    return scores


def auc_table(scores, tag):
    res = {}
    for g, mask in groups.items():
        res[g] = C.auc_safe(rows['y'][mask], scores[mask])
    res['mean4dom'] = float(np.mean([res['cd1'], res['cd2'], res['dfdcp'], res['wild']]))
    log(f'[task3] {tag}:')
    for g in ['ffpp_test', 'cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']:
        log(f'      AUC {g:11s} = {res[g]:.4f}')
    log(f'      AUC {"mean4dom":11s} = {res["mean4dom"]:.4f}   '
        f'(cd1,cd2,dfdcp,wild -- ffiw EXCLUDED)')
    return res


scores_r0 = score_all(model, 'R0-unmodified')
auc_r0 = auc_table(scores_r0, 'R0 UNMODIFIED shipped checkpoint')
S['auc_r0'] = auc_r0
S['scores_r0'] = scores_r0
S['y'] = rows['y']
S['domain'] = rows['domain']
S['split'] = rows['split']

mem_now = mem_mib()
log(f'[gpu] after R0 eval: allocated={mem_now[0]:.0f} MiB  peak={mem_now[1]:.0f} MiB')

# =================================================== APPLY FIX-A ============
log('')
log('=' * 78)
log('APPLY FIX-A  (monkeypatch: (View, Linear, LayerNorm(1)) -> (View, Linear))')
log('=' * 78)
prev, dw, db = C.apply_fix_a(model, verbose=True)
S['fixa_linear_weight_maxdiff'] = float(dw)
S['fixa_linear_bias_maxdiff'] = float(db)
log(f'[2d] replacement Linear weight/bias vs shipped checkpoint copy: '
    f'max|diff| = {dw:.3e} / {db:.3e}   (exact? {dw == 0.0 and db == 0.0})')
log(f'[model] new bridge_adapter_proj.bridge_adapter_proj = '
    f'{model.bridge_adapter_proj.bridge_adapter_proj}')

# ---------------------------------------------------- 2(b) fixed -----------
loss_b, out_b = run_fwd_bwd(model, imgs8, tgt8)
log(f'[2b] FIXED model: loss={loss_b:.6f}  (unmodified loss was {loss_a:.6f})')
grads_b = {}
for name, p in probe_list(model):
    g = C.grad_norm(p)
    grads_b[name] = g
    log(f'      grad_norm  {name:58s} = {g}')
S['grads_before'] = grads_a
S['grads_after'] = grads_b
S['loss_before'] = loss_a
S['loss_after'] = loss_b

# ---------------------------------------------------- 2(c) fixed -----------
emb_after = capture_embed(model, FOUR)
spread_after = row_spread(emb_after)
log('')
log(f'[2c] FIXED clip_adapt_embed (after clip_text_alpha), same 4 images:')
for i in range(4):
    log(f'      row{i} [0:8] = {np.round(emb_after[i][:8], 8).tolist()}  '
        f'norm={np.linalg.norm(emb_after[i]):.8f}')
log(f'[2c] FIXED max |row_i - row_j| over all 6 pairs = {spread_after:.6e}  '
    f'(rows differ? {spread_after > 0})')
S['embed_after'] = emb_after
S['spread_after'] = spread_after

# ---------------------------------------------------- verdict --------------
REQ = ['linear_vit_lst[0].weight', 'linear_vit_lst[2].weight',
       'clip_reduction.weight',
       'bridge_adapter_proj.bridge_adapter_reduction[0].weight']
all_zero_before = all(grads_a[k] == 0.0 for k in REQ)
alpha_nonzero_before = grads_a['clip_text_alpha'] not in (None, 'None') and grads_a['clip_text_alpha'] > 0
all_nonzero_after = all(grads_b[k] not in (None, 'None') and grads_b[k] > 0 for k in REQ)
v_a = all_zero_before and alpha_nonzero_before
v_b = all_nonzero_after
v_c = (spread_before == 0.0) and (spread_after > 0)
v_d = (dw == 0.0 and db == 0.0)
log('')
log(f'[GATE] 2a all-zero-except-alpha : {"PASS" if v_a else "FAIL"}')
log(f'[GATE] 2b all-nonzero           : {"PASS" if v_b else "FAIL"}')
log(f'[GATE] 2c identical -> differ   : {"PASS" if v_c else "FAIL"}  '
    f'({spread_before:.3e} -> {spread_after:.6e})')
log(f'[GATE] 2d weight copy exact     : {"PASS" if v_d else "FAIL"}')
S['gate_2a'] = bool(v_a)
S['gate_2b'] = bool(v_b)
S['gate_2c'] = bool(v_c)
S['gate_2d'] = bool(v_d)
S['gate_task2'] = bool(v_a and v_b and v_c and v_d)

# =================================================== TASK 3 fixed ==========
log('')
log('=' * 78)
log('TASK 3b -- same shipped weights, FIX-A applied, no training')
log('=' * 78)
scores_fix = score_all(model, 'R0-fixed')
auc_fix = auc_table(scores_fix, 'R0 FIXED (no training)')
S['auc_fixed'] = auc_fix
S['scores_fixed'] = scores_fix

# mean feature-level agreement between the two score vectors
log(f'[task3] mean|score_fixed - score_unmodified| = '
    f'{np.abs(scores_fix - scores_r0).mean():.4f}   '
    f'max={np.abs(scores_fix - scores_r0).max():.4f}')

mem_now = mem_mib()
log(f'[gpu] after fixed eval: allocated={mem_now[0]:.0f} MiB  peak={mem_now[1]:.0f} MiB')

# =================================================== TASK 4 ================
log('')
log('=' * 78)
log('TASK 4 -- throughput (forward+backward, batch 32, ~200 images)')
log('=' * 78)

N_TIME = 224
paths_t = [p for p, _ in raw[:N_TIME]]
labels_t = [l for _, l in raw[:N_TIME]]
t0 = time.time()
imgs_t = torch.stack([C.load_image_tensor(p, IMG_TO_TENSOR) for p in paths_t]).to(DEV)
t_load = time.time() - t0
log(f'[task4] loaded+preprocessed {N_TIME} images (+H2D) in {t_load:.2f}s '
    f'= {t_load/N_TIME*1000:.1f} ms/img')

BS = 32
nb = 0
t_fwd = 0.0
t_bwd = 0.0
t0 = time.time()
model.train()
for i in range(0, N_TIME, BS):
    x = imgs_t[i:i + BS]
    # targets: y=1 real -> 0 ; y=0 fake -> 1  (model index0=real, index1=fake)
    y = torch.tensor([0 if l == 1 else 1 for l in labels_t[i:i + BS]],
                     dtype=torch.long, device=DEV)
    model.zero_grad(set_to_none=True)
    ta = time.time()
    out = model(x)
    loss = F.cross_entropy(out, y)
    tb = time.time()
    loss.backward()
    t_fwd += tb - ta
    t_bwd += time.time() - tb
    nb += 1
t_total = time.time() - t0
log(f'[task4] {nb} batches x {BS} = {N_TIME} imgs')
log(f'[task4] fwd={t_fwd:.2f}s  bwd={t_bwd:.2f}s  total(compute)={t_total:.2f}s')
log(f'[task4] epochs per ... : model was on device already (weights resident)')

# realistic epoch accounting: per image = load + fwd + bwd
per_img_compute = t_total / N_TIME
per_img_full = (t_total + t_load) / N_TIME
HRS_TOTAL = 107700 * per_img_full / 3600.0
HRS_COMPUTE = 107700 * per_img_compute / 3600.0
log(f'[task4] compute only   : {per_img_compute*1000:.2f} ms/img -> '
    f'{HRS_COMPUTE:.2f} h / 107,700 imgs')
log(f'[task4] load+compute   : {per_img_full*1000:.2f} ms/img -> '
    f'{HRS_TOTAL:.2f} h / 107,700 imgs')
S['throughput'] = {
    'n_images': N_TIME, 'batch_size': BS, 'n_batches': nb,
    'load_s': t_load, 'fwd_s': t_fwd, 'bwd_s': t_bwd, 'compute_s': t_total,
    'ms_per_img_compute': per_img_compute * 1000,
    'ms_per_img_load_plus_compute': per_img_full * 1000,
    'hours_per_epoch_compute': HRS_COMPUTE,
    'hours_per_epoch_load_plus_compute': HRS_TOTAL,
}


# ===========================================================================
# Save
# ===========================================================================
S['missing_paths'] = missing_paths[:50]
S['gpu_before'] = _gpu_info
S['gpu_index'] = C.GPU_INDEX
S['peak_mem_mib'] = mem_mib()[1]

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(S, f, ensure_ascii=False, indent=2, default=str)

np.savez_compressed(
    OUT_NPZ,
    scores_r0=scores_r0, scores_fixed=scores_fix,
    y=rows['y'], domain=rows['domain'], split=rows['split'],
    embed_before=emb_before, embed_after=emb_after,
    paths=rows['paths'],
)

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('')
print('[done] report ->', REPORT)
print('[done] stats  ->', OUT_JSON, '/', OUT_NPZ)
print(f'[GATE] TASK2={"PASS" if S["gate_task2"] else "FAIL"} '
      f'R0 mean4dom={auc_r0["mean4dom"]:.4f} FIXED mean4dom={auc_fix["mean4dom"]:.4f}')
