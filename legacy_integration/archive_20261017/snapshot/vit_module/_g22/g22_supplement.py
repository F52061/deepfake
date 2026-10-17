# -*- coding: utf-8 -*-
"""
G22 supplement -- CPU only, no GPU.

Explains the one deviation from the stated Task-2(a) expectation:
  bridge_adapter_proj.bridge_adapter_reduction[0].weight grad == 0.027 (NON-ZERO)
even though the bridge upstream is dead.  Also quantifies how much of the
FIXED model's clip_adapt_embed is a constant offset vs. a varying signal.

Appends its output to phase1_report.txt.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import g22_common as C  # noqa: E402

import numpy as np
import torch
import torch.nn as nn

lines = []


def log(m=''):
    print(m, flush=True)
    lines.append(str(m))


sd = torch.load(C.SHIPPED_CKPT, map_location='cpu')

ln_w = sd['bridge_adapter_proj.bridge_adapter_proj.2.weight']   # [1]
ln_b = sd['bridge_adapter_proj.bridge_adapter_proj.2.bias']     # [1]
lin_w = sd['bridge_adapter_proj.bridge_adapter_proj.1.weight']  # [1,64]
lin_b = sd['bridge_adapter_proj.bridge_adapter_proj.1.bias']    # [1]

log('=' * 78)
log('SUPPLEMENT S1 -- why bridge_adapter_reduction DID receive gradient')
log('=' * 78)
log(f'LayerNorm(1).weight = {ln_w.item():.9f}   bias = {ln_b.item():.9f}  (eps=1e-5)')
log('LayerNorm over the last dim of shape [N,1]: mean(x)=x, var(x)=0  ->')
log('   out = ((x - x)/sqrt(0+eps)) * w + b = 0*w + b = b   -> a CONSTANT.')
log(f'   => bridge_adapter_proj() output = {ln_b.item():.9f} for EVERY token / image.')

# constant fed into bridge_adapter_reduction: [B, 2316] all equal to ln_b
const_vec = torch.full((1, 2316), float(ln_b))
red_lin_w = sd['bridge_adapter_proj.bridge_adapter_reduction.0.weight']   # [128,2316]
red_lin_b = sd['bridge_adapter_proj.bridge_adapter_reduction.0.bias']
red_ln_w = sd['bridge_adapter_proj.bridge_adapter_reduction.1.weight']
red_ln_b = sd['bridge_adapter_proj.bridge_adapter_reduction.1.bias']

with torch.no_grad():
    h = const_vec @ red_lin_w.t() + red_lin_b            # [1,128]
    lnorm = nn.LayerNorm(128)
    lnorm.weight.copy_(red_ln_w)
    lnorm.bias.copy_(red_ln_b)
    embed_const = lnorm(h)                               # [1,128]

log('')
log('The INPUT to bridge_adapter_reduction is the SAME non-zero constant vector')
log(f'({float(ln_b):.9f} repeated 2316x) for every sample in every batch.')
log(f'  -> Linear(2316,128) grad_W = delta^T @ x_const  != 0   (x_const != 0)')
log(f'  -> hence grad_norm(bridge_adapter_reduction[0].weight) = 0.0270, NOT 0.')
log('  But that gradient trains the reduction on an input that carries ZERO')
log('  information about the image. The learned "signal" is a pure constant.')
log('')
a_t = float(sd['clip_text_alpha'])
log(f'clip_text_alpha from checkpoint = {a_t:.9f}')
log(f'Analytic alpha*const embed   = '
    f'{np.round(embed_const.numpy()[0][:8] * a_t, 7).tolist()} ...')

stats = np.load(os.path.join(_HERE, 'phase1_stats.npz'), allow_pickle=True)
eb = stats['embed_before']
log(f'Measured embed_before row0   = {np.round(eb[0][:8], 7).tolist()} ...')
d = float(np.abs(embed_const.numpy()[0] * a_t - eb[0]).max())
log(f'>>> max |analytic alpha*const - measured embed_before| = {d:.3e}   '
    f'(exact to fp32? {d < 1e-6})')
log(f'>>> raw (no-alpha) analytic vs measured ratio = '
    f'{(eb[0][5] / embed_const.numpy()[0][5]):.6f}  (should equal clip_text_alpha)')

# --------------------------------------------------------------- S2 ---------
log('')
log('=' * 78)
log('SUPPLEMENT S2 -- R0 score is an affine function that is INDEPENDENT of the bridge')
log('=' * 78)
W = sd['output.weight']   # [2,1664]
b = sd['output.bias']
log('features = cat([clip_vision_cls(768)*a_v, bridge_embed(128)*a_t, vit_features(768)], -1)')
log('score = logit1 - logit0 = (W[1]-W[0]) . features + (b[1]-b[0])')
dW = (W[1] - W[0]).numpy()
# bridge block is features[:, 768:896]; embed_before already contains
# clip_text_alpha, so do NOT multiply by it again.
offset = float(dW[768:896] @ eb[0])
log(f'bridge block of (W[1]-W[0]) has ||.|| = {np.linalg.norm(dW[768:896]):.6f}')
log(f'bridge contributes  offset = dW[768:896] . (clip_text_alpha * embed)')
log(f'                        = {offset:.8f}   (identical for EVERY image, since')
log(f'                          embed is the same constant for every image)')
log('=> removing the bridge entirely shifts every score by the SAME constant, and')
log('   AUC is rank-based, so R0 AUC (unmodified) is EXACTLY the AUC of a model')
log('   whose bridge output is deleted. The dead bridge is invisible to R0 AUC by')
log('   construction. This is a derivation, not an approximation.')

# --------------------------------------------------------------- S3 ---------
log('')
log('=' * 78)
log('SUPPLEMENT S3 -- how much of the FIXED model embed is constant vs varying?')
log('=' * 78)
ea = stats['embed_after']
mu = ea.mean(0, keepdims=True)
dev = ea - mu
log(f'fixed embed: ||mean row||   = {np.linalg.norm(mu):.6f}')
log(f'fixed embed: max ||row-mean|| = {np.linalg.norm(dev, axis=1).max():.6f}')
log(f'fixed embed: per-row norms  = {np.round(np.linalg.norm(ea, axis=1), 6).tolist()}')
r = np.linalg.norm(dev, axis=1).max() / np.linalg.norm(mu)
log(f'  varying / constant ratio = {r:.6f}  ({r*100:.3f} %)')
log('i.e. even after FIX-A the bridge output is dominated by the constant that the')
log('reduction layer learned while trained on a constant input. The fix restores')
log('gradient flow, but it does NOT by itself restore a useful representation --')
log('the bridge must be RETRAINED (which is exactly what phase-1 re-run will do).')

with open(os.path.join(_HERE, 'phase1_report.txt'), 'a', encoding='utf-8') as f:
    f.write('\n' + '\n'.join(lines) + '\n')

np.savez_compressed(os.path.join(_HERE, 'supplement.npz'),
                    embed_const=embed_const.numpy(),
                    delta_W=dW, bridge_offset=np.array([offset]))
print('\n[done] appended to phase1_report.txt')
