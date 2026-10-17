# -*- coding: utf-8 -*-
"""
G22-Phase1 verification -- common helpers.

ABSOLUTE CONSTRAINTS honoured here:
  * everything lives in vit_module/_g22/ ; no other repo file is touched
  * CPU threads pinned to 1 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB + torch)
  * GPU: physical index 1 only  (CUDA_VISIBLE_DEVICES=1 -> in-process cuda:0)
  * fp32 only
  * the model source file is NEVER edited -- FIX-A is a monkeypatch applied
    after the checkpoint has been loaded.

Reused, NOT re-invented:
  * image pipeline / dataset class come from  vit_module/train_bridge_phase1.py
    (FFPPDataset, IMG_TO_TENSOR = Normalize(CLIP mean/std) + ToTensorV2,
     cv2.imread -> BGR2RGB -> cv2.resize((336,336)))
  * model construction mirrors train_bridge_phase1.main()
"""

import os

# ---- CPU thread pinning: MUST happen before numpy/torch import -------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import sys
import time
import subprocess

import numpy as np
import torch
import torch.nn as nn
import cv2

torch.set_num_threads(1)
cv2.setNumThreads(0)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VIT_CKPT = r'E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth'
CLIP_LOCAL = os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336')
SHIPPED_CKPT = os.path.join(PROJECT_ROOT, 'checkpoints', 'stage_1', 'bridge_v2_phase1.pth')

DATA_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
TRAIN_TXT = os.path.join(DATA_ROOT, 'data_2023', 'ffpp_train_split.txt')
LAYER_FEATS = os.path.join(PROJECT_ROOT, 'vit_module', '_g16', 'layer_feats.npz')

GPU_INDEX = 1          # physical GPU we are allowed to use


# --------------------------------------------------------------------------
# GPU bookkeeping
# --------------------------------------------------------------------------
def query_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append(tuple(int(float(x)) for x in parts[:4]))
            except ValueError:
                continue
    return gpus


def gate_on_gpu(gpu_index=GPU_INDEX, max_used_mib=100):
    """Return (ok, info_str).  Hard gate: our card must be essentially free."""
    gpus = query_gpus()
    info = "; ".join(f"gpu{i}: used={u}MiB/{t}MiB util={ut}%" for i, u, t, ut in gpus)
    ok = False
    for i, u, t, ut in gpus:
        if i == gpu_index:
            ok = (u <= max_used_mib)
    return ok, info


def pin_gpu(gpu_index=GPU_INDEX):
    """Must be called BEFORE the first CUDA context is created."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# --------------------------------------------------------------------------
# Image pipeline -- imported verbatim from the training script
# --------------------------------------------------------------------------
def get_pipeline():
    """Return the project's own (FFPPDataset, IMG_TO_TENSOR, CLIP_MEAN/STD)."""
    import importlib
    m = importlib.import_module('vit_module.train_bridge_phase1')
    return m.FFPPDataset, m.IMG_TO_TENSOR, m.CLIP_MEAN, m.CLIP_STD


def load_image_tensor(path, img_to_tensor, size=336):
    """Exactly the training-script __getitem__ body."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'cv2.imread failed: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    return img_to_tensor(image=image)['image']


# --------------------------------------------------------------------------
# Dataset over explicit (path, label) rows -- same pipeline
# --------------------------------------------------------------------------
class RowDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, img_to_tensor, size=336):
        self.paths = [str(p) for p in paths]
        self.labels = [int(l) for l in labels]
        self.img_to_tensor = img_to_tensor
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_tensor(self.paths[idx], self.img_to_tensor, self.size)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def load_eval_rows():
    """G16 layer_feats.npz rows used as the dictionary of evaluation rows."""
    d = np.load(LAYER_FEATS, allow_pickle=True)
    return {
        'paths': np.array([str(p) for p in d['paths']]),
        'y': d['y'].astype(np.int64),
        'domain': np.array([str(x) for x in d['domain']]),
        'split': np.array([str(x) for x in d['split']]),
    }


def eval_groups(rows):
    """Return OrderedDict group_name -> boolean mask over the 4500 rows.

    in-domain : split == 'test'                     (800 rows, FF++)
    cross     : domain in {cd1,cd2,dfdcp,ffiw,wild} (300 rows each)
    """
    groups = {}
    groups['ffpp_test'] = (rows['split'] == 'test')
    for dom in ['cd1', 'cd2', 'dfdcp', 'ffiw', 'wild']:
        groups[dom] = (rows['domain'] == dom)
    return groups


# --------------------------------------------------------------------------
# Model construction -- mirrors train_bridge_phase1.main()
# --------------------------------------------------------------------------
def build_model(device='cuda:0', load_shipped=True, verbose=True):
    """Construct ViT_M2F2Det_Bridge exactly as the training script does, then
    load the ViT backbone from --vit-ckpt and the shipped phase-1 checkpoint."""
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge

    t0 = time.time()
    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=CLIP_LOCAL,
        clip_vision_encoder_name=CLIP_LOCAL,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
    ).to(device)
    if verbose:
        print(f'[build] model constructed in {time.time()-t0:.1f}s')

    # ViT backbone (in train_bridge_phase1 -> model.load_vit_backbone(args.vit_ckpt))
    model.load_vit_backbone(VIT_CKPT, verbose=verbose)

    if load_shipped:
        sd = torch.load(SHIPPED_CKPT, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if verbose:
            print(f'[build] shipped ckpt: {len(sd)} keys, '
                  f'missing={len(missing)} unexpected={len(unexpected)}')
            if missing:
                print('  missing (first 10):', missing[:10])
            if unexpected:
                print('  unexpected (first 10):', unexpected[:10])
        assert len(unexpected) == 0, 'unexpected keys -> checkpoint mismatch'
        assert len(missing) == 0, 'missing keys -> checkpoint mismatch'

    # freeze exactly like the training script (does not matter for eval, but
    # keeps the trainable-parameter set identical for the Task-4 timing run)
    for p in model.vit.parameters():
        p.requires_grad = False
    for p in model.clip_vision_encoder.parameters():
        p.requires_grad = False
    for p in model.clip_text_encoder.model.parameters():
        p.requires_grad = False
    model.clip_text_encoder.prompt_tokens.requires_grad = True
    return model


def apply_fix_a(model, verbose=True):
    """Monkeypatch: replace the buggy Sequential (View, Linear, LayerNorm(1))
    with (View, Linear).  Must be called AFTER the checkpoint was loaded.
    The Linear weight/bias are copied explicitly from the loaded checkpoint
    copy and verified bit-exact."""
    wrap = model.bridge_adapter_proj
    old_seq = wrap.bridge_adapter_proj
    old_linear = old_seq[1]                        # nn.Linear(embed_dim, 1)
    assert isinstance(old_linear, nn.Linear), type(old_linear)
    embed_dim = wrap.embed_dim

    w = old_linear.weight.detach().clone()
    b = old_linear.bias.detach().clone()

    new_linear = nn.Linear(embed_dim, 1)
    with torch.no_grad():
        new_linear.weight.copy_(w)
        new_linear.bias.copy_(b)
    new_linear = new_linear.to(w.device, dtype=w.dtype)

    from vit_module.vit_m2f2_detector_bridge import View
    new_seq = nn.Sequential(View(-1, embed_dim), new_linear)

    if verbose:
        d_w = (new_seq[1].weight.detach() - w).abs().max().item()
        d_b = (new_seq[1].bias.detach() - b).abs().max().item()
        print(f'[FIX-A] replaced bridge_adapter_proj.bridge_adapter_proj '
              f'(View,Linear,LayerNorm(1)) -> (View,Linear)')
        print(f'[FIX-A] linear weight max|diff| vs ckpt = {d_w:.3e}  '
              f'bias max|diff| = {d_b:.3e}')

    prev = (old_seq, old_linear)
    wrap.bridge_adapter_proj = new_seq
    return prev, (new_seq[1].weight.detach() - w).abs().max().item(), \
        (new_seq[1].bias.detach() - b).abs().max().item()


def revert_fix_a(model, prev):
    model.bridge_adapter_proj.bridge_adapter_proj = prev[0]
    return model


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def grad_norm(param):
    """Return 'None' as a string if .grad is None, else the L2 norm."""
    if param is None:
        return None
    if param.grad is None:
        return 'None'
    return float(param.grad.detach().norm().item())


def auc_safe(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, np.asarray(s)))
