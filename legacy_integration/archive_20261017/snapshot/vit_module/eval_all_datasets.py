"""
eval_all_datasets.py — Test the ViT_M2F2Det_Bridge detector on ALL benchmark datasets.

Supports two model types:
  --model-type bridge   → ViT_M2F2Det_Bridge (our BridgeAdapter version)
  --model-type original → M2F2Det EfficientNet-B4 (original project, quick baseline)

Complete data flow:
  Image (disk) → cv2 BGR→RGB → resize → normalize → model.forward()
    ├── _preprocess_for_vit (CLIP norm → [0,1] → 224 → [-1,1])
    ├── ViT backbone (triggers blocks[3][6][9] hooks)
    ├── CLIP Vision (layer 6/10/14 + final)
    ├── CLIP Text (prompt_tokens + text encoder)
    ├── BridgeAdapter 3-stage fusion
    └── classification head → [B,2] → softmax → prediction

Metrics per dataset: AUC-ROC, Accuracy, F1, Precision, Recall, EER

Usage:
  # Quick baseline (original M2F2Det EfficientNet, already trained):
  python vit_module/eval_all_datasets.py --model-type original \
      --checkpoint ./checkpoints/stage_1/current_model_180.pth \
      --data-root ./dataset

  # Our BridgeAdapter version (must train first with train_bridge_phase1.py):
  python vit_module/train_bridge_phase1.py --vit-ckpt ... --train-txt ... --data-root ...
  python vit_module/eval_all_datasets.py --model-type bridge \
      --checkpoint ./checkpoints/stage_1/bridge_phase1.pth \
      --data-root ./dataset
"""

import os, sys, argparse, json, gc, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

from albumentations import Compose, Normalize, ToTensorV2
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

# ── Path ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Constants ───────────────────────────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class EvalDataset(Dataset):
    """Load images from txt file (image_path label), apply normalization.

    Args:
        txt_path: path to txt file
        normalize_type: 'clip' for ViT_Bridge, 'imagenet' for original M2F2Det
    """
    def __init__(self, txt_path, normalize_type='clip'):
        self.normalize_type = normalize_type
        raw_data = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    raw_data.append((parts[0], int(parts[1])))

        self.data = []
        skipped = 0
        for path, label in raw_data:
            if os.path.exists(path):
                self.data.append((path, label))
            else:
                skipped += 1

        if skipped > 0:
            pct = 100. * skipped / max(len(raw_data), 1)
            print(f'   Skipped {skipped}/{len(raw_data)} ({pct:.1f}%) missing files')

        self.name = os.path.splitext(os.path.basename(txt_path))[0]

        if normalize_type == 'clip':
            self.transform = Compose([
                Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                ToTensorV2(),
            ])
            self.size = 336
        else:
            self.transform = Compose([
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
            self.size = 224

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data[idx]
        image = cv2.imread(fn, cv2.IMREAD_COLOR)
        if image is None:
            # fallback for broken images
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = self.transform(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════
def compute_eer(labels, scores):
    """Equal Error Rate — the error rate where FPR == FNR."""
    fpr_list, tpr_list, thresholds = [], [], []
    # Sort by score descending
    sorted_idx = np.argsort(scores)[::-1]
    labels_sorted = labels[sorted_idx]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    tp, fp = 0, 0
    fn = n_pos
    tn = n_neg

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        tpr = tp / n_pos
        fpr = fp / n_neg
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)

    # Find the point where |fpr - (1-tpr)| is minimized
    diffs = np.abs(fpr_list - (1.0 - tpr_list))
    min_idx = np.argmin(diffs)
    eer = (fpr_list[min_idx] + (1.0 - tpr_list[min_idx])) / 2.0

    return eer * 100.0  # percentage


@torch.no_grad()
def evaluate(model, loader, device, model_type='bridge'):
    """Run evaluation on a single dataset."""
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    for images, labels in tqdm(loader, desc='  Eval', leave=False):
        images, labels = images.to(device), labels.to(device)

        if model_type == 'original':
            # M2F2Det forward
            out = model(images, return_dict=True)
            logits = out['pred']
        else:
            # ViT_M2F2Det_Bridge forward
            logits = model(images)

        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)

    n = len(all_labels)
    if n == 0:
        return {'n': 0}

    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()

    # Accuracy
    acc = accuracy_score(all_labels, all_preds) * 100.0

    # AUC
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100.0
    except ValueError:
        auc = float('nan')

    # F1 / Precision / Recall (positive class = real = label 1)
    f1  = f1_score(all_labels, all_preds, zero_division=0) * 100.0
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100.0
    rec  = recall_score(all_labels, all_preds, zero_division=0) * 100.0

    # EER
    eer = compute_eer(all_labels, all_probs)

    return {
        'n': n, 'n_pos': int(n_pos), 'n_neg': int(n_neg),
        'acc': acc, 'auc': auc, 'f1': f1,
        'prec': prec, 'rec': rec, 'eer': eer,
    }


# ═══════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════
def build_bridge_model(device, clip_local_path):
    """Build ViT_M2F2Det_Bridge with local CLIP."""
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=clip_local_path,
        clip_vision_encoder_name=clip_local_path,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float16,
        text_dtype=torch.float16,
        deepfake_dtype=torch.float32,
    ).to(device)
    return model


def build_cosine_model(device, clip_local_path):
    """Build original ViT_M2F2Det (cosine-similarity version)."""
    from vit_module.vit_m2f2_detector import ViT_M2F2Det
    model = ViT_M2F2Det(
        clip_text_encoder_name=clip_local_path,
        clip_vision_encoder_name=clip_local_path,
        hidden_size=1024,
        load_vision_encoder=True,
        pretrained=False,
    ).to(device)
    return model


def build_original_model(device, clip_local_path):
    """Build original M2F2Det (EfficientNet-B4) in OFFLINE mode.

    Uses weights=None for torchvision backbone (avoids download).
    All weights are later loaded from the checkpoint.
    """
    from sequence.models.M2F2_Det.models.model import M2F2Det
    import torchvision.models.efficientnet as effnet_mod

    # ── Patch: build_deepfake_backbone to use weights=None (offline) ─
    _orig_build = None
    import sequence.models.M2F2_Det.models.model as m2f2_mod
    _orig_build = m2f2_mod.build_deepfake_backbone

    def _offline_build(model_name, feature_dim=None, hidden_size=1024):
        """Same as original but uses weights=None to avoid network download."""
        if 'efficientnet' in model_name:
            from torchvision.models import efficientnet
            model_init = {
                "efficientnet_b0": efficientnet.efficientnet_b0,
                "efficientnet_b1": efficientnet.efficientnet_b1,
                "efficientnet_b2": efficientnet.efficientnet_b2,
                "efficientnet_b3": efficientnet.efficientnet_b3,
                "efficientnet_b4": efficientnet.efficientnet_b4,
                "efficientnet_b5": efficientnet.efficientnet_b5,
                "efficientnet_b6": efficientnet.efficientnet_b6,
                "efficientnet_b7": efficientnet.efficientnet_b7,
            }
            model = model_init[model_name](weights=None).features  # offline
        elif 'densenet' in model_name:
            from torchvision.models import densenet
            model_init = {
                "densenet121": densenet.densenet121,
                "densenet161": densenet.densenet161,
                "densenet169": densenet.densenet169,
                "densenet201": densenet.densenet201,
            }
            model = model_init[model_name](weights=None).features  # offline
        else:
            raise ValueError(f'Unsupported deepfake encoder: {model_name}')

        if feature_dim is None:
            from sequence.models.M2F2_Det.models.model import get_feature_dim
            feature_dim = get_feature_dim(model_name)
        import torch.nn as nn
        proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        return model, proj

    m2f2_mod.build_deepfake_backbone = _offline_build
    try:
        model = M2F2Det(
            clip_text_encoder_name=clip_local_path,
            clip_vision_encoder_name=clip_local_path,
            deepfake_encoder_name='efficientnet_b4',
            hidden_size=1792,
        )
    finally:
        if _orig_build is not None:
            m2f2_mod.build_deepfake_backbone = _orig_build

    # NOTE: skip vision_tower.pth — the checkpoint already contains all CLIP
    # vision encoder weights.
    # For single-GPU eval, do NOT use DataParallel.
    model = model.to(device)
    return model


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Evaluate detector on all benchmark datasets')
    parser.add_argument('--model-type', choices=['bridge', 'original', 'vit_cosine'], default='bridge',
                        help='bridge=ViT_M2F2Det_Bridge, original=M2F2Det EfficientNet')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/stage_1/bridge_v2_phase1.pth',
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data-root', type=str, default='./dataset',
                        help='Root path containing data_2023/')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--datasets', type=str, default='all',
                        help='Comma-separated list or "all"')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save JSON results (optional)')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── CLIP local path ──────────────────────────────────────────
    clip_local = os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336')
    assert os.path.isdir(clip_local), f'CLIP model not found: {clip_local}'

    # ── Build model ──────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'Building model (type={args.model_type})...')
    print(f'{"="*70}')

    if args.model_type == 'bridge':
        model = build_bridge_model(device, clip_local)
        normalize_type = 'clip'
    elif args.model_type == 'vit_cosine':
        model = build_cosine_model(device, clip_local)
        normalize_type = 'clip'
    else:
        model = build_original_model(device, clip_local)
        normalize_type = 'imagenet'

    # ── Load checkpoint ─────────────────────────────────────────
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
    else:
        state_dict = ckpt

    if args.model_type == 'original':
        # No DataParallel — load directly
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f'  Missing keys: {len(missing)}  |  Unexpected keys: {len(unexpected)}')
    if len(missing) > 100:
        print(f'  (Large number of missing keys — this is expected for new BridgeAdapter layers)')
        print(f'  Sample missing: {missing[:5]}')

    model.eval()

    # ── Collect datasets ────────────────────────────────────────
    txt_dir = os.path.join(args.data_root, 'data_2023')

    # Categorize datasets
    dataset_categories = {
        'FF++':           ['ffpp_test_split'],
        'FF++ (c0)':      ['ffpp_test_split_c0'],
        'Celeb-DF':       ['CD1_test', 'CD2_test'],
        'DFD/DFR':        ['DFD_test', 'DFR_test'],
        'DFDC':           ['dfdc_test_lip', 'dfdcp_test'],
        'FFIW':           ['FFIW_test'],
        'WildDeepfake':   ['wild_test'],
        'Diffusion':      ['diff_test', 'diff_fe_test', 'diff_fs_test',
                           'diff_i2i_test', 'diff_real_test', 'diff_t2i_test'],
    }

    if args.datasets != 'all':
        requested = set(args.datasets.split(','))
        all_files = []
        for cat, files in dataset_categories.items():
            for f in files:
                txt_name = f + '.txt'
                if f in requested or txt_name in requested:
                    all_files.append((cat, f))
    else:
        all_files = []
        for cat, files in dataset_categories.items():
            for f in files:
                all_files.append((cat, f))

    # Build loaders
    dataloaders = []
    for category, name in all_files:
        txt_path = os.path.join(txt_dir, name + '.txt')
        if not os.path.exists(txt_path):
            print(f'  SKIP: {txt_path} not found')
            continue
        ds = EvalDataset(txt_path, normalize_type=normalize_type)
        if len(ds) == 0:
            print(f'  SKIP: {name} has 0 valid samples')
            continue
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        dataloaders.append((category, name, dl, len(ds)))
        print(f'  Loaded: {name:25s} [{category:15s}] {len(ds):>7d} samples')

    if not dataloaders:
        print('ERROR: No datasets found!')
        sys.exit(1)

    # ── Evaluate ─────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'Running evaluation on {len(dataloaders)} datasets...')
    print(f'{"="*70}')

    all_results = {}
    for category, name, loader, n_samples in dataloaders:
        print(f'\n[{category}] {name} ({n_samples} samples)')
        result = evaluate(model, loader, device, model_type=args.model_type)
        result['category'] = category
        result['dataset'] = name
        all_results[name] = result

        if result.get('n', 0) > 0:
            print(f'  AUC={result["auc"]:6.2f}%  Acc={result["acc"]:6.2f}%  '
                  f'F1={result["f1"]:6.2f}%  EER={result["eer"]:6.2f}%  '
                  f'(real={result["n_pos"]}, fake={result["n_neg"]})')

    # ── Summary table ────────────────────────────────────────────
    print(f'\n\n{"="*90}')
    print(f'SUMMARY — {args.model_type.upper()} model')
    print(f'{"="*90}')
    header = f'{"Dataset":<28s} {"N":>7s} {"AUC%":>8s} {"Acc%":>7s} {"F1%":>7s} {"EER%":>7s} {"Prec%":>7s} {"Rec%":>7s}'
    sep = '-' * len(header)
    print(header)
    print(sep)

    category_avgs = {}
    current_cat = None
    cat_results = []

    for category, name, _, _ in dataloaders:
        r = all_results.get(name, {})
        if r.get('n', 0) == 0:
            continue

        if category != current_cat:
            if cat_results and current_cat:
                avg_auc = np.mean([x['auc'] for x in cat_results if not np.isnan(x['auc'])])
                avg_acc = np.mean([x['acc'] for x in cat_results])
                print(f'  {"  ── avg ──":<28s} {"":>7s} '
                      f'{avg_auc:7.2f}% {avg_acc:6.2f}%')
            current_cat = category
            cat_results = []
            print(f'  [{category}]')

        print(f'  {name:<26s} {r["n"]:>7d} '
              f'{r["auc"]:7.2f}% {r["acc"]:6.2f}% {r["f1"]:6.2f}% '
              f'{r["eer"]:6.2f}% {r["prec"]:6.2f}% {r["rec"]:6.2f}%')
        cat_results.append(r)

    # Last category avg
    if cat_results:
        avg_auc = np.mean([x['auc'] for x in cat_results if not np.isnan(x['auc'])])
        avg_acc = np.mean([x['acc'] for x in cat_results])
        print(f'  {"  ── avg ──":<28s} {"":>7s} {avg_auc:7.2f}% {avg_acc:6.2f}%')

    # Overall average
    all_vals = [r for r in all_results.values() if r.get('n', 0) > 0]
    if all_vals:
        overall_auc = np.mean([x['auc'] for x in all_vals if not np.isnan(x['auc'])])
        overall_acc = np.mean([x['acc'] for x in all_vals])
        overall_f1  = np.mean([x['f1'] for x in all_vals])
        overall_eer = np.mean([x['eer'] for x in all_vals if not np.isnan(x['eer'])])
        print(sep)
        print(f'  {"OVERALL AVERAGE":<28s} {"":>7s} '
              f'{overall_auc:7.2f}% {overall_acc:6.2f}% {overall_f1:6.2f}% {overall_eer:6.2f}%')
    print(sep)

    # ── Save JSON ────────────────────────────────────────────────
    if args.output:
        output_data = {}
        for name, r in all_results.items():
            output_data[name] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                 for k, v in r.items()}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f'\nResults saved to: {args.output}')

    print(f'\nDone. Evaluated {len(all_vals)} datasets.')


if __name__ == '__main__':
    main()
