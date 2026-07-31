"""
Phase 1 training script for ViT_M2F2Det.

Trains only the newly added layers of ViT_M2F2Det:
- deepfake_proj (bridge adapter: 768→1024)
- vision_proj (CLIP fusion)
- text_proj (text fusion)
- output (Mb classification layer)
- clip_vision_alpha, clip_text_alpha

The ViT backbone is loaded from net_050.pth and frozen.

Multi-dataset validation:
  FF++ subsets:  Deepfakes, Face2Face, FaceSwap, NeuralTextures
  Cross-domain:  CD1, CD2, DFD, DFR, FFIW, dfdc, dfdcp, Wild, Diffusion
  Extra:         diff_fe, diff_fs, diff_i2i, diff_real, diff_t2i
"""

import os
import sys
import argparse
import logging
import datetime
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from albumentations import Compose, Normalize, ToTensorV2

# ── Add project root to sys.path so `llava` package is importable ─────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Logging setup ────────────────────────────────────────────────────
def setup_logger(save_dir):
    """Set up logger that writes to both console and file."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'training_{timestamp}.log')

    logger = logging.getLogger('phase1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    return logger, log_file
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vit_m2f2_detector import ViT_M2F2Det


# ── Dataset (loads FF++ style txt files with image paths) ──────────────
# The detector's forward() expects CLIP-normalized [B,3,336,336] images,
# then internally converts them to ViT-normalized [B,3,224,224].
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMG_TO_TENSOR = Compose([
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2()
])

FIVE_CLASSES = ['original_sequences', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

class FFPPDataset(Dataset):
    """FF++ style dataset. Each line: <image_path> <label (0/1)>.
    Supports optional method filtering (e.g. expected_method='Deepfakes').
    Automatically skips missing image files."""
    def __init__(self, txt_path, size=336, expected_method=None, name=None):
        raw_data = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path, label = parts[0], int(parts[1])
                    if expected_method is None or expected_method in path:
                        raw_data.append((path, label))

        # Filter out missing files and log count
        self.data = []
        skipped = 0
        for path, label in raw_data:
            if os.path.exists(path):
                self.data.append((path, label))
            else:
                skipped += 1

        if skipped:
            print(f'  [FFPPDataset] Skipped {skipped}/{len(raw_data)} missing files in {txt_path}')

        self.size = size
        base = os.path.splitext(os.path.basename(txt_path))[0]
        self.name = name if name else base
        if expected_method:
            self.name += f'({expected_method})'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data[idx]
        image = cv2.imread(fn, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = IMG_TO_TENSOR(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Yields balanced batches across 5 classes.
    Each batch: batch_size//5 samples per class, mixed within batch.
    Uses a simple collate + single DataLoader (num_workers=2)
    to avoid multiprocess overhead from 5 parallel loaders.
    """
    def __init__(self, txt_path, batch_size, size=336):
        self.batch_size = batch_size
        self.per_class = max(1, batch_size // 5)
        self.effective_batch = self.per_class * 5

        # Build 5 per-class datasets
        self.class_datasets = {}
        for cls_name in FIVE_CLASSES:
            ds = FFPPDataset(txt_path, size=size, expected_method=cls_name)
            self.class_datasets[cls_name] = ds

        valid_lens = [len(ds) for ds in self.class_datasets.values() if len(ds) > 0]
        self.batches_per_epoch = min(valid_lens) // self.per_class if valid_lens else 0
        self._reshuffle()

    def _reshuffle(self):
        for cls_name in FIVE_CLASSES:
            np.random.shuffle(self.class_datasets[cls_name].data)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        self._reshuffle()
        self._pos = {c: 0 for c in FIVE_CLASSES}
        return self

    def __next__(self):
        if self._pos[FIVE_CLASSES[0]] >= self.batches_per_epoch * self.per_class:
            raise StopIteration
        batch_indices = []
        for cls_name in FIVE_CLASSES:
            ds = self.class_datasets[cls_name]
            pos = self._pos[cls_name]
            for i in range(self.per_class):
                if pos < len(ds):
                    batch_indices.append(ds.data[pos])
                    pos += 1
            self._pos[cls_name] = pos
        np.random.shuffle(batch_indices)
        return batch_indices  # list of (path, label)


def balanced_collate(batch_indices_list):
    """Collate: load images from paths and stack into batch tensors."""
    images, labels = [], []
    for fn, lb in batch_indices_list:
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (336, 336))
        img = IMG_TO_TENSOR(image=img)['image']
        images.append(img)
        labels.append(torch.tensor(lb, dtype=torch.long))
    return torch.stack(images), torch.stack(labels)


def get_val_loaders(data_root, batch_size):
    """
    Build a dict of {name: DataLoader} for all available test sets.
    Returns only datasets whose txt file exists.
    """
    txt_dir = os.path.join(data_root, 'data_2023')
    num_workers = 0  # 0 to avoid multiprocessing issues during eval
    loaders = {}

    # ── FF++ test set (all methods combined) ────────────────
    ffpp_all = os.path.join(txt_dir, 'ffpp_test_split.txt')
    if os.path.exists(ffpp_all):
        ds = FFPPDataset(ffpp_all)
        if len(ds) > 0:
            loaders['FFPP_all'] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── FF++ per-method subsets ─────────────────────────────
    # ffpp_c0 = os.path.join(txt_dir, 'ffpp_test_split_c0.txt')
    # for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
    #     if os.path.exists(ffpp_c0):
    #         ds = FFPPDataset(ffpp_c0, expected_method=method)
    #         if len(ds) > 0:
    #             key = f'FFc0_{method}'
    #             loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── Cross-domain datasets ───────────────────────────────
    cross_domain = {
        # 'CD1': 'CD1_test.txt', 
        'CD2': 'CD2_test.txt',
        # 'DFD': 'DFD_test.txt', 'DFR': 'DFR_test.txt',
        'FFIW': 'FFIW_test.txt',
        # 'dfdc': 'dfdc_test_lip.txt',
        'dfdcp': 'dfdcp_test.txt',
        'Wild': 'wild_test.txt',
        # 'ffpp':'ffpp_test_split.txt',
    }
    for name, fname in cross_domain.items():
        path = os.path.join(txt_dir, fname)
        if os.path.exists(path):
            ds = FFPPDataset(path, name=name)
            if len(ds) > 0:
                loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── Diffusion datasets ──────────────────────────────────
    diff_sets = {
        # 'Diff_all': 'diff_test.txt',
        # 'Diff_fe': 'diff_fe_test.txt',
        # 'Diff_fs': 'diff_fs_test.txt',
        # 'Diff_i2i': 'diff_i2i_test.txt',
        # 'Diff_real': 'diff_real_test.txt',
        # 'Diff_t2i': 'diff_t2i_test.txt',
    }
    for name, fname in diff_sets.items():
        path = os.path.join(txt_dir, fname)
        if os.path.exists(path):
            ds = FFPPDataset(path, name=name)
            if len(ds) > 0:
                loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loaders


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a single dataloader.
    Returns dict with: loss, acc, precision, recall, f1, auc, count
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        B = images.shape[0]

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = F.softmax(outputs, dim=1)[:, 1]   # fake-class probability
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = total_loss / max(len(all_labels), 1)

    # Precision / Recall / F1 (macro)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # AUC (needs both classes present)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return {
        'loss': avg_loss,
        'acc': acc,
        'precision': prec * 100,
        'recall': rec * 100,
        'f1': f1 * 100,
        'auc': auc * 100 if not np.isnan(auc) else float('nan'),
        'count': len(all_labels),
    }


def main(args):
    device = torch.device('cuda:0')

    # ── Logger (save dir = parent of save_path) ──────────────────────────
    save_dir = os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.'
    logger, log_file = setup_logger(save_dir)
    logger.info('=' * 70)
    logger.info('Phase 1 Training: ViT_M2F2Det Fusion Layers')
    logger.info('=' * 70)
    logger.info(f'Log file: {log_file}')

    # Log args
    for k, v in sorted(vars(args).items()):
        logger.info(f'  {k}: {v}')

    # ── Model ───────────────────────────────────────────────────────────
    logger.info('Building ViT_M2F2Det...')
    # Use local CLIP model paths (offline environment)
    clip_local_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336')
    heatmap_dir = os.path.join(save_dir, 'heatmaps')
    model = ViT_M2F2Det(
        clip_text_encoder_name=clip_local_path,
        clip_vision_encoder_name=clip_local_path,
        load_vision_encoder=True,      # ← 启用真实 CLIP 视觉编码器
        pretrained=False,
        save_heatmap=True,
        heatmap_dir=heatmap_dir,
    ).to(device)
    logger.info(f'Heatmaps saved to: {heatmap_dir}')

    # Load ViT backbone from training checkpoint
    logger.info(f'Loading ViT backbone from {args.vit_ckpt}')
    model.load_vit_backbone(args.vit_ckpt, verbose=True)

    # ── Freeze strategy ─────────────────────────────────────────────
    # Freeze ViT backbone
    for p in model.vit.parameters():
        p.requires_grad = False
    logger.info('Frozen: ViT backbone')

    # Freeze CLIP vision encoder (already frozen internally, explicit for safety)
    if model.clip_vision_encoder is not None:
        for p in model.clip_vision_encoder.parameters():
            p.requires_grad = False
        logger.info('Frozen: CLIP Vision Encoder')

    # Freeze CLIP text backbone, but train prompt_tokens
    for p in model.clip_text_encoder.model.parameters():
        p.requires_grad = False
    model.clip_text_encoder.prompt_tokens.requires_grad = True
    logger.info('Frozen: CLIP Text backbone')
    logger.info(f'Trainable: prompt_tokens ({model.clip_text_encoder.prompt_tokens.numel()} params)')

    # ── Trainable layers ────────────────────────────────────────────
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)

    logger.info(f'Trainable parameters ({len(trainable_params)}):')
    for n in trainable_params:
        logger.info(f'  {n}')

    # Count total vs trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params: {total_params:,}  |  Trainable: {trainable_count:,}  ({100.*trainable_count/total_params:.4f}%)')

    # ── Data ────────────────────────────────────────────────────────────
    # 5-class balanced batch sampler (per_class = batch_size//5 from each class)
    batch_sampler = BalancedBatchSampler(args.train_txt, batch_size=args.batch_size)
    logger.info(f'Train samples: 28720 across 5 classes | Effective batch: {batch_sampler.effective_batch} | Batches/epoch: {len(batch_sampler)}')
    for cls_name in FIVE_CLASSES:
        ds = batch_sampler.class_datasets[cls_name]
        logger.info(f'  {cls_name:25s}  {len(ds):>6d} samples')

    # ── Build multi-dataset validation loaders ──────────────────────────
    if args.data_root:
        val_loaders = get_val_loaders(args.data_root, args.batch_size)
        logger.info(f'Validation datasets ({len(val_loaders)}):')
        for name, loader in val_loaders.items():
            logger.info(f'  {name:20s}  {len(loader.dataset):>6d} samples')
    else:
        val_loaders = {}
        logger.info('No data_root specified — skipping multi-dataset validation.')

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ────────────────────────────────────────────────────
    best_acc = 0.0
    best_epoch = -1
    epoch_times = []
    logger.info(f'Starting training for {args.epochs} epochs...')
    logger.info(f'Effective batch: {batch_sampler.effective_batch} | Batches/epoch: {len(batch_sampler)}')

    # ── Eval table header (printed once after epoch 1) ──────────────────
    all_val_names = sorted(val_loaders.keys()) if val_loaders else []

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        epoch_start = datetime.datetime.now()

        pbar = tqdm(batch_sampler, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_data in pbar:
            images, labels = balanced_collate(batch_data)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            B = images.shape[0]
            # model内部自动调用 CLIP Vision Encoder 提取真实特征
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{total_loss/(total/batch_sampler.effective_batch+1e-8):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        scheduler.step()
        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        epoch_times.append(epoch_time)
        train_acc = 100. * correct / total
        train_loss = total_loss / max(len(batch_sampler), 1)

        # ── Train metrics ───────────────────────────────────────────────
        logger.info('')
        logger.info(f'─── Epoch {epoch+1:2d}/{args.epochs}  (Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%) ───')

        # ── Multi-dataset Validation ────────────────────────────────────
        if val_loaders:
            print("----------正在多数据集上验证------------")

            epoch_results = {}
            
            for name in all_val_names:
                loader = val_loaders[name]
                results = evaluate(model, loader, device)
                epoch_results[name] = results

            # ── Per-dataset results ───────────────────────────────────────
            header = f'{"Dataset":22s} {"Loss":>8s} {"Acc%":>7s} {"Prec%":>7s} {"Recall%":>7s} {"F1%":>7s} {"AUC%":>7s} {"Count":>7s}'
            sep = '-' * len(header)

            # Log to file with levelname, console without
            logger.info(sep)
            logger.info(header)
            logger.info(sep)
            for name in all_val_names:
                r = epoch_results[name]
                logger.info(
                    f'{name:22s} {r["loss"]:8.4f} {r["acc"]:6.2f}% '
                    f'{r["precision"]:6.2f}% {r["recall"]:6.2f}% '
                    f'{r["f1"]:6.2f}% {r["auc"]:6.2f}% {r["count"]:>7d}'
                )
            logger.info(sep)
            logger.info(f'Time: {epoch_time:.1f}s')
            logger.info('')

            # ── Compute average metrics (FF++ subsets & cross-domain) ────
            ffpp_keys = [n for n in all_val_names if n.startswith('FF')]
            cd_keys = [n for n in all_val_names if n not in ffpp_keys]
            for group_name, keys in [('FF++ AVG', ffpp_keys), ('Cross-Domain AVG', cd_keys)]:
                if keys:
                    avg_acc = np.mean([epoch_results[k]['acc'] for k in keys])
                    avg_f1 = np.mean([epoch_results[k]['f1'] for k in keys])
                    avg_auc = np.nanmean([epoch_results[k]['auc'] for k in keys])
                    logger.info(f'  {group_name:20s} → Acc: {avg_acc:.2f}%  F1: {avg_f1:.2f}%  AUC: {avg_auc:.2f}%')

            # Best model: use the first FF++ dataset's AUC, or fall back to first dataset's acc
            primary_name = ffpp_keys[0] if ffpp_keys else all_val_names[0]
            primary_auc = epoch_results[primary_name]['auc']
            if primary_auc > best_acc or (best_epoch == -1):
                best_acc = primary_auc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), args.save_path)
                logger.info(f'  ★ New best model (AUC={primary_auc:.2f}% on {primary_name}) → saved to {args.save_path}')
                print(f' best model(AUC={primary_auc:.2f}% on {primary_name})')
        else:
            logger.info(f'Time: {epoch_time:.1f}s')

    # ── Final Summary ────────────────────────────────────────────────────
    total_time = sum(epoch_times)
    logger.info('=' * 70)
    logger.info('Training Complete!')
    logger.info(f'Total time: {total_time:.1f}s ({total_time/60:.1f} min)')
    logger.info(f'Average epoch: {total_time/len(epoch_times):.1f}s')

    if best_epoch > 0:
        logger.info(f'Best model (AUC={best_acc:.2f}%) at epoch {best_epoch}')
        logger.info(f'Best model saved to: {args.save_path}')
    else:
        torch.save(model.state_dict(), args.save_path)
        logger.info(f'Final model saved to: {args.save_path}')
    logger.info('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit-ckpt', type=str,
                        default=r'E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth',
                        help='Path to net_XXX.pth from train_Ama_aps.py')
    parser.add_argument('--train-txt', type=str,
                        default='./dataset/data_2023/ffpp_train_split.txt',
                        help='Path to FF++ train split txt file')
    parser.add_argument('--val-txt', type=str, default=None,
                        help='[DEPRECATED — use --data-root instead] Path to FF++ val split txt file')
    parser.add_argument('--data-root', type=str, default='./dataset',
                        help='Path to dataset root (parent of data_2023/), e.g. "../dataset"')
    parser.add_argument('--save-path', type=str, default='./checkpoints/stage_1/cosine_realclip_phase1.pth',
                        help='Where to save the trained detector')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    args = parser.parse_args()

    # Backward compatibility: if --val-txt is given but --data-root isn't, derive data_root
    if args.data_root is None and args.val_txt is not None:
        args.data_root = os.path.dirname(os.path.dirname(os.path.abspath(args.val_txt)))

    main(args)
