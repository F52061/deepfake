"""
Phase 1 training script for ViT_M2F2Det_Bridge (BridgeAdapter version).

Anti-overfitting strategy (based on previous AUC regression):
  1. ViT backbone: FROZEN (91M params overfits 107K data)
  2. BridgeAdapter layers: trained with higher LR (random init)
  3. Projection layers: trained with moderate LR
  4. prompt_tokens: trained with higher LR (random init)
  5. CosineAnnealing LR schedule
  6. Early stopping on FF++ validation AUC (patience=3)
  7. Gradient clipping (max_norm=1.0)

Frozen:
  — ViT backbone (91.4M)
  — CLIP Vision Encoder (428M)
  — CLIP Text Encoder backbone (123M)

Trainable (~3.2M):
  — BridgeAdapter: clip_reduction, linear_vit_1/2/3, 3×TransformerBlock, BridgeAdapter_Proj
  — Projections: deepfake_proj, vision_proj, text_proj
  — prompt_tokens (learned text prompt)
  — output classifier
  — clip_vision_alpha / clip_text_alpha

Usage:
    python vit_module/train_bridge_phase1.py \
        --vit-ckpt E:/.../PDI/results/Ama1_aps1_1/net_050.pth \
        --train-txt ./dataset/data_2023/ffpp_train_split.txt \
        --data-root ./dataset \
        --save-path ./checkpoints/stage_1/bridge_v2_phase1.pth \
        --epochs 30 --lr 5e-4 --batch-size 6
"""

import os, sys, argparse, logging, datetime, json, gc
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from albumentations import Compose, Normalize, ToTensorV2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════
def setup_logger(save_dir, name='bridge_v2'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'{name}_{timestamp}.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    return logger, log_file


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
IMG_TO_TENSOR = Compose([
    Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2(),
])

FIVE_CLASSES = ['original_sequences', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


class FFPPDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, size=336, expected_method=None, name=None):
        raw_data = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path, label = parts[0], int(parts[1])
                    if expected_method is None or expected_method in path:
                        raw_data.append((path, label))
        self.data = [(p, l) for p, l in raw_data if os.path.exists(p)]
        skipped = len(raw_data) - len(self.data)
        if skipped:
            print(f'  [FFPPDataset] Skipped {skipped}/{len(raw_data)} missing in {txt_path}')
        self.name = name or os.path.splitext(os.path.basename(txt_path))[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data[idx]
        image = cv2.imread(fn, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (336, 336))
        image = IMG_TO_TENSOR(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, txt_path, batch_size, size=336):
        self.batch_size = batch_size
        self.per_class = max(1, batch_size // 5)
        self.effective_batch = self.per_class * 5
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
        return batch_indices


def balanced_collate(batch_indices_list):
    images, labels = [], []
    for fn, lb in batch_indices_list:
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (336, 336))
        img = IMG_TO_TENSOR(image=img)['image']
        images.append(img)
        labels.append(torch.tensor(lb, dtype=torch.long))
    return torch.stack(images), torch.stack(labels)


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════
def get_val_loaders(data_root, batch_size):
    txt_dir = os.path.join(data_root, 'data_2023')
    loaders = {}
    candidates = {
        'FFPP_all': 'ffpp_test_split.txt',
        'CD2': 'CD2_test.txt',
        'FFIW': 'FFIW_test.txt',
        'dfdcp': 'dfdcp_test.txt',
        'Wild': 'wild_test.txt',
    }
    for name, fname in candidates.items():
        path = os.path.join(txt_dir, fname)
        if os.path.exists(path):
            ds = FFPPDataset(path, name=name)
            if len(ds) > 0:
                loaders[name] = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, shuffle=False, num_workers=0
                )
    return loaders


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = F.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = 100.0 * (all_preds == all_labels).sum() / max(len(all_labels), 1)
    avg_loss = total_loss / max(len(all_labels), 1)

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return {
        'loss': avg_loss, 'acc': acc,
        'precision': prec * 100, 'recall': rec * 100, 'f1': f1 * 100,
        'auc': auc * 100 if not np.isnan(auc) else float('nan'),
        'count': len(all_labels),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device('cuda:2')
    save_dir = os.path.dirname(args.save_path) or '.'
    logger, log_file = setup_logger(save_dir)

    logger.info('=' * 70)
    logger.info('ViT_M2F2Det_Bridge — Phase 1 Training (v2: Frozen ViT)')
    logger.info('=' * 70)
    for k, v in sorted(vars(args).items()):
        logger.info(f'  {k}: {v}')
    logger.info(f'Log: {log_file}')

    # ── Model ───────────────────────────────────────────────────
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
    clip_local = os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336')

    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=clip_local,
        clip_vision_encoder_name=clip_local,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
    ).to(device)

    # Load ViT backbone
    logger.info(f'Loading ViT backbone from {args.vit_ckpt}')
    model.load_vit_backbone(args.vit_ckpt, verbose=True)

    # ── Freeze strategy ─────────────────────────────────────────
    # 1. Freeze ViT backbone (prevents overfitting — 91M on 107K data)
    for p in model.vit.parameters():
        p.requires_grad = False
    logger.info('FROZEN: ViT backbone (91.4M) — anti-overfitting')

    # 2. Freeze CLIP vision
    if model.clip_vision_encoder is not None:
        for p in model.clip_vision_encoder.parameters():
            p.requires_grad = False
    logger.info('FROZEN: CLIP Vision Encoder (428M)')

    # 3. Freeze CLIP text backbone, train prompt_tokens
    for p in model.clip_text_encoder.model.parameters():
        p.requires_grad = False
    model.clip_text_encoder.prompt_tokens.requires_grad = True
    logger.info('FROZEN: CLIP Text backbone (123M)')
    logger.info('TRAIN:  prompt_tokens')

    # Log which params are trainable
    trainable_names = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]
    logger.info(f'\nTrainable parameters ({len(trainable_names)}):')
    for n in trainable_names:
        logger.info(f'  {n}')

    # ── Optimizer — manually define groups ──────────────────────
    # ViT backbone is NOT in optimizer (frozen)
    # BridgeAdapter random-init layers → higher LR
    # Projection layers → medium LR
    # Alpha / prompt → high LR

    param_groups = []

    # Group 1: Alpha params (lr=1e-3)
    param_groups.append({'params': [model.clip_vision_alpha], 'lr': 1e-3})
    param_groups.append({'params': [model.clip_text_alpha],   'lr': 3e-3})

    # Group 2: prompt_tokens (lr=1e-3, random init)
    if hasattr(model.clip_text_encoder, 'prompt_tokens'):
        param_groups.append({
            'params': [model.clip_text_encoder.prompt_tokens],
            'lr': 1e-3,
        })

    # Group 3: BridgeAdapter randomly initialized layers (lr=args.lr)
    #   clip_reduction, linear_vit_1/2/3, bridge_adapter, bridge_adapter_proj
    bridge_modules = [
        model.clip_reduction,
        *model.linear_vit_lst,
        *model.bridge_adapter,
        model.bridge_adapter_proj,
    ]
    for mod in bridge_modules:
        param_groups.append({'params': mod.parameters(), 'lr': args.lr})

    # Group 4: Projection layers (lr=args.lr)
    proj_modules = [
        model.deepfake_proj,
        model.vision_proj,
        model.text_proj,
        model.output,
    ]
    for mod in proj_modules:
        param_groups.append({'params': mod.parameters(), 'lr': args.lr * 0.5})

    optimizer = AdamW(param_groups, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Log parameter groups
    total_train = sum(p.numel() for g in param_groups for p in g['params'])
    logger.info(f'\nOptimizer groups ({len(param_groups)}):')
    for i, g in enumerate(param_groups):
        n = sum(p.numel() for p in g['params'])
        logger.info(f'  Group {i}: {n/1e6:7.3f}M  lr={g["lr"]}')
    logger.info(f'Total trainable: {total_train/1e6:.2f}M')

    # ── Data ───────────────────────────────────────────────────
    batch_sampler = BalancedBatchSampler(args.train_txt, batch_size=args.batch_size)
    total_samples = sum(len(ds) for ds in batch_sampler.class_datasets.values())
    logger.info(f'\nTrain: {total_samples} samples, eff_batch={batch_sampler.effective_batch}')
    logger.info(f'Batches/epoch: {len(batch_sampler)}')
    for cls_name in FIVE_CLASSES:
        logger.info(f'  {cls_name:25s} {len(batch_sampler.class_datasets[cls_name]):>6d}')

    # Validation
    val_loaders = get_val_loaders(args.data_root, args.batch_size) if args.data_root else {}
    all_val_names = sorted(val_loaders.keys())
    logger.info(f'\nValidation ({len(val_loaders)} datasets):')
    for name, loader in val_loaders.items():
        logger.info(f'  {name:20s} {len(loader.dataset):>6d} samples')
    logger.info('')

    # ── Training Loop ───────────────────────────────────────────
    best_auc, best_epoch = 0.0, -1
    early_stop_counter = 0
    patience = args.patience
    accum_steps = max(1, 32 // batch_sampler.effective_batch)

    logger.info(f'Starting {args.epochs} epochs (patience={patience}, accum={accum_steps})...\n')

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        epoch_start = datetime.datetime.now()
        optimizer.zero_grad()

        pbar = tqdm(batch_sampler, desc=f'Epoch {epoch+1}/{args.epochs}')
        for step, batch_data in enumerate(pbar):
            images, labels = balanced_collate(batch_data)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{total_loss / max(step + 1, 1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
            })

        scheduler.step()
        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        train_acc = 100. * correct / total
        train_loss = total_loss / max(len(batch_sampler), 1)

        # ── Validation ──────────────────────────────────────
        val_str = 'TRAIN'
        if val_loaders:
            epoch_results = {}
            for name in all_val_names:
                epoch_results[name] = evaluate(model, val_loaders[name], device)

            header = f'{"Dataset":20s}  {"Loss":>7s}  {"Acc%":>6s}  {"Prec%":>6s}  {"Recall%":>6s}  {"F1%":>6s}  {"AUC%":>7s}  {"N":>6s}'
            sep = '-' * len(header)

            logger.info(f'\n─── Epoch {epoch+1}/{args.epochs}  |  Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  {epoch_time:.0f}s ───')
            logger.info(sep)
            logger.info(header)
            logger.info(sep)
            for name in all_val_names:
                r = epoch_results[name]
                logger.info(
                    f'{name:20s}  {r["loss"]:7.4f}  {r["acc"]:6.2f}  {r["precision"]:6.2f}  '
                    f'{r["recall"]:6.2f}  {r["f1"]:6.2f}  {r["auc"]:7.2f}  {r["count"]:>6d}'
                )
            logger.info(sep)

            # Averages
            ffpp_keys = [n for n in all_val_names if n.startswith('FF')]
            cd_keys   = [n for n in all_val_names if n not in ffpp_keys]
            for group_name, keys in [('FF++ AVG', ffpp_keys), ('Cross-Domain AVG', cd_keys)]:
                if keys:
                    avg_f1 = np.mean([epoch_results[k]['f1'] for k in keys])
                    avg_auc = np.nanmean([epoch_results[k]['auc'] for k in keys])
                    avg_acc = np.mean([epoch_results[k]['acc'] for k in keys])
                    logger.info(f'{group_name:20s}  Acc={avg_acc:.2f}%  F1={avg_f1:.2f}%  AUC={avg_auc:.2f}%')

            # Early stopping on primary (FF++) AUC
            primary_name = ffpp_keys[0] if ffpp_keys else all_val_names[0]
            primary_auc = epoch_results[primary_name]['auc']

            if primary_auc > best_auc:
                best_auc = primary_auc
                best_epoch = epoch + 1
                early_stop_counter = 0
                torch.save(model.state_dict(), args.save_path)
                logger.info(f'  ★ BEST: AUC={primary_auc:.2f}%  saved → {args.save_path}')
            else:
                early_stop_counter += 1
                logger.info(f'  No improvement ({early_stop_counter}/{patience})  best AUC={best_auc:.2f}% @ epoch {best_epoch}')

            if early_stop_counter >= patience:
                logger.info(f'\n>>> Early stopping triggered (patience={patience})')
                logger.info(f'>>> Best: AUC={best_auc:.2f}% @ epoch {best_epoch}')
                break

            val_str = f'AUC={primary_auc:.2f}%'

        else:
            # No validation data — just log train
            logger.info(f'Epoch {epoch+1}/{args.epochs}  |  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  {epoch_time:.0f}s')

    # ── Final ──────────────────────────────────────────────────
    logger.info('\n' + '=' * 70)
    logger.info(f'Training complete')
    logger.info(f'Best: AUC={best_auc:.2f}% @ epoch {best_epoch}')
    logger.info(f'Model: {args.save_path}')
    logger.info(f'Log:   {log_file}')
    logger.info('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT_M2F2Det_Bridge Phase-1 v2 (Frozen ViT)')
    parser.add_argument('--vit-ckpt', type=str,
                        default=r'E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth')
    parser.add_argument('--train-txt', type=str,
                        default='./dataset/data_2023/ffpp_train_split.txt')
    parser.add_argument('--data-root', type=str, default='./dataset')
    parser.add_argument('--save-path', type=str, default='./checkpoints/stage_1/bridge_v2_phase1.pth')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()
    main(args)
