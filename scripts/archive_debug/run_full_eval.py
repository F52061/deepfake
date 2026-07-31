"""
Full evaluation runner — evaluates cosine model on ALL datasets sequentially.
Outputs results to console AND saves JSON.

Usage:
    python vit_module/run_full_eval.py
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from vit_module.eval_all_datasets import (
    build_cosine_model, EvalDataset, evaluate
)
from torch.utils.data import DataLoader

# Config
CHECKPOINT = './vit_module/vit_m2f2_phase1.pth'
DATA_ROOT = './dataset'
BATCH_SIZE = 32
OUTPUT = './outputs/cosine_full_results.json'
CLIP_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'clip-vit-large-patch14-336')

# All datasets organized by category
ALL_DATASETS = {
    'FF++': ['ffpp_test_split', 'ffpp_test_split_c0'],
    'Celeb-DF': ['CD1_test', 'CD2_test'],
    'DFD/DFR': ['DFD_test', 'DFR_test'],
    'DFDC': ['dfdc_test_lip', 'dfdcp_test'],
    'FFIW': ['FFIW_test'],
    'WildDeepfake': ['wild_test'],
    'Diffusion': ['diff_test', 'diff_fe_test', 'diff_fs_test',
                  'diff_i2i_test', 'diff_real_test', 'diff_t2i_test'],
}

device = torch.device('cuda:0')
print('=' * 70)
print('Building cosine model...')
print('=' * 70)
model = build_cosine_model(device, CLIP_LOCAL)

print(f'Loading checkpoint: {CHECKPOINT}')
state = torch.load(CHECKPOINT, map_location='cpu')
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Missing: {len(missing)} | Unexpected: {len(unexpected)}')
model.eval()

txt_dir = os.path.join(DATA_ROOT, 'data_2023')
all_results = {}

for category, names in ALL_DATASETS.items():
    for name in names:
        txt_path = os.path.join(txt_dir, name + '.txt')
        if not os.path.exists(txt_path):
            print(f'SKIP: {txt_path}')
            continue
        ds = EvalDataset(txt_path, normalize_type='clip')
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f'\n[{category}] {name} ({len(ds)} samples)...', flush=True)
        result = evaluate(model, dl, device, model_type='vit_cosine')
        result['category'] = category
        result['dataset'] = name
        all_results[name] = result
        print(f'  AUC={result["auc"]:.2f}%  Acc={result["acc"]:.2f}%  F1={result["f1"]:.2f}%  EER={result["eer"]:.2f}%')
        gc.collect()
        torch.cuda.empty_cache()

# Summary
print('\n\n' + '=' * 90)
print('FINAL SUMMARY — VIT COSINE MODEL')
print('=' * 90)
header = f'{"Dataset":<28s} {"N":>7s} {"AUC%":>8s} {"Acc%":>7s} {"F1%":>7s} {"EER%":>7s}'
print(header)
print('-' * len(header))

cat_vals = []
cur_cat = None
for category, names in ALL_DATASETS.items():
    cat_vals = []
    print(f'  [{category}]')
    for name in names:
        r = all_results.get(name)
        if r is None or r.get('n', 0) == 0:
            continue
        print(f'  {name:<26s} {r["n"]:>7d} {r["auc"]:7.2f}% {r["acc"]:6.2f}% {r["f1"]:6.2f}% {r["eer"]:6.2f}%')
        cat_vals.append(r)
    if cat_vals:
        avg_auc = sum(x['auc'] for x in cat_vals if not str(x['auc']) == 'nan') / max(len([x for x in cat_vals if not str(x['auc']) == 'nan']), 1)
        avg_acc = sum(x['acc'] for x in cat_vals) / len(cat_vals)
        print(f'  {"  ── avg ──":<28s} {"":>7s} {avg_auc:7.2f}% {avg_acc:6.2f}%')

all_vals = [r for r in all_results.values() if r.get('n', 0) > 0]
if all_vals:
    valid = [x for x in all_vals if not str(x.get('auc', 'nan')) == 'nan']
    o_auc = sum(x['auc'] for x in valid) / max(len(valid), 1)
    o_acc = sum(x['acc'] for x in all_vals) / len(all_vals)
    o_f1 = sum(x['f1'] for x in all_vals) / len(all_vals)
    o_eer = sum(x.get('eer', 0) for x in valid) / max(len(valid), 1)
    print('-' * len(header))
    print(f'  {"OVERALL AVERAGE":<28s} {"":>7s} {o_auc:7.2f}% {o_acc:6.2f}% {o_f1:6.2f}% {o_eer:6.2f}%')
print('-' * len(header))

# Save
with open(OUTPUT, 'w') as f:
    json.dump({k: {kk: (float(vv) if isinstance(vv, (float, int)) else vv) for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)
print(f'\nResults saved to: {OUTPUT}')
print(f'Evaluated {len(all_vals)} datasets.')
