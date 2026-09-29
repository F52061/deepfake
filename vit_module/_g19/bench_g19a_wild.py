# -*- coding: utf-8 -*-
"""Scratch benchmark: LoRA attach on the frozen ViT + step timing + anchor replication check."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys, time, json
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import cv2
cv2.setNumThreads(0)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vit_module.vit_adaptive_mattn_aps import vit_base_patch16_224

CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_1", "bridge_v2_phase1.pth")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
vit_sd = {k[4:]: v for k, v in sd.items() if k.startswith("vit.")}
print("[ckpt] vit keys", len(vit_sd))

vit = vit_base_patch16_224(pretrained=False, num_classes=0)
miss, unexp = vit.load_state_dict(vit_sd, strict=True)
print("[vit] loaded strict, params=%.1fM" % (sum(p.numel() for p in vit.parameters()) / 1e6))

# attn_drop values
ad = set()
for i, b in enumerate(vit.blocks):
    ad.add(getattr(b.attn, "attn_drop", None))
print("[vit] block attn_drop values:", ad)

from peft import LoraConfig, get_peft_model

for name, p in vit.named_parameters():
    p.requires_grad_(False)

TARGETS = ["blocks.%d.attn.qkv" % i for i in range(12)] + \
          ["blocks.%d.attn.proj" % i for i in range(12)]
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=TARGETS, bias="none")
t0 = time.time()
lora = get_peft_model(vit, cfg)
print("[lora] get_peft_model ok in %.1fs" % (time.time() - t0))
tp = [(n, p.numel()) for n, p in lora.named_parameters() if p.requires_grad]
n_tr = sum(n for _, n in tp)
print("[lora] trainable tensors=%d  trainable params=%d (%.3fM)" % (len(tp), n_tr, n_tr / 1e6))
print("[lora] sample:", tp[:4])
n_lora = sum(p.numel() for n, p in lora.named_parameters() if "lora_" in n)
print("[lora] lora_-only params=%d" % n_lora)

dev = torch.device("cuda:0")
lora.to(dev)
head = nn.Linear(768, 2).to(dev)
opt = torch.optim.AdamW([{"params": [p for p in lora.parameters() if p.requires_grad], "lr": 1e-4},
                         {"params": head.parameters(), "lr": 1e-3}])
x = torch.randn(32, 3, 224, 224, device=dev)
y = torch.randint(0, 2, (32,), device=dev)

def preprocess_for_vit(images):
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device).view(1, 3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device).view(1, 3, 1, 1)
    z = images * CLIP_STD + CLIP_MEAN
    z = torch.nn.functional.interpolate(z, size=(224, 224), mode="bilinear", align_corners=False)
    z = (z - 0.5) / 0.5
    return z

lora.train()
torch.cuda.synchronize()
for i in range(6):
    t = time.time()
    out = lora.forward_features(preprocess_for_vit(x))[:, 0, :]
    logits = head(out)
    loss = nn.functional.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    if i >= 2:
        print("[bench] step %d  %.3fs  loss=%.4f" % (i, time.time() - t, loss.item()))

lora.eval()
with torch.no_grad():
    f = lora.forward_features(preprocess_for_vit(x))[:, 0, :]
print("[bench] feat shape", tuple(f.shape), "mem=%.0f MiB" % (torch.cuda.max_memory_allocated() / 2**20))
print("[bench] torch threads", torch.get_num_threads())
