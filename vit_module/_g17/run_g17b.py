# -*- coding: utf-8 -*-
"""
G17b -- CNN local-detail (texture) feature extraction.

Goal: extract CNN intermediate-layer GAP features for 5300 images
(probe 3000 + multi 2300) using locally-cached ImageNet-pretrained weights
(NO downloads).  Produces cnn_feats.npz whose rows align by index with
probe_feats.npz['V'] (3000) + feats_multi.npz['V'] (2300).

Arms / layers (all GAP-pooled, AdaptiveAvgPool2d(1), fp32):
  densenet121   (main arm; mirrors llava/model/deepfake/encoder.py DenseNet_Deepfake
                 which consumes densenet121(...).features + avgpool):
      dense121_db2   = features.denseblock2 output   (512)
      dense121_db3   = features.denseblock3 output   (1024)
      dense121_final = features (denseblock4+norm5)  (1024)  <- repo "末层" control
  efficientnet_b4 (second arm):
      effnet_b4_blk5  = features[5] output (160)
      effnet_b4_blk6  = features[6] output (272)
      effnet_b4_final = features final   (1792)
  resnet18 (optional third arm):
      resnet18_layer3 = layer3 output (256)
      resnet18_layer4 = layer4 output (512)

Image pipeline (shared read with V, per run_g16):
  cv2.imread -> BGR2RGB -> cv2.resize(336) -> cv2.resize(224, INTER_LINEAR) -> ImageNet norm.
  CNN differs from V ONLY after the 336 stage: V uses CLIP normalize + _preprocess_for_vit
  (F.interpolate to 224); CNN uses cv2.resize 336->224 + ImageNet mean/std.

Label convention: y=1 real, y=0 fake (inherited, unchanged).

Validation (hard gate): the first 10 probe-test images must have path strings
byte-identical to vit_module/_g16/layer_feats.npz['paths'][2200:2210] (the G16
test segment, order-preserving).  Also report full path-set coverage.

Usage (project root):
  C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe vit_module/_g17/run_g17b.py \
      > vit_module/_g17/run_log_g17b.txt 2>&1
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["JOBLIB_NUM_THREADS"] = "2"

import re as _re
import subprocess
import sys
import time

import numpy as np


# --------------------------------------------------------------- GPU pick ----
def query_gpus():
    try:
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
    except Exception:
        return []


GPU_QUERY = query_gpus()
_cands = [g for g in GPU_QUERY if g[1] <= 100 and (g[2] - g[1]) >= 6000]
if _cands:
    GPU_PICK = min(_cands, key=lambda g: g[1])
    GPU_INDEX = GPU_PICK[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    MODE = "gpu"
else:
    GPU_PICK = None
    GPU_INDEX = -1
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    MODE = "cpu"

import torch
torch.set_num_threads(2)
import cv2
cv2.setNumThreads(0)
import torch.nn.functional as F
from torchvision.models import (
    densenet121, efficientnet_b4, resnet18,
    DenseNet121_Weights, EfficientNet_B4_Weights, ResNet18_Weights,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROBE_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_probe", "probe_feats.npz")
MULTI_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_tsne", "feats_multi.npz")
LAYER_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_g16", "layer_feats.npz")
OUT_NPZ = os.path.join(HERE, "cnn_feats.npz")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

IMG_SIZE = 336
CNN_SIZE = 224
BATCH = 16
SEED = 20260910
N_VAL = 10

FEAT_KEYS = [
    "dense121_db2", "dense121_db3", "dense121_final",
    "effnet_b4_blk5", "effnet_b4_blk6", "effnet_b4_final",
    "resnet18_layer3", "resnet18_layer4",
]

AUDIT = {"imgs_forward": 0, "batch_calls": 0, "read_fail": 0}
WEIGHT_SRC = {}


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


# ------------------------------------------------------------- pipeline ----
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        AUDIT["read_fail"] += 1
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_336(rgb):
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def to_cnn_batch(rgb336_list):
    """336 uint8 RGB list -> [B,3,224,224] float32 tensor, ImageNet normalized."""
    tensors = []
    for rgb in rgb336_list:
        img224 = cv2.resize(rgb, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img224.astype(np.float32)).permute(2, 0, 1).div_(255.0)
        t = (t - MEAN_T) / STD_T
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def make_hook(cap, key):
    def hk(module, inp, out):
        cap[key] = out
    return hk


def gap(t):
    return F.adaptive_avg_pool2d(t, 1).flatten(1).float().cpu().numpy()


# ---------------------------------------------------------- weights load ----
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

# torchvision DenseNet checkpoints store old-style keys ('norm.1', 'conv.1', ...);
# torchvision 0.17 remaps them to 'norm1', 'conv1', ... in its own _load_state_dict.
_DENSENET_PATTERN = _re.compile(
    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
)


def _remap_densenet(state_dict):
    out = {k: v for k, v in state_dict.items()}
    for key in list(out.keys()):
        res = _DENSENET_PATTERN.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            out[new_key] = out[key]
            del out[key]
    return out


def build_model(kind, cache_name):
    """Build a CNN with ImageNet weights, hitting the local cache; torch.load fallback."""
    cache_path = os.path.join(CACHE_DIR, cache_name)
    try:
        if kind == "densenet121":
            m = densenet121(weights=DenseNet121_Weights.DEFAULT)
        elif kind == "efficientnet_b4":
            m = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        elif kind == "resnet18":
            m = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(kind)
        return m, f"weights-enum cache hit ({cache_name})"
    except Exception as e1:
        sd = torch.load(cache_path, map_location="cpu")
        if kind == "densenet121":
            sd = _remap_densenet(sd)
            m = densenet121(weights=None)
        elif kind == "efficientnet_b4":
            m = efficientnet_b4(weights=None)
        elif kind == "resnet18":
            m = resnet18(weights=None)
        else:
            raise ValueError(kind)
        m.load_state_dict(sd)
        return m, f"torch.load fallback ({cache_name}) after enum err {type(e1).__name__}"


# ---------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dev_name = (torch.cuda.get_device_name(0)
                if (MODE == "gpu" and torch.cuda.is_available()) else "cpu")
    device = (torch.device("cuda:0")
              if (MODE == "gpu" and torch.cuda.is_available()) else torch.device("cpu"))
    print(f"[gpu] query={GPU_QUERY} pick={GPU_PICK} mode={MODE} device={dev_name} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[torch] version={torch.__version__}", flush=True)

    # -------------------------------------------------------- build models --
    dmodel, WEIGHT_SRC["densenet121"] = build_model("densenet121", "densenet121-a639ec97.pth")
    emodel, WEIGHT_SRC["efficientnet_b4"] = build_model("efficientnet_b4",
                                                        "efficientnet_b4_rwightman-23ab8bcd.pth")
    rmodel, WEIGHT_SRC["resnet18"] = build_model("resnet18", "resnet18-f37072fd.pth")

    for mdl in (dmodel, emodel, rmodel):
        mdl.to(device).eval()
        for p in mdl.parameters():
            p.requires_grad_(False)

    dcap, ecap, rcap = {}, {}, {}
    dmodel.features.denseblock2.register_forward_hook(make_hook(dcap, "denseblock2"))
    dmodel.features.denseblock3.register_forward_hook(make_hook(dcap, "denseblock3"))
    emodel.features[5].register_forward_hook(make_hook(ecap, "blk5"))
    emodel.features[6].register_forward_hook(make_hook(ecap, "blk6"))
    rmodel.layer3.register_forward_hook(make_hook(rcap, "layer3"))
    rmodel.layer4.register_forward_hook(make_hook(rcap, "layer4"))

    print("[weight] densenet121 : " + WEIGHT_SRC["densenet121"], flush=True)
    print("[weight] efficientnet_b4: " + WEIGHT_SRC["efficientnet_b4"], flush=True)
    print("[weight] resnet18   : " + WEIGHT_SRC["resnet18"], flush=True)

    # -------------------------------------------------------- load data ----
    p = np.load(PROBE_NPZ, allow_pickle=True)
    paths_p = np.array([str(s) for s in p["paths"]], dtype=object)
    vids_p = np.array([str(s) for s in p["vids"]], dtype=object)
    y_p = p["y"].astype(np.int64)
    tr_mask = p["train_mask"].astype(bool)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(~tr_mask)[0]
    assert (len(tr_idx), len(te_idx)) == (2200, 800), (len(tr_idx), len(te_idx))

    m = np.load(MULTI_NPZ, allow_pickle=True)
    paths_m = np.array([str(s) for s in m["path"]], dtype=object)
    vids_m = np.array([str(s) for s in m["vid"]], dtype=object)
    y_m = m["y"].astype(np.int64)
    dom_m = np.array([str(s) for s in m["domain"]], dtype=object)

    # --------------------------------------------- validation (hard gate) ----
    l = np.load(LAYER_NPZ, allow_pickle=True)
    paths_l = np.array([str(s) for s in l["paths"]], dtype=object)
    lset = set(paths_l.tolist())

    # order-preserving one-to-one check on the first 10 probe-test images:
    # G16 layer_feats rows 2200..2999 are the 800 probe-test images in te_idx order.
    te_order_ok = bool(np.array_equal(paths_l[2200:3000], paths_p[te_idx]))
    val_ids = te_idx[:N_VAL]
    val_exact = [bool(paths_l[2200 + k] == paths_p[te_idx[k]]) for k in range(N_VAL)]
    n_val_exact = int(sum(val_exact))

    read_ok = []
    for vp in paths_p[val_ids]:
        img = read_rgb(vp)
        read_ok.append(img is not None and img.ndim == 3 and img.shape[2] == 3)

    n_probe_match = int(sum(1 for vp in paths_p if vp in lset))
    n_multi_match = int(sum(1 for vp in paths_m if vp in lset))

    print(f"[validate] te_order(800 probe-test vs layer_feats[2200:3000]) = {te_order_ok}", flush=True)
    print(f"[validate] probe-test 10 order-preserving exact path match: {n_val_exact}/{N_VAL}", flush=True)
    for k, i in enumerate(val_ids):
        print(f"   [{k}] probe_idx={i} match={int(val_exact[k])} read_ok={int(read_ok[k])} "
              f"{paths_p[i]}", flush=True)
    print(f"[validate] coverage vs layer_feats.npz: probe {n_probe_match}/3000, "
          f"multi {n_multi_match}/2300 (expect 3000/1500; multi 'ffpp' 800 absent from G16)", flush=True)

    gate_ok = (n_val_exact == N_VAL) and all(read_ok)
    if not gate_ok:
        _lines = [
            "=" * 90,
            "G17B PRECHECK FAILED (stop): probe-test 10 path strings not all byte-identical "
            "to layer_feats.npz test segment (or unreadable).",
            f"G17B_VAL_PATH_MATCH={n_val_exact}/{N_VAL} (gate {N_VAL}/{N_VAL})",
            f"G17B_TE_ORDER={te_order_ok}",
            f"G17B_READ_OK={int(sum(read_ok))}/{N_VAL}",
            f"wall={time.time()-t0:.1f}s",
            "=" * 90,
        ]
        print("\n".join(_lines), flush=True)
        sys.exit(0)

    # ----------------------------------------- build unified sample order ----
    samples = []  # (path, vid, y, domain, split, source)
    for i in range(len(paths_p)):
        split = "train" if tr_mask[i] else "test"
        samples.append((paths_p[i], vids_p[i], int(y_p[i]), "ffpp_probe", split, "probe"))
    for i in range(len(paths_m)):
        d = dom_m[i]
        samples.append((paths_m[i], vids_m[i], int(y_m[i]), d, d, "multi"))
    n_total = len(samples)
    assert n_total == 5300, n_total

    paths_all = np.array([s[0] for s in samples], dtype=object)
    vids_all = np.array([s[1] for s in samples], dtype=object)
    y_all = np.array([s[2] for s in samples], dtype=np.int64)
    domain_all = np.array([s[3] for s in samples], dtype=object)
    split_all = np.array([s[4] for s in samples], dtype=object)
    source_all = np.array([s[5] for s in samples], dtype=object)

    # -------------------------------------------------------- extraction ----
    feats = {k: [] for k in FEAT_KEYS}
    t_ext = time.time()
    for i in range(0, n_total, BATCH):
        chunk = samples[i:i + BATCH]
        rgbs = []
        for (path, *_rest) in chunk:
            rgb = read_rgb(path)
            if rgb is None:
                rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            rgbs.append(to_336(rgb))
        x = to_cnn_batch(rgbs).to(device)
        AUDIT["batch_calls"] += 1
        AUDIT["imgs_forward"] += len(chunk)
        with torch.no_grad():
            dfinal = dmodel.features(x)
            feats["dense121_db2"].append(gap(dcap["denseblock2"]))
            feats["dense121_db3"].append(gap(dcap["denseblock3"]))
            feats["dense121_final"].append(gap(dfinal))

            efinal = emodel.features(x)
            feats["effnet_b4_blk5"].append(gap(ecap["blk5"]))
            feats["effnet_b4_blk6"].append(gap(ecap["blk6"]))
            feats["effnet_b4_final"].append(gap(efinal))

            _ = rmodel(x)
            feats["resnet18_layer3"].append(gap(rcap["layer3"]))
            feats["resnet18_layer4"].append(gap(rcap["layer4"]))
        if (i // BATCH) % 20 == 0 or i + len(chunk) >= n_total:
            print(f"  [extract] {i + len(chunk)}/{n_total}  "
                  f"imgs_fwd={AUDIT['imgs_forward']} wall={time.time()-t_ext:.0f}s", flush=True)
    wall_ext = time.time() - t_ext

    feats = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in feats.items()}
    dims = {}
    for k in FEAT_KEYS:
        assert feats[k].shape[0] == n_total, (k, feats[k].shape)
        dims[k] = int(feats[k].shape[1])

    # ------------------------------------------------------------ save ----
    np.savez(OUT_NPZ,
             **feats,
             y=y_all, paths=paths_all, vids=vids_all,
             domain=domain_all, split=split_all, source=source_all)
    print(f"[save] cnn_feats.npz -> {OUT_NPZ}", flush=True)

    # ---------------------------------------------------------- audit ----
    peak_smi = float("nan")
    try:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            peak_smi = float((total - free) / (1024.0 ** 2))
    except Exception:
        pass
    peak_torch = 0.0
    try:
        peak_torch = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)) \
            if device.type == "cuda" else 0.0
    except Exception:
        pass
    wall = time.time() - t0

    print("=" * 90, flush=True)
    print("G17B MACHINE_BLOCK", flush=True)
    print(f"G17B_MODE={MODE}", flush=True)
    print(f"G17B_GPU_INDEX={GPU_INDEX}", flush=True)
    print(f"G17B_DEVICE={dev_name}", flush=True)
    print(f"G17B_GPU_QUERY={GPU_QUERY}", flush=True)
    print(f"G17B_TORCH={torch.__version__}", flush=True)
    print(f"G17B_BATCH={BATCH}", flush=True)
    print(f"G17B_N_IMGS={n_total}", flush=True)
    print(f"G17B_FORWARDS={AUDIT['imgs_forward']}", flush=True)
    print(f"G17B_BATCH_CALLS={AUDIT['batch_calls']}", flush=True)
    print(f"G17B_READ_FAIL={AUDIT['read_fail']}", flush=True)
    print(f"G17B_PEAK_TORCH_MIB={fmt(peak_torch, 0)}", flush=True)
    print(f"G17B_PEAK_SMI_MIB={fmt(peak_smi, 0)}", flush=True)
    print(f"G17B_WALL_S={fmt(wall, 1)}", flush=True)
    print(f"G17B_WALL_EXT_S={fmt(wall_ext, 1)}", flush=True)
    print(f"G17B_VAL_PATH_MATCH={n_val_exact}/{N_VAL}", flush=True)
    print(f"G17B_TE_ORDER={te_order_ok}", flush=True)
    print(f"G17B_PROBE_COVERAGE={n_probe_match}/3000", flush=True)
    print(f"G17B_MULTI_COVERAGE={n_multi_match}/2300", flush=True)
    print("G17B_FEATURES:", flush=True)
    for k in FEAT_KEYS:
        print(f"  {k}: dim={dims[k]} pooling=GAP", flush=True)
    print(f"G17B_WEIGHT_SRC={WEIGHT_SRC}", flush=True)
    print(f"G17B_PIPELINE=cv2.imread->BGR2RGB->resize336->cv2.resize224(INTER_LINEAR)->ImageNet_norm", flush=True)
    print(f"G17B_ALL_5300_OK={'YES' if AUDIT['read_fail'] == 0 else 'NO'}", flush=True)
    print("=" * 90, flush=True)
    print(f"[done] wall={wall:.1f}s forwards={AUDIT['imgs_forward']} batches={AUDIT['batch_calls']} "
          f"read_fail={AUDIT['read_fail']} peak_torch={peak_torch:.0f}MiB", flush=True)


if __name__ == "__main__":
    main()
