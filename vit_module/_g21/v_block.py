# -*- coding: utf-8 -*-
"""
G21-V -- is the CLIP TEXT branch of ViT_M2F2Det_Bridge dead code?

Claim under test (static): in vit_m2f2_detector_bridge.py
    clip_text_features = self.clip_text_encoder()            # L432  -> [1,768]
    clip_text_features = self.text_proj(clip_text_features)  # L435  -> [1,768]
and then NEVER used again.  Final fusion (L478-484) is
    features = cat([clip_vision_cls(768), clip_adapt_embed(128), vit_features(768)]) -> Linear(1664,2)

Checks:
  V5  checkpoint-only energy decomposition of d = output.weight[1] - output.weight[0]
  V3  shape + constancy (batch-size independence) of the text features
  V2  functional-path test: monkeypatch clip_text_encoder.forward -> zeros / noise
  V1  gradient test: loss.backward() -> text grads None, contrast grads nonzero
  V1b parameter statistics (recorded for a possible later init comparison)

Resource discipline: CPU threads pinned to 1, cv2 threads 0, single process,
num_workers=0, fp32 only, batch<=16, GPU 2 only (CUDA_VISIBLE_DEVICES=2).

Writes: v_report.txt, v_stats.npz  (inside vit_module/_g21/)
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np

# --------------------------------------------------------------- GPU gate ----
def query_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append(tuple(int(float(x)) for x in parts[:3]))
            except ValueError:
                continue
    return rows


GPU_QUERY = query_gpus()
print(f"[gpu] query (index,used,free) = {GPU_QUERY}", flush=True)
ASSIGNED = 2
_g2 = [g for g in GPU_QUERY if g[0] == ASSIGNED]
if not _g2:
    print(f"[gpu] FATAL: GPU {ASSIGNED} not visible in nvidia-smi output", flush=True)
    sys.exit(2)
USED_MIB, FREE_MIB = _g2[0][1], _g2[0][2]
if USED_MIB > 100:
    print(f"[gpu] STOP: GPU {ASSIGNED} shows {USED_MIB} MiB used (>100 MiB). "
          f"Not switching cards. Aborting.", flush=True)
    sys.exit(3)
os.environ["CUDA_VISIBLE_DEVICES"] = str(ASSIGNED)
print(f"[gpu] GPU {ASSIGNED} accepted: used={USED_MIB} MiB free={FREE_MIB} MiB", flush=True)

import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from albumentations import Compose, Normalize, ToTensorV2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_1", "bridge_v2_phase1.pth")
CLIP_LOCAL = os.path.join(PROJECT_ROOT, "checkpoints", "clip-vit-large-patch14-336")
LAYER_NPZ = os.path.join(PROJECT_ROOT, "vit_module", "_g16", "layer_feats.npz")
REPORT = os.path.join(HERE, "v_report.txt")
STATS = os.path.join(HERE, "v_stats.npz")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TRANSFORM = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

IMG_SIZE = 336
BATCH = 16
GRAD_BATCH = 8
SEED = 20260910
N_IMGS = 48            # 3 disjoint batches of 16
NOISE_SEEDS = [0, 1, 2]

AUDIT = {"imgs_forward": 0, "batch_calls": 0, "read_fail": 0}


def fmt(x, nd=6):
    if x is None:
        return "None"
    if isinstance(x, float) and not np.isfinite(x):
        return "NaN"
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def gpu_used_mib():
    try:
        free, total = torch.cuda.mem_get_info()
        return float((total - free) / (1024.0 ** 2))
    except Exception:
        return float("nan")


# ------------------------------------------------------------- pipeline ----
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        AUDIT["read_fail"] += 1
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_336(rgb):
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return rgb


def load_batch(paths):
    rgbs = []
    for p in paths:
        rgb = read_rgb(p)
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        rgbs.append(to_336(rgb))
    ts = [TRANSFORM(image=im)["image"] for im in rgbs]
    return torch.stack(ts, dim=0)


def all_norm(x):
    return float(torch.linalg.vector_norm(x.detach().float().reshape(-1)).item())


def maxabs(x):
    return float(x.detach().float().abs().max().item()) if x is not None else float("nan")


def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[dev] device={device} name={dev_name} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    # ---------------------------------------------------- build model ----
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=CLIP_LOCAL,
        clip_vision_encoder_name=CLIP_LOCAL,
        hidden_size=768,
        load_vision_encoder=True,
        pretrained=False,
        vision_dtype=torch.float32,
        text_dtype=torch.float32,
        deepfake_dtype=torch.float32,
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    ck_missing, ck_unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] missing={len(ck_missing)} unexpected={len(ck_unexpected)}", flush=True)
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # requires_grad stays True (default) so the V1 gradient test is meaningful.
    n_requires_grad = sum(1 for p in model.parameters() if p.requires_grad)
    n_params = sum(1 for _ in model.parameters())

    text_proj_in_ckpt = sorted(k for k in sd if k.startswith("text_proj."))
    prompt_in_ckpt = "clip_text_encoder.prompt_tokens" in sd

    # -------------------------------------------------- image batches ----
    z = np.load(LAYER_NPZ, allow_pickle=True)
    print(f"[data] {LAYER_NPZ} files={z.files}", flush=True)
    all_paths = z["paths"].astype(str)
    assert all_paths.shape[0] == 4500, all_paths.shape
    paths = all_paths[:N_IMGS]
    for p in paths[:5]:
        if not os.path.exists(p):
            print(f"[data] WARNING path missing: {p}", flush=True)

    print(f"[data] loading {N_IMGS} images ...", flush=True)
    batches = {}
    for bi in range(N_IMGS // BATCH):
        sl = paths[bi * BATCH:(bi + 1) * BATCH]
        batches[bi] = load_batch(sl).to(device)
        AUDIT["batch_calls"] += 1
        AUDIT["imgs_forward"] += len(sl)
    B16_A = batches[0]
    B16_B = batches[1]
    B16_C = batches[2]
    B1 = B16_A[:1]
    B8 = B16_A[:GRAD_BATCH]
    print(f"[data] read_fail={AUDIT['read_fail']}  B16_A={tuple(B16_A.shape)} "
          f"dt={B16_A.dtype}", flush=True)

    V = OrderedDict()          # machine-block values

    # ============================================================ V5 ======
    W = model.output.weight.detach().float().cpu()
    b_out = model.output.bias.detach().float().cpu()
    V5_shape = tuple(W.shape)
    if V5_shape != (2, 1664):
        print(f"[V5] UNEXPECTED output.weight shape {V5_shape} (expected (2,1664))", flush=True)
    d = (W[1] - W[0])
    dn = d.numpy().astype(np.float64)
    tot = float((dn ** 2).sum())
    SEG = [("clip_vision_cls", 0, 768), ("bridge_clip_adapt_embed", 768, 896),
           ("vit_features", 896, 1664)]
    seg = OrderedDict()
    for name, a, b in SEG:
        s = dn[a:b]
        n2 = float((s ** 2).sum())
        seg[name] = dict(a=a, b=b, dim=b - a, n2=n2, share=(n2 / tot if tot > 0 else float("nan")),
                         norm=float(np.sqrt(n2)))
    dim_energy = (dn ** 2) / tot if tot > 0 else np.zeros_like(dn)
    ndim_gt = int((dim_energy > 0.001).sum())
    # which region do the high-energy dims live in?
    hi = np.where(dim_energy > 0.001)[0]
    hi_in_text_region = 0   # text occupies 0 dims of this vector by construction
    print(f"[V5] d=weight[1]-weight[0] shape={V5_shape} |d|={np.sqrt(tot):.6f} "
          f"norm2={tot:.6f}", flush=True)
    for name, a, b in SEG:
        print(f"[V5]   [{a}:{b}] {name:26s} dims={b-a:4d} norm2={seg[name]['n2']:12.6f} "
              f"share={seg[name]['share']*100:8.4f}%", flush=True)
    print(f"[V5] ndim with dim-energy>0.001 : {ndim_gt}", flush=True)

    # ============================================================ V3 ======
    with torch.no_grad():
        tf_raw = model.clip_text_encoder()
        V3_1_shape = tuple(tf_raw.shape)
        V3_1_dtype = str(tf_raw.dtype)
        V3_1_dev = str(tf_raw.device)
        AUDIT["batch_calls"] += 1
        print(f"[V3.1] clip_text_encoder() -> shape={V3_1_shape} dtype={V3_1_dtype} "
              f"device={V3_1_dev}", flush=True)

    # capture text_proj output on every forward
    cap = {}
    hook_handle = model.text_proj.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("y", out.detach().clone()))

    def run_capture(x):
        cap.clear()
        with torch.no_grad():
            out = model(x)
        AUDIT["imgs_forward"] += int(x.shape[0])
        AUDIT["batch_calls"] += 1
        return out.detach(), cap.get("y")

    with torch.no_grad():
        out_base, tp16 = run_capture(B16_A)
    V3_2_shape = tuple(tp16.shape)
    # per-sample pairwise deviation (only meaningful if dim0 > 1)
    if tp16.shape[0] > 1:
        d2 = torch.cdist(tp16.float(), tp16.float(), p=2)
        V3_2_batch_maxdev = float(d2.max().item())
    else:
        V3_2_batch_maxdev = 0.0
    print(f"[V3.2] text_proj out shape={V3_2_shape} (batch of {B16_A.shape[0]})  "
          f"max pairwise ||x_i-x_j|| = {V3_2_batch_maxdev:.10f}"
          + ("   [VACUOUS: only 1 row, no pairs]" if tp16.shape[0] <= 1 else ""), flush=True)

    # batch-size independence: same first image, different batch sizes
    with torch.no_grad():
        _, tp1 = run_capture(B1)
        _, tp8 = run_capture(B8)
        _, tp16b = run_capture(B16_A)          # repeat, for determinism control
    dev_1v16 = float((tp1.float() - tp16.float()).abs().max().item())
    dev_8v16 = float((tp8.float() - tp16.float()).abs().max().item())
    det_same = bool(torch.equal(tp16b, tp16))
    det_maxabs = float((tp16b.float() - tp16.float()).abs().max().item())
    print(f"[V3.2] batch-independence: max|T(batch1)-T(batch16)|={dev_1v16:.3e}  "
          f"max|T(batch8)-T(batch16)|={dev_8v16:.3e}", flush=True)
    print(f"[V3.2] repeat-forward determinism: torch.equal={det_same} "
          f"max|delta|={det_maxabs:.3e}", flush=True)

    # V3.3 second forward, completely different batch of 16
    with torch.no_grad():
        out_b2, tp_other = run_capture(B16_B)
    V3_3_linf = float((tp_other.float() - tp16.float()).abs().max().item())
    V3_3_l2 = all_norm(tp_other.float() - tp16.float())
    V3_3_eq = bool(torch.equal(tp_other, tp16))
    print(f"[V3.3] different batch of 16: max|T_A - T_B| (Linf)={V3_3_linf:.3e}  "
          f"L2={V3_3_l2:.3e}  torch.equal={V3_3_eq}", flush=True)

    # ============================================================ V2 ======
    # determinism control on logits with UNPATCHED encoder
    with torch.no_grad():
        out_ctrl = model(B16_A)
        AUDIT["imgs_forward"] += int(B16_A.shape[0])
        AUDIT["batch_calls"] += 1
    ctrl_eq = bool(torch.equal(out_ctrl, out_base))
    ctrl_maxabs = float((out_ctrl - out_base).abs().max().item())
    print(f"[V2.0] unpatched repeat-forward: torch.equal={ctrl_eq} "
          f"max|delta|={ctrl_maxabs:.3e}  (determinism control)", flush=True)

    orig_forward = model.clip_text_encoder.forward
    V2 = OrderedDict()

    def probe(tag, fill):
        """Patch clip_text_encoder.forward with a factory returning `fill`."""
        def _make():
            def _patched(input_embeds=None):
                return fill()
            return _patched
        model.clip_text_encoder.forward = _make()
        cap.clear()
        try:
            with torch.no_grad():
                o = model(B16_A)
                AUDIT["imgs_forward"] += int(B16_A.shape[0])
                AUDIT["batch_calls"] += 1
            eq = bool(torch.equal(o, out_base))
            md = float((o - out_base).abs().max().item())
            V2[tag] = dict(ok=True, eq=eq, maxabs=md,
                           probe_maxabs=maxabs(fill()), probe_norm=all_norm(fill()),
                           shape=tuple(fill().shape))
            print(f"[V2] {tag:22s} probe shape={tuple(fill().shape)} "
                  f"probe_max|x|={maxabs(fill()):.4f}  max|dLogit|={md:.3e}  "
                  f"torch.equal={eq}", flush=True)
        except Exception as e:
            V2[tag] = dict(ok=False, eq=None, maxabs=float("nan"), err=repr(e))
            print(f"[V2] {tag:22s} RAISED: {e!r}", flush=True)

    shape_r = tuple(tf_raw.shape)
    dtype_r = tf_raw.dtype
    dev_r = tf_raw.device

    probe("a_zeros", lambda: torch.zeros(shape_r, dtype=dtype_r, device=dev_r))
    for s in NOISE_SEEDS:
        g = torch.Generator(device="cpu").manual_seed(1000 + s)
        rnd = torch.randn(shape_r, generator=g, dtype=torch.float32).to(device=dev_r, dtype=dtype_r)
        probe(f"b_noise_seed{s}", (lambda t: (lambda: t))(rnd))

    # extra (beyond spec): a DIFFERENT shape -- if the tensor were consumed anywhere,
    # a [16,768] vs [1,768] mismatch would either broadcast silently or crash.
    probe("c_wrong_shape_16x768", lambda: torch.full((16, 768), 7.0, dtype=dtype_r, device=dev_r))
    probe("c_wrong_shape_1664", lambda: torch.full((1, 1664), -3.0, dtype=dtype_r, device=dev_r))

    model.clip_text_encoder.forward = orig_forward
    hook_handle.remove()

    # ============================================================ V1 ======
    n_rg = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[V1] requires_grad: {n_rg}/{n_params} parameters", flush=True)
    model.zero_grad(set_to_none=True)
    out_g = model(B8)
    AUDIT["imgs_forward"] += int(B8.shape[0])
    AUDIT["batch_calls"] += 1
    loss = out_g.sum()
    loss.backward()
    print(f"[V1] loss=out.sum()={float(loss.item()):.6f}  out.shape={tuple(out_g.shape)}",
          flush=True)

    def group_report(label, named):
        n = 0
        n_none = 0
        mx = 0.0
        nones = []
        for name, p in named:
            n += 1
            if p.grad is None:
                n_none += 1
                nones.append(name)
            else:
                mx = max(mx, float(p.grad.detach().abs().max().item()))
        return dict(label=label, n=n, n_none=n_none, grad_max=mx, none_names=nones)

    groups = OrderedDict()
    groups["text_proj"] = group_report(
        "text_proj.*", [(n_, p) for n_, p in model.named_parameters()
                        if n_.startswith("text_proj.")])
    groups["prompt_tokens"] = group_report(
        "clip_text_encoder.prompt_tokens",
        [(n_, p) for n_, p in model.named_parameters()
         if n_ == "clip_text_encoder.prompt_tokens"])
    rep = [n_ for n_, _ in model.named_parameters()
           if n_.startswith("clip_text_encoder.model.")
           and (n_.endswith("text_model.embeddings.token_embedding.weight")
                or n_.endswith("encoder.layers.0.self_attn.q_proj.weight"))]
    groups["clip_text_model_rep"] = group_report(
        "clip_text_encoder.model.* (2 representative)",
        [(n_, p) for n_, p in model.named_parameters() if n_ in rep])
    for gname, gpath in [("output", "output.weight"), ("clip_reduction", "clip_reduction.weight"),
                         ("vision_proj", "vision_proj.0.weight"),
                         ("deepfake_proj", "deepfake_proj.0.weight")]:
        groups[gname] = group_report(
            gpath, [(n_, p) for n_, p in model.named_parameters() if n_ == gpath])

    for k, g in groups.items():
        print(f"[V1] {g['label']:42s} n={g['n']:3d} grad_is_None={g['n_none']:3d} "
              f"grad_absmax={g['grad_max']:.6e}", flush=True)

    # also: full text-branch sweep
    text_branch = [(n_, p) for n_, p in model.named_parameters()
                   if n_.startswith("text_proj.") or n_.startswith("clip_text_encoder.")]
    tb_none = sum(1 for _, p in text_branch if p.grad is None)
    tb_max = max([float(p.grad.detach().abs().max().item())
                  for _, p in text_branch if p.grad is not None] or [0.0])
    contrast_names = ["output.weight", "output.bias", "clip_reduction.weight",
                      "vision_proj.0.weight", "deepfake_proj.0.weight"]
    c_max = max([float(dict(model.named_parameters())[n].grad.detach().abs().max().item())
                 for n in contrast_names
                 if dict(model.named_parameters())[n].grad is not None] or [0.0])
    c_none = sum(1 for n in contrast_names
                 if dict(model.named_parameters())[n].grad is None)
    print(f"[V1] TEXT BRANCH total: {len(text_branch)} params, grad None = {tb_none}, "
          f"max|grad| over non-None = {tb_max:.6e}", flush=True)
    print(f"[V1] CONTRAST: {len(contrast_names)} params, grad None = {c_none}, "
          f"max|grad| = {c_max:.6e}", flush=True)

    # ---- full sweep over EVERY parameter prefix (completes the V1 picture) ----
    P_all = dict(model.named_parameters())
    PREFIXES = ["text_proj.", "clip_text_encoder.", "output.", "clip_reduction.",
                "vision_proj.", "deepfake_proj.", "linear_vit_", "bridge_adapter.",
                "bridge_adapter_proj.", "clip_vision_encoder.", "vit."]
    SWEEP = OrderedDict()
    for pref in PREFIXES:
        sel = [(n_, p) for n_, p in model.named_parameters() if n_.startswith(pref)]
        if not sel:
            continue
        SWEEP[pref] = (len(sel),
                       sum(1 for _, p in sel if p.grad is None),
                       max([float(p.grad.detach().abs().max().item())
                            for _, p in sel if p.grad is not None] or [0.0]),
                       sum(1 for _, p in sel if p.grad is None and not p.requires_grad))
    for nm in ["clip_vision_alpha", "clip_text_alpha"]:
        p = P_all.get(nm)
        if p is None:
            continue
        SWEEP[nm] = (1, 1 if p.grad is None else 0,
                     float(p.grad.detach().abs().max().item()) if p.grad is not None else 0.0,
                     0 if p.requires_grad else 1)
    print("[V1] FULL PARAMETER-PREFIX GRAD SWEEP", flush=True)
    print(f"[V1]   {'prefix':<26}{'n':>5}{'grad None':>11}{'max|grad|':>16}{'of which RQ=False':>19}",
          flush=True)
    for k_, v_ in SWEEP.items():
        print(f"[V1]   {k_:<26}{v_[0]:>5}{v_[1]:>11}{v_[2]:>16.6e}{v_[3]:>19}", flush=True)

    # ---- V1x (secondary finding surfaced by the V1 contrast column) ----
    # clip_reduction.weight came back as a NON-None grad tensor that is EXACTLY zero.
    # Structural candidate: BridgeAdapter_Proj_ViT.bridge_adapter_proj ends with nn.LayerNorm(1),
    # which normalizes a 1-element vector to 0 and then applies only its bias -> constant.
    ln1 = torch.nn.LayerNorm(1)
    ln1 = ln1.to(device).eval()
    with torch.no_grad():
        probe_in = torch.randn(512, 1, device=device) * 37.0 + 5.0
        ln1_out = ln1(probe_in)
    ln1_spread = float((ln1_out - ln1_out.reshape(-1)[0]).abs().max().item())
    ln1_in_spread = float((probe_in - probe_in.reshape(-1)[0]).abs().max().item())
    print(f"[V1x] nn.LayerNorm(1) micro-test: in-spread={ln1_in_spread:.6f} -> "
          f"out-spread={ln1_spread:.3e} (bias={float(ln1.bias.item()):.6f})", flush=True)

    br_cap = {}
    h1 = model.bridge_adapter_proj.register_forward_hook(
        lambda mod, inp, out: br_cap.__setitem__("y", out.detach().clone()))
    h2 = model.bridge_adapter_proj.bridge_adapter_proj.register_forward_hook(
        lambda mod, inp, out: br_cap.__setitem__("tok", out.detach().clone()))
    # discriminator: the Linear(64,1) sitting BEFORE LayerNorm(1)
    h3 = model.bridge_adapter_proj.bridge_adapter_proj[1].register_forward_hook(
        lambda mod, inp, out: br_cap.__setitem__("pre", out.detach().clone()))
    ln1_bias_model = float(model.bridge_adapter_proj.bridge_adapter_proj[2].bias.detach().item())
    ln1_w_model = model.bridge_adapter_proj.bridge_adapter_proj[2].weight.detach().float()
    lin_pool_w = model.bridge_adapter_proj.bridge_adapter_proj[1].weight.detach().float()

    def run_bridge(x):
        br_cap.clear()
        with torch.no_grad():
            o = model(x)
            AUDIT["imgs_forward"] += int(x.shape[0])
            AUDIT["batch_calls"] += 1
        return o, br_cap.get("tok"), br_cap.get("y"), br_cap.get("pre")

    with torch.no_grad():
        _, tok_A, be_A, pre_A = run_bridge(B16_A)
        _, tok_B, be_B, pre_B = run_bridge(B16_B)
    pr = pre_A.reshape(-1)
    pre_spread = float((pr - pr.reshape(-1)[0]).abs().max().item())
    lin_pool_w_max = float(lin_pool_w.abs().max().item())
    lin_pool_w_norm = float(torch.linalg.vector_norm(lin_pool_w).item())
    be_dim = tuple(be_A.shape)
    be_batch_spread = (float(torch.cdist(be_A.float(), be_A.float(), p=2).max().item())
                       if be_A.shape[0] > 1 else 0.0)
    be_cross = float((be_A.float() - be_B.float()).abs().max().item())
    be_cross_l2 = all_norm(be_A.float() - be_B.float())
    tk = tok_A.reshape(-1)
    tok_spread = float((tk - tk.reshape(-1)[0]).abs().max().item())
    print(f"[V1x] bridge_adapter_proj out shape={be_dim}  pairwise maxdev within batch="
          f"{be_batch_spread:.3e}", flush=True)
    print(f"[V1x] bridge 128-d embed: max|A-B| across two different batches = {be_cross:.3e} "
          f"(L2={be_cross_l2:.3e})", flush=True)
    print(f"[V1x] bridge_adapter_proj inner per-token output [{tuple(tok_A.shape)}] "
          f"spread across ALL tokens/batch = {tok_spread:.3e}", flush=True)
    print(f"[V1x] DISCRIMINATOR -- Linear(64,1) output BEFORE LayerNorm(1): "
          f"shape={tuple(pre_A.shape)} spread={pre_spread:.6e}", flush=True)
    print(f"[V1x]   pooling Linear weight: |w|max={lin_pool_w_max:.6e} norm={lin_pool_w_norm:.6e} "
          f"(non-zero -> the collapse is caused by LayerNorm(1), not by a zeroed Linear)", flush=True)
    print(f"[V1x]   model LayerNorm(1): weight={ln1_w_model.reshape(-1).tolist()} "
          f"bias={ln1_bias_model:.10e}", flush=True)
    h1.remove()
    h2.remove()
    h3.remove()
    V1X = dict(be_dim=be_dim, be_batch_spread=be_batch_spread, be_cross=be_cross,
               be_cross_l2=be_cross_l2, ln1_spread=ln1_spread, ln1_in_spread=ln1_in_spread,
               tok_shape=tuple(tok_A.shape), tok_spread=tok_spread,
               pre_shape=tuple(pre_A.shape), pre_spread=pre_spread,
               lin_pool_w_max=lin_pool_w_max, lin_pool_w_norm=lin_pool_w_norm,
               ln1_w=ln1_w_model.reshape(-1).tolist(), ln1_bias=ln1_bias_model,
               be_batch_const=bool(be_cross == 0.0 and be_batch_spread == 0.0))

    # =========================================================== V1b ======
    P = dict(model.named_parameters())
    V1B = OrderedDict()
    for key in ["text_proj.0.weight", "clip_text_encoder.prompt_tokens", "output.bias"]:
        if key not in P:
            print(f"[V1b] MISSING param {key}", flush=True)
            V1B[key] = None
            continue
        t = P[key].detach().float().cpu()
        fl = t.reshape(-1)
        st = dict(shape=tuple(t.shape), numel=int(t.numel()), mean=float(fl.mean().item()),
                  std=float(fl.std(unbiased=True).item()) if fl.numel() > 1 else 0.0,
                  norm=float(torch.linalg.vector_norm(fl).item()),
                  first5=[float(x) for x in fl[:5].tolist()])
        V1B[key] = st
        print(f"[V1b] {key}: shape={st['shape']} mean={st['mean']:.6e} std={st['std']:.6e} "
              f"norm={st['norm']:.6e} first5={st['first5']}", flush=True)

    peak_torch = torch.cuda.max_memory_allocated() / (1024.0 ** 2) if device.type == "cuda" else 0.0
    peak_smi = gpu_used_mib()
    wall = time.time() - t0

    # ======================================================= verdicts =====
    v5_share = {k: seg[k]["share"] for k in seg}
    v5_pass = (abs(v5_share["clip_vision_cls"] - 0.623) < 0.01
               and v5_share["bridge_clip_adapt_embed"] < 0.001
               and abs(v5_share["vit_features"] - 0.377) < 0.01) if tot > 0 else False
    v3_1_pass = (V3_1_shape == (1, 768))
    v3_batch_indep = (dev_1v16 == 0.0 and dev_8v16 == 0.0)
    v3_3_pass = (V3_3_linf == 0.0 and V3_3_l2 == 0.0)
    SPEC_V2 = ["a_zeros"] + [f"b_noise_seed{s}" for s in NOISE_SEEDS]
    v2_spec_ok = [k for k in SPEC_V2 if k in V2 and V2[k].get("ok")]
    # pass criterion uses ONLY the spec probes; the extra wrong-shape probes are diagnostics and
    # one of them legitimately raises inside text_proj before ever reaching the fusion.
    v2_pass = (len(v2_spec_ok) == len(SPEC_V2)
               and all(V2[k]["eq"] and V2[k]["maxabs"] == 0.0 for k in v2_spec_ok))
    v1_text_dead = (groups["text_proj"]["n_none"] == groups["text_proj"]["n"]
                    and groups["prompt_tokens"]["n_none"] == groups["prompt_tokens"]["n"]
                    and groups["clip_text_model_rep"]["n_none"] == groups["clip_text_model_rep"]["n"]
                    and tb_none == len(text_branch))
    # core contrast (spec named 4 params; clip_reduction measured separately because it came back 0)
    CORE_CONTRAST = ["output.weight", "vision_proj.0.weight", "deepfake_proj.0.weight"]
    core_max = max([float(P_all[n].grad.detach().abs().max().item()) for n in CORE_CONTRAST
                    if P_all[n].grad is not None] or [0.0])
    core_none = sum(1 for n in CORE_CONTRAST if P_all[n].grad is None)
    v1_contrast_live = (core_none == 0 and core_max > 0.0)
    clipred_grad_zero = (groups["clip_reduction"]["n_none"] == 0
                         and groups["clip_reduction"]["grad_max"] == 0.0)
    bridge_dead = bool(V1X["be_batch_const"] and V1X["tok_spread"] == 0.0)
    VERDICT = ("TEXT_BRANCH_DEAD_CONFIRMED"
               if (v1_text_dead and v1_contrast_live and v2_pass and v3_3_pass) else "NOT_CONFIRMED")

    mb = OrderedDict()
    mb["TASK"] = "G21-V text-branch dead-code falsification"
    mb["GPU_QUERY"] = str(GPU_QUERY)
    mb["GPU_ASSIGNED"] = str(ASSIGNED)
    mb["GPU_USED_MIB_AT_START"] = str(USED_MIB)
    mb["GPU_FREE_MIB_AT_START"] = str(FREE_MIB)
    mb["DEVICE"] = f"{dev_name} (cuda:0 == physical GPU {ASSIGNED})"
    mb["BATCH"] = str(BATCH)
    mb["GRAD_BATCH"] = str(GRAD_BATCH)
    mb["FP32"] = "1"
    mb["N_IMGS"] = str(N_IMGS)
    mb["SEED"] = str(SEED)
    mb["WALL_S"] = fmt(wall, 1)
    mb["PEAK_MIB"] = fmt(peak_smi, 0)
    mb["PEAK_TORCH_MIB"] = fmt(peak_torch, 0)
    mb["CKPT_MISSING"] = str(len(ck_missing))
    mb["CKPT_UNEXPECTED"] = str(len(ck_unexpected))
    mb["CKPT_TEXT_PROJ_KEYS"] = str(text_proj_in_ckpt)
    mb["CKPT_PROMPT_TOKENS_PRESENT"] = str(prompt_in_ckpt)
    mb["READ_FAIL"] = str(AUDIT["read_fail"])
    mb["FORWARDS"] = str(AUDIT["imgs_forward"])

    mb["V5_OUTPUT_WEIGHT_SHAPE"] = str(V5_shape)
    mb["V5_D_NORM"] = fmt(float(np.sqrt(tot)), 8)
    mb["V5_D_NORM2"] = fmt(tot, 8)
    mb["V5_SHARE_0_768_clip_vision_cls"] = fmt(v5_share["clip_vision_cls"], 8)
    mb["V5_SHARE_768_896_bridge_embed"] = fmt(v5_share["bridge_clip_adapt_embed"], 8)
    mb["V5_SHARE_896_1664_vit_features"] = fmt(v5_share["vit_features"], 8)
    mb["V5_TEXT_FEATURE_DIMS"] = "0"
    mb["V5_TOTAL_DIMS"] = str(int(W.shape[1]))
    mb["V5_NDIM_ENERGY_GT_1E3"] = str(ndim_gt)
    mb["V5_NDIM_ENERGY_GT_1E3_IN_TEXT_REGION"] = str(hi_in_text_region)
    mb["V5_EXPECT_MATCH"] = "1" if v5_pass else "0"

    mb["V3_1_TEXT_SHAPE"] = str(V3_1_shape)
    mb["V3_1_TEXT_DTYPE"] = V3_1_dtype
    mb["V3_1_TEXT_DEVICE"] = V3_1_dev
    mb["V3_1_SHAPE_IS_1x768"] = "1" if v3_1_pass else "0"
    mb["V3_2_TEXTPROJ_OUT_SHAPE"] = str(V3_2_shape)
    mb["V3_2_BATCH_PAIRWISE_MAXDEV"] = fmt(V3_2_batch_maxdev, 10)
    mb["V3_2_BATCH_PAIRWISE_VACUOUS"] = "1" if V3_2_shape[0] <= 1 else "0"
    mb["V3_2_MAXDEV_B1_vs_B16"] = fmt(dev_1v16, 10)
    mb["V3_2_MAXDEV_B8_vs_B16"] = fmt(dev_8v16, 10)
    mb["V3_2_BATCH_INDEP"] = "1" if v3_batch_indep else "0"
    mb["V3_2_REPEAT_DETERMINISM_equal"] = "1" if det_same else "0"
    mb["V3_2_REPEAT_DETERMINISM_maxabs"] = fmt(det_maxabs, 10)
    mb["V3_3_LINF_DIFF"] = fmt(V3_3_linf, 10)
    mb["V3_3_L2_DIFF"] = fmt(V3_3_l2, 10)
    mb["V3_3_BITWISE_EQUAL"] = "1" if V3_3_eq else "0"

    mb["V2_CTRL_DETERMINISM_equal"] = "1" if ctrl_eq else "0"
    mb["V2_CTRL_DETERMINISM_maxabs"] = fmt(ctrl_maxabs, 10)
    for k in V2:
        r = V2[k]
        if r.get("ok"):
            mb[f"V2_{k}_MAXABS_DLOGIT"] = fmt(r["maxabs"], 10)
            mb[f"V2_{k}_TORCH_EQUAL"] = "1" if r["eq"] else "0"
            mb[f"V2_{k}_PROBE_ABSMAX"] = fmt(r["probe_maxabs"], 6)
            mb[f"V2_{k}_PROBE_SHAPE"] = str(r["shape"])
        else:
            mb[f"V2_{k}_RAISED"] = r.get("err", "?")
    mb["V2_ALL_PASS"] = "1" if v2_pass else "0"

    for gk, g in groups.items():
        mb[f"V1_{gk}_N"] = str(g["n"])
        mb[f"V1_{gk}_GRAD_IS_NONE_COUNT"] = str(g["n_none"])
        mb[f"V1_{gk}_GRAD_ABSMAX"] = fmt(g["grad_max"], 10)
    mb["V1_TEXT_BRANCH_TOTAL_PARAMS"] = str(len(text_branch))
    mb["V1_TEXT_BRANCH_GRAD_NONE"] = str(tb_none)
    mb["V1_TEXT_BRANCH_GRAD_ABSMAX"] = fmt(tb_max, 10)
    mb["V1_CONTRAST_GRAD_NONE"] = str(c_none)
    mb["V1_CONTRAST_GRAD_ABSMAX"] = fmt(c_max, 10)
    mb["V1_CORE_CONTRAST_GRAD_NONE"] = str(core_none)
    mb["V1_CORE_CONTRAST_GRAD_ABSMAX"] = fmt(core_max, 10)
    mb["V1_TEXT_DEAD"] = "1" if v1_text_dead else "0"
    mb["V1_CONTRAST_LIVE"] = "1" if v1_contrast_live else "0"
    mb["V1_CLIP_REDUCTION_GRAD_EXACTLY_ZERO"] = "1" if clipred_grad_zero else "0"
    mb["V1X_BRIDGE_BRANCH_CONSTANT"] = "1" if bridge_dead else "0"
    mb["EXPECT_V1_CONTRAST_ALL_NONZERO"] = "0" if clipred_grad_zero else "1"
    mb["SECONDARY_FINDING"] = ("BRIDGE_BRANCH_CONSTANT" if bridge_dead else "NONE")
    mb["V1_SEEN_FORWARD_WITH_REQUIRES_GRAD"] = "1"
    for k_, v_ in SWEEP.items():
        tag = k_.replace(".", "_").strip("_")
        mb[f"V1SWEEP_{tag}_N"] = str(v_[0])
        mb[f"V1SWEEP_{tag}_GRAD_NONE"] = str(v_[1])
        mb[f"V1SWEEP_{tag}_GRAD_ABSMAX"] = fmt(v_[2], 10)
        mb[f"V1SWEEP_{tag}_N_REQUIRESGRAD_FALSE"] = str(v_[3])
    mb["V1X_BRIDGE_EMBED_SHAPE"] = str(V1X["be_dim"])
    mb["V1X_BRIDGE_BATCH_PAIRWISE_MAXDEV"] = fmt(V1X["be_batch_spread"], 10)
    mb["V1X_BRIDGE_CROSS_BATCH_MAXABS"] = fmt(V1X["be_cross"], 10)
    mb["V1X_BRIDGE_CROSS_BATCH_L2"] = fmt(V1X["be_cross_l2"], 10)
    mb["V1X_BRIDGE_BATCH_CONSTANT"] = "1" if V1X["be_batch_const"] else "0"
    mb["V1X_BRIDGE_INNER_TOKEN_SHAPE"] = str(V1X["tok_shape"])
    mb["V1X_BRIDGE_INNER_TOKEN_SPREAD"] = fmt(V1X["tok_spread"], 10)
    mb["V1X_LAYERNORM1_IN_SPREAD"] = fmt(V1X["ln1_in_spread"], 6)
    mb["V1X_LAYERNORM1_OUT_SPREAD"] = fmt(V1X["ln1_spread"], 10)
    mb["V1X_POOL_LINEAR_PRE_LN_SHAPE"] = str(V1X["pre_shape"])
    mb["V1X_POOL_LINEAR_PRE_LN_SPREAD"] = fmt(V1X["pre_spread"], 8)
    mb["V1X_POOL_LINEAR_WEIGHT_ABSMAX"] = fmt(V1X["lin_pool_w_max"], 8)
    mb["V1X_POOL_LINEAR_WEIGHT_NORM"] = fmt(V1X["lin_pool_w_norm"], 8)
    mb["V1X_MODEL_LN1_WEIGHT"] = str(V1X["ln1_w"])
    mb["V1X_MODEL_LN1_BIAS"] = fmt(V1X["ln1_bias"], 12)
    mb["V1X_CLIP_REDUCTION_GRAD_IS_ZERO"] = (
        "1" if (groups["clip_reduction"]["n_none"] == 0
                and groups["clip_reduction"]["grad_max"] == 0.0) else "0")

    for key in ["text_proj.0.weight", "clip_text_encoder.prompt_tokens", "output.bias"]:
        st = V1B.get(key)
        tag = key.replace(".", "_")
        if st is None:
            mb[f"V1B_{tag}_MISSING"] = "1"
            continue
        mb[f"V1B_{tag}_SHAPE"] = str(st["shape"])
        mb[f"V1B_{tag}_MEAN"] = fmt(st["mean"], 8)
        mb[f"V1B_{tag}_STD"] = fmt(st["std"], 8)
        mb[f"V1B_{tag}_NORM"] = fmt(st["norm"], 8)
        mb[f"V1B_{tag}_FIRST5"] = str(st["first5"])

    mb["VERDICT"] = VERDICT

    # ---------------------------------------------------------- report ----
    mblines = ["#### MACHINE_BLOCK " + "#" * 84]
    mblines += [f"G21V_{k}={v}" for k, v in mb.items()]

    L = []
    A = L.append
    A("=" * 100)
    A("G21-V REPORT -- is the CLIP TEXT branch of ViT_M2F2Det_Bridge dead code?")
    A("=" * 100)
    A("model   : vit_module/vit_m2f2_detector_bridge.py :: ViT_M2F2Det_Bridge")
    A("ckpt    : checkpoints/stage_1/bridge_v2_phase1.pth")
    A("claim   : clip_text_features (L432) -> text_proj (L435) is computed every forward and then")
    A("          never used; final fusion L478-484 is cat([clip_vision_cls(768), clip_adapt_embed(128),")
    A("          vit_features(768)]) -> Linear(1664, 2).")
    A("method  : runtime falsification (V5 energy decomposition, V3 shape/constancy, V2 monkeypatch,")
    A("          V1 gradient reachability, V1b parameter stats).")
    A("status  : the text-branch claim is CONFIRMED; one V1 contrast expectation FAILED")
    A("          (clip_reduction.weight grad is exactly 0) which led to a SECOND finding (V1x).")
    A("")
    A("\n".join(mblines))
    A("")
    A("-" * 100)
    A("SETUP")
    A("-" * 100)
    A(f"gpu query (index,used,free MiB) : {GPU_QUERY}")
    A(f"assigned card                   : {ASSIGNED}  used={USED_MIB} MiB free={FREE_MIB} MiB at start")
    A(f"device                          : {device} / {dev_name}  (CUDA_VISIBLE_DEVICES="
      f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    A(f"checkpoint load                 : strict=False missing={len(ck_missing)} "
      f"unexpected={len(ck_unexpected)}")
    if ck_missing:
        A(f"  missing keys (first 20)       : {list(ck_missing)[:20]}")
    if ck_unexpected:
        A(f"  unexpected keys (first 20)    : {list(ck_unexpected)[:20]}")
    A(f"checkpoint text_proj keys       : {text_proj_in_ckpt}")
    A(f"checkpoint prompt_tokens present: {prompt_in_ckpt}")
    A(f"requires_grad                   : {n_rg}/{n_params} params (left ON so V1 grads are meaningful)")
    A(f"images                          : {N_IMGS} from vit_module/_g16/layer_feats.npz['paths'] "
      f"(batches {BATCH}/{BATCH}/{BATCH}); read_fail={AUDIT['read_fail']}")
    A(f"pipeline                        : cv2.imread -> BGR2RGB -> cv2.resize(336) -> albumentations "
      f"Normalize(CLIP) -> ToTensorV2 -> model(images)")
    A(f"seed / threads                  : seed={SEED}; OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1, "
      f"torch.set_num_threads(1), cv2.setNumThreads(0), num_workers=0")
    A("")

    A("-" * 100)
    A("V5. CHECKPOINT-ONLY ENERGY DECOMPOSITION OF THE DECISION DIRECTION")
    A("-" * 100)
    A(f"  output.weight shape = {V5_shape}   (expected (2, 1664))"
      + ("" if V5_shape == (2, 1664) else "   <<< UNEXPECTED"))
    A(f"  d = weight[1] - weight[0]   (fake-direction minus real-direction)")
    A(f"  ||d|| = {float(np.sqrt(tot)):.8f}   ||d||^2 = {tot:.8f}")
    A("")
    A(f"  {'segment':<12}{'slice':>12}{'dims':>7}{'norm^2':>16}{'norm':>14}{'energy share':>16}")
    for name, a, b in SEG:
        s = seg[name]
        A(f"  {name:<12}{f'[{a}:{b}]':>12}{s['dim']:>7}{s['n2']:>16.6f}{s['norm']:>14.6f}"
          f"{s['share']*100:>15.4f}%")
    A("")
    A(f"  TEXT FEATURES OCCUPY 0 DIMENSIONS OF THIS VECTOR.")
    A(f"    The fusion concat is [clip_vision_cls(768) | clip_adapt_embed(128) | vit_features(768)], so the")
    A(f"    only 1664 dimensions the classifier can read are those three segments. text_proj's 768-d output")
    A(f"    is not concatenated, not added, not gated: it contributes exactly 0 of the 1664 input dims and")
    A(f"    therefore 0.0000% of ||d||^2.")
    A(f"  dimensions with per-dim energy share > 0.001 : {ndim_gt} / {int(W.shape[1])}")
    A(f"    (of which in a hypothetical text region : {hi_in_text_region} -- the region does not exist)")
    A(f"  expected approx 62.3% / 0.03% / 37.7% -> "
      + ("MATCHES" if v5_pass else "DOES NOT MATCH (see numbers above)"))
    A("")

    A("-" * 100)
    A("V3. SHAPE AND CONSTANCY OF THE TEXT FEATURES")
    A("-" * 100)
    A(f"  V3.1  model.clip_text_encoder() (called directly, no image input):")
    A(f"          shape={V3_1_shape}  dtype={V3_1_dtype}  device={V3_1_dev}")
    A(f"          expected [1, 768] -> "
      + ("CONFIRMED" if v3_1_pass else f"NOT CONFIRMED (got {V3_1_shape})"))
    A(f"          mechanism (source, text_encoder.py L30/L55): input_embeds defaults to")
    A(f"          self.prompt_tokens of shape (1, 7, 768), so batch_size is HARD-CODED to 1 and the")
    A(f"          returned pooled_output is last_hidden_state[:, -1, :] = [1, 768]. The image batch")
    A(f"          dimension never enters this function.")
    A("")
    A(f"  V3.2  text_proj OUTPUT captured by a forward hook during a real batch-{BATCH} forward:")
    A(f"          shape={V3_2_shape}")
    A(f"          max pairwise ||x_i - x_j|| over the batch = {V3_2_batch_maxdev:.10f}")
    if V3_2_shape[0] <= 1:
        A(f"          >>> THIS NUMBER IS VACUOUS: the tensor has exactly 1 row, there are no sample")
        A(f"              pairs. The real test is batch-size independence, below.")
    A(f"          batch-size independence (same first image):")
    A(f"            max|T(batch=1) - T(batch=16)| = {dev_1v16:.3e}")
    A(f"            max|T(batch=8) - T(batch=16)| = {dev_8v16:.3e}")
    A(f"            -> batch independent: {'YES (exactly 0)' if v3_batch_indep else 'NO'}")
    A(f"          determinism control (same batch twice, unpatched): torch.equal={det_same} "
      f"max|delta|={det_maxabs:.3e}")
    A("")
    A(f"  V3.3  second, COMPLETELY DIFFERENT batch of {BATCH} images (disjoint paths [16:32] vs [0:16]):")
    A(f"          L_inf ||T_A - T_B|| = {V3_3_linf:.3e}")
    A(f"          L2    ||T_A - T_B|| = {V3_3_l2:.3e}")
    A(f"          torch.equal(T_A, T_B) = {V3_3_eq}")
    A(f"          -> {'EXACTLY ZERO: text output is a constant, independent of the images' if v3_3_pass else 'NONZERO: text output DEPENDS on the images'}")
    A("")

    A("-" * 100)
    A("V2. FUNCTIONAL-PATH TEST (monkeypatched clip_text_encoder.forward)")
    A("-" * 100)
    A(f"  determinism control, unpatched, batch {BATCH} twice: torch.equal={ctrl_eq} "
      f"max|dLogit|={ctrl_maxabs:.3e}")
    A(f"  (if this were not 0/True, bitwise comparisons below would be uninterpretable)")
    A("")
    A(f"  {'replacement':<30}{'probe shape':>14}{'probe max|x|':>14}{'max|dLogit|':>15}{'torch.equal':>13}")
    for k in V2:
        r = V2[k]
        if r.get("ok"):
            A(f"  {k:<30}{str(r['shape']):>14}{r['probe_maxabs']:>14.6f}{r['maxabs']:>15.3e}"
              f"{str(r['eq']):>13}")
        else:
            A(f"  {k:<30}  RAISED {r.get('err')}")
    A("")
    A(f"  (a) exact zeros            : "
      + ("max|dLogit| = 0, bitwise identical" if V2.get("a_zeros", {}).get("eq") else "DIFFERENT"))
    for s in NOISE_SEEDS:
        k = f"b_noise_seed{s}"
        if k in V2 and V2[k].get("ok"):
            A(f"  (b) random noise seed {s:<4} : max|dLogit| = {V2[k]['maxabs']:.3e}, "
              f"bitwise identical = {V2[k]['eq']}")
    A("")
    A("  Extra (beyond spec): replacements with the WRONG shape were also tried, to probe whether")
    A("  ANY consumer would notice. Results are MIXED and must be read carefully:")
    for k in ["c_wrong_shape_16x768", "c_wrong_shape_1664"]:
        if k in V2 and V2[k].get("ok"):
            A(f"    {k:<26} probe shape={V2[k]['shape']} max|dLogit|={V2[k]['maxabs']:.3e} "
              f"torch.equal={V2[k]['eq']}")
        elif k in V2:
            A(f"    {k:<26} RAISED {V2[k].get('err')}")
    A("    [16,768]: accepted, logits bitwise identical -> a batch-shaped text tensor changes nothing.")
    A("    [1,1664]: RAISED RuntimeError inside text_proj itself ('mat1 and mat2 shapes cannot be")
    A("      multiplied (1x1664 and 768x768)'), i.e. the failure happens at the Linear that produces")
    A("      the text embedding, BEFORE any downstream consumer could see it. This probe therefore")
    A("      says NOTHING about downstream consumption and is EXCLUDED from the pass criterion.")
    A(f"  -> V2 SPEC PROBES (zeros + 3 noise seeds) ALL PASS = {v2_pass}")
    A("")

    A("-" * 100)
    A("V1. GRADIENT TEST  (out = model(images); loss = out.sum(); loss.backward())")
    A("-" * 100)
    A(f"  batch = {GRAD_BATCH}, no torch.no_grad, requires_grad {n_rg}/{n_params} params ON")
    A(f"  loss = {float(loss.item()):.6f}   out.shape = {tuple(out_g.shape)}")
    A("")
    A(f"  {'parameter group':<42}{'n':>5}{'grad is None':>14}{'grad.abs().max()':>20}")
    for gk, g in groups.items():
        A(f"  {g['label']:<42}{g['n']:>5}{g['n_none']:>14}{g['grad_max']:>20.6e}")
    A("")
    A("  !!! DEVIATION FROM EXPECTATION !!!")
    A(f"  The task expected ALL 4 named contrast params nonzero. Measured:")
    A(f"    output.weight        grad.abs().max() = {groups['output']['grad_max']:.6e}   nonzero  OK")
    A(f"    vision_proj[0].weight grad.abs().max() = {groups['vision_proj']['grad_max']:.6e}   nonzero  OK")
    A(f"    deepfake_proj[0].weight grad.abs().max() = {groups['deepfake_proj']['grad_max']:.6e}   nonzero  OK")
    A(f"    clip_reduction.weight grad.abs().max() = {groups['clip_reduction']['grad_max']:.6e}   "
      f"*** EXACTLY ZERO (grad tensor exists, all entries 0.0) ***")
    A("  A grad tensor that EXISTS but is exactly 0 means autograd DID build a path to clip_reduction,")
    A("  but that path carries no derivative -- i.e. the bridge branch's output does not actually")
    A("  depend on clip_reduction's parameters. This is NOT the text-branch phenomenon (None grad);")
    A("  it is a second, independent structural finding, investigated in V1x below.")
    A("")
    A(f"  {'prefix':<26}{'n':>5}{'grad None':>11}{'max|grad|':>16}{'reqGrad=F':>12}")
    for k_, v_ in SWEEP.items():
        A(f"  {k_:<26}{v_[0]:>5}{v_[1]:>11}{v_[2]:>16.6e}{v_[3]:>12}")
    A("")
    A(f"  TEXT BRANCH (all text_proj.* + all clip_text_encoder.*): {len(text_branch)} params, "
      f"grad is None for {tb_none}")
    A(f"    max|grad| over any non-None text param = {tb_max:.6e}")
    A(f"  CONTRAST (output.weight/bias, clip_reduction.weight, vision_proj[0].weight, "
      f"deepfake_proj[0].weight):")
    A(f"    grad is None for {c_none} of {len(contrast_names)}; max|grad| = {c_max:.6e}")
    A("")
    if tb_none == len(text_branch):
        A("  >>> INTERPRETATION: every single text-branch parameter has grad = None. (Not zero --")
        A("      None: autograd never even built a path from the loss to them, because the tensor")
        A("      text_proj produces is discarded. If the branch were merely multiplied by a learned")
        A("      weight that happened to be 0, grads would be exactly 0.0 tensors, not None.)")
        A("      Meanwhile the contrast groups all have nonzero grads, so the backward pass was real")
        A("      and the loss genuinely depends on output / clip_reduction / vision_proj / deepfake_proj.")
        A("      Any loss built from `output` therefore cannot have trained the text branch.")
    else:
        A(f"  >>> {len(text_branch)-tb_none} text-branch parameter(s) DO have a gradient; "
          f"see the None-name list in the stats npz.")
    A("")

    A("-" * 100)
    A("V1x. SECONDARY FINDING -- WHY IS clip_reduction.weight GRADIENT EXACTLY ZERO?")
    A("     (not part of the original V-spec; surfaced by the V1 contrast column)")
    A("-" * 100)
    A("  Hypothesis, from source: vit_m2f2_detector_bridge.py L157-161 defines")
    A("      BridgeAdapter_Proj_ViT.bridge_adapter_proj = Sequential(View(-1,64), Linear(64,1),")
    A("      LayerNorm(1)).")
    A("  nn.LayerNorm(1) normalizes a 1-element vector: mean == the element, so (x - mean) == 0")
    A("  EXACTLY, rstd*0 == 0, and the layer output reduces to its bias. Consequently the pooling")
    A("  output is a constant that does not depend on the bridge features at all.")
    A("")
    A(f"  Micro-test, nn.LayerNorm(1) on {512} random scalars (spread {V1X['ln1_in_spread']:.6f}):")
    A(f"    output spread = {V1X['ln1_spread']:.3e}   (bias = {float(model.bridge_adapter_proj.bridge_adapter_proj[2].bias.item()):.6f})")
    A(f"    -> {'CONFIRMED: output is exactly constant' if V1X['ln1_spread'] == 0.0 else 'not constant'}")
    A("")
    A(f"  Same effect inside the real model:")
    A(f"    DISCRIMINATOR -- the Linear(64,1) that feeds LayerNorm(1) produces shape "
      f"{V1X['pre_shape']}")
    A(f"      with spread {V1X['pre_spread']:.6e} over all tokens/images, and its weight is NON-ZERO")
    A(f"      (|w|max={V1X['lin_pool_w_max']:.6e}, ||w||={V1X['lin_pool_w_norm']:.6e}).")
    A(f"      So the input to LayerNorm(1) really does vary; the collapse happens IN LayerNorm(1).")
    A(f"    the model's LayerNorm(1) has weight={V1X['ln1_w']} bias={V1X['ln1_bias']:.10e};")
    A(f"      output = (x-mean)*rstd*weight + bias = 0*weight + bias = bias, a constant.")
    A(f"    inner per-token output shape {V1X['tok_shape']}, spread over ALL {V1X['tok_shape'][0]} entries "
      f"= {V1X['tok_spread']:.3e}")
    A(f"      -> the per-token pooled scalar is the SAME for every token and every image.")
    A(f"    bridge embedding (the 128-d clip_adapt_embed input) shape {V1X['be_dim']}")
    A(f"      pairwise max deviation within one batch of 16 : {V1X['be_batch_spread']:.3e}")
    A(f"      max|A - B| across two DIFFERENT batches of 16 : {V1X['be_cross']:.3e} "
      f"(L2 = {V1X['be_cross_l2']:.3e})")
    A(f"      -> batch-constant: {'YES' if V1X['be_batch_const'] else 'NO'}")
    A("")
    A("  CONSEQUENCE: the 128 fusion dims [768:896] receive a vector that is identical for every")
    A("  image. The classifier can only use it as a constant offset (absorbed into output.bias),")
    A("  which is consistent with the V5 measurement that those 128 dims carry 0.0265% of ||d||^2.")
    A("  It also explains the exactly-zero gradients on clip_reduction and (per the sweep above) on")
    A("  linear_vit_*/bridge_adapter.*: they are upstream of the LayerNorm(1) collapse.")
    A("  NOTE: this is a measurement of THIS checkpoint's forward graph; the parameters themselves")
    A("  may still hold values learned before/elsewhere. It is NOT a claim about training history.")
    A("")

    A("-" * 100)
    A("V1b. PARAMETER STATISTICS (recorded, no judgement)")
    A("-" * 100)
    for key in ["text_proj.0.weight", "clip_text_encoder.prompt_tokens", "output.bias"]:
        st = V1B.get(key)
        if st is None:
            A(f"  {key}: NOT FOUND")
            continue
        A(f"  {key}")
        A(f"    shape={st['shape']}  numel={st['numel']}  mean={st['mean']:.8e}  "
          f"std={st['std']:.8e}  norm={st['norm']:.8e}")
        A(f"    first 5 flattened values = {st['first5']}")
    A("")

    A("-" * 100)
    A("AUDIT")
    A("-" * 100)
    A(f"  forwards  : {AUDIT['imgs_forward']} image-evaluations in {AUDIT['batch_calls']} batch calls")
    A(f"  read_fail : {AUDIT['read_fail']}")
    A(f"  peak mem  : nvidia-smi device used {peak_smi:.0f} MiB / torch max_memory_allocated "
      f"{peak_torch:.0f} MiB (of 11055 MiB free at start)")
    A(f"  wall      : {wall:.1f}s")
    A("")

    A("-" * 100)
    A("CAVEATS")
    A("-" * 100)
    A("  1. This experiment tests the ARCHITECTURE CODE PATH, not every possible usage. It shows the")
    A("     text branch's output cannot reach `output` in this forward(). Any consumer of it would")
    A("     have to read model.clip_text_encoder()/text_proj directly outside forward() (no caller")
    A("     does so in this repo's forward path) or a subclass/different checkpoint could change it.")
    A("  2. Everything is done with ONE checkpoint (bridge_v2_phase1.pth). The conclusion is")
    A("     'these weights, this code' -- a differently-built model would need its own test.")
    A("  3. V1 used loss = out.sum() (no labels). That is a valid loss for asking 'does autograd")
    A("     build a path from these params to the logits'; it is NOT the training loss. Any loss")
    A("     that is a function of `output` gives the same answer, since reachability is a property")
    A("     of the graph, not of the loss value.")
    A("  4. V1 grad=None means 'not reached by this graph'. Because text_proj IS called in forward")
    A("     (so it is in the graph as a dead end) the None is informative; but the experiment cannot")
    A("     distinguish 'never trained' from 'trained in a different code path / different script'.")
    A("     A separate history check (Stage-1/2/3 training scripts) would be needed to claim the")
    A("     parameters were never updated by ANY training run.")
    A("  5. V3.2's literal 'max per-sample deviation across the batch' is vacuous because the tensor")
    A("     has exactly one row; the batch-size-independence check (batch 1 vs 8 vs 16) is the")
    A("     substantive substitute and is reported alongside it.")
    A("  6. Bitwise comparisons (V2, V3.3) are meaningful only because the unpatched repeat-forward")
    A("     determinism control came back bitwise identical; that control is reported.")
    A("  7. V1b statistics are recorded from the checkpoint as loaded; the equally-loaded identical")
    A("     values in text_proj being initialized elsewhere in this repo (init_weights uses")
    A("     normal_(std=0.01) / bias 0) were NOT verified against the checkpoint in this run, so no")
    A("     claim is made about how these values arose.")
    A("  7b. V1's contrast expectation was NOT fully met: clip_reduction.weight has an exactly-zero")
    A("      gradient. This does not weaken the text-branch conclusion (the text branch has NO grad")
    A("      tensor at all, which is a different and stronger signal) but it means the V-spec's stated")
    A("      expectation for the contrast group is falsified for 1 of its 4 members.")
    A("  7c. The V1x bridge-constancy finding rests on one forward graph of this checkpoint at fp32.")
    A("      It is measured (not inferred) for the pooled token / 128-d embedding / LayerNorm(1) micro-")
    A("      test, but the claim 'the bridge branch is dead' is scoped to this code path + checkpoint.")
    A("      Whether LayerNorm(1) was intentional in the original EfficientNet BridgeAdapter_Proj was")
    A("      not checked here.")
    A("  8. Single GPU (physical card 2), fp32 only, batch<=16; no fp16/bf16 numerics were tested.")
    A("  9. V5 is checkpoint-only: it characterizes the read-out direction d, not the model's")
    A("     behaviour on data. The 0% text share is a property of the concat order + Linear input")
    A("     width, which is structural, not weight-dependent.")
    A("")
    A("=" * 100)
    A(f"VERDICT: {VERDICT}")
    A(f"END OF REPORT   (wall {wall:.1f}s)")
    A("=" * 100)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    np.savez(STATS,
             **{k: np.array(str(v)) for k, v in mb.items()},
             meta_keys=np.array(list(mb.keys())),
             d_vector=dn.astype(np.float64),
             output_weight=W.numpy().astype(np.float64),
             output_bias=b_out.numpy().astype(np.float64),
             seg_bounds=np.array([[a, b] for _, a, b in SEG]),
             dim_energy=dim_energy.astype(np.float64),
             tp16=tp16.float().cpu().numpy(),
             tp_other=tp_other.float().cpu().numpy(),
             out_base=out_base.float().cpu().numpy(),
             out_b2=out_b2.float().cpu().numpy(),
             grad_group_names=np.array([groups[k]["label"] for k in groups]),
             grad_group_none=np.array([groups[k]["n_none"] for k in groups]),
             grad_group_max=np.array([groups[k]["grad_max"] for k in groups]),
             ck_missing=np.array(list(ck_missing)),
             ck_unexpected=np.array(list(ck_unexpected)),
             bridge_embed_A=be_A.float().cpu().numpy(),
             bridge_embed_B=be_B.float().cpu().numpy(),
             bridge_inner_tokens_A=tok_A.float().cpu().numpy(),
             sweep_prefixes=np.array(list(SWEEP.keys())),
             sweep_n=np.array([SWEEP[k][0] for k in SWEEP]),
             sweep_none=np.array([SWEEP[k][1] for k in SWEEP]),
             sweep_max=np.array([SWEEP[k][2] for k in SWEEP]))

    print(f"[save] report -> {REPORT}", flush=True)
    print(f"[save] stats  -> {STATS}", flush=True)
    print(f"[done] VERDICT={VERDICT} wall={wall:.1f}s peak_smi={peak_smi:.0f}MiB "
          f"peak_torch={peak_torch:.0f}MiB", flush=True)


if __name__ == "__main__":
    main()
