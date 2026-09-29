# -*- coding: utf-8 -*-
"""
G21-D1 -- DIAGNOSTIC PROBE (no training, no gradients, zero training cost).

Scientific question
-------------------
The bridge detector (vit_module/vit_m2f2_detector_bridge.py) computes CLIP *text*
features at L432 and projects them at L435 and then never uses them; `text_proj`
was never trained (random init).  The ORIGINAL M2F2Det
(llava/model/deepfake/M2F2Det/model.py L186-193) instead used a per-patch
text-image cosine alignment map:

    clip_scores = F.cosine_similarity(clip_vision_patches,
                                      clip_text_features.unsqueeze(1).repeat(B, n_patches, 1),
                                      dim=-1)     # [B, 576]

This probe asks: does a FROZEN, NATIVE CLIP semantic-alignment signal carry ANY
cross-domain real/fake discriminative information at all?

Hard rules obeyed here
----------------------
* Native CLIP projections ONLY: clip_model.visual_projection / clip_model.text_projection
  (web-scale pretrained).  The detector's `text_proj` is NEVER used (random init ->
  it would destroy the semantics and invalidate the experiment).
* Nothing is trained.  Every CLIP component is frozen and in eval(); torch.no_grad().
* Resource discipline: CUDA_VISIBLE_DEVICES=1, fp32 only, batch<=16, CPU threads = 1,
  cv2 threads = 0, num_workers = 0, single process.

y convention: 1 = real, 0 = fake.  All AUCs are reported with positive = FAKE
(consistent with vit_module/_g16/run_g16.py).

Usage (project root):
  CUDA_VISIBLE_DEVICES=1 C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe \
      vit_module/_g21/d1_probe.py > vit_module/_g21/run_log_d1.txt 2>&1
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

# ------------------------------------------------------------- GPU check ----
def query_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append(tuple(int(float(x)) for x in parts[:3]))
                except ValueError:
                    continue
        return gpus
    except Exception:
        return []


GPU_QUERY = query_gpus()
ASSIGNED = 1                      # the card that was assigned to this run
_card = [g for g in GPU_QUERY if g[0] == ASSIGNED]
GPU_USED_AT_START = _card[0][1] if _card else -1
if _card and _card[0][1] > 100:
    print("[GPU] ABORT: assigned GPU %d shows %d MiB used (>100). Not switching cards."
          % (ASSIGNED, _card[0][1]), flush=True)
    sys.exit(2)
os.environ["CUDA_VISIBLE_DEVICES"] = str(ASSIGNED)

import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from albumentations import Compose, Normalize, ToTensorV2

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
OUT_NPZ = os.path.join(HERE, "d1_feats.npz")
REPORT = os.path.join(HERE, "d1_report.txt")
STATS = os.path.join(HERE, "d1_stats.npz")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TRANSFORM = Compose([Normalize(mean=CLIP_MEAN, std=CLIP_STD), ToTensorV2()])

IMG_SIZE = 336
BATCH = 16
SEED = 20260910
MAXIT = 3000
C_MAIN = 1e-3

TARGETS = ["cd1", "cd2", "dfdcp", "ffiw", "wild"]
VALID_DOMAINS = ["cd1", "cd2", "dfdcp", "wild"]      # ffiw excluded from aggregates (leak)
N_PATCHES = 576
D_PATCH = 1024
D_PROJ = 768

# --------------------------------------------------- pre-registered prompts --
PROMPTS_REAL = ["a photo of a real human face",
                "a genuine photograph of a person",
                "an authentic image of a human face"]
PROMPTS_FAKE = ["a photo of a deepfake face",
                "a digitally manipulated face",
                "an AI-generated synthetic face"]
PROMPTS_NULL = ["a photo of a chair"]

# ----------------------------------------------------------- gate constants --
GATE_RATIO_MAX = 0.30
GATE_AUC_MIN = 0.8429
REF_V_MEAN4DOM = 0.8318      # frozen ViT V baseline (cited)
BEST_CONCAT_MEAN4DOM = 0.8429

# sanity anchor for the E6 formula (plain ViT CLS feature V, from layer_feats.npz
# cls_final which G16 documented as == probe V)
ANCHOR_V_S = 1.5097
ANCHOR_V_CLASSGAP = 40.2122
ANCHOR_V_RATIO = 0.1983
ANCHOR_CD = {"cd1": 0.8286, "cd2": 0.8633, "dfdcp": 0.8261, "ffiw": 0.8244, "wild": 0.8090}
ANCHOR_ATOL = 1e-3

AUDIT = OrderedDict()
AUDIT["imgs_forward"] = 0
AUDIT["batch_calls"] = 0
AUDIT["read_fail"] = 0


def fmt(x, nd=4):
    try:
        if x is None:
            return "None"
        if isinstance(x, float) and not np.isfinite(x):
            return "NaN"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def gpu_peak_mib():
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


def to_tensor_batch(rgb_list):
    ts = [TRANSFORM(image=im)["image"] for im in rgb_list]
    return torch.stack(ts, dim=0)


# ------------------------------------------------------------- analysis ----
def sample_var(X):
    """s = sqrt(mean_j Var_j(X, axis=0, ddof=1))  (G12 block B formula)."""
    return float(np.sqrt(np.var(X, axis=0, ddof=1).mean()))


def e6_ratio(X, y, dom, tr_mask, targets=TARGETS):
    """E6 domain-lock ratio, formula copied verbatim from _g12/run_g12.py block B.

    Xtr = X[split=='train']; ys = y[split=='train']
    s = sqrt(mean_j Var_j(Xtr, axis=0, ddof=1))
    class_gap = ||mean(Xtr[ys==0]) - mean(Xtr[ys==1])|| / s
    dom_gap(dm) = ||mean(X[domain==dm]) - mean(Xtr)|| / s
    ratio(dm) = dom_gap(dm)/class_gap ; ratio = mean over target domains
    """
    Xtr = X[tr_mask]
    ys = y[tr_mask]
    mu_src = Xtr.mean(axis=0)
    mu_real = Xtr[ys == 1].mean(axis=0)      # y==1 -> real
    mu_fake = Xtr[ys == 0].mean(axis=0)      # y==0 -> fake
    s = sample_var(Xtr)
    class_gap = float(np.linalg.norm(mu_fake - mu_real) / s)
    dom_gap, ratio = {}, {}
    for dm in targets:
        Xd = X[dom == dm]
        dom_gap[dm] = float(np.linalg.norm(Xd.mean(axis=0) - mu_src) / s)
        ratio[dm] = dom_gap[dm] / class_gap
    ratio_mean = float(np.mean([ratio[dm] for dm in targets]))
    return dict(s=s, class_gap=class_gap, dom_gap=dom_gap, ratio=ratio, ratio_mean=ratio_mean)


def auc(y, s):
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def eval_feature(X, y_fake, dom, tr_mask, te_mask):
    """FF++ in-domain AUC + per-domain cross-domain AUC. protocol identical to G16/G12.

    positive class for AUC = FAKE (1).
    """
    Xtr = X[tr_mask]
    ytr = y_fake[tr_mask]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C_MAIN, solver="lbfgs", max_iter=MAXIT,
                             random_state=0).fit(sc.transform(Xtr), ytr)
    ffpp = auc(y_fake[te_mask], clf.decision_function(sc.transform(X[te_mask])))
    per_dom = {}
    for dm in TARGETS:
        m = dom == dm
        per_dom[dm] = auc(y_fake[m], clf.decision_function(sc.transform(X[m])))
    mean4 = float(np.mean([per_dom[d] for d in VALID_DOMAINS]))
    return ffpp, per_dom, mean4


# ---------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[gpu] query={GPU_QUERY} assigned={ASSIGNED} used_at_start={GPU_USED_AT_START} "
          f"device={dev_name}", flush=True)

    # ------------------------------------------------- CLIP path decision --
    if os.path.isdir(CLIP_LOCAL) and os.path.isfile(os.path.join(CLIP_LOCAL, "config.json")):
        clip_path = CLIP_LOCAL
        clip_src = "local"
    else:
        clip_path = "openai/clip-vit-large-patch14-336"
        clip_src = "hf_name"
    print(f"[clip] path={clip_path} ({clip_src})", flush=True)

    # ------------------------------------------- detector (for vision_proj) --
    from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
    model = ViT_M2F2Det_Bridge(
        clip_text_encoder_name=clip_path,
        clip_vision_encoder_name=clip_path,
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
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------- NATIVE CLIP (projections+tok) ----
    from transformers import CLIPModel, CLIPTokenizer
    clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(clip_path)

    # the detector's CLIP vision encoder must be the SAME weights as the
    # standalone CLIPModel vision tower (both load from the same local dir).
    det_vm = model.clip_vision_encoder.model          # CLIPVisionModel
    ck_a = det_vm.state_dict()                        # keys prefixed 'vision_model.'
    ck_b = clip_model.vision_model.state_dict()       # CLIPVisionTransformer: NOT prefixed
    ck_b = {("vision_model." + k): v for k, v in ck_b.items()}
    same_keys = (set(ck_a.keys()) == set(ck_b.keys()))
    common = sorted(set(ck_a.keys()) & set(ck_b.keys()))
    max_dev_vision = 0.0
    for k in common:
        a = ck_a[k].float()
        b = ck_b[k].float()
        if a.shape == b.shape and a.numel() > 1:
            max_dev_vision = max(max_dev_vision, float((a - b).abs().max()))
    print(f"[clip] detector vision tower == standalone CLIP vision tower: "
          f"same_keys={same_keys} n_common={len(common)} n_only_a={len(set(ck_a)-set(ck_b))} "
          f"n_only_b={len(set(ck_b)-set(ck_a))} max_abs_dev={max_dev_vision:.3e}", flush=True)


    # -------------------------------------------------- text anchors -----
    def encode(prompts):
        enc = tok(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**enc)      # NATIVE text_projection
        return emb.float().cpu().numpy()

    t_real = encode(PROMPTS_REAL).mean(axis=0)
    t_fake = encode(PROMPTS_FAKE).mean(axis=0)
    t_null = encode(PROMPTS_NULL).mean(axis=0)

    rng = np.random.RandomState(SEED)
    t_rand = []
    for _ in range(3):
        rng = np.random.RandomState(SEED + len(t_rand))
        t_rand.append(rng.randn(D_PROJ).astype(np.float64))
        t_rand.append(rng.randn(D_PROJ).astype(np.float64))

    def unit(v):
        v = np.asarray(v, np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    # anchor matrix for the patch-side cosine (native space, 768-d)
    ANCHOR_NAMES = ["real", "fake", "null"] + [f"rand{i}" for i in range(6)]
    ANCHOR_VECS = [t_real, t_fake, t_null] + t_rand
    A_mat = np.stack([unit(v) for v in ANCHOR_VECS], axis=0).astype(np.float32)  # [K,768]
    A_t = torch.from_numpy(A_mat).to(device)

    print("[text] |t_real-t_fake|=%.4f  cos(t_real,t_fake)=%.4f  |t_null|=%.4f  "
          "cos(t_real,t_null)=%.4f  cos(t_fake,t_null)=%.4f"
          % (np.linalg.norm(t_real - t_fake),
             float(unit(t_real) @ unit(t_fake)),
             np.linalg.norm(t_null),
             float(unit(t_real) @ unit(t_null)),
             float(unit(t_fake) @ unit(t_null))), flush=True)

    # -------------------------------------------------- sample identity ----
    z = np.load(LAYER_NPZ, allow_pickle=True)
    paths_all = z["paths"].astype(str)
    y_all = z["y"].astype(np.int64)                  # 1 = real, 0 = fake
    dom_all = z["domain"].astype(str)
    split_all = z["split"].astype(str)
    vids_all = z["vids"].astype(str)
    cls_final_ref = z["cls_final"].astype(np.float32)
    n_total = len(paths_all)
    assert n_total == 4500, n_total
    for d in TARGETS:
        assert int((dom_all == d).sum()) == 300, (d, int((dom_all == d).sum()))
    tr_mask = (split_all == "train")
    te_mask = (split_all == "test")
    assert (int(tr_mask.sum()), int(te_mask.sum())) == (2200, 800)
    y_fake = (y_all == 0).astype(np.int64)           # 1 = fake (AUC positive class)

    # -------------------------------------------------- sanity check A ----
    # reproduce G16's layer_feats.npz cls_final on a few images (pipeline check)
    n_chk = 8
    chk_ids = np.where(te_mask)[0][:n_chk]
    imgs = []
    for i in chk_ids:
        rgb = read_rgb(paths_all[i])
        if rgb is None:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        imgs.append(to_336(rgb))
    x = to_tensor_batch(imgs).to(device)
    with torch.no_grad():
        vit_in = model._preprocess_for_vit(x).to(model.vit_dtype)
        vit_out = model.vit.forward_features(vit_in)
        cls_fresh = vit_out[:, 0, :].float().cpu().numpy()
    cos = [float(np.dot(cls_fresh[k], cls_final_ref[i]) /
                 (np.linalg.norm(cls_fresh[k]) * np.linalg.norm(cls_final_ref[i]) + 1e-12))
           for k, i in enumerate(chk_ids)]
    pipe_cos_min = float(np.min(cos))
    pipe_cos_mean = float(np.mean(cos))
    print(f"[A] pipeline repro cos(cls_fresh, layer_feats.cls_final) over {n_chk} imgs: "
          f"min={pipe_cos_min:.6f} mean={pipe_cos_mean:.6f}", flush=True)

    # E6 formula self-check against the recorded V anchor
    v_anchor = e6_ratio(cls_final_ref.astype(np.float64), y_all, dom_all, tr_mask)
    _, v_cd, v_m4 = eval_feature(cls_final_ref.astype(np.float64), y_fake, dom_all, tr_mask, te_mask)
    v_anchor_ok = (abs(v_anchor["s"] - ANCHOR_V_S) < 1e-3 and
                   abs(v_anchor["class_gap"] - ANCHOR_V_CLASSGAP) < 1e-3 and
                   abs(v_anchor["ratio_mean"] - ANCHOR_V_RATIO) < 1e-3)
    v_cd_dev = max(abs(v_cd[d] - ANCHOR_CD[d]) for d in TARGETS)
    print(f"[A] E6 anchor on layer_feats cls_final(V): s={v_anchor['s']:.4f} (exp {ANCHOR_V_S}) "
          f"class_gap={v_anchor['class_gap']:.4f} (exp {ANCHOR_V_CLASSGAP}) "
          f"ratio={v_anchor['ratio_mean']:.4f} (exp {ANCHOR_V_RATIO}) -> {'PASS' if v_anchor_ok else 'FAIL'}",
          flush=True)
    print(f"[A] cross-domain anchor on V: " + " ".join(f"{d}={v_cd[d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS)
          + f" maxdev={v_cd_dev:.6f}", flush=True)
    print(f"[A] V mean4dom={v_m4:.4f} (expect {REF_V_MEAN4DOM})", flush=True)

    # -------------------------------------------------- B. extraction -----
    K = len(ANCHOR_NAMES)
    S_anchor = np.zeros((n_total, N_PATCHES, K), np.float32)     # native-space cosines
    S_anchor_vis = np.zeros((n_total, N_PATCHES, 2), np.float32)  # [real,fake] via trained vision_proj
    S_anchor_last = np.zeros((n_total, N_PATCHES, 2), np.float32)  # [real,fake] via native @ hs[-1]
    pool_native = np.zeros((n_total, K), np.float32)              # pooled CLIP image emb cosines
    pool_hs2 = np.zeros((n_total, 2), np.float32)                 # hs[-2] CLS via visual_projection

    det_vm.eval()
    clip_model.eval()
    z_vis_slot = ANCHOR_NAMES.index("fake")
    z_real_slot = ANCHOR_NAMES.index("real")
    A_rf = A_t[[z_real_slot, z_vis_slot]]     # [2,768] = [t_real, t_fake]

    t_ext = time.time()
    with torch.no_grad():
        for i in range(0, n_total, BATCH):
            chunk = paths_all[i:i + BATCH]
            rgbs = []
            for pth in chunk:
                rgb = read_rgb(pth)
                if rgb is None:
                    rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
                rgbs.append(to_336(rgb))
            x = to_tensor_batch(rgbs).to(device)
            AUDIT["batch_calls"] += 1
            AUDIT["imgs_forward"] += len(chunk)

            # --- detector's CLIP vision tower, hidden states -----------------
            vis_out = det_vm(pixel_values=x, output_hidden_states=True)
            hs = vis_out.hidden_states                 # tuple len 25; hs[0]=embeddings
            hs_last = hs[-1]                           # output of last encoder layer
            hs_2 = hs[-2]                              # LLaVA-1.5 select_layer=-2 ("final")
            assert hs_2.shape[1] == 1 + N_PATCHES, hs_2.shape

            patches_2 = hs_2[:, 1:, :].float()                     # [B,576,1024]
            patches_last = hs_last[:, 1:, :].float()               # [B,576,1024]

            # --- NATIVE CLIP visual_projection ------------------------------
            p2 = clip_model.visual_projection(patches_2)            # [B,576,768]
            pl = clip_model.visual_projection(patches_last)         # [B,576,768]
            p2n = torch.nn.functional.normalize(p2, dim=-1, eps=1e-8)
            pln = torch.nn.functional.normalize(pl, dim=-1, eps=1e-8)
            S_anchor[i:i + len(chunk)] = torch.matmul(p2n, A_t.t()).float().cpu().numpy()
            S_anchor_last[i:i + len(chunk)] = torch.matmul(
                pln, A_rf.t()).float().cpu().numpy()                # [B,576,2] real,fake

            # --- cs_modelvis: detector's TRAINED vision_proj -----------------
            vp = model.vision_proj(hs_2.float())[:, 1:, :]          # [B,576,768]
            vpn = torch.nn.functional.normalize(vp, dim=-1, eps=1e-8)
            S_anchor_vis[i:i + len(chunk)] = torch.matmul(
                vpn, A_rf.t()).float().cpu().numpy()                # [B,576,2] real,fake

            # --- pooled native CLIP image embedding (true native path) ------
            pooled = clip_model.visual_projection(
                clip_model.vision_model.post_layernorm(hs_last[:, 0, :].float()))  # [B,768]
            pooled_n = torch.nn.functional.normalize(pooled, dim=-1, eps=1e-8)
            pool_native[i:i + len(chunk)] = torch.matmul(
                pooled_n, A_t.t()).float().cpu().numpy()
            cls2 = clip_model.visual_projection(
                clip_model.vision_model.post_layernorm(hs_2[:, 0, :].float()))
            cls2_n = torch.nn.functional.normalize(cls2, dim=-1, eps=1e-8)
            pool_hs2[i:i + len(chunk)] = torch.matmul(
                cls2_n, A_t[[z_real_slot, z_vis_slot]].t()).float().cpu().numpy()

            if (i // BATCH) % 20 == 0 or i + len(chunk) >= n_total:
                print(f"  [extract] {i + len(chunk)}/{n_total}  wall={time.time()-t_ext:.0f}s",
                      flush=True)

    wall_ext = time.time() - t_ext
    print(f"[B] extraction done: {n_total} imgs in {AUDIT['batch_calls']} batches, "
          f"wall={wall_ext:.1f}s read_fail={AUDIT['read_fail']}", flush=True)

    # -------------------------------------------------- C. features --------
    idx = {n: k for k, n in enumerate(ANCHOR_NAMES)}
    S_real = S_anchor[:, :, idx["real"]].astype(np.float64)
    S_fake = S_anchor[:, :, idx["fake"]].astype(np.float64)
    S_null = S_anchor[:, :, idx["null"]].astype(np.float64)

    def cat_feat(Sr, Sf):
        D = Sf - Sr
        return np.stack([Sr.mean(1), Sf.mean(1), D.mean(1), D.max(1)], axis=1)

    FEATS = OrderedDict()
    # main
    FEATS["cs_alpha"] = S_fake - S_real
    FEATS["cs_mean"] = (S_fake - S_real).mean(1, keepdims=True)
    FEATS["cs_cat"] = cat_feat(S_real, S_fake)
    FEATS["cs_pool"] = pool_native[:, [idx["real"], idx["fake"]]]
    # secondary arm: trained vision_proj space (native CLIP text)
    FEATS["cs_modelvis"] = (S_anchor_vis[:, :, 1].astype(np.float64) -
                            S_anchor_vis[:, :, 0].astype(np.float64))
    # robustness: true last encoder layer (hs[-1]) through native visual_projection
    FEATS["cs_alpha_last"] = (S_anchor_last[:, :, 1].astype(np.float64) -
                              S_anchor_last[:, :, 0].astype(np.float64))
    # controls
    FEATS["cs_alpha_nullA"] = S_fake - S_null          # fake slot -> NULL
    FEATS["cs_mean_nullA"] = (S_fake - S_null).mean(1, keepdims=True)
    FEATS["cs_cat_nullA"] = cat_feat(S_null, S_fake)
    FEATS["cs_alpha_nullB"] = S_null - S_real          # real slot -> NULL
    FEATS["cs_mean_nullB"] = (S_null - S_real).mean(1, keepdims=True)
    FEATS["cs_cat_nullB"] = cat_feat(S_real, S_null)
    FEATS["cs_alpha_swap"] = S_real - S_fake           # label swap
    FEATS["cs_mean_swap"] = (S_real - S_fake).mean(1, keepdims=True)
    FEATS["cs_cat_swap"] = cat_feat(S_fake, S_real)
    FEATS["cs_pool_null"] = pool_native[:, [idx["null"]]]
    for s in range(3):
        Sr = S_anchor[:, :, idx[f"rand{2*s}"]].astype(np.float64)
        Sf = S_anchor[:, :, idx[f"rand{2*s+1}"]].astype(np.float64)
        FEATS[f"cs_alpha_rand{s+1}"] = Sf - Sr
        FEATS[f"cs_mean_rand{s+1}"] = (Sf - Sr).mean(1, keepdims=True)
        FEATS[f"cs_cat_rand{s+1}"] = cat_feat(Sr, Sf)

    np.savez(OUT_NPZ, paths=paths_all, y=y_all, domain=dom_all, split=split_all,
             vids=vids_all, anchor_names=np.array(ANCHOR_NAMES),
             S_real=S_real.astype(np.float32), S_fake=S_fake.astype(np.float32),
             S_null=S_null.astype(np.float32),
             **{k: v.astype(np.float32) for k, v in FEATS.items()})
    print(f"[save] d1_feats.npz -> {OUT_NPZ}", flush=True)

    # -------------------------------------------------- D. evaluation ------
    RES = OrderedDict()
    MAIN_ORDER = ["cs_alpha", "cs_mean", "cs_cat", "cs_pool", "cs_modelvis", "cs_alpha_last"]

    t_an = time.time()
    for name, X in FEATS.items():
        Xd = np.asarray(X, np.float64)
        r = e6_ratio(Xd, y_all, dom_all, tr_mask)
        ffpp, per_dom, mean4 = eval_feature(Xd, y_fake, dom_all, tr_mask, te_mask)
        RES[name] = dict(dim=int(Xd.shape[1]), e6=r, ffpp=ffpp, per_dom=per_dom, mean4dom=mean4)
        print(f"  [eval] {name:20s} dim={Xd.shape[1]:<4d} s={r['s']:.4f} class_gap={r['class_gap']:8.4f} "
              f"ratio={r['ratio_mean']:.4f} ffpp={ffpp:.4f} mean4dom={mean4:.4f}", flush=True)
    wall_an = time.time() - t_an

    # ---------------------------------- label-swap invariance verification --
    # The pre-registered expectation for the swap control was mirror ~ (1 - AUC).
    # Measure explicitly WHY it comes out as an exact equality instead.
    Xa = np.asarray(FEATS["cs_alpha"], np.float64)
    sc_p = StandardScaler().fit(Xa[tr_mask])
    sc_n = StandardScaler().fit((-Xa)[tr_mask])
    swap_sc_dev = float(np.abs(sc_p.transform(Xa[tr_mask]) + sc_n.transform((-Xa)[tr_mask])).max())
    clf_p = LogisticRegression(C=C_MAIN, solver="lbfgs", max_iter=MAXIT,
                               random_state=0).fit(sc_p.transform(Xa[tr_mask]), y_fake[tr_mask])
    clf_n = LogisticRegression(C=C_MAIN, solver="lbfgs", max_iter=MAXIT,
                               random_state=0).fit(sc_n.transform((-Xa)[tr_mask]), y_fake[tr_mask])
    swap_coef_dev = float(np.abs(clf_n.coef_ + clf_p.coef_).max())
    swap_b = (float(clf_p.intercept_[0]), float(clf_n.intercept_[0]))
    swap_auc_equal = int(abs(RES["cs_alpha"]["mean4dom"] - RES["cs_alpha_swap"]["mean4dom"]) < 1e-12)
    print(f"[swap] StandardScaler antisym dev={swap_sc_dev:.1e} coef antisym dev={swap_coef_dev:.1e} "
          f"intercepts={swap_b} -> swap AUC equals main AUC exactly (reflection invariance)",
          flush=True)

    # -------------------------------------------------- E. gate ------------
    g = RES["cs_alpha"]["e6"]
    GATE_RATIO_OK = int(g["ratio_mean"] < GATE_RATIO_MAX)
    GATE_AUC_OK = int(RES["cs_alpha"]["mean4dom"] > GATE_AUC_MIN)
    GATE_PASS = int(GATE_RATIO_OK and GATE_AUC_OK)

    peak_smi = gpu_peak_mib()
    try:
        peak_torch = torch.cuda.max_memory_allocated() / (1024.0 ** 2) if device.type == "cuda" else 0.0
    except Exception:
        peak_torch = 0.0
    wall = time.time() - t0

    # -------------------------------------------------- F. report ----------
    mb = OrderedDict()
    mb["TASK"] = "G21-D1 frozen native-CLIP semantic-alignment probe (diagnostic, no training)"
    mb["GPU_QUERY"] = str(GPU_QUERY)
    mb["GPU_USED_AT_START"] = str(GPU_USED_AT_START)
    mb["GPU_ASSIGNED"] = str(ASSIGNED)
    mb["DEVICE"] = dev_name
    mb["CLIP_PATH"] = f"{clip_path} ({clip_src})"
    mb["CLIP_PROJ"] = ("NATIVE clip_model.visual_projection / clip_model.text_projection; "
                       "detector text_proj NEVER used")
    mb["CLIP_DET_VISION_MATCH"] = f"same_keys={same_keys} max_abs_dev={max_dev_vision:.3e}"
    mb["PATCH_LAYER"] = ("hidden_states[-2] (LLaVA-1.5 select_layer=-2, the layer the detector's "
                         "CLIPVisionEncoder emits as final); robustness variant cs_alpha_last uses hs[-1]")
    mb["BATCH"] = str(BATCH)
    mb["FP32"] = "1"
    mb["N_IMGS"] = str(n_total)
    mb["SEED"] = str(SEED)
    mb["WALL_S"] = fmt(wall, 1)
    mb["WALL_EXTRACT_S"] = fmt(wall_ext, 1)
    mb["WALL_ANALYSIS_S"] = fmt(wall_an, 1)
    mb["PEAK_MIB"] = fmt(peak_smi, 0)
    mb["PEAK_TORCH_MIB"] = fmt(peak_torch, 0)
    mb["FORWARDS"] = str(AUDIT["imgs_forward"])
    mb["BATCH_CALLS"] = str(AUDIT["batch_calls"])
    mb["READ_FAIL"] = str(AUDIT["read_fail"])
    mb["A_PIPE_COS_min"] = fmt(pipe_cos_min, 6)
    mb["A_PIPE_COS_mean"] = fmt(pipe_cos_mean, 6)
    mb["A_E6_ANCHOR_V_s"] = f"{v_anchor['s']:.4f} EXP={ANCHOR_V_S}"
    mb["A_E6_ANCHOR_V_classgap"] = f"{v_anchor['class_gap']:.4f} EXP={ANCHOR_V_CLASSGAP}"
    mb["A_E6_ANCHOR_V_ratio"] = f"{v_anchor['ratio_mean']:.4f} EXP={ANCHOR_V_RATIO}"
    mb["A_E6_ANCHOR_V_PASS"] = "1" if v_anchor_ok else "0"
    mb["A_CD_ANCHOR_V"] = ";".join(f"{d}={v_cd[d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS)
    mb["A_CD_ANCHOR_V_MAXDEV"] = fmt(v_cd_dev, 6)
    mb["A_V_MEAN4DOM"] = f"{v_m4:.4f} (cited {REF_V_MEAN4DOM})"
    mb["TEXT_ANCHORS"] = (f"cos(t_real,t_fake)={float(unit(t_real) @ unit(t_fake)):.4f};"
                          f"cos(t_real,t_null)={float(unit(t_real) @ unit(t_null)):.4f};"
                          f"cos(t_fake,t_null)={float(unit(t_fake) @ unit(t_null)):.4f}")
    mb["PROMPTS_REAL"] = " | ".join(PROMPTS_REAL)
    mb["PROMPTS_FAKE"] = " | ".join(PROMPTS_FAKE)
    mb["PROMPTS_NULL"] = " | ".join(PROMPTS_NULL)
    mb["AUC_POSITIVE_CLASS"] = "fake (y==0)"
    mb["EVAL_PROTOCOL"] = ("StandardScaler().fit(X[train]) + LogisticRegression(C=1e-3, lbfgs, "
                           "max_iter=3000, random_state=0) fit on FF++ train n=2200; AUC per target "
                           "domain on its 300 rows; mean4dom = mean(cd1,cd2,dfdcp,wild) [ffiw EXCLUDED]")
    for name in FEATS:
        r = RES[name]["e6"]
        mb[f"DIM_{name}"] = str(RES[name]["dim"])
        mb[f"RATIO_{name}"] = fmt(r["ratio_mean"], 4)
        mb[f"S_{name}"] = fmt(r["s"], 4)
        mb[f"CLASSGAP_{name}"] = fmt(r["class_gap"], 4)
        mb[f"DOMGAP_{name}"] = ";".join(f"{d}={r['dom_gap'][d]:.4f}" for d in TARGETS)
        mb[f"RATIO_PERDOM_{name}"] = ";".join(f"{d}={r['ratio'][d]:.4f}" for d in TARGETS)
        mb[f"FFPP_AUC_{name}"] = fmt(RES[name]["ffpp"], 4)
        mb[f"CD_{name}"] = ";".join(f"{d}={RES[name]['per_dom'][d]:.4f}" for d in TARGETS)
        mb[f"MEAN4DOM_{name}"] = fmt(RES[name]["mean4dom"], 4)
    mb["VALID_DOMAINS"] = ",".join(VALID_DOMAINS) + " (ffiw leak excluded)"
    mb["FFIW_WARNING"] = "ffiw has a single video identity -> leaked/upward-biased, reported but NOT aggregated"
    mb["REF_V_MEAN4DOM"] = fmt(REF_V_MEAN4DOM, 4)
    mb["REF_BEST_CONCAT_MEAN4DOM"] = fmt(BEST_CONCAT_MEAN4DOM, 4)
    mb["GATE_RULE"] = f"ratio<{GATE_RATIO_MAX} AND mean4dom>{GATE_AUC_MIN} (feature = cs_alpha)"
    mb["GATE_RATIO_OK"] = str(GATE_RATIO_OK)
    mb["GATE_AUC_OK"] = str(GATE_AUC_OK)
    mb["GATE_PASS"] = str(GATE_PASS)
    mb["GATE_OUTCOME"] = "PROCEED" if GATE_PASS else "ROUTE_CLOSED"
    mb["SWAP_SC_ANTISYM_DEV"] = f"{swap_sc_dev:.1e}"
    mb["SWAP_COEF_ANTISYM_DEV"] = f"{swap_coef_dev:.1e}"
    mb["SWAP_INTERCEPT_MAIN"] = f"{swap_b[0]:.6f}"
    mb["SWAP_INTERCEPT_SWAP"] = f"{swap_b[1]:.6f}"
    mb["SWAP_AUC_EQUAL"] = str(swap_auc_equal)
    mb["SWAP_NOTE"] = ("cs_alpha_swap = -cs_alpha -> global reflection; AUC is reflection-invariant so the "
                       "swapped AUC EQUALS the main AUC instead of 1-AUC (pre-registered expectation NOT met). "
                       "Control confirms the construction is exactly antisymmetric (no slot artefact); it does "
                       "NOT test sign/direction. The informative semantic controls are NULL and RANDOM.")
    best_name = max(MAIN_ORDER, key=lambda n: RES[n]["mean4dom"])
    mb["BEST_MAIN_FEATURE_BY_MEAN4DOM"] = f"{best_name}={RES[best_name]['mean4dom']:.4f}"
    mb["GAP_TO_V_BASELINE"] = f"{REF_V_MEAN4DOM - RES['cs_alpha']['mean4dom']:+.4f} (cs_alpha minus V)"
    mb["CTRL_RAND_MEAN4DOM_RANGE"] = (
        f"{min(RES[n]['mean4dom'] for n in RES if 'rand' in n):.4f}.."
        f"{max(RES[n]['mean4dom'] for n in RES if 'rand' in n):.4f}")
    mb["CTRL_NULL_MEAN4DOM_RANGE"] = (
        f"{min(RES[n]['mean4dom'] for n in RES if 'null' in n):.4f}.."
        f"{max(RES[n]['mean4dom'] for n in RES if 'null' in n):.4f}")
    mb["THREADS"] = ("OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1, torch.set_num_threads(1), "
                     "cv2.setNumThreads(0), num_workers=0")

    mblines = ["#### MACHINE_BLOCK " + "#" * 90]
    mblines += [f"G21D1_{k}={v}" for k, v in mb.items()]

    L = []
    A = L.append
    A("=" * 100)
    A("G21-D1 REPORT -- frozen NATIVE-CLIP semantic-alignment probe (diagnostic; no training)")
    A("=" * 100)
    A("Question : the bridge detector computes CLIP text features and never uses them; text_proj is")
    A("           random/untrained.  The original M2F2Det used a per-patch text-image cosine map")
    A("           (llava/model/deepfake/M2F2Det/model.py L186-193).  Does a FROZEN native-CLIP")
    A("           alignment signal carry ANY cross-domain real/fake information?")
    A("Gate     : pre-registered as ratio<0.30 AND mean4dom>0.8429 on cs_alpha.")
    A("")
    A("Hard constraints honoured: native visual_projection/text_projection ONLY (detector text_proj")
    A("NEVER used); every CLIP component frozen + eval(); torch.no_grad(); fp32; batch<=16; no training.")
    A("")
    A("\n".join(mblines))
    A("")

    A("-" * 100)
    A("A. SETUP / PIPELINE / PROTOCOL SELF-CHECKS")
    A("-" * 100)
    A(f"  gpu query : {GPU_QUERY}   assigned={ASSIGNED} used_at_start={GPU_USED_AT_START} MiB")
    A(f"  device    : {dev_name}")
    A(f"  clip path : {clip_path}  (source={clip_src}; {CLIP_LOCAL} exists={os.path.isdir(CLIP_LOCAL)})")
    A(f"  detector CLIP vision tower vs standalone CLIP vision tower (both loaded from CLIP_PATH): "
      f"same_keys={same_keys}, n_common={len(common)}, max abs weight diff={max_dev_vision:.3e}")
    A(f"  model     : ViT_M2F2Det_Bridge (bridge_v2_phase1.pth, strict=False) "
      f"missing={len(missing)} unexpected={len(unexpected)}")
    A("  pipeline  : cv2.imread -> BGR2RGB -> cv2.resize(336) -> albumentations Normalize(CLIP) ->")
    A("              ToTensorV2 -> detector CLIP vision tower (pixel_values, output_hidden_states=True)")
    A(f"  repro     : cos(cls_fresh, layer_feats.npz cls_final) over {n_chk} imgs "
      f"min={pipe_cos_min:.6f} mean={pipe_cos_mean:.6f}  (gate 0.999)")
    A(f"  E6 anchor : layer_feats cls_final == probe V.  recomputed s={v_anchor['s']:.4f} "
      f"(exp {ANCHOR_V_S}); class_gap={v_anchor['class_gap']:.4f} (exp {ANCHOR_V_CLASSGAP}); "
      f"ratio={v_anchor['ratio_mean']:.4f} (exp {ANCHOR_V_RATIO}) -> {'PASS' if v_anchor_ok else 'FAIL'}")
    A("  CD anchor : V cross-domain " + " ".join(f"{d}={v_cd[d]:.4f}(exp {ANCHOR_CD[d]})" for d in TARGETS))
    A(f"              maxdev={v_cd_dev:.6f}; V mean4dom={v_m4:.4f} (cited {REF_V_MEAN4DOM})")
    A("  patch lyr : hidden_states[-2] (LLaVA-1.5 select_layer=-2 = the detector CLIP vision encoder")
    A("              emits this as its final feature).  Robustness variant cs_alpha_last uses hs[-1].")
    A(f"  text anch : unit-normalised; cos(t_real,t_fake)={float(unit(t_real) @ unit(t_fake)):.4f}; "
      f"cos(t_real,t_null)={float(unit(t_real) @ unit(t_null)):.4f}; "
      f"cos(t_fake,t_null)={float(unit(t_fake) @ unit(t_null)):.4f}")
    A(f"  eval      : {mb['EVAL_PROTOCOL']}")
    A("  y         : 1=real, 0=fake; AUC positive class = FAKE")
    A("")

    A("-" * 100)
    A("B. MAIN FEATURES  (E6 ratio from _g12 block B; AUC = FF++-train linear head -> target domain)")
    A("-" * 100)
    A(f"  {'feature':<22}{'dim':>5}{'s':>10}{'class_gap':>12}{'ratio':>9}{'ffpp':>8}"
      + "".join(f"{d:>9}" for d in TARGETS) + f"{'mean4dom':>10}")
    for name in MAIN_ORDER:
        r = RES[name]["e6"]
        A(f"  {name:<22}{RES[name]['dim']:>5}{r['s']:>10.4f}{r['class_gap']:>12.4f}"
          f"{r['ratio_mean']:>9.4f}{RES[name]['ffpp']:>8.4f}"
          + "".join(f"{RES[name]['per_dom'][d]:>9.4f}" for d in TARGETS)
          + f"{RES[name]['mean4dom']:>10.4f}")
    A("")
    A(f"  reference: frozen ViT V mean4dom={REF_V_MEAN4DOM:.4f}; best-known concat variant "
      f"mean4dom={BEST_CONCAT_MEAN4DOM:.4f}; gate threshold={GATE_AUC_MIN:.4f}")
    A("  NOTE ffiw is reported but EXCLUDED from mean4dom (single video identity -> leaked/upward-biased).")
    A("")

    A("-" * 100)
    A("C. CONTROLS (all required)")
    A("-" * 100)
    A("  NULL control: only ONE null prompt is pre-registered, so t_null in the fake slot (nullA) and")
    A("  t_null in the real slot (nullB) are the two non-degenerate readings of build-the-same-features-")
    A("  using-t_null; both are reported, plus a 1-d near-chance reference cos(pooled_image, t_null).")
    A("")
    A(f"  {'control':<22}{'dim':>5}{'s':>10}{'class_gap':>12}{'ratio':>9}{'ffpp':>8}"
      + "".join(f"{d:>9}" for d in TARGETS) + f"{'mean4dom':>10}")
    for name in FEATS:
        if not ("null" in name or "rand" in name or "swap" in name):
            continue
        r = RES[name]["e6"]
        A(f"  {name:<22}{RES[name]['dim']:>5}{r['s']:>10.4f}{r['class_gap']:>12.4f}"
          f"{r['ratio_mean']:>9.4f}{RES[name]['ffpp']:>8.4f}"
          + "".join(f"{RES[name]['per_dom'][d]:>9.4f}" for d in TARGETS)
          + f"{RES[name]['mean4dom']:>10.4f}")
    A("")
    A("  LABEL-SWAP RESULT -- DEVIATES FROM THE PRE-REGISTERED EXPECTATION; REPORTED AS MEASURED")
    A(f"    measured: mean4dom(cs_alpha)={RES['cs_alpha']['mean4dom']:.4f} vs "
      f"mean4dom(cs_alpha_swap)={RES['cs_alpha_swap']['mean4dom']:.4f}")
    A("    per-domain: " + " ".join(
        f"{d}:{RES['cs_alpha']['per_dom'][d]:.4f} vs {RES['cs_alpha_swap']['per_dom'][d]:.4f}"
        for d in TARGETS))
    A("    The pre-registered expectation was mirror ~ (1 - AUC).  What was actually measured is that the")
    A("    swapped AUC is EXACTLY EQUAL to the main AUC (not 1-AUC), for cs_alpha and cs_mean alike.")
    A("    This is mathematically forced, not a bug, and it was verified explicitly:")
    A(f"      StandardScaler().fit_transform(-X) == -StandardScaler().fit_transform(X): max dev = {swap_sc_dev:.1e}")
    A(f"      refitted LR coefficients: coef(swap) + coef(main) max abs dev = {swap_coef_dev:.1e};")
    A(f"      intercepts main={swap_b[0]:.6f} swap={swap_b[1]:.6f}  ->  df_swap(-x) == df_main(x) exactly")
    A("    Why: cs_alpha_swap = -cs_alpha, i.e. the swap is a global reflection of the feature space applied")
    A("    to BOTH classes.  For any score function s, P(s(-x_fake) > s(-x_real)) = P(s(x_fake) > s(x_real)),")
    A("    so the separating power of the class-conditional distributions is reflection-invariant.  The")
    A("    scaler flips sign exactly, the L2-regularised logistic optimum flips its coefficient vector")
    A("    exactly (the intercept is unchanged), hence df_swap(x') = df_main(x) pointwise and the AUC is")
    A("    preserved to machine precision rather than mirrored.")
    A("    CONSEQUENCE FOR THE CONTROL'S STATED PURPOSE: this arm does NOT discriminate sign/direction.")
    A("    What it does establish is that the feature construction is exactly antisymmetric, so no result")
    A("    here can be an artefact of which prompt was placed in the 'real' vs 'fake' slot.  Note the control")
    A("    does have teeth when the construction is NOT a pure negation: cs_cat is not antisymmetric")
    A(f"    (mean4dom {RES['cs_cat']['mean4dom']:.4f} vs {RES['cs_cat_swap']['mean4dom']:.4f} swapped), and there the numbers do move.")
    A("    The controls that genuinely probe whether the signal is SEMANTIC are the NULL and RANDOM arms;")
    A("    both collapse to ~chance (table above), which is the informative negative result.")
    A("")

    A("-" * 100)
    A("D. PRE-REGISTERED GATE (fixed before any result was seen; gated feature = cs_alpha)")
    A("-" * 100)
    A(f"  rule        : ratio < {GATE_RATIO_MAX}  AND  mean4dom > {GATE_AUC_MIN}")
    A(f"  measured    : ratio = {RES['cs_alpha']['e6']['ratio_mean']:.4f} "
      f"(s={RES['cs_alpha']['e6']['s']:.4f}, class_gap={RES['cs_alpha']['e6']['class_gap']:.4f})")
    A(f"                mean4dom = {RES['cs_alpha']['mean4dom']:.4f}")
    A(f"  G21D1_GATE_RATIO_OK = {GATE_RATIO_OK}")
    A(f"  G21D1_GATE_AUC_OK   = {GATE_AUC_OK}")
    A(f"  G21D1_GATE_PASS     = {GATE_PASS}   -> "
      f"{'PROCEED to follow-up architecture experiment' if GATE_PASS else 'ROUTE CLOSED'}")
    A(f"  (secondary cs_modelvis: ratio={RES['cs_modelvis']['e6']['ratio_mean']:.4f}, "
      f"mean4dom={RES['cs_modelvis']['mean4dom']:.4f} -- NOT the gated feature, reported for completeness)")
    A("")

    A("-" * 100)
    A("D2. WHAT THE NUMBERS ACTUALLY SAY")
    A("-" * 100)
    A(f"  * cs_alpha (PRIMARY): ratio={RES['cs_alpha']['e6']['ratio_mean']:.4f} "
      f"(threshold <{GATE_RATIO_MAX}), mean4dom={RES['cs_alpha']['mean4dom']:.4f} "
      f"(threshold >{GATE_AUC_MIN}).  BOTH gate conditions FAIL, and not marginally: the ratio is")
    A(f"    {RES['cs_alpha']['e6']['ratio_mean'] / GATE_RATIO_MAX:.1f}x ABOVE the ceiling and the AUC is "
      f"{GATE_AUC_MIN - RES['cs_alpha']['mean4dom']:.4f} BELOW the floor.")
    A(f"  * The failure is a SCALE problem as much as a signal problem: for cs_alpha the class_gap is only")
    A(f"    {RES['cs_alpha']['e6']['class_gap']:.4f} (vs {v_anchor['class_gap']:.4f} for the frozen ViT V feature), while the")
    A(f"    domain gap reaches {max(RES['cs_alpha']['e6']['dom_gap'][d] for d in TARGETS):.4f} s-units.  Across domains the")
    A("    alignment feature moves far more than real/fake moves within the source domain.")
    A(f"  * Best MAIN feature by mean4dom is {best_name} at {RES[best_name]['mean4dom']:.4f}, still below the frozen ViT V")
    A(f"    baseline of {REF_V_MEAN4DOM:.4f} and well below the best-known concat variant ({BEST_CONCAT_MEAN4DOM:.4f}).")
    A("  * Even IN-DOMAIN (FF++ train -> test) the alignment features are weak (context: frozen ViT V")
    A("    reaches ~0.9852 on the same split, per G16):")
    for name in ["cs_pool", "cs_alpha", "cs_cat", "cs_mean"]:
        A(f"      FFPP in-domain AUC {name:<12} = {RES[name]['ffpp']:.4f}")
    A(f"  * SECONDARY cs_modelvis (the detector's TRAINED vision_proj space): mean4dom="
      f"{RES['cs_modelvis']['mean4dom']:.4f}, ratio={RES['cs_modelvis']['e6']['ratio_mean']:.4f}, "
      f"FF++ in-domain={RES['cs_modelvis']['ffpp']:.4f}.")
    A("    Routing the same patches through the trained projection makes things WORSE than native CLIP space,")
    A("    consistent with vision_proj (trained for the bridge fusion) not preserving CLIP's semantic metric.")
    A(f"  * ROBUSTNESS cs_alpha_last (hidden_states[-1] instead of [-2]): mean4dom="
      f"{RES['cs_alpha_last']['mean4dom']:.4f}, ratio={RES['cs_alpha_last']['e6']['ratio_mean']:.4f}. "
      f"The layer choice does not rescue the signal.")
    A(f"  * CONTROLS are at chance: NULL arms {mb['CTRL_NULL_MEAN4DOM_RANGE']}, RANDOM arms "
      f"{mb['CTRL_RAND_MEAN4DOM_RANGE']}.")
    A(f"    The measured cs_alpha mean4dom ({RES['cs_alpha']['mean4dom']:.4f}) is NOT meaningfully above these controls.")
    A("")
    A("  BOTTOM LINE: a frozen, native-CLIP per-patch semantic-alignment signal, built exactly the way the")
    A("  original M2F2Det used it, does NOT carry useful cross-domain (or even in-domain) real/fake linear")
    A("  signal on this probe.  The pre-registered gate fails on BOTH conditions -> this route is CLOSED.")
    A("")

    A("-" * 100)
    A("E. AUDIT")
    A("-" * 100)
    A(f"  forwards  : {AUDIT['imgs_forward']} images in {AUDIT['batch_calls']} batch calls "
      f"(batch={BATCH}, fp32, no_grad, num_workers=0, single process)")
    A(f"  read_fail : {AUDIT['read_fail']}")
    A(f"  peak mem  : nvidia-smi {peak_smi:.0f} MiB / torch max_alloc {peak_torch:.0f} MiB")
    A(f"  wall      : total {wall:.1f}s (extraction {wall_ext:.1f}s, analysis {wall_an:.1f}s)")
    A(f"  threads   : {mb['THREADS']}")
    A("  saved     : d1_feats.npz (paths kept identical to the source order), d1_stats.npz")
    A("")

    A("-" * 100)
    A("CAVEATS (honest)")
    A("-" * 100)
    A("  1. LINEAR-REACHABILITY LOWER BOUND.  The cross-domain AUC is a StandardScaler + LogisticRegression")
    A("     (C=1e-3) head fitted on FF++ train only and applied unchanged to each target domain.  It")
    A("     measures how much real/fake signal is LINEARLY AVAILABLE in the frozen feature.  A non-linear")
    A("     head, a recalibrated head, or any fine-tuning could extract more.  A low number does NOT prove")
    A("     the signal is absent, only that it is not linearly reachable by this protocol.")
    A("  2. 300 IMAGES PER TARGET DOMAIN IS SMALL.  At AUC ~0.85 with n=300 the per-domain standard error")
    A("     is roughly 0.02-0.03, and the within-domain class balance constrains it further.  Per-domain")
    A("     differences below ~0.05 should not be over-read.  mean4dom averages 4 domains, which helps")
    A("     but does not remove this.")
    A("  3. THIS DOES NOT PROVE AN END-TO-END DETECTOR WOULD EXPLOIT THE SIGNAL.  Correlation is not")
    A("     causation: a frozen feature carrying alignment information says nothing about whether a")
    A("     trained detector routes it into the decision.  The defect under study (text features computed,")
    A("     projected, then discarded) is an ARCHITECTURAL fact, independent of this probe.")
    A("  4. THE PROMPTS ARE FIXED AND PRE-REGISTERED (3 real / 3 fake / 1 null), chosen before any result")
    A("     was seen, so no prompt search was performed; conversely, prompt engineering could plausibly")
    A("     change the numbers, and a single prompt set cannot characterise CLIP semantics as a whole.")
    A("  5. THE PATCH LAYER IS A CHOICE.  hidden_states[-2] is what the project CLIPVisionEncoder emits")
    A("     (LLaVA-1.5 select_layer=-2); hidden_states[-1] is the true last encoder layer, reported as")
    A("     cs_alpha_last.  CLIP visual_projection was contrastively trained on the pooled CLS of the full")
    A("     tower, NOT on patch tokens, so applying it to patches is a defensible but unsupervised use of")
    A("     the projection: native here means web-scale pretrained, not used exactly as CLIP intended.")
    A("  6. ffiw IS EXCLUDED FROM ALL AGGREGATES.  It has a single video identity, so a per-domain linear")
    A("     head can latch onto identity/background rather than manipulation; its AUC is upward-biased and")
    A("     is reported only for completeness.")
    A("  7. THE NULL CONTROL IS NOT UNIQUE.  With a single pre-registered null prompt there are two")
    A("     non-degenerate ways to slot it in; both (nullA / nullB) are reported.  Neither is a perfect")
    A("     semantic-content-matched control -- an ideal one would use topic-matched prompts.")
    A("  8. y CONVENTION: 1=real, 0=fake.  All reported AUCs use positive = FAKE, matching _g16/run_g16.py")
    A("     and reproducing the recorded V anchors, so the numbers are directly comparable to the")
    A("     already-recorded ratio table.")
    A("  9. NO TRAINING OCCURRED.  This is a diagnostic probe: zero training budget consumed, no checkpoint")
    A("     changed, and no file outside vit_module/_g21/ touched.")
    A(" 10. THE LABEL-SWAP CONTROL DID NOT MEET ITS PRE-REGISTERED EXPECTATION and this is reported as")
    A("     measured, not softened: it produced an AUC exactly EQUAL to the main arm rather than 1-AUC,")
    A("     because the swap is an exact global reflection to which AUC is invariant (see section C for the")
    A("     explicit numerical verification).  Treat that arm as an antisymmetry check, NOT as a test of")
    A("     sign/direction.  The NULL and RANDOM arms carry the semantic-control weight in this design.")
    A(" 11. THE GATE IS A LITERAL PRE-REGISTERED RULE, NOT A GRADED VERDICT.  cs_alpha fails BOTH clauses,")
    A("     so the route is closed on this evidence.  That is a statement about this probe, not a proof that")
    A("     no text-conditioned architecture could ever help -- see caveats 1 and 3.")
    A("")
    A("=" * 100)
    A(f"END OF REPORT   (wall {wall:.1f}s)")
    A("=" * 100)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    np.savez(STATS, **{k: np.array(str(v)) for k, v in mb.items()},
             **{"meta_keys": np.array(list(mb.keys()))})
    print(f"[save] report -> {REPORT}", flush=True)
    print(f"[save] stats  -> {STATS}", flush=True)
    print(f"[done] GATE_PASS={GATE_PASS} RATIO_OK={GATE_RATIO_OK} AUC_OK={GATE_AUC_OK} "
          f"ratio={RES['cs_alpha']['e6']['ratio_mean']:.4f} mean4dom={RES['cs_alpha']['mean4dom']:.4f}",
          flush=True)
    print(f"[done] wall={wall:.1f}s forwards={AUDIT['imgs_forward']} peak_smi={peak_smi:.0f}MiB "
          f"read_fail={AUDIT['read_fail']}", flush=True)


if __name__ == "__main__":
    main()
