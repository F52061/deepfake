"""
ViT_M2F2Det_Bridge: Multi-modal Deepfake Detector with ViT backbone
and BridgeAdapter cross-modal/cross-scale fusion.

This is the BridgeAdapter variant — replaces the cosine-similarity fusion
in ViT_M2F2Det with the same 3-stage Transformer-based BridgeAdapter used
by the original EfficientNet M2F2Det.

Key differences from vit_m2f2_detector.py (cosine-similarity version):
 ┌──────────────────────────┬─────────────────────────────────┐
 │  Cosine-similarity       │  BridgeAdapter (this file)      │
 ├──────────────────────────┼─────────────────────────────────┤
 │  cos_sim(patches, text)  │  3× TransformerEncoderBlock     │
 │  → [B,576] scalar scores │   逐层递进融合 ViT + CLIP 特征    │
 │                          │  → [B,128] rich embedding       │
 ├──────────────────────────┼─────────────────────────────────┤
 │  No intermediate layers  │  ViT blocks[3][6][9] hooks      │
 │                          │  CLIP layers 6/10/14            │
 ├──────────────────────────┼─────────────────────────────────┤
 │  output: 576+1024+1024   │  output: 128 + 2*hidden_size    │
 │       = 2624 → 2         │       = 128+2*h → 2             │
 └──────────────────────────┴─────────────────────────────────┘

Environment: M2F2_Det conda env
  torch 2.2.2+cu118, transformers 4.37.0, timm 0.9.2, flash_attn installed
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# CLIP encoders (shared with original M2F2Det)
from llava.model.deepfake.M2F2Det.text_encoder import CLIPTextEncoder
from llava.model.deepfake.M2F2Det.vision_encoder import CLIPVisionEncoder

# flash_attn MHA — used by TransformerEncoderBlock
# 本环境无 cu118 Windows 轮子时回退到纯 PyTorch shim (非 flash 路径, 数学一致)
try:
    from flash_attn.modules.mha import MHA
except ImportError:
    try:
        from .flash_attn_shim.mha import MHA
    except ImportError:
        from vit_module.flash_attn_shim.mha import MHA

# ViT backbone (shared with cosine-similarity ViT_M2F2Det)
try:
    from .vit_adaptive_mattn_aps import vit_base_patch16_224
except ImportError:
    from vit_adaptive_mattn_aps import vit_base_patch16_224


# ═══════════════════════════════════════════════════════════════════
# CLIP 归一化常量 (用于 _preprocess_for_vit 的反归一化)
# ═══════════════════════════════════════════════════════════════════
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

# ViT patch 维度常量 (vit_base_patch16_224)
VIT_NUM_PATCHES = 196    # (224/16)^2 = 14×14 = 196, 不含 CLS
VIT_EMBED_DIM   = 768
CLIP_NUM_PATCHES = 576   # (336/14)^2 = 24×24 = 576, 不含 CLS
CLIP_HIDDEN_SIZE = 1024  # CLIP ViT-L/14 隐藏维度

# BridgeAdapter 内部维度常量
BRIDGE_EMBED_DIM = 64    # 每阶段对齐后的统一维度
BRIDGE_NUM_HEADS = 4     # MHA 注意力头数
BRIDGE_FF_DIM    = 32    # FFN 中间维度

# 各阶段累积序列长度
# Stage 0: vit(196) + clip(576) = 772
# Stage 1: 772 + 196 + 576 = 1544
# Stage 2: 1544 + 196 + 576 = 2316
BRIDGE_TOTAL_SEQ_LEN = 2316


# ═══════════════════════════════════════════════════════════════════
# Utility Modules (复用原始 M2F2Det 的设计，本地定义避免交叉依赖)
# ═══════════════════════════════════════════════════════════════════

class View(nn.Module):
    """Reshape module, equivalent to Tensor.view()."""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Permute(nn.Module):
    """Permute module, equivalent to Tensor.permute()."""
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block with flash_attn MHA + FFN.
    Identical to the one in sequence/models/M2F2_Det/models/model.py.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, causal=False):
        super().__init__()
        self.attn = MHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
        ).to(torch.bfloat16)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        orig_dtype = x.dtype
        attn_out = self.attn(x.to(torch.bfloat16)).to(orig_dtype)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class BridgeAdapter_Proj_ViT(nn.Module):
    """
    Projection head for BridgeAdapter output → compact embedding [B, bridge_adapter_dim].

    Equivalent to original BridgeAdapter_Proj but with parameterized total_seq_len
    (2316 for ViT) instead of hardcoded 2731 (EfficientNet).

    Pipeline:
        input [total_seq, B, embed_dim]
          → View(-1, embed_dim)            [total*B, embed_dim]
          → Linear(embed_dim, 1)            [total*B, 1]
          → LayerNorm(1)
          → View(B, total_seq)              [B, total_seq]
          → Linear(total_seq, out_dim)      [B, out_dim]
          → LayerNorm(out_dim)
    """
    def __init__(self, embed_dim, out_dim, total_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.total_seq_len = total_seq_len

        self.bridge_adapter_proj = nn.Sequential(
            View(-1, self.embed_dim),
            nn.Linear(self.embed_dim, 1),
            nn.LayerNorm(1),
        )

        self.bridge_adapter_reduction = nn.Sequential(
            nn.Linear(self.total_seq_len, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, x, batch_size):
        # x: [total_seq, B, embed_dim]
        x = self.bridge_adapter_proj(x)       # [total*B, 1]
        x = x.view(batch_size, -1)             # [B, total_seq]
        x = self.bridge_adapter_reduction(x)   # [B, out_dim]
        return x


# ═══════════════════════════════════════════════════════════════════
# Main Detector Class
# ═══════════════════════════════════════════════════════════════════

class ViT_M2F2Det_Bridge(nn.Module):
    """
    Multi-modal deepfake detector using ViT backbone with BridgeAdapter fusion.

    Architecture (3-branch → BridgeAdapter fusion → classification):
      Branch 1: ViT blocks[3][6][9] → hook features [B,196,768] ×3
      Branch 2: CLIP Vision layer6/10/14 → [B,576,1024] ×3
      Branch 3: ViT CLS token → deepfake_proj → [B,hidden_size]

      BridgeAdapter: 3-stage progressive fusion of branches 1+2
        Stage0: vit_block3 + clip_layer6  → TransformerBlock → [772,B,64]
        Stage1: + vit_block6 + clip_layer10 → TransformerBlock → [1544,B,64]
        Stage2: + vit_block9 + clip_layer14 → TransformerBlock → [2316,B,64]
        Then BridgeAdapter_Proj → [B,128]

      Final: cat(Branch3_CLS, Bridge128, Branch1_CLS) → Linear → [B,2]

    Constructor signature is identical to ViT_M2F2Det for drop-in replacement.
    """

    def __init__(
        self,
        clip_text_encoder_name: str  = "openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        deepfake_encoder_name: str   = 'vit',       # API compatibility
        hidden_size: int             = 768,          # common projection dim (match ViT CLS)
        vision_dtype: torch.dtype    = torch.float32,
        # NOTE: vision_dtype should match deepfake_dtype for numeric stability.
        # Using fp16 for vision features can cause NaN loss in the bridge adapter.
        text_dtype: torch.dtype      = torch.float32,
        deepfake_dtype: torch.dtype  = torch.float32,
        load_vision_encoder: bool    = True,
        pretrained: bool             = False,
        save_heatmap: bool           = False,        # reserved, not used in Bridge version
        heatmap_dir: str             = "./outputs/heatmaps",
    ):
        super().__init__()

        # ── 1. ViT backbone ────────────────────────────────────────────
        self.vit = vit_base_patch16_224(pretrained=pretrained, num_classes=0)
        self.vit_dtype = deepfake_dtype
        self.vit.to(deepfake_dtype)

        # ── 2. Register hooks on 3 intermediate ViT blocks ─────────────
        self.vit_block_outputs: dict = {}
        def _make_hook(name: str):
            def hook(module, input, output):
                self.vit_block_outputs[name] = output
            return hook
        # blocks[3] = 浅层 (纹理/边缘), blocks[6] = 中层 (局部形状), blocks[9] = 深层 (语义)
        self.vit.blocks[3].register_forward_hook(_make_hook("b_1"))
        self.vit.blocks[6].register_forward_hook(_make_hook("b_2"))
        self.vit.blocks[9].register_forward_hook(_make_hook("b_3"))

        # ── 3. Deepfake projection ─────────────────────────────────────
        # ViT CLS token (768) → hidden_size (融合空间)
        self.deepfake_proj = nn.Sequential(
            nn.Linear(VIT_EMBED_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.deepfake_proj.to(deepfake_dtype)

        # ── 4. CLIP text encoder ────────────────────────────────────────
        self.clip_text_encoder = CLIPTextEncoder(
            clip_text_encoder_name, dtype=text_dtype
        )

        # ── 5. CLIP vision encoder (optional) ──────────────────────────
        if load_vision_encoder:
            self.clip_vision_encoder = CLIPVisionEncoder(
                clip_vision_encoder_name, dtype=vision_dtype
            )
        else:
            self.clip_vision_encoder = None

        # ── 6. Projection layers ────────────────────────────────────────
        clip_text_hidden = self.clip_text_encoder.model.config.hidden_size  # 768

        self.vision_proj = nn.Sequential(
            nn.Linear(CLIP_HIDDEN_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.vision_proj.to(vision_dtype)

        self.text_proj = nn.Sequential(
            nn.Linear(clip_text_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.text_proj.to(text_dtype)

        # ── 7. BridgeAdapter components ─────────────────────────────────
        # CLIP patch features → 64-dim unified space
        self.clip_reduction = nn.Linear(CLIP_HIDDEN_SIZE, BRIDGE_EMBED_DIM)  # 1024→64

        # ViT patch features → 64-dim unified space (one per hook level)
        self.linear_vit_1 = nn.Linear(VIT_EMBED_DIM, BRIDGE_EMBED_DIM)       # 768→64
        self.linear_vit_2 = nn.Linear(VIT_EMBED_DIM, BRIDGE_EMBED_DIM)       # 768→64
        self.linear_vit_3 = nn.Linear(VIT_EMBED_DIM, BRIDGE_EMBED_DIM)       # 768→64
        self.linear_vit_lst = nn.ModuleList([
            self.linear_vit_1, self.linear_vit_2, self.linear_vit_3,
        ])

        # 3-stage progressive fusion transformers
        self.bridge_adapter = nn.ModuleList([
            TransformerEncoderBlock(BRIDGE_EMBED_DIM, BRIDGE_NUM_HEADS, BRIDGE_FF_DIM)
            for _ in range(3)
        ])

        # Final projection: [2316, B, 64] → [B, 128]
        self.bridge_adapter_proj = BridgeAdapter_Proj_ViT(
            embed_dim=BRIDGE_EMBED_DIM,
            out_dim=128,
            total_seq_len=BRIDGE_TOTAL_SEQ_LEN,
        )

        # ── 8. Learnable fusion weights ──────────────────────────────────
        self.clip_vision_alpha = nn.Parameter(torch.tensor(0.5))
        self.clip_text_alpha   = nn.Parameter(torch.tensor(4.0))
        self.clip_vision_alpha.to(vision_dtype)
        self.clip_text_alpha.to(text_dtype)

        # ── 9. Final classifier ──────────────────────────────────────────
        # input = cat([clip_vision_cls(hidden), bridge_embed(128), vit_feat(hidden)])
        self.output = nn.Linear(2 * hidden_size + 128, 2)
        self.output.to(deepfake_dtype)

        # ── 10. Cache & metadata ─────────────────────────────────────────
        self.cached_clip_text_features = None
        self.hidden_size = hidden_size
        self.vision_dtype = vision_dtype
        self.text_dtype = text_dtype
        self.deepfake_dtype = deepfake_dtype

        # ── 11. Weight initialization for new layers ──────────────────────
        self._init_new_components()

    # ═══════════════════════════════════════════════════════════════════
    # Initialization helpers
    # ═══════════════════════════════════════════════════════════════════

    def _init_new_components(self):
        """Initialize projection and BridgeAdapter layers with small random weights."""
        components_to_init = [
            self.deepfake_proj,
            self.vision_proj,
            self.text_proj,
            self.output,
            self.clip_reduction,
            self.bridge_adapter_proj,
        ]
        for component in components_to_init:
            for m in component.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ═══════════════════════════════════════════════════════════════════
    # Preprocessing
    # ═══════════════════════════════════════════════════════════════════

    def _preprocess_for_vit(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert CLIP-normalized images to ViT-compatible format.

        Pipeline: CLIP norm → [0,1] → resize 224 → [-1,1]
        """
        device = images.device
        mean = CLIP_MEAN.to(device).view(1, 3, 1, 1)
        std  = CLIP_STD.to(device).view(1, 3, 1, 1)
        x = images * std + mean                                             # [0, 1]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5                                                 # [-1, 1]
        return x

    # ═══════════════════════════════════════════════════════════════════
    # Forward
    # ═══════════════════════════════════════════════════════════════════

    def forward(
        self,
        images: torch.Tensor,
        clip_vision_features: Optional[torch.Tensor] = None,
        use_cached_clip_text_features: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            images: [B, 3, 336, 336] CLIP-normalized input images.
            clip_vision_features: Optional pre-computed CLIP vision features
                                  (used when load_vision_encoder=False and
                                   an external CLIP tower provides them).
            use_cached_clip_text_features: Reuse cached text features.
            labels: Optional ground-truth labels (reserved).

        Returns:
            output: [B, 2] classification logits (logit[0]=real, logit[1]=fake).
        """
        B = images.shape[0]

        # ──= Branch 1: ViT backbone + intermediate hooks ──────────────
        vit_input = self._preprocess_for_vit(images)
        vit_input = vit_input.to(self.vit_dtype)

        # forward_features triggers hooks → self.vit_block_outputs populated
        vit_out = self.vit.forward_features(vit_input)                      # [B, 197, 768]

        # ViT CLS token → high-level semantic features
        vit_cls_token = vit_out[:, 0, :]                                    # [B, 768]
        vit_features  = self.deepfake_proj(vit_cls_token)                   # [B, hidden_size]

        # 3 intermediate hook outputs → drop CLS → patch tokens only
        vit_feat_0 = self.vit_block_outputs["b_1"][:, 1:, :]               # [B, 196, 768]
        vit_feat_1 = self.vit_block_outputs["b_2"][:, 1:, :]               # [B, 196, 768]
        vit_feat_2 = self.vit_block_outputs["b_3"][:, 1:, :]               # [B, 196, 768]

        # ──= Branch 2: CLIP vision encoder ────────────────────────────
        if clip_vision_features is None:
            if self.clip_vision_encoder is not None:
                # Returns 4 tensors: clip_0, clip_1, clip_2, clip_vision_features
                clip_0, clip_1, clip_2, clip_vision_features = \
                    self.clip_vision_encoder(images)
            else:
                raise ValueError(
                    "clip_vision_features must be provided when "
                    "load_vision_encoder=False."
                )
        else:
            # External features provided; we still need intermediate layers if
            # the caller provides them as a tuple/extra arg. If not, we fall
            # back gracefully — but this path expects a tuple.
            if isinstance(clip_vision_features, (list, tuple)) and len(clip_vision_features) == 4:
                clip_0, clip_1, clip_2, clip_vision_features = clip_vision_features
            else:
                # Caller only provided final features — cannot do BridgeAdapter.
                raise ValueError(
                    "BridgeAdapter requires 4-tuple of CLIP features "
                    "(clip_0, clip_1, clip_2, clip_vision_features). "
                    "When load_vision_encoder=False, pass them as a tuple."
                )

        # Project CLIP vision features through shared projection
        clip_vision_features = clip_vision_features.to(self.vision_dtype)
        clip_vision_features = self.vision_proj(clip_vision_features)       # [B, 577, hidden_size]
        clip_vision_cls = clip_vision_features[:, 0, :]                     # [B, hidden_size]

        # ──= Branch 3: CLIP text encoder ──────────────────────────────
        if use_cached_clip_text_features:
            if self.cached_clip_text_features is None:
                self.cached_clip_text_features = self.clip_text_encoder()
            clip_text_features = self.cached_clip_text_features
        else:
            clip_text_features = self.clip_text_encoder()                   # [B, 768]

        clip_text_features = clip_text_features.to(self.text_dtype)
        clip_text_features = self.text_proj(clip_text_features)             # [B, hidden_size]

        # ──= BridgeAdapter: 3-stage progressive fusion ────────────────
        device = vit_input.device
        vit_feat_lst  = [vit_feat_0, vit_feat_1, vit_feat_2]               # 3×[B,196,768]
        clip_feat_lst = [clip_0, clip_1, clip_2]                            # 3×[B,576,1024]

        bridge_adapter_output = None
        for i, (vit_feat, clip_feat) in enumerate(zip(vit_feat_lst, clip_feat_lst)):
            # --- CLIP features → 64-dim ---
            clip_feat = clip_feat.to(device)
            clip_feat = self.clip_reduction(clip_feat)                      # [B, 576, 64]

            # --- ViT features → 64-dim ---
            vit_feat = vit_feat.to(device)
            self.linear_vit_lst[i] = self.linear_vit_lst[i].to(device)
            vit_feat = self.linear_vit_lst[i](vit_feat)                     # [B, 196, 64]

            # --- Concatenation with previous stage output ---
            if bridge_adapter_output is None:
                combined = torch.cat((vit_feat, clip_feat), dim=1)          # [B, 772, 64]
            else:
                bridge_adapter_output = bridge_adapter_output.permute(1, 0, 2)  # [B, prev_len, 64]
                combined = torch.cat(
                    (bridge_adapter_output, vit_feat, clip_feat), dim=1
                )

            # --- Transformer block ---
            combined = combined.permute(1, 0, 2)                            # [seq, B, 64]
            self.bridge_adapter[i] = self.bridge_adapter[i].to(device)
            bridge_adapter_output = self.bridge_adapter[i](combined)

        # ──= BridgeAdapter → compact embedding ────────────────────────
        clip_adapt_embed = self.bridge_adapter_proj(
            bridge_adapter_output, B
        )                                                                    # [B, 128]
        clip_adapt_embed = self.clip_text_alpha * clip_adapt_embed

        # ──= Final fusion + classification ────────────────────────────
        clip_vision_cls = self.clip_vision_alpha * clip_vision_cls.to(
            self.deepfake_dtype
        )

        features = torch.cat([
            clip_vision_cls,     # [B, hidden_size]
            clip_adapt_embed,    # [B, 128]
            vit_features,        # [B, hidden_size]
        ], dim=-1)                                                          # [B, 2*hidden_size+128]

        output = self.output(features)                                      # [B, 2]

        return output

    # ═══════════════════════════════════════════════════════════════════
    # Weight loading (ViT backbone only)
    # ═══════════════════════════════════════════════════════════════════

    def load_vit_backbone(self, checkpoint_path: str, verbose: bool = True):
        """
        Load pre-trained ViT backbone weights from a checkpoint.

        The checkpoint is expected to have keys like:
            model.blocks.0.attn.qkv.weight
            model.head.weight   ← skipped (classification head)

        These are mapped to:
            vit.blocks.0.attn.qkv.weight

        New BridgeAdapter layers are NOT loaded — they remain randomly
        initialized and will be trained from scratch.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {}
        loaded_count = 0
        skipped_head = 0

        for k, v in ckpt.items():
            # Skip classification head
            if k.startswith('model.head.'):
                skipped_head += 1
                continue
            # Map model.xxx → vit.xxx
            if k.startswith('model.'):
                vit_key = 'vit.' + k[6:]
                if vit_key in self.state_dict():
                    state_dict[vit_key] = v.to(
                        self.state_dict()[vit_key].dtype
                    )
                    loaded_count += 1

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if verbose:
            print(
                f'[ViT_M2F2Det_Bridge] Loaded {loaded_count} ViT backbone keys, '
                f'skipped {skipped_head} classification head keys.'
            )
            print(f'  Missing keys (BridgeAdapter etc, expected): {len(missing)}')
            print(f'  Unexpected keys: {len(unexpected)}')

        return missing, unexpected

    # ═══════════════════════════════════════════════════════════════════
    # Optimizer parameter groups (for Stage-1 training)
    # ═══════════════════════════════════════════════════════════════════

    def assign_lr_dict_list(self, lr: float = 1e-4):
        """
        Return a list of parameter-group dicts for torch.optim.Adam.

        Format matches original M2F2Det.assign_lr_dict_list() so that
        stage_1_detection.py can use this class as a drop-in replacement.

        Learning rate assignments:
          - Alpha params: 1e-3 / 3e-3 (higher, learn fusion weights fast)
          - All other trainable params: `lr` (default 1e-4)
        """
        params_dict_list = []

        # Alpha parameters — separate, higher learning rates
        params_dict_list.append({
            'params': [self.clip_vision_alpha], 'lr': 1e-3,
        })
        params_dict_list.append({
            'params': [self.clip_text_alpha], 'lr': 3e-3,
        })

        # ViT backbone — fine-tune
        self._add_params(self.vit, lr, params_dict_list)

        # CLIP text prompt tokens (learnable — must be trained from random init)
        if hasattr(self.clip_text_encoder, 'prompt_tokens'):
            params_dict_list.append({
                'params': [self.clip_text_encoder.prompt_tokens], 'lr': lr * 10,
            })

        # Projection layers
        self._add_params(self.deepfake_proj, lr, params_dict_list)
        self._add_params(self.vision_proj, lr, params_dict_list)
        self._add_params(self.text_proj, lr, params_dict_list)

        # Classifier head
        self._add_params(self.output, lr, params_dict_list)

        # BridgeAdapter components
        self._add_params(self.clip_reduction, lr, params_dict_list)
        self._add_params(self.bridge_adapter_proj, lr, params_dict_list)
        for blk in self.bridge_adapter:
            self._add_params(blk, lr, params_dict_list)
        for lin in self.linear_vit_lst:
            self._add_params(lin, lr, params_dict_list)

        return params_dict_list

    @staticmethod
    def _add_params(module, lr, params_dict_list):
        """Helper: append a {'params': ..., 'lr': ...} dict for a module."""
        params_dict_list.append({
            'params': module.parameters(), 'lr': lr,
        })
