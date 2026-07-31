"""
ViT_M2F2Det_Unified: Multi-modal Deepfake Detector with ViT backbone.

Two fusion modes selectable via `fusion_mode` parameter:

  fusion_mode='cosine' (方案A):
    ViT CLS + CLIP final features
    Fusion: cosine_similarity(patches, text) → scalar scores
    Output: cat([scores×α_t, CLS×α_v, ViT_CLS]) → Linear(2*h+576, 2)
    - No ViT intermediate hooks
    - No BridgeAdapter
    - Supports heatmap visualization

  fusion_mode='bridge' (方案B):
    ViT CLS + ViT blocks[3][6][9] hooks + CLIP layers 6/10/14
    Fusion: 3-stage BridgeAdapter (TransformerEncoderBlock)
    Output: cat([CLS×α_v, bridge_128, ViT_CLS]) → Linear(2*h+128, 2)
    - 3 ViT intermediate hooks
    - 3×TransformerEncoderBlock + BridgeAdapter_Proj

Usage:
    from vit_module.vit_m2f2_detector_unified import ViT_M2F2Det_Unified as ViT_M2F2Det

    # 方案A: Cosine
    model = ViT_M2F2Det(fusion_mode='cosine', hidden_size=1024, ...)

    # 方案B: BridgeAdapter
    model = ViT_M2F2Det(fusion_mode='bridge', hidden_size=768, ...)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from llava.model.deepfake.M2F2Det.text_encoder import CLIPTextEncoder
from llava.model.deepfake.M2F2Det.vision_encoder import CLIPVisionEncoder
from flash_attn.modules.mha import MHA

try:
    from .vit_adaptive_mattn_aps import vit_base_patch16_224
except ImportError:
    from vit_adaptive_mattn_aps import vit_base_patch16_224


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

VIT_EMBED_DIM   = 768
CLIP_HIDDEN_SIZE = 1024
CLIP_NUM_PATCHES = 576

# BridgeAdapter constants
BRIDGE_EMBED_DIM  = 64
BRIDGE_NUM_HEADS  = 4
BRIDGE_FF_DIM     = 32
BRIDGE_TOTAL_SEQ_LEN = 2316


# ═══════════════════════════════════════════════════════════════════
# Utility Modules (BridgeAdapter)
# ═══════════════════════════════════════════════════════════════════

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with flash_attn MHA + FFN."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, causal=False):
        super().__init__()
        self.attn = MHA(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, causal=causal,
        ).to(torch.bfloat16)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim),
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
    BridgeAdapter output → compact embedding [B, bridge_adapter_dim].
    [total_seq, B, embed_dim] → View(-1, embed_dim) → Linear(embed_dim→1)
    → LN → View(B, total_seq) → Linear(total_seq→out_dim) → LN
    """
    def __init__(self, embed_dim, out_dim, total_seq_len):
        super().__init__()
        self.bridge_adapter_proj = nn.Sequential(
            View(-1, embed_dim),
            nn.Linear(embed_dim, 1),
            nn.LayerNorm(1),
        )
        self.bridge_adapter_reduction = nn.Sequential(
            nn.Linear(total_seq_len, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, batch_size):
        x = self.bridge_adapter_proj(x)
        x = x.view(batch_size, -1)
        x = self.bridge_adapter_reduction(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Heatmap Visualization (Cosine mode only)
# ═══════════════════════════════════════════════════════════════════

def save_heatmap_visualization(
    images, clip_scores, clip_vision_cls, vit_features,
    output, labels=None, save_dir="./outputs/heatmaps",
    prefix="", max_samples=4
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(save_dir, exist_ok=True)
    B = min(images.shape[0], max_samples)
    num_patches = int(np.sqrt(clip_scores.shape[1]))
    patch_size = 336 // num_patches

    for i in range(B):
        img = images[i].detach().cpu()
        mean = CLIP_MEAN.view(3, 1, 1)
        std = CLIP_STD.view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()

        raw_scores = clip_scores[i].detach().cpu().numpy()
        heatmap = raw_scores.reshape(num_patches, num_patches)

        probs = F.softmax(output[i], dim=0).detach().cpu().numpy()
        pred_label = 'FAKE' if probs[1] > probs[0] else 'REAL'
        gt_label = 'FAKE' if labels is not None and labels[i] == 1 else 'REAL' if labels is not None else 'N/A'
        confidence = float(max(probs) * 100)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image\nGT: {gt_label} | Pred: {pred_label} ({confidence:.1f}%)', fontsize=11)
        axes[0].axis('off')

        im = axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
        axes[1].set_title(f'CLIP Patch-Text Similarity\n(mean={raw_scores.mean():.3f}, std={raw_scores.std():.3f})', fontsize=11)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        hm_resized = np.clip(heatmap, -1, 1)
        hm_resized = (hm_resized + 1) / 2
        hm_resized = np.kron(hm_resized, np.ones((patch_size, patch_size)))
        hm_resized = hm_resized[:336, :336]
        axes[2].imshow(img_np)
        axes[2].imshow(hm_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Overlay: Similarity on Image', fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()
        fname = f"{prefix}sample_{i}" if prefix else f"sample_{i}"
        plt.savefig(os.path.join(save_dir, f'{fname}_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Unified Detector
# ═══════════════════════════════════════════════════════════════════

class ViT_M2F2Det_Unified(nn.Module):
    """
    Unified multi-modal deepfake detector with selectable fusion mode.

    Args:
        fusion_mode: 'cosine' (方案A) or 'bridge' (方案B)
        hidden_size: projection dimension (768 recommended for bridge, 1024 for cosine)
        Other args: shared between both modes.
    """

    def __init__(
        self,
        fusion_mode: str               = 'cosine',
        deepfake_encoder_name: str     = 'vit',    # API compatibility, unused
        clip_text_encoder_name: str    = "openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name: str  = "openai/clip-vit-large-patch14-336",
        hidden_size: int               = 768,
        vision_dtype: torch.dtype      = torch.float32,
        text_dtype: torch.dtype        = torch.float32,
        deepfake_dtype: torch.dtype    = torch.float32,
        load_vision_encoder: bool      = True,
        pretrained: bool               = False,
        save_heatmap: bool             = False,
        heatmap_dir: str               = "./outputs/heatmaps",
    ):
        super().__init__()
        assert fusion_mode in ('cosine', 'bridge'), f"fusion_mode must be 'cosine' or 'bridge', got {fusion_mode}"
        self.fusion_mode = fusion_mode
        self.save_heatmap = save_heatmap
        self.heatmap_dir = heatmap_dir

        # ════════════════════════════════════════════════════════
        # Shared components (both modes)
        # ════════════════════════════════════════════════════════

        # ViT backbone
        self.vit = vit_base_patch16_224(pretrained=pretrained, num_classes=0)
        self.vit_dtype = deepfake_dtype
        self.vit.to(deepfake_dtype)

        # Deepfake projection: ViT CLS [768] → hidden_size
        self.deepfake_proj = nn.Sequential(
            nn.Linear(VIT_EMBED_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.deepfake_proj.to(deepfake_dtype)

        # CLIP text encoder
        self.clip_text_encoder = CLIPTextEncoder(
            clip_text_encoder_name, dtype=text_dtype
        )

        # CLIP vision encoder
        if load_vision_encoder:
            self.clip_vision_encoder = CLIPVisionEncoder(
                clip_vision_encoder_name, dtype=vision_dtype
            )
        else:
            self.clip_vision_encoder = None

        # Projection layers
        clip_text_hidden = self.clip_text_encoder.model.config.hidden_size

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

        # Learnable fusion weights
        self.clip_vision_alpha = nn.Parameter(torch.tensor(0.5))
        self.clip_text_alpha   = nn.Parameter(torch.tensor(4.0))
        self.clip_vision_alpha.to(vision_dtype)
        self.clip_text_alpha.to(text_dtype)

        # ════════════════════════════════════════════════════════
        # Mode-specific components
        # ════════════════════════════════════════════════════════

        if self.fusion_mode == 'bridge':
            # ── ViT intermediate hooks (blocks[3/6/9]) ──────────
            self.vit_block_outputs: dict = {}
            def _make_hook(name: str):
                def hook(module, input, output):
                    self.vit_block_outputs[name] = output
                return hook
            self.vit.blocks[3].register_forward_hook(_make_hook("b_1"))
            self.vit.blocks[6].register_forward_hook(_make_hook("b_2"))
            self.vit.blocks[9].register_forward_hook(_make_hook("b_3"))

            # ── CLIP reduction: 1024 → 64 ──────────────────────
            self.clip_reduction = nn.Linear(CLIP_HIDDEN_SIZE, BRIDGE_EMBED_DIM)

            # ── ViT patch projections: 768 → 64 (one per level) ─
            self.linear_vit_1 = nn.Linear(VIT_EMBED_DIM, BRIDGE_EMBED_DIM)
            self.linear_vit_2 = nn.Linear(VIT_EMBED_DIM, BRIDGE_EMBED_DIM)
            self.linear_vit_3 = nn.Linear(VIT_EMBED_DIM, BRIDGE_EMBED_DIM)
            self.linear_vit_lst = nn.ModuleList([
                self.linear_vit_1, self.linear_vit_2, self.linear_vit_3,
            ])

            # ── 3-stage TransformerBridge ─────────────────────
            self.bridge_adapter = nn.ModuleList([
                TransformerEncoderBlock(BRIDGE_EMBED_DIM, BRIDGE_NUM_HEADS, BRIDGE_FF_DIM)
                for _ in range(3)
            ])

            # ── Bridge final projection ─────────────────────────
            self.bridge_adapter_proj = BridgeAdapter_Proj_ViT(
                embed_dim=BRIDGE_EMBED_DIM,
                out_dim=128,
                total_seq_len=BRIDGE_TOTAL_SEQ_LEN,
            )

            # ── Output: hidden + 128 + hidden ──────────────────
            self.output = nn.Linear(2 * hidden_size + 128, 2)

        else:
            # cosine mode — no bridge components
            # Output: 576 (scores) + hidden + hidden
            self.output = nn.Linear(CLIP_NUM_PATCHES + 2 * hidden_size, 2)

        self.output.to(deepfake_dtype)

        # Cache & metadata
        self.cached_clip_text_features = None
        self.hidden_size = hidden_size
        self.vision_dtype = vision_dtype
        self.text_dtype = text_dtype
        self.deepfake_dtype = deepfake_dtype

        # Weight initialization
        self._init_new_components()

    # ════════════════════════════════════════════════════════════
    # Initialization
    # ════════════════════════════════════════════════════════════

    def _init_new_components(self):
        components = [
            self.deepfake_proj, self.vision_proj, self.text_proj, self.output,
        ]
        if self.fusion_mode == 'bridge':
            components += [self.clip_reduction, self.bridge_adapter_proj]
        for comp in components:
            for m in comp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ════════════════════════════════════════════════════════════
    # Preprocessing
    # ════════════════════════════════════════════════════════════

    def _preprocess_for_vit(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        mean = CLIP_MEAN.to(device).view(1, 3, 1, 1)
        std  = CLIP_STD.to(device).view(1, 3, 1, 1)
        x = images * std + mean
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    # ════════════════════════════════════════════════════════════
    # Forward
    # ════════════════════════════════════════════════════════════

    def forward(
        self,
        images: torch.Tensor,
        clip_vision_features: Optional[torch.Tensor] = None,
        use_cached_clip_text_features: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):
        B = images.shape[0]

        # ── Shared: ViT backbone ─────────────────────────────
        vit_input = self._preprocess_for_vit(images)
        vit_input = vit_input.to(self.vit_dtype)
        vit_out = self.vit.forward_features(vit_input)              # [B, 197, 768]
        vit_cls_token = vit_out[:, 0, :]                            # [B, 768]
        vit_features = self.deepfake_proj(vit_cls_token)            # [B, hidden_size]

        # ── Shared: CLIP text ─────────────────────────────────
        if use_cached_clip_text_features:
            if self.cached_clip_text_features is None:
                self.cached_clip_text_features = self.clip_text_encoder()
            clip_text_features = self.cached_clip_text_features
        else:
            clip_text_features = self.clip_text_encoder()

        clip_text_features = clip_text_features.to(self.text_dtype)
        clip_text_features = self.text_proj(clip_text_features)     # [B, hidden_size]

        device = vit_input.device

        # ═══════════════════════════════════════════════════════
        # Mode-specific fusion
        # ═══════════════════════════════════════════════════════

        if self.fusion_mode == 'bridge':
            return self._forward_bridge(
                B, device, images, vit_out, vit_features,
                clip_text_features, clip_vision_features,
            )
        else:
            return self._forward_cosine(
                B, device, images, vit_features,
                clip_text_features, clip_vision_features,
            )

    # ════════════════════════════════════════════════════════════
    # Cosine mode forward
    # ════════════════════════════════════════════════════════════

    def _forward_cosine(self, B, device, images, vit_features,
                         clip_text_features, clip_vision_features):
        # CLIP vision
        if clip_vision_features is None:
            if self.clip_vision_encoder is not None:
                clip_vision_features = self.clip_vision_encoder(images)[-1]
            else:
                raise ValueError(
                    "clip_vision_features must be provided when load_vision_encoder=False"
                )

        clip_vision_features = clip_vision_features.to(self.vision_dtype)
        clip_vision_features = self.vision_proj(clip_vision_features)
        clip_vision_cls = clip_vision_features[:, 0, :]
        clip_vision_patches = clip_vision_features[:, 1:, :]

        # Cosine similarity
        clip_scores = F.cosine_similarity(
            clip_vision_patches,
            clip_text_features.unsqueeze(1).expand(B, -1, -1),
            dim=-1,
        )

        clip_vision_cls = self.clip_vision_alpha * clip_vision_cls
        clip_scores_scaled = self.clip_text_alpha * clip_scores

        # Fusion
        features = torch.cat(
            [clip_scores_scaled, clip_vision_cls, vit_features], dim=-1
        )
        output = self.output(features)

        # Heatmap visualization
        if self.save_heatmap and (not self.training):
            save_heatmap_visualization(
                images=images, clip_scores=clip_scores_scaled,
                clip_vision_cls=clip_vision_cls, vit_features=vit_features,
                output=output, labels=labels,
                save_dir=self.heatmap_dir,
            )

        return output

    # ════════════════════════════════════════════════════════════
    # Bridge mode forward
    # ════════════════════════════════════════════════════════════

    def _forward_bridge(self, B, device, images, vit_out, vit_features,
                         clip_text_features, clip_vision_features):
        # ViT intermediate hooks
        vit_feat_0 = self.vit_block_outputs["b_1"][:, 1:, :]        # [B, 196, 768]
        vit_feat_1 = self.vit_block_outputs["b_2"][:, 1:, :]        # [B, 196, 768]
        vit_feat_2 = self.vit_block_outputs["b_3"][:, 1:, :]        # [B, 196, 768]

        # CLIP vision
        if clip_vision_features is None:
            if self.clip_vision_encoder is not None:
                clip_0, clip_1, clip_2, clip_vision_features = \
                    self.clip_vision_encoder(images)
            else:
                raise ValueError(
                    "clip_vision_features must be a 4-tuple "
                    "when load_vision_encoder=False"
                )
        else:
            if isinstance(clip_vision_features, (list, tuple)) and len(clip_vision_features) == 4:
                clip_0, clip_1, clip_2, clip_vision_features = clip_vision_features
            else:
                raise ValueError("BridgeAdapter requires 4-tuple CLIP features")

        clip_vision_features = clip_vision_features.to(self.vision_dtype)
        clip_vision_features = self.vision_proj(clip_vision_features)
        clip_vision_cls = clip_vision_features[:, 0, :]

        # BridgeAdapter 3-stage
        vit_feat_lst  = [vit_feat_0, vit_feat_1, vit_feat_2]
        clip_feat_lst = [clip_0, clip_1, clip_2]

        bridge_adapter_output = None
        for i, (vit_feat, clip_feat) in enumerate(zip(vit_feat_lst, clip_feat_lst)):
            clip_feat = clip_feat.to(device)
            clip_feat = self.clip_reduction(clip_feat)               # [B, 576, 64]

            vit_feat = vit_feat.to(device)
            self.linear_vit_lst[i] = self.linear_vit_lst[i].to(device)
            vit_feat = self.linear_vit_lst[i](vit_feat)               # [B, 196, 64]

            if bridge_adapter_output is None:
                combined = torch.cat((vit_feat, clip_feat), dim=1)
            else:
                bridge_adapter_output = bridge_adapter_output.permute(1, 0, 2)
                combined = torch.cat((bridge_adapter_output, vit_feat, clip_feat), dim=1)

            combined = combined.permute(1, 0, 2)
            self.bridge_adapter[i] = self.bridge_adapter[i].to(device)
            bridge_adapter_output = self.bridge_adapter[i](combined)

        clip_adapt_embed = self.bridge_adapter_proj(bridge_adapter_output, B)
        clip_adapt_embed = self.clip_text_alpha * clip_adapt_embed

        # Final fusion
        clip_vision_cls = self.clip_vision_alpha * clip_vision_cls.to(self.deepfake_dtype)

        features = torch.cat([
            clip_vision_cls, clip_adapt_embed, vit_features,
        ], dim=-1)

        output = self.output(features)
        return output

    # ════════════════════════════════════════════════════════════
    # Weight loading
    # ════════════════════════════════════════════════════════════

    def load_vit_backbone(self, checkpoint_path: str, verbose: bool = True):
        """
        Load pre-trained weights from a checkpoint.

        Supports two checkpoint formats:
          1. PDI training checkpoint: keys start with 'model.xxx'
             → mapped to 'vit.xxx' (only loads ViT backbone)
          2. Our trained bridge checkpoint: keys already match model.state_dict()
             → full load (all components: vit + bridge + projections + output)

        In both cases, mismatched keys are reported but not fatal.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {}
        loaded_count = 0
        skipped_head = 0
        direct_match = 0

        # Check format: if any key starts with 'model.', use PDI format
        has_model_prefix = any(k.startswith('model.') for k in ckpt.keys() if not k.startswith('model.head.'))

        if has_model_prefix:
            # PDI checkpoint format: model.xxx → vit.xxx
            for k, v in ckpt.items():
                if k.startswith('model.head.'):
                    skipped_head += 1
                    continue
                if k.startswith('model.'):
                    vit_key = 'vit.' + k[6:]
                    if vit_key in self.state_dict():
                        state_dict[vit_key] = v.to(self.state_dict()[vit_key].dtype)
                        loaded_count += 1
        else:
            # Our trained checkpoint: direct key match
            model_sd = self.state_dict()
            for k, v in ckpt.items():
                if k in model_sd:
                    state_dict[k] = v.to(model_sd[k].dtype)
                    loaded_count += 1
                elif k.startswith('vit.'):
                    # ViT key that doesn't match architecture somehow
                    if verbose:
                        print(f'  Skipping unmatched vit key: {k}')
                else:
                    # May be classification head or other non-essential
                    if 'head' in k or 'classifier' in k:
                        skipped_head += 1

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if verbose:
            fmt = 'PDI' if has_model_prefix else 'trained'
            print(
                f'[ViT_M2F2Det_Unified/{self.fusion_mode}] '
                f'Loaded {loaded_count} keys ({fmt} format), '
                f'skipped {skipped_head} head keys.'
            )
            print(f'  Missing (will use random init): {len(missing)}')
            if len(missing) > 0 and verbose > 1:
                for mk in sorted(missing)[:10]:
                    print(f'    - {mk}')
                if len(missing) > 10:
                    print(f'    ... and {len(missing)-10} more')
            print(f'  Unexpected keys: {len(unexpected)}')

        return missing, unexpected

    # ════════════════════════════════════════════════════════════
    # Optimizer parameter groups
    # ════════════════════════════════════════════════════════════

    def assign_lr_dict_list(self, lr: float = 1e-4):
        """
        Return parameter-group dicts for torch.optim.Adam.
        Includes BridgeAdapter components only in 'bridge' mode.
        """
        params_dict_list = []

        # Alpha params
        params_dict_list.append({'params': [self.clip_vision_alpha], 'lr': 1e-3})
        params_dict_list.append({'params': [self.clip_text_alpha],   'lr': 3e-3})

        # ViT backbone
        self._add_params(self.vit, lr, params_dict_list)

        # prompt_tokens
        if hasattr(self.clip_text_encoder, 'prompt_tokens'):
            params_dict_list.append({
                'params': [self.clip_text_encoder.prompt_tokens], 'lr': lr * 10,
            })

        # Projection layers
        self._add_params(self.deepfake_proj, lr, params_dict_list)
        self._add_params(self.vision_proj, lr, params_dict_list)
        self._add_params(self.text_proj, lr, params_dict_list)

        # Classifier
        self._add_params(self.output, lr, params_dict_list)

        # BridgeAdapter-specific (only in bridge mode)
        if self.fusion_mode == 'bridge':
            self._add_params(self.clip_reduction, lr, params_dict_list)
            self._add_params(self.bridge_adapter_proj, lr, params_dict_list)
            for blk in self.bridge_adapter:
                self._add_params(blk, lr, params_dict_list)
            for lin in self.linear_vit_lst:
                self._add_params(lin, lr, params_dict_list)

        return params_dict_list

    @staticmethod
    def _add_params(module, lr, params_dict_list):
        params_dict_list.append({'params': module.parameters(), 'lr': lr})
