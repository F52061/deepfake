"""
ViT-M2F2Det: Multi-modal Deepfake Detector with ViT backbone.

Replaces the DenseNet backbone in M2F2Det with the ViT (vit_adaptive_mattn_aps)
trained via train_Ama_aps.py.

Architecture:
    ViT (backbone) → CLS token [B,768] → deepfake_proj (bridge adapter) → [B,1024]
    CLIP Vision → [B,577,1024] → vision_proj → [B,577,1024]
      ├─ cls × α_v ──→ [B,1024]
      └─ patches ──→ cos_sim(text) ──→ scores [B,576] × α_t
    concat([scores, cls×α, vit_feat]) ──→ [B,2624] ──→ self.output (Mb @ x) ──→ [B,2]
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from llava.model.deepfake.M2F2Det.text_encoder import CLIPTextEncoder
from llava.model.deepfake.M2F2Det.vision_encoder import CLIPVisionEncoder

try:
    from .vit_adaptive_mattn_aps import vit_base_patch16_224  # as package submodule
except ImportError:
    from vit_adaptive_mattn_aps import vit_base_patch16_224    # as top-level module


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


# ═════════════════════════════════════════════════════════════════════
# Heatmap Visualization Utility
# ═════════════════════════════════════════════════════════════════════
def save_heatmap_visualization(
    images, clip_scores, clip_vision_cls, vit_features,
    output, labels=None, save_dir="./outputs/heatmaps",
    prefix="", max_samples=4
):
    """
    Save heatmap visualizations of the fused features.

    For each sample (up to max_samples), saves a 3-panel figure:
    1. Original image (denormalized from CLIP norm)
    2. CLIP patch-text similarity heatmap (24x24 grid)
    3. Heatmap overlay on image

    Args:
        images: [B, 3, 336, 336] CLIP-normalized input
        clip_scores: [B, 576] CLIP patch-text similarity (x alpha_t)
        clip_vision_cls: [B, 1024] CLIP vision CLS features
        vit_features: [B, 1024] ViT deepfake features
        output: [B, 2] logits
        labels: [B] or None (0=real, 1=fake)
        save_dir: output directory for PNGs
        prefix: filename prefix
        max_samples: max samples to visualize
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(save_dir, exist_ok=True)
    B = min(images.shape[0], max_samples)
    num_patches = int(np.sqrt(clip_scores.shape[1]))  # 576 -> 24
    patch_size = 336 // num_patches  # 14

    for i in range(B):
        # Denormalize image: CLIP norm -> [0,1]
        img = images[i].detach().cpu()
        mean = CLIP_MEAN.view(3, 1, 1)
        std = CLIP_STD.view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()

        # CLIP patch-text similarity heatmap
        raw_scores = clip_scores[i].detach().cpu().numpy()  # [576]
        heatmap = raw_scores.reshape(num_patches, num_patches)  # [24,24]

        # Prediction
        probs = F.softmax(output[i], dim=0).detach().cpu().numpy()
        pred_label = 'FAKE' if probs[1] > probs[0] else 'REAL'
        gt_label = 'FAKE' if labels is not None and labels[i] == 1 else 'REAL' if labels is not None else 'N/A'
        confidence = float(max(probs) * 100)

        # 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Panel 1: Original image
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image\nGT: {gt_label} | Pred: {pred_label} ({confidence:.1f}%)', fontsize=11)
        axes[0].axis('off')

        # Panel 2: Heatmap
        im = axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
        axes[1].set_title(
            f'CLIP Patch-Text Similarity\n'
            f'(mean={raw_scores.mean():.3f}, std={raw_scores.std():.3f})',
            fontsize=11
        )
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Heatmap overlay on image
        hm_resized = np.clip(heatmap, -1, 1)
        hm_resized = (hm_resized + 1) / 2  # [0,1]
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


class ViT_M2F2Det(nn.Module):
    """
    Multi-modal deepfake detector using ViT as the image backbone.

    The detector fuses three modalities:
    1. ViT deepfake features (from the user's trained ViT)
    2. CLIP vision features (from the project's CLIP vision tower)
    3. CLIP text features (learnable prompt tokens)

    The fused features are classified by a linear layer (Mb) to [B,2] logits.
    """

    def __init__(
        self,
        clip_text_encoder_name: str = "openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        deepfake_encoder_name: str = 'vit',  # kept for API compatibility
        hidden_size: int = 1024,
        vision_dtype: torch.dtype = torch.float32,
        text_dtype: torch.dtype = torch.float32,
        deepfake_dtype: torch.dtype = torch.float32,
        load_vision_encoder: bool = True,
        pretrained: bool = False,
        save_heatmap: bool = False,
        heatmap_dir: str = "./outputs/heatmaps",
    ):
        super().__init__()

        self.save_heatmap = save_heatmap
        self.heatmap_dir = heatmap_dir

        # ViT backbone (replaces DenseNet)
        self.vit = vit_base_patch16_224(pretrained=pretrained, num_classes=0)
        self.vit_dtype = deepfake_dtype
        self.vit.to(deepfake_dtype)

        # Bridge adapter: ViT [768] -> CLIP space [1024]
        self.deepfake_proj = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.deepfake_proj.to(deepfake_dtype)

        # CLIP text encoder
        self.clip_text_encoder = CLIPTextEncoder(
            clip_text_encoder_name, dtype=text_dtype
        )

        # CLIP vision encoder (optional, usually provided externally)
        if load_vision_encoder:
            self.clip_vision_encoder = CLIPVisionEncoder(
                clip_vision_encoder_name, dtype=vision_dtype
            )
        else:
            self.clip_vision_encoder = None

        # CLIP feature projectors
        clip_text_hidden = self.clip_text_encoder.model.config.hidden_size  # 768
        clip_vision_hidden = 1024  # CLIP ViT-L/14-336 vision encoder
        self.vision_proj = nn.Sequential(
            nn.Linear(clip_vision_hidden, hidden_size),
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
        self.clip_text_alpha = nn.Parameter(torch.tensor(4.0))
        self.clip_vision_alpha.to(vision_dtype)
        self.clip_text_alpha.to(text_dtype)

        # Final classifier
        self.output = nn.Linear(2 * hidden_size + 576, 2)
        self.output.to(deepfake_dtype)

        # Cache for text features
        self.cached_clip_text_features = None

        self.hidden_size = hidden_size
        self.vision_dtype = vision_dtype
        self.text_dtype = text_dtype
        self.deepfake_dtype = deepfake_dtype

        # Init new layers
        self._init_new_components()

    def _init_new_components(self):
        for name, module in [
            ('deepfake_proj', self.deepfake_proj),
            ('vision_proj', self.vision_proj),
            ('text_proj', self.text_proj),
            ('output', self.output),
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _preprocess_for_vit(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        mean = CLIP_MEAN.to(device).view(1, 3, 1, 1)
        std = CLIP_STD.to(device).view(1, 3, 1, 1)
        x = images * std + mean
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    def forward(
        self,
        images: torch.Tensor,
        clip_vision_features: Optional[torch.Tensor] = None,
        use_cached_clip_text_features: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):
        B = images.shape[0]

        # 1. ViT deepfake features
        vit_input = self._preprocess_for_vit(images)
        vit_input = vit_input.to(self.vit_dtype)
        vit_out = self.vit.forward_features(vit_input)
        vit_cls = vit_out[:, 0]
        vit_features = self.deepfake_proj(vit_cls)

        # 2. CLIP vision features
        if clip_vision_features is None:
            if self.clip_vision_encoder is not None:
                # CLIPVisionEncoder returns 4-tuple: (clip_0, clip_1, clip_2, final_features)
                clip_vision_features = self.clip_vision_encoder(images)[-1]
            else:
                raise ValueError(
                    "clip_vision_features must be provided when "
                    "load_vision_encoder=False"
                )

        clip_vision_features = clip_vision_features.to(self.vision_dtype)
        clip_vision_features = self.vision_proj(clip_vision_features)
        clip_vision_cls = clip_vision_features[:, 0, :]
        clip_vision_patches = clip_vision_features[:, 1:, :]

        # 3. CLIP text features
        if use_cached_clip_text_features:
            if self.cached_clip_text_features is None:
                self.cached_clip_text_features = self.clip_text_encoder()
            clip_text_features = self.cached_clip_text_features
        else:
            clip_text_features = self.clip_text_encoder()

        clip_text_features = clip_text_features.to(self.text_dtype)
        clip_text_features = self.text_proj(clip_text_features)

        # 4. Cosine similarity: CLIP patches vs text features
        clip_scores = F.cosine_similarity(
            clip_vision_patches,
            clip_text_features.unsqueeze(1).expand(B, -1, -1),
            dim=-1,
        )

        clip_vision_cls = self.clip_vision_alpha * clip_vision_cls
        clip_scores_scaled = self.clip_text_alpha * clip_scores

        # 5. Fusion and classification
        features = torch.cat(
            [clip_scores_scaled, clip_vision_cls, vit_features], dim=-1
        )
        output = self.output(features)

        # 6. Save heatmap visualization (if enabled)
        if self.save_heatmap and (not self.training):
            save_heatmap_visualization(
                images=images,
                clip_scores=clip_scores_scaled,
                clip_vision_cls=clip_vision_cls,
                vit_features=vit_features,
                output=output,
                labels=labels,
                save_dir=self.heatmap_dir,
                prefix="",
                max_samples=4,
            )

        return output

    def load_vit_backbone(self, checkpoint_path: str, verbose: bool = True):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {}
        loaded_count = 0
        skipped_head = 0
        for k, v in ckpt.items():
            if k.startswith('model.head.'):
                skipped_head += 1
                continue
            if k.startswith('model.'):
                vit_key = 'vit.' + k[6:]
                if vit_key in self.state_dict():
                    state_dict[vit_key] = v.to(self.state_dict()[vit_key].dtype)
                    loaded_count += 1

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if verbose:
            print(f'[ViT_M2F2Det] Loaded {loaded_count} ViT backbone keys, '
                  f'skipped {skipped_head} classification head keys.')
            print(f'  Missing keys (new layers, expected): {len(missing)}')
            print(f'  Unexpected keys (none expected): {len(unexpected)}')

        return missing, unexpected
