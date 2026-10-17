"""Frozen native ViT detector, including its original two-class head.

Input is independently preprocessed RGB, not CLIP-normalized pixels. No CLIP,
Bridge, feature projection or replacement classifier is constructed here.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeViTImageProcessor:
    """Explicit baseline pipeline; verify it against PDI before claiming AUC.

    PIL RGB -> float32 [0,1] -> resize 224 bilinear (no antialias) -> normalize.
    Tensor inputs are rejected to prevent accidental double normalization.
    """
    def __init__(self, size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        if size != 224 or len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError("Native ViT requires size=224 and three valid mean/std values")
        if not all(torch.isfinite(torch.tensor(float(v))) for v in (*mean, *std)):
            raise ValueError("Non-finite normalization constants")
        self.size, self.mean, self.std = size, tuple(mean), tuple(std)

    def preprocess(self, images, return_tensors="pt"):
        from PIL import Image
        import numpy as np
        if return_tensors != "pt":
            raise ValueError("Only PyTorch output is supported")
        images = images if isinstance(images, (list, tuple)) else [images]
        if not images:
            raise ValueError("At least one PIL image is required")
        pixels = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("Pass original PIL images, not processed tensors")
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            tensor = F.interpolate(tensor, (self.size, self.size), mode="bilinear", align_corners=False, antialias=False)[0]
            mean = tensor.new_tensor(self.mean).view(3, 1, 1)
            std = tensor.new_tensor(self.std).view(3, 1, 1)
            pixels.append((tensor - mean) / std)
        return {"pixel_values": torch.stack(pixels)}


def build_native_backbone():
    # Direct construction uses exactly the local ViT-B/16 architecture, avoiding
    # timm registry defaults and any pretrained download. Head is retained.
    from vit_module.vit_adaptive_mattn_aps import VisionTransformer
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=768,
                             depth=12, num_heads=12, num_classes=2)


def _native_state(checkpoint, expected):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a tensor state_dict")
    nested = [key for key in ("state_dict", "model_state_dict", "model")
              if key in checkpoint and isinstance(checkpoint[key], Mapping)]
    if len(nested) > 1:
        raise ValueError("Ambiguous checkpoint wrappers")
    state = checkpoint[nested[0]] if nested else checkpoint
    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Use a pure native ViT state_dict; no fusion or mixed metadata")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if len(state) != len(checkpoint[nested[0]] if nested else checkpoint):
        raise ValueError("Duplicate checkpoint keys after stripping module prefix")
    for prefix in ("model.", "vit."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
            break
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Native ViT checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}. "
                         "The original head.weight/head.bias are required; fusion weights are not accepted.")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.is_meta:
            raise ValueError(f"Wrong shape or meta tensor for {key}: {tuple(value.shape)}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
    return state


class NativeViTDetector(nn.Module):
    uses_clip_features = False

    def __init__(self, fake_logit_index, preprocessing=None, backbone=None):
        super().__init__()
        if fake_logit_index not in (0, 1):
            raise ValueError("Specify native fake_logit_index explicitly as 0 or 1")
        self.fake_logit_index = fake_logit_index
        self.vit = backbone if backbone is not None else build_native_backbone()
        if not isinstance(self.vit.head, nn.Linear) or self.vit.head.out_features != 2:
            raise ValueError("Native ViT must retain its original two-class Linear head")
        self.image_processor = NativeViTImageProcessor(**(preprocessing or {}))
        # Persistent guards also survive a correctly saved full HF checkpoint.
        # An old LLaVA/fusion checkpoint does not contain them and stays blocked
        # until load_deepfake_encoder() explicitly loads the original native head.
        self.register_buffer("_native_checkpoint_verified", torch.tensor(False))
        self.register_buffer("_native_checkpoint_sha256", torch.zeros(32, dtype=torch.uint8))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent LLaVA.train() must not enable stochastic native detector paths.
        super().train(False)
        return self

    def load_checkpoint(self, path, expected_sha256=None):
        self._native_checkpoint_verified.fill_(False)
        path = Path(path)
        actual_hash = file_sha256(path)
        if expected_sha256 is not None and actual_hash.lower() != expected_sha256.lower():
            raise ValueError("Native checkpoint SHA256 does not match the configured checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = _native_state(checkpoint, self.vit.state_dict())
        if any(p.is_meta for p in self.vit.parameters()):
            raise RuntimeError("Materialize the native detector before loading its checkpoint; disk/meta offload is unsupported")
        self.vit.load_state_dict(state, strict=True)
        self._native_checkpoint_sha256.copy_(torch.tensor(list(bytes.fromhex(actual_hash)), dtype=torch.uint8,
                                                        device=self._native_checkpoint_sha256.device))
        self._native_checkpoint_verified.fill_(True)
        self.requires_grad_(False)
        self.eval()
        return {"checkpoint": str(path), "sha256": actual_hash, "loaded_keys": len(state),
                "head_loaded": True, "native_fake_logit_index": self.fake_logit_index,
                "native_class_order": ["fake", "real"] if self.fake_logit_index == 0 else ["real", "fake"]}

    def forward(self, images, clip_vision_features=None, use_cached_clip_text_features=False):
        if not bool(self._native_checkpoint_verified.item()):
            raise RuntimeError("Native ViT weights are not verified. Call load_checkpoint()/load_deepfake_encoder() first.")
        if clip_vision_features is not None:
            raise ValueError("vit_native does not consume CLIP features")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("Expected independently normalized native pixels [batch,3,224,224]")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("Native pixels must be finite floating-point tensors")
        parameter = next(self.vit.parameters())
        return self.vit(images.to(device=parameter.device, dtype=parameter.dtype))

    def token_logits(self, native_logits):
        """Explicit token order [fake, real], independent of native head order."""
        return native_logits[:, [self.fake_logit_index, 1 - self.fake_logit_index]]
