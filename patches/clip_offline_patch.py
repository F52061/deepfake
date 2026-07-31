"""
Offline CLIP loader patch for M2F2-Det on Windows.
Redirects CLIP downloads to local directory.
"""
import os
import torch
import torch.nn as nn
from transformers import (
    CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel,
    CLIPTextConfig, CLIPTextModel, AutoTokenizer, AutoConfig
)

# Path to local CLIP model (downloaded by user)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CLIP_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'clip-vit-large-patch14-336')

# ===== Patch 1: CLIPVisionTower (load from local dir) =====
import llava.model.multimodal_encoder.clip_encoder as clip_encoder_module

def patched_load_model(self, device_map=None):
    if self.is_loaded:
        return
    print(f'Loading CLIP vision tower from: {LOCAL_CLIP_DIR}')
    self.image_processor = CLIPImageProcessor.from_pretrained(LOCAL_CLIP_DIR)
    self.vision_tower = CLIPVisionModel.from_pretrained(LOCAL_CLIP_DIR)
    self.vision_tower = self.vision_tower.half()  # match LLaMA fp16
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True

def patched_init(self, vision_tower, args, delay_load=False):
    nn.Module.__init__(self)
    self.is_loaded = False
    self.vision_tower_name = vision_tower
    self.select_layer = args.mm_vision_select_layer
    self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
    if not delay_load:
        self.load_model()
    elif getattr(args, 'unfreeze_mm_vision_tower', False):
        self.load_model()
    else:
        self.cfg_only = CLIPVisionConfig.from_pretrained(LOCAL_CLIP_DIR)

clip_encoder_module.CLIPVisionTower.load_model = patched_load_model
clip_encoder_module.CLIPVisionTower.__init__ = patched_init

# ===== Patch 2: Redirect openai/clip to local dir for all modules =====
def _redirect_to_local(func):
    def wrapper(pretrained_model_name_or_path, *args, **kwargs):
        if isinstance(pretrained_model_name_or_path, str) and 'openai/clip' in pretrained_model_name_or_path:
            return func(LOCAL_CLIP_DIR, *args, **kwargs)
        return func(pretrained_model_name_or_path, *args, **kwargs)
    return wrapper

CLIPTextModel.from_pretrained = _redirect_to_local(CLIPTextModel.from_pretrained)
CLIPVisionModel.from_pretrained = _redirect_to_local(CLIPVisionModel.from_pretrained)
CLIPImageProcessor.from_pretrained = _redirect_to_local(CLIPImageProcessor.from_pretrained)
AutoConfig.from_pretrained = _redirect_to_local(AutoConfig.from_pretrained)
AutoTokenizer.from_pretrained = _redirect_to_local(AutoTokenizer.from_pretrained)
