"""Test the flash_attn patch and model imports."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply flash_attn patch first
import patches.flash_attn_patch

# Now test importing with flash_attn mocked
from flash_attn.modules.mha import MHA
print(f'MHA class: {MHA}')
print('flash_attn patch applied OK')

# Test project model import
from sequence.models.M2F2_Det.models.model import M2F2Det, TransformerEncoderBlock
print(f'M2F2Det class: {M2F2Det}')
print(f'TransformerEncoderBlock: {TransformerEncoderBlock}')
print('Project model imports OK')

# Test dataset imports
from dataset import ImageFolderH5Dataset, ImageFolderH5Dataset_inference
print('Dataset imports OK')

# Quick model instantiation test
import torch
model = M2F2Det(
    clip_text_encoder_name="openai/clip-vit-large-patch14-336",
    clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
    deepfake_encoder_name='efficientnet_b4',
    hidden_size=1024,
    load_vision_encoder=False,
)
print(f'Model created: {type(model).__name__}')
print('ALL PROJECT IMPORTS AND INITIALIZATION OK')
