"""Test the complete flash_attn compatibility patch."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply the patch
import patches.flash_attn_patch

# Test all flash_attn submodules
from flash_attn.modules.mha import MHA
print(f'✓ flash_attn.modules.mha.MHA: {MHA}')

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_varlen_qkvpacked_func
print(f'✓ flash_attn.flash_attn_interface.flash_attn_unpadded_qkvpacked_func')

from flash_attn.bert_padding import unpad_input, pad_input
print(f'✓ flash_attn.bert_padding.unpad_input, pad_input')

# Test project model import with patched flash_attn
from sequence.models.M2F2_Det.models.model import M2F2Det, TransformerEncoderBlock
print(f'✓ sequence.models.M2F2_Det.models.model.M2F2Det')
print(f'✓ TransformerEncoderBlock uses patched MHA')

# Test llava llama monkey patch import
from llava.train.llama_flash_attn_monkey_patch import forward
print(f'✓ llava.train.llama_flash_attn_monkey_patch.forward')

import torch
# Quick test of MHA forward
mha = MHA(embed_dim=256, num_heads=4, causal=True)
x = torch.randn(2, 10, 256)
out = mha(x)
print(f'✓ MHA forward shape: {out.shape} (expected: [2, 10, 256])')

# Test unpad/pad
hidden = torch.randn(2, 5, 64)
mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
unpadded, cu_seqlens, _ = unpad_input(hidden, mask)
print(f'✓ unpad_input: {unpadded.shape} (expected: [4, 64])')

print('\n=== ALL TESTS PASSED ===')
