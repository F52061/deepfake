"""Experimental, in-memory repair. Changes the bridge function, needs retraining.

The calling detector retains its existing sequence-first outer interface.
Do not treat repaired old weights as a trained replacement checkpoint.
"""
from types import MethodType

import torch
from torch import nn


def _sequence_first_block(self, x):
    dtype = next(self.attn.parameters()).dtype
    # flash_attn MHA and the project's shim both expect batch-first input.
    batch_first = x.transpose(0, 1).contiguous().to(dtype)
    attention = self.attn(batch_first).transpose(0, 1).contiguous().to(x.dtype)
    x = self.norm1(x + self.dropout(attention))
    return self.norm2(x + self.dropout(self.ffn(x)))


def _sample_preserving_projection(self, x, batch_size):
    if x.ndim != 3 or x.shape[1] != batch_size:
        raise ValueError("Expected sequence-first [tokens, batch, channels]")
    # Transpose BEFORE flattening so each output row contains only one image.
    scalar = self.bridge_adapter_proj(x.transpose(0, 1).contiguous())
    return self.bridge_adapter_reduction(scalar.reshape(batch_size, -1))


def apply_layout_candidate(model):
    """Preserve loaded Linear weights and dtypes; remove only scalar LayerNorm.

    Call after loading the legacy state_dict, before creating an optimizer.
    No parameters are added, and no files are written. Reapplication is refused.
    The removed LN's learned bias is intentionally not preserved: the legacy
    bridge was constant. This is a new function, not checkpoint equivalence.
    """
    if getattr(model, "_bridge_layout_candidate_applied", False):
        raise ValueError("Candidate already applied")
    wrap = model.bridge_adapter_proj
    layers = list(wrap.bridge_adapter_proj.children())
    if len(layers) != 3 or not isinstance(layers[1], nn.Linear):
        raise ValueError("Expected legacy View -> Linear -> LayerNorm(1)")
    if not isinstance(layers[2], nn.LayerNorm) or layers[2].normalized_shape != (1,):
        raise ValueError("Expected scalar LayerNorm")
    wrap.bridge_adapter_proj = nn.Sequential(*layers[:2])
    wrap.forward = MethodType(_sample_preserving_projection, wrap)
    for block in model.bridge_adapter:
        block.forward = MethodType(_sequence_first_block, block)
    model._bridge_layout_candidate_applied = True
    return model
