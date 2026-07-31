"""
Flash Attention compatibility patch for Windows.
Replaces flash_attn with PyTorch-native implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from types import ModuleType


class MHA(nn.Module):
    """Drop-in replacement for flash_attn.modules.mha.MHA using PyTorch-native attention."""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, causal=False, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Wqkv combined
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.embed_dim = embed_dim
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Wqkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.Wqkv.bias is not None:
            nn.init.constant_(self.Wqkv.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, x_kv=None, key_padding_mask=None, *args, **kwargs):
        batch_size, seq_len, embed_dim = x.shape
        # Project to Q, K, V
        qkv = self.Wqkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (batch, heads, seq_len, head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        # Apply key_padding_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        return output


# ---- Stubs for flash_attn.flash_attn_interface ----
def flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False):
    """Fallback: basic attention for llama monkey patch."""
    # Unpack QKV
    total_tokens, _, hidden = qkv.shape
    num_heads = 1  # simplified; llama handles head splitting separately
    q, k, v = qkv[:, 0, :], qkv[:, 1, :], qkv[:, 2, :]
    scale = softmax_scale or (hidden ** -0.5)
    attn = torch.matmul(q.unsqueeze(1), k.unsqueeze(1).transpose(-2, -1)) * scale
    if causal:
        seq_len = q.size(0)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
        attn = attn + causal_mask[:seq_len, :seq_len]
    attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
    attn = F.dropout(attn, p=dropout_p, training=attn.requires_grad)
    out = torch.matmul(attn, v.unsqueeze(1)).squeeze(1)
    return out, None

def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False):
    return flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal)


# Stub for bert_padding
def unpad_input(hidden_states, attention_mask):
    """Simplified unpad: flatten non-padded tokens."""
    # attention_mask: (batch, seq_len), 1 = keep, 0 = pad
    batch_size, seq_len = attention_mask.shape
    indices = attention_mask.nonzero(as_tuple=True)
    total_tokens = indices[0].size(0)
    if total_tokens == 0:
        return hidden_states.view(-1, hidden_states.size(-1))[0:0], None, None
    seqlens_in_batch = attention_mask.sum(dim=-1).int()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    output = hidden_states[indices[0], indices[1]]
    return output, cu_seqlens.to(hidden_states.device), None

def pad_input(hidden_states, indices, batch_size, seq_len):
    """Simplified pad: scatter back into (batch, seq_len, hidden)."""
    if hidden_states.size(0) == 0:
        return hidden_states.new_zeros(batch_size, seq_len, hidden_states.size(-1))
    out = hidden_states.new_zeros(batch_size, seq_len, hidden_states.size(-1))
    out[indices[0], indices[1]] = hidden_states
    return out


# ---- Build the flash_attn package hierarchy ----
import importlib.util
import importlib.machinery

def _make_spec(name):
    """Create a ModuleSpec for a fake module."""
    loader = importlib.machinery.BuiltinImporter
    return importlib.machinery.ModuleSpec(name, loader, origin='built-in', is_package=True)

flash_attn_pkg = ModuleType('flash_attn')
flash_attn_pkg.__spec__ = _make_spec('flash_attn')

# flash_attn.flash_attn_interface
flash_attn_interface_mod = ModuleType('flash_attn.flash_attn_interface')
flash_attn_interface_mod.__spec__ = _make_spec('flash_attn.flash_attn_interface')
flash_attn_interface_mod.flash_attn_unpadded_qkvpacked_func = flash_attn_unpadded_qkvpacked_func
flash_attn_interface_mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
flash_attn_pkg.flash_attn_interface = flash_attn_interface_mod

# flash_attn.bert_padding
bert_padding_mod = ModuleType('flash_attn.bert_padding')
bert_padding_mod.__spec__ = _make_spec('flash_attn.bert_padding')
bert_padding_mod.unpad_input = unpad_input
bert_padding_mod.pad_input = pad_input
flash_attn_pkg.bert_padding = bert_padding_mod

# flash_attn.modules + flash_attn.modules.mha
modules_pkg = ModuleType('flash_attn.modules')
modules_pkg.__spec__ = _make_spec('flash_attn.modules')
mha_mod = ModuleType('flash_attn.modules.mha')
mha_mod.__spec__ = _make_spec('flash_attn.modules.mha')
mha_mod.MHA = MHA
modules_pkg.mha = mha_mod
flash_attn_pkg.modules = modules_pkg

# Register in sys.modules
sys.modules['flash_attn'] = flash_attn_pkg
sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface_mod
sys.modules['flash_attn.bert_padding'] = bert_padding_mod
sys.modules['flash_attn.modules'] = modules_pkg
sys.modules['flash_attn.modules.mha'] = mha_mod
