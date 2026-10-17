"""flash_attn.modules.mha.MHA 的纯 PyTorch 等价实现 (非 flash 路径)。

复刻 flash_attn 2.5.9 的 MHA 在 use_flash_attn=False 时的完整前向数学:

    MHA.forward(x):
        qkv = Wqkv(x)                                  # [B, S, 3E]
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=head_dim)
        context = SelfAttention(qkv):
            q, k, v = qkv.unbind(dim=2)
            softmax_scale = 1 / sqrt(head_dim)         # 默认值
            scores = einsum("bthd,bshd->bhts", q, k * softmax_scale)
            attention = softmax(scores, dim=-1, dtype=v.dtype)
            attention_drop = dropout(attention)        # 训练时 dropout, 推理时恒等
            output = einsum("bhts,bshd->bthd", attention_drop, v)
        out = out_proj(rearrange(context, "... h d -> ... (h d)"))   # [B, S, E]

本项目 (vit_m2f2_detector_unified.py / _bridge.py) 调用 MHA 时:
    MHA(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
未传 use_flash_attn=True, 且 causal=False, 无 mask / rotary / kv cache,
所以上面的简化路径与原实现完全等价。

参数结构与原版一致 (Wqkv / out_proj, 均带 bias), 因此 state_dict 键名相同,
可直接加载训练保存的权重。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """scaled dot-product attention (flash_attn 的非 flash 实现, einsum 版)."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """qkv: (B, S, 3, H, D)"""
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + padding_mask.unsqueeze(1).unsqueeze(1)
        if causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention — 兼容 flash_attn.modules.mha.MHA 的非 flash 路径。"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert not cross_attn, "shim MHA 仅支持 self-attention"
        assert not dwconv and rotary_emb_dim == 0 and not use_alibi, \
            "shim MHA 仅覆盖本项目使用路径 (无 dwconv/rotary/alibi)"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        self.causal = causal
        self.layer_idx = layer_idx
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        self.use_flash_attn = use_flash_attn

        assert self.num_heads % self.num_heads_kv == 0, \
            "num_heads must be divisible by num_heads_kv"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)
        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # 与原版一致: 若构造时显式要求 flash, shim 无法提供, 直接报错而非静默降级
        assert not use_flash_attn, "shim MHA 不支持 use_flash_attn=True (无 CUDA 内核)"

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        assert x_kv is None and mixer_subset is None, "shim MHA 仅支持 self-attention"
        assert cu_seqlens is None and max_seqlen is None and inference_params is None, \
            "shim MHA 不支持 varlen / kv-cache 路径"
        qkv = self.Wqkv(x)
        qkv = qkv.view(*qkv.shape[:-1], 3, self.num_heads, self.head_dim)
        context = self.inner_attn(qkv, key_padding_mask=key_padding_mask)
        out = self.out_proj(context.reshape(*context.shape[:-2], self.embed_dim))
        return out if not self.return_residual else (out, x)
