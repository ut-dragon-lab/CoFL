"""Neural components shared where their numerical contracts agree."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def prepare_attention_context(attention: nn.MultiheadAttention, context: torch.Tensor):
    """Project context tokens into per-head keys and values."""
    width, heads = attention.embed_dim, attention.num_heads
    bias = attention.in_proj_bias
    projected = F.linear(
        context, attention.in_proj_weight[width:], None if bias is None else bias[width:]
    )
    key, value = projected.chunk(2, dim=-1)
    shape = (*context.shape[:2], heads, width // heads)
    return key.reshape(shape).transpose(1, 2), value.reshape(shape).transpose(1, 2)


def attention_from_context(attention, query, query_context, attn_mask=None, key_padding_mask=None):
    """Attend to projected context using PyTorch's scaled dot-product kernel.

    Boolean masks follow MultiheadAttention's convention: true entries are
    blocked. Floating masks are additive attention-logit biases.
    """
    width, heads = attention.embed_dim, attention.num_heads
    bias = attention.in_proj_bias
    projected = F.linear(
        query, attention.in_proj_weight[:width], None if bias is None else bias[:width]
    )
    projected = projected.reshape(*query.shape[:2], heads, width // heads).transpose(1, 2)
    mask = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=projected.dtype).masked_fill(
                attn_mask, float("-inf")
            )
        if attn_mask.ndim == 3:
            attn_mask = attn_mask.reshape(query.shape[0], heads, *attn_mask.shape[-2:])
        mask = attn_mask
    if key_padding_mask is not None:
        padding = key_padding_mask
        if padding.dtype == torch.bool:
            padding = torch.zeros_like(padding, dtype=projected.dtype).masked_fill(
                padding, float("-inf")
            )
        padding = padding[:, None, None, :]
        mask = padding if mask is None else mask + padding
    output = F.scaled_dot_product_attention(
        projected,
        *query_context,
        attn_mask=mask,
        dropout_p=attention.dropout if attention.training else 0.0,
    )
    output = output.transpose(1, 2).reshape(*query.shape[:2], width)
    return F.linear(output, attention.out_proj.weight, attention.out_proj.bias)


class GaussianFourierEncoding(nn.Module):
    def __init__(self, in_features=2, out_features=512, scale=10.0):
        super().__init__()
        self.register_buffer("weight", torch.randn(in_features, out_features // 2) * scale)

    def forward(self, x):
        if not self.training and torch.is_autocast_enabled(x.device.type):
            # Half-precision phase errors are amplified by the Fourier scale.
            # Keep inference coordinates precise while later layers use AMP.
            with torch.autocast(device_type=x.device.type, enabled=False):
                dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
                x_proj = torch.matmul(x.to(dtype), self.weight.to(dtype)) * 2 * math.pi
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x_proj = torch.matmul(x, self.weight) * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PostNormDecoderBlock(nn.Module):
    """Post-normalized attention block for the CoFL field decoder."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        need_self_attn: bool = False,
    ):
        super().__init__()
        if need_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.normq = nn.LayerNorm(d_model)
        else:
            self.self_attn = None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropoutq = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decode(q, self.prepare_context(kv), attn_mask, key_padding_mask)

    def prepare_context(self, context):
        return prepare_attention_context(self.cross_attn, context)

    def decode(self, q, query_context, attn_mask=None, key_padding_mask=None):
        if self.self_attn is not None:
            (attn_output, _) = self.self_attn(q, q, q, need_weights=False)
            q = q + self.dropoutq(attn_output)
            q = self.normq(q)
        attn_output = attention_from_context(
            self.cross_attn, q, query_context, attn_mask, key_padding_mask
        )
        q = q + self.dropout1(attn_output)
        q = self.norm1(q)
        ffn_output = self.ffn(q)
        q = q + self.dropout2(ffn_output)
        q = self.norm2(q)
        return q


class PreNormDecoderBlock(nn.Module):
    """Pre-normalized query attention for field and action heads."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        need_self_attn: bool = False,
    ):
        super().__init__()
        if need_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.normq = nn.LayerNorm(d_model)
        else:
            self.self_attn = None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropoutq = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decode(q, self.prepare_context(kv), attn_mask, key_padding_mask)

    def prepare_context(self, context):
        return prepare_attention_context(self.cross_attn, context)

    def decode(self, q, query_context, attn_mask=None, key_padding_mask=None):
        if self.self_attn is not None:
            normalized = self.normq(q)
            (attn_output, _) = self.self_attn(
                normalized, normalized, normalized, need_weights=False
            )
            q = q + self.dropoutq(attn_output)
        attn_output = attention_from_context(
            self.cross_attn, self.norm1(q), query_context, attn_mask, key_padding_mask
        )
        q = q + self.dropout1(attn_output)
        ffn_output = self.ffn(self.norm2(q))
        q = q + self.dropout2(ffn_output)
        return q
