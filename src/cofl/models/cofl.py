"""CoFLFieldDecoder query decoder, independent of training and simulation frameworks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import GaussianFourierEncoding, PreNormDecoderBlock


class CoFLFieldDecoder(nn.Module):
    """Query a static image-coordinate field using visual-language context.

    Queries are [B,N,2] in [0,1]^2; context is [B,T,D]. Output vectors
    use the same image-coordinate basis as the queries.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        need_self_attn: bool = False,
    ):
        super().__init__()
        if d_model < 4 or d_model % 2:
            raise ValueError("d_model must be even and at least 4")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.pos_proj = nn.Sequential(
            GaussianFourierEncoding(in_features=2, out_features=d_model, scale=10.0),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cross_attn_layer = nn.ModuleList(
            [
                PreNormDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=hidden_dim,
                    dropout=dropout,
                    need_self_attn=need_self_attn,
                )
                for _ in range(num_layers)
            ]
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.mag_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2)
        )

    def prepare_context(self, context: torch.Tensor):
        """Project image-language tokens into each decoder layer's key/value space."""
        if context.ndim != 3 or context.shape[-1] != self.d_model:
            raise ValueError("context must have shape [B,T,d_model]")
        return tuple(layer.prepare_context(context) for layer in self.cross_attn_layer)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        mag_enable: bool = True,
    ) -> torch.Tensor:
        return self.decode(x, self.prepare_context(context), mag_enable=mag_enable)

    def decode(self, x: torch.Tensor, query_context, *, mag_enable: bool = True):
        """Return direction times nonnegative magnitude at image queries, with shape [B,N,2]."""
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError("query must have shape [B,N,2]")
        if len(query_context) != len(self.cross_attn_layer):
            raise ValueError("query context must contain one key/value pair per decoder layer")
        if any(key.shape[0] != x.shape[0] for key, value in query_context):
            raise ValueError("query and context must have the same batch size")
        local_emb = self.pos_proj(x)
        for layer, layer_context in zip(self.cross_attn_layer, query_context):
            local_emb = layer.decode(local_emb, layer_context)
        features = self.feature_norm(local_emb)
        mag = self.mag_head(features)
        dir = self.dir_head(features)
        dir = F.normalize(dir, dim=-1, eps=1e-08)
        if not mag_enable:
            mag = torch.ones_like(mag)
        velocity = dir * mag
        return velocity
