"""Vision-language cross-attention."""

from __future__ import annotations

import torch
from torch import nn


def _text_key_padding_mask(
    text_tokens: torch.Tensor, text_attention_mask: torch.Tensor | None
) -> torch.Tensor | None:
    """Convert a batch's valid-token mask to PyTorch's ignored-token mask."""
    if text_attention_mask is None:
        return None
    if (
        not isinstance(text_attention_mask, torch.Tensor)
        or text_attention_mask.ndim != 2
        or text_attention_mask.shape != text_tokens.shape[:2]
    ):
        raise ValueError("text_attention_mask must have shape [B,L] matching text_tokens")
    valid = text_attention_mask.to(device=text_tokens.device, dtype=torch.bool)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("text_attention_mask must retain at least one valid token per sample")
    return ~valid


class VisionLanguageFusion(nn.Module):
    """Pre-normalized visual self-attention and cross-attention to language tokens."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse tokens with an optional [B,L] mask where True means valid text."""
        text_key_padding_mask = _text_key_padding_mask(text_tokens, text_attention_mask)
        for layer in self.layers:
            visual_tokens = layer(
                visual_tokens, text_tokens, memory_key_padding_mask=text_key_padding_mask
            )
        context = visual_tokens
        return context
