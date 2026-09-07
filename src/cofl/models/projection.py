"""Shared projections from pretrained visual/language features to policy tokens."""

from __future__ import annotations

import torch
from torch import nn


class VisionLanguageProjection(nn.Module):
    """Project each modality independently into the common context dimension."""

    def __init__(self, d_model: int = 768, backbone_dim: int = 1152):
        super().__init__()
        self.vision_proj = nn.Linear(backbone_dim, d_model)
        self.text_proj = nn.Linear(backbone_dim, d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        visual_tokens = self.visual_norm(self.vision_proj(image_features))
        language_tokens = self.text_norm(self.text_proj(text_features))
        return visual_tokens, language_tokens
