"""Shared single-frame visual-language conditioning for both field policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import torch
from torch import nn

from .backbone import SigLIPBackbone
from .depth import DepthFusion
from .fusion import VisionLanguageFusion
from .projection import VisionLanguageProjection


def _encode_text_for_fusion(backbone, texts_or_input_ids, attention_mask):
    """Keep tokenizer validity separate from pretrained SigLIP self-attention."""
    if not isinstance(texts_or_input_ids, torch.Tensor):
        processed = backbone.preprocess(texts=texts_or_input_ids)
        texts_or_input_ids = processed["input_ids"].to(backbone.device)
        if attention_mask is None:
            attention_mask = processed.get("attention_mask")
    if texts_or_input_ids.ndim != 2:
        raise ValueError("input text tokens must have shape [B,L]")
    # SigLIP uses fixed-length padded text and pools the final position. Keep
    # that pretrained computation intact; the task fusion masks padding keys.
    text_features = backbone.encode_text(texts_or_input_ids)
    expected_shape = (texts_or_input_ids.shape[0], texts_or_input_ids.shape[1] + 1)
    if text_features.ndim != 3 or text_features.shape[:2] != expected_shape:
        raise ValueError(
            "Backbone text features must have shape [B,L+1,D] with a pooled prefix token"
        )
    if attention_mask is None:
        return text_features, None
    if not isinstance(attention_mask, torch.Tensor) or (
        attention_mask.ndim != 2 or attention_mask.shape != texts_or_input_ids.shape
    ):
        raise ValueError("attention_mask must match the input text token shape [B,L]")
    text_mask = attention_mask.to(device=text_features.device, dtype=torch.bool)
    text_mask = torch.cat([text_mask.new_ones((text_mask.shape[0], 1)), text_mask], dim=1)
    return text_features, text_mask


class EncoderOutput(NamedTuple):
    """Fused context, current visual features, and projected language tokens.

    ``context`` and ``language_tokens`` use the policy's ``d_model`` width;
    ``vision_tokens`` retain the backbone width and include any depth fusion.
    """

    context: torch.Tensor
    vision_tokens: torch.Tensor
    language_tokens: torch.Tensor


def _validate_single_frame_images(images, *, method: str = "ConditionEncoder") -> None:
    """Reject frame sequences before a processor could flatten them into a batch."""
    if isinstance(images, torch.Tensor):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape) < 1:
            raise ValueError(
                f"{method} pixel_values must have shape [B,3,H,W] with positive dimensions"
            )
    elif getattr(images, "ndim", 0) > 4:
        raise ValueError(f"{method} accepts one current image per sample, not frame sequences")
    elif isinstance(images, (list, tuple)):
        if not images:
            raise ValueError(f"{method} image batch must not be empty")
        if any(
            isinstance(image, (list, tuple)) or getattr(image, "ndim", 0) > 3 for image in images
        ):
            raise ValueError(
                f"{method} expects a flat image batch with one current image per sample"
            )


class ConditionEncoder(nn.Module):
    """Shared pooled-text and pre-normalized visual-language encoder.

    Both field policies receive final normalized context. ``use_depth`` adds
    depth conditioning to the current visual features before projection.
    """

    def __init__(
        self,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        unfreeze_last_n_layers: int = 0,
        siglip_deterministic_embeddings: bool = True,
        use_depth: bool = False,
        depth_use_mask_channel: bool = True,
        d_model: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 4,
        local_files_only: bool = False,
        backbone: nn.Module | None = None,
        vision_unfreeze_last_n_layers: int | None = None,
        text_unfreeze_last_n_layers: int | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.use_depth = bool(use_depth)
        self.backbone = (
            backbone
            if backbone is not None
            else SigLIPBackbone(
                model_name=siglip_model_name,
                unfreeze_last_n_layers=unfreeze_last_n_layers,
                vision_unfreeze_last_n_layers=vision_unfreeze_last_n_layers,
                text_unfreeze_last_n_layers=text_unfreeze_last_n_layers,
                deterministic_embeddings=siglip_deterministic_embeddings,
                local_files_only=local_files_only,
            )
        )
        backbone_dim = self.backbone.vision_dim
        self.projection = VisionLanguageProjection(d_model=d_model, backbone_dim=backbone_dim)
        self.fusion = VisionLanguageFusion(
            d_model=d_model, num_heads=num_heads, num_layers=num_fusion_layers, d_ff=d_model * 4
        )
        self.depth_fusion: DepthFusion | None = None
        if self.use_depth:
            self.depth_fusion = DepthFusion(
                patch_size=self.backbone.patch_size,
                d_model=backbone_dim,
                use_mask_channel=bool(depth_use_mask_channel),
            )
        self.ctx_norm = nn.LayerNorm(d_model)

    @property
    def total_text_length(self) -> int:
        """Maximum token feature count, including the pooled prefix."""
        return self.backbone.max_text_length + 1

    def forward(
        self,
        images_or_pixel_values: torch.Tensor | Any | Sequence[Any],
        texts_or_input_ids: torch.Tensor | str | Sequence[str],
        attention_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Encode one current image and instruction per sample."""
        _validate_single_frame_images(images_or_pixel_values)
        image_features = self.backbone.encode_image(images_or_pixel_values)
        if depth is not None and self.depth_fusion is not None:
            image_features = self.depth_fusion(image_features, depth, depth_valid_mask)
        text_features, text_mask = _encode_text_for_fusion(
            self.backbone, texts_or_input_ids, attention_mask
        )
        visual_tokens, language_tokens = self.projection(image_features, text_features)
        context = self.fusion(visual_tokens, language_tokens, text_attention_mask=text_mask)
        context = self.ctx_norm(context)
        return EncoderOutput(context, image_features, language_tokens)
