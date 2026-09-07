"""Vision-language policies composed from encoders and query decoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .action import NavigationActionHead
from .cofl import CoFLFieldDecoder
from .cofl_s import CoFLSFieldDecoder
from .encoders import ConditionEncoder, _validate_single_frame_images


def _prepare_inputs(backbone, images, input_ids, attention_mask):
    """Treat raw image lists as a batch and preserve processor padding masks."""
    raw_images = not isinstance(images, torch.Tensor)
    raw_text = not isinstance(input_ids, torch.Tensor)
    if raw_images or raw_text:
        processed = backbone.preprocess(
            images=images if raw_images else None, texts=input_ids if raw_text else None
        )
        if raw_images:
            images = processed["pixel_values"].to(backbone.device)
        if raw_text:
            input_ids = processed["input_ids"].to(backbone.device)
            if attention_mask is None:
                attention_mask = processed.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(backbone.device)
    return (images, input_ids, attention_mask)


def _components(decoder_type, encoder, decoder, encoder_kwargs, decoder_kwargs, *, use_depth=False):
    enc_args = dict(encoder_kwargs or {})
    dec_args = dict(decoder_kwargs or {})
    if encoder is not None and enc_args:
        raise ValueError("Pass either an encoder or encoder_kwargs")
    if decoder is not None and dec_args:
        raise ValueError("Pass either a decoder or decoder_kwargs")
    if decoder is not None:
        width = decoder.d_model
    elif "d_model" in dec_args:
        width = dec_args["d_model"]
    elif encoder is not None:
        width = encoder.d_model
    else:
        width = enc_args.get("d_model", 768)
    if int(enc_args.get("d_model", width)) != int(width):
        raise ValueError("Encoder and decoder d_model must match")
    if encoder is None:
        enc_args.setdefault("d_model", width)
        enc_args.setdefault("use_depth", use_depth)
        encoder = ConditionEncoder(**enc_args)
    if decoder is None:
        dec_args.setdefault("d_model", width)
        dec_args.setdefault("hidden_dim", 4 * width)
        dec_args.setdefault("num_layers", 2)
        decoder = decoder_type(**dec_args)
    if encoder.d_model != width:
        raise ValueError("Encoder and decoder d_model must match")
    return (encoder, decoder)


class CoFLPolicy(nn.Module):
    """CoFL image/language policy with separately reusable encoding and querying.

    ``images`` is a preprocessed [B,3,H,W] tensor or raw images accepted by
    the SigLIP processor. ``input_ids`` is token IDs or raw instruction text.
    ``query`` is [B,N,2] in normalized image coordinates. Loading the
    default encoder requires the ``vision`` installation extra.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module | None = None,
        decoder: CoFLFieldDecoder | None = None,
        encoder_kwargs: Mapping[str, Any] | None = None,
        decoder_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        (self.encoder, self.decoder) = _components(
            CoFLFieldDecoder, encoder, decoder, encoder_kwargs, decoder_kwargs
        )

    @property
    def backbone(self):
        """Pretrained backbone, including its processor and preprocess method."""
        return self.encoder.backbone

    def encode(self, images, input_ids, attention_mask=None) -> torch.Tensor:
        _validate_single_frame_images(images, method="CoFL")
        if not isinstance(images, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            (images, input_ids, attention_mask) = _prepare_inputs(
                self.backbone, images, input_ids, attention_mask
            )
        return self.encoder(images, input_ids, attention_mask)[0]

    def query(self, query: torch.Tensor, query_context, **kwargs) -> torch.Tensor:
        """Evaluate coordinates using the decoder context from prepare_query_context."""
        return self.decoder.decode(query, query_context, **kwargs)

    def prepare_query_context(self, context: torch.Tensor):
        return self.decoder.prepare_context(context)

    def forward(self, query, images, input_ids, attention_mask=None, *, mag_enable=True):
        context = self.encode(images, input_ids, attention_mask)
        return self.query(query, self.prepare_query_context(context), mag_enable=mag_enable)


class CoFLSPolicy(nn.Module):
    """CoFL-S single-frame RGB-D/language policy with an action head.

    The field query uses normalized polar coordinates. Field outputs have
    ego-frame components (forward, left), normalized by ``decoder.v_norm``.
    ``action_logits(context)`` predicts STOP/FORWARD/LEFT/RIGHT separately.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module | None = None,
        decoder: CoFLSFieldDecoder | None = None,
        encoder_kwargs: Mapping[str, Any] | None = None,
        decoder_kwargs: Mapping[str, Any] | None = None,
        use_action_head: bool = True,
        action_head_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        (self.encoder, self.decoder) = _components(
            CoFLSFieldDecoder,
            encoder,
            decoder,
            encoder_kwargs,
            decoder_kwargs,
            use_depth=True,
        )
        head_args = dict(action_head_kwargs or {})
        if int(head_args.get("d_model", self.decoder.d_model)) != self.decoder.d_model:
            raise ValueError("Action head and decoder d_model must match")
        if int(head_args.get("num_actions", 4)) != 4:
            raise ValueError("CoFL-S uses four action classes")
        if not use_action_head and head_args:
            raise ValueError("action_head_kwargs requires use_action_head=True")
        head_args.setdefault("d_model", self.decoder.d_model)
        head_args.setdefault("num_heads", self.decoder.num_heads)
        self.action_head = NavigationActionHead(**head_args) if use_action_head else None

    @property
    def backbone(self):
        """Pretrained backbone, including its processor and preprocess method."""
        return self.encoder.backbone

    def encode(
        self,
        images,
        input_ids,
        attention_mask=None,
        *,
        depth=None,
        depth_valid_mask=None,
    ) -> torch.Tensor:
        _validate_single_frame_images(images)
        if not isinstance(images, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            (images, input_ids, attention_mask) = _prepare_inputs(
                self.backbone, images, input_ids, attention_mask
            )
        return self.encoder(
            images,
            input_ids,
            attention_mask,
            depth=depth,
            depth_valid_mask=depth_valid_mask,
        )[0]

    def query(self, query: torch.Tensor, query_context, **kwargs) -> torch.Tensor:
        """Evaluate coordinates using the decoder context from prepare_query_context."""
        return self.decoder.decode(query, query_context, **kwargs)

    def prepare_query_context(self, context: torch.Tensor):
        return self.decoder.prepare_context(context)

    def action_logits(self, context: torch.Tensor, context_padding_mask=None) -> torch.Tensor:
        if self.action_head is None:
            raise RuntimeError("This policy was constructed without an action head")
        return self.action_head(context, context_padding_mask)

    def forward(
        self,
        query,
        images,
        input_ids,
        attention_mask=None,
        *,
        depth=None,
        depth_valid_mask=None,
        hfov_rad=None,
    ):
        context = self.encode(
            images,
            input_ids,
            attention_mask,
            depth=depth,
            depth_valid_mask=depth_valid_mask,
        )
        return self.query(query, self.prepare_query_context(context), hfov_rad=hfov_rad)
