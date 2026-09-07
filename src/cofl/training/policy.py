"""Construct, load and query field policies from native samples."""

import numpy as np
import torch
from torch.nn import functional as F

from .config import ModelConfig


def build_policy(config: ModelConfig, *, backbone=None):
    from cofl.models import CoFLPolicy, CoFLSPolicy

    freeze_kwargs = {
        "vision_unfreeze_last_n_layers": config.vision_unfreeze_last_n_layers,
        "text_unfreeze_last_n_layers": config.text_unfreeze_last_n_layers,
    }
    if backbone is not None:
        backbone.set_freeze_strategy(**freeze_kwargs)
    encoder_kwargs = {
        **freeze_kwargs,
        "backbone": backbone,
        "siglip_model_name": config.siglip_model_name,
        "local_files_only": config.local_files_only,
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "num_fusion_layers": config.fusion_layers,
    }
    decoder_kwargs = {
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "hidden_dim": config.d_model * 4,
        "num_layers": config.decoder_layers,
    }
    if config.method == "cofl":
        return CoFLPolicy(encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)
    encoder_kwargs.update(use_depth=config.use_depth)
    decoder_kwargs.update(
        bev_x_max=config.r_max_m,
        hfov_rad=config.hfov_rad,
        v_norm=config.normalization_scale_m,
    )
    return CoFLSPolicy(encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)


def prepare_inputs(processor, samples, *, max_text_length=64, color_jitter=0.0, use_depth=False):
    """Batch official SigLIP preprocessing in a CPU DataLoader worker."""
    from PIL import Image
    from torchvision.transforms.v2 import ColorJitter

    images = [Image.fromarray(sample["image"]) for sample in samples]
    if color_jitter:
        augment = ColorJitter(brightness=color_jitter, contrast=color_jitter)
        images = [augment(image) for image in images]
    processed = processor(
        images=images,
        text=[sample["instruction"] for sample in samples],
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    inputs = {key: processed[key] for key in ("pixel_values", "input_ids", "attention_mask")}
    inputs["depth_groups"] = []
    if use_depth:
        if any(sample.get("depth") is None for sample in samples):
            raise ValueError(
                "RGB-D training requires metric depth for every sample; set model.use_depth=false for RGB training"
            )
        grouped = {}
        for index, sample in enumerate(samples):
            depth = np.asarray(sample["depth"])
            if depth.ndim != 2 or min(depth.shape) < 1:
                raise ValueError("Metric depth must have shape [H,W] with positive dimensions")
            mask = sample.get("depth_valid")
            mask = (
                np.ones(depth.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
            )
            if mask.shape != depth.shape:
                raise ValueError("Depth validity mask must match metric depth shape [H,W]")
            grouped.setdefault(depth.shape, []).append(
                (index, depth, mask, sample["geometry"]["r_max_m"])
            )
        for group in grouped.values():
            indices, depths, masks, scales = zip(*group)
            inputs["depth_groups"].append(
                {
                    "indices": torch.tensor(indices, dtype=torch.long),
                    "depth": torch.from_numpy(np.stack(depths))[:, None],
                    "depth_valid": torch.from_numpy(np.stack(masks))[:, None],
                    "scale_m": torch.tensor(scales, dtype=torch.float32)[:, None, None, None],
                }
            )
    return inputs


def encode_prepared(policy, config, inputs, device):
    """Transfer a prepared batch and encode it without invoking the processor."""
    pixels = inputs["pixel_values"].to(device, non_blocking=True)
    tokens = inputs["input_ids"].to(device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)
    kwargs = {}
    if config.method == "cofl-s" and config.use_depth:
        if not inputs["depth_groups"]:
            raise ValueError("RGB-D encoding requires prepared depth groups")
        size = tuple(pixels.shape[-2:])
        shape = (pixels.shape[0], 1, *size)
        depths = torch.empty(shape, dtype=torch.float32, device=device)
        masks = torch.empty(shape, dtype=torch.bool, device=device)
        resize_kwargs = {"align_corners": False} if config.depth_interpolation == "bicubic" else {}
        for group in inputs["depth_groups"]:
            depth = group["depth"].to(device, non_blocking=True).float()
            depth_scale_m = group["scale_m"].to(device, non_blocking=True)
            valid = torch.isfinite(depth) & (depth > config.depth_min_m) & (depth <= depth_scale_m)
            valid &= group["depth_valid"].to(device, non_blocking=True)
            normalized = torch.where(valid, depth / depth_scale_m, 0)
            resized = F.interpolate(
                normalized, size=size, mode=config.depth_interpolation, **resize_kwargs
            ).clamp(0, 1)
            resized_mask = F.interpolate(valid.float(), size=size, mode="nearest").bool()
            indices = group["indices"].to(device, non_blocking=True)
            depths.index_copy_(0, indices, resized)
            masks.index_copy_(0, indices, resized_mask)
        kwargs.update(depth=depths, depth_valid_mask=masks)
    return policy.encode(pixels, tokens, attention_mask, **kwargs)


def encode_samples(policy, config, samples, device):
    inputs = prepare_inputs(
        policy.backbone.processor,
        samples,
        max_text_length=policy.backbone.max_text_length,
        use_depth=config.method == "cofl-s" and config.use_depth,
    )
    return encode_prepared(policy, config, inputs, device)


def query_policy_kwargs(config, samples, *, device, dtype, inference=False):
    """Prepare decoder arguments; inference clamps conditioning, never geometry.

    Training retains each sample's actual HFOV. Deployment conditions the
    decoder within its trained band while rollout uses the original geometry.
    """
    kwargs = {}
    if config.method == "cofl-s":
        query_radius = config.r_max_m
        normalization_scale = config.normalization_scale_m
        for sample in samples:
            geometry = sample["geometry"]
            if not np.isclose(geometry["r_max_m"], query_radius, rtol=1e-6):
                raise ValueError("Dataset ground query radius differs from the model/checkpoint")
            if not np.isclose(geometry["normalization_scale_m"], normalization_scale, rtol=1e-6):
                raise ValueError(
                    "Dataset ground field normalization differs from the model/checkpoint"
                )
        hfovs = [s["geometry"]["hfov_rad"] for s in samples]
        if inference:
            from cofl.models.cofl_s import clamp_hfov_for_model

            hfovs = [clamp_hfov_for_model(value) for value in hfovs]
        kwargs["hfov_rad"] = torch.tensor(
            hfovs,
            device=device,
            dtype=dtype,
        )
    return kwargs


def query_policy(policy, config, queries, context, samples):
    kwargs = query_policy_kwargs(config, samples, device=queries.device, dtype=queries.dtype)
    return policy.query(queries, policy.prepare_query_context(context), **kwargs)


def action_labels(samples, device):
    values = [sample["target_action"] for sample in samples]
    if any(value is not None and value not in (0, 1, 2, 3) for value in values):
        raise ValueError("Action targets must be 0..3 or null")
    return torch.tensor(
        [-100 if value is None else value for value in values], dtype=torch.long, device=device
    )


def load_policy(checkpoint_path, *, device="cpu"):
    """Load an independent PyTorch policy from one complete Lightning checkpoint."""
    from cofl.models.backbone import SigLIPBackbone

    from .checkpoint import read_checkpoint

    checkpoint = read_checkpoint(checkpoint_path)
    config = ModelConfig(**checkpoint["hyper_parameters"]["model"])
    backbone = SigLIPBackbone.from_assets(checkpoint["cofl"]["backbone_assets"])
    model = build_policy(config, backbone=backbone)
    state = {
        key.removeprefix("policy."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("policy.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config
