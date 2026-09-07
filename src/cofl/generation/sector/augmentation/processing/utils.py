"""Pose conversion and metric-depth image preprocessing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

(_TARGET_H, _TARGET_W) = (224, 224)


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / tuple structures into JSON types."""
    import dataclasses

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if dataclasses.is_dataclass(obj) and (not isinstance(obj, type)):
        return _to_json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return str(obj)


def _quat_xyzw_to_heading(rot_xyzw: np.ndarray) -> float:
    """Convert Habitat's (x, y, z, w) quaternion to a scalar heading (radians)."""
    q1 = float(rot_xyzw[1])
    q3 = float(rot_xyzw[3])
    return math.atan2(q3 * q3 - q1 * q1, -2.0 * q1 * q3)


def _compute_depth_valid_mask(depth_np: np.ndarray, z_max: float) -> np.ndarray:
    """Return a uint8 mask with 1 where the (metric) depth is finite, >0, ≤ z_max."""
    d = np.asarray(depth_np, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    d_m = d
    return (np.isfinite(d_m) & (d_m > 0.0) & (d_m <= float(z_max))).astype(np.uint8)


def _resize_rgb_224(rgb_np: np.ndarray) -> np.ndarray:
    """Crop alpha + resize to 224×224 using PIL bilinear."""
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    rgb_np = rgb_np.astype(np.uint8)
    if rgb_np.shape[0] != _TARGET_H or rgb_np.shape[1] != _TARGET_W:
        from PIL import Image as _PILImg

        rgb_np = np.array(
            _PILImg.fromarray(rgb_np).resize((_TARGET_W, _TARGET_H), _PILImg.BILINEAR)
        ).astype(np.uint8)
    return rgb_np


def _resize_depth_224(depth_np: np.ndarray) -> np.ndarray:
    """Resize finite metric depth to 224 by 224 with bilinear interpolation."""
    depth_np = depth_np.astype(np.float32)
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    if depth_np.shape[0] != _TARGET_H or depth_np.shape[1] != _TARGET_W:
        from PIL import Image as _PILImg

        depth_np = np.array(
            _PILImg.fromarray(depth_np, mode="F").resize((_TARGET_W, _TARGET_H), _PILImg.BILINEAR)
        ).astype(np.float32)
    return depth_np


def _resize_depth_with_holes_224(depth_np: np.ndarray, *, min_valid_m: float = 0.0) -> np.ndarray:
    """Fill invalid depth from the nearest valid pixel, then resize.

    An entirely invalid image remains all zero; all-valid input needs no
    inpainting. The fill avoids creating phantom near-camera obstacles."""
    raw = np.asarray(depth_np, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    valid_raw = np.isfinite(raw) & (raw > float(min_valid_m))
    if valid_raw.all():
        return _resize_depth_224(raw)
    if not valid_raw.any():
        _TARGET_H if raw.shape[0] != _TARGET_H else raw.shape[0]
        _TARGET_W if raw.shape[1] != _TARGET_W else raw.shape[1]
        return np.zeros((_TARGET_H, _TARGET_W), dtype=np.float32)
    from scipy.ndimage import distance_transform_edt

    (_, idx) = distance_transform_edt(~valid_raw, return_indices=True)
    filled = raw[tuple(idx)].astype(np.float32)
    return _resize_depth_224(filled)
