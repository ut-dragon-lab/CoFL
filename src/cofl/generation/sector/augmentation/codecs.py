"""JPEG encoding for portable simulator RGB observations."""

from __future__ import annotations

import cv2
import numpy as np

JPEG_QUALITY_DEFAULT = 92


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY_DEFAULT) -> bytes:
    """Encode an (H, W, 3) uint8 RGB frame to JPEG bytes."""
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"encode_rgb_jpeg expected (H, W, 3) uint8, got shape={rgb.shape}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    (ok, enc) = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG failed")
    return enc.tobytes()


__all__ = ["JPEG_QUALITY_DEFAULT", "encode_rgb_jpeg"]
