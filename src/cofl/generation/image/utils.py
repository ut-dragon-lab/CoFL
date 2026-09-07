"""
Utility functions for VLA dataset generation.

Contains low-level image processing and geometric utilities.
"""

import cv2
import numpy as np


def clean_walkable_mask(
    walkable_mask: np.ndarray,
    min_region_area: int = 64,
    closing_kernel: int = 3,
) -> np.ndarray:
    """Clean small isolated regions and apply morphological closing."""
    walkable_u8 = walkable_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walkable_u8, connectivity=8)

    clean = np.zeros_like(walkable_u8, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_region_area:
            clean[labels == i] = 1

    if closing_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)

    return clean.astype(bool)


def sample_start_positions(
    walkable_mask: np.ndarray,
    n_samples: int = 10,
    min_clearance: float = 5.0,
) -> list[list[float]]:
    """Sample valid start positions from walkable mask.

    Returns positions in normalized coordinates [0, 1].
    """
    H, W = walkable_mask.shape

    # Get largest connected component
    free_u8 = walkable_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_u8, connectivity=8)

    if num_labels <= 1:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    main_mask = (labels == largest_label).astype(np.uint8)

    # Compute distance transform for clearance
    dist_transform = cv2.distanceTransform(main_mask, cv2.DIST_L2, 5)

    # Get valid points with sufficient clearance
    valid_mask = dist_transform >= min_clearance
    ys, xs = np.where(valid_mask)

    if len(xs) == 0:
        # Fallback: use any walkable point
        ys, xs = np.where(main_mask > 0)
        if len(xs) == 0:
            return []

    # Sample points
    n_available = len(xs)
    n_to_sample = min(n_samples, n_available)

    if n_to_sample < n_samples:
        # With replacement
        indices = np.random.choice(n_available, n_samples, replace=True)
    else:
        indices = np.random.choice(n_available, n_to_sample, replace=False)

    # Return normalized coordinates
    starts = []
    for idx in indices:
        x_norm = xs[idx] / W
        y_norm = ys[idx] / H
        starts.append([float(x_norm), float(y_norm)])

    return starts


def resize_field(field: np.ndarray, target_size: tuple[int, int] = (50, 50)) -> np.ndarray:
    """Resize a field (2D or 3D with channels first) to target size."""
    if field.ndim == 2:
        return cv2.resize(field, target_size, interpolation=cv2.INTER_LINEAR)
    elif field.ndim == 3:
        # [C, H, W] -> resize each channel
        C = field.shape[0]
        result = np.zeros((C, target_size[1], target_size[0]), dtype=field.dtype)
        for c in range(C):
            result[c] = cv2.resize(field[c], target_size, interpolation=cv2.INTER_LINEAR)
        return result
    else:
        raise ValueError(f"Unsupported field shape: {field.shape}")
