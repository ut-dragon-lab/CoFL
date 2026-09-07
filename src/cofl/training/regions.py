"""Assign sampled queries to observed navigation regions for loss diagnostics.

Label assignment never consumes random numbers; training can optionally use the
labels to exclude obstacle queries from supervision. Ground ``free``
means visible and walkable; ``occluded`` is walkable outside the visible mask
(including rasterized sector edges). Image navigation masks have no visibility
split. ``obstacle`` means non-walkable, and missing information stays ``unknown``.
"""

import numpy as np
import torch


REGION_NAMES = ("free", "obstacle", "occluded", "unknown")
FREE, OBSTACLE, OCCLUDED, UNKNOWN = range(len(REGION_NAMES))

# Project only diagnostic masks, not semantic images, trajectories or potentials.
REGION_EXTRA_KEYS = {
    "observation": (
        "navigation_mask",
        "bev_mask",
        "bev_walkable",
        "source_bev_mask",
        "source_bev_walkable",
    ),
}


def _mask(extras, *names):
    for name in names:
        if name not in extras:
            continue
        value = np.asarray(extras[name])
        if value.ndim != 2 or min(value.shape) < 1:
            raise ValueError(f"Region mask {name} must be a nonempty [H,W] array")
        if not np.isin(value, (0, 1)).all():
            raise ValueError(f"Region mask {name} must contain only binary values")
        return value.astype(bool, copy=False)
    return None


def _nearest(mask, row, col):
    h, w = mask.shape
    return mask[
        np.rint(row).astype(np.int64).clip(0, h - 1),
        np.rint(col).astype(np.int64).clip(0, w - 1),
    ]


def prepare_query_regions(samples, queries, *, mode="continuous"):
    """Return CPU int8 ``[B,N]`` labels ordered by :data:`REGION_NAMES`.

    Query positions use the same Cartesian readout as their field targets, with
    nearest mask cells instead of bilinear vectors. Image continuous readout
    uses half-pixel coordinates; grid readout uses stored endpoint coordinates.
    Ground masks share the field's inclusive-endpoint Cartesian grid. A mask
    edge receives one discrete label even when its target blends both regions.

    Only explicit navigation extras establish occupancy. In particular the
    annotation supervision ``mask`` cannot distinguish free space and obstacles.
    With just one of the ground masks, only its unambiguous cells are labeled.
    """
    if (
        not isinstance(queries, torch.Tensor)
        or queries.device.type != "cpu"
        or not queries.is_floating_point()
        or queries.ndim != 3
        or queries.shape[-1] != 2
        or len(queries) != len(samples)
    ):
        raise ValueError("Region queries must be a CPU floating tensor of shape [B,N,2]")
    if mode not in {"grid", "continuous"}:
        raise ValueError("Region query mode must be grid or continuous")
    coordinates = queries.detach().numpy()
    if not np.isfinite(coordinates).all():
        raise ValueError("Region queries must be finite")
    labels = np.full(queries.shape[:2], UNKNOWN, dtype=np.int8)
    for index, (sample, query) in enumerate(zip(samples, coordinates)):
        extras = sample.get("observation_extras", {})
        profile = sample["profile"]
        if profile == "image_field_v1":
            navigation = _mask(extras, "navigation_mask")
            if navigation is None:
                continue
            h, w = navigation.shape
            if mode == "continuous":
                row, col = query[:, 1] * h - 0.5, query[:, 0] * w - 0.5
            else:
                row, col = query[:, 1] * (h - 1), query[:, 0] * (w - 1)
            free = _nearest(navigation, row, col)
            labels[index] = np.where(free, FREE, OBSTACLE)
        elif profile == "ground_sector_v1":
            visible = _mask(extras, "bev_mask", "source_bev_mask")
            walkable = _mask(extras, "bev_walkable", "source_bev_walkable")
            if visible is None and walkable is None:
                continue
            geometry = sample["geometry"]
            shape = tuple(geometry["grid_shape"])
            if any(mask.shape != shape for mask in (visible, walkable) if mask is not None):
                raise ValueError("Ground region masks must match geometry.grid_shape")
            if visible is not None and walkable is not None and np.any(visible & ~walkable):
                raise ValueError("Visible ground region mask must be a subset of walkable")
            theta = query[:, 0] * (geometry["hfov_rad"] / 2)
            radius = query[:, 1] * (
                geometry["r_max_m"] / geometry["normalization_scale_m"]
            )
            h, w = shape
            row = radius * np.cos(theta) * (h - 1)
            col = (radius * np.sin(theta) + 1) * ((w - 1) / 2)
            seen = None if visible is None else _nearest(visible, row, col)
            walk = None if walkable is None else _nearest(walkable, row, col)
            if walk is not None:
                labels[index, ~walk] = OBSTACLE
            if seen is not None:
                labels[index, seen] = FREE
                if walk is not None:
                    labels[index, walk & ~seen] = OCCLUDED
        else:
            raise ValueError(f"Unknown field profile: {profile!r}")
    return torch.from_numpy(labels)
