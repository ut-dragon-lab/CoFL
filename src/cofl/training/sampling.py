"""Query sampling and continuous supervision in each policy's query chart."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def _check_finite_field(field):
    if not bool(torch.isfinite(field).all()):
        raise ValueError("Continuous supervision requires finite values over the full field")


def prepare_field_groups(samples, *, dtype=None):
    """Validate and stack continuous fields on the CPU, grouped by profile/shape.

    Returned dictionaries contain CPU tensors so a preprocessing worker can pin
    them before transfer. Supplying the query dtype performs conversion and its
    finite check here, avoiding further CPU work at readout. Ground geometry
    stays per sample. The mask does not remove obstacle escape supervision.
    """
    if not samples:
        raise ValueError("Continuous supervision requires at least one sample")
    if dtype is not None and not dtype.is_floating_point:
        raise ValueError("Continuous field preparation requires a floating dtype")
    grouped = {}
    for index, sample in enumerate(samples):
        profile = sample["profile"]
        if profile not in {"image_field_v1", "ground_sector_v1"}:
            raise ValueError("Unknown field profile")
        field = np.asarray(sample["field"])
        if field.ndim != 3 or field.shape[0] != 2 or min(field.shape[1:]) < 1:
            raise ValueError("Continuous supervision fields must have shape [2,H,W]")
        key = profile, field.shape
        group = grouped.setdefault(key, {"profile": profile, "indices": [], "fields": []})
        group["indices"].append(index)
        group["fields"].append(field)
        if profile == "ground_sector_v1":
            geometry = sample["geometry"]
            hfov = float(geometry["hfov_rad"])
            radius = float(geometry["r_max_m"])
            scale = float(geometry["normalization_scale_m"])
            if not (
                math.isfinite(hfov)
                and 0 < hfov < math.pi
                and math.isfinite(radius)
                and radius > 0
                and math.isfinite(scale)
                and scale > 0
            ):
                raise ValueError("Ground field geometry requires valid HFOV, radius, and scale")
            group.setdefault("hfov_half", []).append(hfov / 2)
            group.setdefault("radius_scale", []).append(radius / scale)
    for group in grouped.values():
        # Stacking owns the storage; checking after conversion also rejects
        # finite source values that overflow the requested query precision.
        field = torch.from_numpy(np.stack(group["fields"]))
        if dtype is not None:
            field = field.to(dtype=dtype)
        _check_finite_field(field)
        group["fields"] = field
        group["indices"] = torch.tensor(group["indices"], dtype=torch.int64)
        if group["profile"] == "ground_sector_v1":
            group["hfov_half"] = torch.tensor(group["hfov_half"], dtype=torch.float64)
            group["radius_scale"] = torch.tensor(group["radius_scale"], dtype=torch.float64)
    return list(grouped.values())


def interpolate_field_groups(groups, queries):
    """Transfer each prepared group once and read all its fields in one call.

    Image readout uses half-pixel coordinates with zero padding. Ground readout
    converts polar queries to Cartesian storage with inclusive endpoints and
    border padding. Results retain the original sample order and vector units.
    """
    if queries.ndim != 3 or queries.shape[-1] != 2 or not queries.is_floating_point():
        raise ValueError("Continuous queries must be a floating tensor of shape [B,N,2]")
    if not groups or sum(len(group["indices"]) for group in groups) != len(queries):
        raise ValueError("Field groups and query batch sizes must match")
    targets = queries.new_empty(queries.shape)
    for group in groups:
        field = group["fields"]
        if field.dtype != queries.dtype:
            field = field.to(dtype=queries.dtype)
            _check_finite_field(field)
        field = field.to(device=queries.device, non_blocking=True)
        indices = group["indices"].to(device=queries.device, non_blocking=True)
        query = queries.index_select(0, indices)
        if group["profile"] == "image_field_v1":
            grid = 2 * query - 1
            align_corners, padding_mode = False, "zeros"
        else:
            hfov_half = group["hfov_half"].to(
                device=queries.device, dtype=queries.dtype, non_blocking=True
            )
            radius_scale = group["radius_scale"].to(
                device=queries.device, dtype=queries.dtype, non_blocking=True
            )
            theta = query[:, :, 0] * hfov_half[:, None]
            radius = query[:, :, 1] * radius_scale[:, None]
            forward, left = radius * theta.cos(), radius * theta.sin()
            grid = torch.stack((left, 2 * forward - 1), dim=-1)
            align_corners, padding_mode = True, "border"
        values = F.grid_sample(
            field,
            grid[:, :, None, :],
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )[:, :, :, 0].transpose(1, 2)
        targets.index_copy_(0, indices, values)
    return targets


def interpolate_targets(samples, queries):
    """Prepare CPU fields and read their continuous targets at the given queries."""
    return interpolate_field_groups(prepare_field_groups(samples, dtype=queries.dtype), queries)


def prepare_queries(samples, count, rng, *, mode="grid", grid_size=10):
    """Return CPU queries and grid targets; continuous targets are deferred.

    Query RNG consumption is independent of device transfers and interpolation,
    allowing preprocessing to run ahead without changing the sampled sequence.
    """
    from cofl.fields import queries_for_grid, query_valid_mask

    if not samples:
        raise ValueError("Query sampling requires at least one sample")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Query count must be a positive integer")
    if mode not in {"grid", "continuous"}:
        raise ValueError("query sampling must be grid or continuous")
    if any(sample["profile"] not in {"image_field_v1", "ground_sector_v1"} for sample in samples):
        raise ValueError("Unknown field profile")
    if mode == "continuous":
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, (int, np.integer))
            or grid_size < 1
        ):
            raise ValueError("Continuous grid_size must be a positive integer")
        # Each complete tile visits every stratum, with independent jitter.
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        base = np.stack((x, y), axis=-1).reshape(-1, 2)
        tiles = math.ceil(count / len(base))
        noise = rng.random((len(samples), tiles, len(base), 2))
        raw = ((base[None, None] + noise) / grid_size).reshape(len(samples), -1, 2)[:, :count]
        queries = raw.astype(np.float32)
        for index, sample in enumerate(samples):
            if sample["profile"] == "ground_sector_v1":
                queries[index, :, 0] = 2 * queries[index, :, 0] - 1
                queries[index, :, 1] = np.sqrt(queries[index, :, 1])
        return torch.from_numpy(queries), None
    queries, targets = [], []
    for sample in samples:
        grid = queries_for_grid(sample["profile"], sample["geometry"])
        valid = sample["mask"] & query_valid_mask(sample["profile"], sample["geometry"])
        indices = np.flatnonzero(valid.ravel())
        if not len(indices):
            raise ValueError(f"Sample {sample['sample_id']} has no valid in-domain supervision")
        selected = rng.choice(indices, size=count, replace=len(indices) < count)
        queries.append(grid.reshape(-1, 2)[selected])
        targets.append(np.moveaxis(sample["field"], 0, -1).reshape(-1, 2)[selected])
    return (
        torch.as_tensor(np.stack(queries), dtype=torch.float32),
        torch.as_tensor(np.stack(targets), dtype=torch.float32),
    )


def sample_queries(samples, count, rng, device, *, mode="grid", grid_size=10):
    """Prepare queries and return supervision tensors on the requested device."""
    queries, targets = prepare_queries(samples, count, rng, mode=mode, grid_size=grid_size)
    queries = queries.to(device=device, non_blocking=True)
    if targets is None:
        targets = interpolate_targets(samples, queries)
    else:
        targets = targets.to(device=device, non_blocking=True)
    return queries, targets
