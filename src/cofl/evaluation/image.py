"""Image navigation scores against an explicit goal and observed free-space raster."""

from __future__ import annotations

import numpy as np

from .metrics import resample_trajectory


def _touches_obstacle(points: np.ndarray, navigation_mask: np.ndarray) -> bool:
    """Check all raster cells touched by grid-coordinate points, including edges."""
    height, width = navigation_mask.shape
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] <= width)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height)
    )
    points = points[inside]
    if not len(points):
        return False
    nearest = np.rint(points)
    on_edge = np.abs(points - nearest) <= 1e-10
    points = np.where(on_edge, nearest, points)
    cells = np.floor(points).astype(np.int64)
    for dx, dy in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        selected = np.ones(len(points), dtype=bool)
        if dx:
            selected &= on_edge[:, 0]
        if dy:
            selected &= on_edge[:, 1]
        x, y = cells[:, 0] + dx, cells[:, 1] + dy
        selected &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(~navigation_mask[y[selected], x[selected]]):
            return True
    return False


def _path_collision(prediction: np.ndarray, navigation_mask: np.ndarray) -> bool:
    """Visit every segment's grid crossings and intervening cells in batches.

    Crossing times and interval midpoints cover every touched raster cell,
    including boundary contacts. Segment batches bound temporary storage for
    long paths and permit early exit after a collision.
    """
    height, width = navigation_mask.shape
    points = prediction * np.array([width, height], dtype=np.float64)
    if len(points) == 1:
        return _touches_obstacle(points, navigation_mask)
    for offset in range(0, len(points) - 1, 128):
        starts = points[offset : offset + 128]
        ends = points[offset + 1 : offset + 129]
        starts = starts[: len(ends)]
        delta = ends - starts
        segment_ids = [np.repeat(np.arange(len(starts)), 2)]
        crossings = [np.tile([0.0, 1.0], len(starts))]
        for axis, size in enumerate((width, height)):
            lower = np.ceil(np.maximum(0, np.minimum(starts[:, axis], ends[:, axis])))
            upper = np.floor(np.minimum(size, np.maximum(starts[:, axis], ends[:, axis])))
            selected = np.flatnonzero((delta[:, axis] != 0) & (lower <= upper))
            if not len(selected):
                continue
            counts = (upper[selected] - lower[selected] + 1).astype(np.int64)
            ids = np.repeat(selected, counts)
            offsets = np.repeat(np.cumsum(counts) - counts, counts)
            grid_lines = np.repeat(lower[selected], counts) + np.arange(len(ids)) - offsets
            segment_ids.append(ids)
            crossings.append((grid_lines - starts[ids, axis]) / delta[ids, axis])
        ids, times = np.concatenate(segment_ids), np.concatenate(crossings)
        valid = (times >= 0) & (times <= 1)
        ids, times = ids[valid], times[valid]
        order = np.lexsort((times, ids))
        ids, times = ids[order], times[order]
        # Duplicate crossings add only repeated points or zero-length intervals.
        same_segment = ids[:-1] == ids[1:]
        midpoint_ids = ids[:-1][same_segment]
        midpoints = ((times[:-1] + times[1:]) / 2)[same_segment]
        ids = np.concatenate((ids, midpoint_ids))
        times = np.concatenate((times, midpoints))
        if _touches_obstacle(starts[ids] + times[:, None] * delta[ids], navigation_mask):
            return True
    return False


def evaluate_image_navigation(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    goal: np.ndarray,
    navigation_mask: np.ndarray,
    resample_points: int = 101,
) -> dict:
    """Score an image-space path without modifying it or using labels to stop it.

    Coordinates are normalized image ``(right, down)``. ``navigation_mask`` is
    a nonempty boolean raster: true means traversable, false means obstacle.
    The reference and independently supplied goal must lie in ``[0, 1]^2``.
    Prediction points outside that domain are counted as unsafe, never clipped.

    FGE is endpoint-to-goal Euclidean distance. PLR divides lengths measured
    after independently resampling both paths uniformly in arc length. Curv
    is the mean absolute heading change, in radians, on that resampled path;
    segments of length at most ``1e-8`` have no defined heading. A reference
    shorter than ``1e-8`` has undefined PLR, and fewer than two directed
    segments give undefined Curv. Both return ``None`` with explicit counts.

    Collision checks every cell touched by every original path segment. Grid
    edges and corners touch cells on both sides, so even contact with an
    obstacle counts. CR is a per-path binary unsafe indicator: collision OR
    leaving the image. Its mean across paths is the conservative collision
    rate; the two causes are also reported separately. It is a point-path
    raster score and does not model a robot footprint or physical clearance.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    mask = np.asarray(navigation_mask)
    if mask.ndim != 2 or not mask.size or mask.dtype != np.bool_:
        raise ValueError("navigation_mask must be a nonempty two-dimensional boolean raster")
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("goal must be a finite normalized image point of shape (2,)")
    if np.any((target < 0) | (target > 1)):
        raise ValueError("goal must be inside the normalized image")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            sampled_pred = resample_trajectory(pred, resample_points)
            sampled_truth = resample_trajectory(truth, resample_points)
            if np.any((truth < 0) | (truth > 1)):
                raise ValueError("reference must be inside the normalized image")
            segments = np.diff(sampled_pred, axis=0)
            segment_lengths = np.linalg.norm(segments, axis=1)
            predicted_length = float(segment_lengths.sum())
            reference_length = float(np.linalg.norm(np.diff(sampled_truth, axis=0), axis=1).sum())
            segments = segments[segment_lengths > 1e-8]
            turn_count = max(0, len(segments) - 1)
            curv = None
            if turn_count:
                headings = np.arctan2(segments[:, 1], segments[:, 0])
                turns = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
                curv = float(np.abs(turns).mean())
            fge = float(np.linalg.norm(pred[-1] - target))
            collision = _path_collision(pred, mask)
    except FloatingPointError as error:
        raise ValueError("Image trajectory arithmetic must remain finite") from error
    out_of_bounds_count = int(np.any((pred < 0) | (pred > 1), axis=1).sum())
    out_of_bounds = out_of_bounds_count > 0
    degenerate_reference = reference_length < 1e-8
    return {
        "fge": fge,
        "cr": float(collision or out_of_bounds),
        "plr": None if degenerate_reference else predicted_length / reference_length,
        "curv": curv,
        "collision": collision,
        "out_of_bounds": out_of_bounds,
        "out_of_bounds_count": out_of_bounds_count,
        "degenerate_reference": degenerate_reference,
        "turn_count": turn_count,
        "predicted_path_length": predicted_length,
        "reference_path_length": reference_length,
        "prediction_point_count": len(pred),
        "reference_point_count": len(truth),
        "resample_points": int(resample_points),
        "coordinate_frame": "image_xy_right_down",
        "distance_unit": "image_fraction",
        "curv_unit": "rad_per_sampled_vertex",
        "plr_unit": "dimensionless",
        "cr_definition": "per_path_collision_or_out_of_bounds",
        "collision_definition": "segment_supercover_with_boundary_contact",
        "length_definition": "arc_resampled_polyline",
        "length_epsilon": 1e-8,
    }
