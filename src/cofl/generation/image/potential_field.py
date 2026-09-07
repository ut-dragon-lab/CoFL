"""
Potential field computation for navigation.

Contains obstacle distances, potential fields, and static velocity fields.
"""

from typing import NamedTuple

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra


class PotentialFieldResult(NamedTuple):
    """Potential and weighted-path geometry at the source image resolution."""

    potential: np.ndarray
    weighted_distance: np.ndarray
    pixel_distance: np.ndarray
    predecessors: np.ndarray


def compute_obstacle_distance(obstacle_map: np.ndarray) -> np.ndarray:
    """Compute an obstacle distance field.

    Returns a float32 array where:
    - inside obstacles: positive distance to the nearest obstacle boundary
    - outside obstacles: 0

    Note: This is not a full signed distance field (it has no non-zero values
    in free space). Callers should not rely on a negative/positive sign split
    between free/occupied.
    """
    obs = obstacle_map.astype(bool)
    dist_in = distance_transform_edt(obs).astype(np.float32)
    sdf = np.zeros_like(obstacle_map, dtype=np.float32)
    sdf[obs] = dist_in[obs]
    return sdf


def analyze_target_accessibility(
    target_mask: np.ndarray,
    walkable_mask: np.ndarray,
    bbox: list[int],
    center: tuple[int, int],
    check_margin: int = 20,
    min_walkable_pixels: int = 50,
) -> dict:
    """
    Analyze how a target object interfaces with the walkable area.

    Determines which directions (front/back/left/right) have walkable access
    by checking if there is walkable area on each side of the target's bounding box.

    Args:
        target_mask: Binary mask of the target object
        walkable_mask: Binary mask of walkable areas
        bbox: [x_min, y_min, x_max, y_max] of the target
        center: (cx, cy) center of the target
        check_margin: How far to check for walkable area
        min_walkable_pixels: Minimum walkable pixels to consider accessible

    Returns:
        Dict with:
        - 'boundary_mask': mask of target pixels adjacent to walkable
        - 'accessible_directions': list of directions with walkable access
        - 'direction_boundaries': dict mapping direction -> boundary mask for that direction
    """
    H, W = walkable_mask.shape
    x_min, y_min, x_max, y_max = bbox
    cx, cy = center

    # Get largest connected component of walkable area (main walkable region).
    # NOTE: For the non-directional (center) goal we should allow ANY walkable
    # component that actually touches the target boundary. Directional goals
    # (front/back/left/right) are only meaningful when that side touches the
    # main walkable region.
    walkable_u8 = walkable_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walkable_u8, connectivity=8)

    if num_labels <= 1:
        return {
            "boundary_mask": np.zeros((H, W), dtype=bool),
            "accessible_directions": [],
            "direction_boundaries": {},
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    main_walkable = labels == largest_label

    # Dilate target mask to find adjacent pixels
    kernel = np.ones((3, 3), np.uint8)
    target_u8 = target_mask.astype(np.uint8)
    dilated_target = cv2.dilate(target_u8, kernel, iterations=1)

    # Boundary pixels in WALKABLE space adjacent to target.
    # - boundary_any: adjacent pixels in ANY walkable component (for center goal)
    # - boundary_main: subset adjacent to the largest walkable component (for directional goals)
    boundary_any = (dilated_target > 0) & walkable_mask & (~target_mask)
    boundary_main = boundary_any & main_walkable

    if not np.any(boundary_any):
        # Target is isolated/inside obstacle.
        # Find closest point in ANY walkable area to target center.
        ys, xs = np.where(walkable_mask)
        if len(xs) == 0:
            return {
                "boundary_mask": np.zeros((H, W), dtype=bool),
                "accessible_directions": [],
                "direction_boundaries": {},
            }

        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        idx = np.argmin(d2)
        nearest_y, nearest_x = ys[idx], xs[idx]

        fallback_boundary = np.zeros((H, W), dtype=bool)
        fallback_boundary[nearest_y, nearest_x] = True

        return {
            "boundary_mask": fallback_boundary,
            "accessible_directions": [],  # Only center allowed
            "direction_boundaries": {},
        }

    # Check each direction by looking at regions OUTSIDE the target bbox
    # A direction is accessible if there is walkable area on that side of the target
    accessible_directions = []
    direction_boundaries = {}

    # Front = above target (y < y_min)
    front_region_y_start = max(0, y_min - check_margin)
    front_region = main_walkable[front_region_y_start:y_min, x_min:x_max]
    # Front boundary: boundary pixels (main component) that are above the target (y < y_min)
    front_boundary = boundary_main.copy()
    front_boundary[y_min:, :] = False  # Keep only y < y_min
    # Only mark as accessible if we have enough walkable region AND valid boundary points
    if np.sum(front_region) >= min_walkable_pixels and np.sum(front_boundary) > 0:
        accessible_directions.append("front")
        direction_boundaries["front"] = front_boundary

    # Back = below target (y > y_max)
    back_region_y_end = min(H, y_max + check_margin)
    back_region = main_walkable[y_max:back_region_y_end, x_min:x_max]
    # Back boundary: boundary pixels (main component) that are below the target (y >= y_max)
    back_boundary = boundary_main.copy()
    back_boundary[:y_max, :] = False  # Keep only y >= y_max
    if np.sum(back_region) >= min_walkable_pixels and np.sum(back_boundary) > 0:
        accessible_directions.append("back")
        direction_boundaries["back"] = back_boundary

    # Left = left of target (x < x_min)
    left_region_x_start = max(0, x_min - check_margin)
    left_region = main_walkable[y_min:y_max, left_region_x_start:x_min]
    # Left boundary: boundary pixels (main component) that are left of the target (x < x_min)
    left_boundary = boundary_main.copy()
    left_boundary[:, x_min:] = False  # Keep only x < x_min
    if np.sum(left_region) >= min_walkable_pixels and np.sum(left_boundary) > 0:
        accessible_directions.append("left")
        direction_boundaries["left"] = left_boundary

    # Right = right of target (x > x_max)
    right_region_x_end = min(W, x_max + check_margin)
    right_region = main_walkable[y_min:y_max, x_max:right_region_x_end]
    # Right boundary: boundary pixels (main component) that are right of the target (x >= x_max)
    right_boundary = boundary_main.copy()
    right_boundary[:, :x_max] = False  # Keep only x >= x_max
    if np.sum(right_region) >= min_walkable_pixels and np.sum(right_boundary) > 0:
        accessible_directions.append("right")
        direction_boundaries["right"] = right_boundary

    return {
        # Center goal should accept reaching ANY walkable-adjacent boundary
        # of the target, not just the main walkable component.
        "boundary_mask": boundary_any,
        "accessible_directions": accessible_directions,
        "direction_boundaries": direction_boundaries,
    }


def select_goal_directions(
    accessible_directions: list[str],
    max_directional: int = 2,
) -> list[str]:
    """
    Select which directions to use for goal generation.

    Rules:
    - Always include 'center' (general approach)
    - Randomly select up to max_directional directions from accessible ones

    Args:
        accessible_directions: List of accessible directions from analyze_target_accessibility
        max_directional: Maximum number of directional goals (besides center)

    Returns:
        List of selected directions (always includes 'center')
    """
    import random

    selected = ["center"]

    if not accessible_directions:
        return selected

    # Randomly select up to max_directional from accessible directions
    available = list(accessible_directions)
    random.shuffle(available)
    selected.extend(available[:max_directional])

    return selected


def build_safety_cost_map(
    free_mask: np.ndarray,
    safe_radius: float,
    safety_cost_weight: float,
) -> np.ndarray:
    """Build per-pixel traversal cost map used by weighted geodesic.

    This is view-invariant (depends only on free_mask), so callers can compute it
    once per view and reuse for all goal boundaries.
    """
    dist_to_obs = distance_transform_edt(free_mask)
    safety_cost = np.zeros_like(dist_to_obs, dtype=np.float32)
    near_mask = dist_to_obs < safe_radius
    if near_mask.any():
        safety_cost[near_mask] = safety_cost_weight * (safe_radius - dist_to_obs[near_mask])
    return (1.0 + safety_cost).astype(np.float32)


def build_weighted_grid_graph(
    free_mask: np.ndarray,
    cost_map: np.ndarray,
    allow_diagonal: bool = True,
) -> sp.csr_matrix:
    """Build a weighted grid graph (CSR) for Dijkstra.

    The graph depends only on free_mask + cost_map, so it can be cached per view.
    Edge weights combine geometric step length with the mean endpoint cost.
    """
    H, W = free_mask.shape
    num_pixels = H * W
    indices = np.arange(num_pixels).reshape(H, W)

    rows = []
    cols = []
    data = []

    def add_edges(dy: int, dx: int, weight_mult: float):
        y_start, y_end = max(0, -dy), min(H, H - dy)
        x_start, x_end = max(0, -dx), min(W, W - dx)

        src_idx = indices[y_start:y_end, x_start:x_end]
        dst_idx = indices[y_start + dy : y_end + dy, x_start + dx : x_end + dx]

        src_free = free_mask[y_start:y_end, x_start:x_end]
        dst_free = free_mask[y_start + dy : y_end + dy, x_start + dx : x_end + dx]
        valid = src_free & dst_free
        if not valid.any():
            return

        s = src_idx[valid]
        d = dst_idx[valid]

        c_src = cost_map[y_start:y_end, x_start:x_end][valid]
        c_dst = cost_map[y_start + dy : y_end + dy, x_start + dx : x_end + dx][valid]
        w = 0.5 * (c_src + c_dst) * weight_mult

        rows.append(s)
        cols.append(d)
        data.append(w)

    add_edges(1, 0, 1.0)
    add_edges(-1, 0, 1.0)
    add_edges(0, 1, 1.0)
    add_edges(0, -1, 1.0)

    if allow_diagonal:
        sqrt2 = np.sqrt(2)
        add_edges(1, 1, sqrt2)
        add_edges(1, -1, sqrt2)
        add_edges(-1, 1, sqrt2)
        add_edges(-1, -1, sqrt2)

    if not rows:
        return sp.csr_matrix((num_pixels, num_pixels), dtype=np.float32)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data).astype(np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(num_pixels, num_pixels)).tocsr()


def _weighted_geodesic_on_graph(
    graph: sp.spmatrix,
    free_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted distance, pixel path length, and predecessor indices.

    Pixel length follows the minimum-cost predecessor tree: cardinal edges
    contribute one pixel, diagonal edges sqrt(2). Unreachable pixels retain
    invalid predecessors even when their distances use nearest-reachable fill.
    """
    H, W = free_mask.shape
    N = H * W

    sources = np.where(boundary_mask.ravel() & free_mask.ravel())[0]
    if len(sources) == 0:
        inf = np.full((H, W), np.inf, dtype=np.float32)
        return inf, inf.copy(), np.full((H, W), -9999, dtype=np.int64)

    # A super-source connects every goal pixel with a zero-cost edge.
    S = N
    G = graph.tocsr()

    row = np.zeros(len(sources), dtype=np.int32)
    col = sources.astype(np.int32)
    data = np.zeros(len(sources), dtype=np.float32)
    S_to_sources = sp.csr_matrix((data, (row, col)), shape=(1, N))

    G_top = sp.vstack([G, S_to_sources], format="csr")  # (N+1) x N
    G_ext = sp.hstack(
        [G_top, sp.csr_matrix((N + 1, 1), dtype=np.float32)], format="csr"
    )  # (N+1)x(N+1)

    dist_all, pred_all = dijkstra(G_ext, indices=S, return_predecessors=True)
    dist_flat = dist_all[:N].astype(np.float32)
    pred = pred_all[:N].astype(np.int64)
    dist_geo = dist_flat.reshape(H, W).astype(np.float32)

    pred_map = pred.reshape(H, W).astype(np.int64)

    dist_pix_flat = np.full(N, np.inf, dtype=np.float32)
    order = np.argsort(dist_flat, kind="mergesort")
    for v in order:
        if not np.isfinite(dist_flat[v]):
            break
        p = pred[v]
        if p == S:
            dist_pix_flat[v] = 0.0
            continue
        if p < 0 or not np.isfinite(dist_pix_flat[p]):
            continue

        vy, vx = divmod(int(v), W)
        py, px = divmod(int(p), W)
        dy = float(abs(vy - py))
        dx = float(abs(vx - px))
        # With 4/8-neighborhood edges, this is either 1 or sqrt(2).
        step = float(np.sqrt(dx * dx + dy * dy))
        dist_pix_flat[v] = dist_pix_flat[p] + step

    dist_geo_pixel = dist_pix_flat.reshape(H, W).astype(np.float32)

    # Apply the same nearest-reachable fill to the distance outputs.
    reachable = np.isfinite(dist_geo)
    if not reachable.all() and reachable.any():
        missing = free_mask & (~reachable)
        if missing.any():
            idx = distance_transform_edt(~reachable, return_distances=False, return_indices=True)
            ny = idx[0][missing]
            nx = idx[1][missing]
            dist_geo[missing] = dist_geo[ny, nx]
            dist_geo_pixel[missing] = dist_geo_pixel[ny, nx]

    return dist_geo, dist_geo_pixel, pred_map


def compute_potential_field_from_boundary(
    obstacle_map: np.ndarray,
    sdf_map: np.ndarray,
    boundary_mask: np.ndarray,
    walkable_mask: np.ndarray,
    goal_weight: float = 1.0,
    sdf_band_weight: float = 5.0,
    safe_radius: float = 10.0,  # Distance to start penalizing
    safety_cost_weight: float = 5.0,  # Max added cost for safety
    precomputed_graph: sp.spmatrix | None = None,
) -> PotentialFieldResult:
    """Compute the potential and complete geometry of a weighted shortest path.

    Safety costs make near-obstacle steps more expensive. A cached graph may
    be supplied for reuse across goals in the same view; otherwise it is built
    from the walkable mask, safe radius, and safety cost weight.
    """
    H, W = obstacle_map.shape
    potential = np.zeros((H, W), dtype=np.float32)

    # Use the full walkable mask; graph-based Dijkstra naturally isolates components.
    free_mask = walkable_mask.astype(bool)

    graph = precomputed_graph
    if graph is None:
        cost_map = build_safety_cost_map(
            free_mask=free_mask,
            safe_radius=safe_radius,
            safety_cost_weight=safety_cost_weight,
        )
        graph = build_weighted_grid_graph(free_mask, cost_map, allow_diagonal=True)
    dist_geo, dist_geo_pixel, pred_map = _weighted_geodesic_on_graph(
        graph, free_mask, boundary_mask
    )

    reachable = np.isfinite(dist_geo)
    max_geo_dist = 0.0
    if reachable.any():
        d = dist_geo[reachable]
        max_geo_dist = d.max()
        potential[reachable] = goal_weight * d

    inside = sdf_map > 0
    if inside.any():
        max_free_potential = goal_weight * max_geo_dist
        effective_penalty = abs(max_free_potential) + 50.0
        potential[inside] = sdf_band_weight * sdf_map[inside] + effective_penalty

    return PotentialFieldResult(potential, dist_geo, dist_geo_pixel, pred_map)


def compute_velocity_field(
    potential_field: np.ndarray,
    geodesic_distance: np.ndarray,
    sdf_map: np.ndarray | None = None,
    smoothing_iterations: int = 2,
    smoothing_kernel_size: int = 5,
    smoothing_sigma: float = 1.0,
    inside_obstacle_speed_norm: float = 1.0,
    max_speed_norm: float | None = None,
    static_length_scale: float | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return a static (2,H,W) field in normalized image coordinates.

    Direction follows the smoothed negative potential gradient. Magnitude is
    remaining path length divided by image width/height per component, with
    bounded escape vectors where the obstacle-distance field is positive.
    """
    H, W = potential_field.shape
    phi = potential_field.astype(np.float32)

    # Smooth potential for stable gradients
    phi = cv2.GaussianBlur(phi, (smoothing_kernel_size, smoothing_kernel_size), smoothing_sigma)

    # Compute gradient (direction toward LOWER potential = toward goal)
    grad_x_pix = -cv2.Sobel(phi, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_pix = -cv2.Sobel(phi, cv2.CV_32F, 0, 1, ksize=3)

    # Compute magnitude and normalize to unit vectors
    mag_pix = np.sqrt(grad_x_pix**2 + grad_y_pix**2)

    # Unit direction vectors
    dir_x = grad_x_pix / (mag_pix + eps)
    dir_y = grad_y_pix / (mag_pix + eps)

    # Vector Field Smoothing
    for _ in range(smoothing_iterations):
        dir_x = cv2.GaussianBlur(
            dir_x, (smoothing_kernel_size, smoothing_kernel_size), smoothing_sigma
        )
        dir_y = cv2.GaussianBlur(
            dir_y, (smoothing_kernel_size, smoothing_kernel_size), smoothing_sigma
        )

        # Re-normalize after smoothing
        mag = np.sqrt(dir_x**2 + dir_y**2)
        dir_x = dir_x / (mag + eps)
        dir_y = dir_y / (mag + eps)

    # Static mode: return f(x) (NOT divided by (1-t)).
    # Choose how to convert geodesic length to a normalized distance scale.
    if geodesic_distance is None:
        raise ValueError("geodesic_distance is required for a static field")

    geo = geodesic_distance.astype(np.float32)
    finite = np.isfinite(geo)
    geo_safe = np.where(finite, geo, 0.0).astype(np.float32)

    # Static mode magnitude is locked to image-scale conversion.
    # Convert pixel-distance to normalized-coordinate velocity per-axis.
    # Semantics: f_pix(x) ~ dir(x) * d_geo(x), then v_norm = f_pix / [W, H].
    if static_length_scale is None:
        denom_x = float(W)
        denom_y = float(H)
    else:
        denom_x = denom_y = None
        if (
            isinstance(static_length_scale, (tuple, list, np.ndarray))
            and len(static_length_scale) == 2
        ):
            denom_x = float(static_length_scale[0])
            denom_y = float(static_length_scale[1])
        else:
            denom_x = float(static_length_scale)
            denom_y = float(static_length_scale)

        if denom_x <= 0 or not np.isfinite(denom_x):
            denom_x = float(W)
        if denom_y <= 0 or not np.isfinite(denom_y):
            denom_y = float(H)

    geo_use = np.where(finite, geo_safe, 0.0).astype(np.float32)
    mag_norm_x = (geo_use / (denom_x + eps)).astype(np.float32)
    mag_norm_y = (geo_use / (denom_y + eps)).astype(np.float32)

    # Inside-obstacle handling: bounded escape speed (constant over time).
    # Blend magnitude first, then multiply by direction once.
    if sdf_map is not None:
        sdf = sdf_map.astype(np.float32)
        inside = sdf > 0
        if inside.any():
            esc_mag = float(inside_obstacle_speed_norm)
            mag_norm_x = np.where(inside, esc_mag, mag_norm_x).astype(np.float32)
            mag_norm_y = np.where(inside, esc_mag, mag_norm_y).astype(np.float32)

    v_norm_x = (dir_x * mag_norm_x).astype(np.float32)
    v_norm_y = (dir_y * mag_norm_y).astype(np.float32)

    # Optional global speed cap (normalized coords per unit t)
    if max_speed_norm is not None and max_speed_norm > 0:
        mag = np.sqrt(v_norm_x**2 + v_norm_y**2)
        scale = np.minimum(1.0, float(max_speed_norm) / (mag + eps))
        v_norm_x = v_norm_x * scale
        v_norm_y = v_norm_y * scale

    # Defensive: remove any non-finite values
    v_norm_x = np.where(np.isfinite(v_norm_x), v_norm_x, 0.0)
    v_norm_y = np.where(np.isfinite(v_norm_y), v_norm_y, 0.0)

    return np.stack([v_norm_x, v_norm_y], axis=0).astype(np.float32)  # [2, H, W]
