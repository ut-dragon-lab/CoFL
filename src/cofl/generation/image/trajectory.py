"""Generate goal-reaching trajectories from the weighted geodesic predecessor tree."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import splev, splprep


class TrajectoryGenerator:
    def __init__(self, parameters: dict):
        self.traj_params = parameters

    def generate(
        self,
        geodesic: np.ndarray,
        geodesic_predecessors: np.ndarray,
        walkable: np.ndarray,
        sdf: np.ndarray,
        goal_boundary_mask: np.ndarray,
    ) -> dict | None:
        """Try to generate a single trajectory from a geodesic predecessor tree.

        This is used during flow generation to validate that a trajectory can be
        generated. Only flows that successfully produce trajectories are kept,
        ensuring 1:1 correspondence.

        Inputs use the source image resolution; resizing happens when saving.

        Args:
            geodesic: Weighted geodesic distance field (H, W) at source resolution
            geodesic_predecessors: Per-pixel predecessor map (H, W) of linear indices
                returned by the multi-source Dijkstra run (with a super-source).
            walkable: Walkable mask at source resolution
            sdf: Obstacle distance field (H, W) at source resolution for clearance
            goal_boundary_mask: Goal set as a boolean mask (H, W) of acceptable goal pixels.
                               For center: any target-adjacent walkable boundary.
                               For directional goals: boundary on that side.

        Returns:
            Dict with 'states' and 'actions' if successful, None otherwise.
        """
        H, W = geodesic.shape
        if geodesic_predecessors.shape != (H, W):
            return None
        if walkable.shape != (H, W) or sdf.shape != (H, W):
            return None

        goal_boundary_mask = goal_boundary_mask.astype(bool)
        if goal_boundary_mask.shape != (H, W):
            return None
        if goal_boundary_mask.sum() == 0:
            return None

        # Precompute distance-to-goal-boundary map in normalized units.
        goal_dt_input = np.ones((H, W), dtype=np.uint8)
        goal_dt_input[goal_boundary_mask] = 0
        goal_dist_px = cv2.distanceTransform(goal_dt_input, cv2.DIST_L2, 3).astype(np.float32)
        scale = float(max(H, W))
        if scale <= 0:
            return None
        goal_dist_norm = goal_dist_px / scale

        min_len = self.traj_params.get("min_length", 16)
        max_steps = self.traj_params.get("max_steps", 200)
        num_output_length = self.traj_params.get("num_output_length", None)
        if num_output_length is None:
            num_output_length = max_steps
        min_dist = self.traj_params.get("min_start_goal_dist", 0.1)
        distance_bias_power = self.traj_params.get("distance_bias_power", 2.0)
        disable_resample = bool(self.traj_params.get("disable_resample", False))

        # Use predecessor validity for reachability to avoid accidentally treating
        # filled/propagated distances as reachable.
        N = int(H * W)
        S = N  # super-source index used in geodesic predecessor construction
        pred = geodesic_predecessors.astype(np.int64)
        reachable_mask = np.isfinite(geodesic) & (geodesic > 0) & (pred >= 0) & (pred != S)
        if reachable_mask.sum() < 10:
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            reachable_mask.astype(np.uint8), connectivity=8
        )
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = int(np.argmax(areas)) + 1
            reachable_mask = labels == largest_label
            if reachable_mask.sum() < 10:
                return None

        max_trials = int(self.traj_params.get("start_sampling_trials", 20))
        max_trials = max(1, min(max_trials, 200))

        trajectory = None
        start_norm = None
        for _ in range(max_trials):
            start_norm = self._sample_biased_start(
                reachable_mask=reachable_mask,
                width=W,
                height=H,
                geodesic=geodesic,
                sdf=sdf,
                goal_dist_norm=goal_dist_norm,
                min_dist=min_dist,
                bias_power=distance_bias_power,
            )
            if start_norm is None:
                continue

            trajectory = self._backtrack_geodesic_trajectory(
                start_norm=start_norm,
                predecessors=pred,
                width=W,
                height=H,
                max_steps=max_steps,
            )

            if trajectory is not None and len(trajectory) >= 2:
                raw_traj = trajectory
                if not disable_resample:
                    num_samples = int(num_output_length)
                    if num_samples < 2:
                        continue

                    use_spline_resample = bool(self.traj_params.get("use_spline_resample", True))
                    if use_spline_resample:
                        spline_s = self.traj_params.get("spline_s", None)
                        smooth_strength = float(self.traj_params.get("smooth_strength", 0.002))
                        trajectory = self._resample_spline(
                            points=trajectory,
                            num_samples=num_samples,
                            spline_s=spline_s,
                            smooth_strength=smooth_strength,
                        )

                        # If spline resampling fails (SciPy missing / degenerate path),
                        # fall back to simple arc-length resampling.
                        if trajectory is None:
                            trajectory = self._resample_polyline(raw_traj, num_samples=num_samples)
                            if trajectory is None:
                                continue
                    else:
                        preserve_vertices = bool(
                            self.traj_params.get("polyline_preserve_vertices", True)
                        )
                        if preserve_vertices:
                            trajectory = self._resample_polyline_preserve_vertices(
                                raw_traj, num_samples=num_samples
                            )
                        else:
                            trajectory = self._resample_polyline(raw_traj, num_samples=num_samples)
                        if trajectory is None:
                            continue

                if len(trajectory) >= min_len:
                    break

        if trajectory is None or len(trajectory) < min_len:
            return None

        states = np.array(trajectory, dtype=np.float32)
        actions = np.zeros_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]

        goal_norm = states[-1]

        return {
            "states": states,
            "actions": actions,
            "start": states[0].tolist(),
            "goal": goal_norm.tolist(),
        }

    def _backtrack_geodesic_trajectory(
        self,
        start_norm: np.ndarray,
        predecessors: np.ndarray,
        width: int,
        height: int,
        max_steps: int,
    ) -> list[np.ndarray] | None:
        """Backtrack a cost-optimal path using the per-pixel predecessor map."""
        H, W = int(height), int(width)
        if predecessors.shape != (H, W):
            return None

        px = int(np.clip(start_norm[0] * W, 0, W - 1))
        py = int(np.clip(start_norm[1] * H, 0, H - 1))

        N = int(H * W)
        S = N
        pred_flat = predecessors.reshape(-1)

        v = int(py * W + px)
        if v < 0 or v >= N:
            return None
        if int(pred_flat[v]) < 0:
            return None

        # Follow predecessors until we hit a goal-boundary source (pred == S).
        path_lin = [v]
        reached = False
        # Hard safety cap to avoid infinite loops on corrupted predecessor maps.
        max_hops = min(int(H * W), max(int(max_steps) * 20, 1000))
        for _ in range(max_hops):
            p = int(pred_flat[v])
            if p == S:
                reached = True
                break
            if p < 0 or p >= N:
                return None
            if p == v:
                return None
            v = p
            path_lin.append(v)

        if not reached or len(path_lin) < 2:
            return None

        # Convert to normalized coordinates (x/W, y/H).
        traj = []
        for idx in path_lin:
            y, x = divmod(int(idx), W)
            traj.append(np.array([float(x / W), float(y / H)], dtype=np.float64))
        return traj

    def _resample_polyline(
        self,
        points: list[np.ndarray],
        num_samples: int,
    ) -> list[np.ndarray] | None:
        """Resample a 2D polyline to a fixed number of points by arc length."""
        if num_samples < 2:
            return None
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            return None

        seg = np.diff(pts, axis=0)
        seg_len = np.sqrt(np.sum(seg * seg, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(s[-1])
        if not np.isfinite(total) or total <= 1e-12:
            out = [pts[0].copy() for _ in range(num_samples)]
            return [np.clip(p, 0.0, 1.0) for p in out]

        targets = np.linspace(0.0, total, num_samples, dtype=np.float64)
        out = []
        j = 0
        for t in targets:
            while j < len(s) - 2 and s[j + 1] < t:
                j += 1
            s0 = float(s[j])
            s1 = float(s[j + 1])
            if s1 <= s0 + 1e-12:
                p = pts[j].copy()
            else:
                a = float((t - s0) / (s1 - s0))
                p = (1.0 - a) * pts[j] + a * pts[j + 1]
            out.append(np.clip(p, 0.0, 1.0))
        return out

    def _resample_polyline_preserve_vertices(
        self,
        points: list[np.ndarray],
        num_samples: int,
    ) -> list[np.ndarray] | None:
        """Resample by arc length while avoiding corner cutting.

        Pure arc-length resampling can change the *shape* when visualizing the
        resampled points as a new polyline: if two consecutive output points lie
        on different original segments, the straight line between them becomes a
        chord that cuts across the corner.

        This variant snaps the next output point to the intermediate vertex when
        the sampling crosses into a later segment, so the reconstructed polyline
        stays on the original polyline geometry as much as possible.

        Limitation: if the original polyline contains more corners than
        `num_samples`, it is impossible to preserve all corners with a fixed
        output length.
        """
        if num_samples < 2:
            return None
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            return None

        # Remove consecutive duplicates to avoid degenerate segments.
        keep = [0]
        for i in range(1, pts.shape[0]):
            if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > 1e-12:
                keep.append(i)
        pts = pts[keep]
        if pts.shape[0] < 2:
            out = [pts[0].copy() for _ in range(num_samples)]
            return [np.clip(p, 0.0, 1.0) for p in out]

        seg = np.diff(pts, axis=0)
        seg_len = np.sqrt(np.sum(seg * seg, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(s[-1])
        if not np.isfinite(total) or total <= 1e-12:
            out = [pts[0].copy() for _ in range(num_samples)]
            return [np.clip(p, 0.0, 1.0) for p in out]

        targets = np.linspace(0.0, total, num_samples, dtype=np.float64)
        out: list[np.ndarray] = []
        j = 0
        last_seg = 0
        for t in targets:
            while j < len(s) - 2 and s[j + 1] < t:
                j += 1
            seg_idx = int(j)

            # If we advanced to a later segment, emit the next vertex first.
            if seg_idx > last_seg:
                v_idx = min(last_seg + 1, pts.shape[0] - 1)
                out.append(np.clip(pts[v_idx].copy(), 0.0, 1.0))
                last_seg = v_idx
                continue

            s0 = float(s[seg_idx])
            s1 = float(s[seg_idx + 1])
            if s1 <= s0 + 1e-12:
                p = pts[seg_idx].copy()
            else:
                a = float((t - s0) / (s1 - s0))
                p = (1.0 - a) * pts[seg_idx] + a * pts[seg_idx + 1]
            out.append(np.clip(p, 0.0, 1.0))

        if len(out) >= 2:
            out[0] = np.clip(pts[0].copy(), 0.0, 1.0)
            out[-1] = np.clip(pts[-1].copy(), 0.0, 1.0)
        return out

    def _resample_spline(
        self,
        points: list[np.ndarray],
        num_samples: int,
        spline_s: float | None = None,
        smooth_strength: float = 0.002,
    ) -> list[np.ndarray] | None:
        """Smooth + resample using scipy.interpolate.splprep + splev.

        This is the preferred top-level post-process: it both increases smoothness
        and outputs exactly `num_samples` points.

        Args:
            points: list of normalized (x, y) points in [0, 1]
            num_samples: number of samples to output
            spline_s: optional explicit SciPy smoothing factor `s`.
                     If None, derive from `smooth_strength`.
            smooth_strength: normalized knob used only when spline_s is None.
        """
        if num_samples < 2:
            return None
        if points is None or len(points) < 2:
            return None

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None

        # Remove consecutive duplicates to avoid splprep failures.
        keep = [0]
        for i in range(1, pts.shape[0]):
            if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > 1e-12:
                keep.append(i)
        pts = pts[keep]

        if pts.shape[0] < 2:
            out = [np.clip(pts[0].copy(), 0.0, 1.0) for _ in range(num_samples)]
            return out

        # Parameterize by arc length for stability.
        seg = np.diff(pts, axis=0)
        seg_len = np.sqrt(np.sum(seg * seg, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(s[-1])
        if not np.isfinite(total) or total <= 1e-12:
            out = [np.clip(pts[0].copy(), 0.0, 1.0) for _ in range(num_samples)]
            return out
        u = (s / total).astype(np.float64)

        # Baseline (linear) resample used to stabilize endpoints.
        baseline = self._resample_polyline([p.copy() for p in pts], num_samples=int(num_samples))
        baseline_arr = None
        if baseline is not None:
            baseline_arr = np.asarray(baseline, dtype=np.float64)

        try:
            n = int(pts.shape[0])
            k = int(min(3, n - 1))

            if spline_s is None:
                # Scale with number of points so the knob is less sensitive.
                smooth_strength = float(np.clip(smooth_strength, 0.0, 1.0))
                spline_s_val = float(smooth_strength) * float(n)
            else:
                spline_s_val = float(spline_s)

            # Heavily weight endpoints (and optionally near-endpoints) to reduce
            # boundary artifacts / endpoint kinks.
            endpoint_weight = float(self.traj_params.get("endpoint_weight", 50.0))
            endpoint_weight = float(np.clip(endpoint_weight, 1.0, 1e6))
            w = np.ones(n, dtype=np.float64)
            w[0] = endpoint_weight
            w[-1] = endpoint_weight
            if n >= 4:
                w[1] = max(1.0, endpoint_weight * 0.25)
                w[-2] = max(1.0, endpoint_weight * 0.25)

            tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=spline_s_val, k=k, w=w)
            u_new = np.linspace(0.0, 1.0, int(num_samples), dtype=np.float64)
            x_new, y_new = splev(u_new, tck)
            out = np.stack([x_new, y_new], axis=1)
        except (ValueError, TypeError, RuntimeError):
            return None

        out = np.clip(out, 0.0, 1.0)

        # Blend endpoints with a linear baseline to avoid creating sharp turns
        # when the spline doesn't pass exactly through the first/last point.
        if baseline_arr is not None and baseline_arr.shape == out.shape:
            blend_len = int(self.traj_params.get("endpoint_blend_len", 4))
            blend_len = max(0, min(blend_len, max(0, out.shape[0] // 2 - 1)))

            out[0] = baseline_arr[0]
            out[-1] = baseline_arr[-1]

            for i in range(1, blend_len + 1):
                a = float(i) / float(blend_len + 1)
                out[i] = (1.0 - a) * baseline_arr[i] + a * out[i]
                j = out.shape[0] - 1 - i
                out[j] = (1.0 - a) * baseline_arr[j] + a * out[j]

            out = np.clip(out, 0.0, 1.0)

        return [out[i].astype(np.float64) for i in range(out.shape[0])]

    def _sample_biased_start(
        self,
        reachable_mask: np.ndarray,
        width: int,
        height: int,
        geodesic: np.ndarray,
        sdf: np.ndarray,
        goal_dist_norm: np.ndarray,
        min_dist: float,
        bias_power: float = 2.0,
        min_clearance: float = 5.0,
        max_attempts: int = 100,
    ) -> np.ndarray | None:
        """Sample a start position with bias toward farther points."""
        clearance_mask = reachable_mask & (sdf >= min_clearance)
        valid_y, valid_x = np.where(clearance_mask)

        if len(valid_x) == 0:
            valid_y, valid_x = np.where(reachable_mask)
            if len(valid_x) == 0:
                return None

        n_points = len(valid_x)
        distances = geodesic[valid_y, valid_x].astype(np.float64)

        max_geo = distances.max()
        if max_geo > 0:
            normalized = distances / max_geo
            weights = np.power(normalized + 0.01, bias_power)
        else:
            weights = np.ones(n_points, dtype=np.float64)

        probs = weights / weights.sum()

        for _ in range(max_attempts):
            idx = np.random.choice(n_points, p=probs)
            cand_norm = np.array(
                [
                    valid_x[idx] / width,
                    valid_y[idx] / height,
                ],
                dtype=np.float64,
            )

            px = int(np.clip(cand_norm[0] * width, 0, width - 1))
            py = int(np.clip(cand_norm[1] * height, 0, height - 1))
            if float(goal_dist_norm[py, px]) >= min_dist:
                return cand_norm

        return None
