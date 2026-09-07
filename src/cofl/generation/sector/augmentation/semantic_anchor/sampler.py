"""AlternativeSampler — emit K-1 contrastive CandidateCommands per frame.

Given a confirmed anchor command plus the scene context
(visible objects / regions, pose, pathfinder) this class draws up to
``K-1`` alternative ``CandidateCommand``s that are semantically distinct
from the anchor and form valid training negatives.  The pool is assembled
from three families and then sub-sampled at random:

1. **Object-grounded** — one candidate per visible object passing the
   strict pixel-count gate, in two variants (``toward_object``,
   ``go_past``). ``toward_object`` carries both "go to X" and "stop at
   X" surface forms — same flow field, single template pool.
2. **Direction-grounded** — the ``forward`` probe (strict corridor check).
3. **Region-grounded** — ``region_enter`` for rooms that are both visible
   AND reachable via a clean geodesic (``_sample_region_alternatives``).
4. **Geometric object fallback** — when no visible-objects list is
   provided (e.g. inference), pull from ``semantic_scene.objects`` with a
   LOS + FOV gate (``_sample_object_alternatives``).

The last step uniformly samples ``K-1`` entries from the assembled pool.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..scene_analyzer.navigation import _geodesic_distance, heading_to_dirs
from .geometry import (
    _collect_stairs_aabbs,
    _has_line_of_sight,
    _path_in_fov,
    _project_object_onto_navmesh,
    _region_entry_point,
    _sample_forward_far_point,
)
from .types import CandidateCommand, VisibleObject, VisibleRegion, _load_sampler_cfg


class AlternativeSampler:
    """Sample ``K - 1`` alternative ``CandidateCommand``s for a given frame."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = _load_sampler_cfg(cfg)
        self._K = int(self.cfg["K"])

    @property
    def K(self) -> int:
        return self._K

    def _family_enabled(self, family: str) -> bool:
        """Return the configured enable state of a supported instruction family."""
        enabled = self.cfg.get("families_enabled") or {}
        return bool(enabled.get(family, True))

    def sample(
        self,
        *,
        anchor: CandidateCommand,
        visible_objects: list[VisibleObject],
        agent_position: np.ndarray,
        agent_heading: float,
        pathfinder: Any,
        semantic_scene: Any,
        current_region: str | None,
        rng: np.random.Generator,
        visible_regions: list[VisibleRegion] | None = None,
    ) -> list[CandidateCommand]:
        n_alt = max(0, self._K - 1)
        if n_alt == 0:
            return []
        cfg = self.cfg
        agent_pos = np.asarray(agent_position, dtype=np.float64).reshape(3)
        (forward_w, left_w) = heading_to_dirs(agent_heading)
        anchor_obj_id = anchor.object_id
        anchor_region = anchor.region_name
        anchor_is_last = bool((anchor.metadata or {}).get("is_last_subinstr", False))
        pool: list[CandidateCommand] = []
        vo_qualifier: dict[int, str] = {}
        cat_groups: dict[str, list[int]] = {}
        for j, vo in enumerate(visible_objects):
            cat_groups.setdefault(vo.mpcat40, []).append(j)
        fwd_xz = forward_w.astype(np.float64)
        for cat, idxs in cat_groups.items():
            if len(idxs) <= 1:
                continue
            info: list[tuple[int, float, float]] = []
            for j in idxs:
                vo = visible_objects[j]
                d_xyz = np.asarray(vo.centroid, dtype=np.float64) - agent_pos
                lat = -fwd_xz[2] * float(d_xyz[0]) + fwd_xz[0] * float(d_xyz[2])
                info.append((j, float(vo.distance_m), lat))
            info.sort(key=lambda r: r[2])
            spread = info[-1][2] - info[0][2]
            len(info)
            if spread > 1.0:
                vo_qualifier[info[0][0]] = "on the left"
                vo_qualifier[info[-1][0]] = "on the right"
                middle = sorted(info[1:-1], key=lambda r: r[1])
                if len(middle) == 1:
                    vo_qualifier[middle[0][0]] = "in the middle"
                elif len(middle) >= 2:
                    vo_qualifier[middle[0][0]] = "the closer one in the middle"
                    vo_qualifier[middle[-1][0]] = "the farther one in the middle"
            else:
                info.sort(key=lambda r: r[1])
                vo_qualifier[info[0][0]] = "nearer"
                vo_qualifier[info[-1][0]] = "farther"
                middle = sorted(info[1:-1], key=lambda r: r[2])
                if len(middle) == 1:
                    vo_qualifier[middle[0][0]] = "in between"
                elif len(middle) >= 2:
                    vo_qualifier[middle[0][0]] = "the leftmost one in between"
                    vo_qualifier[middle[-1][0]] = "the rightmost one in between"
        proj_max_m = float(cfg.get("object_navmesh_projection_max_m", 1.0))
        strict_obj_px_pool = int(cfg.get("object_min_visible_pixels_strict", 500))
        soft_obj_px_floor = int(cfg.get("object_pair_bypass_min_pixels", 100))
        _strict_noun_set: set = set()
        if strict_obj_px_pool > 0:
            _noun_strict_count: dict[str, int] = {}
            for _vo in visible_objects:
                if int(getattr(_vo, "pixel_count", 0)) >= strict_obj_px_pool:
                    _noun_strict_count[_vo.mpcat40] = _noun_strict_count.get(_vo.mpcat40, 0) + 1
            _strict_noun_set = {n for (n, c) in _noun_strict_count.items() if c >= 1}
        for j_vo, vo in enumerate(visible_objects):
            if vo.object_id == anchor_obj_id:
                continue
            if strict_obj_px_pool > 0:
                px = int(getattr(vo, "pixel_count", 0))
                if px < strict_obj_px_pool:
                    if vo.mpcat40 not in _strict_noun_set or px < soft_obj_px_floor:
                        continue
            if len(cat_groups.get(vo.mpcat40, [])) >= 2 and j_vo not in vo_qualifier:
                continue
            q = vo_qualifier.get(j_vo, "")
            display_noun = f"{vo.display_name} {q}".strip() if q else vo.display_name
            base_meta = {
                "object_id": vo.object_id,
                "mpcat40": vo.mpcat40,
                "pixel_count": vo.pixel_count,
                "distance_m": vo.distance_m,
                "qualifier": q,
            }
            proj_pt: np.ndarray | None = _project_object_onto_navmesh(
                pathfinder, agent_pos, vo.centroid.astype(np.float64), max_dist_m=proj_max_m
            )
            if proj_pt is None:
                continue
            proj_dist = float(
                np.linalg.norm(proj_pt[[0, 2]] - vo.centroid.astype(np.float32)[[0, 2]])
            )
            if self._family_enabled("toward_object"):
                pool.append(
                    CandidateCommand(
                        family="toward_object",
                        goal_xyz=proj_pt,
                        kind="alternative",
                        object_id=vo.object_id,
                        object_name=display_noun,
                        object_mpcat40=vo.mpcat40,
                        metadata={**base_meta, "navmesh_projection_dist_m": proj_dist},
                    )
                )
            if self._family_enabled("go_past"):
                pool.append(
                    CandidateCommand(
                        family="go_past",
                        goal_xyz=vo.centroid.astype(np.float32).copy(),
                        kind="alternative",
                        object_id=vo.object_id,
                        object_name=display_noun,
                        object_mpcat40=vo.mpcat40,
                        metadata={
                            **base_meta,
                            "navmesh_projection_dist_m": proj_dist,
                            "goal_is_placeholder": True,
                        },
                    )
                )
        if self._family_enabled("forward"):
            fwd_result = _sample_forward_far_point(
                pathfinder,
                agent_pos,
                float(agent_heading),
                r_min_m=float(cfg.get("forward_r_min_m", 1.0)),
                r_max_m=float(cfg.get("forward_r_max_m", 3.0)),
                theta_half_rad=float(cfg.get("forward_theta_half_rad", math.radians(5.0))),
                corridor_step_m=float(cfg.get("forward_corridor_step_m", 0.3)),
                max_attempts=int(cfg.get("forward_max_sample_attempts", 8)),
                rng=rng,
            )
            if fwd_result is not None:
                (fwd_target, fwd_stats) = fwd_result
                pool.append(
                    CandidateCommand(
                        family="forward",
                        goal_xyz=fwd_target,
                        kind="alternative",
                        metadata=dict(fwd_stats),
                    )
                )
        region_families_all = (
            "region_enter",
            "turn_left_enter",
            "turn_right_enter",
            "exit_enter",
            "turn_left_exit_enter",
            "turn_right_exit_enter",
            "turn_left",
            "turn_right",
        )
        any_region_enabled = any((self._family_enabled(f) for f in region_families_all))
        region_pool: list[CandidateCommand] = []
        if semantic_scene is not None and any_region_enabled:
            region_pool = self._sample_region_alternatives(
                semantic_scene=semantic_scene,
                pathfinder=pathfinder,
                agent_position=agent_pos,
                forward_w=forward_w.astype(np.float64),
                left_w=left_w.astype(np.float64),
                current_region=current_region,
                anchor_region=anchor_region,
                visible_regions=visible_regions,
                rng=rng,
            )
        pool.extend(region_pool)
        run_geom_pool = (
            bool(cfg.get("object_enabled", True))
            and semantic_scene is not None
            and (int(cfg.get("max_object_alts", 0)) > 0)
            and (visible_objects is None or len(visible_objects) == 0)
        )
        if run_geom_pool:
            anchor_obj_id_for_dedup = anchor.object_id
            object_pool = self._sample_object_alternatives(
                semantic_scene=semantic_scene,
                pathfinder=pathfinder,
                agent_position=agent_pos,
                forward_w=forward_w.astype(np.float64),
                anchor_obj_id=anchor_obj_id_for_dedup,
                max_alts=int(cfg["max_object_alts"]),
                visible_objects=visible_objects,
            )
            pool.extend(object_pool)
        term_prob = float(cfg.get("terminal_alt_prob", 0.0))
        if (
            term_prob > 0.0
            and (not anchor_is_last)
            and self._family_enabled("terminal")
            and (rng.random() < term_prob)
        ):
            pool.append(
                CandidateCommand(
                    family="terminal",
                    goal_xyz=agent_pos.astype(np.float32),
                    kind="alternative",
                    metadata={"injected_terminal": True},
                )
            )
        if not pool:
            return []
        n_backup = max(0, int(cfg.get("backup_alt_pool_size", n_alt)))
        n_total = n_alt + n_backup
        if len(pool) <= n_total:
            result = list(pool)
            rng.shuffle(result)
            return result
        priority_set = set(cfg.get("prioritize_families") or [])
        priority_cands: list[CandidateCommand] = []
        other_cands: list[CandidateCommand] = []
        if priority_set:
            for c in pool:
                (priority_cands if c.family in priority_set else other_cands).append(c)
        else:
            other_cands = list(pool)
        if len(priority_cands) >= n_alt:
            idx = rng.choice(len(priority_cands), size=n_alt, replace=False)
            primary = [priority_cands[int(i)] for i in idx]
            primary_set_ids = {id(c) for c in primary}
            backup_priority_pool = [c for c in priority_cands if id(c) not in primary_set_ids]
            backup_other_pool = list(other_cands)
        else:
            primary = list(priority_cands)
            remaining = n_alt - len(primary)
            backup_priority_pool = []
            if remaining > 0 and other_cands:
                k = min(remaining, len(other_cands))
                idx = rng.choice(len(other_cands), size=k, replace=False)
                primary_other = [other_cands[int(i)] for i in idx]
                primary.extend(primary_other)
                primary_other_ids = {id(c) for c in primary_other}
                backup_other_pool = [c for c in other_cands if id(c) not in primary_other_ids]
            else:
                backup_other_pool = list(other_cands)
        rng.shuffle(primary)
        backup: list[CandidateCommand] = []
        if backup_priority_pool:
            rng.shuffle(backup_priority_pool)
            backup.extend(backup_priority_pool)
        if len(backup) < n_backup and backup_other_pool:
            rng.shuffle(backup_other_pool)
            backup.extend(backup_other_pool)
        backup = backup[:n_backup]
        return primary + backup

    def _sample_region_alternatives(
        self,
        *,
        semantic_scene: Any,
        pathfinder: Any,
        agent_position: np.ndarray,
        forward_w: np.ndarray,
        left_w: np.ndarray,
        current_region: str | None,
        anchor_region: str | None,
        visible_regions: list[VisibleRegion] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[CandidateCommand]:
        from ..scene_analyzer.regions import (
            _find_current_region,
            _get_region_category,
            _point_in_aabb,
            _region_aabb,
            _sample_navigable_candidates_in_aabb,
        )

        cfg = self.cfg
        n_samples = int(cfg["region_n_sample_pts"])
        max_geo = float(cfg["region_max_geo_m"])
        snap_tol = 0.5
        floor_y = float(agent_position[1])
        stairs_aabbs = _collect_stairs_aabbs(semantic_scene)
        fov_rad = float(cfg.get("region_fov_rad", math.pi / 2.0))
        fov_margin = float(cfg.get("region_fov_margin", 0.95))
        goal_fov_rad = float(cfg.get("region_goal_fov_rad", math.pi))
        path_fov_frac = float(cfg.get("region_path_fov_frac", 0.6))
        half_fov = 0.5 * fov_rad * fov_margin
        half_goal_fov = 0.5 * goal_fov_rad
        vis_gate_enabled = bool(cfg.get("region_require_visible", True))
        visible_region_names: set | None
        if vis_gate_enabled and visible_regions is not None:
            visible_region_names = {
                str(vr.region_name).strip().lower()
                for vr in visible_regions
                if getattr(vr, "region_name", None)
            }
        else:
            visible_region_names = None
        strict_px = int(cfg.get("region_min_visible_pixels_strict", 3000))
        strict_visible_names: set = set()
        if visible_regions is not None:
            for vr in visible_regions:
                nm = getattr(vr, "region_name", None)
                if nm and int(getattr(vr, "pixel_count", 0)) >= strict_px:
                    strict_visible_names.add(str(nm).strip().lower())
        cur_region_lc = current_region.strip().lower() if current_region else None

        def _combined_ok(target_cat: str) -> bool:
            """Are we allowed to emit `*_exit_enter*` for this target?"""
            if cur_region_lc is None:
                return False
            if cur_region_lc not in strict_visible_names:
                return False
            if str(target_cat).strip().lower() not in strict_visible_names:
                return False
            return len(strict_visible_names) >= 2

        out: list[CandidateCommand] = []
        seen_names: set = set()
        import os as _os

        _dbg = _os.environ.get("COFL_SECTOR_REGION_DEBUG", "") == "1"
        _dbg_stats = {
            "regions_seen": 0,
            "no_cat": 0,
            "is_current": 0,
            "anchor_dup": 0,
            "no_aabb": 0,
            "agent_inside": 0,
            "no_goal": 0,
            "geo_fail": 0,
            "fov_reject": 0,
            "emitted": 0,
        }
        for region in getattr(semantic_scene, "regions", None) or []:
            _dbg_stats["regions_seen"] += 1
            cat = _get_region_category(region)
            if cat is None or cat in seen_names:
                _dbg_stats["no_cat"] += 1
                continue
            if cat == current_region:
                _dbg_stats["is_current"] += 1
                continue
            if anchor_region is not None and cat == anchor_region:
                _dbg_stats["anchor_dup"] += 1
                continue
            (aabb_min, aabb_max) = _region_aabb(region)
            if aabb_min is None:
                _dbg_stats["no_aabb"] += 1
                continue
            if _point_in_aabb(np.asarray(agent_position, dtype=np.float32), aabb_min, aabb_max):
                _dbg_stats["agent_inside"] += 1
                continue
            if visible_region_names is not None and cat.lower() not in strict_visible_names:
                _dbg_stats["not_visible"] = _dbg_stats.get("not_visible", 0) + 1
                continue
            cand_pts = _sample_navigable_candidates_in_aabb(
                pathfinder,
                aabb_min,
                aabb_max,
                floor_y,
                n_samples,
                snap_tol,
                stairs_aabbs=stairs_aabbs,
            )
            if not cand_pts:
                _dbg_stats["no_goal"] += 1
                continue
            best_goal: np.ndarray | None = None
            best_geo: float = float("inf")
            best_fov_stats: dict[str, float] = {}
            best_in_fov: bool = False
            best_off_goal: np.ndarray | None = None
            best_off_geo: float = float("inf")
            fov_ok_cands: list[tuple[float, np.ndarray, dict[str, float]]] = []
            for cand in cand_pts:
                g = _geodesic_distance(pathfinder, agent_position, cand)
                if not math.isfinite(g) or g > max_geo:
                    continue
                (ok, fov_stats) = _path_in_fov(
                    pathfinder,
                    agent_position,
                    cand,
                    forward_w,
                    half_fov_rad=half_fov,
                    half_goal_fov_rad=half_goal_fov,
                    min_in_fov_frac=path_fov_frac,
                )
                if ok:
                    fov_ok_cands.append((g, np.asarray(cand), fov_stats))
                elif not best_in_fov and g < best_off_geo:
                    best_off_goal = cand
                    best_off_geo = g
            if not fov_ok_cands:
                if best_off_goal is None:
                    _dbg_stats["geo_fail"] += 1
                else:
                    _dbg_stats["fov_reject"] += 1
                continue
            best_in_fov = True
            cand_arr = np.asarray(cand_pts, dtype=np.float64)
            centroid_xz = np.array(
                [float(cand_arr[:, 0].mean()), float(cand_arr[:, 2].mean())], dtype=np.float64
            )
            sigma_open_m = float(cfg.get("region_openness_search_radius_m", 2.0))
            w_center = float(cfg.get("region_center_dist_weight", 1.0))

            def _openness_score(c: np.ndarray) -> float:
                """Metres to nearest obstacle, clipped to ``sigma_open_m``."""
                try:
                    d_obs = float(
                        pathfinder.distance_to_closest_obstacle(
                            c, max_search_radius=float(sigma_open_m)
                        )
                    )
                except Exception:
                    d_obs = 0.0
                if not math.isfinite(d_obs):
                    d_obs = 0.0
                return min(max(d_obs, 0.0), float(sigma_open_m))

            def _xz_dist_m(c: np.ndarray) -> float:
                return float(np.hypot(c[0] - centroid_xz[0], c[2] - centroid_xz[1]))

            def _combined_score(gp: tuple[float, np.ndarray, dict[str, float]]) -> float:
                """Higher = better."""
                (_, cand, _) = gp
                return _openness_score(cand) - w_center * _xz_dist_m(cand)

            best_pair = max(fov_ok_cands, key=_combined_score)
            (best_geo, best_goal, best_fov_stats) = best_pair
            entry_pt = _region_entry_point(
                pathfinder, agent_position, best_goal, aabb_min, aabb_max
            )
            (los_ok, los_stats) = _has_line_of_sight(
                pathfinder, agent_position, entry_pt, geo_slack_m=0.5, stairs_aabbs=stairs_aabbs
            )
            if not los_ok:
                _dbg_stats["los_fail"] = _dbg_stats.get("los_fail", 0) + 1
                continue
            hit_region = _find_current_region(
                semantic_scene, np.asarray(best_goal, dtype=np.float64)
            )
            if hit_region != cat:
                _dbg_stats["region_membership_fail"] = (
                    _dbg_stats.get("region_membership_fail", 0) + 1
                )
                continue
            goal = best_goal
            geo = best_geo
            fov_stats = best_fov_stats
            goal_arr = np.asarray(goal, dtype=np.float32)
            to_entry = np.asarray(entry_pt, dtype=np.float64) - np.asarray(
                agent_position, dtype=np.float64
            )
            to_entry[1] = 0.0
            en = float(np.linalg.norm(to_entry))
            if en >= 0.001:
                fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
                ent2 = np.array([to_entry[0], to_entry[2]], dtype=np.float64) / en
                cos_a = float(np.clip(np.dot(ent2, fwd2), -1.0, 1.0))
                entry_angle_deg = math.degrees(math.acos(cos_a))
                lateral = float(to_entry[0] * left_w[0] + to_entry[2] * left_w[2]) / en
            else:
                entry_angle_deg = 0.0
                lateral = 0.0
            fwd_angle_deg = float(cfg.get("region_forward_angle_deg", 25.0))
            if entry_angle_deg < fwd_angle_deg:
                side: str | None = None
            else:
                side = "left" if lateral > 0 else "right"
            base_family = "region_enter" if side is None else f"turn_{side}_enter"
            framing_probs = cfg.get("framing_probs") or {}
            forms: list[tuple[str, float]] = []
            if side is not None:
                forms.append(("plain_turn", float(framing_probs.get("plain_turn", 0.0))))
            if current_region is not None:
                forms.append(("combined", float(framing_probs.get("combined", 0.0))))
            forms.append(("plain_enter", float(framing_probs.get("plain_enter", 0.0))))
            flip_rng = rng if rng is not None else np.random.default_rng()
            tot = sum((w for (_, w) in forms))
            if tot <= 0.0:
                framing_tag = "plain_enter"
            else:
                draw = float(flip_rng.uniform(0.0, tot))
                acc = 0.0
                framing_tag = forms[-1][0]
                for name, w in forms:
                    acc += w
                    if draw <= acc:
                        framing_tag = name
                        break
            if framing_tag == "plain_turn":
                family = f"turn_{side}"
                cur_region_used: str | None = None
                region_name_out: str | None = None
            elif framing_tag == "combined":
                family = (
                    "exit_enter"
                    if base_family == "region_enter"
                    else base_family.replace("_enter", "_exit_enter")
                )
                cur_region_used = current_region
                region_name_out = cat
            else:
                family = base_family
                cur_region_used = None
                region_name_out = cat
            if not self._family_enabled(family):
                _dbg_stats["family_disabled"] = _dbg_stats.get("family_disabled", 0) + 1
                continue
            out.append(
                CandidateCommand(
                    family=family,
                    goal_xyz=goal_arr.copy(),
                    kind="alternative",
                    region_name=region_name_out,
                    cur_region_name=cur_region_used,
                    metadata={
                        "geodesic_m": float(geo),
                        "entry_angle_deg": float(entry_angle_deg),
                        "entry_lateral_m": float(lateral),
                        "in_fov_frac": float(fov_stats.get("in_fov_frac", float("nan"))),
                        "framing": framing_tag,
                        "side": side,
                        "target_region": cat,
                    },
                )
            )
            seen_names.add(cat)
            _dbg_stats["emitted"] += 1
        if _dbg:
            print(
                f"[REGION_DBG] cur={current_region} anchor_region={anchor_region} out={len(out)} stats={_dbg_stats}",
                flush=True,
            )
        return out

    def _sample_object_alternatives(
        self,
        *,
        semantic_scene: Any,
        pathfinder: Any,
        agent_position: np.ndarray,
        forward_w: np.ndarray,
        anchor_obj_id: int | None,
        max_alts: int,
        visible_objects: list[VisibleObject] | None = None,
    ) -> list[CandidateCommand]:
        """Sample reachable object goals using semantic visibility and navmesh geometry."""
        from ..semantic_vocab import ALL_ALLOWED_OBJECTS, L1_NAV_OBJECTS, canonical_object_noun

        cfg = self.cfg
        agent_pos = np.asarray(agent_position, dtype=np.float64).reshape(3)
        fwd = np.asarray(forward_w, dtype=np.float64).reshape(3)
        fwd[1] = 0.0
        fwd_n = float(np.linalg.norm(fwd))
        if fwd_n < 1e-06:
            return []
        fwd /= fwd_n
        d_min = float(cfg["object_min_dist_m"])
        d_max = float(cfg["object_max_dist_m"])
        floor = float(cfg["object_floor_band_m"])
        slack = float(cfg["object_los_geo_slack_m"])
        half_fov = 0.5 * float(cfg["object_fov_rad"])
        n_force_nav = int(cfg.get("object_force_nav_anchors", 2))
        stairs_aabbs_obj = _collect_stairs_aabbs(semantic_scene)
        vis_gate_enabled = bool(cfg.get("object_require_visible", True))
        vis_match_m = float(cfg.get("object_visible_match_m", 1.0))
        strict_obj_px = int(cfg.get("object_min_visible_pixels_strict", 1000))
        visible_ids: set = set()
        visible_centroids_by_cat: dict[str, list[np.ndarray]] = {}
        if vis_gate_enabled and visible_objects:
            for vo in visible_objects:
                if strict_obj_px > 0 and int(getattr(vo, "pixel_count", 0)) < strict_obj_px:
                    continue
                try:
                    visible_ids.add(int(vo.object_id))
                except Exception:
                    pass
                cat_v = str(getattr(vo, "mpcat40", "")).strip().lower()
                if cat_v:
                    visible_centroids_by_cat.setdefault(cat_v, []).append(
                        np.asarray(vo.centroid, dtype=np.float64).reshape(3)
                    )
        gate_active = vis_gate_enabled and visible_objects is not None
        objs = getattr(semantic_scene, "objects", None) or []
        seen_keys: set = set()
        cands: list[tuple[float, str, np.ndarray, str, int, bool]] = []
        for obj_idx, o in enumerate(objs):
            cat_obj = getattr(o, "category", None)
            if cat_obj is None:
                continue
            try:
                cat = str(cat_obj.name()).strip().lower()
            except Exception:
                cat = ""
            if cat not in ALL_ALLOWED_OBJECTS:
                continue
            aabb = getattr(o, "aabb", None)
            if aabb is None:
                continue
            try:
                center = np.asarray(aabb.center, dtype=np.float64).reshape(3)
            except Exception:
                continue
            sizes = np.asarray(getattr(aabb, "sizes", [0, 0, 0]), dtype=np.float64).reshape(3)
            ymin = float(center[1] - 0.5 * sizes[1])
            ymax = float(center[1] + 0.5 * sizes[1])
            if not ymin - 0.5 < agent_pos[1] < ymax + floor:
                continue
            delta = center.copy()
            delta[1] = 0.0
            agent_xz = agent_pos.copy()
            agent_xz[1] = 0.0
            d = float(np.linalg.norm(delta - agent_xz))
            if d < d_min or d > d_max:
                continue
            to = (delta - agent_xz) / max(d, 1e-06)
            cos_g = float(np.clip(np.dot(to, fwd), -1.0, 1.0))
            ang = math.acos(cos_g)
            if ang > half_fov:
                continue
            if gate_active:
                id_candidates = {obj_idx}
                raw_id = getattr(o, "id", None)
                if raw_id is not None:
                    try:
                        id_candidates.add(int(raw_id))
                    except Exception:
                        try:
                            id_candidates.add(int(str(raw_id).split("_")[-1]))
                        except Exception:
                            pass
                if not id_candidates & visible_ids:
                    matched = False
                    for vc in visible_centroids_by_cat.get(cat, ()):
                        if float(np.linalg.norm(vc - center)) <= vis_match_m:
                            matched = True
                            break
                    if not matched:
                        continue
            (ok, _stats) = _has_line_of_sight(
                pathfinder, agent_pos, center, geo_slack_m=slack, stairs_aabbs=stairs_aabbs_obj
            )
            if not ok:
                continue
            key = (cat, round(float(center[0]) * 2) / 2, round(float(center[2]) * 2) / 2)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            obj_id_raw = getattr(o, "id", None)
            obj_id_int = obj_idx
            if obj_id_raw is not None:
                try:
                    obj_id_int = int(obj_id_raw)
                except Exception:
                    try:
                        obj_id_int = int(str(obj_id_raw).split("_")[-1])
                    except Exception:
                        obj_id_int = obj_idx
            if obj_id_int == anchor_obj_id:
                continue
            is_nav = cat in L1_NAV_OBJECTS
            cands.append((d, cat, center, canonical_object_noun(cat), obj_id_int, is_nav))
        if not cands:
            return []
        cands.sort(key=lambda r: r[0])
        nav_first = [c for c in cands if c[5]][:n_force_nav]
        rest = [c for c in cands if c not in nav_first]
        ordered = nav_first + rest
        same_noun_groups: dict[str, list[tuple[Any, ...]]] = {}
        for c in ordered:
            same_noun_groups.setdefault(c[3], []).append(c)
        priority_pairs: list[tuple[Any, ...]] = []
        for noun, members in same_noun_groups.items():
            if len(members) < 2:
                continue
            for c in members[:2]:
                if c not in priority_pairs:
                    priority_pairs.append(c)
        if priority_pairs:
            head = list(nav_first)
            for c in priority_pairs:
                if c not in head:
                    head.append(c)
            tail = [c for c in ordered if c not in head]
            ordered = head + tail
        to_emit = ordered[:max_alts]
        groups: dict[str, list[int]] = {}
        for j, (_d, _cat, _c, _n, _oid, _nav) in enumerate(to_emit):
            groups.setdefault(_cat, []).append(j)
        qualifier: dict[int, str] = {}
        for cat, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            info: list[tuple[int, float, float]] = []
            for j in idxs:
                (_d, _, center, _n, _oid, _nav) = to_emit[j]
                delta = center - agent_pos
                lat = -fwd[2] * float(delta[0]) + fwd[0] * float(delta[2])
                info.append((j, _d, lat))
            info.sort(key=lambda r: r[2])
            spread = info[-1][2] - info[0][2]
            n = len(info)
            if spread > 1.0:
                for k, (j, _, _lat) in enumerate(info):
                    if k == 0:
                        qualifier[j] = "on the left"
                    elif k == n - 1:
                        qualifier[j] = "on the right"
                    else:
                        qualifier[j] = "in the middle"
            else:
                info.sort(key=lambda r: r[1])
                for k, (j, _d, _) in enumerate(info):
                    if k == 0:
                        qualifier[j] = "nearer"
                    elif k == n - 1:
                        qualifier[j] = "farther"
                    else:
                        qualifier[j] = "in between"
        proj_max_m = float(cfg.get("object_navmesh_projection_max_m", 2.0))
        out: list[CandidateCommand] = []
        for j, (d, cat, center, noun, obj_id_int, is_nav) in enumerate(to_emit):
            q = qualifier.get(j, "")
            display_noun = f"{noun} {q}".strip() if q else noun
            proj_pt = _project_object_onto_navmesh(
                pathfinder, agent_pos, np.asarray(center, dtype=np.float64), max_dist_m=proj_max_m
            )
            if proj_pt is None:
                continue
            proj_dist = float(
                np.linalg.norm(proj_pt[[0, 2]] - np.asarray(center, dtype=np.float32)[[0, 2]])
            )
            out.append(
                CandidateCommand(
                    family="toward_object",
                    goal_xyz=proj_pt,
                    kind="alternative",
                    object_id=obj_id_int,
                    object_name=display_noun,
                    object_mpcat40=cat,
                    metadata={
                        "source": "mp3d_aabb",
                        "tier": "nav" if is_nav else "landmark",
                        "distance_m": float(d),
                        "mpcat40": cat,
                        "qualifier": q,
                        "navmesh_projection_dist_m": proj_dist,
                    },
                )
            )
        return out
