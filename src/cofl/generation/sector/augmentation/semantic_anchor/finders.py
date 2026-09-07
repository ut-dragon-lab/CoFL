"""Per-frame finders that consume the semantic image + SemanticScene.

Two public classes:

:class:`VisibleObjectFinder`
    Walks the semantic image, keeps object_ids with enough pixel coverage,
    then filters by mpcat40 whitelist, distance band, and geodesic
    reachability.  Output feeds the per-object candidate pool in the
    sampler and the QA UI's "visible objects" list.

:class:`VisibleRegionFinder`
    Room-level mirror of the object finder.  Attributes visible pixels to
    their parent region (via ``SemanticObject.region`` when present,
    otherwise by AABB-XZ containment of the object centroid) and emits one
    :class:`VisibleRegion` per room whose aggregated pixel count clears
    ``region_min_visible_pixels``.  Used by the region visibility gate in
    the sampler.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..scene_analyzer.navigation import _geodesic_distance, heading_to_dirs
from ..scene_analyzer.objects import _obj_centroid
from .types import VisibleObject, VisibleRegion, _display_name, _load_finder_cfg, _mpcat40_of


class VisibleObjectFinder:
    """Identify objects truly visible in the camera frame via the SEMANTIC sensor.

    The semantic sensor returns an ``(H, W)`` image of object_id values.
    For each object_id present with pixel_count ≥ ``min_visible_pixels``,
    look up the SemanticObject in ``semantic_scene.objects`` and apply the
    mpcat40 whitelist / blacklist + distance / reachability filters.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = _load_finder_cfg(cfg)
        self._whitelist = set((str(w).lower() for w in self.cfg["whitelist"]))
        self._blacklist = set((str(w).lower() for w in self.cfg["blacklist"]))
        self._min_px = int(self.cfg["min_visible_pixels"])
        self._min_d = float(self.cfg["min_dist_m"])
        self._max_d = float(self.cfg["max_dist_m"])
        self._geo_ratio = float(self.cfg["max_geo_euc_ratio"])

    def find(
        self,
        semantic_image_hw: np.ndarray,
        semantic_scene: Any,
        agent_position: np.ndarray,
        agent_heading: float,
        pathfinder: Any,
    ) -> list[VisibleObject]:
        if semantic_image_hw is None or semantic_scene is None:
            return []
        sem = np.asarray(semantic_image_hw)
        if sem.ndim == 3:
            sem = sem[..., 0]
        try:
            (ids, counts) = np.unique(sem, return_counts=True)
        except Exception:
            return []
        id_to_obj: dict[int, Any] = {}
        for idx, o in enumerate(getattr(semantic_scene, "objects", None) or []):
            id_to_obj[idx] = o
            raw_id = getattr(o, "id", None)
            if raw_id is not None:
                try:
                    id_to_obj[int(raw_id)] = o
                except Exception:
                    try:
                        tail = str(raw_id).split("_")[-1]
                        id_to_obj[int(tail)] = o
                    except Exception:
                        pass
            sid = getattr(o, "semantic_id", None)
            if sid is not None:
                try:
                    id_to_obj[int(sid)] = o
                except Exception:
                    pass
        agent_pos = np.asarray(agent_position, dtype=np.float64).reshape(3)
        (forward_w, left_w) = heading_to_dirs(agent_heading)
        fwd2 = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
        lft2 = np.array([left_w[0], left_w[2]], dtype=np.float64)
        out: list[VisibleObject] = []
        seen_ids: set = set()
        for raw_id, count in zip(ids.tolist(), counts.tolist()):
            try:
                pid = int(raw_id)
            except Exception:
                continue
            if pid <= 0 or count < self._min_px or pid in seen_ids:
                continue
            seen_ids.add(pid)
            obj = id_to_obj.get(pid)
            if obj is None:
                continue
            mpcat = _mpcat40_of(obj)
            if mpcat is None or mpcat in self._blacklist:
                continue
            if mpcat not in self._whitelist:
                continue
            centroid = _obj_centroid(obj)
            if centroid is None:
                continue
            euc = float(np.linalg.norm(centroid - agent_pos))
            if not self._min_d <= euc <= self._max_d:
                continue
            geo = _geodesic_distance(pathfinder, agent_pos, centroid)
            if not math.isfinite(geo):
                continue
            if euc > 0.0001 and geo / euc > self._geo_ratio:
                continue
            to_obj = (centroid - agent_pos).astype(np.float64)
            to_obj[1] = 0.0
            n = float(np.linalg.norm(to_obj))
            if n < 1e-06:
                angle_signed = 0.0
            else:
                u = to_obj / n
                obj2 = np.array([u[0], u[2]], dtype=np.float64)
                cos_a = float(np.clip(np.dot(fwd2, obj2), -1.0, 1.0))
                ang = math.acos(cos_a)
                sign = float(np.dot(obj2, lft2))
                angle_signed = math.copysign(ang, sign)
            out.append(
                VisibleObject(
                    object_id=int(pid),
                    mpcat40=mpcat,
                    display_name=_display_name(mpcat),
                    centroid=centroid.astype(np.float32),
                    pixel_count=int(count),
                    distance_m=euc,
                    geodesic_m=float(geo),
                    angle_rad=float(angle_signed),
                )
            )
        if not out:
            return out
        merged: dict[tuple[str, int, int], VisibleObject] = {}
        merged_px: dict[tuple[str, int, int], int] = {}
        for vo in out:
            key = (
                vo.mpcat40,
                int(round(float(vo.centroid[0]) * 2.0)),
                int(round(float(vo.centroid[2]) * 2.0)),
            )
            best = merged.get(key)
            if best is None or vo.pixel_count > best.pixel_count:
                merged[key] = vo
            merged_px[key] = merged_px.get(key, 0) + vo.pixel_count
        out_dedup: list[VisibleObject] = []
        for key, vo in merged.items():
            if merged_px[key] != vo.pixel_count:
                vo = VisibleObject(
                    object_id=vo.object_id,
                    mpcat40=vo.mpcat40,
                    display_name=vo.display_name,
                    centroid=vo.centroid,
                    pixel_count=merged_px[key],
                    distance_m=vo.distance_m,
                    geodesic_m=vo.geodesic_m,
                    angle_rad=vo.angle_rad,
                )
            out_dedup.append(vo)
        return out_dedup


class VisibleRegionFinder:
    """Identify *regions* visible in the camera frame.

    Mirror of :class:`VisibleObjectFinder` but at the room level.  Walks
    the semantic image once, attributes every visible object_id to its
    parent region (via ``SemanticObject.region`` if available, else by
    AABB-XZ containment of the object's centroid), and emits one
    :class:`VisibleRegion` per region whose total pixel count clears
    ``region_min_visible_pixels``.  This is consumed by
    :class:`AlternativeSampler` to gate ``*_enter`` / ``enter_then_*``
    candidates so they only fire for rooms that are actually in view.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = _load_finder_cfg(cfg)
        self._min_px = int(self.cfg.get("region_min_visible_pixels", 200))
        self._aabb_tol = float(self.cfg.get("region_visible_aabb_tol_m", 0.5))
        _default_arch = {"wall", "door", "window", "misc", "void", "unknown", "remove"}
        _user_bl = self.cfg.get("region_visible_pixel_blacklist", None)
        if _user_bl is None:
            self._region_pixel_blacklist = _default_arch
        else:
            self._region_pixel_blacklist = set((str(w).strip().lower() for w in _user_bl))

    def find(self, semantic_image_hw: np.ndarray, semantic_scene: Any) -> list[VisibleRegion]:
        if semantic_image_hw is None or semantic_scene is None:
            return []
        sem = np.asarray(semantic_image_hw)
        if sem.ndim == 3:
            sem = sem[..., 0]
        try:
            (ids, counts) = np.unique(sem, return_counts=True)
        except Exception:
            return []
        objects = list(getattr(semantic_scene, "objects", None) or [])
        id_to_obj: dict[int, Any] = {}
        for idx, o in enumerate(objects):
            id_to_obj[idx] = o
            raw_id = getattr(o, "id", None)
            if raw_id is not None:
                try:
                    id_to_obj[int(raw_id)] = o
                except Exception:
                    try:
                        tail = str(raw_id).split("_")[-1]
                        id_to_obj[int(tail)] = o
                    except Exception:
                        pass
            sid = getattr(o, "semantic_id", None)
            if sid is not None:
                try:
                    id_to_obj[int(sid)] = o
                except Exception:
                    pass
        from ..scene_analyzer.regions import _get_region_category, _region_aabb

        region_aabbs: list[tuple[str, np.ndarray, np.ndarray]] = []
        regions = list(getattr(semantic_scene, "regions", None) or [])
        for r in regions:
            cat = _get_region_category(r)
            if not cat:
                continue
            (mn, mx) = _region_aabb(r)
            if mn is None or mx is None:
                continue
            region_aabbs.append((cat, mn, mx))
        per_region_px: dict[str, int] = {}
        per_region_ids: dict[str, list[int]] = {}
        for raw_id, count in zip(ids.tolist(), counts.tolist()):
            try:
                pid = int(raw_id)
            except Exception:
                continue
            if pid <= 0:
                continue
            obj = id_to_obj.get(pid)
            if obj is None:
                continue
            mpcat = _mpcat40_of(obj)
            if mpcat is not None and str(mpcat).strip().lower() in self._region_pixel_blacklist:
                continue
            cat: str | None = None
            r_attr = getattr(obj, "region", None)
            if r_attr is not None:
                try:
                    cat = _get_region_category(r_attr)
                except Exception:
                    cat = None
            if cat is None:
                centroid = _obj_centroid(obj)
                if centroid is not None and region_aabbs:
                    p = np.asarray(centroid, dtype=np.float32)
                    tol = self._aabb_tol
                    for r_cat, mn, mx in region_aabbs:
                        if (
                            mn[0] - tol <= p[0] <= mx[0] + tol
                            and mn[1] <= p[1] <= mx[1]
                            and (mn[2] - tol <= p[2] <= mx[2] + tol)
                        ):
                            cat = r_cat
                            break
            if not cat:
                continue
            per_region_px[cat] = per_region_px.get(cat, 0) + int(count)
            per_region_ids.setdefault(cat, []).append(pid)
        out: list[VisibleRegion] = []
        for cat, px in per_region_px.items():
            if px < self._min_px:
                continue
            out.append(
                VisibleRegion(
                    region_name=cat,
                    pixel_count=int(px),
                    object_ids=list(per_region_ids.get(cat, [])),
                )
            )
        out.sort(key=lambda r: -r.pixel_count)
        return out
