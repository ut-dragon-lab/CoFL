"""Per-frame processing: visibility, sampler, scene context, slot loop.

For every kept (stride-filtered) frame:
  1. Unpack the oracle trajectory tuple (pos / rot / RGB / depth / sem
     / habitat action).
  2. Find visible objects + regions (semantic-image based gates).
  3. Sample the K-1 alternatives from :class:`AlternativeSampler`.
  4. Precompute the goal-independent frame context once
     (``flow.precompute_frame``).
  5. Iterate ``commands = [anchor] + alts`` through the ``SlotFiller``
     pipeline, advancing a ``next_k`` cursor with backup-pool fallback
     so a transient slot failure (renderer empty / go_past pick fails /
     flow exception) gets replaced by the next backup instead of
     leaving a hole at this slot index.

Returns a per-frame dictionary for the native sink, or ``None`` when no
candidate is valid. Unexpected computation errors propagate to the unit runner.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from ..codecs import JPEG_QUALITY_DEFAULT, encode_rgb_jpeg
from ..flow_field_generator import FlowFieldGenerator
from ..scene_analyzer.regions import _find_current_region
from ..scene_analyzer.types import SceneContext
from ..semantic_anchor.finders import VisibleObjectFinder, VisibleRegionFinder
from ..semantic_anchor.sampler import AlternativeSampler
from ..semantic_anchor.types import CandidateCommand, VisibleObject, VisibleRegion
from .anchor_factory import AnchorContext
from .slot_filler import SlotFiller
from .utils import _TARGET_H, _TARGET_W, _compute_depth_valid_mask, _quat_xyzw_to_heading


def process_frame(
    *,
    t: int,
    trajectory: list[tuple[Any, ...]],
    anchor_ctx: AnchorContext,
    semantic_scene: Any,
    pathfinder: Any,
    finder: VisibleObjectFinder,
    region_finder: VisibleRegionFinder,
    sampler: AlternativeSampler,
    flow: FlowFieldGenerator,
    slot_filler: SlotFiller,
    bev_x_max: float,
    H_bev: int,
    W_bev: int,
    baseline_horizon: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Build per-frame command list + flow fields. Returns a dict of
    arrays ready to be stacked into the per-episode (T, K, ...) tensors,
    or ``None`` if the frame should be skipped entirely.
    """
    pos_t = np.asarray(trajectory[t][0], dtype=np.float64).reshape(3)
    rot_t = np.asarray(trajectory[t][1], dtype=np.float64).reshape(4)
    rgb_t = trajectory[t][2]
    depth_t = trajectory[t][3]
    sem_t = trajectory[t][4]
    action_id_t = int(trajectory[t][5]) if len(trajectory[t]) > 5 else -1
    heading_t = _quat_xyzw_to_heading(rot_t)
    visible_objects: list[VisibleObject] = []
    if sem_t is not None and semantic_scene is not None and (pathfinder is not None):
        visible_objects = finder.find(sem_t, semantic_scene, pos_t, heading_t, pathfinder)
    visible_regions: list[VisibleRegion] = []
    if sem_t is not None and semantic_scene is not None:
        visible_regions = region_finder.find(sem_t, semantic_scene)
    current_region: str | None = None
    if semantic_scene is not None:
        current_region = _find_current_region(semantic_scene, pos_t)
    alternatives = sampler.sample(
        anchor=anchor_ctx.command,
        visible_objects=visible_objects,
        agent_position=pos_t,
        agent_heading=heading_t,
        pathfinder=pathfinder,
        semantic_scene=semantic_scene,
        current_region=current_region,
        rng=rng,
        visible_regions=visible_regions,
    )
    commands: list[CandidateCommand] = [anchor_ctx.command] + alternatives
    rgb_arr = (
        rgb_t.astype(np.uint8)
        if rgb_t is not None
        else np.zeros((_TARGET_H, _TARGET_W, 3), dtype=np.uint8)
    )
    depth_arr = (
        depth_t.astype(np.float32)
        if depth_t is not None
        else np.zeros((_TARGET_H, _TARGET_W), dtype=np.float32)
    )
    depth_valid = _compute_depth_valid_mask(depth_arr, z_max=float(bev_x_max))
    scene_ctx = SceneContext(
        position=np.asarray(pos_t, dtype=np.float32),
        heading=float(heading_t),
        pathfinder=pathfinder,
        semantic_scene=semantic_scene,
    )
    frame_ctx = flow.precompute_frame(
        scene_ctx, depth_image_hw=depth_arr, depth_valid_mask_hw=depth_valid
    )
    K = int(sampler.K)
    H_bl = int(baseline_horizon)
    v_K = np.zeros((K, 2, H_bev, W_bev), dtype=np.float16)
    dp_traj_K = np.zeros((K, H_bl + 1, 2), dtype=np.float32)
    path_len_K = np.zeros((K,), dtype=np.float32)
    slot_valid = np.zeros((K,), dtype=np.uint8)
    texts: list[str] = [""] * K
    cmd_records: list[dict[str, Any] | None] = [None] * K
    family_K: list[str | None] = [None] * K
    rejected_candidates = []
    next_k = 0
    for cmd_idx, cmd in enumerate(commands):
        if next_k >= K:
            break
        k = next_k
        is_anchor_cmd = cmd_idx == 0
        if is_anchor_cmd and k != 0:
            continue
        if not is_anchor_cmd and k == 0:
            next_k = 1
            k = 1
        plan = slot_filler.fill(
            k=k,
            is_anchor=is_anchor_cmd,
            cmd=cmd,
            anchor_text=anchor_ctx.text,
            anchor_subpath=anchor_ctx.subpath,
            is_last_subinstr=anchor_ctx.is_last_subinstr,
            pos_t=pos_t,
            pathfinder=pathfinder,
            frame_ctx=frame_ctx,
            scene_ctx=scene_ctx,
            rng=rng,
        )
        if plan is None or plan.flow_result is None:
            rejected_candidates.append(
                {
                    "candidate_index": cmd_idx,
                    "attempted_slot": k,
                    "reason": "candidate_unfillable",
                    "command": asdict(cmd),
                }
            )
            continue
        ff = plan.flow_result
        v_K[k] = ff.bev_v_field
        dp_traj_K[k] = ff.dp_traj_cart
        path_len_K[k] = float(ff.dp_traj_cart_total_length_m)
        slot_valid[k] = 1
        texts[k] = plan.text
        family_K[k] = plan.family_tag
        eff_goal_xyz = (
            [float(x) for x in plan.effective_goal_xyz.tolist()]
            if plan.effective_goal_xyz is not None
            else [float(x) for x in np.asarray(plan.cmd.goal_xyz).tolist()]
        )
        merged_meta = dict(plan.cmd.metadata or {})
        merged_meta.update(plan.meta_extra)
        cmd_records[k] = {
            "family": plan.cmd.family,
            "kind": plan.cmd.kind,
            "object_id": plan.cmd.object_id,
            "object_name": plan.cmd.object_name,
            "object_mpcat40": plan.cmd.object_mpcat40,
            "region_name": plan.cmd.region_name,
            "cur_region_name": plan.cmd.cur_region_name,
            "goal_xyz": eff_goal_xyz,
            "metadata": merged_meta,
        }
        next_k += 1
    if frame_ctx is not None:
        bev_walkable_t = np.asarray(frame_ctx["bev_walkable"], dtype=np.uint8)
        bev_mask_t = np.asarray(frame_ctx["bev_mask"], dtype=np.uint8)
    else:
        bev_walkable_t = np.zeros((H_bev, W_bev), dtype=np.uint8)
        bev_mask_t = np.zeros((H_bev, W_bev), dtype=np.uint8)
    rgb_blob = encode_rgb_jpeg(rgb_arr, quality=JPEG_QUALITY_DEFAULT)
    depth_fp16 = depth_arr.astype(np.float16)
    return {
        "frame_index": int(t),
        "generation_diagnostics": {
            "rejected_candidates": rejected_candidates,
            "unallocated_slots": [int(k) for k in np.flatnonzero(slot_valid == 0)],
        },
        "rgb_blob": rgb_blob,
        "depth": depth_fp16,
        "position": pos_t.astype(np.float32),
        "heading": float(heading_t),
        "action_id": int(action_id_t),
        "v_K": v_K,
        "dp_traj_K": dp_traj_K,
        "path_len_K": path_len_K,
        "family_K": list(family_K),
        "bev_mask": bev_mask_t,
        "bev_walkable": bev_walkable_t,
        "slot_valid": slot_valid,
        "texts": texts,
        "cmd_records": cmd_records,
        "current_region": current_region,
        "visible_objects": [
            {
                "object_id": vo.object_id,
                "mpcat40": vo.mpcat40,
                "pixel_count": int(vo.pixel_count),
                "distance_m": float(vo.distance_m),
            }
            for vo in visible_objects
        ],
        "visible_regions": [
            {
                "region_name": vr.region_name,
                "pixel_count": int(vr.pixel_count),
                "object_ids": [int(o) for o in vr.object_ids],
            }
            for vr in visible_regions
        ],
    }


__all__ = ["process_frame"]
