"""Resolve candidate text, goal, arrival radius, field and action family per slot."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..flow_field_generator import FlowFieldGenerator, FlowFieldResult
from ..scene_analyzer.navigation import _geodesic_distance
from ..scene_analyzer.types import SceneContext
from ..semantic_anchor.types import CandidateCommand
from .go_past import pick_go_past_goal_bev
from .rendering import maybe_render_corner_variant, render_alt

_TERMINAL_PATH_LEN_EPS_M: float = 0.05


@dataclass
class SlotPlan:
    """All decisions for one (t, k) slot, populated stage-by-stage."""

    k: int
    is_anchor: bool
    cmd: CandidateCommand
    is_last_subinstr: bool
    text: str = ""
    slot_goal: np.ndarray | None = None
    slot_ref: np.ndarray | None = None
    cmd_target_xyz: np.ndarray | None = None
    stop_snap_eligible: bool = False
    stop_snap_fired: bool = False
    approach_radius_m: float = 0.0
    flow_result: FlowFieldResult | None = None
    family_tag: str | None = None
    effective_goal_xyz: np.ndarray | None = None
    meta_extra: dict[str, Any] = field(default_factory=dict)


class SlotFiller:
    """Per-slot fill pipeline shared by the episode generator.

    Constructor takes the per-builder constants (flow generator, geometric
    radii, sampler config). :meth:`fill` takes per-(t, k) inputs and
    returns a populated :class:`SlotPlan`, or ``None`` if the slot must
    be skipped (renderer empty, go_past pick fails, flow exception).
    """

    def __init__(
        self, *, flow: FlowFieldGenerator, stop_radius_m: float, sampler_cfg: dict[str, Any]
    ) -> None:
        self._flow = flow
        self._stop_radius_m = float(stop_radius_m)
        self._sampler_cfg = sampler_cfg or {}

    def fill(
        self,
        *,
        k: int,
        is_anchor: bool,
        cmd: CandidateCommand,
        anchor_text: str,
        anchor_subpath: np.ndarray | None,
        is_last_subinstr: bool,
        pos_t: np.ndarray,
        pathfinder: Any,
        frame_ctx: dict[str, Any],
        scene_ctx: SceneContext,
        rng: np.random.Generator,
    ) -> SlotPlan | None:
        plan = SlotPlan(
            k=k,
            is_anchor=is_anchor,
            cmd=cmd,
            is_last_subinstr=is_last_subinstr,
            cmd_target_xyz=np.asarray(cmd.goal_xyz, dtype=np.float32).reshape(3),
        )
        if not self._pick_text(plan, anchor_text=anchor_text, rng=rng):
            return None
        if not self._pick_goal(plan, anchor_subpath=anchor_subpath, frame_ctx=frame_ctx, rng=rng):
            return None
        self._apply_stop_snap(plan, pos_t=pos_t, pathfinder=pathfinder)
        self._decide_approach_radius(plan)
        self._generate_flow(plan, scene_ctx=scene_ctx, frame_ctx=frame_ctx)
        self._maybe_rerender_corner(plan, rng=rng)
        self._tag_family(plan)
        self._record_meta(plan)
        return plan

    def _pick_text(self, plan: SlotPlan, *, anchor_text: str, rng: np.random.Generator) -> bool:
        """Render text. Anchor uses verbatim dataset text; alts go through
        :func:`render_alt`. Empty text → drop the slot."""
        if plan.is_anchor:
            plan.text = anchor_text or ""
        else:
            plan.text = render_alt(plan.cmd, rng) or ""
        return bool(plan.text)

    def _pick_goal(
        self,
        plan: SlotPlan,
        *,
        anchor_subpath: np.ndarray | None,
        frame_ctx: dict[str, Any],
        rng: np.random.Generator,
    ) -> bool:
        """Decide ``slot_goal`` + ``slot_ref`` before stop-snap kicks in.

        Anchor (``k=0``): uses the dataset's sub-instruction endpoint and
        forwards ``anchor_subpath`` so sector_gt can walk the reference
        path forward to the farthest depth-visible waypoint.

        Alt (``k≥1``): uses the candidate's own ``cmd.goal_xyz`` with no
        reference path — each alternative must produce a distinct
        flow field, otherwise every K collapses to the same field.

        ``go_past`` (alt only): the sampler emits the object centroid as
        a placeholder; we replace it here with a navmesh-walkable cell
        on the *agent → object → c* corridor. A failed pick (object too
        close / fully occluded / no path-threading cell) drops the slot.
        """
        if plan.is_anchor:
            plan.slot_goal = np.asarray(plan.cmd.goal_xyz, dtype=np.float32).reshape(3)
            plan.slot_ref = anchor_subpath
        else:
            plan.slot_goal = np.asarray(plan.cmd.goal_xyz, dtype=np.float32).reshape(3)
            plan.slot_ref = None
        if not plan.is_anchor and plan.cmd.family == "go_past":
            gp_goal = pick_go_past_goal_bev(
                frame_ctx=frame_ctx,
                obj_centroid_world=plan.cmd.goal_xyz,
                alt_cfg=self._sampler_cfg,
                rng=rng,
            )
            if gp_goal is None:
                return False
            plan.slot_goal = np.asarray(gp_goal, dtype=np.float32).reshape(3)
        return True

    def _apply_stop_snap(self, plan: SlotPlan, *, pos_t: np.ndarray, pathfinder: Any) -> None:
        """Snap ``slot_goal`` to the agent's own position when geodesic
        distance to the goal is within ``stop_radius_m``.

        Eligibility:
          * Anchor (``k=0``): ONLY on the last sub-instruction frame.
            Mid-episode sub-instructions hand off to the next one in the
            next frame, so collapsing the field to r=0 near a sub-path
            endpoint would teach the model to pause at every
            sub-instruction boundary — wrong for continuous navigation.
          * Alternative (``k≥1``): only explicit arrival families.  A
            ``region_enter`` / ``go_past`` / plain-turn target that is
            close to the agent is still a directional task, not a stop
            task; snapping it to the agent erases the geodesic target and
            leaves a degenerate trajectory.

        When stop-snap fires:
          * ``slot_goal`` ← agent_pos
          * ``slot_ref``  ← None (reference walk is meaningless at goal)
          * ``stop_snap_fired`` ← True (invariant gate for family="terminal")
        """
        alt_arrival_family = not plan.is_anchor and plan.cmd.family in ("terminal", "toward_object")
        plan.stop_snap_eligible = alt_arrival_family or (plan.is_anchor and plan.is_last_subinstr)
        plan.stop_snap_fired = False
        if self._stop_radius_m <= 0.0 or not plan.stop_snap_eligible:
            return
        g = np.asarray(plan.slot_goal, dtype=np.float32).reshape(3)
        p = np.asarray(pos_t, dtype=np.float32).reshape(3)
        geo = _geodesic_distance(pathfinder, p, g)
        if not math.isfinite(geo):
            dx = float(g[0] - p[0])
            dz = float(g[2] - p[2])
            geo = math.sqrt(dx * dx + dz * dz)
        if geo <= self._stop_radius_m:
            plan.slot_goal = p.copy()
            plan.slot_ref = None
            plan.stop_snap_fired = True

    def _decide_approach_radius(self, plan: SlotPlan) -> None:
        """Ring-of-approach radius: places the dijkstra source on a ring
        of radius ``stop_radius_m`` around the candidate goal instead of
        at the goal itself, so the field's basin lives on a circle of
        "good enough to stop here" cells.

        Two structurally identical use cases share the same radius —
        the model sees one physical arrival scale:

          (a) ANCHOR on the last sub-instruction — ring around the
              episode endpoint.
          (b) ``toward_object`` ALTERNATIVES — ring around the object's
              navmesh-projected goal (sampler emits a projected point,
              not the raw centroid).

        Critical: ``stop_snap_fired`` takes precedence — once the agent
        is inside the ring (geodesic ≤ radius), sector_gt collapses the
        field to r=0 and enabling ring on top would pick a cell on a
        circle around the agent and pull them away from rest.
        """
        if plan.stop_snap_fired:
            plan.approach_radius_m = 0.0
            return
        if self._stop_radius_m <= 0.0:
            plan.approach_radius_m = 0.0
            return
        is_last_anchor = plan.is_anchor and plan.is_last_subinstr
        is_toward_obj_alt = not plan.is_anchor and plan.cmd.family == "toward_object"
        if is_last_anchor or is_toward_obj_alt:
            plan.approach_radius_m = float(self._stop_radius_m)
        else:
            plan.approach_radius_m = 0.0

    def _generate_flow(self, plan, *, scene_ctx, frame_ctx) -> bool:
        """Generate the goal-dependent field using the required shared frame context."""
        goal_w = np.asarray(plan.slot_goal, dtype=np.float32).reshape(3)
        plan.flow_result = self._flow.generate_from_context(
            frame_ctx,
            scene_ctx,
            goal_w,
            reference_path_world_xyz=plan.slot_ref,
            approach_radius_m=plan.approach_radius_m,
        )

    def _maybe_rerender_corner(self, plan: SlotPlan, *, rng: np.random.Generator) -> None:
        """Corner-template degradation for ``turn_left`` / ``turn_right``
        alts. Anchor text is verbatim from the dataset and never
        re-rendered."""
        if plan.is_anchor:
            return
        ff = plan.flow_result
        if ff is None:
            return
        plan.text = maybe_render_corner_variant(
            cmd=plan.cmd, dp_traj_cart=ff.dp_traj_cart, fallback_text=plan.text, rng=rng
        )

    def _tag_family(self, plan: SlotPlan) -> None:
        """Tag ``family_K[k]``.

        ::

            INVARIANT
            ─────────
            family == "terminal"
              ⟺  is_anchor=True  AND  is_last_subinstr
                  AND  ( stop_snap_fired
                         OR  path_length_m collapsed inside the BEV ring )

            ⟹  action_id == STOP  ⟸  dp_traj_cart ≡ 0

        Two gates fire ``terminal`` on a last-anchor frame:

        * ``stop_snap_fired`` — pathfinder-side test (geodesic ≤
          ``stop_radius_m``); ``slot_goal`` was snapped to the agent foot.
        * BEV-side fallback — ``approach_radius_m > 0`` (a ring was active)
          AND the actual dijkstra polyline collapsed to ~zero length
          AND ``walk_term_reason`` reports a geometric-reach outcome
          (``goal_reached`` / ``stop_D`` / ``stop_snap_at_agent``). This
          happens at a sub-cell boundary where pathfinder geodesic
          reports just above ``stop_radius_m`` while the grid-discretised
          ring on BEV already covers the agent's cell. Without this
          fallback, the BEV-zero anchor would land as ``ACTION_IGNORE``
          (degenerate cart) on the very frame the agent enters the ring,
          one tick before the geodesic check catches up. The
          ``walk_term_reason`` gate is critical: ``sector_clip`` and
          ``unreachable`` ALSO yield ``path_len ≈ 0`` but only because
          the walk gave up (goal behind agent / oracle still rotating /
          no walkable bridge) — those frames must NOT be tagged terminal.

        The geometric gate is still load-bearing for *non-terminal*
        frames: an agent walking the last sub-instruction's segment can
        be 1–5 m from the goal vp while still on that segment, producing
        a long-forward trajectory — ``family`` stays at the dataset
        anchor's empty value so the slot is bucketed as FWD/TURN.
        """
        if not (plan.is_anchor and plan.is_last_subinstr):
            plan.family_tag = plan.cmd.family
            return
        if plan.stop_snap_fired:
            plan.family_tag = "terminal"
            return
        ff = plan.flow_result
        ring_active = float(plan.approach_radius_m) > 0.0
        path_len = float(ff.dp_traj_cart_total_length_m) if ff is not None else float("inf")
        walk_term = str((ff.raw or {}).get("walk_term_reason", "")) if ff is not None else ""
        local_goal_world = (ff.raw or {}).get("local_goal_world") if ff is not None else None
        if local_goal_world is not None and plan.slot_goal is not None:
            lg = np.asarray(local_goal_world, dtype=np.float64).reshape(3)
            sg = np.asarray(plan.slot_goal, dtype=np.float64).reshape(3)
            d_src_to_goal = float(np.linalg.norm(lg[[0, 2]] - sg[[0, 2]]))
            source_is_goal = d_src_to_goal < 0.5 * float(plan.approach_radius_m)
        else:
            source_is_goal = False
        agent_at_goal = source_is_goal and walk_term in (
            "goal_reached",
            "stop_D",
            "stop_snap_at_agent",
        )
        if ring_active and path_len < _TERMINAL_PATH_LEN_EPS_M and agent_at_goal:
            plan.family_tag = "terminal"
            return
        plan.family_tag = plan.cmd.family

    def _record_meta(self, plan: SlotPlan) -> None:
        """Compute ``effective_goal_xyz`` (priority: cmd if snapped, else local > cmd)
        and the ``meta_extra`` dict that lands on samples.jsonl."""
        ff = plan.flow_result
        raw = ff.raw if ff is not None and ff.raw else {}
        picked_world = raw.get("picked_goal_world")
        local_world = raw.get("local_goal_world")
        if plan.stop_snap_fired:
            eff = np.asarray(plan.cmd.goal_xyz).reshape(3)
        elif local_world is not None:
            eff = np.asarray(local_world).reshape(3)
        else:
            eff = np.asarray(plan.cmd.goal_xyz).reshape(3)
        plan.effective_goal_xyz = eff
        meta_extra: dict[str, Any] = {
            "cmd_target_xyz": [float(x) for x in np.asarray(plan.cmd.goal_xyz).tolist()],
            "slot_goal_xyz": [float(x) for x in np.asarray(plan.slot_goal).reshape(3).tolist()],
            "goal_source": str(raw.get("goal_source", "")),
            "label_family": plan.family_tag,
            "stop_snap_eligible": bool(plan.stop_snap_eligible),
            "stop_snap_fired": bool(plan.stop_snap_fired),
        }
        if local_world is not None:
            meta_extra["local_goal_world"] = [
                float(x) for x in np.asarray(local_world).reshape(3).tolist()
            ]
        if picked_world is not None:
            meta_extra["picked_goal_world"] = [
                float(x) for x in np.asarray(picked_world).reshape(3).tolist()
            ]
        if "walk_term_reason" in raw:
            meta_extra["walk_term_reason"] = str(raw.get("walk_term_reason", ""))
        if plan.approach_radius_m > 0.0:
            meta_extra["approach_radius_m"] = float(plan.approach_radius_m)
            if picked_world is not None:
                meta_extra["candidate_centroid_xyz"] = [
                    float(x) for x in np.asarray(plan.cmd.goal_xyz).tolist()
                ]
        plan.meta_extra = meta_extra


__all__ = ["SlotFiller", "SlotPlan"]
