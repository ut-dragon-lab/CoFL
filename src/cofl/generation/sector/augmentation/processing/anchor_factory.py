"""Anchor construction from the dataset's sub-instruction stream.

The anchor (``k=0`` slot of every frame) is the dataset's own labelled
sub-instruction: its TEXT, GOAL, and sub-PATH all come straight from the
FG-R2R / Landmark-RxR loader. No internal geometric re-analysis happens
here — this module is just the bridge between the loader and a
``CandidateCommand``.

Public surface
--------------
:func:`build_anchor` — assemble a ``CandidateCommand`` for the current
    agent position, or ``None`` when the frame doesn't map to any sub.
:func:`fetch_anchor_context` — bundle anchor text, sub-path slice, and the
    ``is_last_subinstr`` flag in a single call. Used by ``frame_processor``
    before invoking the slot filler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fgr2r_loader import FGR2RLoader
from ..scene_analyzer.regions import _find_current_region
from ..semantic_anchor.types import CandidateCommand


@dataclass(frozen=True)
class AnchorContext:
    """Everything the slot filler needs about the anchor for one frame."""

    command: CandidateCommand
    text: str
    subpath: np.ndarray | None
    is_last_subinstr: bool


def build_anchor(
    fgr2r_info: Any,
    reference_path: np.ndarray,
    agent_pos_t: np.ndarray,
    semantic_scene: Any,
    *,
    agent_future_positions: np.ndarray,
) -> CandidateCommand | None:
    """Build a per-frame anchor ``CandidateCommand`` from the dataset's
    sub-instruction stream (FG-R2R / Landmark-RxR).

    Returns ``None`` when the frame doesn't map to any sub-instruction,
    or when the goal/subpath lookup fails. Region names are derived from
    the goal endpoint (target room) and the agent's current position
    (from-room) so the sampler can dedup region alts that point at the
    room the anchor already occupies.
    """
    g_sp = FGR2RLoader.goal_and_subpath_for_position(
        fgr2r_info, reference_path, agent_pos_t, agent_future_positions=agent_future_positions
    )
    if g_sp is None:
        return None
    (goal, _subpath) = g_sp
    goal_xyz = np.asarray(goal, dtype=np.float32).reshape(3)
    region_name: str | None = None
    cur_region_name: str | None = None
    if semantic_scene is not None:
        region_name = _find_current_region(semantic_scene, goal_xyz.astype(np.float64))
        cur_region_name = _find_current_region(
            semantic_scene, np.asarray(agent_pos_t, dtype=np.float64)
        )
    return CandidateCommand(
        family=None,
        goal_xyz=goal_xyz,
        kind="anchor",
        region_name=region_name,
        cur_region_name=cur_region_name,
        metadata={},
    )


def fetch_anchor_context(
    fgr2r_info: Any,
    reference_path: np.ndarray,
    agent_pos_t: np.ndarray,
    semantic_scene: Any,
    *,
    agent_future_positions: np.ndarray,
) -> AnchorContext | None:
    """Resolve text, goal, reference suffix and terminal eligibility for the active chunk."""
    cmd = build_anchor(
        fgr2r_info,
        reference_path,
        agent_pos_t,
        semantic_scene,
        agent_future_positions=agent_future_positions,
    )
    if cmd is None:
        return None
    text = FGR2RLoader.subinstr_for_position(fgr2r_info, reference_path, agent_pos_t)
    if not text:
        return None
    g_sp = FGR2RLoader.goal_and_subpath_for_position(
        fgr2r_info, reference_path, agent_pos_t, agent_future_positions=agent_future_positions
    )
    subpath = g_sp[1] if g_sp is not None else None
    is_last = bool(
        FGR2RLoader.is_last_subinstr_for_position(fgr2r_info, reference_path, agent_pos_t)
    )
    cmd.metadata["is_last_subinstr"] = is_last
    return AnchorContext(command=cmd, text=str(text), subpath=subpath, is_last_subinstr=is_last)


__all__ = ["AnchorContext", "build_anchor", "fetch_anchor_context"]
