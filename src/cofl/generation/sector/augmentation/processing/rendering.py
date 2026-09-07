"""Surface-text rendering for alternative slots.

Anchors (``k=0``) are never rendered here — their text comes verbatim
from the dataset's sub-instruction stream (FG-R2R / Landmark-RxR). Only
``k≥1`` alternatives reach this module, and each carries a non-``None``
``cmd.family``.

The corner-template degradation is also here: once the dijkstra
trajectory is in hand we can tell whether a ``turn_left`` / ``turn_right``
slot is geometrically a sharp corner (vs a smooth arc). If yes, we
re-render with ``corner_eligible=True`` so the pool may swap to the
direction-neutral ``_TURN_CORNER`` templates and force the model to
ground the side from visible geometry instead of surface text.
"""

from __future__ import annotations

import numpy as np

from ..instruction_renderer import (
    COMBINED_REGION_FAMILIES,
    OBJECT_FAMILIES,
    REGION_FAMILIES,
    is_corner_trajectory,
)
from ..instruction_renderer import render as render_instruction
from ..semantic_anchor.types import CandidateCommand


def render_alt(cmd: CandidateCommand, rng: np.random.Generator) -> str:
    """Render an alternative slot's surface text from its ``CandidateCommand``.

    Returns ``""`` when the command can't be rendered for any reason
    (missing object / region name, unknown family). The caller treats
    empty strings as "drop this slot".
    """
    fam = cmd.family
    if not fam:
        return ""
    if fam in OBJECT_FAMILIES:
        if not cmd.object_name:
            return ""
        return render_instruction(fam, object_name=cmd.object_name, rng=rng)
    if fam in COMBINED_REGION_FAMILIES:
        if not cmd.region_name or not cmd.cur_region_name:
            return ""
        return render_instruction(
            fam, region_name=cmd.region_name, cur_region_name=cmd.cur_region_name, rng=rng
        )
    if fam in REGION_FAMILIES:
        if not cmd.region_name:
            return ""
        return render_instruction(fam, region_name=cmd.region_name, rng=rng)
    return render_instruction(fam, rng=rng)


def maybe_render_corner_variant(
    *, cmd: CandidateCommand, dp_traj_cart: np.ndarray, fallback_text: str, rng: np.random.Generator
) -> str:
    """If the slot is a corner-shaped ``turn_left``/``turn_right`` alt,
    re-render with ``corner_eligible=True`` to potentially pull from the
    direction-neutral ``_TURN_CORNER`` pool. Returns ``fallback_text``
    when the slot isn't eligible or the re-render fails.

    Anchors are never re-rendered (caller is responsible for skipping).
    """
    if cmd.family not in ("turn_left", "turn_right"):
        return fallback_text
    if not is_corner_trajectory(dp_traj_cart):
        return fallback_text
    try:
        new_text = render_instruction(cmd.family, corner_eligible=True, rng=rng)
    except Exception:
        return fallback_text
    return new_text if new_text else fallback_text


__all__ = ["maybe_render_corner_variant", "render_alt"]
