"""Natural-language template pools for the instruction renderer.

Pure data — one ``List[str]`` per family, grouped here for ease of review.
The aggregation into ``_FAMILY_TEMPLATES`` and the family set constants
live in :mod:`.families`.

Only families actually emitted by the pipeline are kept here.  See
``families.ACTIVE_FAMILIES`` for the canonical live set.
"""

from __future__ import annotations

# ── Basic motion (no anchor) ──────────────────────────────────────────────────

_FORWARD: list[str] = [
    "walk forward",
    "go forward",
    "move ahead",
    "keep going straight",
    "continue forward",
    "proceed straight ahead",
    "walk straight",
]

_TURN_LEFT: list[str] = [
    "turn left",
]

_TURN_RIGHT: list[str] = [
    "turn right",
]

# Corner variant — direction-neutral. Both ``turn_left`` and ``turn_right``
# can degrade to this pool (with low probability) when the trajectory passes
# ``is_corner_trajectory``. Intentionally NOT carrying left/right so the
# model learns to read the actual turn direction off the visible scene
# geometry rather than the surface text. Trains the "turn at the corner"
# phrasing that benchmarks use without committing the language to a side.
_TURN_CORNER: list[str] = [
    "turn at the corner",
    "turn at the next corner",
    "take the corner",
    "go around the corner",
    "round the corner",
    "make the turn at the corner",
]

_TERMINAL: list[str] = [
    "stop here",
]


# ── Object-anchored ───────────────────────────────────────────────────────────

# `toward_object` covers BOTH "go to X" and "stop at X" surface forms:
# at training time the underlying flow field is identical (ring of radius
# `stop_radius_m` around the navmesh-projected object position, plus stop
# snap when the agent is inside the ring), so the policy learns one
# behaviour from two equivalent text patterns.
_TOWARD_OBJECT: list[str] = [
    # "go toward / approach" wording
    "walk toward the {object}",
    "go to the {object}",
    "move to the {object}",
    "head to the {object}",
    "approach the {object}",
    "go toward the {object}",
    "head toward the {object}",
    # "stop at" wording — same flow, different surface form
    "stop at the {object}",
    "stop near the {object}",
    "stop when you reach the {object}",
    "halt at the {object}",
    "come to a stop near the {object}",
]

_GO_PAST: list[str] = [
    "go past the {object}",
    "walk past the {object}",
    "continue past the {object}",
    "pass the {object}",
    "walk by the {object}",
]


# ── Region enter (single-step) ────────────────────────────────────────────────

_REGION_ENTER: list[str] = [
    "enter the {region}",
    "go into the {region}",
    "walk into the {region}",
    "step into the {region}",
    "head into the {region}",
    # "exit into" framings — describe the action as leaving the current
    # space to enter the {region}, without naming the source room. Useful
    # when the agent is between rooms / in a hallway and needs to commit
    # to the destination. Same flow field as the plain enter templates.
    "exit into the {region}",
    "step out into the {region}",
    "head out into the {region}",
    "go out into the {region}",
]


# ── Compound: turn + enter region ────────────────────────────────────────────

_TURN_LEFT_ENTER: list[str] = [
    "turn left to enter the {region}",
    "turn left and go into the {region}",
    "turn left into the {region}",
    "head left and step into the {region}",
    "bear left to enter the {region}",
    "veer left into the {region}",
    "go left and walk into the {region}",
]

_TURN_RIGHT_ENTER: list[str] = [
    "turn right to enter the {region}",
    "turn right and go into the {region}",
    "turn right into the {region}",
    "head right and step into the {region}",
    "bear right to enter the {region}",
    "veer right into the {region}",
    "go right and walk into the {region}",
]


# ── Combined exit_enter (dual-room: cur_region + region) ──────────────────────
# Templates carry BOTH `{cur_region}` and `{region}` slots.  Only emitted
# when (a) cur_region and target_region are both visible with ≥
# `region_min_visible_pixels_strict`, AND (b) agent ∈ cur_region (implicit
# via `_find_current_region`), AND (c) target goal ∈ target_region (implicit
# by sampling from its AABB).

_EXIT_ENTER: list[str] = [
    "leave the {cur_region} and enter the {region}",
    "step out of the {cur_region} into the {region}",
    "head out of the {cur_region} and into the {region}",
    "go from the {cur_region} into the {region}",
    "exit the {cur_region} into the {region}",
    "walk out of the {cur_region} and into the {region}",
    # Simplified "exit current region" framings — the destination room is
    # left implicit, but the underlying flow field still terminates in
    # ``{region}``, so the model learns to ground the exit action
    # against the visible doorway / opening. Useful when the source
    # region is the salient cue (e.g. "exit the bedroom" — where to is
    # obvious from context).
    "exit the {cur_region}",
    "leave the {cur_region}",
    "step out of the {cur_region}",
    "walk out of the {cur_region}",
    "head out of the {cur_region}",
]

_TURN_LEFT_EXIT_ENTER: list[str] = [
    "turn left out of the {cur_region} and into the {region}",
    "turn left, leave the {cur_region} and enter the {region}",
    "bear left out of the {cur_region} into the {region}",
    "head left out of the {cur_region} into the {region}",
]

_TURN_RIGHT_EXIT_ENTER: list[str] = [
    "turn right out of the {cur_region} and into the {region}",
    "turn right, leave the {cur_region} and enter the {region}",
    "bear right out of the {cur_region} into the {region}",
    "head right out of the {cur_region} into the {region}",
]
