"""Family → templates mapping and family-set classifications.

Consumers import the family sets to decide which ``{...}`` placeholders
the renderer must fill for a given family:

* ``OBJECT_FAMILIES``           — fills ``{object}``.
* ``REGION_FAMILIES``           — fills ``{region}`` (superset of combined).
* ``COMBINED_REGION_FAMILIES``  — fills BOTH ``{cur_region}`` and ``{region}``.
* ``DIRECTION_FAMILIES``        — parameterless (basic motion + terminal).
* ``ACTIVE_FAMILIES``           — every family the pipeline can actually emit.
                                  Must equal the keys of ``_FAMILY_TEMPLATES``.
                                  Used to validate ``families_enabled`` in the
                                  sampler config.
"""

from __future__ import annotations

from .templates import (
    _EXIT_ENTER,
    _FORWARD,
    _GO_PAST,
    _REGION_ENTER,
    _TERMINAL,
    _TOWARD_OBJECT,
    _TURN_LEFT,
    _TURN_LEFT_ENTER,
    _TURN_LEFT_EXIT_ENTER,
    _TURN_RIGHT,
    _TURN_RIGHT_ENTER,
    _TURN_RIGHT_EXIT_ENTER,
)

# ── Family → template-pool lookup ────────────────────────────────────────────

_FAMILY_TEMPLATES: dict[str, list[str]] = {
    # direction-only (parameterless)
    "forward": _FORWARD,
    "turn_left": _TURN_LEFT,
    "turn_right": _TURN_RIGHT,
    "terminal": _TERMINAL,
    # object-grounded — `toward_object` carries both "go to X" and
    # "stop at X" surface forms (same flow field).
    "toward_object": _TOWARD_OBJECT,
    "go_past": _GO_PAST,
    # region-grounded, single-step enter framing
    "region_enter": _REGION_ENTER,
    "turn_left_enter": _TURN_LEFT_ENTER,
    "turn_right_enter": _TURN_RIGHT_ENTER,
    # combined "exit X into Y" (dual-room)
    "exit_enter": _EXIT_ENTER,
    "turn_left_exit_enter": _TURN_LEFT_EXIT_ENTER,
    "turn_right_exit_enter": _TURN_RIGHT_EXIT_ENTER,
}


# ── Family set classifications (placeholder requirements) ────────────────────

# Families that need BOTH `{cur_region}` (current room) AND `{region}` (target
# room) filled.  See sampler gates: only emitted when both rooms are visible
# with ≥ region_min_visible_pixels_strict and agent ∈ cur_region.
COMBINED_REGION_FAMILIES = frozenset(
    {
        "exit_enter",
        "turn_left_exit_enter",
        "turn_right_exit_enter",
    }
)

# Families that need ``{object}`` filled
OBJECT_FAMILIES = frozenset(
    {
        "toward_object",
        "go_past",
    }
)

# Families that need ``{region}`` filled (superset of COMBINED_REGION_FAMILIES)
REGION_FAMILIES = (
    frozenset(
        {
            "region_enter",
            "turn_left_enter",
            "turn_right_enter",
        }
    )
    | COMBINED_REGION_FAMILIES
)

# Families that take no parameters
DIRECTION_FAMILIES = frozenset(
    {
        "forward",
        "turn_left",
        "turn_right",
        "terminal",
    }
)


# ── Active set: canonical list of families the pipeline can emit ─────────────
# Used to validate ``families_enabled`` in the sampler config so a typo or
# stale entry fails fast at startup instead of silently being ignored.

ACTIVE_FAMILIES = frozenset(_FAMILY_TEMPLATES.keys())

# Self-check: every classification set must be a subset of ACTIVE_FAMILIES.
assert OBJECT_FAMILIES <= ACTIVE_FAMILIES
assert REGION_FAMILIES <= ACTIVE_FAMILIES
assert COMBINED_REGION_FAMILIES <= ACTIVE_FAMILIES
assert DIRECTION_FAMILIES <= ACTIVE_FAMILIES
# And the four sets must partition ACTIVE_FAMILIES (region ⊃ combined, so
# only the disjoint trio counts toward the cover).
assert (OBJECT_FAMILIES | REGION_FAMILIES | DIRECTION_FAMILIES) == ACTIVE_FAMILIES
assert OBJECT_FAMILIES.isdisjoint(REGION_FAMILIES)
assert OBJECT_FAMILIES.isdisjoint(DIRECTION_FAMILIES)
assert REGION_FAMILIES.isdisjoint(DIRECTION_FAMILIES)
