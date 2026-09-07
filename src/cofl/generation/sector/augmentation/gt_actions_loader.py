"""Ground-truth discrete-action sequence loader.

VLN-CE ships per-split ``*_gt.json.gz`` files containing the dataset authors'
canonical replay sequence — a list of habitat action IDs that, when stepped
in the simulator, produce the human-described path used for sub-instruction
annotation. Replaying these directly is preferable to running an
oracle/follower at augmentation time:

* The trajectory matches ``reference_path`` exactly (no shortest-path
  shortcuts that would misalign sub-instructions to off-route frames).
* No dependence on ``SHORTEST_PATH_SENSOR`` / ``GreedyGeodesicFollower``,
  so RxR-CE episodes don't break with T=1 when those fall through.
* Determinism: same input → same trajectory.

File layouts (both keyed by episode_id string):

* R2R-CE  → ``R2R_VLNCE_v1-3_preprocessed/<split>/<split>_gt.json.gz``
            actions ∈ [0,3]: STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
* RxR-CE  → ``RxR_VLNCE_v0/<split>/<split>_guide_gt.json.gz``
            actions ∈ [0,5]: …+ LOOK_UP / LOOK_DOWN

Record schema:
    {episode_id: {"actions": [int, ...], "locations": [[x,y,z], ...],
                  "forward_steps": int}}
"""

from __future__ import annotations

import gzip
import json


class GtActionsLoader:
    """Per-split lookup: ``episode_id -> List[int] (habitat action IDs)``."""

    def __init__(self, json_path: str) -> None:
        self.json_path = str(json_path)
        if self.json_path.endswith(".gz"):
            with gzip.open(self.json_path, "rt", encoding="utf-8") as f:
                self._raw: dict[str, dict] = json.load(f)
        else:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._raw = json.load(f)
        if not isinstance(self._raw, dict):
            raise ValueError(
                f"GT file must be a top-level dict keyed by episode_id, got {type(self._raw).__name__}: {self.json_path}"
            )

    def __len__(self) -> int:
        return len(self._raw)

    def lookup(self, episode_id) -> list[int] | None:
        rec = self._raw.get(str(episode_id))
        if not isinstance(rec, dict):
            return None
        actions = rec.get("actions")
        if not isinstance(actions, list):
            return None
        try:
            return [int(a) for a in actions]
        except Exception:
            return None


__all__ = ["GtActionsLoader"]
