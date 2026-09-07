"""Load Landmark-RxR instruction IDs and map subpaths to inclusive viewpoint ranges."""

from __future__ import annotations

import json

from .fgr2r_loader import _MatchedInstr


class LandmarkRxRLoader:
    """Per-split Landmark-RxR lookup: ``instruction_id -> sub-instructions``."""

    def __init__(self, json_path: str) -> None:
        self.json_path = str(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self._raw: list[dict] = json.load(f)
        self._by_iid: dict[int, _MatchedInstr] = {}
        for rec in self._raw:
            iid = rec.get("instruction_id")
            if iid is None:
                continue
            try:
                iid = int(iid)
            except Exception:
                continue
            matched = self._build(rec)
            if matched is not None:
                self._by_iid[iid] = matched

    def __len__(self) -> int:
        return len(self._by_iid)

    def lookup(self, instruction_id: int) -> _MatchedInstr | None:
        """Return subinstructions by the RxR instruction ID."""
        try:
            iid = int(instruction_id)
        except Exception:
            return None
        return self._by_iid.get(iid)

    @staticmethod
    def _build(rec: dict) -> _MatchedInstr | None:
        """Build ``_MatchedInstr`` from one Landmark-RxR record.

        Converts ``sub_paths`` (lists of viewpoint hashes that tile ``path``
        with shared endpoints) into the same 1-based inclusive vp_ranges that
        FG-R2R's ``chunk_view`` provides, so the downstream geometric matcher
        works without modification.
        """
        path = rec.get("path") or []
        sub_paths = rec.get("sub_paths") or []
        sub_texts = rec.get("sub_instructions") or []
        if not path or not sub_paths or (not sub_texts):
            return None
        idx_of: dict[str, int] = {}
        for i, h in enumerate(path):
            if h not in idx_of:
                idx_of[h] = i
        S = min(len(sub_paths), len(sub_texts))
        out_texts: list[str] = []
        out_ranges: list[tuple[int, int]] = []
        for s in range(S):
            sp = sub_paths[s]
            text = sub_texts[s]
            if not sp or not text:
                continue
            try:
                a = idx_of[sp[0]] + 1
                b = idx_of[sp[-1]] + 1
            except KeyError:
                continue
            out_texts.append(str(text))
            out_ranges.append((a, b))
        if not out_texts:
            return None
        return _MatchedInstr(sub_texts=out_texts, vp_ranges=out_ranges)


__all__ = ["LandmarkRxRLoader"]
