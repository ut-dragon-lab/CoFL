"""FG-R2R subinstructions and segment-based reference-path alignment."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class _MatchedInstr:
    """One VLN-CE episode's resolved FG-R2R alignment."""

    sub_texts: list[str]
    vp_ranges: list[tuple[int, int]]


def _parse_field(x):
    """``new_instructions``/``chunk_view`` may be a literal Python repr-string
    or already a list (the public file ships them as strings; some downstream
    re-saves convert to lists). Handle both."""
    if x is None or isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            try:
                return json.loads(x)
            except Exception:
                return None
    return None


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().split()).lower().rstrip(".")


def _locate_agent_on_path(ref: np.ndarray, agent: np.ndarray) -> tuple[int, float]:
    """Return the closest XZ path segment index and its clamped progress."""
    P = int(ref.shape[0])
    if P == 0:
        return (0, 0.0)
    if P == 1:
        return (0, 0.0)
    a_xz = np.stack([ref[:-1, 0], ref[:-1, 2]], axis=1)
    b_xz = np.stack([ref[1:, 0], ref[1:, 2]], axis=1)
    seg = b_xz - a_xz
    seg_len2 = np.sum(seg * seg, axis=1)
    safe = np.where(seg_len2 > 1e-10, seg_len2, 1.0)
    p_xz = np.array([agent[0], agent[2]], dtype=np.float64)
    rel = p_xz[None, :] - a_xz
    t_un = np.sum(rel * seg, axis=1) / safe
    t_cl = np.clip(t_un, 0.0, 1.0)
    proj = a_xz + t_cl[:, None] * seg
    d2_seg = np.sum((p_xz[None, :] - proj) ** 2, axis=1)
    d2_seg = np.where(seg_len2 > 1e-10, d2_seg, np.inf)
    seg_i = int(np.argmin(d2_seg))
    t = float(t_cl[seg_i])
    return (seg_i, t)


def _match_subinstr_for_segment(
    vp_ranges: list[tuple[int, int]],
    seg_i: int,
    t: float,
    P: int,
    *,
    agent_xyz: np.ndarray | None,
    reference_path_xyz: np.ndarray | None,
    stop_radius_m: float = 1.0,
) -> int | None:
    """Match a segment to inclusive viewpoint ranges. Degenerate point events
    within the arrival radius take priority; gaps use segment progress."""
    if not vp_ranges:
        return None
    if P <= 1:
        return 0
    a_seg = seg_i + 1
    b_seg = seg_i + 2
    ref_arr = np.asarray(reference_path_xyz, dtype=np.float64).reshape(-1, 3)
    ag_xz_ = np.asarray(agent_xyz, dtype=np.float64).reshape(3)[[0, 2]]
    best_degenerate: int | None = None
    for i, (a, b) in enumerate(vp_ranges):
        if a != b:
            continue
        anchor_idx_0b = max(0, min(P - 1, int(a) - 1))
        anchor_xz = ref_arr[anchor_idx_0b][[0, 2]]
        if float(np.linalg.norm(anchor_xz - ag_xz_)) <= float(stop_radius_m):
            best_degenerate = i
    if best_degenerate is not None:
        return best_degenerate
    for i, (a, b) in enumerate(vp_ranges):
        if a <= a_seg and b_seg <= b:
            return i
    if t < 0.5:
        for i, (a, b) in enumerate(vp_ranges):
            if b == a_seg:
                return i
    else:
        for i, (a, b) in enumerate(vp_ranges):
            if a == b_seg:
                return i
    first_a = int(vp_ranges[0][0])
    last_b = int(vp_ranges[-1][1])
    if b_seg <= first_a:
        return 0
    if a_seg >= last_b:
        return len(vp_ranges) - 1
    mid = a_seg + t
    (best_i, best_d) = (None, float("inf"))
    for i, (a, b) in enumerate(vp_ranges):
        if mid < a:
            d = float(a - mid)
        elif mid > b:
            d = float(mid - b)
        else:
            d = 0.0
        if d < best_d:
            (best_d, best_i) = (d, i)
    return best_i


class FGR2RLoader:
    """Per-split FG-R2R lookup: ``(trajectory_id, instruction_text) -> sub-instructions``."""

    def __init__(self, json_path: str) -> None:
        self.json_path = str(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self._raw: list[dict] = json.load(f)
        self._by_pid: dict[int, dict] = {}
        for rec in self._raw:
            pid = rec.get("path_id")
            if pid is None:
                continue
            try:
                self._by_pid[int(pid)] = rec
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._by_pid)

    def lookup(self, trajectory_id: int, instruction_text: str) -> _MatchedInstr | None:
        """Return the FG-R2R sub-instruction list aligned to a VLN-CE episode.

        Matches the VLN-CE ``instruction_text`` against the 3 R2R instructions
        for the given ``trajectory_id`` (== FG-R2R ``path_id``) by normalised
        string equality, then returns the corresponding sub-instructions and
        viewpoint ranges. Returns ``None`` if no exact match.
        """
        try:
            tid = int(trajectory_id)
        except Exception:
            return None
        rec = self._by_pid.get(tid)
        if rec is None:
            return None
        instrs = rec.get("instructions") or []
        ni = _parse_field(rec.get("new_instructions"))
        cv = _parse_field(rec.get("chunk_view"))
        if not isinstance(ni, list) or not isinstance(cv, list):
            return None
        target = _norm_text(instruction_text or "")
        idx = None
        for i, instr in enumerate(instrs):
            if _norm_text(instr) == target:
                idx = i
                break
        if idx is None:
            idx = 0
        if idx >= len(ni) or idx >= len(cv):
            return None
        sub_token_lists = ni[idx]
        sub_ranges = cv[idx]
        S = min(len(sub_token_lists), len(sub_ranges))
        if S == 0:
            return None
        sub_texts: list[str] = []
        vp_ranges: list[tuple[int, int]] = []
        for s in range(S):
            toks = sub_token_lists[s]
            try:
                text = " ".join((str(t) for t in toks)).strip()
            except Exception:
                text = ""
            try:
                (a, b) = (int(sub_ranges[s][0]), int(sub_ranges[s][1]))
            except Exception:
                continue
            if not text:
                continue
            sub_texts.append(text)
            vp_ranges.append((a, b))
        if not sub_texts:
            return None
        return _MatchedInstr(sub_texts=sub_texts, vp_ranges=vp_ranges)

    @staticmethod
    def subinstr_for_position(
        info: _MatchedInstr, reference_path_xyz: np.ndarray, agent_position_xyz: np.ndarray
    ) -> str | None:
        """Return active subinstruction text from the closest reference-path segment."""
        if reference_path_xyz is None:
            return None
        ref = np.asarray(reference_path_xyz, dtype=np.float64).reshape(-1, 3)
        if ref.shape[0] == 0:
            return None
        agent = np.asarray(agent_position_xyz, dtype=np.float64).reshape(3)
        (seg_i, t) = _locate_agent_on_path(ref, agent)
        idx = _match_subinstr_for_segment(
            info.vp_ranges, seg_i, t, P=ref.shape[0], agent_xyz=agent, reference_path_xyz=ref
        )
        if idx is None:
            return None
        return info.sub_texts[idx]

    @staticmethod
    def goal_and_subpath_for_position(
        info: _MatchedInstr,
        reference_path_xyz: np.ndarray,
        agent_position_xyz: np.ndarray,
        *,
        stop_radius_m: float = 1.0,
        agent_future_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the active subinstruction goal and forward reference suffix.

        Near a nonfinal subinstruction endpoint, choose a dense future replay
        position at the arrival radius and retain that partial path as the
        geometry. Text and terminal eligibility remain with the active chunk."""
        if reference_path_xyz is None:
            return None
        ref = np.asarray(reference_path_xyz, dtype=np.float32).reshape(-1, 3)
        P = int(ref.shape[0])
        if P == 0:
            return None
        agent = np.asarray(agent_position_xyz, dtype=np.float64).reshape(3)
        (seg_i, t) = _locate_agent_on_path(ref, agent)
        idx = _match_subinstr_for_segment(
            info.vp_ranges,
            seg_i,
            t,
            P=P,
            agent_xyz=agent,
            reference_path_xyz=ref,
            stop_radius_m=float(stop_radius_m),
        )
        if idx is None:
            return None
        (a_vp, b_vp) = info.vp_ranges[idx]
        end_vp = int(b_vp)
        end_idx = max(0, min(P - 1, end_vp - 1))
        local_goal = ref[end_idx].copy()
        promoted_subpath: np.ndarray | None = None
        if idx + 1 < len(info.vp_ranges):
            ag_xz = agent[[0, 2]]
            cur_end_xz = ref[end_idx][[0, 2]].astype(np.float64)
            d_xz = float(np.linalg.norm(ag_xz - cur_end_xz))
            if d_xz <= float(stop_radius_m):
                future = np.asarray(agent_future_positions, dtype=np.float64).reshape(-1, 3)
                _LOOKAHEAD_HORIZON = 32
                upper = min(int(future.shape[0]), _LOOKAHEAD_HORIZON)
                picked_k: int | None = None
                for k in range(upper):
                    if float(np.linalg.norm(ag_xz - future[k][[0, 2]])) >= float(stop_radius_m):
                        picked_k = k
                        break
                if picked_k is None and upper > 0:
                    dists = np.linalg.norm(ag_xz[None, :] - future[:upper][:, [0, 2]], axis=1)
                    far_k = int(np.argmax(dists))
                    if float(dists[far_k]) >= 0.1:
                        picked_k = far_k
                if picked_k is not None:
                    local_goal = future[picked_k].astype(np.float32).copy()
                    seg = np.vstack(
                        [
                            agent.astype(np.float32).reshape(1, 3),
                            future[: picked_k + 1].astype(np.float32),
                        ]
                    )
                    promoted_subpath = seg
        if promoted_subpath is not None:
            sub_path = promoted_subpath
        else:
            sub_start = seg_i if t < 0.5 else seg_i + 1
            sub_start = max(0, min(sub_start, end_idx))
            if end_idx >= sub_start:
                sub_path = ref[sub_start : end_idx + 1].copy()
            else:
                sub_path = ref[sub_start : sub_start + 1].copy()
            if sub_path.shape[0] < 2:
                sub_path = np.vstack([agent.astype(np.float32).reshape(1, 3), sub_path])
        return (local_goal, sub_path)

    @staticmethod
    def is_last_subinstr_for_position(
        info: _MatchedInstr, reference_path_xyz: np.ndarray, agent_position_xyz: np.ndarray
    ) -> bool:
        """Only a valid match to the final subinstruction permits terminal labels."""
        if info is None or not info.vp_ranges:
            return False
        if reference_path_xyz is None:
            return False
        ref = np.asarray(reference_path_xyz, dtype=np.float64).reshape(-1, 3)
        if ref.shape[0] == 0:
            return False
        agent = np.asarray(agent_position_xyz, dtype=np.float64).reshape(3)
        (seg_i, t) = _locate_agent_on_path(ref, agent)
        idx = _match_subinstr_for_segment(
            info.vp_ranges, seg_i, t, P=ref.shape[0], agent_xyz=agent, reference_path_xyz=ref
        )
        if idx is None:
            return False
        return idx == len(info.vp_ranges) - 1


__all__ = ["FGR2RLoader"]
