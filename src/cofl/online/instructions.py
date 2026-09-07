"""Instruction provenance, GT progress selection and revisioned live commands.

The policy only receives the selected text. An oracle may inspect the reference
route and simulator pose to select that text; it never supplies a control path.
Live commands are committed at planner boundaries, so superseded inference
results cannot move the simulator after a newer command has been accepted.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Condition, RLock

import numpy as np


@dataclass(frozen=True)
class InstructionSnapshot:
    text: str
    source: str
    command_id: str
    revision: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "text": self.text,
            "source": self.source,
            "command_id": self.command_id,
            "revision": self.revision,
            "metadata": copy.deepcopy(self.metadata),
        }


class FullInstructionProvider:
    interactive = False

    def __init__(self, episode):
        self.episode = episode

    def describe(self):
        return {
            "instruction_mode": "full_instruction",
            "instruction_source": "episode",
            "uses_gt_progress": False,
        }

    def current(self, environment, frame, step_index):
        return InstructionSnapshot(self.episode["instruction"], "full_instruction", "episode")

    def execute_if_current(self, snapshot, callback):
        return True, callback()


class FullInstructionFallbackProvider(FullInstructionProvider):
    """Explicitly permitted SecVLA fallback when genuine sub-instructions are absent."""

    def __init__(self, episode, rejections):
        super().__init__(episode)
        self.texts = [episode["instruction"]]
        self.selector = None
        self.provenance = {
            "annotation_source": "instruction_fallback",
            "fallback_source": "episode",
            "reason": "oracle_sub_instructions_unavailable",
            "requested_instruction_mode": "oracle",
            "oracle_unavailable": "full_instruction",
            "instruction_id": _episode_instruction_id(episode),
            "rejected_sources": copy.deepcopy(rejections),
        }

    def describe(self):
        return {
            "instruction_mode": "instruction_fallback",
            "requested_instruction_mode": "oracle",
            "instruction_source": "instruction_fallback",
            "uses_gt_progress": False,
            "instruction_provenance": copy.deepcopy(self.provenance),
        }

    def current(self, environment, frame, step_index):
        return InstructionSnapshot(
            self.episode["instruction"],
            "instruction_fallback",
            "episode",
            metadata=copy.deepcopy(self.provenance),
        )


class LiveInstructionProvider:
    interactive = True

    def __init__(self, initial_text=""):
        self._condition = Condition(RLock())
        self._revision = 0
        self._command_id = None
        self._text = ""
        self._active = self._paused = self._closed = False
        self._generation = 0
        self._restart_pose = {"position": None, "rotation": None}
        self._reset_pending = False
        self._reset_error = None
        self._completion_reason = None
        if initial_text:
            self.submit(initial_text)

    def describe(self):
        return {"instruction_mode": "live", "instruction_source": "user", "uses_gt_progress": False}

    def state(self):
        with self._condition:
            return {
                "revision": self._revision,
                "command_id": self._command_id,
                "text": self._text,
                "status": "paused" if self._paused else "running" if self._active else "waiting",
                "restart_generation": self._generation,
                "restart_pose": copy.deepcopy(self._restart_pose),
                "reset_pending": self._reset_pending,
                "reset_error": self._reset_error,
                "completion_reason": self._completion_reason,
                "closed": self._closed,
            }

    def _changed(self):
        self._revision += 1
        self._condition.notify_all()
        return self.state()

    def submit(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 2048:
            raise ValueError("A live instruction must contain 1..2048 characters")
        with self._condition:
            if self._closed:
                raise RuntimeError("The live instruction session is closed")
            self._text = text.strip()
            self._command_id = uuid.uuid4().hex
            self._active = True
            self._completion_reason = None
            return self._changed()

    def pause(self):
        with self._condition:
            self._paused = True
            return self._changed()

    def resume(self):
        with self._condition:
            self._paused = False
            return self._changed()

    def restart(self, *, position=None, rotation=None, instruction=None):
        for name, value, size in (("position", position, 3), ("rotation", rotation, 4)):
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (size,) or not np.isfinite(array).all():
                    raise ValueError("Reset " + name + " has invalid coordinates")
                if name == "rotation" and np.linalg.norm(array) < 1e-6:
                    raise ValueError("Reset quaternion must be nonzero")
        if instruction is not None and (
            not isinstance(instruction, str) or len(instruction) > 2048
        ):
            raise ValueError("Reset instruction must contain at most 2048 characters")
        with self._condition:
            if self._closed:
                raise RuntimeError("The live instruction session is closed")
            self._generation += 1
            self._restart_pose = {
                "position": None if position is None else list(map(float, position)),
                "rotation": None if rotation is None else list(map(float, rotation)),
            }
            self._reset_pending = True
            self._reset_error = None
            self._active = bool(instruction and instruction.strip())
            self._paused = False
            if instruction is not None:
                self._text = instruction.strip()
                self._command_id = uuid.uuid4().hex if self._active else None
            return self._changed()

    def acknowledge_restart(self, generation, error=None):
        with self._condition:
            if generation == self._generation and self._reset_pending:
                self._reset_pending = False
                self._reset_error = None if error is None else str(error)
                if error is not None:
                    self._active = False
                return self._changed()
            return self.state()

    def current(self, environment, frame, step_index):
        with self._condition:
            if self._closed or self._paused or not self._active or self._reset_pending:
                return None
            return InstructionSnapshot(
                self._text,
                "user",
                self._command_id,
                self._revision,
                {"restart_generation": self._generation},
            )

    def execute_if_current(self, snapshot, callback):
        with self._condition:
            if (
                self._closed
                or self._paused
                or not self._active
                or self._reset_pending
                or snapshot.revision != self._revision
            ):
                return False, None
            # Linearization boundary: accepting another command waits for this
            # already-started short control segment. Inference holds no lock.
            return True, callback()

    def complete(self, snapshot, reason="action_head_stop"):
        with self._condition:
            if snapshot.revision != self._revision or not self._active:
                return False
            self._active = False
            self._completion_reason = reason
            self._changed()
            return True

    def wait(self, timeout=0.1):
        with self._condition:
            self._condition.wait(timeout=timeout)

    def close(self):
        with self._condition:
            self._closed = True
            self._active = False
            self._changed()


def _normalize(text):
    return " ".join(str(text).split()).lower()


def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


@lru_cache(maxsize=8)
def _rows(path, mtime_ns, size):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, list):
        raise ValueError("Instruction annotation JSON must contain a list")  # noqa: TRY004
    return result


def _annotation_rows(path):
    stat = Path(path).stat()
    return _rows(str(path), stat.st_mtime_ns, stat.st_size)


def _resample_edges(texts, count):
    if not texts or not any(texts) or count < 1:
        return []
    # Legacy edge mapping fills gaps from the preceding assigned edge, then
    # fills leading gaps backwards. It does not invent new instruction text.
    for index in range(1, len(texts)):
        if not texts[index]:
            texts[index] = texts[index - 1]
    for index in range(len(texts) - 2, -1, -1):
        if not texts[index]:
            texts[index] = texts[index + 1]
    return [texts[round(i * (len(texts) - 1) / max(count - 1, 1))] for i in range(count)]


def _fgr2r_alignment(recipe, episode, reference_count):
    if recipe.fgr2r_json is None:
        return None
    scene = Path(str(episode["scene_id"])).stem
    matches = []
    for row in _annotation_rows(recipe.fgr2r_json):
        if str(row.get("scan", "")) != scene:
            continue
        for index, instruction in enumerate(row.get("instructions", [])):
            if _normalize(instruction) == _normalize(episode["instruction"]):
                matches.append((row, index))
    if len(matches) != 1:
        return None  # Reject ambiguous/wrong-scene matches instead of guessing.
    row, index = matches[0]
    chunks_raw = _literal(row.get("new_instructions", []))[index]
    spans = _literal(row.get("chunk_view", []))[index]
    chunks = []
    for chunk in chunks_raw:
        text = chunk if isinstance(chunk, str) else " ".join(map(str, chunk))
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        chunks.append(_normalize(text))
    count = len(row.get("path", [])) - 1
    if count < 1 or len(chunks) != len(spans):
        return None
    owners = [None] * count
    spans = [tuple(sorted(map(int, span))) for span in spans]
    for i, (start, end) in enumerate(spans):
        for edge in range(max(0, start - 1), min(count, end - 1)):
            owner = owners[edge]
            if owner is None or end - start < spans[owner][1] - spans[owner][0]:
                owners[edge] = i
    for i, (start, end) in enumerate(spans):
        if start != end:
            continue
        target = next(
            (owners[e] for e in (start - 2, start - 1) if 0 <= e < count and owners[e] is not None),
            None,
        )
        if target is not None:
            chunks[target] = (chunks[target] + " " + chunks[i]).strip()
    texts = ["" if owner is None else chunks[owner] for owner in owners]
    return (
        _resample_edges(texts, reference_count - 1),
        "edge",
        {
            "annotation_source": "fgr2r",
            "annotation_path": str(recipe.fgr2r_json),
            "path_id": row.get("path_id"),
            "instruction_index": index,
            "alignment": "chunk_view_edge",
            "source_path_length": count + 1,
        },
    )


def _rxr_text_key(text):
    """Match SecVLA's comparison only; the policy receives the original text."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _meaningful_rxr_chunks(texts, full_instruction):
    unique = {_rxr_text_key(text) for text in texts} - {""}
    if len(unique) >= 2:
        return True
    if not unique:
        return False
    chunk = next(iter(unique))
    full = _rxr_text_key(full_instruction)
    return chunk != full and len(chunk.split()) < max(20, int(0.75 * max(1, len(full.split()))))


def _rxr_instruction_id(value):
    # IDs are integers, never row/path/episode positions or rounded floats.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _episode_instruction_id(episode):
    original = episode.get("instruction_data", {})
    identifier = original.get("instruction_id") if isinstance(original, dict) else None
    return _rxr_instruction_id(episode.get("instruction_id") if identifier is None else identifier)


def _landmark_alignment(recipe, episode, reference_count, *, rejections=None):
    if recipe.landmark_rxr_json is None:
        return None
    identifier = _episode_instruction_id(episode)

    def reject(reason):
        if rejections is not None:
            rejections.append(
                {
                    "annotation_source": "landmark_rxr",
                    "annotation_path": str(recipe.landmark_rxr_json),
                    "instruction_id": identifier,
                    "reason": reason,
                }
            )

    if identifier is None:
        return reject("missing_instruction_id")
    matches = [
        row
        for row in _annotation_rows(recipe.landmark_rxr_json)
        if isinstance(row, dict) and _rxr_instruction_id(row.get("instruction_id")) == identifier
    ]
    if len(matches) != 1:
        return reject("ambiguous_instruction_id" if matches else "no_matching_instruction_id")
    row = matches[0]
    scan = str(row.get("scan") or "").strip()
    scene = Path(str(episode["scene_id"]).replace("\\", "/")).stem
    if scan and scene and scan != scene:
        return reject("scan_mismatch")
    raw_chunks = row.get("sub_instructions")
    if not isinstance(raw_chunks, list):
        return reject("malformed_sub_instructions")
    chunks = [chunk.strip() for chunk in raw_chunks if isinstance(chunk, str) and chunk.strip()]
    # Match SecVLA's source acceptance before looking at edge availability.
    node_texts = (
        [
            chunks[min(index * len(chunks) // reference_count, len(chunks) - 1)]
            for index in range(reference_count)
        ]
        if chunks
        else []
    )
    if not _meaningful_rxr_chunks(node_texts, episode["instruction"]):
        return reject("degenerate_full_instruction")
    raw_path = row.get("path")
    path = [str(node) for node in raw_path] if isinstance(raw_path, list) else []
    subpaths = row.get("sub_paths")
    provenance = {
        "annotation_source": "landmark_rxr",
        "annotation_path": str(recipe.landmark_rxr_json),
        "instruction_id": identifier,
        "path_id": row.get("path_id"),
        "scan": scan,
        "language": str(row.get("language") or ""),
        "match_kind": "exact",
        "source_path_length": len(path),
        "reference_path_length": reference_count,
    }

    def uniform_fallback(reason):
        return (
            node_texts,
            "node",
            dict(provenance, alignment="uniform_reference_node", edge_alignment_unavailable=reason),
        )

    if len(path) < 2 or not isinstance(subpaths, list) or len(chunks) != len(subpaths):
        return uniform_fallback("malformed_sub_paths")
    locations = {node: index for index, node in enumerate(path)}
    texts = [""] * (len(path) - 1)
    for chunk, subpath in zip(chunks, subpaths, strict=True):
        if not isinstance(subpath, list):
            continue
        indices = [locations[str(node)] for node in subpath if str(node) in locations]
        if len(indices) >= 2:
            for edge in range(min(indices), max(indices)):
                texts[edge] = chunk
    if not any(texts):
        return uniform_fallback("no_matched_sub_path_edges")
    return (
        _resample_edges(texts, reference_count - 1),
        "edge",
        dict(
            provenance,
            alignment="sub_path_edge",
            edge_resampling="identity" if len(path) == reference_count else "normalized_edge_index",
        ),
    )


class OracleInstructionProvider(FullInstructionProvider):
    def __init__(self, episode, texts, selector, provenance, *, lookahead=6):
        super().__init__(episode)
        self.reference = np.asarray(episode["reference_path"], dtype=np.float64)
        self.texts, self.selector, self.provenance = texts, selector, provenance
        self.lookahead = lookahead
        self.index = 0

    def describe(self):
        description = {
            "instruction_mode": "oracle",
            "instruction_source": self.provenance["annotation_source"],
            "uses_gt_progress": True,
            "progress_selector": "nearest_reference_" + self.selector + "_monotonic",
            "progress_lookahead": self.lookahead,
            "instruction_provenance": self.provenance,
        }
        if self.provenance.get("oracle_source") == "landmark_rxr":
            description["oracle_source"] = "landmark_rxr"
        return description

    def current(self, environment, frame, step_index):
        position = np.asarray(frame["state"]["position"], dtype=np.float64)
        candidates = range(self.index, min(len(self.texts), self.index + self.lookahead + 1))
        distances = []
        for index in candidates:
            if self.selector == "edge":
                start, end = self.reference[index : index + 2, [0, 2]]
                segment = end - start
                fraction = np.clip(
                    np.dot(position[[0, 2]] - start, segment)
                    / max(np.dot(segment, segment), 1e-12),
                    0,
                    1,
                )
                distance = np.linalg.norm(position[[0, 2]] - (start + fraction * segment))
            else:
                # Native timed sentences map to reference nodes. Use the
                # worker's actual navmesh distance; no silent Euclidean fallback.
                distance = environment.geodesic_distance(position, self.reference[index])
            distances.append((float(distance), index))
        distance, self.index = min(distances)
        if not np.isfinite(distance):
            raise ValueError("Oracle progress is unreachable on the navmesh")
        return InstructionSnapshot(
            self.texts[self.index],
            "oracle",
            "oracle-" + str(self.index),
            metadata=dict(
                self.provenance,
                index=self.index,
                count=len(self.texts),
                selector=self.selector,
                reference_distance_m=distance,
                full_instruction=self.episode["instruction"],
            ),
        )


def make_instruction_provider(recipe, episode, mode=None):
    mode = mode or recipe.instruction_mode
    if mode == "full_instruction":
        return FullInstructionProvider(episode)
    if mode != "oracle":
        raise ValueError("Benchmark instruction_mode must be oracle or full_instruction")
    reference = np.asarray(episode.get("reference_path", []), dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) < 2
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Oracle sub-instructions require the episode's GT reference path")
    oracle_source = getattr(recipe, "oracle_source", "auto")
    if oracle_source not in ("auto", "landmark_rxr"):
        raise ValueError("oracle_source must be auto or landmark_rxr")
    if oracle_source == "landmark_rxr":
        rejections = []
        try:
            aligned = _landmark_alignment(recipe, episode, len(reference), rejections=rejections)
        except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as error:
            raise ValueError(
                "Required oracle source landmark_rxr is unavailable: " + str(error)
            ) from error
        if aligned is not None:
            texts, selector, provenance = aligned
            if texts and all(str(text).strip() for text in texts):
                return OracleInstructionProvider(
                    episode, texts, selector,
                    dict(provenance, oracle_source="landmark_rxr"),
                    lookahead=recipe.progress_lookahead,
                )
        reasons = [item["reason"] for item in rejections]
        if not reasons:
            reasons = ["landmark_rxr_json_not_configured" if recipe.landmark_rxr_json is None
                       else "no_usable_landmark_alignment"]
        raise ValueError(
            "Required oracle source landmark_rxr is unavailable: " + "; ".join(reasons)
        )

    from .native_instructions import extract_sub_instructions

    rejections = []

    def native_alignment():
        native = extract_sub_instructions(
            dict(episode, instruction=episode.get("instruction_data", {})), len(reference)
        )
        if native and _meaningful_rxr_chunks(native, episode["instruction"]):
            return (
                native,
                "node",
                {
                    "annotation_source": "native",
                    "alignment": "timed_sentence_to_reference_node",
                    "instruction_id": _episode_instruction_id(episode),
                    "reference_path_length": len(reference),
                    "timed_instruction_source": "episode.instruction.timed_instruction",
                },
            )
        if native:
            rejections.append(
                {"annotation_source": "native", "reason": "degenerate_full_instruction"}
            )
        return None

    # Evaluate sources lazily: a lower-priority optional source cannot invalidate
    # an already matched annotation. R2R genuine single-segment annotations count.
    candidates = (
        lambda: _landmark_alignment(recipe, episode, len(reference), rejections=rejections),
        native_alignment,
        lambda: _fgr2r_alignment(recipe, episode, len(reference)),
    )
    for candidate in candidates:
        try:
            aligned = candidate()
        except (TypeError, IndexError, KeyError, AttributeError) as error:
            raise ValueError("Malformed oracle instruction annotations: " + str(error)) from error
        if aligned is None:
            continue
        texts, selector, provenance = aligned
        if not texts or not all(str(text).strip() for text in texts):
            continue
        if rejections:
            provenance = dict(provenance, rejected_sources=copy.deepcopy(rejections))
        return OracleInstructionProvider(
            episode, texts, selector, provenance, lookahead=recipe.progress_lookahead
        )
    if getattr(recipe, "oracle_unavailable", "error") == "full_instruction":
        return FullInstructionFallbackProvider(episode, rejections)
    raise ValueError(
        "Oracle sub-instructions are unavailable for this episode. Supply matching FGR2R/Landmark-RxR annotations or explicitly select full_instruction."
    )
