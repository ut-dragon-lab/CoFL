"""Native RxR timed-word sentence alignment, migrated from the benchmark loader."""

from __future__ import annotations

from typing import Any


def extract_sub_instructions(ep, T: int) -> list[str]:
    """Return a per-timestep sub-instruction list of length T.

    For RxR episodes that carry ``timed_instruction`` data, align the active
    sentence to rollout time using the instruction timestamps. For all other
    episodes (R2R, or RxR without timing data) return an empty list.

    Args:
        ep:  Habitat episode object (or a raw episode dict).
        T:   Number of rollout timesteps (length of the returned list).

    Returns:
        List[str] of length T. Each entry is the sub-instruction active at
        that timestep.
    """
    if T <= 0:
        return []

    def _timed_get(item: Any, *keys: str) -> Any:
        if isinstance(item, dict):
            for key in keys:
                if key in item and item[key] is not None:
                    return item[key]
            return None
        for key in keys:
            val = getattr(item, key, None)
            if val is not None:
                return val
        return None

    def _coerce_float(v: Any) -> float | None:
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError, OverflowError):
            return None

    def _align_segments_to_timesteps(
        segments: list[dict[str, Any]],
        *,
        T: int,
    ) -> list[str]:
        if not segments:
            return []

        last_with_end = next((seg for seg in reversed(segments) if seg["end"] is not None), None)
        total_dur = float(last_with_end["end"]) if last_with_end is not None else 1.0
        total_dur = max(total_dur, 1e-6)

        result: list[str] = []
        for t in range(T):
            t_sec = (t / max(T - 1, 1)) * total_dur
            active = next(
                (
                    seg
                    for seg in segments
                    if seg["start"] is not None
                    and seg["end"] is not None
                    and seg["start"] <= t_sec <= seg["end"]
                ),
                None,
            )
            if active is None:
                active = min(
                    segments,
                    key=lambda seg: min(
                        abs((seg["start"] if seg["start"] is not None else t_sec) - t_sec),
                        abs((seg["end"] if seg["end"] is not None else t_sec) - t_sec),
                    ),
                )
            result.append(str(active["text"]))
        return result

    def _timed_words_to_sentences(timed_words: list[Any]) -> list[dict[str, Any]]:
        sentences: list[dict[str, Any]] = []
        cur_words: list[str] = []
        cur_start: float | None = None

        for word_item in timed_words:
            word = str(_timed_get(word_item, "word", "text", "instruction_text") or "").strip()
            if not word:
                continue
            t_start = _coerce_float(_timed_get(word_item, "start_time", "startTime", "start"))
            t_end = _coerce_float(_timed_get(word_item, "end_time", "endTime", "end"))
            if cur_start is None:
                cur_start = t_start if t_start is not None else 0.0
            cur_words.append(word)
            stripped = word.rstrip("\"'")
            if stripped.endswith((".", "?", "!")):
                sentences.append(
                    {
                        "text": " ".join(cur_words).strip(),
                        "start": float(cur_start if cur_start is not None else 0.0),
                        "end": float(
                            t_end
                            if t_end is not None
                            else (cur_start if cur_start is not None else 0.0)
                        ),
                    }
                )
                cur_words = []
                cur_start = None

        if cur_words:
            last_end = _coerce_float(_timed_get(timed_words[-1], "end_time", "endTime", "end"))
            start = float(cur_start if cur_start is not None else 0.0)
            end = float(last_end if last_end is not None else start)
            sentences.append(
                {
                    "text": " ".join(cur_words).strip(),
                    "start": start,
                    "end": end,
                }
            )
        return [seg for seg in sentences if seg["text"]]

    if isinstance(ep, dict):
        instr_obj = ep.get("instruction", None)
    else:
        instr_obj = getattr(ep, "instruction", None)

    timed = None
    if instr_obj is not None:
        if isinstance(instr_obj, dict):
            timed = instr_obj.get("timed_instruction", None)
        else:
            timed = getattr(instr_obj, "timed_instruction", None)

    if timed and len(timed) > 0:
        timed_list = list(timed)

        # RxR raw JSON stores word-level timed tokens. Convert those into
        # sentence spans and align them by timestamp to rollout time.
        if _timed_get(timed_list[0], "word") is not None:
            sentence_spans = _timed_words_to_sentences(timed_list)
            if sentence_spans:
                return _align_segments_to_timesteps(sentence_spans, T=T)

        # Fallback: some callers may expose sentence-level timed segments
        # directly. Use provided timestamps when available, otherwise keep
        # the previous uniform broadcast behavior across segment order.
        timed_segments: list[dict[str, Any]] = []
        for seg in timed_list:
            txt = str(_timed_get(seg, "instruction_text", "text", "word") or "").strip()
            if not txt:
                continue
            timed_segments.append(
                {
                    "text": txt,
                    "start": _coerce_float(_timed_get(seg, "start_time", "startTime", "start")),
                    "end": _coerce_float(_timed_get(seg, "end_time", "endTime", "end")),
                }
            )
        if timed_segments:
            if all(seg["start"] is not None and seg["end"] is not None for seg in timed_segments):
                return _align_segments_to_timesteps(timed_segments, T=T)
            n = len(timed_segments)
            return [timed_segments[min(int(t * n / T), n - 1)]["text"] for t in range(T)]

    # No timed sub-instructions (e.g. R2R) -> return empty list.
    # Callers should treat [] as "no sub-instruction available".
    return []
