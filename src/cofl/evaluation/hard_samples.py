"""Mine offline field errors with deterministic temporal suppression, without inference."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from .protocol import file_sha256


def _valid_error(sample, metric):
    count = sample.get("valid_count" if metric == "mag_error" else "directional_count", 0)
    value = sample.get(metric)
    return (
        sample.get("status") == "ok"
        and isinstance(count, (int, float)) and count > 0
        and isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )


def _coverage(samples):
    return {
        "samples": len(samples),
        "observations": len({item["observation_id"] for item in samples}),
        "episodes": len({item["episode_id"] for item in samples}),
        "scenes": len({item["scene_id"] for item in samples if item["scene_id"] != "unknown"}),
    }


def select_hard_samples(samples, *, top_fraction=0.1, stride=10, max_samples=None):
    """Take the union of two error tails, then keep the hardest separated frames.

    ``stride`` is a minimum gap in original frame indices within a replay episode,
    not a slice step through annotations. All instructions at a frame compete for
    one slot. Threshold ties are included; the fraction applies before suppression.
    """
    if isinstance(top_fraction, bool) or not isinstance(top_fraction, (int, float)) or not (
        math.isfinite(top_fraction) and 0 < top_fraction <= 1
    ):
        raise ValueError("top_fraction must be in (0, 1]")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive integer or null")
    samples = [dict(sample) for sample in samples]
    for sample in samples:
        scene = sample.get("scene_id")
        sample["scene_id"] = (
            "unknown" if scene is None or scene == "" else scene
            if isinstance(scene, str) else json.dumps(scene, sort_keys=True, ensure_ascii=False)
        )
    ids = [sample["sample_id"] for sample in samples]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("sample_id must be a nonempty string")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in evaluation records")
    thresholds, distributions = {}, {}
    for metric in ("mag_error", "dir_error"):
        values = np.sort([sample[metric] for sample in samples if _valid_error(sample, metric)])
        distributions[metric] = values
        thresholds[metric] = (
            float(values[-math.ceil(len(values) * top_fraction)]) if len(values) else None
        )
    candidates = []
    for sample in samples:
        reasons = []
        for metric, prefix, reason in (
            ("mag_error", "mag", "magnitude"), ("dir_error", "dir", "direction")
        ):
            valid = _valid_error(sample, metric)
            values = distributions[metric]
            percentile = (
                float(np.searchsorted(values, sample[metric], side="right") / len(values))
                if valid else None
            )
            sample[f"{prefix}_percentile"] = percentile
            if valid and sample[metric] > 0 and sample[metric] >= thresholds[metric]:
                reasons.append(reason)
        if not reasons:
            continue
        frame = sample.get("frame_index")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(f"Sample {sample['sample_id']} lacks a valid original frame_index")
        group = sample.get("temporal_group") or sample.get("episode_id")
        if not isinstance(group, str) or not group:
            raise ValueError(f"Sample {sample['sample_id']} lacks a temporal group")
        sample["temporal_group"] = group
        sample["score"] = max(sample["mag_percentile"] or 0, sample["dir_percentile"] or 0)
        sample["selected_by"] = reasons
        candidates.append(sample)
    candidates.sort(key=lambda sample: (
        -sample["score"],
        -((sample["mag_percentile"] or 0) + (sample["dir_percentile"] or 0)),
        sample["sample_id"],
    ))
    selected, positions = [], defaultdict(list)
    suppressed, capped = 0, 0
    for sample in candidates:
        frames = positions[sample["temporal_group"]]
        frame = sample["frame_index"]
        insertion = bisect_left(frames, frame)
        if (
            (insertion and frame - frames[insertion - 1] < stride)
            or (insertion < len(frames) and frames[insertion] - frame < stride)
        ):
            suppressed += 1
            continue
        if max_samples is not None and len(selected) >= max_samples:
            capped += 1
            continue
        insort(frames, frame)
        selected.append(sample)
    scene_evaluated = Counter(sample["scene_id"] for sample in samples)
    scene_candidates = Counter(sample["scene_id"] for sample in candidates)
    scene_selected = Counter(sample["scene_id"] for sample in selected)
    summary = {
        "top_fraction_per_metric": top_fraction,
        "stride_frames": stride,
        "max_samples": max_samples,
        "thresholds": thresholds,
        "metric_counts": {key: len(values) for key, values in distributions.items()},
        "evaluated": _coverage(samples),
        "candidates": _coverage(candidates),
        "selected": _coverage(selected),
        "temporal_suppressed": suppressed,
        "limit_excluded": capped,
        "failed_samples": sum(sample.get("status") != "ok" for sample in samples),
        "selected_by": dict(Counter(reason for sample in selected for reason in sample["selected_by"])),
        "by_scene": {
            scene: {
                "evaluated": scene_evaluated[scene],
                "candidates": scene_candidates[scene],
                "selected": scene_selected[scene],
            }
            for scene in sorted(scene_evaluated)
        },
        "selection_rule": "positive error >= top-fraction cutoff in either metric; ties included",
        "priority": "max empirical percentile, then sum of percentiles, then sample_id",
        "temporal_rule": "at most one annotation per frame; original frame gap >= stride within each replay episode",
    }
    return selected, summary


def _load_records(evaluation):
    """Read a coherent snapshot of the authoritative SQLite store, or its export."""
    protocol = json.loads((evaluation / "protocol.json").read_text())
    recipe = json.loads((evaluation / "recipe.json").read_text())
    recipe_digest = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if recipe_digest != protocol["protocol"].get("recipe_sha256"):
        raise ValueError("Evaluation recipe does not match its saved protocol")
    digest = protocol["protocol_sha256"]
    database = evaluation / "results.sqlite"
    if database.is_file():
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("BEGIN")
            stored = connection.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
            if stored is None or json.loads(stored[0]) != protocol:
                raise ValueError("SQLite and exported evaluation protocols disagree")
            records = [json.loads(row[0]) for row in connection.execute(
                "SELECT record_json FROM results ORDER BY sequence"
            )]
        finally:
            connection.close()
    else:
        with (evaluation / "results.jsonl").open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    if any(record.get("protocol_sha256") != digest for record in records):
        raise ValueError("Evaluation records belong to a different protocol")
    selection = recipe["selection"]
    if len(records) != selection["selected_samples"]:
        raise ValueError(
            f"Offline evaluation is incomplete ({len(records)}/{selection['selected_samples']}); "
            "finish the same evaluation before selecting hard samples"
        )
    if "field" not in recipe["tasks"]:
        raise ValueError("Hard sample selection requires offline field evaluation")
    return records, protocol, recipe


def mine_hard_samples(
    *, evaluation, dataset, output, top_fraction=0.1, stride=10, max_samples=None
):
    """Join scored annotation IDs to native metadata and publish a selection manifest."""
    from cofl.data.collection import manifest_path
    from cofl.data.metadata import iter_sample_metadata

    evaluation, dataset, output = map(Path, (evaluation, dataset, output))
    if output.exists():
        raise ValueError(f"Selection output already exists: {output}; choose a new directory")
    records, protocol, recipe = _load_records(evaluation)
    selection = recipe["selection"]
    dataset_digest = file_sha256(manifest_path(dataset))
    if dataset_digest != selection["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest does not match the offline evaluation")
    lookup = {}
    for record in records:
        identity = record["sample_id"]
        if identity in lookup:
            raise ValueError("Duplicate sample_id in evaluation records")
        field = record.get("metrics", {}).get("field", {})
        lookup[identity] = {
            "sample_id": identity,
            "status": record.get("diagnostics", {}).get("status", "failed"),
            "mag_error": field.get("magnitude_error_mean"),
            "dir_error": field.get("angular_error_deg_mean"),
            "valid_count": field.get("valid_count", 0),
            "directional_count": field.get("directional_count", 0),
            "vector_error_unit": field.get("vector_error_unit"),
        }
    del records
    samples = []
    for metadata in iter_sample_metadata(dataset, split=selection["split"], eligible_only=True):
        record = lookup.pop(metadata["sample_id"], None)
        if record is not None:
            samples.append({**metadata, **record})
    if lookup:
        raise ValueError(f"{len(lookup)} evaluated annotation IDs are missing from the native split")
    selected, summary = select_hard_samples(
        samples, top_fraction=top_fraction, stride=stride, max_samples=max_samples
    )
    summary.update(
        schema_version=1,
        purpose="offline_hard_example_selection",
        source_split=selection["split"],
        evaluation=str(evaluation.resolve()),
        dataset=str(dataset.resolve()),
        dataset_manifest_sha256=dataset_digest,
        protocol_sha256=protocol["protocol_sha256"],
        checkpoint_sha256=protocol["protocol"].get("checkpoint_sha256"),
        evaluation_selection=selection,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hard-samples-", dir=output.parent) as temporary:
        staging = Path(temporary) / "selection"
        staging.mkdir()
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        with (staging / "selected.jsonl").open("w") as handle:
            for sample in selected:
                handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
        columns = [
            "sample_id", "scene_id", "episode_id", "trajectory_id", "observation_id",
            "frame_index", "timestamp", "instruction", "annotation_key", "is_anchor",
            "mag_error", "dir_error", "mag_percentile", "dir_percentile", "score", "selected_by",
        ]
        with (staging / "selected.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for sample in selected:
                writer.writerow({**sample, "selected_by": "+".join(sample["selected_by"])})
        (staging / "sample_ids.txt").write_text(
            "".join(sample["sample_id"] + "\n" for sample in selected)
        )
        staging.rename(output)
    return summary
