"""Batched offline evaluation with a common model, dataset and result lifecycle."""

import hashlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from cofl.data import open_dataset
from cofl.runtime.inference_rules import inference_rule_metadata
from cofl.training.config import METHOD_PROFILES
from cofl.training.policy import load_policy

from .config import EvaluationConfig
from .data import build_evaluation_loader, select_evaluation_samples
from .image import evaluate_image_navigation
from .metrics import evaluate_field
from .prediction import predict_batch
from .protocol import EvaluationProtocol, ResultWriter, file_sha256
from .rollout import rollout_image_fields
from .summary import summarize_results


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _code_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _diagnostics(sample):
    episode = sample.get("episode_metadata", {})
    annotation = sample.get("annotation_metadata", {})
    source = annotation.get("source_record", {}).get("command_metadata", {})
    return {
        "status": "ok",
        "instruction": sample["instruction"],
        "scene_id": sample.get("scene_id")
        or episode.get("scene_id")
        or episode.get("source_record", {}).get("scene_id")
        or "unknown",
        "category": annotation.get("target_category")
        or annotation.get("episode", {}).get("target_category")
        or source.get("object_mpcat40")
        or "unknown",
        "annotation_key": sample.get("annotation_key"),
        "episode_id": sample["episode_id"],
        "observation_id": sample["observation_id"],
    }


def score_batch(batch, predictions, config):
    samples = batch["samples"]
    trajectories = {}
    if predictions.navigation_fields is not None:
        valid = np.isfinite(predictions.navigation_fields).all(axis=(1, 2, 3))
        indices = np.flatnonzero(valid)
        if len(indices):
            starts = np.stack(
                [samples[i]["annotation_extras"]["trajectory_state"][0] for i in indices]
            )
            settings = asdict(config.trajectory)
            settings.pop("grid_size")
            settings.pop("resample_points")
            rollout = rollout_image_fields(
                predictions.navigation_fields[indices], starts, **settings
            )
            trajectories = {
                int(i): (rollout.points[j], int(rollout.clamped_steps[j]))
                for j, i in enumerate(indices)
            }
    records = []
    for index, sample in enumerate(samples):
        metrics, diagnostics, errors = {}, _diagnostics(sample), []
        if (
            "field" in config.tasks
            and not np.isfinite(predictions.fields[index][sample["mask"]]).all()
        ):
            errors.append("nonfinite_field_prediction")
        if "image_navigation" in config.tasks and index not in trajectories:
            errors.append("nonfinite_navigation_field")
        if "action" in config.tasks and not np.isfinite(predictions.actions[index]).all():
            errors.append("nonfinite_action_logits")
        if errors:
            diagnostics.update(status="failed", error=",".join(errors))
            if "image_navigation" in config.tasks:
                metrics["image_navigation"] = {"cr": 1.0, "fge": None, "plr": None, "curv": None}
        else:
            if "field" in config.tasks:
                metrics["field"] = evaluate_field(
                    predictions.fields[index],
                    np.moveaxis(sample["field"], 0, -1),
                    profile=sample["profile"],
                    geometry=sample["geometry"],
                    mask=sample["mask"],
                )
            if "image_navigation" in config.tasks:
                points, clamped = trajectories[index]
                extras = sample["annotation_extras"]
                metrics["image_navigation"] = evaluate_image_navigation(
                    points,
                    extras["trajectory_state"],
                    goal=extras["goal"],
                    navigation_mask=sample["observation_extras"]["navigation_mask"],
                    resample_points=config.trajectory.resample_points,
                )
                diagnostics.update(prediction_trajectory=points.tolist(), clamped_steps=clamped)
            if "action" in config.tasks:
                label = sample.get("target_action")
                logits = predictions.actions[index]
                metrics["action"] = {
                    "target": label,
                    "prediction": int(logits.argmax()),
                    "cross_entropy": float(
                        F.cross_entropy(torch.from_numpy(logits)[None], torch.tensor([label]))
                    )
                    if label is not None
                    else None,
                }
        records.append(
            {"sample_id": sample["sample_id"], "metrics": metrics, "diagnostics": diagnostics}
        )
    return records


def _figure_positions(selection, config):
    count = min(config.visualization_count, len(selection.sample_ids))
    return np.sort(
        np.random.default_rng(config.seed).choice(len(selection.sample_ids), count, replace=False)
    )


def _render_figures(selection, writer, policy, model_config, config, assets, cached_fields):
    from .data import EvaluationCollator, image_navigation_labels
    from .report import render_sample

    if config.visualization_count == 0:
        return []
    positions = _figure_positions(selection, config)
    wanted = {selection.sample_ids[i] for i in positions}
    records = {
        record["sample_id"]: record for record in writer.records() if record["sample_id"] in wanted
    }
    plot_config = replace(config, tasks=("field",))
    collator = EvaluationCollator(
        processor_files=assets["processor_files"],
        profile=selection.identity["profile"],
        max_text_length=assets["max_text_length"],
        use_depth=model_config.method == "cofl-s" and model_config.use_depth,
        tasks=plot_config.tasks,
    )
    # The inference reader omits unused payloads. Figures load their own full view.
    dataset = open_dataset(
        config.dataset,
        expected_profile=selection.identity["profile"],
        split=config.split,
        eligible_only=True,
    )
    samples = []
    for position in positions:
        sample_id = selection.sample_ids[position]
        if sample_id not in records:
            continue
        sample = dataset[int(selection.indices[position])]
        if "image_navigation" in config.tasks:
            annotation, observation = image_navigation_labels(sample)
            sample = {**sample, "annotation_extras": annotation, "observation_extras": observation}
        samples.append(sample)
    missing = [
        sample
        for sample in samples
        if "field" in config.tasks
        and records[sample["sample_id"]]["diagnostics"]["status"] == "ok"
        and sample["sample_id"] not in cached_fields
    ]
    # A resumed run may need fields from earlier executions; infer these in batches.
    for start in range(0, len(missing), config.batch_size):
        batch = missing[start : start + config.batch_size]
        predictions = predict_batch(policy, model_config, collator(batch), plot_config)
        cached_fields.update(zip((sample["sample_id"] for sample in batch), predictions.fields))
    figures = []
    for sample in samples:
        sample_id = sample["sample_id"]
        path = (
            writer.output_dir
            / "figures"
            / f"{hashlib.sha256(sample_id.encode()).hexdigest()[:20]}.png"
        )
        prediction = (
            cached_fields.get(sample_id)
            if records[sample_id]["diagnostics"]["status"] == "ok"
            else None
        )
        render_sample(path, sample, prediction, records[sample_id])
        figures.append(
            {
                "sample_id": sample_id,
                "path": str(path.relative_to(writer.output_dir)),
                "instruction": sample["instruction"],
            }
        )
    return figures


def _score_timed(batch, predictions, config):
    started = time.perf_counter()
    with torch.profiler.record_function("cofl.score"):
        records = score_batch(batch, predictions, config)
    return records, time.perf_counter() - started


def _run_batches(loader, policy, model_config, config, writer, execution, figure_ids, fields):
    """Overlap CPU scoring with inference; commit a bounded, ordered result stream."""
    timings = execution["stages"]
    pending = deque()
    commit_failed = False
    pool = ThreadPoolExecutor(max_workers=config.score_workers) if config.score_workers else None
    profiler = nullcontext(None)
    if config.profile_batches and len(loader):
        warmup = min(1, len(loader) - 1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(config.device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=warmup,
                active=min(config.profile_batches, len(loader) - warmup),
                repeat=1,
            ),
            record_shapes=True,
            on_trace_ready=lambda trace: trace.export_chrome_trace(
                str(writer.output_dir / "profile.json")
            ),
        )

    def commit(result, progress):
        records, seconds = result
        timings["scoring_seconds"] += seconds
        started = time.perf_counter()
        with torch.profiler.record_function("cofl.commit"):
            writer.write_many(records)
        timings["commit_seconds"] += time.perf_counter() - started
        execution["processed_samples"] += len(records)
        progress.update(1)

    def commit_next(progress):
        nonlocal commit_failed
        started = time.perf_counter()
        try:
            result = pending[0].result()
            timings["scoring_wait_seconds"] += time.perf_counter() - started
            commit(result, progress)
        except BaseException:
            commit_failed = True
            raise
        pending.popleft()

    started = time.perf_counter()
    try:
        with (
            profiler as trace,
            tqdm(total=len(loader), desc="Evaluating", unit="batch") as progress,
        ):
            try:
                waiting = time.perf_counter()
                iterator = iter(loader)
                timings["data_wait_seconds"] += time.perf_counter() - waiting
                while True:
                    # Bound retained input/prediction batches and preserve commit order.
                    while pending and (
                        len(pending) >= config.score_workers + 1 or pending[0].done()
                    ):
                        commit_next(progress)
                    waiting = time.perf_counter()
                    with torch.profiler.record_function("cofl.data_wait"):
                        batch = next(iterator, None)
                    timings["data_wait_seconds"] += time.perf_counter() - waiting
                    if batch is None:
                        break
                    predicting = time.perf_counter()
                    with torch.profiler.record_function("cofl.predict"):
                        predictions = predict_batch(policy, model_config, batch, config)
                    timings["prediction_seconds"] += time.perf_counter() - predicting
                    if figure_ids and "field" in config.tasks:
                        for sample, field in zip(batch["samples"], predictions.fields):
                            if sample["sample_id"] in figure_ids:
                                fields[sample["sample_id"]] = field.copy()
                    # Release preprocessed model tensors before queuing CPU scoring.
                    scoring_batch = {"samples": batch["samples"]}
                    if pool is None:
                        commit(_score_timed(scoring_batch, predictions, config), progress)
                    else:
                        pending.append(
                            pool.submit(_score_timed, scoring_batch, predictions, config)
                        )
                    del batch, predictions, scoring_batch
                    if trace is not None:
                        trace.step()
            except BaseException as error:
                # Preserve successful in-flight batches on inference/interruption errors.
                # A scoring/commit error stays at the front, so later rows cannot pass it.
                while pending and not commit_failed:
                    try:
                        commit_next(progress)
                    except BaseException as pending_error:  # noqa: BLE001
                        error.add_note(f"Pending scoring/commit also failed: {pending_error}")
                        break
                raise
            else:
                while pending:
                    commit_next(progress)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        timings["loop_seconds"] = time.perf_counter() - started


def evaluate(config: EvaluationConfig):
    startup_started = time.perf_counter()
    checkpoint_digest = file_sha256(config.checkpoint)
    policy, model_config = load_policy(config.checkpoint, device=config.device)
    if file_sha256(config.checkpoint) != checkpoint_digest:
        raise ValueError("Checkpoint changed while loading; evaluate a stable saved checkpoint")
    profile = METHOD_PROFILES[model_config.method]
    if "image_navigation" in config.tasks and profile != "image_field_v1":
        raise ValueError(
            "image_navigation requires an image-field policy; ground navigation uses a separate task protocol"
        )
    if "action" in config.tasks and not callable(getattr(policy, "action_logits", None)):
        raise ValueError("The checkpoint has no action head")
    selection = select_evaluation_samples(
        config.dataset,
        profile=profile,
        split=config.split,
        seed=config.seed,
        max_samples=config.max_samples,
        tasks=config.tasks,
        use_depth=model_config.method == "cofl-s" and model_config.use_depth,
    )
    inference_settings = {}
    if "image_navigation" in config.tasks:
        inference_settings = asdict(config.trajectory)
        inference_settings.pop("resample_points")
    inference_rules = inference_rule_metadata(profile, **inference_settings)
    recipe = {
        "selection": selection.identity,
        "tasks": list(config.tasks),
        "trajectory": asdict(config.trajectory) if "image_navigation" in config.tasks else None,
        "field": {
            "query_grid": "native_profile_grid",
            "magnitude_epsilon": 1e-8,
            "zero_prediction_angle_deg": 180,
        },
        "image_collision": "segment_supercover_or_out_of_bounds",
        "precision": config.precision,
        "inference_rules": inference_rules,
        "inference_applications": {
            "field_queries": "native_profile_grid" if "field" in config.tasks else None,
            "trajectory": "image_navigation" in config.tasks,
            "action_head": "action" in config.tasks,
        },
    }
    protocol = EvaluationProtocol(
        name="offline_policy_v2",
        version=2,
        profile=profile,
        split=config.split,
        seed=config.seed,
        recipe="offline_policy_v2",
        task=",".join(config.tasks),
        instruction_mode="provided_annotation",
        instruction_source="instruction bound to each dataset annotation",
        checkpoint_sha256=checkpoint_digest,
        dataset_sha256=selection.identity["dataset_manifest_sha256"],
        recipe_sha256=_digest(recipe),
        code_revision="sha256:" + _code_digest(),
        inference_rules=inference_rules,
    )
    assets = policy.backbone.export_assets()
    output = Path(config.output)
    with ResultWriter(output, protocol) as writer:
        (output / "recipe.json").write_text(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        loader = build_evaluation_loader(
            selection,
            processor_files=assets["processor_files"],
            max_text_length=assets["max_text_length"],
            use_depth=model_config.method == "cofl-s" and model_config.use_depth,
            tasks=config.tasks,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            multiprocessing_context=config.multiprocessing_context,
            pin_memory=torch.device(config.device).type == "cuda",
            completed_sample_ids=writer.completed_sample_ids,
        )
        started = time.perf_counter()
        execution = {
            "startup_seconds": started - startup_started,
            "device": config.device,
            "batch_size": config.batch_size,
            "query_chunk_size": config.query_chunk_size,
            "num_workers": config.num_workers,
            "precision": config.precision,
            "score_workers": config.score_workers,
            "profile_batches": config.profile_batches,
            "processed_samples": 0,
            "torch": torch.__version__,
            "stages": dict.fromkeys(
                (
                    "data_wait_seconds",
                    "prediction_seconds",
                    "scoring_seconds",
                    "scoring_wait_seconds",
                    "commit_seconds",
                    "loop_seconds",
                ),
                0.0,
            ),
        }
        figure_ids = {selection.sample_ids[i] for i in _figure_positions(selection, config)}
        cached_fields = {}
        try:
            _run_batches(
                loader, policy, model_config, config, writer, execution, figure_ids, cached_fields
            )
        finally:
            summarizing = time.perf_counter()
            summary, table = summarize_results(
                writer, selected_samples=len(selection.sample_ids), tasks=config.tasks
            )
            execution["stages"]["summary_seconds"] = time.perf_counter() - summarizing
            execution.update(seconds=time.perf_counter() - started, complete=summary["complete"])
            with (output / "executions.jsonl").open("a") as handle:
                handle.write(json.dumps(execution) + "\n")
        figures = _render_figures(
            selection, writer, policy, model_config, config, assets, cached_fields
        )
        from .report import write_report

        score = (
            "image_navigation.fge" if "image_navigation" in config.tasks else "field.vector_l2_mean"
        )
        worst = (
            table.sort_values(score, ascending=False)
            .head(20)
            .replace({np.nan: None})
            .to_dict("records")
            if score in table
            else []
        )
        write_report(
            output,
            summary,
            {**protocol.to_dict(), "fingerprint": protocol.fingerprint},
            figures,
            worst,
        )
        return summary
