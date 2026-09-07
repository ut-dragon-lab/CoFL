"""Deterministic native evaluation cohorts and standard CPU DataLoader batches."""

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cofl.data import open_dataset
from cofl.data.collection import manifest_path
from cofl.fields import queries_for_grid, query_valid_mask
from cofl.training.policy import prepare_inputs

from .protocol import file_sha256


def _tasks(profile, tasks):
    if isinstance(tasks, str) or not tasks:
        raise ValueError("Evaluation tasks must be a nonempty sequence")
    values = tuple(sorted(set(tasks)))
    if set(values) - {"field", "image_navigation", "action"}:
        raise ValueError("Evaluation tasks must be field, image_navigation, or action")
    if "image_navigation" in values and profile != "image_field_v1":
        raise ValueError("image_navigation requires image_field_v1")
    if "action" in values and profile != "ground_sector_v1":
        raise ValueError("action evaluation requires ground_sector_v1")
    return values


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class EvaluationSelection:
    """A fixed, ordered cohort chosen before completed results are excluded."""

    dataset: Any = field(repr=False)
    indices: np.ndarray = field(repr=False)
    sample_ids: tuple[str, ...]
    identity: dict


def select_evaluation_samples(
    dataset_root,
    *,
    profile,
    split,
    seed=42,
    max_samples=None,
    tasks=("field",),
    use_depth=False,
):
    """Select identities without dense reads and project subsequent task payloads.

    Depth payloads are loaded only when the model's ``use_depth`` setting is true.
    """
    _positive_integer(seed, "seed", minimum=0)
    if not isinstance(split, str) or not split.strip():
        raise ValueError("An explicit nonempty evaluation split is required")
    if max_samples is not None:
        _positive_integer(max_samples, "max_samples")
    tasks = _tasks(profile, tasks)
    dataset = open_dataset(
        dataset_root,
        expected_profile=profile,
        split=split,
        eligible_only=True,
        include_extras="image_navigation" in tasks,
        include_field="field" in tasks,
        include_depth=use_depth,
        extra_keys={
            "annotation": (
                "goal", "start", "trajectory_state", "trajectory_valid", "trajectory_length"
            ),
            "observation": ("navigation_mask",),
        },
    )
    if not len(dataset):
        raise ValueError(f"The eligible {split!r} evaluation split is empty")
    count = min(len(dataset), max_samples or len(dataset))
    indices = (
        np.arange(count, dtype=np.int64)
        if count == len(dataset)
        else np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False))
    )
    sample_ids = tuple(dataset.sample_id(int(index)) for index in indices)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Evaluation cohort contains duplicate annotation identities")
    indices.setflags(write=False)
    identity = {
        "dataset_manifest_sha256": file_sha256(manifest_path(dataset_root)),
        "dataset_hash_scope": "native manifest only",
        "profile": profile,
        "split": split,
        "eligible_only": True,
        "eligible_samples": len(dataset),
        "selected_samples": count,
        "seed": seed,
        "max_samples": max_samples,
        "tasks": list(tasks),
        "selection_sha256": hashlib.sha256(
            json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_partial": bool(dataset.manifest.get("source", {}).get("partial", False)),
    }
    return EvaluationSelection(dataset, indices, sample_ids, identity)


def image_navigation_labels(sample):
    """Read navigation labels and trim a declared valid trajectory prefix."""
    annotation = sample.get("annotation_extras", {})
    observation = sample.get("observation_extras", {})
    required = ("goal", "trajectory_state")
    if any(name not in annotation for name in required) or "navigation_mask" not in observation:
        raise ValueError(
            f"Sample {sample['sample_id']} lacks image_navigation labels: "
            "annotation goal/trajectory_state and observation navigation_mask are required"
        )
    goal, trajectory = (np.asarray(annotation[name]) for name in required)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError(f"Sample {sample['sample_id']} has an invalid navigation goal")
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) < 1:
        raise ValueError(f"Sample {sample['sample_id']} has an invalid navigation trajectory")
    capacity = len(trajectory)
    length = capacity
    if "trajectory_valid" in annotation:
        valid = np.asarray(annotation["trajectory_valid"])
        if valid.shape != (capacity,) or not np.isin(valid, (0, 1)).all():
            raise ValueError(f"Sample {sample['sample_id']} has an invalid trajectory_valid mask")
        length = int(np.count_nonzero(valid))
        if not np.array_equal(valid, np.arange(capacity) < length):
            raise ValueError(f"Sample {sample['sample_id']} trajectory_valid must select a prefix")
    if "trajectory_length" in annotation:
        declared = np.asarray(annotation["trajectory_length"])
        if declared.shape != () or declared.dtype.kind not in "iu":
            raise ValueError(
                f"Sample {sample['sample_id']} trajectory_length must be an integer scalar"
            )
        declared_length = int(declared)
        if "trajectory_valid" in annotation and declared_length != length:
            raise ValueError(
                f"Sample {sample['sample_id']} trajectory_length conflicts with trajectory_valid"
            )
        length = declared_length
    if not 1 <= length <= capacity:
        raise ValueError(
            f"Sample {sample['sample_id']} trajectory length must be within 1..{capacity}"
        )
    trajectory = trajectory[:length]
    if not np.isfinite(trajectory).all():
        raise ValueError(f"Sample {sample['sample_id']} has an invalid navigation trajectory")
    navigation = np.asarray(observation["navigation_mask"])
    if navigation.ndim != 2 or min(navigation.shape) < 1 or not np.isin(navigation, (0, 1)).all():
        raise ValueError(f"Sample {sample['sample_id']} has an invalid navigation_mask")
    # Keep the independent goal. A GT trajectory endpoint is not its substitute.
    selected_observation = {
        key: observation[key]
        for key in ("navigation_mask", "label_mask", "sdf")
        if key in observation
    }
    selected_observation["navigation_mask"] = navigation.astype(bool, copy=False)
    return (
        {
            "goal": goal,
            "trajectory_state": trajectory,
            **({"start": annotation["start"]} if "start" in annotation else {}),
        },
        selected_observation,
    )


@dataclass
class EvaluationCollator:
    """Prepare model tensors in workers; retain scoring arrays on the CPU."""

    processor_files: dict = field(repr=False)
    profile: str
    tasks: tuple[str, ...]
    max_text_length: int = 64
    use_depth: bool = False
    _processor: Any = field(default=None, init=False, repr=False)
    _grid_cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def _grid(self, geometry):
        key = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
        if key not in self._grid_cache:
            queries = queries_for_grid(self.profile, geometry)
            queries.setflags(write=False)
            valid = (
                None
                if self.profile == "image_field_v1"
                else query_valid_mask(self.profile, geometry)
            )
            if valid is not None:
                valid.setflags(write=False)
            self._grid_cache[key] = queries, valid
            if len(self._grid_cache) > 16:
                self._grid_cache.popitem(last=False)
        self._grid_cache.move_to_end(key)
        return self._grid_cache[key]

    def __call__(self, samples):
        records = []
        for sample in samples:
            if sample["profile"] != self.profile:
                raise ValueError("Evaluation batch profile does not match its cohort")
            geometry = sample["geometry"]
            record = {
                key: sample[key]
                for key in (
                    "sample_id",
                    "annotation_id",
                    "annotation_key",
                    "observation_id",
                    "episode_id",
                    "frame_index",
                    "timestamp",
                    "profile",
                    "split",
                    "instruction",
                    "instruction_source",
                    "task",
                    "is_anchor",
                    "target_action",
                    "executed_action",
                    "geometry",
                    "episode_metadata",
                    "observation_metadata",
                    "annotation_metadata",
                )
            }
            if "scene_id" in sample:
                record["scene_id"] = sample["scene_id"]
            record["image_shape"] = tuple(sample["image"].shape)
            if "field" in self.tasks:
                query_grid, valid = self._grid(geometry)
                target = np.asarray(sample["field"])
                mask = np.asarray(sample["mask"], dtype=bool)
                if target.shape != (2, *query_grid.shape[:2]) or mask.shape != query_grid.shape[:2]:
                    raise ValueError(
                        f"Sample {sample['sample_id']} field/mask do not match its geometry"
                    )
                record.update(
                    field=target,
                    mask=mask if valid is None else mask & valid,
                    query_grid=query_grid,
                )
            if "image_navigation" in self.tasks:
                annotation, observation = image_navigation_labels(sample)
                record.update(annotation_extras=annotation, observation_extras=observation)
            if "action" in self.tasks and sample["target_action"] not in (None, 0, 1, 2, 3):
                raise ValueError(f"Sample {sample['sample_id']} action target must be 0..3 or null")
            records.append(record)
        if self._processor is None:
            from cofl.models.backbone import processor_from_files

            self._processor = processor_from_files(self.processor_files)
        return {
            "inputs": prepare_inputs(
                self._processor,
                samples,
                max_text_length=self.max_text_length,
                use_depth=self.use_depth,
            ),
            "samples": records,
        }


def build_evaluation_loader(
    selection,
    *,
    processor_files,
    max_text_length=64,
    use_depth=False,
    tasks=("field",),
    batch_size=16,
    num_workers=4,
    multiprocessing_context="forkserver",
    pin_memory=False,
    completed_sample_ids=(),
):
    """Skip completed IDs after cohort selection, independently of batch size."""
    _positive_integer(batch_size, "batch_size")
    _positive_integer(num_workers, "num_workers", minimum=0)
    _positive_integer(max_text_length, "max_text_length")
    if multiprocessing_context not in {"forkserver", "spawn"}:
        raise ValueError("multiprocessing_context must be forkserver or spawn")
    tasks = _tasks(selection.identity["profile"], tasks)
    if list(tasks) != selection.identity["tasks"]:
        raise ValueError("Loader tasks must match the selected evaluation cohort")
    if use_depth and selection.identity["profile"] != "ground_sector_v1":
        raise ValueError("Depth inputs require ground_sector_v1")
    if isinstance(completed_sample_ids, str):
        raise ValueError("completed_sample_ids must be a collection of annotation IDs")
    completed = set(completed_sample_ids)
    if completed - set(selection.sample_ids):
        raise ValueError("Completed sample IDs are outside the selected evaluation cohort")
    pending = [
        int(index)
        for index, sample_id in zip(selection.indices, selection.sample_ids)
        if sample_id not in completed
    ]
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(selection.identity["seed"]),
        collate_fn=EvaluationCollator(
            processor_files=processor_files,
            profile=selection.identity["profile"],
            tasks=tasks,
            max_text_length=max_text_length,
            use_depth=use_depth,
        ),
    )
    if num_workers:
        kwargs.update(multiprocessing_context=multiprocessing_context, prefetch_factor=2)
    return DataLoader(Subset(selection.dataset, pending), **kwargs)
