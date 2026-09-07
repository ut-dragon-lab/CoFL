"""Bounded, read-only views of the native dataset and collection readers.

Browser coordinates always describe the field's component frame. In particular,
ground previews expose metric (forward, left), never the decoder's polar chart.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from threading import RLock

import numpy as np

from cofl.data.profiles import field_query_grid
from cofl.fields import queries_for_grid, query_valid_mask

from .adapters import DomainConstraint


def coordinate_metadata(profile: str) -> dict:
    if profile == "image_field_v1":
        return {
            "coordinate_frame": "image_xy_right_down",
            "coordinate_unit": "image_fraction",
            "vector_unit": "image_fraction_per_policy_time",
        }
    if profile == "ground_sector_v1":
        return {
            "coordinate_frame": "body_xy_forward_left_fixed_at_observation",
            "coordinate_unit": "m",
            "vector_unit": "m_per_policy_time",
        }
    raise ValueError(f"Unsupported field profile: {profile!r}")


def finite_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def image_data_url(image: np.ndarray) -> str:
    from PIL import Image

    encoded = BytesIO()
    Image.fromarray(image).save(encoded, format="PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def sampled_field(sample: dict, max_points: int = 1024) -> dict:
    """Select actual supervised cells, retaining model and display coordinates.

    Selection is deterministic and bounded. The returned target remains in the
    native units expected by ``evaluate_field``; scaling is only for display.
    """
    profile, geometry = sample["profile"], sample["geometry"]
    grid = queries_for_grid(profile, geometry)
    mask = np.asarray(sample["mask"])
    if mask.shape != grid.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("Dataset mask must be boolean with the native field grid shape")
    field = np.asarray(sample["field"])
    if field.shape != (2, *mask.shape):
        raise ValueError("Dataset field must have shape (2, *geometry.grid_shape)")
    valid = mask & query_valid_mask(profile, geometry)
    indices = np.flatnonzero(valid)
    total = len(indices)
    if total > max_points:
        indices = indices[np.linspace(0, total - 1, max_points, dtype=np.int64)]
    target = field.transpose(1, 2, 0).reshape(-1, 2)[indices]
    positions = field_query_grid(geometry).reshape(-1, 2)[indices]
    scale = 1.0 if profile == "image_field_v1" else geometry["normalization_scale_m"]
    return {
        "queries": grid.reshape(-1, 2)[indices],
        "positions": finite_array(positions * scale, "Field positions"),
        "target": finite_array(target, "Supervised target vectors"),
        "vectors": finite_array(target * scale, "Display target vectors"),
        "supervised_count": total,
    }


def _stored_trajectory(sample: dict) -> np.ndarray | None:
    extras = sample.get("annotation_extras", {})
    if sample["profile"] == "ground_sector_v1":
        points = extras.get("trajectory_body_m")
        if points is None:
            return None
        points = finite_array(points, "Ground trajectory")
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("Ground trajectory must have nonempty shape (N, 2)")
        return points
    if "trajectory_state" not in extras:
        return None
    # The established label reader validates padding, lengths, and coordinates.
    from cofl.evaluation.data import image_navigation_labels

    annotation, _ = image_navigation_labels(sample)
    return annotation["trajectory_state"]


class DatasetService:
    """Resolve configured resource IDs through at most two native reader views."""

    def __init__(self, resources: dict[str, Path]):
        self.resources = {key: Path(value) for key, value in resources.items()}
        self._readers = OrderedDict()
        self._lock = RLock()
        self._closed = False
        self._pending_close_readers = []

    def describe(self) -> list[dict]:
        # Listing resources must not open large Parquet indices or dense arrays.
        with self._lock:
            return [{"id": key, "label": key} for key in self.resources]

    @staticmethod
    def _open(path: Path, split: str):
        from cofl.data import open_dataset

        return open_dataset(
            path,
            split=split,
            cache_size=1,
            metadata_cache_size=2,
            extra_keys={
                "annotation": (
                    "goal",
                    "start",
                    "trajectory_state",
                    "trajectory_valid",
                    "trajectory_length",
                    "trajectory_body_m",
                ),
                "observation": ("navigation_mask",),
            },
        )

    def _remember(self, key: tuple[str, str], reader):
        previous = self._readers.pop(key, None)
        if previous is not None:
            previous.clear_payload_cache()
        self._readers[key] = reader
        while len(self._readers) > 2:
            _, removed = self._readers.popitem(last=False)
            removed.clear_payload_cache()

    def register(
        self,
        resource_id: str,
        path: Path,
        *,
        split: str = "val",
        domain: DomainConstraint | None = None,
    ) -> dict:
        """Validate the native split and read one sample before publishing it."""
        from .resources import resolved_path

        path = resolved_path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        with self._lock:
            if self._closed:
                raise RuntimeError("Dataset service is closed")
            previous = self.resources.get(resource_id)
            if previous is not None and previous.resolve() != path:
                raise ValueError("This dataset ID already belongs to another directory")
            reader = self._open(path, split)
            validated = False
            try:
                if domain is not None:
                    domain.check_dataset(reader.profile)
                if not len(reader):
                    raise ValueError(f"Dataset split {split!r} contains no samples")
                reader[0]
                validated = True
            finally:
                if not validated:
                    reader.clear_payload_cache()
            self.resources[resource_id] = path
            self._remember((resource_id, split), reader)
            return {"profile": reader.profile, "count": len(reader), "split": split}

    def _reader(self, dataset_id: str, split: str):
        if self._closed:
            raise RuntimeError("Dataset service is closed")
        if dataset_id not in self.resources:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        key = (dataset_id, split)
        if key not in self._readers:
            self._remember(key, self._open(self.resources[dataset_id], split))
        self._readers.move_to_end(key)
        return self._readers[key]

    def describe_dataset(self, dataset_id: str, split: str = "val") -> dict:
        with self._lock:
            reader = self._reader(dataset_id, split)
            return {
                "id": dataset_id,
                "label": dataset_id,
                "profile": reader.profile,
                "count": len(reader),
                "split": split,
            }

    def sample(self, dataset_id: str, index: int, split: str = "val") -> dict:
        with self._lock:
            reader = self._reader(dataset_id, split)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(reader)
            ):
                raise ValueError("Sample index is outside the selected dataset split")
            return reader[index]

    def view(self, dataset_id: str, index: int, split: str = "val") -> dict:
        with self._lock:
            sample = self.sample(dataset_id, index, split)
            count = len(self._reader(dataset_id, split))
        preview = sampled_field(sample)
        trajectory = _stored_trajectory(sample)
        start = [0.5, 0.5] if sample["profile"] == "image_field_v1" else [0.0, 0.0]
        if trajectory is not None:
            start = trajectory[0].tolist()
        if sample["profile"] == "image_field_v1":
            declared = sample.get("annotation_extras", {}).get("start")
            if declared is not None:
                declared = finite_array(declared, "Stored start")
                if declared.shape != (2,):
                    raise ValueError("Stored start must have shape (2,)")
                start = declared.tolist()
        return {
            "dataset_id": dataset_id,
            "index": index,
            "count": count,
            "split": split,
            "sample_id": sample["sample_id"],
            "observation_id": sample["observation_id"],
            "profile": sample["profile"],
            "instruction": sample["instruction"],
            "image": image_data_url(sample["image"]),
            "geometry": sample["geometry"],
            "start": start,
            "queries": preview["positions"].tolist(),
            "vectors": preview["vectors"].tolist(),
            "trajectory": None if trajectory is None else trajectory.tolist(),
            "target_action": sample["target_action"],
            "supervised_count": preview["supervised_count"],
            "preview_count": len(preview["queries"]),
            **coordinate_metadata(sample["profile"]),
        }

    def close(self):
        """Drop every reader, retaining failed cleanups only until an explicit retry."""
        with self._lock:
            self._closed = True
            readers = list(self._readers.values()) + self._pending_close_readers
            self._readers.clear()
            self._pending_close_readers = []
            errors, seen = [], set()
            for reader in readers:
                if id(reader) in seen:
                    continue
                seen.add(id(reader))
                try:
                    reader.clear_payload_cache()
                except Exception as error:  # noqa: BLE001 — finish all readers before reporting failure
                    self._pending_close_readers.append(reader)
                    errors.append(type(reader).__name__ + ": " + str(error))
            if errors:
                raise RuntimeError("Dataset cleanup incomplete: " + "; ".join(errors))
