"""Optional local export using the installed LeRobot writer.

This module performs no network upload. Compatibility must be verified
against the installed LeRobot version with the optional integration test.
The native CoFL format itself is not a LeRobot dataset.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from .profiles import validate_geometry
from .storage import CoFLDataset, _validate_manifest, validate_dataset

IMAGE_KEY = "observation.images.camera"
DEPTH_KEY = "observation.depth"
DEPTH_MASK_KEY = "observation.depth_valid"


def _lerobot_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot export requires an independently installed, compatible lerobot package. "
            "See docs/dataset-format.md; the native CoFL reader does not need LeRobot."
        ) from exc
    return LeRobotDataset


def export_lerobot(source: str | Path, output: str | Path, *, repo_id: str, fps: int = 10) -> Path:
    """Export actual observations plus a linked CoFL field extension locally.

    Alternatives remain annotations; they never become extra episode frames.
    Noncontiguous source frames, inconsistent timestamps, or partially
    available action/depth columns are rejected before export. No sequence
    of action chunks is inferred from generated target labels.
    """
    validate_dataset(source)
    dataset = CoFLDataset(source)
    if dataset.manifest["storage"]["image_encoding"] != "rgb":
        raise ValueError("LeRobot export currently requires a dense-RGB native shard")
    if any(name in dataset.arrays for name in ("observation_extras", "annotation_extras")):
        raise ValueError("LeRobot export does not support extra arrays; preserve the native shard")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"LeRobot export output must not already exist: {output}")
    observation_rows = [
        row for ep in dataset.episodes for row in dataset.episode_observations(ep["episode_id"])
    ]
    has_actions = [row["executed_action"] is not None for row in observation_rows]
    has_depths = [row["depth_index"] is not None for row in observation_rows]
    if any(has_actions) and not all(has_actions):
        raise ValueError(
            "LeRobot export requires executed actions on all observations or none; no labels are invented"
        )
    if any(has_depths) and not all(has_depths):
        raise ValueError("LeRobot export currently requires depth on all observations or none")
    for episode in dataset.episodes:
        rows = dataset.episode_observations(episode["episode_id"])
        if any(b["frame_index"] != a["frame_index"] + 1 for a, b in zip(rows, rows[1:])):
            raise ValueError(
                "LeRobot export requires contiguous source frames; preserve missing observations first"
            )
        if rows[0]["timestamp"] is not None:
            if any(
                not np.isclose(b["timestamp"] - a["timestamp"], 1 / fps, atol=1e-6)
                for a, b in zip(rows, rows[1:])
            ):
                raise ValueError("Source timestamps do not match export fps")
    image_shape = tuple(dataset.arrays["images"].shape[1:])
    features = {
        IMAGE_KEY: {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    }
    if all(has_actions):
        features["action"] = {"dtype": "int64", "shape": (1,), "names": ["executed_action_id"]}
    if all(has_depths):
        features[DEPTH_KEY] = {
            "dtype": "float32",
            "shape": image_shape[:2],
            "names": ["height", "width"],
        }
        features[DEPTH_MASK_KEY] = {
            "dtype": "bool",
            "shape": image_shape[:2],
            "names": ["height", "width"],
        }
    lr_class = _lerobot_class()
    base = lr_class.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output,
        robot_type="cofl",
        use_videos=False,
    )
    mapping = []
    global_index = 0
    static_observations = dataset.profile == "image_field_v1" and not any(has_actions)
    export_episodes = []
    for episode in dataset.episodes:
        rows = dataset.episode_observations(episode["episode_id"])
        if static_observations:
            export_episodes.extend((episode, [row]) for row in rows)
        else:
            export_episodes.append((episode, rows))
    for episode_index, (episode, rows) in enumerate(export_episodes):
        for frame_index, row in enumerate(rows):
            frame = {
                "task": episode["task"],
                IMAGE_KEY: np.asarray(dataset.arrays["images"][row["image_index"]]),
            }
            if all(has_actions):
                frame["action"] = np.asarray([row["executed_action"]], dtype=np.int64)
            if all(has_depths):
                frame[DEPTH_KEY] = np.asarray(dataset.arrays["depths"][row["depth_index"]])
                frame[DEPTH_MASK_KEY] = np.asarray(
                    dataset.arrays["depth_masks"][row["depth_index"]]
                )
            base.add_frame(frame)
            mapping.append(
                {
                    **row,
                    "lerobot_episode_index": episode_index,
                    "lerobot_frame_index": frame_index,
                    "lerobot_index": global_index,
                }
            )
            global_index += 1
        base.save_episode()
    base.finalize()
    extension = output / "cofl"
    extension.mkdir()
    pq.write_table(pa.Table.from_pylist(mapping), extension / "observations.parquet")
    for filename in ("episodes.parquet", "annotations.parquet"):
        pq.write_table(pq.read_table(dataset.path / filename), extension / filename)
    dense = zarr.open_group(str(extension / "fields.zarr"), mode="w")
    for key in ("fields", "masks"):
        original = dataset.arrays[key]
        copied = dense.create_dataset(
            key,
            shape=original.shape,
            chunks=original.chunks,
            dtype=original.dtype,
            compressor=original.compressor,
        )
        for index in range(original.shape[0]):
            copied[index] = original[index]
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    manifest = {
        **dataset.manifest,
        "extension_format": "cofl_lerobot_extension_v1",
        "base": {
            "repo_id": repo_id,
            "lerobot_version": version,
            "fps": fps,
            "cofl_source_revision": dataset.manifest["revision"],
            "temporal_mapping": (
                "one_static_observation_per_episode"
                if static_observations
                else "one_source_observation_per_frame"
            ),
        },
    }
    (extension / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class LeRobotCoFLDataset:
    """Join LeRobot's image reader to the separately stored CoFL annotations.

    This custom join is required. A standard LeRobot policy loader does not
    automatically train on CoFL fields or alternative instructions.
    """

    def __init__(
        self, path: str | Path, *, expected_profile: str | None = None, split: str | None = None
    ):
        self.path = Path(path)
        extension = self.path / "cofl"
        self.manifest = json.loads((extension / "manifest.json").read_text(encoding="utf-8"))
        _validate_manifest(self.manifest)
        if self.manifest.get("extension_format") != "cofl_lerobot_extension_v1":
            raise ValueError("Unsupported CoFL LeRobot extension")
        self.profile = self.manifest["profile"]
        if expected_profile is not None and self.profile != expected_profile:
            raise ValueError("Dataset profile does not match expected profile")
        self._observations = {
            row["observation_id"]: row
            for row in pq.read_table(extension / "observations.parquet").to_pylist()
        }
        self._episodes = {
            row["episode_id"]: row
            for row in pq.read_table(extension / "episodes.parquet").to_pylist()
        }
        annotations = pq.read_table(extension / "annotations.parquet").to_pylist()
        for row in annotations:
            obs = self._observations[row["observation_id"]]
            if row["split"] != self._episodes[obs["episode_id"]]["split"]:
                raise ValueError("Annotation split must inherit episode split")
        self.samples = [row for row in annotations if split is None or row["split"] == split]
        self.arrays = zarr.open_group(str(extension / "fields.zarr"), mode="r")
        self.base = _lerobot_class()(repo_id=self.manifest["base"]["repo_id"], root=self.path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        annotation = self.samples[index]
        observation = self._observations[annotation["observation_id"]]
        episode = self._episodes[observation["episode_id"]]
        frame = self.base[observation["lerobot_index"]]
        image = _numpy(frame[IMAGE_KEY])
        # LeRobot's default image transform returns floating-point CHW.
        if image.dtype != np.uint8:
            if (
                image.ndim != 3
                or image.shape[0] != 3
                or not np.isfinite(image).all()
                or image.min() < 0
                or image.max() > 1
            ):
                raise ValueError("Unexpected LeRobot image transform; expected CHW RGB in [0,1]")
            image = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
        geometry = json.loads(annotation["geometry_json"])
        validate_geometry(geometry, self.profile)
        result = {
            "sample_id": annotation["annotation_id"],
            "annotation_id": annotation["annotation_id"],
            "observation_id": observation["observation_id"],
            "episode_id": episode["episode_id"],
            "frame_index": observation["frame_index"],
            "timestamp": observation["timestamp"],
            "profile": self.profile,
            "split": episode["split"],
            "image": image,
            "instruction": annotation["instruction"],
            "instruction_source": annotation["instruction_source"],
            "is_anchor": annotation["is_anchor"],
            "geometry": geometry,
            "field": np.asarray(self.arrays["fields"][annotation["field_index"]]),
            "mask": np.asarray(self.arrays["masks"][annotation["field_index"]]),
            "executed_action": int(_numpy(frame["action"]).item()) if "action" in frame else None,
            "target_action": annotation["target_action"],
            "task": episode["task"],
            "source": self.manifest["source"],
            "recipe": self.manifest["recipe"],
        }
        if DEPTH_KEY in frame:
            result["depth"] = _numpy(frame[DEPTH_KEY])
            result["depth_valid"] = _numpy(frame[DEPTH_MASK_KEY]).astype(np.bool_)
        return result
