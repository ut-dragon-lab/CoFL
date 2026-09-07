"""Generate native image-field shards from rendered RGB and semantic views."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from cofl.data.profiles import geometry_for_profile
from cofl.generation.metadata import portable_metadata
from cofl.generation.types import GenerationUnit

from .view import ViewProcessor

_DEFAULT_POTENTIAL = {
    "safe_radius": 50.0,
    "safety_cost_weight": 1.0,
    "goal_weight": 1.0,
    "inside_obstacle_speed_norm": 1.0,
    "smoothing_iterations": 2,
    "smoothing_kernel_size": 5,
    "smoothing_sigma": 1.0,
    "check_margin": 20,
    "min_walkable_pixels": 50,
    "max_directional_goals": 2,
}
_DEFAULT_TRAJECTORY = {
    "enabled": True,
    "min_length": 16,
    "max_steps": 200,
    "num_output_length": 100,
    "min_start_goal_dist": 0.1,
    "distance_bias_power": 2.0,
    "disable_resample": False,
    "use_spline_resample": True,
    "polyline_preserve_vertices": False,
    "smooth_strength": 0.0001,
    "spline_s": None,
    "endpoint_weight": 50.0,
    "endpoint_blend_len": 4,
    "start_sampling_trials": 20,
}
_PIPELINE_OPTIONS = {
    "data_dirs",
    "label_config",
    "split",
    "scene_ids",
    "exclude_scenes",
    "val_scenes",
    "top_view_only",
    "max_regions",
    "flow_resolution",
    "field_resolution",
    "image_resolution",
    "num_start_samples",
    "potential_field",
    "trajectory",
}


def _resolution(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [width,height]")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 2 for item in value):
        raise ValueError(f"{name} dimensions must be integers of at least two")
    return tuple(value)


class ImagePipeline:
    """A rendered view is one independent unit; the shared runner owns storage.

    Source scene directories contain annotations.json plus either top.png and
    top_mask.npy, or region*/ directories with those files. Additional views
    use view0.png/view0_mask.npy through view7. Each semantic .npy contains the
    source dictionary {mask: integer HxW array, label_map: integer-to-name map}.
    """

    profile = "image_field_v1"
    recipe_version = "1"
    writer_options: ClassVar[dict] = {
        "image_encoding": "rgb",
        "field_dtype": "float16",
        "chunk_rows": 8,
        "compression_shuffle": "byte",
    }

    def __init__(self, options: dict, *, base_dir: Path):
        self.options = dict(options)
        unknown = self.options.keys() - _PIPELINE_OPTIONS
        if unknown:
            raise ValueError(f"Unsupported image pipeline options: {', '.join(sorted(unknown))}")
        self.base_dir = Path(base_dir).resolve()
        roots = self.options.get("data_dirs", [])
        if not isinstance(roots, list) or not roots:
            raise ValueError("Image generation requires a nonempty data_dirs list")
        self.data_dirs = [self._resolve(path) for path in roots]
        for path in self.data_dirs:
            if not path.is_dir():
                raise FileNotFoundError(f"Rendered source directory does not exist: {path}")
        if "label_config" not in self.options:
            raise ValueError("Image generation requires an explicit label_config")
        self.label_config = self._resolve(self.options["label_config"])
        self.split = self.options.get("split", "train")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("Image generation split must be a nonempty string")
        for key in ("scene_ids", "exclude_scenes", "val_scenes"):
            value = self.options.get(key)
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            ):
                raise ValueError(f"{key} must be a list of scene names")
        if self.options.get("val_scenes") and self.split not in {"train", "val"}:
            raise ValueError("val_scenes selection requires split='train' or split='val'")
        if not isinstance(self.options.get("top_view_only", False), bool):
            raise TypeError("top_view_only must be a boolean")
        max_regions = self.options.get("max_regions")
        if max_regions is not None and (
            isinstance(max_regions, bool) or not isinstance(max_regions, int) or max_regions < 1
        ):
            raise ValueError("max_regions must be a positive integer or null")
        self.flow_resolution = _resolution(
            self.options.get("flow_resolution", [224, 224]), "flow_resolution"
        )
        self.field_resolution = _resolution(
            self.options.get("field_resolution", [50, 50]), "field_resolution"
        )
        self.image_resolution = int(self.options.get("image_resolution", 224))
        if self.image_resolution < 2:
            raise ValueError("image_resolution must be at least two")
        potential = dict(self.options.get("potential_field", {}))
        unknown = set(potential) - {*_DEFAULT_POTENTIAL, "max_speed_norm", "static_length_scale"}
        if unknown:
            raise ValueError(
                f"Unsupported static potential_field options: {', '.join(sorted(unknown))}"
            )
        self.potential = {**_DEFAULT_POTENTIAL, **potential}
        trajectory = dict(self.options.get("trajectory", {}))
        unknown = set(trajectory) - _DEFAULT_TRAJECTORY.keys()
        if unknown:
            raise ValueError(f"Unsupported trajectory options: {', '.join(sorted(unknown))}")
        self.trajectory = {**_DEFAULT_TRAJECTORY, **trajectory}
        for key in (
            "enabled",
            "disable_resample",
            "use_spline_resample",
            "polyline_preserve_vertices",
        ):
            if not isinstance(self.trajectory[key], bool):
                raise TypeError(f"trajectory.{key} must be a boolean")
        self.processor = ViewProcessor(
            self.label_config,
            flow_resolution=self.flow_resolution,
            field_resolution=self.field_resolution,
            num_start_samples=int(self.options.get("num_start_samples", 5)),
            generate_trajectories=self.trajectory["enabled"],
            trajectory_params={
                key: value for key, value in self.trajectory.items() if key != "enabled"
            },
            potential_field_params=self.potential,
        )

    def _resolve(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.base_dir / path).resolve()

    def units(self) -> Iterable[GenerationUnit]:
        exclude = set(self.options.get("exclude_scenes", []))
        validation = set(self.options.get("val_scenes", []))
        selected = self.options.get("scene_ids")
        selected = set(selected) if selected is not None else None
        scene_paths = sorted(
            path for root in self.data_dirs for path in root.iterdir() if path.is_dir()
        )
        seen = set()
        for scene in scene_paths:
            scene_id = scene.name
            if scene_id in exclude or (selected is not None and scene_id not in selected):
                continue
            if validation and (self.split == "val") != (scene_id in validation):
                continue
            annotations = scene / "annotations.json"
            if not annotations.is_file():
                continue
            regions = sorted(
                path for path in scene.iterdir() if path.is_dir() and path.name.startswith("region")
            )
            if not regions:
                regions = [scene]
            max_regions = self.options.get("max_regions")
            if max_regions is not None:
                regions = regions[:max_regions]
            for region in regions:
                region_id = "region0" if region == scene else region.name
                views = (
                    ["top"]
                    if self.options.get("top_view_only", False)
                    else ["top", *(f"view{i}" for i in range(8))]
                )
                for view_id in views:
                    rgb, semantic = region / f"{view_id}.png", region / f"{view_id}_mask.npy"
                    if rgb.is_file() != semantic.is_file():
                        raise FileNotFoundError(
                            f"Incomplete rendered view {scene_id}/{region_id}/{view_id}: "
                            "both RGB and semantic mask are required"
                        )
                    if not rgb.is_file():
                        continue
                    key = f"{scene_id}__{region_id}__{view_id}"
                    if key in seen:
                        raise ValueError(f"Duplicate rendered view identity: {key}")
                    seen.add(key)
                    yield GenerationUnit(
                        key=key,
                        payload={
                            "scene_id": scene_id,
                            "region_id": region_id,
                            "view_id": view_id,
                            "rgb": rgb,
                            "semantic": semantic,
                        },
                        inputs=(rgb, semantic, self.label_config, annotations),
                    )

    def generate(self, unit: GenerationUnit, writer, *, seed: int) -> dict:
        """Append one view; the shared runner scopes Python/NumPy RNG to seed."""
        source = unit.payload
        result = self.processor.process_view(source["rgb"], source["semantic"])
        if result is None:
            return {"reason": "no_usable_walkable_area_or_target", "annotations": 0, "seed": seed}
        if not result["n_flows"]:
            return {
                "reason": "no_direction_survived_trajectory_validation",
                "annotations": 0,
                "seed": seed,
            }
        metadata = result["metadata"]
        annotations = []
        for target_index, target in enumerate(metadata["targets"]):
            for direction, info in target["directions"].items():
                trajectory = info.pop("trajectory", None)
                annotations.append((target_index, target, direction, info, trajectory))
        if len(annotations) != result["n_flows"]:
            raise ValueError("Generated field and instruction counts disagree")
        if self.trajectory["enabled"] and any(item[-1] is None for item in annotations):
            raise ValueError("A generated instruction is missing its trajectory")
        lengths = [len(item[-1]["states"]) for item in annotations if item[-1] is not None]
        trajectory_capacity = max(lengths, default=0)
        sample_metadata = {
            "sample_id": unit.key,
            "scene_id": source["scene_id"],
            "region_id": source["region_id"],
            "view_id": source["view_id"],
            "source_image": f"{source['scene_id']}/{source['region_id']}/{source['rgb'].name}",
            **metadata,
        }
        image = cv2.resize(
            result["image"],
            (self.image_resolution, self.image_resolution),
            interpolation=cv2.INTER_AREA,
        )
        episode = writer.add_episode(
            unit.key,
            split=self.split,
            task="language_navigation",
            metadata={"scene_id": source["scene_id"], "source_view": unit.key},
        )
        observation = writer.add_observation(
            episode,
            frame_index=0,
            image=image,
            metadata=portable_metadata(sample_metadata),
            extras={
                "navigation_mask": result["mask"],
                "sdf": result["sdf"],
                "label_mask": result["label_mask"],
            },
        )
        geometry = geometry_for_profile(self.profile, tuple(result["flows"].shape[-2:]))
        supervision = np.ones(geometry["grid_shape"], dtype=bool)
        for target_index, target, direction, info, trajectory in annotations:
            flow_index = int(info["flow_idx"])
            extras = {
                "potential": result["potentials"][flow_index],
                "geodesic": result["geodesics"][flow_index],
                "goal": np.asarray(info["goal"], dtype=np.float32),
            }
            annotation_metadata = {
                "flow_idx": flow_index,
                "target_index": target_index,
                "target_category": target["category"],
                "target_instance_id": target["instance_id"],
                "target_label_id": target["label_id"],
                "direction": direction,
                "training_eligible": True,
            }
            if trajectory is not None:
                length = len(trajectory["states"])
                states = np.zeros((trajectory_capacity, 2), dtype=np.float32)
                actions = np.zeros_like(states)
                states[:length], actions[:length] = trajectory["states"], trajectory["actions"]
                extras.update(
                    trajectory_state=states,
                    trajectory_action=actions,
                    trajectory_valid=np.arange(trajectory_capacity) < length,
                    trajectory_length=np.asarray(length, dtype=np.int64),
                    start=np.asarray(trajectory["start"], dtype=np.float32),
                )
                annotation_metadata.update(
                    trajectory_length=length, trajectory_action_semantics="next_position_xy"
                )
            writer.add_annotation(
                observation,
                annotation_key=f"flow-{flow_index}",
                instruction=info["instruction"],
                field=result["flows"][flow_index],
                mask=supervision,
                geometry=geometry,
                metadata=portable_metadata(annotation_metadata),
                extras=extras,
            )
        return {
            "observations": 1,
            "annotations": len(annotations),
            "targets": result["n_targets"],
            "trajectories": len(lengths),
            "source_image_shape": list(result["image"].shape),
            "seed": seed,
        }

    def close(self) -> None:
        """The image pipeline holds no persistent file or simulator handles."""
