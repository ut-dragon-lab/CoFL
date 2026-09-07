"""
Static image-field generation for one rendered semantic view.

Returns native-ready arrays without scene orchestration or storage.
"""

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .instance_detector import InstanceDetector
from .label_processor import LabelProcessor
from .potential_field import (
    analyze_target_accessibility,
    build_safety_cost_map,
    build_weighted_grid_graph,
    compute_obstacle_distance,
    compute_potential_field_from_boundary,
    compute_velocity_field,
    select_goal_directions,
)
from .trajectory import TrajectoryGenerator
from .utils import clean_walkable_mask, resize_field, sample_start_positions


class ViewProcessor:
    """
    Compute static labels for one source view without holding dataset state.
    """

    def __init__(
        self,
        config_path: str,
        flow_resolution: tuple[int, int] = (50, 50),
        field_resolution: tuple[int, int] | None = None,
        num_start_samples: int = 10,
        generate_trajectories: bool = False,
        trajectory_params: dict | None = None,
        potential_field_params: dict | None = None,
    ):
        """
        Initialize the scene processor.

        Args:
            config_path: Path to label configuration YAML
            flow_resolution: (W, H) resolution for flow fields
            num_start_samples: Number of start positions to sample per view (for benchmark)
            generate_trajectories: Whether to keep only directions with a valid trajectory
            trajectory_params: Dictionary of trajectory generation parameters
            potential_field_params: Dictionary of potential field generation parameters
        """
        self.flow_resolution = flow_resolution
        self.field_resolution = (
            tuple(field_resolution) if field_resolution is not None else tuple(flow_resolution)
        )
        self.num_start_samples = num_start_samples
        self.generate_trajectories = generate_trajectories
        self.pf_params = potential_field_params or {}

        # Trajectory parameters for weighted predecessor paths and resampling
        self.traj_params = trajectory_params or {}
        self.trajectory_generator = TrajectoryGenerator(self.traj_params)

        # Initialize sub-processors
        self.label_processor = LabelProcessor(config_path)
        self.instance_detector = InstanceDetector(self.label_processor)

    def generate_instruction(self, furniture: dict, direction: str) -> str:
        """
        Generate natural language instruction for navigation.

        Uses description strategies:
        1. Relative descriptors (e.g., upper-left, upper-left 2nd-from-left) for disambiguation
        2. Plain object reference when there's only one instance

        Args:
            furniture: Instance dictionary with category and relative_desc
            direction: Target direction

        Returns:
            Instruction string
        """
        target_obj = furniture["category"]
        verb = random.choice(self.label_processor.action_verbs)

        # Get disambiguation info
        relative_desc = furniture.get("relative_desc")
        use_relative = bool(relative_desc) and len(relative_desc) > 0

        # Build object phrase
        if use_relative:
            desc = random.choice(relative_desc)
            # desc is a natural phrase like "in the upper left".
            obj_phrase = f"the {target_obj} {desc}"
        else:
            obj_phrase = f"the {target_obj}"

        if self.label_processor.is_non_directional(target_obj) or direction == "center":
            templates_no_pos = [
                "{verb} {obj}",
                "can you {verb} {obj}",
                "please {verb} {obj}",
            ]
            template = random.choice(templates_no_pos)
            return template.format(verb=verb, obj=obj_phrase).strip()
        else:
            pos_opts = self.label_processor.position_descriptors.get(direction, [direction])
            pos_with_of = [p for p in pos_opts if p and "of" in p]
            if pos_with_of:
                pos = random.choice(pos_with_of)
            else:
                pos = direction + " of"

            templates = [
                "{verb} {pos} {obj}",
                "please {verb} {pos} {obj}",
                "can you {verb} {pos} {obj}",
            ]
            template = random.choice(templates)

            return template.format(verb=verb, obj=obj_phrase, pos=pos).strip()

    def process_view(self, rgb_path: Path, mask_path: Path) -> dict | None:
        """
        Process a single view and return data.

        Args:
            rgb_path: Path to RGB image
            mask_path: Path to semantic mask (.npy)

        Returns:
            Dictionary with processed data, or None if view is invalid
        """
        # Load semantic data
        semantic_data = np.load(mask_path, allow_pickle=True).item()
        if not isinstance(semantic_data, dict) or not {"mask", "label_map"} <= semantic_data.keys():
            raise ValueError("Semantic source must contain mask and label_map")
        semantic_mask = np.asarray(semantic_data["mask"])
        label_map = semantic_data["label_map"]
        if (
            semantic_mask.ndim != 2
            or min(semantic_mask.shape) < 2
            or not np.issubdtype(semantic_mask.dtype, np.integer)
        ):
            raise ValueError("Semantic mask must be a nonempty 2D integer label array")
        if not isinstance(label_map, dict) or not all(
            isinstance(key, (int, np.integer)) and isinstance(value, str)
            for key, value in label_map.items()
        ):
            raise ValueError("label_map must map integer label IDs to names")

        # Preprocess mask
        semantic_mask = self.label_processor.preprocess_semantic_mask(semantic_mask, label_map)
        H, W = semantic_mask.shape

        # Get walkable mask
        walkable = self.label_processor.get_walkable_mask(semantic_mask, label_map)
        walkable = clean_walkable_mask(walkable, min_region_area=64)

        if walkable.sum() < 100:
            return None

        # Get furniture instances
        furniture_list = self.instance_detector.get_furniture_instances(semantic_mask, label_map)
        if len(furniture_list) == 0:
            return None

        # Exclude very small targets (policy: drop such targets rather than using coordinate-based instructions).
        furniture_list = [f for f in furniture_list if not bool(f.get("exclude", False))]
        if len(furniture_list) == 0:
            return None

        # Assign a stable instance_id per category within this view.
        # label_id is the semantic class id and is NOT unique across instances.
        by_cat = defaultdict(list)
        for idx, inst in enumerate(furniture_list):
            by_cat[inst["category"]].append((idx, inst))
        for items in by_cat.values():
            items_sorted = sorted(items, key=lambda x: (x[1]["center"][0], x[1]["center"][1]))
            for inst_id, (orig_idx, _) in enumerate(items_sorted):
                furniture_list[orig_idx]["instance_id"] = int(inst_id)

        # Sample start positions
        start_positions = sample_start_positions(walkable, self.num_start_samples)
        if len(start_positions) == 0:
            return None

        # Compute base obstacle map and SDF
        base_obstacle = (~walkable).astype(np.uint8)
        sdf_full = compute_obstacle_distance(base_obstacle)

        # Resize for storage
        sdf_small = resize_field(sdf_full, self.field_resolution).astype(np.float16)
        mask_small = resize_field(walkable.astype(np.float32), self.field_resolution).astype(bool)
        label_mask_small = cv2.resize(
            semantic_mask.astype(np.int16), self.field_resolution, interpolation=cv2.INTER_NEAREST
        )

        # Process each target
        targets_json = []
        all_flows = []
        all_potentials = []
        all_geodesics = []

        # Share one weighted grid graph across all boundaries in this view.
        free_mask = walkable.astype(bool)
        cost_map_full = build_safety_cost_map(
            free_mask=free_mask,
            safe_radius=self.pf_params.get("safe_radius", 10.0),
            safety_cost_weight=self.pf_params.get("safety_cost_weight", 1.0),
        )
        graph_full = build_weighted_grid_graph(
            free_mask=free_mask,
            cost_map=cost_map_full,
            allow_diagonal=True,
        )

        # Precompute connected-component labels per label_id to reconstruct precise instance masks.
        cc_labels_by_label_id: dict[int, np.ndarray] = {}
        for lid in sorted({int(f["label_id"]) for f in furniture_list}):
            obj_mask_u8 = (semantic_mask == lid).astype(np.uint8)
            if obj_mask_u8.sum() == 0:
                continue
            _, cc_labels, _, _ = cv2.connectedComponentsWithStats(obj_mask_u8, connectivity=8)
            cc_labels_by_label_id[lid] = cc_labels

        for furniture in furniture_list:
            x_min, y_min, x_max, y_max = furniture["bbox"]
            cx, cy = furniture["center"]

            # Connected-component membership keeps nearby same-label goals separate.
            lid = int(furniture["label_id"])
            member_ids = np.asarray(furniture["member_comp_ids"], dtype=np.int32)
            if not member_ids.size:
                raise ValueError("Target instances require connected-component membership")
            target_mask = np.isin(cc_labels_by_label_id[lid], member_ids)

            # Analyze accessibility of this target
            accessibility = analyze_target_accessibility(
                target_mask=target_mask,
                walkable_mask=walkable,
                bbox=furniture["bbox"],
                center=furniture["center"],
                check_margin=self.pf_params.get("check_margin", 20),
                min_walkable_pixels=self.pf_params.get("min_walkable_pixels", 50),
            )

            # Select which directions to generate
            if self.label_processor.is_non_directional(furniture["category"]):
                selected_directions = ["center"]
            else:
                selected_directions = select_goal_directions(
                    accessibility["accessible_directions"],
                    max_directional=self.pf_params.get("max_directional_goals", 2),
                )

            directions_data = {}

            for direction in selected_directions:
                boundary_mask = (
                    accessibility["boundary_mask"]
                    if direction == "center"
                    else accessibility["direction_boundaries"].get(direction)
                )
                if boundary_mask is None:
                    continue
                by, bx = np.where(boundary_mask)
                if len(bx) == 0:
                    continue

                # The boundary pixel nearest its centroid is a walkable goal.
                d2 = (bx - float(np.mean(bx))) ** 2 + (by - float(np.mean(by))) ** 2
                nearest = int(d2.argmin())
                goal_for_meta = (int(bx[nearest]), int(by[nearest]))

                boundary_indices = None
                if direction == "center":
                    scale_x = self.flow_resolution[0] / W
                    scale_y = self.flow_resolution[1] / H
                    bx_small = np.clip((bx * scale_x).astype(int), 0, self.flow_resolution[0] - 1)
                    by_small = np.clip((by * scale_y).astype(int), 0, self.flow_resolution[1] - 1)
                    boundary_indices = np.unique(
                        np.stack([bx_small, by_small], axis=1), axis=0
                    ).tolist()

                geometry = compute_potential_field_from_boundary(
                    obstacle_map=base_obstacle,
                    sdf_map=sdf_full,
                    boundary_mask=boundary_mask,
                    walkable_mask=walkable,
                    goal_weight=self.pf_params.get("goal_weight", 1.0),
                    precomputed_graph=graph_full,
                )
                potential = geometry.potential
                geodesic = geometry.weighted_distance
                if not np.isfinite(potential).any():
                    continue

                # Generate only the static normalized image field.

                flow_full = compute_velocity_field(
                    potential,
                    geodesic_distance=geometry.pixel_distance,
                    sdf_map=sdf_full,
                    smoothing_iterations=self.pf_params.get("smoothing_iterations", 2),
                    smoothing_kernel_size=self.pf_params.get("smoothing_kernel_size", 5),
                    smoothing_sigma=self.pf_params.get("smoothing_sigma", 1.0),
                    inside_obstacle_speed_norm=self.pf_params.get(
                        "inside_obstacle_speed_norm", 1.0
                    ),
                    max_speed_norm=self.pf_params.get("max_speed_norm", None),
                    static_length_scale=self.pf_params.get("static_length_scale", None),
                )

                flow_small = resize_field(flow_full, self.flow_resolution).astype(np.float16)

                # Resize geodesic distance field for storage/visualization
                # Use INTER_LINEAR for smooth interpolation
                geodesic_small = cv2.resize(
                    geodesic.astype(np.float32),
                    self.field_resolution,
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float16)

                # Also store a resized potential for this directional goal
                # Clip extreme values before resizing to prevent obstacle penalties from
                # bleeding into walkable areas during linear interpolation
                # Use a reasonable range based on typical geodesic distances
                reachable_mask = np.isfinite(geodesic)
                if reachable_mask.any():
                    max_reachable_dist = geodesic[reachable_mask].max()
                    max_potential = 10.0 * max_reachable_dist + 100.0  # reasonable ceiling
                else:
                    max_potential = 1000.0
                potential_clipped = np.clip(potential, 0.0, max_potential)
                potential_small = resize_field(potential_clipped, self.field_resolution).astype(
                    np.float16
                )

                # Generate instruction
                instruction = self.generate_instruction(furniture, direction)

                # Representative goal used when trajectory generation is disabled.
                goal_norm = np.array([float(goal_for_meta[0] / W), float(goal_for_meta[1] / H)])

                # Keep only fields with a valid trajectory when requested.
                traj_data = None
                if self.generate_trajectories:
                    traj_data = self.trajectory_generator.generate(
                        geodesic=geodesic,
                        geodesic_predecessors=geometry.predecessors,
                        walkable=walkable,
                        sdf=sdf_full,
                        goal_boundary_mask=boundary_mask,
                    )
                    if traj_data is None:
                        # Cannot generate valid trajectory - skip this direction
                        continue

                    # For metadata, store the actual reached goal point on the boundary.
                    goal_norm = np.array(traj_data["goal"], dtype=np.float64)

                flow_idx = len(all_flows)
                directions_data[direction] = {
                    "goal": goal_norm.tolist(),
                    "instruction": instruction,
                    "flow_idx": flow_idx,
                    "trajectory": traj_data,  # Store trajectory data inline (will be extracted later)
                }

                if boundary_indices is not None:
                    directions_data[direction]["boundary_indices"] = boundary_indices

                all_flows.append(flow_small)
                all_potentials.append(potential_small)
                all_geodesics.append(geodesic_small)

            if len(directions_data) == 0:
                continue

            targets_json.append(
                {
                    "category": furniture["category"],
                    "instance_id": int(furniture["instance_id"]),
                    "label_id": furniture["label_id"],
                    "center": [float(cx / W), float(cy / H)],
                    "bbox": [
                        float(x_min / W),
                        float(y_min / H),
                        float(x_max / W),
                        float(y_max / H),
                    ],
                    "area": furniture["area"],
                    "accessible_directions": accessibility["accessible_directions"],
                    "directions": directions_data,
                }
            )

        # Load image (always; even if no targets/flows survive filtering we keep the view)
        with Image.open(rgb_path) as source_image:
            image = np.array(source_image.convert("RGB"))
        if image.shape[:2] != semantic_mask.shape:
            raise ValueError("RGB and semantic mask must have matching source dimensions")

        # Views without surviving labels retain their image and auxiliary arrays.
        if all_flows:
            flows_array = np.stack(all_flows, axis=0).astype(np.float16)
            potentials_array = np.stack(all_potentials, axis=0).astype(np.float16)
            geodesics_array = np.stack(all_geodesics, axis=0).astype(np.float16)
        else:
            flows_array = np.zeros((0, 2, *self.flow_resolution[::-1]), dtype=np.float16)
            potentials_array = np.zeros((0, *self.field_resolution[::-1]), dtype=np.float16)
            geodesics_array = np.zeros((0, *self.field_resolution[::-1]), dtype=np.float16)

        return {
            "metadata": {
                "image_size": [W, H],
                "start_positions": start_positions,
                "targets": targets_json,
                "label_map": label_map,  # Save label map for visualization
            },
            "image": image,
            "sdf": sdf_small,
            "mask": mask_small,
            "label_mask": label_mask_small,
            "flows": flows_array,
            "potentials": potentials_array,
            "geodesics": geodesics_array,
            "n_targets": len(targets_json),
            "n_flows": len(all_flows),
        }
