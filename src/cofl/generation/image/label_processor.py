"""
Label processing for semantic segmentation masks.

Handles label mapping, category classification, and mask preprocessing.
"""

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt


class LabelProcessor:
    """
    Processes semantic labels for VLA dataset generation.

    Handles:
    - Label mapping (e.g., "sofa chair" -> "sofa")
    - Category classification (walkable, obstacle, target)
    - Mask preprocessing (ceiling removal, interior wall removal)
    """

    def __init__(self, config_path: str | Path):
        """
        Initialize the label processor.

        Args:
            config_path: Path to the label configuration YAML file
        """
        self.config = self._load_config(config_path)

        # Extract categories from config
        self.label_mapping = self.config.get("label_mapping", {})
        self.ignore_labels = set(self.config.get("ignore_labels", []))
        self.walkable_cats = set(self.config.get("walkable_categories", []))
        self.obstacle_cats = set(self.config.get("obstacle_categories", []))
        self.target_cats = set(self.config.get("target_categories", []))
        self.non_directional_cats = set(self.config.get("non_directional_categories", []))
        self.occluder_cats = set(self.config.get("occluder_categories", []))

        # Instruction config
        instr_conf = self.config.get("instruction_config", {})
        self.action_verbs = instr_conf.get("action_verbs", ["go to"])
        self.position_descriptors = instr_conf.get("position_descriptors", {})

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        with Path(config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise TypeError("Label configuration must be a mapping")
        return config

    def _matches_category(self, raw_label: str, category_key: str) -> bool:
        """Check if raw_label matches category_key using word-boundary matching."""
        raw = raw_label.lower().strip()
        key = category_key.lower().strip()

        if raw == key:
            return True

        pattern = r"\b" + re.escape(key) + r"\b"
        return bool(re.search(pattern, raw))

    def map_and_classify_label(self, raw_label: str) -> tuple[str, list[str]]:
        """
        Map raw label to clean category and determine types.

        Args:
            raw_label: The raw label string from the dataset

        Returns:
            (clean_category, types) where types is a list containing
            any of: 'walkable', 'obstacle', 'target'
        """
        raw_lower = raw_label.lower().strip()

        # Check ignore list
        for ignore in self.ignore_labels:
            if self._matches_category(raw_lower, ignore):
                return raw_lower, []

        # Apply mapping - sort by key length descending to prefer longer/more specific matches
        # e.g., "toilet paper" should match before "toilet"
        clean_category = raw_lower
        sorted_mappings = sorted(self.label_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        for source, target in sorted_mappings:
            if self._matches_category(raw_lower, source):
                clean_category = target
                break

        # Determine types
        types = []

        for walk in self.walkable_cats:
            if self._matches_category(clean_category, walk):
                types.append("walkable")
                break

        for obs in self.obstacle_cats:
            if self._matches_category(clean_category, obs):
                types.append("obstacle")
                break

        for tgt in self.target_cats:
            if self._matches_category(clean_category, tgt):
                types.append("target")
                break

        return clean_category, types

    def get_walkable_mask(self, semantic_mask: np.ndarray, label_map: dict[int, str]) -> np.ndarray:
        """
        Get binary walkable mask from semantic segmentation.

        Args:
            semantic_mask: 2D array of label IDs
            label_map: Mapping from label ID to label name

        Returns:
            Binary mask where True = walkable
        """
        walkable = np.zeros_like(semantic_mask, dtype=bool)
        for label_id, raw_label in label_map.items():
            _, types = self.map_and_classify_label(raw_label)
            if "walkable" in types:
                walkable |= semantic_mask == label_id
        return walkable

    def preprocess_semantic_mask(
        self, semantic_mask: np.ndarray, label_map: dict[int, str]
    ) -> np.ndarray:
        """
        Preprocess semantic mask by removing ignored labels, ceiling and interior walls.

        Args:
            semantic_mask: 2D array of label IDs
            label_map: Mapping from label ID to label name

        Returns:
            Preprocessed semantic mask
        """
        semantic_mask = semantic_mask.copy()
        ignore_ids = []
        ceiling_ids = []
        wall_ids = []
        floor_ids = []

        for label_id, raw_label in label_map.items():
            cleaned, types = self.map_and_classify_label(raw_label)

            # Only remove labels explicitly in ignore_labels config (delete, unknown, etc.)
            # Check by matching against ignore_labels directly
            raw_lower = raw_label.lower().strip()
            is_ignored = False
            for ignore in self.ignore_labels:
                if self._matches_category(raw_lower, ignore) or self._matches_category(
                    cleaned, ignore
                ):
                    is_ignored = True
                    break
            if is_ignored:
                ignore_ids.append(label_id)

            if "obstacle" in types and "ceiling" in cleaned.lower():
                ceiling_ids.append(label_id)
            if "obstacle" in types and "wall" in cleaned.lower():
                wall_ids.append(label_id)
            if "walkable" in types:
                floor_ids.append(label_id)

        # Remove ignored labels (delete, unknown, etc.) by inpainting with nearest valid label
        if ignore_ids:
            ignore_mask = np.isin(semantic_mask, ignore_ids)
            if np.any(ignore_mask) and not np.all(ignore_mask):
                indices = distance_transform_edt(
                    ignore_mask, return_distances=False, return_indices=True
                )
                semantic_mask = semantic_mask[tuple(indices)]

        # Remove ceiling by inpainting with nearest non-ceiling label
        if ceiling_ids:
            ceiling_mask = np.isin(semantic_mask, ceiling_ids)
            if np.any(ceiling_mask) and not np.all(ceiling_mask):
                indices = distance_transform_edt(
                    ceiling_mask, return_distances=False, return_indices=True
                )
                semantic_mask = semantic_mask[tuple(indices)]

        # Remove interior walls surrounded by floor
        if wall_ids and floor_ids:
            wall_mask = np.isin(semantic_mask, wall_ids)
            if np.any(wall_mask):
                num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
                    wall_mask.astype(np.uint8), connectivity=8
                )
                mask_to_remove = np.zeros_like(wall_mask, dtype=bool)
                kernel = np.ones((3, 3), np.uint8)

                for i in range(1, num_labels):
                    component_mask = labels == i
                    # Skip if touches image boundary
                    if (
                        np.any(component_mask[0, :])
                        or np.any(component_mask[-1, :])
                        or np.any(component_mask[:, 0])
                        or np.any(component_mask[:, -1])
                    ):
                        continue

                    dilated = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1)
                    boundary = (dilated == 1) & (~component_mask)
                    if not np.any(boundary):
                        continue

                    boundary_labels = semantic_mask[boundary]
                    is_surrounded = np.all(np.isin(boundary_labels, floor_ids))
                    if is_surrounded:
                        mask_to_remove |= component_mask

                if np.any(mask_to_remove):
                    indices = distance_transform_edt(
                        mask_to_remove, return_distances=False, return_indices=True
                    )
                    semantic_mask = semantic_mask[tuple(indices)]

        return semantic_mask

    def is_non_directional(self, category: str) -> bool:
        """Check if a category should not have directional targets."""
        return category in self.non_directional_cats
