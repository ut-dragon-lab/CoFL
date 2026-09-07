"""
Instance detection for furniture objects.

Handles connected component analysis and intelligent merging of
fragmented instances (e.g., beds split by pillows).
"""

from collections import defaultdict

import cv2
import numpy as np

from .label_processor import LabelProcessor


class InstanceDetector:
    """
    Detects and distinguishes multiple instances of furniture objects.

    Features:
    - Connected component analysis for instance detection
    - Intelligent merging of fragments split by occluding objects
    - Relative position descriptors for disambiguation
    """

    def __init__(self, label_processor: LabelProcessor):
        """
        Initialize the instance detector.

        Args:
            label_processor: LabelProcessor instance for category classification
        """
        self.label_processor = label_processor
        self.occluder_cats = self.label_processor.occluder_cats

    def get_furniture_instances(
        self, semantic_mask: np.ndarray, label_map: dict[int, str]
    ) -> list[dict]:
        """
        Extract furniture instances using connected components analysis.

        This properly handles multiple instances of the same object type
        (e.g., two chairs with the same label_id but separate masks).

        Uses intelligent merging to avoid splitting a single occluded object
        into multiple instances. Analyzes occluding objects (like pillows on beds)
        to determine if components belong to the same object.

        Args:
            semantic_mask: 2D array of label IDs
            label_map: Mapping from label ID to label name

        Returns:
            List of instance dictionaries with keys:
            - category: str
            - label_id: int
            - bbox: [x_min, y_min, x_max, y_max]
            - center: (x, y)
            - area: int
            - relative_desc: Optional[List[str]] - position descriptors
        """
        instances = []
        H, W = semantic_mask.shape

        # Build a map of occluder labels for quick lookup
        occluder_label_ids = set()
        for lid, raw_label in label_map.items():
            cat, _ = self.label_processor.map_and_classify_label(raw_label)
            if cat.lower() in self.occluder_cats:
                occluder_label_ids.add(lid)

        for label_id, raw_label in label_map.items():
            category, type = self.label_processor.map_and_classify_label(raw_label)
            if "target" not in type:
                continue

            # Get binary mask for this label
            obj_mask = (semantic_mask == label_id).astype(np.uint8)
            if obj_mask.sum() < 100:
                continue

            # Find connected components
            num_components, _, stats, centroids = cv2.connectedComponentsWithStats(
                obj_mask, connectivity=8
            )

            # Collect valid components
            components = []
            for comp_id in range(1, num_components):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area < 100:
                    continue

                x_min = stats[comp_id, cv2.CC_STAT_LEFT]
                y_min = stats[comp_id, cv2.CC_STAT_TOP]
                bbox_w = stats[comp_id, cv2.CC_STAT_WIDTH]
                bbox_h = stats[comp_id, cv2.CC_STAT_HEIGHT]
                x_max = x_min + bbox_w
                y_max = y_min + bbox_h

                # Skip very sparse components (likely noise)
                bbox_area = bbox_w * bbox_h
                if bbox_area > area * 5:
                    continue

                cx, cy = centroids[comp_id]

                components.append(
                    {
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "center": (int(cx), int(cy)),
                        "area": int(area),
                        "member_comp_ids": [int(comp_id)],
                    }
                )

            if len(components) == 0:
                continue

            # Merge components that likely belong to same object
            merged = self._merge_fragmented_components(
                category, components, H, W, semantic_mask, occluder_label_ids
            )

            for comp in merged:
                instances.append(
                    {
                        "category": category,
                        "label_id": label_id,
                        "bbox": comp["bbox"],
                        "center": comp["center"],
                        "area": comp["area"],
                        # For deterministic per-instance mask reconstruction in ViewProcessor.
                        # When fragments are merged, this contains multiple component ids.
                        "member_comp_ids": comp["member_comp_ids"],
                    }
                )

        # Add relative position descriptors for duplicate categories
        instances = self._add_relative_descriptors(instances, semantic_mask.shape)
        return instances

    def _merge_fragmented_components(
        self,
        category: str,
        components: list[dict],
        H: int,
        W: int,
        semantic_mask: np.ndarray,
        occluder_label_ids: set[int],
    ) -> list[dict]:
        """
        Merge components that likely belong to the same object.

        Criteria for merging:
        1. Components whose centers are very close relative to their size
        2. Components with significant bounding box overlap
        3. Small fragments near a larger component
        4. Components separated by occluding objects (pillow, cushion, etc.)
        """
        if len(components) <= 1:
            return components

        # Sort by area (largest first)
        components = sorted(components, key=lambda x: x["area"], reverse=True)

        merged = []
        used = set()

        for i, comp_a in enumerate(components):
            if i in used:
                continue

            # Start a new merged component
            merge_group = [comp_a]
            used.add(i)

            for j, comp_b in enumerate(components):
                if j in used:
                    continue

                if self._should_merge_components(
                    category,
                    comp_b,
                    H,
                    W,
                    merge_group,
                    semantic_mask,
                    occluder_label_ids,
                ):
                    merge_group.append(comp_b)
                    used.add(j)

            # Combine the merge group into one instance
            if len(merge_group) == 1:
                merged.append(comp_a)
            else:
                merged.append(self._combine_components(merge_group))

        return merged

    def _should_merge_components(
        self,
        category: str,
        comp_b: dict,
        H: int,
        W: int,
        current_group: list[dict],
        semantic_mask: np.ndarray,
        occluder_label_ids: set[int],
    ) -> bool:
        """
        Determine if two components should be merged.

        Returns True if comp_b should be merged with current_group.
        """
        # Get the combined bounding box of current group
        group_x_min = min(c["bbox"][0] for c in current_group)
        group_y_min = min(c["bbox"][1] for c in current_group)
        group_x_max = max(c["bbox"][2] for c in current_group)
        group_y_max = max(c["bbox"][3] for c in current_group)
        group_area = sum(c["area"] for c in current_group)

        b_x_min, b_y_min, b_x_max, b_y_max = comp_b["bbox"]
        b_cx, b_cy = comp_b["center"]
        b_area = comp_b["area"]

        # 1. Check if comp_b is a small fragment (< 20% of group area)
        #    and its center is inside or very close to group bbox
        if b_area < group_area * 0.2:
            # Expand group bbox slightly
            margin_x = (group_x_max - group_x_min) * 0.3
            margin_y = (group_y_max - group_y_min) * 0.3

            if (
                group_x_min - margin_x <= b_cx <= group_x_max + margin_x
                and group_y_min - margin_y <= b_cy <= group_y_max + margin_y
            ):
                return True

        # 2. Check bounding box overlap (IoU-like)
        inter_x_min = max(group_x_min, b_x_min)
        inter_y_min = max(group_y_min, b_y_min)
        inter_x_max = min(group_x_max, b_x_max)
        inter_y_max = min(group_y_max, b_y_max)

        if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            b_bbox_area = (b_x_max - b_x_min) * (b_y_max - b_y_min)

            # If comp_b's bbox significantly overlaps with group
            if inter_area > b_bbox_area * 0.3:
                return True

        # For most categories (especially small/moveable furniture like tables/chairs),
        # avoid aggressive merging based on proximity or occluder-filled gaps.
        # These heuristics can incorrectly merge two distinct objects that are close.
        aggressive_merge_categories = {
            "bed",
            "sofa",
            "couch",
            "sectional",
        }
        if category not in aggressive_merge_categories:
            return False

        # 3. Check center distance relative to object size
        group_cx = (group_x_min + group_x_max) / 2
        group_cy = (group_y_min + group_y_max) / 2
        group_diag = np.sqrt((group_x_max - group_x_min) ** 2 + (group_y_max - group_y_min) ** 2)

        center_dist = np.sqrt((b_cx - group_cx) ** 2 + (b_cy - group_cy) ** 2)

        # If centers are very close relative to object size, likely same object
        if center_dist < group_diag * 0.5:
            return True

        # 4. Check if gap is occupied by occluding objects (pillow, cushion, blanket, etc.)
        #    But only if the components are reasonably close (within 2x the group diagonal)
        #    This prevents merging distant objects just because they have occluders between them
        if (
            center_dist < group_diag * 2.0
        ):  # Only consider gap merging if components are somewhat close
            total_gap, _occluding_px, ratio = self._analyze_gap_labels(
                current_group, comp_b, H, W, semantic_mask, occluder_label_ids
            )
            if total_gap > 0 and ratio > 0.3:  # If >30% of gap is occluding objects
                return True

        return False

    def _analyze_gap_labels(
        self,
        comp_group: list[dict],
        comp_b: dict,
        H: int,
        W: int,
        semantic_mask: np.ndarray,
        occluder_label_ids: set[int],
    ) -> tuple[int, int, float]:
        """
        Analyze what labels occupy the gap between two component groups.

        Returns:
            (total_gap_pixels, occluding_pixels, occluding_ratio)
        """
        # Get combined bbox of the group
        group_x_min = min(c["bbox"][0] for c in comp_group)
        group_y_min = min(c["bbox"][1] for c in comp_group)
        group_x_max = max(c["bbox"][2] for c in comp_group)
        group_y_max = max(c["bbox"][3] for c in comp_group)

        # Union bbox with comp_b
        union_x_min = min(group_x_min, comp_b["bbox"][0])
        union_y_min = min(group_y_min, comp_b["bbox"][1])
        union_x_max = max(group_x_max, comp_b["bbox"][2])
        union_y_max = max(group_y_max, comp_b["bbox"][3])

        # Clamp to image bounds
        union_x_min = max(0, union_x_min)
        union_y_min = max(0, union_y_min)
        union_x_max = min(W, union_x_max)
        union_y_max = min(H, union_y_max)

        # Create mask of the gap region
        # Gap = union bbox minus the component masks
        gap_mask = np.zeros((H, W), dtype=bool)
        gap_mask[union_y_min:union_y_max, union_x_min:union_x_max] = True

        # Remove the component masks from gap
        for comp in comp_group:
            x1, y1, x2, y2 = comp["bbox"]
            gap_mask[y1:y2, x1:x2] = False
        x1, y1, x2, y2 = comp_b["bbox"]
        gap_mask[y1:y2, x1:x2] = False

        total_gap_pixels = np.sum(gap_mask)
        if total_gap_pixels == 0:
            return 0, 0, 0.0

        # Count pixels of occluding categories in the gap
        occluding_pixels = 0
        for label_id in np.unique(semantic_mask[gap_mask]):
            if label_id in occluder_label_ids:
                occluding_pixels += np.sum((semantic_mask == label_id) & gap_mask)

        ratio = occluding_pixels / total_gap_pixels if total_gap_pixels > 0 else 0.0
        return total_gap_pixels, occluding_pixels, ratio

    def _combine_components(self, components: list[dict]) -> dict:
        """Combine multiple components into a single instance."""
        total_area = sum(c["area"] for c in components)

        # Weighted center by area
        cx = sum(c["center"][0] * c["area"] for c in components) / total_area
        cy = sum(c["center"][1] * c["area"] for c in components) / total_area

        # Combined bounding box
        x_min = min(c["bbox"][0] for c in components)
        y_min = min(c["bbox"][1] for c in components)
        x_max = max(c["bbox"][2] for c in components)
        y_max = max(c["bbox"][3] for c in components)

        member_comp_ids = sorted(
            {int(component_id) for c in components for component_id in c["member_comp_ids"]}
        )

        return {
            "bbox": [x_min, y_min, x_max, y_max],
            "center": (int(cx), int(cy)),
            "area": total_area,
            "member_comp_ids": member_comp_ids,
        }

    def _add_relative_descriptors(
        self,
        instances: list[dict],
        img_shape: tuple[int, int],
        small_target_ratio: float = 0.0001,
    ) -> list[dict]:
        """Describe same-category instances by image region and ordinal.

        Single instances need no descriptor. Duplicates use a 3x3 region,
        then an ordinal along the region's more separated axis if needed.
        Targets below ``small_target_ratio`` of image area are excluded.
        """

        def _ordinal_word(n: int) -> str:
            """Human-ish ordinal for small n; falls back to numeric suffix."""
            n = int(n)
            words = {
                1: "first",
                2: "second",
                3: "third",
                4: "fourth",
                5: "fifth",
            }
            if n in words:
                return words[n]
            if 10 <= (n % 100) <= 20:
                suf = "th"
            else:
                suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            return f"{n}{suf}"

        def _region_desc(nx: float, ny: float) -> tuple[tuple[int, int], str]:
            """Return ((row, col), prepositional region phrase) for a normalized point in [0,1].

            Region naming follows a human-friendly 3x3 partition:
            - Corners: "in the upper left", "in the upper right", "in the lower left", "in the lower right"
            - Edges: "on the left", "on the right", "at the top", "at the bottom"
            - Center: "in the center"
            """
            nx = float(nx)
            ny = float(ny)
            col = 0 if nx < (1.0 / 3.0) else (1 if nx < (2.0 / 3.0) else 2)
            row = 0 if ny < (1.0 / 3.0) else (1 if ny < (2.0 / 3.0) else 2)

            phrases = (
                ("in the upper left", "at the top", "in the upper right"),
                ("on the left", "in the center", "on the right"),
                ("in the lower left", "at the bottom", "in the lower right"),
            )
            return (row, col), phrases[row][col]

        H, W = img_shape
        img_area = H * W

        # Step 1: Add basic info to all instances
        for inst in instances:
            cx, cy = inst["center"]
            # Use higher precision to avoid collisions when two instances are close.
            # Collisions can cause UI/instruction grouping to look like "one target".
            inst["norm_coord"] = (round(cx / W, 3), round(cy / H, 3))
            # NOTE: using only visible pixel area can misclassify large-but-occluded objects
            # (e.g., bed heavily covered by pillows) as "small". We also consider bbox area
            # as a robust proxy for object extent.
            x1, y1, x2, y2 = inst["bbox"]
            bbox_area = max(0, (x2 - x1)) * max(0, (y2 - y1))

            area_ratio = (inst["area"] / img_area) if img_area > 0 else 0.0
            bbox_ratio = (bbox_area / img_area) if img_area > 0 else 0.0

            # Mark as small only if BOTH visible area and bbox extent are small.
            inst["is_small_target"] = (area_ratio < small_target_ratio) and (
                bbox_ratio < small_target_ratio
            )
            # Small targets should be excluded by the caller (rather than forcing coordinate descriptions).
            inst["exclude"] = bool(inst["is_small_target"])

        # Step 2: Group by category
        category_groups = defaultdict(list)
        for i, inst in enumerate(instances):
            category_groups[inst["category"]].append((i, inst))

        # Step 3: Decide disambiguation strategy for each group
        for group in category_groups.values():
            indices = [i for i, _ in group]

            # Single instance: no disambiguation needed (small targets are handled via exclusion)
            if len(group) == 1:
                idx = indices[0]
                instances[idx]["relative_desc"] = None
                continue

            # Multiple instances: 3x3 region descriptor + ordinal within region if needed.
            region_groups: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
            for idx, inst in group:
                nx, ny = inst["norm_coord"]
                key, desc = _region_desc(nx, ny)
                inst["region_3x3"] = key
                inst["region_desc"] = desc
                region_groups[key].append((idx, inst))

            # Assign descriptors per region.
            for key, sub in region_groups.items():
                if len(sub) == 1:
                    idx, inst = sub[0]
                    instances[idx]["relative_desc"] = [inst["region_desc"]]
                    continue

                # Multiple in same region: add ordinal along a dominant axis.
                # Axis choice is based on which dimension has larger separation among instances.
                xs = [float(inst["center"][0]) for _, inst in sub]
                ys = [float(inst["center"][1]) for _, inst in sub]

                def _max_gap(vals: list[float]) -> float:
                    if len(vals) < 2:
                        return 0.0
                    v = sorted(vals)
                    return float(max(v[i + 1] - v[i] for i in range(len(v) - 1)))

                x_range = (max(xs) - min(xs)) / max(1.0, float(W))
                y_range = (max(ys) - min(ys)) / max(1.0, float(H))
                x_gap = _max_gap(xs) / max(1.0, float(W))
                y_gap = _max_gap(ys) / max(1.0, float(H))

                # Prefer the axis with the larger max-gap; fall back to total range.
                use_x = (x_gap > y_gap) or (abs(x_gap - y_gap) < 1e-9 and x_range >= y_range)

                if use_x:
                    sub_sorted = sorted(sub, key=lambda t: (t[1]["center"][0], t[1]["center"][1]))
                    pos_suffix = "from the left"
                    neg_suffix = "from the right"
                else:
                    sub_sorted = sorted(sub, key=lambda t: (t[1]["center"][1], t[1]["center"][0]))
                    pos_suffix = "from the top"
                    neg_suffix = "from the bottom"

                n = len(sub_sorted)
                for j, (idx, inst) in enumerate(sub_sorted, start=1):
                    # Choose a direction that yields a smaller ordinal when possible.
                    # Example (n=4): left->right rank=3 becomes "second from the right".
                    j_from_pos = j
                    j_from_neg = n - j + 1
                    if j_from_pos <= j_from_neg:
                        ord_str = _ordinal_word(j_from_pos)
                        suffix = pos_suffix
                    else:
                        ord_str = _ordinal_word(j_from_neg)
                        suffix = neg_suffix
                    instances[idx]["relative_desc"] = [f"{inst['region_desc']}, {ord_str} {suffix}"]

        return instances
