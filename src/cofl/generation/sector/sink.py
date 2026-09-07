"""Write replay observations and sparse alternative targets directly as native data."""

from __future__ import annotations

import numpy as np

from cofl.data.profiles import geometry_for_profile
from cofl.fields import query_valid_mask
from cofl.generation.metadata import portable_metadata

from .qa import validate_frame


class NativeSink:
    def __init__(self, writer, *, key, split, extent, hfov, grid_shape):
        self.writer, self.key, self.split = writer, key, split
        self.geometry = geometry_for_profile(
            "ground_sector_v1", grid_shape, normalization_scale_m=extent, hfov_rad=hfov
        )
        self.mask = query_valid_mask("ground_sector_v1", self.geometry)
        self.episode_id = None
        self.frames = 0
        self.qa = {"valid_slots": 0, "ignored_action_slots": 0}

    def begin_episode(self, metadata):
        instruction = metadata["source_episode"].get("instruction", {})
        task = (
            instruction.get("instruction_text", "")
            if isinstance(instruction, dict)
            else instruction
        )
        if not str(task).strip():
            raise ValueError("Source replay episode requires its original instruction")
        self.episode_id = self.writer.add_episode(
            self.key, split=self.split, task=task, metadata=portable_metadata(metadata)
        )

    def add_frame(self, frame):
        if self.episode_id is None or int(frame["frame_index"]) != self.frames:
            raise ValueError("Replay observations must be complete and contiguous from frame zero")
        counts = validate_frame(frame, self.geometry["grid_shape"])
        for key, count in counts.items():
            self.qa[key] += count
        actions = np.asarray(frame["slot_actions"])
        anchor = int(actions[0])
        # The executed action advances the actual replay; generated alternatives
        # never create a temporal transition or replace that action.
        metadata = {
            name: frame[name]
            for name in (
                "label_status",
                "current_region",
                "visible_objects",
                "visible_regions",
                "texts",
                "cmd_records",
                "family_K",
            )
        }
        metadata.update(
            anchor_action=None if anchor == -100 else anchor,
            raw_anchor_action=anchor,
            slot_valid=np.asarray(frame["slot_valid"]).tolist(),
            slot_actions=actions.tolist(),
        )
        if "generation_diagnostics" in frame:
            metadata["generation_diagnostics"] = frame["generation_diagnostics"]
        depth = np.asarray(frame["depth"], np.float16)
        obs = self.writer.add_observation(
            self.episode_id,
            frame_index=self.frames,
            image=frame["rgb_blob"],
            depth=depth,
            depth_valid=depth > 0,
            executed_action=int(frame["action_id"]),
            metadata=portable_metadata(metadata),
            extras={
                "position_world": np.asarray(frame["position"], np.float32),
                "rotation_xyzw": np.asarray(frame["rotation_xyzw"], np.float32),
                "heading_rad": np.asarray(frame["heading"], np.float32),
                "semantic_labels": np.asarray(frame["semantic_labels"], np.int32),
                "bev_mask": np.asarray(frame["bev_mask"], np.uint8),
                "bev_walkable": np.asarray(frame["bev_walkable"], np.uint8),
            },
        )
        for slot in np.flatnonzero(frame["slot_valid"]):
            action = int(actions[slot])
            self.writer.add_annotation(
                obs,
                annotation_key="slot:" + str(slot),
                instruction=frame["texts"][slot],
                field=np.asarray(frame["v_K"][slot], np.float16),
                mask=self.mask,
                geometry=self.geometry,
                target_action=None if action == -100 else action,
                is_anchor=bool(slot == 0),
                instruction_source="source_subinstruction" if slot == 0 else "semantic_template",
                metadata=portable_metadata(
                    {
                        "slot_index": int(slot),
                        "family": frame["family_K"][slot],
                        "command": frame["cmd_records"][slot],
                        "training_eligible": action != -100,
                        "raw_slot_action": action,
                    }
                ),
                extras={
                    "trajectory_body_m": np.asarray(frame["dp_traj_K"][slot], np.float32),
                    "trajectory_delta_body_m": np.asarray(
                        frame["trajectory_delta"][slot], np.float32
                    ),
                    "path_length_m": np.asarray(frame["path_len_K"][slot], np.float32),
                },
            )
        self.frames += 1
