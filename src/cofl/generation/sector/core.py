"""Episode replay and semantic augmentation, usable in a Python 3.7 simulator worker.

The sink receives complete replay observations and aligned generated slots.
It controls storage; this module never imports the native dataset stack.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from copy import deepcopy

import numpy as np

from .augmentation.baseline_labels import compute_frame_baseline_labels
from .augmentation.codecs import encode_rgb_jpeg
from .augmentation.fgr2r_loader import FGR2RLoader
from .augmentation.flow_field_generator import FlowFieldGenerator
from .augmentation.gt_actions_loader import GtActionsLoader
from .augmentation.landmark_rxr_loader import LandmarkRxRLoader
from .augmentation.processing.anchor_factory import fetch_anchor_context
from .augmentation.processing.frame_processor import process_frame
from .augmentation.processing.oracle import collect_follower_replay, collect_gt_replay
from .augmentation.processing.slot_filler import SlotFiller
from .augmentation.processing.utils import _quat_xyzw_to_heading
from .augmentation.semantic_anchor.finders import VisibleObjectFinder, VisibleRegionFinder
from .augmentation.semantic_anchor.sampler import AlternativeSampler
from .defaults import DEFAULT_AUGMENTATION


def merge_augmentation(overrides):
    from .augmentation.semantic_anchor.types import _DEFAULT_FINDER_CFG, _DEFAULT_SAMPLER_CFG

    result = deepcopy(DEFAULT_AUGMENTATION)
    result["semantic_anchor"] = {**deepcopy(_DEFAULT_FINDER_CFG), **result["semantic_anchor"]}
    result["alternative_sampler"] = {
        **deepcopy(_DEFAULT_SAMPLER_CFG),
        **result["alternative_sampler"],
    }

    def merge(destination, values):
        for key, value in values.items():
            if key not in destination:
                raise ValueError("Unknown augmentation parameter: " + key)
            if isinstance(destination[key], bool) and not isinstance(value, bool):
                raise ValueError("Augmentation parameter must be a boolean: " + key)
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = deepcopy(value)

    merge(result, overrides or {})
    return result


def episode_identity(episode):
    """Match a serialized episode and its Habitat object without source-path prefixes."""
    from pathlib import Path

    if isinstance(episode, dict):
        get = episode.get
    else:

        def get(key, default=None):
            return getattr(episode, key, default)

    instruction = get("instruction", {})
    iid = (
        instruction.get("instruction_id", "")
        if isinstance(instruction, dict)
        else getattr(instruction, "instruction_id", "")
    )
    return (Path(str(get("scene_id", ""))).stem, str(get("episode_id", "")), str(iid))


class SectorCore:
    """Stateful simulator/semantic collaborators; all random choices use a unit seed."""

    def __init__(self, options, env=None):
        self.options = options
        self.cfg = merge_augmentation(options.get("augmentation"))
        self.env = env
        self.owns_env = env is None
        self._episode_lookup = None
        self.mode = options.get("replay_mode", "gt")
        if self.mode not in ("gt", "follower"):
            raise ValueError("replay_mode must be gt or follower")
        alignment = options["subinstructions"]
        if alignment["kind"] == "fgr2r":
            self.alignment = FGR2RLoader(alignment["path"])
        elif alignment["kind"] == "landmark_rxr":
            self.alignment = LandmarkRxRLoader(alignment["path"])
        else:
            raise ValueError("subinstructions.kind must be fgr2r or landmark_rxr")
        self.alignment_kind = alignment["kind"]
        self.gt = GtActionsLoader(options["gt_actions_path"]) if self.mode == "gt" else None
        self.hfov = float(self.cfg["hfov_rad"])
        self.extent = float(self.cfg["bev_x_max"])
        self.resolution = float(self.cfg["bev_resolution"])
        self.v_norm = float(self.cfg["v_norm"])
        if self.v_norm != self.extent:
            raise ValueError("Native ground-sector profile requires v_norm == bev_x_max")
        if not 0 < self.hfov < math.pi or self.extent <= 0 or self.resolution <= 0:
            raise ValueError("Invalid sector geometry")
        steps = self.extent / self.resolution
        if not np.isclose(steps, round(steps)):
            raise ValueError("bev_x_max must be an integer multiple of bev_resolution")
        self.grid_shape = (int(round(steps)) + 1, 2 * int(round(steps)) + 1)
        builder = self.cfg["dataset_builder"]
        self.max_steps = int(builder["max_steps_per_ep"])
        self.stride = int(builder["frame_stride"])
        labels = builder["baseline_labels"]
        self.horizon = int(labels["horizon"])
        if min(self.max_steps, self.stride, self.horizon) < 1:
            raise ValueError("Step, stride and trajectory horizon must be positive")
        self.finder = VisibleObjectFinder(cfg=self.cfg["semantic_anchor"])
        self.region_finder = VisibleRegionFinder(cfg=self.cfg["semantic_anchor"])
        sampler_cfg = dict(self.cfg["alternative_sampler"])
        sampler_cfg["region_fov_rad"] = self.hfov
        self.sampler = AlternativeSampler(cfg=sampler_cfg)
        flow_cfg = dict(self.cfg["flow_field_generator"])
        flow_cfg.update(
            hfov_rad=self.hfov,
            bev_x_max=self.extent,
            bev_resolution=self.resolution,
            v_norm=self.v_norm,
            horizon=self.horizon,
        )
        self.flow = FlowFieldGenerator(cfg=flow_cfg)
        self.filler = SlotFiller(
            flow=self.flow,
            stop_radius_m=float(self.cfg["stop_radius_m"]),
            sampler_cfg=self.cfg["alternative_sampler"],
        )

    def _environment(self):
        if self.env is None:
            from .simulator.environment import build_habitat_env

            self.env = build_habitat_env(
                self.options["habitat_config"],
                self.options["split"],
                dataset_data_path=self.options["dataset_data_path"],
                scenes_dir=self.options["scenes_dir"],
                hfov_deg=math.degrees(self.hfov),
            )
        return self.env

    def _lookup_alignment(self, episode):
        instruction = getattr(episode, "instruction", None)
        if self.alignment_kind == "fgr2r":
            trajectory_id = getattr(episode, "trajectory_id", None)
            text = getattr(instruction, "instruction_text", None)
            return (
                self.alignment.lookup(trajectory_id, text)
                if trajectory_id is not None and text
                else None
            )
        iid = getattr(instruction, "instruction_id", getattr(episode, "instruction_id", None))
        return self.alignment.lookup(iid) if iid is not None else None

    def generate(self, payload, *, seed, sink):
        random.seed(int(seed))
        np.random.seed(int(seed) % 2**32)
        rng = np.random.default_rng(int(seed))
        env = self._environment()
        if self._episode_lookup is None:
            self._episode_lookup = {episode_identity(ep): ep for ep in env.episodes}
        identity = episode_identity(payload)
        if identity not in self._episode_lookup:
            raise ValueError("Episode inventory does not match the configured Habitat dataset")
        episode = self._episode_lookup[identity]
        info = self._lookup_alignment(episode)
        reference = getattr(episode, "reference_path", None)
        if info is None or reference is None or len(reference) == 0:
            return {
                "status": "skipped",
                "reason": "missing_subinstruction_alignment_or_reference_path",
                "observations": 0,
                "annotations": 0,
            }
        actions = self.gt.lookup(str(episode.episode_id)) if self.gt is not None else None
        if self.mode == "gt":
            if not actions:
                raise ValueError(
                    "Missing required GT actions; choose follower replay explicitly for a different recipe"
                )
            if (
                len(actions) > self.max_steps
                or int(actions[-1]) != 0
                or any((int(a) not in (0, 1, 2, 3) for a in actions))
            ):
                raise ValueError(
                    "GT actions must fit max_steps, use the four recorded actions and terminate with STOP"
                )
            if 0 in [int(a) for a in actions[:-1]]:
                raise ValueError("GT action sequence contains a premature STOP")
        env.seed(int(seed) % 2**32)
        env._current_episode = None
        env._episodes = [episode]
        env._episode_iterator = iter([episode])
        observations = env.reset()
        if self.mode == "gt":
            trajectory = collect_gt_replay(env, observations, actions)
        else:
            trajectory = collect_follower_replay(env, episode, observations, self.max_steps)
        if self.mode == "gt" and [int(frame[5]) for frame in trajectory] != [
            int(a) for a in actions
        ]:
            raise ValueError("GT replay ended early or changed the executed action sequence")
        if not trajectory:
            raise ValueError("Replay produced no observations")
        if self.mode == "follower" and int(trajectory[-1][5]) != 0:
            raise ValueError("Follower replay failed to terminate with STOP")
        if any((frame[2] is None or frame[3] is None or frame[4] is None for frame in trajectory)):
            raise ValueError(
                "Replay requires RGB, metric depth and semantic observations at every frame"
            )
        sim = getattr(env, "sim", env)
        scene = getattr(sim, "semantic_scene", None)
        pathfinder = getattr(sim, "pathfinder", None)
        if scene is None or pathfinder is None:
            raise ValueError("Sector generation requires a semantic scene and navmesh pathfinder")
        positions = np.stack([np.asarray(frame[0], dtype=np.float64) for frame in trajectory])
        reference = np.asarray(reference, dtype=np.float32)
        sink.begin_episode(
            {
                "source_episode": payload,
                "replay_mode": self.mode,
                "seed": int(seed),
                "replay_frames": len(trajectory),
                "augmentation": self.cfg,
            }
        )
        (families, statuses) = (Counter(), Counter())
        annotations = 0
        for t, frame in enumerate(trajectory):
            anchor = None
            if t % self.stride == 0:
                anchor = fetch_anchor_context(
                    info,
                    reference,
                    positions[t],
                    scene,
                    agent_future_positions=positions[t + 1 :],
                )
            result = None
            status = "stride_excluded" if t % self.stride else "missing_frame_alignment"
            if anchor is not None:
                result = process_frame(
                    t=t,
                    trajectory=trajectory,
                    anchor_ctx=anchor,
                    semantic_scene=scene,
                    pathfinder=pathfinder,
                    finder=self.finder,
                    region_finder=self.region_finder,
                    sampler=self.sampler,
                    flow=self.flow,
                    slot_filler=self.filler,
                    bev_x_max=self.extent,
                    H_bev=self.grid_shape[0],
                    W_bev=self.grid_shape[1],
                    baseline_horizon=self.horizon,
                    rng=rng,
                )
                status = (
                    "labeled"
                    if result is not None and np.any(result["slot_valid"])
                    else "no_valid_generated_slots"
                )
            if result is None:
                (k, h, w) = (int(self.sampler.K), *self.grid_shape)
                result = {
                    "frame_index": t,
                    "rgb_blob": encode_rgb_jpeg(frame[2]),
                    "depth": np.asarray(frame[3], dtype=np.float16),
                    "position": np.asarray(frame[0], dtype=np.float32),
                    "heading": _quat_xyzw_to_heading(frame[1]),
                    "action_id": int(frame[5]),
                    "v_K": np.zeros((k, 2, h, w), np.float16),
                    "dp_traj_K": np.zeros((k, self.horizon + 1, 2), np.float32),
                    "path_len_K": np.zeros(k, np.float32),
                    "family_K": [None] * k,
                    "bev_mask": np.zeros((h, w), np.uint8),
                    "bev_walkable": np.zeros((h, w), np.uint8),
                    "slot_valid": np.zeros(k, np.uint8),
                    "texts": [""] * k,
                    "cmd_records": [None] * k,
                    "current_region": None,
                    "visible_objects": [],
                    "visible_regions": [],
                }
            baseline = compute_frame_baseline_labels(
                dp_traj_cart_K=result["dp_traj_K"],
                path_length_m_K=result["path_len_K"],
                slot_valid_K=result["slot_valid"],
                family_K=result["family_K"],
            )
            result.update(
                slot_actions=baseline["action_id"],
                trajectory_delta=baseline["dp_traj_cart_delta"],
                rotation_xyzw=np.asarray(frame[1], np.float32),
                semantic_labels=np.asarray(frame[4], np.int32),
                label_status=status,
            )
            statuses[status] += 1
            for k in np.flatnonzero(result["slot_valid"]):
                annotations += 1
                families[str(result["family_K"][k] or "anchor")] += 1
            sink.add_frame(result)
        return {
            "status": "generated" if annotations else "skipped",
            "reason": None if annotations else "no_aligned_valid_slots",
            "observations": len(trajectory),
            "annotations": annotations,
            "frame_status_counts": dict(statuses),
            "family_counts": dict(families),
            "replay_mode": self.mode,
            "terminal_action": int(trajectory[-1][5]),
        }

    def close(self):
        if self.env is not None and self.owns_env:
            self.env.close()
        self.env = None
