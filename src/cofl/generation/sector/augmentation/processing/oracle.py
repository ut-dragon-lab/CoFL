"""Explicit GT and greedy-follower replay, recording sensors before each action."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ...simulator.habitat_utils import get_pose, obs_get_rgb_depth_semantic
from ...simulator.oracle import ACTION_NAMES, create_geodesic_follower
from .utils import _TARGET_H, _TARGET_W, _resize_depth_with_holes_224, _resize_rgb_224


def _capture_frame(env, obs):
    """Return pose, RGB, metric depth and semantic labels; sensor failures raise."""
    pos, rotation = get_pose(env)
    rgb, depth, semantic = obs_get_rgb_depth_semantic(obs)
    if semantic.shape != (_TARGET_H, _TARGET_W):
        semantic = np.asarray(
            Image.fromarray(semantic.astype(np.int32), mode="I").resize(
                (_TARGET_W, _TARGET_H), Image.NEAREST
            )
        )
    return (
        pos,
        rotation,
        _resize_rgb_224(rgb),
        _resize_depth_with_holes_224(depth),
        semantic.astype(np.int64),
    )


def collect_gt_replay(env, initial_obs, actions):
    """Replay a complete four-action sequence ending in one STOP.

    Every frame records the action executed from that pose. A simulator or
    sensor failure aborts the unit; it cannot publish a truncated episode.
    """
    if not actions or actions[-1] != 0 or any(action not in range(4) for action in actions):
        raise ValueError("GT actions must use IDs 0..3 and end with STOP")
    if 0 in actions[:-1]:
        raise ValueError("GT actions contain a premature STOP")
    trajectory = []
    obs = initial_obs
    for index, action in enumerate(actions):
        frame = _capture_frame(env, obs)
        trajectory.append((*frame, int(action)))
        if action == 0:
            return trajectory
        obs = env.step({"action": ACTION_NAMES[action]})
        if env.episode_over:
            raise RuntimeError("GT replay ended early after action {}".format(index))
    raise RuntimeError("GT replay did not reach its terminal action")


def collect_follower_replay(env, episode, initial_obs, max_steps):
    """Follow the reference waypoints with Habitat's greedy geodesic follower.

    Intermediate waypoints are passed within 0.5 m; the final goal uses
    the episode's arrival radius. Follower failures and step limits raise,
    rather than inserting an arbitrary navigation action or partial trace.
    """
    if not episode.goals:
        raise ValueError("Follower replay requires an episode goal")
    goal = np.asarray(episode.goals[0].position, dtype=np.float32).reshape(3)
    radius = episode.goals[0].radius
    radius = 0.5 if radius is None else float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Follower goal radius must be positive and finite")
    follower = create_geodesic_follower(env, radius)
    waypoints = [np.asarray(point, dtype=np.float32) for point in episode.reference_path]
    waypoints.append(goal)
    waypoint_index = 0
    trajectory = []
    obs = initial_obs
    for index in range(int(max_steps)):
        frame = _capture_frame(env, obs)
        while (
            waypoint_index < len(waypoints) - 1
            and np.linalg.norm(frame[0] - waypoints[waypoint_index]) <= 0.5
        ):
            waypoint_index += 1
        action = follower.next_action_along(waypoints[waypoint_index])
        # A wide final-goal radius may also satisfy an intermediate waypoint.
        while action == "STOP" and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            action = follower.next_action_along(waypoints[waypoint_index])
        if action not in ACTION_NAMES:
            raise ValueError("Follower returned an unsupported action: {!r}".format(action))
        trajectory.append((*frame, ACTION_NAMES.index(action)))
        if action == "STOP":
            return trajectory
        obs = env.step({"action": action})
        if env.episode_over:
            raise RuntimeError("Follower replay ended before STOP after action {}".format(index))
    raise RuntimeError("Follower replay exceeded max_steps without STOP")
