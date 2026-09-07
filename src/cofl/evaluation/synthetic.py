"""Deterministic, goal-assisted ground execution demo with an analytic field.

Run ``python -m cofl.evaluation.synthetic --output results/synthetic``.
No trained policy, simulator assets, obstacle scoring or paper benchmark is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cofl.runtime import PlanarState, PurePursuitController, integrate_unicycle
from .metrics import evaluate_ground_trajectory
from .protocol import EvaluationProtocol, ResultWriter, TimingSpec


def _plan(position: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Euler-integrate an analytic attractive field in metric Cartesian space."""
    points = [position.copy()]
    for _ in range(30):
        delta = goal - points[-1]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.01:
            break
        velocity = delta / max(distance, 1.0)
        points.append(points[-1] + 0.15 * velocity)
    # The known goal is explicitly provided in this synthetic assisted protocol.
    points.append(goal.copy())
    return np.asarray(points)


def run_synthetic_benchmark(output_dir: str | Path, *, seed: int = 0) -> dict[str, Any]:
    """Run three unobstructed unicycle episodes, write traces and return summary."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    goals = np.array([[3.0, 0.0], [2.0, 2.0], [-2.0, 1.0]], dtype=np.float64)
    recipe = {
        "name": "synthetic_ground_pursuit_v1",
        "goals_m": goals.tolist(),
        "initial_xy_m": [0.0, 0.0],
        "initial_heading_distribution": "uniform(-pi,pi)",
        "seed": seed,
        "plant_hz": 20.0,
        "controller_hz": 20.0,
        "planner_hz": 5.0,
        "max_time_s": 30.0,
        "planner": "analytic_goal_attraction_euler",
        "rollout_steps": 30,
        "rollout_dt": 0.15,
        "rollout_stop_distance_m": 0.01,
        "known_goal_appended": True,
        "controller": asdict(PurePursuitController()),
    }
    recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    protocol = EvaluationProtocol(
        name="synthetic_ground_pursuit",
        profile="ground_sector_v1",
        split="synthetic",
        seed=seed,
        recipe=recipe["name"],
        task="synthetic_ground",
        instruction_mode="goal_assisted",
        instruction_source="explicit_synthetic_goal_coordinates",
        timing=TimingSpec(
            execution="fixed_step", planner_hz=5.0, controller_hz=20.0, plant_hz=20.0
        ),
        recipe_sha256=hashlib.sha256(recipe_json.encode()).hexdigest(),
    )
    writer = ResultWriter(output_dir, protocol)
    if writer.completed_sample_ids:
        raise ValueError("Synthetic demo needs an output directory without existing results")
    controller = PurePursuitController()
    rng = np.random.default_rng(seed)
    episodes = []
    for index, goal in enumerate(goals):
        state = PlanarState(0.0, 0.0, float(rng.uniform(-math.pi, math.pi)))
        states = [[state.x_m, state.y_m, state.heading_rad]]
        commands, timestamps = [], [0.0]
        reference = _plan(state.position, goal)
        for tick in range(600):
            if np.linalg.norm(state.position - goal) <= controller.goal_radius_m:
                break
            if tick % 4 == 0:
                reference = _plan(state.position, goal)
            command = controller.compute(state, reference)
            state = integrate_unicycle(state, command, 0.05)
            states.append([state.x_m, state.y_m, state.heading_rad])
            timestamps.append((tick + 1) * 0.05)
            commands.append([command.linear_mps, command.angular_radps])
        path = np.asarray(states)[:, :2]
        metrics = evaluate_ground_trajectory(
            path, np.vstack([[0.0, 0.0], goal]), success_radius_m=controller.goal_radius_m
        )
        metrics["duration_s"] = timestamps[-1]
        metrics["stop_reason"] = "goal_radius" if metrics["euclidean_goal_reached"] else "timeout"
        episodes.append(metrics)
        writer.write(
            f"synthetic_{index:03d}",
            metrics,
            diagnostics={
                "goal_m": goal.tolist(),
                "state_columns": ["x_m", "y_m", "heading_rad"],
                "states": states,
                "timestamps_s": timestamps,
                "command_columns": ["linear_mps", "angular_radps"],
                "commands": commands,
            },
        )
    summary = {
        "benchmark": "synthetic_ground_pursuit",
        "synthetic": True,
        "trained_policy": False,
        "episode_count": len(episodes),
        "euclidean_goal_reach_fraction": float(
            np.mean([row["euclidean_goal_reached"] for row in episodes])
        ),
        "final_goal_error_m_mean": float(np.mean([row["final_goal_error"] for row in episodes])),
        "recipe": recipe,
    }
    writer.write_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_synthetic_benchmark(args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
