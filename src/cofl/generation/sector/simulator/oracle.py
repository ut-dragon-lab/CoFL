"""Explicit greedy-follower support for Habitat-Sim 0.1.7."""

from __future__ import annotations

ACTION_NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")


def create_geodesic_follower(env, goal_radius):
    """Use the installed follower API with the four recorded action names."""
    from habitat_sim.nav import GreedyGeodesicFollower

    return GreedyGeodesicFollower(
        env.sim.pathfinder,
        env.sim.get_agent(0),
        goal_radius=float(goal_radius),
        stop_key="STOP",
        forward_key="MOVE_FORWARD",
        left_key="TURN_LEFT",
        right_key="TURN_RIGHT",
    )
