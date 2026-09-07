"""Headless evaluation using the same protocol as Studio benchmark sessions."""

import argparse
import json
from time import perf_counter

from .cli import parse_args
from .config import (
    episode_ground_truth,
    ground_truth_paths,
    load_ground_truth,
    load_recipe,
    read_episodes,
    runtime_availability,
)
from .habitat import HabitatEnvironment
from .instructions import make_instruction_provider
from .policy import NativePolicy
from .runner import protocol, run_episode
from .store import ResultStore, file_identity


def selected_indices(args, total):
    if args.episode is not None:
        indices = list(dict.fromkeys(args.episode))
    else:
        indices = list(range(args.start, total, args.stride))
        if args.count is not None:
            indices = indices[: args.count]
    if not indices or any(not 0 <= index < total for index in indices):
        raise ValueError("Episode selection is empty or outside the benchmark")
    return indices


def group_indices_by_scene(indices, episodes):
    """Stable scene grouping after cohort selection; identities keep dataset indices."""
    groups = {}
    for index in indices:
        groups.setdefault(episodes[index]["scene_id"], []).append(index)
    return [index for group in groups.values() for index in group]


def main(argv=None):
    args = parse_args(argv)
    if getattr(args, "policy", None) is not None:
        from .external import run_evaluation

        return run_evaluation(args)
    parser = argparse.ArgumentParser(description="Evaluate native CoFL-S on fixed VLN episodes")
    recipe = load_recipe(args.benchmark)
    ready, reason = runtime_availability(recipe)
    if not ready:
        parser.error(reason)
    try:
        episodes = read_episodes(recipe)
        indices = selected_indices(args, len(episodes))
        ground_truth = load_ground_truth(recipe)
        for index in indices:
            episode_ground_truth(ground_truth, episodes[index])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    parameters = args.parameters
    mode = args.instruction_mode or recipe.instruction_mode
    recorded_protocol = protocol(parameters, ndtw_fdtw=recipe.ndtw_fdtw)
    recorded_protocol.update(
        instruction_mode=mode,
        uses_gt_progress=mode == "oracle",
        progress_lookahead=recipe.progress_lookahead,
    )
    if mode == "oracle" and recipe.oracle_source == "landmark_rxr":
        recorded_protocol["oracle_source"] = "landmark_rxr"
    if mode == "oracle" and recipe.oracle_unavailable != "error":
        # This run requests an oracle but may use full text for some episodes.
        # Each episode records its actual source and uses_gt_progress value.
        recorded_protocol.update(
            oracle_unavailable=recipe.oracle_unavailable,
            gt_progress_scope="per_episode",
        )
    manifest = {
        "protocol": recorded_protocol["protocol"],
        "parameters": recorded_protocol,
        "checkpoint_identity": file_identity(args.checkpoint),
        "recipe_identity": file_identity(recipe.path, hash_content=True),
        "dataset_identity": file_identity(recipe.dataset_data_path, hash_content=True),
        "instruction_sources": {
            name: file_identity(path, hash_content=True)
            for name in ("fgr2r_json", "landmark_rxr_json")
            if (path := getattr(recipe, name)) is not None
        },
        "device": args.device,
        "habitat_config_identity": file_identity(recipe.habitat_config, hash_content=True),
        "ground_truth_identity": [
            file_identity(path, hash_content=True) for path in ground_truth_paths(recipe)
        ],
    }
    with ResultStore(args.output, manifest=manifest, resume=args.resume) as store:
        indices = [i for i in indices if not store.is_completed(episodes[i], i)]
        if args.workers > 1:
            from .parallel import run_parallel

            execution = run_parallel(
                args=args,
                recipe=recipe,
                episodes=episodes,
                ground_truth=ground_truth,
                indices=indices,
                store=store,
            )
            if execution is not None:
                # Execution settings stay outside the frozen navigation protocol.
                # The run lock also serializes this append across resumed runs.
                with (store.output_dir / "executions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(execution, allow_nan=False) + "\n")
                print(json.dumps({"execution": execution}, allow_nan=False), flush=True)
        else:
            if args.group_by_scene:
                indices = group_indices_by_scene(indices, episodes)
            run_serial(
                args=args,
                recipe=recipe,
                episodes=episodes,
                ground_truth=ground_truth,
                indices=indices,
                store=store,
                mode=mode,
            )
        summary = store.summary()
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "episodes"},
                allow_nan=False,
            )
        )


def run_serial(*, args, recipe, episodes, ground_truth, indices, store, mode):
    """Preserve the single-process loop for existing runs and comparisons."""
    parameters = args.parameters
    policy = environment = None
    try:
        for index in indices:
            episode = episodes[index]
            writer = store.begin_episode(episode, index)
            try:
                provider = make_instruction_provider(recipe, episode, mode)
                if policy is None:
                    policy = NativePolicy(
                        args.checkpoint,
                        device=args.device,
                        inference_backend=args.inference_backend,
                    )
                if environment is None:
                    environment = HabitatEnvironment(recipe)
                started = perf_counter()
                result = run_episode(
                    environment,
                    policy,
                    episode,
                    parameters,
                    instruction_provider=provider,
                    gt_path=episode_ground_truth(ground_truth, episode),
                    ndtw_fdtw=recipe.ndtw_fdtw,
                    on_step=lambda step, writer=writer: writer.write_step(dict(step, world_path=[])),
                )
                result["execution"] = {
                    "episode_wall_s": perf_counter() - started,
                    "group_by_scene": args.group_by_scene,
                    "reuse_scene": args.reuse_scene,
                    "inference_backend_requested": args.inference_backend,
                    "inference_backend": getattr(policy, "active_inference_backend", "eager"),
                    "acceleration": getattr(policy, "acceleration_status", None),
                }
            except Exception as error:  # noqa: BLE001 — persist failed execution; storage failures propagate
                if environment is not None:
                    environment.close()
                    environment = None
                writer.fail(error)
                print(
                    json.dumps({"episode_index": index, "error": str(error)}, allow_nan=False)
                )
            else:
                writer.finish(result)
                print(
                    json.dumps(
                        {
                            "episode_index": index,
                            "episode_id": str(episode["episode_id"]),
                            "reason": result["reason"],
                            "metrics": result["metrics"],
                        },
                        allow_nan=False,
                    )
                )
            finally:
                if not args.reuse_scene and environment is not None:
                    environment.close()
                    environment = None
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            if policy is not None:
                policy.close()


if __name__ == "__main__":
    main()
