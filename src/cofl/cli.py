"""Command-line entry points, with optional dependencies imported on demand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cofl import __version__

PROFILES = ("image_field_v1", "ground_sector_v1")


def _generation_templates():
    """Find package data in a wheel installation or an editable checkout."""
    from importlib.metadata import PackageNotFoundError, distribution

    required = {"image.yaml", "image_labels.yaml", "sector_r2r.yaml", "sector_rxr.yaml"}
    try:
        package = distribution("cofl-navigation")
        candidates = {
            Path(str(entry)).name: Path(package.locate_file(entry))
            for entry in package.files or ()
            if any(
                str(entry).replace("\\", "/").endswith(f"share/cofl/generation/{name}")
                for name in required
            )
        }
        # pip --target relocates data files below the target while its RECORD
        # can retain prefix-relative paths. Resolve that installed layout too.
        for name in required:
            path = candidates.get(name)
            if path is None or not path.is_file():
                candidates[name] = (
                    Path(package.locate_file("")) / "share" / "cofl" / "generation" / name
                )
        if set(candidates) == required and all(path.is_file() for path in candidates.values()):
            return candidates
    except PackageNotFoundError:
        pass
    root = Path(__file__).resolve().parents[2] / "configs" / "generation"
    candidates = {name: root / name for name in required}
    if all(path.is_file() for path in candidates.values()):
        return candidates
    raise FileNotFoundError("Bundled generation recipes are missing; reinstall cofl-navigation")


def _parser():
    parser = argparse.ArgumentParser(
        prog="cofl", description="CoFL and CoFL-S navigation research tools"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Build, inspect, and validate CoFLDataset v1")
    actions = data.add_subparsers(dest="data_command", required=True)
    templates = actions.add_parser(
        "init-generation", help="Copy editable generation recipes bundled with the package"
    )
    templates.add_argument("--output", required=True, type=Path)
    camera_export = actions.add_parser(
        "export-cameras", help="Collect historical scene annotations into one portable camera JSON"
    )
    camera_export.add_argument("--matterport-root", type=Path, help="Matterport rendered scene root")
    camera_export.add_argument("--scannet-root", type=Path, help="ScanNet rendered scene root")
    camera_export.add_argument("--output", required=True, type=Path)
    render = actions.add_parser(
        "render", help="Render Matterport3D or ScanNet meshes into image-generation sources"
    )
    render.add_argument("--dataset", required=True, choices=("matterport", "scannet"))
    render.add_argument("--root", required=True, type=Path, help="Official scans directory")
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--scan-ids", nargs="+", help="Selected scene IDs; default: all scenes")
    render.add_argument("--seed", type=int, default=42)
    camera_options = render.add_mutually_exclusive_group()
    camera_options.add_argument(
        "--cameras-from", type=Path,
        help="Replay a combined camera JSON or ROOT/SCAN/annotations.json directory",
    )
    camera_options.add_argument(
        "--bundled-cameras", action="store_true",
        help="Replay an optional local historical camera catalog (not included in the release)",
    )
    render.add_argument("--resume", action="store_true", help="Verify and reuse completed scenes")
    generate = actions.add_parser(
        "generate", help="Generate real observations and labels from a method recipe"
    )
    generate.add_argument(
        "--config", required=True, type=Path, help="GenerationConfig YAML or JSON"
    )
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument(
        "--resume", action="store_true", help="Verify and reuse finalized generation units"
    )
    for flag in ("seed", "limit-units", "num-shards", "shard-index"):
        generate.add_argument(f"--{flag}", type=int)
    merge = actions.add_parser(
        "merge-generation", help="Verify and index every process partition below one parent"
    )
    merge.add_argument("partitions", nargs="+", type=Path)
    merge.add_argument("--output", required=True, type=Path)
    verify = actions.add_parser(
        "verify-generation", help="Verify generated receipts and stored payload checksums"
    )
    verify.add_argument("dataset", type=Path)
    verify.add_argument(
        "--metadata-only", action="store_true", help="Skip rereading dense payload bytes"
    )
    for name in ("validate", "inspect"):
        sub = actions.add_parser(name)
        sub.add_argument("dataset", type=Path)
        sub.add_argument("--profile", choices=PROFILES)
    compare = actions.add_parser("compare", help="Compare complete dataset contents exactly")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, help="Save the comparison report as JSON")
    export = actions.add_parser(
        "export-lerobot", help="Export observation episodes using an installed LeRobot"
    )
    export.add_argument("dataset", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--repo-id", required=True)
    export.add_argument("--fps", type=int, default=10)
    hard = actions.add_parser(
        "select-hard", help="Select offline mag/dir error tails with temporal frame spacing"
    )
    hard.add_argument("--evaluation", required=True, type=Path)
    hard.add_argument("--dataset", required=True, type=Path)
    hard.add_argument("--output", required=True, type=Path)
    hard.add_argument("--top-fraction", type=float, default=0.1,
                      help="Top fraction per error metric before temporal suppression (default: 0.1)")
    hard.add_argument("--stride", type=int, default=10,
                      help="Minimum original-frame gap within each replay episode (default: 10)")
    hard.add_argument("--max-samples", type=int, help="Optional final annotation count cap")
    commands.add_parser(
        "train",
        help="Train with LightningCLI; use cofl train --help for its arguments",
        add_help=False,
    )
    commands.add_parser(
        "evaluate",
        help="Evaluate fields, image navigation and actions; use cofl evaluate --help",
        add_help=False,
    )
    commands.add_parser("app", help="Launch CoFL Studio; use cofl app --help", add_help=False)
    rollout = commands.add_parser(
        "rollout", help="Integrate a predicted field with explicit policy-time steps"
    )
    rollout.add_argument("--dataset", required=True, type=Path)
    rollout.add_argument("--checkpoint", required=True, type=Path)
    rollout.add_argument("--output", required=True, type=Path)
    rollout.add_argument("--split", default="val")
    rollout.add_argument("--sample-index", type=int, default=0)
    rollout.add_argument(
        "--start",
        type=float,
        nargs=2,
        help="Image x/y fractions for CoFL; body forward/left metres for CoFL-S",
    )
    rollout.add_argument("--max-steps", type=int, default=100)
    rollout.add_argument("--policy-dt", type=float, default=0.01)
    rollout.add_argument("--device", default="cpu")
    benchmark = commands.add_parser("benchmark", help="Run a named execution protocol")
    benchmark.add_argument("protocol", choices=("synthetic-ground",))
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "app":
        from cofl.app.cli import main as app_main

        return app_main(argv[1:])
    if argv and argv[0] == "train":
        from cofl.training.cli import main as training_main

        training_main(["fit", *argv[1:]])
        return 0
    if argv and argv[0] == "evaluate":
        from cofl.evaluation.cli import main as evaluation_main

        return evaluation_main(argv[1:])
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    if args.command == "data" and args.data_command == "compare":
        return 0 if result["semantic_equal"] and result["sample_order_equal"] else 1
    return 0


def _run(args):
    if args.command == "data":
        if args.data_command == "select-hard":
            from cofl.evaluation.hard_samples import mine_hard_samples

            return mine_hard_samples(
                evaluation=args.evaluation, dataset=args.dataset, output=args.output,
                top_fraction=args.top_fraction, stride=args.stride, max_samples=args.max_samples,
            )
        if args.data_command == "export-cameras":
            from cofl.generation.rendering.catalog import build_camera_catalog

            roots = {
                name: getattr(args, f"{name}_root") for name in ("matterport", "scannet")
                if getattr(args, f"{name}_root") is not None
            }
            return build_camera_catalog(roots, args.output)
        if args.data_command == "render":
            from cofl.generation.rendering.runner import render_sources

            return render_sources(
                args.dataset, args.root, args.output,
                scan_ids=args.scan_ids, seed=args.seed, resume=args.resume,
                cameras_from=args.cameras_from,
                **({"bundled_cameras": True} if args.bundled_cameras else {}),
            )
        if args.data_command == "init-generation":
            sources = _generation_templates()
            if args.output.exists() and any((args.output / name).exists() for name in sources):
                raise FileExistsError(
                    "Generation configuration already exists; choose another output directory"
                )
            contents = {name: path.read_bytes() for name, path in sources.items()}
            args.output.mkdir(parents=True, exist_ok=True)
            for name, content in contents.items():
                with (args.output / name).open("xb") as stream:
                    stream.write(content)
            return {"path": str(args.output), "files": sorted(contents)}
        if args.data_command == "merge-generation":
            from cofl.generation.verification import merge_generation

            return merge_generation(args.partitions, args.output)
        if args.data_command == "verify-generation":
            from cofl.generation.verification import verify_generation

            return verify_generation(args.dataset, full=not args.metadata_only)
        if args.data_command == "generate":
            import logging
            from cofl.generation import GenerationConfig, generate_dataset, load_mapping

            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
            values = load_mapping(args.config)
            for key in ("seed", "limit_units", "num_shards", "shard_index"):
                if getattr(args, key) is not None:
                    values[key] = getattr(args, key)
            return generate_dataset(
                GenerationConfig.from_mapping(values),
                args.output,
                base_dir=args.config.resolve().parent,
                resume=args.resume,
            )
        from cofl.data import open_dataset, validate_dataset, validate_collection

        if args.data_command == "validate":
            validate = (
                validate_collection
                if (args.dataset / "collection.json").is_file()
                else validate_dataset
            )
            return validate(args.dataset, expected_profile=args.profile)
        if args.data_command == "compare":
            from cofl.data.comparison import compare_datasets, write_comparison_report

            report = compare_datasets(args.reference, args.candidate)
            if args.output is not None:
                write_comparison_report(report, args.output)
            return report
        if args.data_command == "export-lerobot":
            from cofl.data.lerobot_bridge import export_lerobot

            return export_lerobot(args.dataset, args.output, repo_id=args.repo_id, fps=args.fps)
        dataset = open_dataset(args.dataset, expected_profile=args.profile)
        if not len(dataset):
            return {"samples": 0}
        sample = dataset[0]
        return {
            "samples": len(dataset),
            "profile": sample["profile"],
            "first_sample": {
                key: sample[key]
                for key in (
                    "sample_id",
                    "episode_id",
                    "observation_id",
                    "instruction",
                    "split",
                    "geometry",
                )
            },
            "image_shape": list(sample["image"].shape),
            "field_shape": list(sample["field"].shape),
        }
    if args.command == "rollout":
        from cofl.training.inference import rollout_sample

        return rollout_sample(
            args.dataset,
            args.checkpoint,
            args.output,
            split=args.split,
            sample_index=args.sample_index,
            start=args.start,
            max_steps=args.max_steps,
            policy_dt=args.policy_dt,
            device=args.device,
        )
    if args.command == "benchmark":
        from cofl.evaluation.synthetic import run_synthetic_benchmark

        return run_synthetic_benchmark(args.output, seed=args.seed)
    raise ValueError("Unknown command")
