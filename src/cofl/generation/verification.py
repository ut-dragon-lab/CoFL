"""Verify generated outputs and join complete distributed generation partitions."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from cofl.data._io import file_sha256
from cofl.data.collection import _local_child, publish_collection

from .io import digest_json, output_lock, write_json
from .runner import _read_json, _verify_shard


def verify_generation(output, *, full=True):
    """Check generation receipts; full mode also rehashes every stored payload.

    These checks establish integrity of generated output, not equality to a
    reference dataset. Reproduction against a reference uses data compare.
    """
    output = Path(output).resolve()
    if (output / "generation-merge.json").is_file():
        merge = _read_json(output / "generation-merge.json")
        for child in merge["children"]:
            path = _local_child(output, child["path"])
            if file_sha256(path / "generation.json") != child["journal_sha256"]:
                raise ValueError("Merged generation journal changed")
            verify_generation(path, full=full)
        if file_sha256(output / "collection.json") != merge["collection_sha256"]:
            raise ValueError("Merged generation collection changed")
        return {"status": "passed", "full_payload_check": full, "counts": merge["counts"]}
    state = _read_json(output / "generation.json")
    if state.get("format") != "cofl_generation" or state.get("status") != "completed":
        raise ValueError("A completed generation journal is required")
    definition = state["definition"]
    if digest_json(definition) != state["definition_sha256"]:
        raise ValueError("Generation definition hash differs from its recorded value")
    if set(state["units"]) != set(definition["units"]):
        raise ValueError("Generation journal does not cover the selected source units")
    if state.get("empty_partition"):
        if definition["config"]["num_shards"] <= 1 or (output / "collection.json").exists():
            raise ValueError("An empty partition cannot advertise a native dataset collection")
        for item in state["units"].values():
            if item["status"] != "skipped" or not item["summary"].get("reason"):
                raise ValueError("Empty partition contains unfinished or unexplained source units")
        return {
            "status": "passed",
            "full_payload_check": full,
            "counts": {"episodes": 0, "observations": 0, "annotations": 0},
        }
    manifest = _read_json(output / "collection.json")
    if not manifest.get("complete"):
        raise ValueError("Generation collection is incomplete")
    for key in ("dataset_id", "revision"):
        if manifest[key] != definition["config"][key]:
            raise ValueError("Generation collection identity differs from the recipe")
    if manifest["recipe"] != definition["recipe"] or manifest["profile"] != definition["profile"]:
        raise ValueError("Generation collection recipe or profile changed")
    source = definition["config"]["source"]
    partial = bool(source.get("partial", False)) or len(definition["units"]) < len(
        definition["available_units"]
    )
    if manifest["source"] != {**source, "partial": partial}:
        raise ValueError("Generation collection source identity or partial status changed")
    entries = {entry["path"]: entry for entry in manifest["shards"]}
    completed = []
    totals = {"episodes": 0, "observations": 0, "annotations": 0}
    for key in definition["units"]:
        item = state["units"][key]
        if item["status"] == "skipped":
            if not item["summary"].get("reason"):
                raise ValueError("Skipped generation unit has no reason")
            continue
        if item["status"] != "completed":
            raise ValueError("Generation contains an unfinished unit")
        path = _local_child(output, item["path"])
        if file_sha256(path / "generation-report.json") != item["report_sha256"]:
            raise ValueError("Generation unit report changed")
        report = (
            _verify_shard(path, item["report_sha256"])
            if full
            else _read_json(path / "generation-report.json")
        )
        if report["unit_sha256"] != item["unit_sha256"] or report["key"] != key:
            raise ValueError("Generation report belongs to a different source unit")
        if report["definition_sha256"] != state["definition_sha256"]:
            raise ValueError("Generation unit used a different recipe")
        entry = entries.get(item["path"])
        if entry is None or file_sha256(path / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("Generation shard manifest changed or is absent from the collection")
        for count in totals:
            if report["counts"][count] != entry["counts"][count]:
                raise ValueError("Generation report and collection counts differ")
            totals[count] += report["counts"][count]
        completed.append(item["path"])
    if completed != [entry["path"] for entry in manifest["shards"]] or totals != manifest["counts"]:
        raise ValueError("Generation collection order or totals differ from its journal")
    return {"status": "passed", "full_payload_check": full, "counts": totals}


def merge_generation(inputs, output):
    """Publish a complete collection only after verifying every process shard.

    Worker outputs must already live below output. This indexes immutable
    child directories without copying payloads or using external symlinks.
    """
    output = Path(output).resolve()
    children = [Path(path).resolve() for path in inputs]
    if not children or len(set(children)) != len(children):
        raise ValueError("Provide distinct generation partition directories")
    for child in children:
        if child == output or not child.is_relative_to(output):
            raise ValueError("Each generation partition must live below the merge output directory")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output), ExitStack() as locks:
        for child in sorted(children):
            locks.enter_context(output_lock(child))
        states = []
        for child in children:
            verify_generation(child)
            state = _read_json(child / "generation.json")
            states.append((state["definition"]["config"]["shard_index"], child, state))
        states.sort(key=lambda item: item[0])
        first = states[0][2]["definition"]
        config = first["config"]
        count = config["num_shards"]
        if [index for index, _, _ in states] != list(range(count)):
            raise ValueError("A complete set of process shard indices is required")
        normalized = {key: value for key, value in config.items() if key != "shard_index"}
        available = first["available_units"]
        limit = config["limit_units"]
        selected = available[:limit] if limit is not None else available
        source_fingerprints = {}
        unit_paths = {}
        for index, _, state in states:
            definition = state["definition"]
            other = {
                key: value for key, value in definition["config"].items() if key != "shard_index"
            }
            if other != normalized or definition["recipe"] != first["recipe"]:
                raise ValueError("Generation partitions used different recipes")
            if (
                definition["available_units"] != available
                or definition["units"] != selected[index::count]
            ):
                raise ValueError("Generation partitions do not cover the same source inventory")
        for _, path, state in states:
            for key, unit in state["units"].items():
                if unit["status"] == "completed":
                    native = _local_child(path, unit["path"])
                    unit_paths[key] = native
                    inputs = _read_json(native / "generation-report.json")["inputs"]
                else:
                    inputs = unit["inputs"]
                for item in inputs:
                    previous = source_fingerprints.setdefault(item["id"], item)
                    if previous != item:
                        raise ValueError(
                            "A shared source input changed between generation partitions"
                        )
        existing = output / "generation-merge.json"
        receipts = [
            {
                "path": path.relative_to(output).as_posix(),
                "journal_sha256": file_sha256(path / "generation.json"),
            }
            for _, path, _ in states
        ]
        if existing.exists() and _read_json(existing)["children"] != receipts:
            raise ValueError("Existing merged generation has different inputs")
        if (output / "collection.json").exists() and not existing.exists():
            raise ValueError("Refusing to replace an unrelated collection")
        partial = bool(config["source"].get("partial", False)) or len(selected) < len(available)
        publish_collection(
            output,
            [unit_paths[key] for key in selected if key in unit_paths],
            profile=first["profile"],
            dataset_id=config["dataset_id"],
            revision=config["revision"],
            source={**config["source"], "partial": partial},
            recipe=first["recipe"],
        )
        manifest_path = output / "collection.json"
        manifest = _read_json(manifest_path)
        # Each child is a partial process partition. The checks above prove
        # their union covers the recipe's inventory; only here can it become
        # a complete corpus. Explicit source/pilot restrictions remain partial.
        manifest["source"]["partial"] = partial
        write_json(manifest_path, manifest)
        write_json(
            existing,
            {
                "format": "cofl_generation_merge",
                "version": "1",
                "children": receipts,
                "collection_sha256": file_sha256(manifest_path),
                "counts": manifest["counts"],
                "partial": partial,
            },
        )
        return {
            "path": str(output),
            "status": "completed",
            "partial": partial,
            "counts": manifest["counts"],
            "partitions": count,
        }
