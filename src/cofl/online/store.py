"""Atomic episode commits and resumable summaries, independent of verbosity.

Each committed episode directory contains its streamed steps and final result.
Summaries derive from all committed episodes; resume repairs interrupted summary
updates. Summaries retain metadata and metrics, with links to full episode results
instead of duplicating trajectories. An advisory lock permits one writer per run.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock, Timeout

_TRAJECTORY_FIELDS = frozenset(
    {
        "executed_path",
        "reference_path",
        "dense_path",
        "predicted_path_cart",
        "trajectory",
        "sector_trajectory",
        "world_trajectory",
        "world_path",
        "control_trajectory",
    }
)


def _summary_record(record, key):
    """Keep summaries and resident records independent of trajectory length.

    The committed episode.json is authoritative and always contains the complete
    result, including arrays omitted here. The relative link also works when a
    result directory is moved or copied elsewhere.
    """
    return {
        **record,
        "episode_file": f"episodes/{key}/episode.json",
        "result": {
            name: value
            for name, value in record["result"].items()
            if name not in _TRAJECTORY_FIELDS
        },
    }


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sync_directory(path):
    # Windows has no directory fsync; file flushes and atomic rename still apply.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_identity(path, hash_content=False):
    """Resolved path/stat identity, optionally with a content SHA256.

    Stat is inexpensive for large checkpoints; hashing is useful for recipes,
    datasets and instruction sources. Record the chosen identity in the manifest.
    """
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Identity requires a regular file: {path}")
    before = path.stat()
    identity = {"path": str(path), "size": before.st_size, "mtime_ns": before.st_mtime_ns}
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(f"File changed while computing its identity: {path}")
        identity["sha256"] = digest.hexdigest()
    return identity


class ResultStore:
    """Strict run manifest, with episode ranges deliberately kept outside it.

    All manifest fields participate in resume comparison. Include the protocol,
    parameters and checkpoint/recipe/dataset identities, plus identities for any
    external instruction sources. Exclude episode selection to allow extensions.
    """

    def __init__(self, output_dir, manifest, resume=False):
        if not isinstance(manifest, dict):
            raise TypeError("Run manifest must be an object")
        if not isinstance(manifest.get("parameters"), dict):
            raise TypeError("Run manifest requires parameters as an object")
        if not isinstance(manifest.get("protocol"), (str, dict)) or not manifest["protocol"]:
            raise ValueError("Run manifest requires a nonempty protocol")
        for key in ("checkpoint_identity", "recipe_identity", "dataset_identity"):
            if not manifest.get(key):
                raise ValueError(f"Run manifest requires {key}")
        self.manifest = json.loads(_json(manifest))
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(self.output_dir / ".run.lock", timeout=0, thread_local=False)
        self._writers = set()
        self._closed = False
        try:
            try:
                self._lock.acquire()
            except Timeout as error:
                raise RuntimeError(
                    f"Result directory is already in use: {self.output_dir}"
                ) from error
            path = self.output_dir / "manifest.json"
            expected = {"schema_version": 1, "manifest": self.manifest}
            if path.exists():
                if not resume:
                    raise FileExistsError(f"Run already exists; use resume: {self.output_dir}")
                if json.loads(path.read_text(encoding="utf-8")) != expected:
                    raise ValueError(
                        "Cannot resume: run manifest differs from requested configuration"
                    )
            else:
                if resume:
                    raise FileNotFoundError(f"Cannot resume without a run manifest: {path}")
                if any(p.name != ".run.lock" for p in self.output_dir.iterdir()):
                    raise FileExistsError(f"Result directory is not empty: {self.output_dir}")
                _atomic_text(path, _json(expected) + "\n")
            self.episodes_dir = self.output_dir / "episodes"
            self.episodes_dir.mkdir(exist_ok=True)
            self._records = self._read_records()
            self._write_summary()
        except BaseException:
            self._lock.release()
            self._closed = True
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Result store is closed")

    def _episode_identity(self, episode, index):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Episode index must be a nonnegative integer")
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id or episode.get("episode_id") is None:
            raise ValueError("Episode requires a nonempty episode_id")
        identity = {
            "dataset_identity": self.manifest["dataset_identity"],
            "episode_id": episode_id,
            "episode_index": index,
        }
        return hashlib.sha256(_json(identity).encode("utf-8")).hexdigest(), identity

    def _read_records(self):
        records = {}
        for directory in sorted(self.episodes_dir.iterdir()):
            if directory.name.startswith("."):
                continue  # Uncommitted crash staging does not count as completed.
            try:
                record = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
                key, identity = self._episode_identity(record, record["episode_index"])
                if (
                    key != directory.name
                    or record["identity"] != identity
                    or record["status"] not in ("completed", "failed")
                    or not isinstance(record["result"], dict)
                    or not (directory / "steps.jsonl").is_file()
                ):
                    raise ValueError("invalid identity, result or missing step log")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid committed episode result: {directory}") from error
            records[key] = _summary_record(record, key)
        return records

    def is_completed(self, episode, index):
        """Both committed normal outcomes and execution failures are skipped on resume."""
        self._ensure_open()
        key, _ = self._episode_identity(episode, index)
        return key in self._records

    def begin_episode(self, episode, index):
        self._ensure_open()
        key, identity = self._episode_identity(episode, index)
        if key in self._records or any(writer.key == key for writer in self._writers):
            raise FileExistsError(f"Episode already recorded or being written: {index}")
        writer = EpisodeWriter(self, key, identity, episode.get("scene_id"))
        self._writers.add(writer)
        return writer

    def summary(self):
        """Return metrics and compact records; episode_file links to full results.

        Instruction sources count completed episodes only. Legacy records with
        no source metadata count as unknown; failures retain their established
        contribution to the overall success-rate denominator.
        """
        self._ensure_open()
        records = sorted(
            self._records.values(), key=lambda r: (r["episode_index"], r["episode_id"])
        )
        values, successes = {}, 0
        instruction_source_counts = {}
        for record in records:
            if record["status"] == "completed":
                parameters = record["result"].get("parameters")
                source = (
                    parameters.get("instruction_source") if isinstance(parameters, dict) else None
                )
                source = source.strip() if isinstance(source, str) else ""
                source = source or "unknown"
                instruction_source_counts[source] = instruction_source_counts.get(source, 0) + 1
            metrics = record["result"].get("metrics") or {}
            success = metrics.get("SR", metrics.get("success", False))
            successes += int(record["status"] == "completed" and bool(success))
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.setdefault(name, []).append(float(value))
        completed = sum(record["status"] == "completed" for record in records)
        result = {
            "n_episodes": len(records),
            "n_completed": completed,
            "n_failed": len(records) - completed,
            "n_success": successes,
            "success_rate": successes / len(records) if records else 0.0,
            "metric_means": {k: math.fsum(v) / len(v) for k, v in sorted(values.items())},
            "metric_counts": {k: len(v) for k, v in sorted(values.items())},
            "instruction_source_counts": dict(sorted(instruction_source_counts.items())),
            "episodes": records,
        }
        return json.loads(_json(result))  # Prevent callers mutating committed records.

    def _write_summary(self):
        summary = self.summary()
        _atomic_text(self.output_dir / "summary.json", _json(summary) + "\n")
        metric_names = sorted(summary["metric_means"])
        fields = [
            "episode_index",
            "episode_id",
            "scene_id",
            "status",
            "reason",
            "total_steps",
            "error",
        ]
        fields.extend(f"metric.{name}" for name in metric_names)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in summary["episodes"]:
            result = record["result"]
            row = {name: record.get(name, "") for name in fields[:4]}
            row.update(reason=result.get("reason", ""), total_steps=result.get("total_steps", 0))
            row["error"] = record.get("error", "")
            metrics = result.get("metrics") or {}
            row.update({f"metric.{name}": metrics.get(name, "") for name in metric_names})
            writer.writerow(row)
        _atomic_text(self.output_dir / "summary.csv", stream.getvalue())

    def close(self):
        if not self._closed:
            for writer in list(self._writers):
                writer.abort()
            self._lock.release()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()


class EpisodeWriter:
    """Streaming transaction; interrupted episodes remain eligible for retry."""

    def __init__(self, store, key, identity, scene_id):
        self.store, self.key, self.identity, self.scene_id = store, key, identity, scene_id
        self.path = Path(tempfile.mkdtemp(prefix=f".pending-{key}-", dir=store.episodes_dir))
        self._stream = (self.path / "steps.jsonl").open("w", encoding="utf-8")
        self._closed = False
        self.step_count = 0

    def write_step(self, step):
        self.store._ensure_open()
        if self._closed:
            raise RuntimeError("Episode writer is closed")
        self._stream.write(_json(step) + "\n")
        self._stream.flush()
        self.step_count += 1

    def import_steps(self, source):
        """Adopt a worker's completed JSONL log before committing its episode.

        Workers stage logs on the output filesystem, so the normal path is a
        rename rather than copying image payloads through process pipes. Only
        the parent owns this writer and the run manifest. An imported log is
        still uncommitted until ``finish`` or ``fail`` succeeds.
        """
        self.store._ensure_open()
        if self._closed:
            raise RuntimeError("Episode writer is closed")
        if self.step_count or self._stream.tell():
            raise ValueError("Steps can only be imported into an empty episode writer")
        source = Path(source)
        destination = self.path / "steps.jsonl"
        if source.resolve() == destination.resolve():
            raise ValueError("Source steps must differ from the writer's own log")

        def reject_constant(value):
            raise ValueError("Imported steps cannot contain nonfinite JSON numbers: " + value)

        count = 0
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n") or not isinstance(
                    json.loads(line, parse_constant=reject_constant), dict
                ):
                    raise ValueError("Imported steps must be complete JSONL objects")
                count += 1
        self._stream.close()
        try:
            try:
                os.replace(source, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                # Copy to a private file first: a failed cross-device copy must
                # leave the writer's original empty log usable for recovery.
                descriptor, temporary = tempfile.mkstemp(prefix=".import-", dir=self.path)
                os.close(descriptor)
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                self.step_count = count
                source.unlink()
        finally:
            self._stream = destination.open("a", encoding="utf-8")
        self.step_count = count

    def _finish(self, result, status, error=None):
        self.store._ensure_open()
        if self._closed:
            raise RuntimeError("Episode writer is closed")
        if not isinstance(result, dict):
            raise TypeError("Episode result must be an object")
        if result.get("metrics") is not None and not isinstance(result["metrics"], dict):
            raise TypeError("Episode metrics must be an object")
        record = {
            "identity": self.identity,
            "episode_id": self.identity["episode_id"],
            "episode_index": self.identity["episode_index"],
            "scene_id": self.scene_id,
            "status": status,
            "step_count": self.step_count,
            "result": result,
        }
        if error is not None:
            record["error"] = error
        record = json.loads(_json(record))
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
        _atomic_text(self.path / "episode.json", _json(record) + "\n")
        os.replace(self.path, self.store.episodes_dir / self.key)
        self._closed = True
        self.store._writers.discard(self)
        self.store._records[self.key] = _summary_record(record, self.key)
        _sync_directory(self.store.episodes_dir)
        self.store._write_summary()
        return json.loads(_json(record))

    def finish(self, result):
        return self._finish(result, "completed")

    def fail(self, error, result=None):
        message = (
            f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        )
        return self._finish(result or {"reason": "error", "metrics": {}}, "failed", message)

    def abort(self):
        if not self._closed:
            self._stream.close()
            shutil.rmtree(self.path, ignore_errors=True)
            self._closed = True
            self.store._writers.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, _traceback):
        if not self._closed:
            if kind is not None and issubclass(kind, Exception):
                try:
                    self.fail(error)
                finally:
                    self.abort()
            else:
                self.abort()
