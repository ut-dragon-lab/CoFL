"""Immutable evaluation identity and transactional SQLite result persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

from .metrics import PROFILES


@dataclass(frozen=True)
class TimingSpec:
    """Execution timing; simulated rates do not assert real-time capability."""

    execution: str = "offline"
    planner_hz: float | None = None
    controller_hz: float | None = None
    plant_hz: float | None = None
    inference_latency_affects_execution: bool = False

    def __post_init__(self) -> None:
        if self.execution not in ("offline", "fixed_step", "teleport", "wall_time"):
            raise ValueError("Unsupported timing execution mode")
        rates = (self.planner_hz, self.controller_hz, self.plant_hz)
        for rate in rates:
            if rate is not None and (not math.isfinite(rate) or rate <= 0):
                raise ValueError("Timing rates must be finite and positive when provided")
        if self.execution == "offline" and any(rate is not None for rate in rates):
            raise ValueError("Offline evaluation must not specify simulated rates")
        if self.execution in ("fixed_step", "wall_time") and any(rate is None for rate in rates):
            raise ValueError("Closed-loop timing requires planner, controller and plant rates")
        if all(rate is not None for rate in rates) and not (
            self.planner_hz <= self.controller_hz <= self.plant_hz
        ):
            raise ValueError("Require planner_hz <= controller_hz <= plant_hz")
        if self.execution == "offline" and self.inference_latency_affects_execution:
            raise ValueError("Offline inference latency cannot affect simulated execution")


@dataclass(frozen=True)
class EvaluationProtocol:
    """Complete, immutable protocol metadata carried alongside every result.

    Missing artifact hashes remain explicit ``None``. A recipe name or an
    output metric alone does not establish reproduction of a publication.
    """

    name: str
    profile: str
    split: str
    seed: int
    recipe: str
    instruction_mode: str = "full_instruction"
    timing: TimingSpec = field(default_factory=TimingSpec)
    version: int = 1
    task: str = "offline_field"
    dataset_sha256: str | None = None
    checkpoint_sha256: str | None = None
    recipe_sha256: str | None = None
    code_revision: str | None = None
    instruction_source: str | None = None
    inference_rules: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for key in ("name", "split", "recipe", "task"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        if self.profile not in PROFILES:
            raise ValueError(f"Unsupported profile: {self.profile!r}")
        if self.instruction_mode not in (
            "full_instruction",
            "provided_annotation",
            "oracle_assisted",
            "learned_progress",
            "goal_assisted",
        ):
            raise ValueError("Unknown instruction mode")
        if self.instruction_mode != "full_instruction" and not self.instruction_source:
            raise ValueError("Assisted or learned instruction modes require instruction_source")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.timing, TimingSpec):
            raise TypeError("timing must be a TimingSpec")
        for key in ("dataset_sha256", "checkpoint_sha256", "recipe_sha256"):
            value = getattr(self, key)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{key} must be a lowercase SHA-256 hex digest or None")
        if self.inference_rules is not None:
            if not isinstance(self.inference_rules, dict):
                raise TypeError("inference_rules must be a JSON object or None")
            # Detach protocol identity from mutable caller-owned nested metadata.
            object.__setattr__(
                self, "inference_rules", json.loads(_encode_json(self.inference_rules))
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.inference_rules is None:
            # Non-policy scoring protocols retain their existing serialized identity.
            payload.pop("inference_rules")
        return payload

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_encode_json(self.to_dict()).encode("utf-8")).hexdigest()


def _encode_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    encoded = _encode_json(payload)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


class ResultWriter:
    """Single-writer SQLite records with reconstructible JSON/JSONL exports.

    SQLite is the authoritative result store. Each batch commits atomically;
    reopening recovers completed IDs from committed rows. Protocol mismatches
    and duplicate IDs are errors. Use one writer process per directory;
    concurrent writes from multiple threads or processes are unsupported.
    """

    def __init__(self, output_dir: str | Path, protocol: EvaluationProtocol):
        if not isinstance(protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol
        self.protocol_path = self.output_dir / "protocol.json"
        self.results_path = self.output_dir / "results.jsonl"
        self.database_path = self.output_dir / "results.sqlite"
        payload = {"protocol_sha256": protocol.fingerprint, "protocol": protocol.to_dict()}
        encoded_protocol = _encode_json(payload)
        has_export = self.results_path.exists() and self.results_path.stat().st_size > 0
        if not self.database_path.exists() and has_export:
            raise ValueError(
                "Existing JSONL results have no SQLite store; use a fresh output directory"
            )
        if (
            self.protocol_path.exists()
            and json.loads(self.protocol_path.read_text(encoding="utf-8")) != payload
        ):
            raise ValueError("Output directory belongs to a different evaluation protocol")
        self._connection = sqlite3.connect(self.database_path)
        try:
            self._connection.execute("PRAGMA synchronous=FULL")
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS results ("
                    "sequence INTEGER PRIMARY KEY, sample_id TEXT NOT NULL UNIQUE, "
                    "record_json TEXT NOT NULL)"
                )
                stored = self._connection.execute(
                    "SELECT value FROM metadata WHERE key = 'protocol'"
                ).fetchone()
                if stored is None:
                    if has_export:
                        raise ValueError(
                            "Existing JSONL results have no SQLite protocol; use a fresh output directory"
                        )
                    if self._connection.execute("SELECT 1 FROM results LIMIT 1").fetchone():
                        raise ValueError("SQLite results have no protocol identity")
                    self._connection.execute(
                        "INSERT INTO metadata (key, value) VALUES ('protocol', ?)",
                        (encoded_protocol,),
                    )
                elif stored[0] != encoded_protocol:
                    raise ValueError("SQLite store belongs to a different evaluation protocol")
            self.completed_sample_ids = {
                row[0] for row in self._connection.execute("SELECT sample_id FROM results")
            }
            if not self.protocol_path.exists():
                _atomic_json(self.protocol_path, payload)
        except BaseException:
            self._connection.close()
            raise

    def write(
        self,
        sample_id: str,
        metrics: Mapping[str, Any],
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.write_many(
            [{"sample_id": sample_id, "metrics": metrics, "diagnostics": diagnostics}]
        )[0]

    def write_many(self, records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Validate and commit an entire batch, or leave every row uncommitted."""
        prepared, rows = [], []
        for item in records:
            sample_id = item["sample_id"]
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("sample_id must be a nonempty string")
            record = {
                "schema_version": 1,
                "protocol_sha256": self.protocol.fingerprint,
                "sample_id": sample_id,
                "metrics": dict(item["metrics"]),
                "diagnostics": dict(item.get("diagnostics") or {}),
            }
            rows.append((sample_id, _encode_json(record)))
            prepared.append(record)
        if not rows:
            return []
        try:
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO results (sample_id, record_json) VALUES (?, ?)", rows
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate sample_id in evaluation batch") from error
        self.completed_sample_ids.update(row[0] for row in rows)
        return prepared

    def records(self) -> Iterator[dict[str, Any]]:
        """Yield committed records in insertion order without loading the full stream."""
        cursor = self._connection.execute("SELECT record_json FROM results ORDER BY sequence")
        try:
            for row in cursor:
                yield json.loads(row[0])
        finally:
            cursor.close()

    def export_results(self) -> Path:
        """Atomically rebuild JSONL from committed SQLite rows."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.output_dir, delete=False
            ) as handle:
                temporary = handle.name
                cursor = self._connection.execute(
                    "SELECT record_json FROM results ORDER BY sequence"
                )
                try:
                    for row in cursor:
                        handle.write(row[0] + "\n")
                finally:
                    cursor.close()
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.results_path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return self.results_path

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        self.export_results()
        path = self.output_dir / "summary.json"
        _atomic_json(
            path,
            {
                "protocol_sha256": self.protocol.fingerprint,
                "sample_count": len(self.completed_sample_ids),
                "summary": dict(summary),
            },
        )
        return path

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
