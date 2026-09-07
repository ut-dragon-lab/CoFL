"""Atomic local reports and streaming source/payload checksums."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from cofl.data._io import file_sha256


def digest_json(value) -> str:
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tree_checksums(path: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.relative_to(path).as_posix() not in exclude
    }


class InputFingerprints:
    """Hash shared assets once; detect files changed during this invocation."""

    def __init__(self):
        self._cache = {}

    def describe(self, inputs) -> list[dict]:
        result = []
        for path in inputs:
            path = Path(path).resolve(strict=True)
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            previous = self._cache.get(path)
            if previous is not None and previous[0] != identity:
                raise ValueError(f"Source changed during generation: {path.name}")
            if previous is None:
                previous = (identity, file_sha256(path))
                self._cache[path] = previous
            result.append(
                {
                    "id": digest_json(path.as_posix()),
                    "name": path.name,
                    "size": stat.st_size,
                    "sha256": previous[1],
                }
            )
        return result

    def verify_all(self) -> None:
        """Check inputs of earlier units again before completing a collection."""
        self.describe(tuple(self._cache))


@contextmanager
def output_lock(output: Path):
    """A released-on-exit lock; stale lock files do not prevent crash recovery."""
    import fcntl

    with (output / ".generation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another generator owns this output; use a separate shard directory"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
