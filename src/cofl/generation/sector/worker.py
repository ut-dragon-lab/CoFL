"""Python 3.7 simulator worker: JSON control and non-pickle NumPy transport.

Run through SectorPipeline; stdout is reserved for one JSON response per request.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, default=_json_default, allow_nan=False), encoding="utf-8"
    )


class TransportSink:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.frames = 0

    def begin_episode(self, metadata):
        write_json(self.root / "episode.json", metadata)

    def add_frame(self, frame):
        metadata, arrays = {}, {}
        for key, value in frame.items():
            if isinstance(value, bytes):
                arrays[key] = np.frombuffer(value, dtype=np.uint8)
            elif isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value
        stem = f"frame-{self.frames:06d}"
        np.savez(self.root / (stem + ".npz"), **arrays)
        write_json(self.root / (stem + ".json"), metadata)
        self.frames += 1


def runtime_provenance():
    import hashlib
    import importlib
    import platform

    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "cv2", "PIL", "habitat", "habitat_sim", "habitat_extensions"):
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        entry = {"version": str(version) if version is not None else "unversioned"}
        if name == "habitat_extensions":
            root = Path(module.__file__).parent
            digest = hashlib.sha256()
            for path in sorted(root.rglob("*.py")):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
            entry["python_source_sha256"] = digest.hexdigest()
        versions[name] = entry
    return {"simulator_runtime": versions, "transport": "json-npz-v1"}


def main():
    # Habitat's C++ renderer writes directly to fd 1; Python redirect_stdout
    # alone cannot protect the JSON channel. Keep a private duplicate for it.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    core = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            with contextlib.redirect_stdout(sys.stderr):
                if command == "initialize":
                    from .core import SectorCore

                    core = SectorCore(request["options"])
                    response = {"ok": True, "runtime": runtime_provenance()}
                elif command == "generate":
                    if core is None:
                        raise RuntimeError("Worker has not been initialized")
                    sink = TransportSink(request["scratch"])
                    summary = core.generate(request["episode"], seed=request["seed"], sink=sink)
                    response = {"ok": True, "summary": summary, "frames": sink.frames}
                elif command == "close":
                    if core is not None:
                        core.close()
                    response = {"ok": True}
                else:
                    raise ValueError("Unknown worker command")
            print(
                json.dumps(response, default=_json_default, allow_nan=False),
                file=protocol,
                flush=True,
            )
            if command == "close":
                return
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(
                json.dumps({"ok": False, "error": str(error), "type": type(error).__name__}),
                file=protocol,
                flush=True,
            )
    if core is not None:
        with contextlib.redirect_stdout(sys.stderr):
            core.close()


if __name__ == "__main__":
    main()
