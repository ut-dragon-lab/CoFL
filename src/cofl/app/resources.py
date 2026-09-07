"""Browse local resource locations without parsing models or dataset formats."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from fastapi import Request

    from .contracts import ResourceKind

MODEL_SUFFIXES = {".ckpt", ".pt", ".pth"}
BENCHMARK_SUFFIXES = {".yaml", ".yml", ".json"}


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resource_id(kind: ResourceKind, path: Path) -> str:
    """Stable across sessions and spelling/symlink aliases of the same path."""
    digest = hashlib.sha256(os.fsencode(path)).hexdigest()[:24]
    return f"{kind}-{digest}"


def _dataset_candidate(path: Path) -> bool:
    # Discovery hints only. Registration validates through the native reader.
    return any((path / name).is_file() for name in ("manifest.json", "collection.json"))


def browse_resources(kind: ResourceKind, path: str | None = None) -> dict:
    location = resolved_path(Path.cwd() if path is None else path)
    selected = None
    suffixes = MODEL_SUFFIXES if kind == "model" else BENCHMARK_SUFFIXES
    if kind in ("model", "benchmark") and location.is_file():
        if location.suffix.lower() not in suffixes:
            raise ValueError(f"Choose a {kind} file with one of {sorted(suffixes)}")
        selected = str(location)
        location = location.parent
    if not location.is_dir():
        if not location.exists():
            raise FileNotFoundError(f"Directory does not exist: {location}")
        raise ValueError("Choose a directory to browse")
    entries = []
    with os.scandir(location) as children:
        for child in children:
            try:
                is_dir = child.is_dir()
                is_resource = child.is_file() and Path(child.name).suffix.lower() in suffixes
                if not is_dir and not (kind in ("model", "benchmark") and is_resource):
                    continue
                child_path = resolved_path(child.path)
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child_path),
                        "is_dir": is_dir,
                        "selectable": is_resource
                        if kind in ("model", "benchmark")
                        else _dataset_candidate(child_path),
                    }
                )
            except (OSError, RuntimeError):
                # Broken/inaccessible individual links should not hide siblings.
                continue
    entries.sort(key=lambda entry: (not entry["is_dir"], entry["name"].casefold()))
    return {
        "kind": kind,
        "cwd": str(location),
        "parent": None if location == location.parent else str(location.parent),
        "selectable": kind == "dataset" and _dataset_candidate(location),
        "selected_path": selected,
        "entries": entries,
        "shortcuts": [
            {"id": "cwd", "label": "Working directory", "path": str(Path.cwd().resolve())},
            {"id": "home", "label": "Home", "path": str(Path.home().resolve())},
            {"id": "filesystem", "label": "Filesystem", "path": location.anchor},
        ],
    }


def require_same_origin(request: Request) -> None:
    """Protect host-file access from cross-site requests and DNS rebinding.

    Vite's default proxy retains the browser-facing Host header. Non-browser
    local API clients (including TestClient) need not invent an Origin header.
    """
    from fastapi import HTTPException

    if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
        raise HTTPException(403, "Local resources require a same-origin request")
    try:
        target = urlsplit(str(request.url))
        host = (target.hostname or "").rstrip(".").lower()
        server = request.scope.get("server")
        trusted_hosts = {"localhost", socket.gethostname().lower()}
        if server:
            trusted_hosts.add(str(server[0]).rstrip(".").lower())
        if host not in trusted_hosts:
            # Numeric addresses allow explicitly bound LAN/remote development;
            # an arbitrary DNS name cannot rebind into a local file browser.
            ipaddress.ip_address(host)
        origin = request.headers.get("origin")
        if origin is not None:
            source = urlsplit(origin)

            def origin_tuple(url):
                return (
                    url.scheme.lower(),
                    (url.hostname or "").rstrip(".").lower(),
                    url.port or (443 if url.scheme == "https" else 80),
                )

            if (
                source.scheme not in ("http", "https")
                or source.username is not None
                or source.password is not None
                or source.path not in ("", "/")
                or source.query
                or source.fragment
                or origin_tuple(source) != origin_tuple(target)
            ):
                raise ValueError("Origin does not match Host")
    except ValueError as error:
        raise HTTPException(403, "Local resources require the Studio origin and host") from error
