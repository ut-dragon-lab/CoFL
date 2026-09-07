"""Local Studio launcher, with source-checkout frontend builds on request."""

import argparse
import shutil
import subprocess
import threading
import webbrowser
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cofl app", description="Launch CoFL Studio locally")
    parser.add_argument("--config", type=Path, help="Studio YAML resource catalog")
    parser.add_argument("--scene", type=Path, help="Add one GLB to the scene catalog")
    parser.add_argument(
        "--scene-up-axis",
        choices=("y", "z"),
        help="Vertical axis of --scene (default y; legacy Matterport GLBs use z)",
    )
    parser.add_argument("--checkpoint", type=Path, help="Add a self-contained CoFL checkpoint")
    parser.add_argument("--dataset", type=Path, help="Add a native dataset or collection")
    parser.add_argument(
        "--device", help="Policy device (default cuda:0); cpu explicitly opts into CPU inference"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8787, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--build", action="store_true", help="Build the web UI in a source checkout"
    )
    parser.add_argument("--api-only", action="store_true", help="API server for Vite development")
    parser.add_argument(
        "--export-openapi", type=Path, help="Write the schema without starting a server"
    )
    args = parser.parse_args(argv)
    server = None

    def request_shutdown():
        if server is not None:
            server.should_exit = True

    try:
        from .config import DatasetResource, FileResource, SceneResource, load_config
        from .server import create_app

        config = load_config(args.config)
        if args.scene_up_axis and args.scene is None:
            raise ValueError("--scene-up-axis requires --scene; use up_axis in a YAML catalog")
        if args.device:
            config.device = args.device
        for arg, resources, cls in (
            (args.scene, config.scenes, SceneResource),
            (args.checkpoint, config.models, FileResource),
            (args.dataset, config.datasets, DatasetResource),
        ):
            if arg is not None:
                resources["local"] = cls(path=arg.resolve(), label=arg.stem)
        if args.scene_up_axis:
            config.scenes["local"].up_axis = args.scene_up_axis
        if args.export_openapi:
            import json

            schema = create_app(config).openapi()
            args.export_openapi.parent.mkdir(parents=True, exist_ok=True)
            args.export_openapi.write_text(json.dumps(schema, indent=2) + "\n")
            return 0
        if args.build:
            frontend = Path(__file__).resolve().parents[3] / "app"
            if not (frontend / "package-lock.json").is_file() or not shutil.which("npm"):
                raise ValueError(
                    "Building Studio requires the CoFL source checkout and Node.js/npm"
                )
            subprocess.run(["npm", "ci"], cwd=frontend, check=True)
            subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)
        static = Path(__file__).parent / "static"
        if not args.api_only and not (static / "index.html").is_file():
            raise ValueError(
                "Studio web assets are missing. In a source checkout use: cofl app --build"
            )
        app = create_app(config, shutdown_callback=request_shutdown)
    except ImportError as error:
        parser.error(
            f"Install Studio dependencies with uv sync --locked or pip install 'cofl-navigation[app]': {error}"
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    import uvicorn

    if not args.no_browser and not args.api_only:
        host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        timer = threading.Timer(1.5, webbrowser.open, args=[f"http://{host}:{args.port}"])
        timer.daemon = True
        timer.start()
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
    server.run()
    return 0
