"""Mode composition independent of page layout, models and simulator runtimes."""

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter

from cofl.online.dynamics import RunParameters

from .contracts import DomainDescriptor, ModeDescriptor


@dataclass(frozen=True)
class StudioMode:
    """A discoverable mode and optional mode-specific HTTP routes.

    A future simulator mode can own its session/step/stream contracts. It need
    not implement static field prediction or reuse image evaluation metrics.
    """

    descriptor: ModeDescriptor
    router: APIRouter | None = None
    static_input: Literal["image", "dataset"] | None = None


def builtin_domains():
    return [
        DomainDescriptor(
            id="bev",
            title="BEV",
            description="Bird's-eye image fields with CoFL",
            methods=["cofl"],
            profiles=["image_field_v1"],
        ),
        DomainDescriptor(
            id="ego2d",
            title="2D egocentric",
            description="RGB-D ground sectors with CoFL-S",
            methods=["cofl-s"],
            profiles=["ground_sector_v1"],
            resource_kinds=["model", "dataset", "benchmark"],
        ),
    ]


def builtin_modes():
    return [
        StudioMode(
            ModeDescriptor(
                id="scene",
                domain_id="bev",
                task="playground",
                execution="static",
                title="Scene Playground",
                renderer="scene",
                description="Explore a space. Choose a view. Query a path.",
                capabilities=["scene", "image_field", "camera_capture", "floor_cutaway"],
            ),
            static_input="image",
        ),
        StudioMode(
            ModeDescriptor(
                id="validation",
                domain_id="bev",
                task="validation",
                execution="static",
                title="Val Inspector",
                renderer="validation",
                description="Inspect native validation samples and compare predicted fields.",
                capabilities=["dataset", "image_field", "reference_metrics"],
            ),
            static_input="dataset",
        ),
        StudioMode(
            ModeDescriptor(
                id="ego2d-interactive",
                domain_id="ego2d",
                task="interactive",
                execution="session",
                title="Interactive Playground",
                renderer="online",
                description="Explore a scene with live instructions, pause and editable spawn poses.",
                capabilities=["benchmark", "sessions", "commands", "reset"],
                default_parameters=RunParameters().to_dict(),
            ),
        ),
        StudioMode(
            ModeDescriptor(
                id="ego2d-playground",
                domain_id="ego2d",
                task="benchmark",
                execution="session",
                title="VLN Benchmark",
                renderer="online",
                description="Evaluate CoFL-S on fixed VLN episodes and benchmark instructions.",
                capabilities=["benchmark", "sessions", "step", "stream"],
                default_parameters=RunParameters().to_dict(),
            ),
        ),
    ]
