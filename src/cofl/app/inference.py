"""Single-model inference service using the public encoding/query contracts.

No checkpoint is loaded until a prediction is requested. One model and one
content-addressed observation context and integration field are retained, and a
lock serializes their use across HTTP workers.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import pickle
import sys
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from cofl.data.profiles import geometry_for_profile
from cofl.fields import field_query_grid, queries_for_grid, query_valid_mask
from cofl.runtime.device import DeviceManager, policy_device_operation
from cofl.runtime.device import release_cuda as release_cuda_memory

from .adapters import DomainConstraint
from .datasets import coordinate_metadata, finite_array, sampled_field


def _decode_image(value: str) -> np.ndarray:
    from PIL import Image, UnidentifiedImageError

    if not isinstance(value, str) or "," not in value:
        raise ValueError("Scene inference requires a PNG/JPEG image data URL")
    prefix, encoded = value.split(",", 1)
    formats = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
    if prefix not in formats:
        raise ValueError("Scene image must be a PNG/JPEG base64 data URL")
    if len(encoded) > 16 * 1024 * 1024:
        raise ValueError("Encoded scene image exceeds 16 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.format != formats[prefix]:
                raise ValueError("Scene image format does not match its data URL")
            if max(image.size) > 4096 or min(image.size) < 1:
                raise ValueError("Scene image dimensions must be within 1..4096")
            return np.array(image.convert("RGB"), copy=True)
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Scene image is not a valid PNG/JPEG") from error


def _observation_key(sample: dict) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"instruction": sample["instruction"], "geometry": sample["geometry"]},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    )
    for name in ("image", "depth", "depth_valid"):
        digest.update(name.encode())
        value = sample.get(name)
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _integer(request: Mapping, name: str, default: int, low: int, high: int) -> int:
    value = request.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer within {low}..{high}")
    return value


class InferenceService:
    def __init__(self, models: dict[str, Path], device: str = "cuda:0", *, device_manager=None):
        self.models = {key: Path(value) for key, value in models.items()}
        self.device_manager = device_manager or DeviceManager(device)
        self._lock = RLock()
        self._loaded_key = None
        self._model = self._config = None
        self._context_key = None
        self._context = self._query_context = self._query_kwargs = None
        self._integration_field = None
        self._methods = {}

    @property
    def device(self):
        return self.device_manager.device

    @device.setter
    def device(self, value):
        self.device_manager.select(value)

    def describe(self) -> list[dict]:
        with self._lock:
            return [
                {"id": key, "label": key, "method": self._methods.get(key)} for key in self.models
            ]

    @policy_device_operation
    def register(
        self, resource_id: str, path: Path, *, domain: DomainConstraint | None = None
    ) -> dict:
        """Load a real checkpoint before publishing its resource ID.

        Loading another policy releases the previous GPU cache first. If it
        fails, existing resource mappings remain usable and reload on demand.
        """
        from cofl.training.config import METHOD_PROFILES

        from .resources import MODEL_SUFFIXES, resolved_path

        path = resolved_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
        if path.suffix.lower() not in MODEL_SUFFIXES:
            raise ValueError("Choose a .ckpt, .pt, or .pth checkpoint")
        with self._lock:
            previous = self.models.get(resource_id)
            if previous is not None and previous.resolve() != path:
                raise ValueError("This model ID already belongs to another checkpoint")
            stat = path.stat()
            key = (resource_id, str(path), stat.st_mtime_ns, stat.st_size)
            if key != self._loaded_key:
                from cofl.training.policy import load_policy

                self.close()
                self.device_manager.validate()
                try:
                    model, config = load_policy(path, device=self.device)
                except (KeyError, TypeError, EOFError, pickle.UnpicklingError) as error:
                    raise ValueError(f"Checkpoint could not be loaded: {error}") from error
                if domain is not None:
                    try:
                        domain.check_model(config.method, METHOD_PROFILES[config.method])
                    except ValueError:
                        # Do not retain rejected GPU weights in the error traceback.
                        del model
                        self.close()
                        raise
                self._model, self._config = model, config
                self._loaded_key = key
                self._methods[resource_id] = config.method
            profile = METHOD_PROFILES[self._config.method]
            if domain is not None:
                domain.check_model(self._config.method, profile)
            self.models[resource_id] = path
            return {"method": self._config.method, "profile": profile}

    def _load(self, model_id: str):
        if model_id not in self.models:
            raise KeyError(f"Unknown model: {model_id}")
        path = self.models[model_id]
        stat = path.stat()
        key = (model_id, str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        if key != self._loaded_key:
            from cofl.training.policy import load_policy

            self.close()
            self.device_manager.validate()
            self._model, self._config = load_policy(path, device=self.device)
            self._loaded_key = key
            self._methods[model_id] = self._config.method

    @policy_device_operation
    def predict(
        self,
        request: Mapping,
        sample: dict | None = None,
        *,
        domain: DomainConstraint | None = None,
    ) -> dict:
        """Predict an image field or inspect a native RGB-D sector annotation.

        Changing only the start reuses the encoded observation and its dense
        integration field. Display grid_size controls arrows independently.
        Observation IDs are echoed for UI cancellation, never used as cache keys.
        """
        grid_size = _integer(request, "grid_size", 24, 8, 64)
        max_steps = _integer(request, "max_steps", 100, 1, 500)
        policy_dt = request.get("policy_dt", 0.01)
        if isinstance(policy_dt, bool) or not isinstance(policy_dt, (int, float)):
            raise ValueError("policy_dt must be finite and positive")  # noqa: TRY004 — request validation
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("policy_dt must be finite and positive")
        instruction = request.get("instruction")
        if instruction is None and sample is not None:
            instruction = sample["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Instruction must be a nonempty string")
        if sample is not None and request.get("image") is not None:
            raise ValueError("Use either a native dataset sample or a scene image")
        if sample is None:
            observation = {
                "profile": "image_field_v1",
                "geometry": geometry_for_profile("image_field_v1"),
                "image": _decode_image(request.get("image")),
                "instruction": instruction,
            }
        else:
            observation = dict(sample, instruction=instruction)

        # Importing the app/catalog does not import PyTorch or load a backbone.
        import torch

        from cofl.evaluation.metrics import evaluate_field
        from cofl.runtime.field_grid import predict_field_grid
        from cofl.runtime.inference_rules import (
            DEFAULT_FIELD_GRID_SIZE,
            DEFAULT_FIELD_QUERY_CHUNK_SIZE,
            inference_rule_metadata,
        )
        from cofl.runtime.rollout import rollout_image_field, rollout_sector_field
        from cofl.training.config import METHOD_PROFILES
        from cofl.training.policy import encode_samples, query_policy_kwargs

        started = perf_counter()
        with self._lock, torch.inference_mode():
            model_id = request["model_id"]
            self._load(model_id)
            profile = METHOD_PROFILES[self._config.method]
            if domain is not None:
                domain.check_model(self._config.method, profile)
            if sample is None and self._config.method == "cofl-s":
                raise ValueError(
                    "CoFL-S requires a native ground-sector sample with geometry/depth"
                )
            if observation["profile"] != profile:
                raise ValueError("Dataset profile does not match the selected model")
            key = _observation_key(observation)
            cache_hit = key == self._context_key
            if not cache_hit:
                self._integration_field = None
                kwargs = query_policy_kwargs(
                    self._config,
                    [observation],
                    device=self.device,
                    dtype=torch.float32,
                    inference=True,
                )
                context = encode_samples(self._model, self._config, [observation], self.device)
                query_context = self._model.prepare_query_context(context)
                self._context, self._query_context, self._query_kwargs = (
                    context,
                    query_context,
                    kwargs,
                )
                self._context_key = key

            def field(queries):
                queries = torch.as_tensor(queries, device=self.device, dtype=torch.float32)[None]
                vectors = (
                    self._model.query(queries, self._query_context, **self._query_kwargs)[0]
                    .float()
                    .cpu()
                    .numpy()
                )
                if vectors.shape != (queries.shape[1], 2):
                    raise ValueError("Policy field output must have shape (N, 2)")
                return finite_array(vectors, "Predicted vectors")

            geometry = observation["geometry"]
            if sample is not None:
                preview = sampled_field(sample, max_points=grid_size**2)
                queries, positions = preview["queries"], preview["positions"]
            else:
                view_geometry = dict(geometry, grid_shape=[grid_size, grid_size])
                valid = query_valid_mask(profile, view_geometry)
                queries = queries_for_grid(profile, view_geometry)[valid]
                positions = field_query_grid(view_geometry)[valid]
            vectors = field(queries) if len(queries) else np.empty((0, 2), dtype=np.float32)
            metrics = {}
            if sample is not None and instruction == sample["instruction"]:
                metrics = evaluate_field(
                    vectors, preview["target"], profile=profile, geometry=geometry
                )
                metrics.update(
                    scope="sampled_supervised_cells",
                    supervised_count=preview["supervised_count"],
                )
            start = request.get("start")
            if start is None and profile == "image_field_v1":
                start = [0.5, 0.5]
            rollout_kwargs = {
                "geometry": geometry,
                "max_steps": max_steps,
                "policy_dt": float(policy_dt),
                "boundary": "project",
                "nonfinite": "raise",
            }
            if self._integration_field is None:
                self._integration_field = predict_field_grid(
                    self._model,
                    self._query_context,
                    self._query_kwargs,
                    geometry=geometry,
                    device=self.device,
                    grid_size=DEFAULT_FIELD_GRID_SIZE,
                    query_chunk_size=DEFAULT_FIELD_QUERY_CHUNK_SIZE,
                )
            if profile == "image_field_v1":
                rollout = rollout_image_field(
                    self._integration_field, start=start, **rollout_kwargs
                )
            else:
                rollout = rollout_sector_field(
                    self._integration_field,
                    start_m=start,
                    field_grid_size=DEFAULT_FIELD_GRID_SIZE,
                    **rollout_kwargs,
                )
                vectors = finite_array(
                    vectors * geometry["normalization_scale_m"], "Metric predicted vectors"
                )
            actions = None
            if self._config.method == "cofl-s" and self._model.action_head is not None:
                actions = finite_array(
                    self._model.action_logits(self._context)[0].float().cpu().numpy(),
                    "Action logits",
                ).tolist()
            return {
                "observation_id": request.get("observation_id", ""),
                "model_id": model_id,
                "method": self._config.method,
                "profile": profile,
                **coordinate_metadata(profile),
                "queries": positions.tolist(),
                "vectors": vectors.tolist(),
                "trajectory": rollout.points.tolist(),
                "stop_reason": rollout.stop_reason,
                "elapsed_ms": (perf_counter() - started) * 1000,
                "cache_hit": cache_hit,
                "inference_rules": inference_rule_metadata(
                    profile,
                    grid_size=DEFAULT_FIELD_GRID_SIZE,
                    num_steps=max_steps,
                    policy_dt=float(policy_dt),
                ),
                "metrics": metrics,
                "actions": actions,
                "action_semantics": "logits_stop_forward_left_right"
                if actions is not None
                else None,
            }

    def close(self):
        with self._lock:
            torch = sys.modules.get("torch")
            release_cuda = (
                str(self.device).startswith("cuda")
                and torch is not None
                and torch.cuda.is_initialized()
            )
            self._model = self._config = None
            self._loaded_key = self._context_key = None
            self._context = self._query_context = self._query_kwargs = None
            self._integration_field = None
            if release_cuda:
                release_cuda_memory(self.device)
