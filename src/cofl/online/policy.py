"""CoFL-S replanning: encode once, materialize a dense field, then interpolate."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cofl.runtime.inference_rules import DEFAULT_FIELD_GRID_SIZE, DEFAULT_FIELD_QUERY_CHUNK_SIZE


def configure_geometry_sensors(geometry, sensors):
    """Return request-local geometry using the aligned cameras' actual HFOV."""
    import math

    rgb = sensors.get("rgb", {}).get("hfov_deg")
    depth = sensors.get("depth", {}).get("hfov_deg")
    if rgb is None or depth is None:
        raise ValueError("The simulator must report both RGB and depth camera FoV")
    rgb, depth = float(rgb), float(depth)
    if not (math.isfinite(rgb) and math.isfinite(depth) and 0 < rgb <= 180):
        raise ValueError("Camera FoV must be finite and within (0, 180] degrees")
    if not math.isclose(rgb, depth, rel_tol=1e-6):
        raise ValueError("CoFL-S requires aligned RGB and depth camera FoV")
    return dict(geometry, hfov_rad=math.radians(rgb))


def finish_prediction(field_response, geometry, parameters):
    """Integrate a returned field on its environment worker's CPU."""
    from cofl.runtime.field_grid import RasterField
    from cofl.runtime.rollout import rollout_sector_field

    started = perf_counter()
    field = RasterField(field_response["field_values"], profile=geometry["profile"])
    result = rollout_sector_field(
        field,
        geometry=geometry,
        max_steps=parameters.policy_steps,
        policy_dt=parameters.policy_dt,
        field_grid_size=getattr(parameters, "field_grid_size", DEFAULT_FIELD_GRID_SIZE),
        boundary="project",
        nonfinite="raise",
    )
    return {
        "trajectory": np.asarray(result.points),
        "actions": np.asarray(field_response["actions"]),
        "inference_ms": field_response["inference_ms"] + (perf_counter() - started) * 1000,
        "field_stop_reason": result.stop_reason,
    }


class NativePolicy:
    def __init__(self, checkpoint, *, device="cpu", inference_backend="auto"):
        from cofl.data.profiles import geometry_for_profile
        from cofl.training.policy import load_policy

        self.device = device
        self.inference_backend = inference_backend
        self.active_inference_backend = "eager"
        self.inference_fallback_reason = None
        self._warned_graph_fallback = False
        self._field_grid_metadata = None
        self.model, self.config = load_policy(checkpoint, device=device)
        if self.config.method != "cofl-s":
            self.close()
            raise ValueError("2D egocentric closed-loop navigation requires a CoFL-S checkpoint")
        self.geometry = geometry_for_profile(
            "ground_sector_v1",
            normalization_scale_m=self.config.normalization_scale_m,
            hfov_rad=self.config.hfov_rad,
        )
        self.geometry["r_max_m"] = self.config.r_max_m

    @property
    def inference_backend(self):
        return self._inference_backend

    @inference_backend.setter
    def inference_backend(self, value):
        """Select grid generation execution; every backend uses a complete raster."""
        if value not in {"auto", "eager", "cuda_graph"}:
            raise ValueError("inference_backend must be 'auto', 'eager', or 'cuda_graph'")
        self._inference_backend = value

    @property
    def acceleration_status(self):
        return {
            "requested_backend": self.inference_backend,
            "active_backend": self.active_inference_backend,
            "fallback_reason": self.inference_fallback_reason,
            "graph_captures": 0,
            "graph_transfers_captured": False,
            "graph_transfer_fallback_reason": None,
            "field_grid": self._field_grid_metadata,
        }

    def _select_grid_backend(self):
        # Dense batches eliminate per-integration-step decoder launches and
        # transfers. The older single-query graph is not an integration backend.
        self.active_inference_backend = "eager"
        self.inference_fallback_reason = None
        if self.inference_backend == "cuda_graph":
            self.inference_fallback_reason = (
                "Dense-grid CUDA graph capture is unavailable; using batched eager grid generation"
            )
            if not self._warned_graph_fallback:
                import warnings

                warnings.warn(self.inference_fallback_reason, RuntimeWarning, stacklevel=3)
                self._warned_graph_fallback = True

    def _grid_field(self, prepared, kwargs, parameters):
        from cofl.runtime.field_grid import predict_field_grid

        self._select_grid_backend()
        chunk_size = getattr(parameters, "field_query_chunk_size", DEFAULT_FIELD_QUERY_CHUNK_SIZE)
        field = predict_field_grid(
            self.model,
            prepared,
            kwargs,
            geometry=self.geometry,
            device=self.device,
            grid_size=getattr(parameters, "field_grid_size", DEFAULT_FIELD_GRID_SIZE),
            query_chunk_size=chunk_size,
        )
        self._field_grid_metadata = dict(field.metadata, query_chunk_size=chunk_size)
        return field

    def configure_sensors(self, sensors):
        """Condition queries on the actual camera FoV, including RxR's 79°.

        Radius and normalization belong to the checkpoint. Camera intrinsics
        belong to the observation and may vary without changing model weights.
        """
        self.geometry = configure_geometry_sensors(self.geometry, sensors)

    def predict_fields_batch(self, samples, parameters):
        """Encode observations together and return fields for local CPU rollout.

        Samples retain their own camera geometry and may use different sensor
        resolutions. Only decoder conditioning clamps HFOV to its trained band.
        The full batch service time is attached to every response, not divided
        by batch size; queue and transport time belong to the service caller.
        """
        import math

        import torch

        from cofl.runtime.field_grid import predict_field_grids
        from cofl.training.policy import encode_samples, query_policy_kwargs

        samples = [dict(sample, geometry=dict(sample["geometry"])) for sample in samples]
        if not samples:
            return []
        if any(sample["geometry"].get("profile") != "ground_sector_v1" for sample in samples):
            raise ValueError("CoFL-S field batches require ground_sector_v1 geometry")
        for sample in samples:
            hfov = sample["geometry"].get("hfov_rad")
            if (
                isinstance(hfov, (bool, np.bool_))
                or not isinstance(hfov, (int, float, np.integer, np.floating))
                or not math.isfinite(hfov)
                or not 0 < hfov < math.pi
            ):
                raise ValueError("Sample geometry.hfov_rad must be finite and within (0, pi)")
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        self._select_grid_backend()
        chunk_size = getattr(parameters, "field_query_chunk_size", DEFAULT_FIELD_QUERY_CHUNK_SIZE)
        with torch.inference_mode():
            kwargs = query_policy_kwargs(
                self.config, samples, device=self.device, dtype=torch.float32, inference=True
            )
            context = encode_samples(self.model, self.config, samples, self.device)
            prepared = self.model.prepare_query_context(context)
            fields = predict_field_grids(
                self.model,
                prepared,
                kwargs,
                geometries=[sample["geometry"] for sample in samples],
                device=self.device,
                grid_size=getattr(parameters, "field_grid_size", DEFAULT_FIELD_GRID_SIZE),
                query_chunk_size=chunk_size,
            )
            probabilities = self.model.action_logits(context).softmax(-1).cpu().numpy()
        if probabilities.shape != (len(samples), 4) or not np.isfinite(probabilities).all():
            raise ValueError("Policy action probabilities must be finite with shape [B,4]")
        elapsed_ms = (perf_counter() - started) * 1000
        self._field_grid_metadata = dict(fields[0].metadata, query_chunk_size=chunk_size)
        responses = []
        for field, actions in zip(fields, probabilities):
            metadata = dict(field.metadata, query_chunk_size=chunk_size)
            responses.append(
                {
                    "field_values": field.values,
                    "actions": actions.copy(),
                    "inference_ms": elapsed_ms,
                    "field_metadata": metadata,
                    "acceleration": dict(self.acceleration_status, field_grid=metadata),
                }
            )
        return responses

    def predict(self, image, depth, instruction, parameters):
        import torch

        from cofl.runtime.rollout import rollout_sector_field
        from cofl.training.policy import encode_samples, query_policy_kwargs

        sample = {
            "image": image,
            "depth": depth,
            "instruction": instruction,
            "geometry": self.geometry,
            "profile": "ground_sector_v1",
        }
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode():
            context = encode_samples(self.model, self.config, [sample], self.device)
            prepared = self.model.prepare_query_context(context)
            kwargs = query_policy_kwargs(
                self.config, [sample], device=self.device, dtype=torch.float32, inference=True
            )

            result = rollout_sector_field(
                self._grid_field(prepared, kwargs, parameters),
                geometry=self.geometry,
                max_steps=parameters.policy_steps,
                policy_dt=parameters.policy_dt,
                field_grid_size=getattr(parameters, "field_grid_size", DEFAULT_FIELD_GRID_SIZE),
                boundary="project",
                nonfinite="raise",
            )
            probabilities = self.model.action_logits(context).softmax(-1)[0].cpu().numpy()
        return {
            "trajectory": np.asarray(result.points),
            "actions": probabilities,
            "inference_ms": (perf_counter() - started) * 1000,
            "field_stop_reason": result.stop_reason,
        }

    def close(self):
        import gc

        self.model = None
        gc.collect()
        if str(self.device).startswith("cuda"):
            import torch

            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
