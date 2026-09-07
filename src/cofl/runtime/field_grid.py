"""Complete policy query rasters and CPU streamline field interpolation.

Image rasters use rows=down, columns=right over [0,1]^2. Ground policy
rasters use rows=normalized radius, columns=normalized theta, independently
of the Cartesian BEV grid used by ground dataset annotations. Ground field
components remain Cartesian (forward,left), never polar derivatives.
"""

from __future__ import annotations

import numpy as np

from .inference_rules import DEFAULT_FIELD_GRID_SIZE, DEFAULT_FIELD_QUERY_CHUNK_SIZE

_PROFILES = {"image_field_v1", "ground_sector_v1"}


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def field_grid_queries(profile, *, grid_size=DEFAULT_FIELD_GRID_SIZE):
    """Return the full FP32 endpoint lattice as [H*W,2] policy-chart queries."""
    import torch

    if profile not in _PROFILES:
        raise ValueError(f"Unsupported policy field profile: {profile!r}")
    size = _positive_integer(grid_size, "grid_size", minimum=2)
    if profile == "ground_sector_v1":
        from cofl.models.cofl_s import CoFLSFieldDecoder

        return CoFLSFieldDecoder.build_sector_query_lattice(
            size, size, device=torch.device("cpu"), dtype=torch.float32
        ).numpy()
    # Preserve the established offline image lattice's FP32 rounding exactly.
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.stack((x, y), axis=-1).reshape(-1, 2)


class RasterField:
    """CPU bilinear/border readout from one complete [H,W,2] policy raster."""

    def __init__(self, values, *, profile):
        import torch

        if profile not in _PROFILES:
            raise ValueError(f"Unsupported policy field profile: {profile!r}")
        values = np.asarray(values)
        if values.ndim != 3 or values.shape[-1] != 2 or min(values.shape[:2]) < 2:
            raise ValueError("Field grid must have shape [H,W,2] with H,W >= 2")
        if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Field grid must contain float32 or float64 vectors")
        if not np.isfinite(values).all():
            raise ValueError("Predicted field grid contains a nonfinite vector")
        self.profile = profile
        self.values = np.array(values, copy=True, order="C")
        self._tensor = torch.from_numpy(self.values).permute(2, 0, 1).unsqueeze(0)

    @property
    def metadata(self):
        ground = self.profile == "ground_sector_v1"
        return {
            "representation": "dense_query_grid",
            "query_grid_shape": list(self.values.shape[:2]),
            "query_grid_axes": ["radius_normalized", "theta_normalized"]
            if ground
            else ["down", "right"],
            "query_grid_bounds": [[0.0, 1.0], [-1.0, 1.0]] if ground else [[0.0, 1.0], [0.0, 1.0]],
            "interpolation": "bilinear",
            "padding_mode": "border",
            "align_corners": True,
        }

    def __call__(self, queries):
        import torch
        from torch.nn import functional as F

        queries = np.asarray(queries)
        if queries.ndim != 2 or queries.shape[-1] != 2 or not np.isfinite(queries).all():
            raise ValueError("Field queries must be a finite array of shape [N,2]")
        if not len(queries):
            return np.empty((0, 2), dtype=self.values.dtype)
        with torch.inference_mode():
            coordinates = torch.as_tensor(queries, dtype=self._tensor.dtype, device="cpu")
            if self.profile == "ground_sector_v1":
                coordinates = torch.stack((coordinates[:, 0], 2 * coordinates[:, 1] - 1), dim=-1)
            else:
                coordinates = 2 * coordinates - 1
            sampled = F.grid_sample(
                self._tensor,
                coordinates.reshape(1, 1, -1, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return sampled[0, :, 0].transpose(0, 1).contiguous().numpy()


def materialize_field_grid(field, *, profile, grid_size=DEFAULT_FIELD_GRID_SIZE):
    """Call ``field`` once on the complete lattice and return a CPU interpolator."""
    queries = field_grid_queries(profile, grid_size=grid_size)
    values = np.asarray(field(queries))
    if values.shape != queries.shape:
        raise ValueError("Policy field output must have shape [grid_size**2,2]")
    return RasterField(values.reshape(grid_size, grid_size, 2), profile=profile)


def predict_field_grid(
    model,
    prepared,
    query_kwargs,
    *,
    geometry,
    device,
    grid_size=DEFAULT_FIELD_GRID_SIZE,
    query_chunk_size=DEFAULT_FIELD_QUERY_CHUNK_SIZE,
):
    """Query a full lattice in GPU/CPU batches with one final host transfer.

    Encoding and prepared-context construction belong to the caller, so every
    grid chunk shares the same observation context. No decoder calls occur
    when the returned field is sampled during integration.
    """
    import torch

    chunk_size = _positive_integer(query_chunk_size, "query_chunk_size")

    def predict(queries):
        with torch.inference_mode():
            queries = torch.as_tensor(queries, device=device, dtype=torch.float32)[None]
            output = torch.empty((1, queries.shape[1], 2), dtype=torch.float32, device=device)
            for start in range(0, queries.shape[1], chunk_size):
                stop = min(start + chunk_size, queries.shape[1])
                vectors = model.query(queries[:, start:stop], prepared, **query_kwargs)
                if vectors.shape != (1, stop - start, 2):
                    raise ValueError("Policy field output must have shape [1,N,2]")
                output[:, start:stop] = vectors
            return output[0].cpu().numpy()

    return materialize_field_grid(predict, profile=geometry["profile"], grid_size=grid_size)


def predict_field_grids(
    model,
    prepared,
    query_kwargs,
    *,
    geometries,
    device,
    grid_size=DEFAULT_FIELD_GRID_SIZE,
    query_chunk_size=DEFAULT_FIELD_QUERY_CHUNK_SIZE,
):
    """Decode complete fields for one observation batch, with one host transfer.

    Every decoder call receives [B,N,2] queries and the same prepared batch.
    Geometry conditioning such as per-observation HFOV belongs in query_kwargs;
    all observations share the profile lattice and configured raster dimensions.
    """
    import torch

    chunk_size = _positive_integer(query_chunk_size, "query_chunk_size")
    geometries = list(geometries)
    if not geometries:
        raise ValueError("A field batch requires at least one geometry")
    profile = geometries[0]["profile"]
    if any(geometry["profile"] != profile for geometry in geometries):
        raise ValueError("A field batch must use one policy field profile")
    lattice = field_grid_queries(profile, grid_size=grid_size)
    batch_size, query_count = len(geometries), len(lattice)
    with torch.inference_mode():
        queries = torch.as_tensor(lattice, device=device, dtype=torch.float32)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        output = torch.empty((batch_size, query_count, 2), dtype=torch.float32, device=device)
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            vectors = model.query(queries[:, start:stop], prepared, **query_kwargs)
            if vectors.shape != (batch_size, stop - start, 2):
                raise ValueError("Policy field output must have shape [B,N,2]")
            output[:, start:stop] = vectors
        values = output.cpu().numpy().reshape(batch_size, grid_size, grid_size, 2)
    return [RasterField(value, profile=profile) for value in values]
