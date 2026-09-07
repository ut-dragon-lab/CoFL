# CoFLDataset v1

CoFLDataset separates observation sequences from instruction-conditioned field
annotations. The profiles are `image_field_v1` for CoFL and `ground_sector_v1`
for CoFL-S. Native storage uses Parquet and Zarr 2. An optional LeRobot v3 view
joins observations written through LeRobot's API to a CoFL annotation extension.
The [locked uv environment](../README.md#installation) includes native data
tools. Run the commands below from the repository root and Python examples
with `uv run --locked python`.

## Records and identities

| Record | Required meaning | Optional information |
| --- | --- | --- |
| Episode | Source sequence key, task and split | JSON metadata |
| Observation | Episode, source frame index, RGB image | Timestamp, metric depth, executed action, metadata and extra arrays |
| Annotation | Observation, instruction and provenance, field, mask, geometry, anchor flag, inherited split | Target action, metadata and extra arrays |

“Approach the door” and “Turn toward the table” can annotate the same frame.
They form two samples referencing one observation, without creating additional
frames or executed transitions. Observations without annotations are permitted.

IDs derive from dataset ID, immutable revision and source keys; annotation IDs
also include the recipe and annotation key. Record order and shard boundaries
do not determine IDs. Changed source data or labels require a new revision.
All annotations inherit their episode's split.

Native frame indices may have explicit gaps. Timestamps must be absent throughout
an episode or finite, nonnegative and strictly increasing. Stationary turn frames
remain distinct observations. `executed_action` describes the source action
following an observation; a final STOP may label a terminal observation.
`target_action` belongs to its annotation's instruction. Missing actions are
`None`. Nonnegative action IDs are defined by the source and recipe.

## Native storage

```text
dataset/
  manifest.json
  episodes.parquet
  observations.parquet
  annotations.parquet
  arrays.zarr/
    images               uint8 [O,H_image,W_image,3], or JPEG bytes [O]
    depths               float32/float16 [D,H_image,W_image], optional
    depth_masks          bool [D,H_image,W_image], optional
    fields               float32/float16 [A,2,H_field,W_field]
    masks                bool [A,H_field,W_field]
    observation_extras/  optional numeric/bool arrays, one row per observation
    annotation_extras/   optional numeric/bool arrays, one row per annotation
```

`O`, `D` and `A` count observations, observations with depth, and annotations.
Parquet rows reference arrays using `image_index`, optional `depth_index` and
`field_index`. Extras use their source image/field index. Each extra name, dtype
and trailing shape is fixed within a shard. Every record has a `metadata_json`
string column; unspecified metadata is the JSON object `{}`.

The manifest declares `format="cofl_dataset"`, `schema_version="1.0"`, profile,
dataset ID, revision, source and recipe objects with `id`/`version`, storage and
record counts. Storage explicitly includes `image_encoding`, `field_dtype` and
`depth_dtype`, including when writer defaults are used. Readers require the
declared Parquet record schemas and Zarr v2
metadata. Geometry is never inferred from dimensions or masks.

`DatasetWriter` defaults to dense RGB, float32 fields/depth and one numerical
row per compressed chunk. Optional settings are `image_encoding="jpeg"`,
`field_dtype="float16"`, `depth_dtype="float16"` and `chunk_rows`. JPEG input is
encoded bytes, stored unchanged and decoded as uint8 HWC when read. JPEG chunks
contain up to 64 records. Float16 storage rejects conversions that change input
values. Zarr compression is lossless. Images and each array have fixed trailing
shapes within a shard; separate shards can retain different image dimensions.

Depth is in metres; unavailable depth can be zero. `depth_valid=True` requires
finite positive depth. Field `mask=True` means that a cell contributes
supervision. Navigation and walkability masks can be extras with separate
meanings. Fields must be finite even outside their supervision mask, and every
annotation must supervise at least one cell.

## Collections and reading

```text
collection/
  collection.json
  shard-000000/  # complete native dataset
  shard-000001/
```

`publish_collection` indexes completed shards or child collections with the same
profile. The `cofl_dataset_collection` schema, version `1.0`, records portable
relative paths, manifest SHA-256 hashes, record counts, per-split counts and
eligible sample counts. Incomplete collections cannot be opened. A partial
child keeps its parent marked `source.partial=true`.

```python
from cofl.data import open_dataset, validate_collection

print(validate_collection("datasets/cofl-s"))
dataset = open_dataset("datasets/cofl-s", split="train", include_extras=False)
sample = dataset[0]
image = sample["image"]  # uint8 HWC
field = sample["field"]  # float32 or float16, 2HW
mask = sample["mask"]    # bool HW
```

`CoFLDataset` opens one shard. `open_dataset` opens a shard or collection, loading
children lazily with compact Arrow metadata and bounded decoded-array caches.
Cached native arrays are read-only; copy before in-place augmentation.
`include_extras=False` skips extra arrays while retaining JSON metadata.
`eligible_only=True` excludes annotations whose metadata explicitly sets
`training_eligible=false`.

Scene generation is described in [generation.md](generation.md).

Create datasets through `DatasetWriter.add_episode`, `add_observation`,
`add_annotation` and `finalize`; output must be absent or empty. Validation checks
record schemas, identities, references, array bounds, splits, time ordering,
masks, fields, metric depth and geometry. `validate_collection` checks child
manifest hashes and counts, then validates every child payload. Opening a
collection checks accessed child manifests; it does not hash every array byte.

## Coordinate profiles

| Contract | `image_field_v1` | `ground_sector_v1` |
| --- | --- | --- |
| Observation view | BEV RGB | Egocentric RGB, optional metric depth |
| Vector components | right, down | forward, left |
| Grid rows, columns | down, right | forward, left |
| Cartesian `grid_bounds` | `[[0,1],[0,1]]` | `[[0,1],[-1,1]]` |
| `query_axes` | right, down | theta_normalized, radius_normalized |
| Model `query_bounds` | `[[0,1],[0,1]]` | `[[-1,1],[0,1]]` |
| Vector unit | Image fractions per policy time | Metres / normalization scale per policy time |

Geometry includes `profile`, `grid_shape`, `field_layout`, `coordinate_frame`,
`observation_view`, `component_axes`, `grid_axes`, `grid_bounds`, `query_axes`,
`query_bounds`, `grid_alignment`, `velocity_unit` and `time_convention`.
`field_layout` is `channels_height_width`; `grid_alignment` is
`inclusive_endpoints`. Each grid dimension must be at least two.

`grid_bounds` follows vector component order. Image columns run rightward from
zero to one and rows downward from zero to one. Ground rows run forward from
zero to one; columns run leftward from minus one to one.

Ground geometry also requires finite positive `normalization_scale_m`,
`r_max_m` and `hfov_rad`, with `r_max_m == normalization_scale_m` and
`hfov_rad < pi`. A Cartesian grid position maps to model queries as:

```text
theta_norm = atan2(left, forward) / (hfov_rad / 2)
r_norm = hypot(forward, left) * normalization_scale_m / r_max_m
```

`cofl.fields.queries_for_grid` returns model queries; `query_valid_mask` selects
`abs(theta_norm) <= 1` and `r_norm <= 1` with numerical boundary tolerance.
Ground supervision masks must stay inside this sector. Vectors retain their
Cartesian forward/left components after query conversion.

Both profiles describe a static velocity per unit **policy integration time**.
This integration parameter is distinct from timestamps and simulator steps.
A normalized field alone does not specify physical actuator speeds.

## Exact comparison

```bash
uv run --locked cofl data compare data/first data/second --output reports/comparison.json
```

Comparison validates both inputs and supports shards and recursive collections.
Collections require the same shard layout and order; arbitrary reshards are not
joined. Reports distinguish:

- `semantic_equal`: exact manifest, table schema/metadata, identity-associated
  array values, dtypes/shapes and Zarr attributes. Physical row offsets,
  annotation table order and compression do not affect this result.
- `sample_order_equal`: annotation IDs appear in the same reader order,
  preserving the mapping from seeded indices to examples.
- `byte_equal`: every relative file path and its contents match, independently
  of timestamps and permissions.

The CLI succeeds when semantics and sample order match. Each side includes
`semantic_sha256`, `sample_order_sha256`, `byte_sha256` and detailed fingerprints.
No floating-point tolerance is used. Save reports outside both inputs.
`compare_datasets` and `write_comparison_report` expose the same Python API.
Matching seeds alone do not establish equality when assets, recipes or runtimes
change.

## Optional LeRobot view

For an individual dense-RGB shard without extra arrays:

```python
from cofl.data import export_lerobot, LeRobotCoFLDataset

export_lerobot(
    "datasets/dense-rgb-shard", "datasets/cofl-s-lerobot",
    repo_id="local/cofl-s", fps=10,
)
dataset = LeRobotCoFLDataset("datasets/cofl-s-lerobot", split="train")
```

Export uses LeRobot's `create`, `add_frame`, `save_episode` and `finalize` API
with `use_videos=False`. Its base stores observations once, tasks, optional
executed actions, `observation.images.camera`, and optional
`observation.depth`/`observation.depth_valid`. The `cofl/` extension stores fields,
masks, instructions, splits and stable observation/frame mappings.
`LeRobotCoFLDataset` performs the join; a standard LeRobot policy loader does not
automatically consume these annotations.

Image-profile observations without executed actions become separate one-frame
LeRobot episodes, retaining source grouping in the extension. Temporal episodes
require contiguous source frames and timestamps matching the integer FPS.
Actions and depth must each be present everywhere or absent throughout.
Generated target actions never fill executed actions.

The bridge accepts one dense-RGB native shard. Collections, JPEG storage and
extra arrays are rejected. Optional JSON metadata remains in extension tables
but is not exposed by its sample reader. Exports are local; automatic streaming,
Hub upload and remote extension resolution are outside this API.

The optional LeRobot SDK is outside `uv.lock` and is not required for native
training or evaluation. Bridge checks have exercised dense RGB, float32 depth
where present, fields, masks, actions and static/temporal mappings against
LeRobot 0.6.1 with v3.0 metadata. They do not establish a supported, fully
resolved LeRobot environment. The two SDK roundtrip tests are skipped when
that optional package is unavailable. Float16 depth and broader LeRobot
features are outside the tested scope; use a separately resolved environment
for work on this bridge.
