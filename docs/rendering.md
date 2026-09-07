# Rendering source scenes

`cofl data render` builds the RGB/semantic source files consumed by CoFL's
image-field generator. The complete chain is:

```text
official scene meshes + semantic annotations
  → cofl data render (Open3D, 640 × 480)
  → scene/annotations.json + top/view0…7 PNG and mask.npy
  → cofl data generate --config configs/generation/image.yaml
  → native observations, instructions, fields and trajectories
```

The renderers are adapted from the project's original
`Matterport3D/visualizer_v2.py` and `ScanNet/render_scannet_scenes.py`. Their
implementations now live in
[`rendering/matterport.py`](../src/cofl/generation/rendering/matterport.py) and
[`rendering/scannet.py`](../src/cofl/generation/rendering/scannet.py).
Each module records the original script's SHA-256. The port preserves camera
directions, zoom, RGB capture and semantic rasterization rules, removes machine
paths and implicit output writes, and makes missing required assets fatal.

## Obtain the upstream assets

Use the [official Matterport3D repository](https://github.com/niessner/Matterport)
and [official ScanNet repository](https://github.com/ScanNet/ScanNet) for dataset
access and download instructions. A repository clone alone does not include the
scene data. Obtain the assets through the respective dataset access process;
their data terms remain separate from CoFL's source code.

For Matterport3D, extract the `region_segmentations` archives. The original
renderer expects this extraction layout under `--root`:

```text
mp3d/v1/scans/
└── 17DRP5sb8fy/
    └── region_segmentations/
        └── 17DRP5sb8fy/
            └── region_segmentations/
                ├── region0.ply
                ├── region0.semseg.json
                ├── region0.fsegs.json
                └── ...
```

Each region mesh needs its face-segment map and semantic segment groups. See
the [official data organization](https://github.com/niessner/Matterport/blob/master/data_organization.md)
for the source formats. The extra nesting above reflects the archive extraction
used by this adapter; set up that layout before rendering.

For ScanNet, use annotated scans with all four files below:

```text
scans/
└── scene0000_00/
    ├── scene0000_00_vh_clean_2.ply
    ├── scene0000_00.aggregation.json
    ├── scene0000_00_vh_clean_2.0.010000.segs.json
    └── scene0000_00.txt
```

The renderer reads category strings directly from aggregation groups. A
`scannetv2-labels.combined.tsv` is not used by this implementation. The `.txt`
file supplies `axisAlignment` when present; a file without that entry uses the
identity transform. Unannotated test scans cannot generate these labels.

## Install and render

From the CoFL checkout, install the optional renderer into the locked host
environment:

```bash
uv sync --locked --extra rendering
```

This adds Open3D 0.19.0 to the host generation dependencies. It does not require
Habitat. The legacy Open3D `Visualizer` creates an invisible window but still
needs an OpenGL context. On a Linux machine without a desktop display, use an
X server such as Xvfb with a working GLX/Mesa setup, or configure a compatible
[Open3D headless build](https://www.open3d.org/docs/release/tutorial/visualization/headless_rendering.html).
A headless build is a different renderer environment and must be validated
separately. Installing the Python package alone does not supply system graphics
libraries or an X server.

To sample new views, start with one scene from each source:

```bash
uv run --locked --extra rendering cofl data render \
  --dataset matterport --root /path/to/mp3d/v1/scans \
  --output data/raw/matterport-rendered --scan-ids 17DRP5sb8fy --seed 42

uv run --locked --extra rendering cofl data render \
  --dataset scannet --root /path/to/scannet/scans \
  --output data/raw/scannet-rendered --scan-ids scene0000_00 --seed 42
```

For Xvfb, prefix either command with `xvfb-run -a`. For the CPU dependency group,
also pass `--no-group cu124 --group cpu` to each `uv sync`/`uv run` invocation.

Omit `--scan-ids` to select every discoverable scene. Use a new output directory
when changing the selection or seed; a preview directory cannot be resumed
with a different inventory. There are no automatic downloads or overwrites of
existing rendered directories. Source and output trees must not overlap.

## Replay existing camera records

Historical CoFL camera records are outside the first model/data release.
Supply a licensed portable camera catalog with `--cameras-from` to fix recorded
**intrinsics, extrinsics and image dimensions** instead of sampling camera pitch:

```bash
uv run --locked --extra rendering cofl data render \
  --dataset scannet --root /path/to/scannet/scans \
  --cameras-from /path/to/cameras.json \
  --scan-ids scene0000_00 --output data/raw/scannet-camera-replay

uv run --locked --extra rendering cofl data render \
  --dataset matterport --root /path/to/mp3d/v1/scans \
  --cameras-from /path/to/cameras.json \
  --scan-ids 17DRP5sb8fy --output data/raw/matterport-camera-replay
```

`--cameras-from /path/to/rendered-root` also reads the original
`<scan_id>/annotations.json` layout. Neither camera source requires the old
RGB/mask files. The legacy `--bundled-cameras` option remains available for local
checkouts that retain a historical catalog; the public packages omit that data.

The catalog format is
`cofl-render-cameras`, schema version 1, with records indexed as:

```text
datasets.matterport.scenes[scan_id].regions[region_id][view_id]
datasets.scannet.scenes[scan_id].regions.region0[view_id]
```

Each available record contains `width`, `height`, `intrinsic` and `extrinsic`;
a missing historical camera is explicitly `null`. Per-dataset counts and each
scene's source-annotation SHA-256 are included. Camera numbers are preserved
without rounding or orthogonalization. The catalog contains no RGB images,
semantic masks, object labels, meshes or machine-specific source paths.
Official scene meshes and semantic annotations are still external inputs.

Every selected source region must have all nine available camera records.
The entire selection is checked before any scene is rendered. Missing or
invalid records fail explicitly; replay never fills a missing camera randomly.
The runner reads a combined catalog once per invocation and fingerprints the
entire file for resume, then selects the requested scene records in memory.

Both renderer modules also accept the catalog directly from Python:

```python
from cofl.generation.rendering.scannet import render_scan

render_scan(
    "scene0000_00", "/path/to/scannet/scans", "/path/to/new-scene-output",
    cameras_from="/path/to/cameras.json",
)
```

`rendering.matterport.render_scan` and both renderer classes accept the same
camera-source options. The CLI adds source fingerprinting, staged publication
and resume around these low-level render functions. Omitting both camera-source
options retains seeded sampling through the CLI for generating new views.

To create a new catalog from your own per-scene annotations:

```bash
uv run --locked cofl data export-cameras \
  --matterport-root /path/to/matterport-rendered \
  --scannet-root /path/to/scannet-rendered \
  --output my-cameras.json
```

Supply one or both roots. Export uses stable ordering, retains source checksums,
does not overwrite an existing file, and does not import Open3D. Missing camera
fields become `null`; malformed cameras, duplicate identities and missing view
records are errors. The renderer's internal `.staging` directory is excluded.
This also makes the bundled catalog reproducible from the original annotations.

Historical records omit image dimensions. Both original renderers used
640×480, which is the fallback for these records; an explicit different size is
rejected. New annotations include width and height. All existing local source
PNGs were checked and use RGB at 640×480.

The same camera is restored for RGB capture, semantic capture and mesh changes.
After applying it, the renderer checks that Open3D returns the recorded
parameters within numerical tolerance. Keep the original Open3D convention:
legacy oblique-view extrinsic matrices can contain nonorthogonal up/front rows.
They are loaded without inversion or orthogonalization by CoFL, and passed to
Open3D's own camera conversion. A generic rigid-pose conversion would change
these records.

The camera JSON is fingerprinted alongside meshes and semantic inputs. A changed
camera file invalidates resume. `--seed` remains part of the receipt but does
not select camera angles in replay mode. Output cannot overlap the camera root.

The bundled historical inventory contains 33,093 complete camera records out of
33,363 views. ScanNet has all 13,617; Matterport has 19,476 of 19,746. The missing
270 are in `E9uDoFAP3SH`, regions 0–3 and 10–35. That scene cannot be fully
replayed from its current annotations. Retain its existing rendered images or
recover a camera-metadata backup; a new random view is not a historical replay.

## Outputs and resume

Matterport3D writes one scene directory with `annotations.json` and a directory
per region. ScanNet puts the views directly inside each scene directory:

```text
rendered-root/
├── rendering.json
└── scene0000_00/
    ├── annotations.json
    ├── render-report.json
    ├── top.png
    ├── top_mask.npy
    ├── top_mask.png
    ├── view0.png
    ├── view0_mask.npy
    ├── view0_mask.png
    └── ...                  # view1 through view7
```

The `.npy` dictionary contains an integer `[H,W]` mask and an integer-to-category
`label_map`; `-1` denotes unlabeled/background pixels. The colored `_mask.png`
files are previews. `annotations.json` records camera intrinsics, extrinsics and
image dimensions for every newly rendered view. Archive these files to enable
fixed-camera replay without retaining the original random-generator state.

In sampling mode, the runner derives a seed from the base seed, dataset and scan ID. Selecting
other scans does not change that scan's random camera choices. Matterport
regions are rendered in sorted order within a scene; changing its region
inventory can change later regions' random draws and label IDs.

Every scene is built in `.staging`, checked for all nine RGB/mask/camera records
per view directory, and renamed into place after success. `rendering.json`
records the selection, seed, source/camera roots, camera mode, code fingerprint,
Python/package versions and completion status. Each scene report records its source file
fingerprints, derived seed and output checksums.

Repeat the same command with `--resume` to verify and reuse completed scenes.
An interrupted scene is rendered again in full. Source, recipe, code or recorded
environment changes require a new output; altered finalized payloads are
rejected. Existing historical directories without these receipts cannot be
adopted by `render --resume`, but remain valid inputs to `data generate`.
Run label generation only after rendering completes for the selected inventory;
ImagePipeline itself does not enforce the rendering journal's status.

The receipt records Python package versions, not a complete GPU/driver/OpenGL
environment. Checksums prove which outputs were retained, not that another
graphics stack will recreate their pixels.

## Generate the native dataset

Point `pipeline.data_dirs` in your image recipe at the rendered roots. The
bundled `configs/generation/image.yaml` already resolves its defaults to
`data/raw/matterport-rendered` and `data/raw/scannet-rendered` in the checkout.
When copying a recipe elsewhere, adjust its relative paths or use absolute ones.
List only the roots you actually have.

```bash
uv run --locked --extra rendering cofl data generate \
  --config configs/generation/image.yaml --output runs/image-preview --limit-units 1
uv run --locked --extra rendering cofl data verify-generation runs/image-preview
```

Continue with the [generation guide](generation.md) for split selection,
trajectory settings, process shards and full native-dataset verification.
Open3D is only used by the preceding `render` step.

## Verification and limits

The bundled catalog was checked against all 1,603 source annotation files:
source hashes match, and all 827,325 matrix float64 values are preserved bit for
bit. A second export with the input roots supplied in reverse order produced
identical file bytes. The wheel and source archive contain those same catalog
bytes, and the extracted wheel can load the resource outside the checkout.
Using `--bundled-cameras` for the actual ScanNet scene below produced the same
nine RGB/mask pairs as replaying its original annotations directory.

The port retains the historical red-channel semantic encoding, including its
255-value ceiling. Matterport uses face labels; ScanNet uses the first vertex's
label for each face and fills interior unlabeled regions, preserving background
connected to the image boundary. Changing these rules would define new labels.

The separate historical `ScanNet/postprocess_masks.py` algorithm is preserved
as [`rendering.postprocess.process_mask`](../src/cofl/generation/rendering/postprocess.py).
It is an optional array-level API, not an automatic extra rendering pass. Its
default `all` method adds morphological closing and a 500-pixel hole threshold;
the renderer's built-in cleanup uses a 1000-pixel threshold without that closing
step. Applying this extra cleanup changes the dataset and should be tracked as
a separate revision, with RGB, annotations and metadata preserved by the caller.

Automated tests use small local fixtures for scene publication, seed isolation,
resume and damaged-input rejection. A local Open3D 0.19.0 check under Xvfb also
rendered one synthetic floor/object scene in each official input layout: nine
views per scene, exact same-environment repeat output, verified resume, and
successful downstream native generation (three field/trajectory annotations
per top view, including full payload verification). This did not render the
official full corpus or compare it against historical images. Real OpenGL
rendering must still be checked in the intended graphics environment.
Fixed-camera replay was also checked on both synthetic scenes with a different
base seed and random-angle sampling disabled. All 18 replayed RGB images and
semantic masks matched the originals pixel for pixel on the same graphics stack.

One actual historical ScanNet scene (`scene0000_00`, nine views) was then
replayed through the CLI from its official mesh and archived camera records.
Intrinsics matched exactly and the maximum extrinsic difference was below
9e-16. Five masks matched exactly; each of the other four differed at one pixel,
and all label maps matched. RGB pixels were not identical (per-image mean
absolute channel error 0.026–0.030 on the 0–255 scale). The cause of those
historical pixel differences has not been isolated. This confirms that camera
reuse works while historical pixel equality still needs its own verification.

These checks do not establish a full historical-corpus match across graphics
environments or reconstruct downstream random label choices.
See the
[historical reproduction boundary](generation.md#historical-datasets-versus-a-new-deterministic-generation)
before treating regenerated samples as an existing dataset release.
