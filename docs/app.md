# CoFL Studio

Studio opens with a choice of **BEV** or **2D egocentric**. Each domain owns
its compatible resources and tasks:

| Domain | Method / profile | Tasks |
| --- | --- | --- |
| BEV (`bev`) | CoFL / `image_field_v1` | **Scene Playground** (`scene`): GLB camera views and field trajectories; **Val Inspector** (`validation`): native image-field samples and predictions |
| 2D egocentric (`ego2d`) | CoFL-S / `ground_sector_v1` | **Interactive Playground** (`ego2d-interactive`): explore a real Habitat scene with live commands; **VLN Benchmark** (`ego2d-playground`): fixed episodes with oracle instruction progress and metrics |

The IDs in parentheses are API mode IDs. Entering the egocentric workspace opens
the Interactive Playground. Ground-sector dataset inspection remains available
through the native dataset API; it is no longer an egocentric navigation entry.
No simulator is needed for BEV validation or the example scene.

## Run locally

Use Python 3.12 and the repository's documented `uv` environment. Building the
browser assets requires Node.js 20.19+ or 22.12+ and npm. From the CoFL repository:

```bash
uv sync --locked
uv run --locked cofl app --build
```

The launcher builds the frontend using `npm ci` and `npm run build`, starts the
local server, and opens `http://127.0.0.1:8787`. Subsequent starts can omit
`--build`. The BEV example building lets users explore the interface immediately;
real predictions require a self-contained checkpoint for the selected domain.

You can connect resources directly from the interface without restarting:

1. Choose **BEV** or **2D egocentric** on the landing page. Use the navigation
   rail to change domains and tasks. Each workspace remembers its selections.
2. Click **Load policy** beside the policy selector. Browse folders, use the
   working-directory/home/filesystem shortcuts, or paste a checkpoint path and
   click **Go**. Select a `.ckpt`, `.pt`, or `.pth` file and click **Load policy**.
   Studio validates the checkpoint's method and profile for the selected domain,
   loads it on the configured device, and selects it when loading succeeds.
3. Open **BEV → Val Inspector → Open dataset**. Navigate into a native dataset or
   collection directory, choose its split (default `val`), and click
   **Open dataset**. Studio checks the split and reads its first sample, then
   opens the inspector at sample zero. Ground-truth inspection works without a
   policy checkpoint. Choose the split present in the dataset.
4. For either **2D egocentric** task, use **Open benchmark** to choose a
   `.yaml`, `.yml`, or `.json` benchmark recipe. This selects a configuration
   file, not a validation dataset folder; its split belongs in the recipe. See
   [online configuration](#online-environment-and-benchmark-configuration) below.
5. Use the resource selectors to switch between connected resources.
   **Resources** opens the picker for the current task. The picker remembers
   the most recent folder for each domain/resource type while the page is open.

The picker browses the machine running Studio; with a remote server, these are
the server's paths. Files are read in place, so large weights and datasets are
not copied through the browser. Display names are optional. Invalid files or
splits stay in the picker with an error and are not added to the catalog. A valid
benchmark recipe can be registered before its simulator assets are available;
the selected task displays the availability reason and disables execution.
BEV offers checkpoint/dataset resources; 2D egocentric also offers benchmarks.
Resource selectors filter known method/profile/domain metadata, and the backend
checks compatibility again when registering resources and executing requests.
Renaming a checkpoint or changing a domain hint cannot bypass that validation.

Connections remain available across browser refreshes for the running Studio
session. To restore them after a server restart, use the YAML catalog below or
the optional command-line flags:

```bash
uv run --locked cofl app \
  --scene /data/scenes/room.glb \
  --checkpoint /data/checkpoints/cofl.ckpt \
  --dataset /data/cofl-validation \
  --device cuda:0
```

Startup catalogs do not eagerly load every model or dataset. Entries with
unverified metadata can appear as candidates; selecting one calls
`POST /api/v1/resources/{kind}/{resource_id}/activate` with the selected
`domain_id`. Successful activation verifies and binds its method/profile before
the UI uses it. An optional `domain_id` in YAML helps place the candidate in the
right workspace, but does not replace validation.

`--dataset` accepts a native dataset revision or collection. Studio defaults to
`--device cuda:0`. `--no-browser`, `--host`, and `--port` control the launcher.
The workspace toolbar shows the selected **GPU** or **CPU** device.

If the selected GPU is unavailable, or model execution reports a CUDA failure
such as an out-of-memory error, Studio asks whether to **Use CPU**. **Keep GPU**
keeps the GPU selected so you can resolve the problem and retry. Dismissing a
failure does not repeatedly reopen the same prompt while the page is open;
**Review device** lets you reconsider. The landing page and scene browsing do
not require CPU fallback approval.

CPU is selected only after an explicit **Use CPU** confirmation or an explicit
`--device cpu` / `device: cpu` configuration. Changing devices releases the
loaded model cache; it does not automatically repeat a failed operation. After
the confirmation succeeds, run that operation again. A failed device change
retains the previous selection. **Use GPU** returns to `cuda:0`; finish any
current inference operation or end the active online session before changing
devices. Runtime selection lasts for the running Studio process; use the
startup configuration to preserve a choice across server restarts.

The shared `GET /api/v1/runtime/device` reports the selected device, GPU
availability, latest GPU failure and whether switching is currently allowed.
`POST /api/v1/runtime/device` with an explicit `device` updates both static
inference and online sessions. Failed requests preserve the current selection.

The CPU dependency group selects installed Torch packages; explicitly select
CPU execution too when launching with a CPU-only environment:

```bash
uv sync --locked --no-group cu124 --group cpu
uv run --locked --no-group cu124 --group cpu cofl app --build --device cpu
```

The Python application dependencies are also available as
`cofl-navigation[app]`. A source checkout is needed for `--build`; an installed
wheel must already contain the built frontend assets.

## Close Studio and release resources

Click **Shut down Studio** in the navigation footer from the landing page or any
workspace. This ends active sessions and unloads model, observation and dataset
resources. The browser immediately leaves the workspace and stops its polling
and other query requests while the server completes cleanup.

The page shows **Studio closed** only after the server confirms cleanup. You can
then close the tab. The standard `cofl app` launcher stops its server process;
when Studio is embedded in another server, Studio closes while the host remains
available. Reopen Studio by starting the launcher again.

If the shutdown request fails or the connection drops before acknowledgement,
the page reports **Shutdown not confirmed** and offers **Retry shutdown**.
Workspace requests stay stopped; a lost connection is not treated as proof that
cleanup finished. `POST /api/v1/runtime/shutdown` provides the acknowledgement,
including whether the launcher will stop its server.

Closing or refreshing a browser tab by itself does not shut down Studio. This
preserves active sessions when reconnecting and avoids affecting other tabs.
Use the explicit shutdown button when you want to release the server's resources.

## Configure reusable resources and floors

Use `cofl app --config studio.yaml` for a named catalog. Paths are resolved
relative to the YAML file. IDs are stable API identifiers; labels are display
names. Command-line resource flags add or replace the corresponding `local` ID.

```yaml
device: cuda:0

models:
  aerial:
    label: CoFL
    path: checkpoints/cofl.ckpt
    domain_id: bev
  ground:
    label: CoFL-S
    path: checkpoints/cofl-s.ckpt
    domain_id: ego2d

datasets:
  aerial-val:
    label: Aerial validation
    path: datasets/image-field
    split: val
    domain_id: bev

benchmarks:
  ground-online:
    label: Ground VLN episodes
    path: benchmarks/ground-online.yaml
    domain_id: ego2d

scenes:
  house:
    label: Two-storey house
    path: scenes/house.glb
    up_axis: y
    floors:
      - id: ground
        label: Ground floor
        elevation: 0.0
        min_height: -0.15
        max_height: 2.5
      - id: upper
        label: Upper floor
        elevation: 3.5
        min_height: 3.35
        max_height: 6.0
```

Scene files must be self-contained `.glb` assets with embedded textures. glTF
normally uses Y-up; an explicitly configured Z-up asset is rotated at the scene
root. Floor heights refer to the resulting Y-up scene, in the asset's world
units. Correctly scaled glTF scenes use metres.

The legacy Matterport GLBs used by Habitat can instead contain Z-up mesh data.
For those files, set `up_axis: z`, launch with `--scene house.glb --scene-up-axis z`,
or use the **Up axis** selector in the browser. Axis changes reset the floor
suggestions, start, and prediction so all coordinates remain aligned.

Draco, KTX2, and the legacy `GOOGLE_texture_basis` textures found in Habitat's
Matterport GLBs use the upstream decoders distributed with Three.js. The build
copies these assets into Studio, so loading scenes does not require a decoder
CDN. A small loader adapter bridges the legacy Basis extension. See
[`app/THIRD_PARTY.md`](../app/THIRD_PARTY.md) for attribution. Local GLBs can also
be opened directly in the browser with the upload control.

Without configured floors, Studio suggests levels from approximately horizontal
upward-facing mesh surfaces. These are geometric suggestions: furniture,
ceilings, uneven scans, and split levels can confuse them. Explicit floor
configuration supplies reliable boundaries. The cutaway retains the selected
height interval, hiding both upper structures and lower floors.

## BEV Playground

Orbit, pan, and zoom to choose an observation. Select a floor, adjust its
cutaway, and click an exposed area of its floor to select the start. Enter an
instruction and request a prediction. The observation sent to the model is the
rendered scene with the chosen cutaway; start markers and prediction overlays
are removed during capture.

CoFL returns an image-coordinate field. The displayed scene trajectory is
projected back onto the selected floor plane using the camera that produced
that observation. This assumes a locally planar floor. It does not establish
collision freedom, snap predictions to a navigation mesh, or implement stair
traversal. Multiple-floor browsing and cross-floor navigation are separate
capabilities.

Free camera movement can produce views outside the checkpoint's training
distribution. Use the top-view/reset controls to return to a reproducible view.
The browser renders interactively; inference latency depends on the selected
checkpoint and device.

## BEV Val Inspector and static coordinate contracts

The inspector reads the original RGB image, instruction, mask, and field from
the native reader. The static dataset API also supports CoFL-S using stored depth
and the declared sector geometry, even though it has no built-in Val navigation
entry. It does not derive CoFL-S observations from RGB-only scene captures.

| Profile | Displayed query/start/trajectory coordinates | Displayed field vectors |
| --- | --- | --- |
| `image_field_v1` | Image fractions `(right, down)` in `[0,1]²` | Image fractions per policy time |
| `ground_sector_v1` | Metres `(forward, left)` in the fixed observation body frame | Metres per policy time |

Ground decoder queries internally use normalized `(angle, radius)`. The service
uses the existing profile conversion helpers; it converts returned positions
and vectors to metric Cartesian values for display. Policy time is an
integration parameter, not wall-clock seconds. The CoFL-S action output is four
raw logits in `STOP, FORWARD, LEFT, RIGHT` order, identified by
`action_semantics` in the response.

Static and online trajectories use the shared `cofl-inference-v1` rules, also
recorded in each prediction's `inference_rules` metadata. They first build a
100×100 field and sample it on CPU with bilinear interpolation. Each of the
`T` Euler updates uses effective step
`policy_dt / ((1 - t) + bias * t**10)`, with `t = linspace(0, 1, T)` and bias
`0.5` for image fields or `0.2` for ground fields. Updates are projected into
the image box or ground box/physical sector, and all `T` steps run even at a
boundary or in a zero field. The default is 100 updates.

Ground starts default to the centerline at normalized radius
`max(0.001, 0.5 / (100 - 1))`, about 2.53 cm for a 5 m field. The Inspector
leaves the ground start unset until you choose a location; the backend applies
that default. Explicit starts use at least half a radial grid cell. Decoder
HFOV conditioning is limited to 79–90 degrees during inference, while camera
geometry, displayed coordinates and sector boundaries keep the actual HFOV.
These rules do not enable a separate low-speed endpoint postprocess.

GT previews select at most 1,024 actual supervised cells. Prediction requests
select at most `grid_size²` native supervised cells, with `grid_size` in 8–64.
Metrics use the shared `evaluate_field` implementation and explicitly report
`scope: sampled_supervised_cells`, the selected count, and the available
supervised count. They are sample previews, not a replacement for the offline
evaluation command. Editing the instruction removes GT metrics for that
prediction because the original annotation does not supervise the edited text.

Stored image trajectories use their declared valid prefix. Stored
`trajectory_body_m` ground trajectories are already metric and are not scaled
again. If a sample has no trajectory label, Studio reports it as unavailable.

## 2D egocentric Interactive Playground

Select a CoFL-S checkpoint, use **Open benchmark** to connect an environment
recipe, choose a **Navigation scene**, then click **Open scene**. An initial
RGB-D observation appears before any instruction or policy prediction. You can
open the scene with an empty instruction and decide what to do after seeing it.

Type an instruction and click **Send command**, or press Enter; Shift + Enter
inserts a newline. Commands take effect at the next planner boundary. The
**Current instruction** panel shows the instruction used by the policy, with a
queued command shown separately until it takes effect. **Pause** freezes motion;
**Resume** continues. A policy STOP or per-command limit returns the session to
waiting for the next command rather than closing the scene.

Expand **Starting position & heading** to edit world XYZ metres and heading in
degrees. Leave all XYZ fields empty to request a navigable spawn point. **Reset
pose** restarts the scene at the chosen pose and clears the current instruction;
the response's `epoch` separates pre-reset predictions from the new observation.
The UI surfaces reset errors. **End session** releases the simulator and policy.
Free exploration has no goal-conditioned benchmark score.

The display shares a current-instruction panel, camera, local sector plan and
world-XZ path with VLN Benchmark. Command history records recent interactive
commands. Only one online session owns the simulator at a time. Switching views
or refreshing the page reconnects to it; **Open active session** returns to its
task, and **End active session** releases it. Static inference and policy loading
wait until online ownership is released.

## VLN Benchmark

Choose **2D egocentric → VLN Benchmark**, connect a recipe with episode data,
select an episode and simulation time limit, then click **Run episode**. The
**Global task** remains the full episode instruction; **Current instruction**
shows the oracle-selected local instruction actually supplied to the policy.
This progress oracle uses reference path annotations to select language. It does
not substitute a reference trajectory for CoFL-S predictions.

The view reports the executed and planned paths and final metrics. **Stop
execution** cancels the episode. **Explore this scene** ends an active benchmark
and opens an interactive session in the same scene at the robot's current pose,
ready for your commands. This new session does not continue benchmark scoring.

## Online environment and benchmark configuration

A benchmark recipe supplies simulator and episode assets independently of the
Studio catalog. Save a file such as `benchmarks/ground-online.yaml`:

```yaml
online:
  habitat_config: /data/habitat/tasks/ground-vln.yaml
  dataset_data_path: /data/vln/{split}/{split}.json.gz
  gt_path: /data/vln/{split}/{split}_gt.json.gz
  ndtw_fdtw: true
  scenes_dir: /data/scene_datasets
  habitat_python: /opt/habitat-env/bin/python
  split: val_unseen
  language_filter: en
  gpu_id: 0
  runtime_pythonpath: []
  instruction_mode: oracle
  oracle_source: auto
  oracle_unavailable: error
  progress_lookahead: 6
  fgr2r_json: /data/annotations/FGR2R.json
  # landmark_rxr_json: /data/annotations/landmark-rxr.json
  # hfov_deg: 90
```

`habitat_config` and `scenes_dir` are required. `dataset_data_path` is required
for benchmark evaluation, together with `gt_path` for official nDTW trajectories;
both can be omitted for free exploration. Without
episode data, supply scene GLBs through the Studio `scenes` catalog or `--scene`;
a scene directory alone does not populate the scene selector. Scene geometry
still needs a usable navigation mesh and simulator configuration.

Paths are relative to the recipe file unless absolute; `{split}` is expanded.
`gt_path` maps episode IDs to records containing `locations: [[x,y,z], ...]`,
as in VLN-CE's official `*_gt.json.gz` files. JSON and gzipped JSON are accepted.
For R2R, use the GT file distributed with `R2R_VLNCE_v1-3_preprocessed`.
For RxR guide episodes, use `{split}_guide_gt.json.gz`; alternatively a
`{split}_{role}_gt.json.gz` template loads both `guide` and `follower` files.
Every selected episode must have a valid GT trajectory. Missing, duplicate or
ambiguous IDs are errors; episode `reference_path` is never a GT substitute.
`ndtw_fdtw: true` uses FastDTW with radius 1, matching the official default;
`false` selects exact DTW. The algorithm is recorded in the result protocol.

The `online:` wrapper is optional. The authoritative schema and defaults are in
[`online/config.py`](../src/cofl/online/config.py). The old SecVLA `export:`
configuration is not accepted directly: move the needed runtime/data/annotation
paths into this centralized schema. FGR2R and Landmark-RxR files are optional
sources for oracle sub-instructions; omit an annotation path you do not use.
`oracle_source` defaults to `auto`, which tries the available annotation sources.
In oracle mode, set `oracle_source: landmark_rxr` to require Landmark-RxR
exclusively. The canonical RxR recipe pairs it with `instruction_mode: oracle`,
`oracle_unavailable: error` and a `landmark_rxr_json` path. This source policy
never falls back to native timed or full instructions.
The source policy is recorded in the run manifest and episode protocol.

The episode source is JSON or gzipped JSON with an `episodes` list. Episodes
need IDs, scene IDs, instructions, finite XYZ start/goal/reference positions and
a unit XYZW start rotation. Oracle annotations select the policy's language;
reference geometry also supplies benchmark metrics. `language_filter` defaults
to `en` and filters language-tagged episodes.

Use your own working Habitat interpreter in `habitat_python`. Model inference
and the app run in the main CoFL Python 3.12 environment; Habitat-Sim runs in a
separate worker process and can use an older compatible Python environment.
The worker needs `habitat_sim`, `numpy`, `quaternion`, and `yaml`; add directories
to `runtime_pythonpath` only when that interpreter needs additional import
paths. The CoFL source directory is added automatically. No SecVLA import is
required. If `habitat_python` is omitted, the current interpreter is used.

The worker reads the Habitat task's `SIMULATOR` settings (or follows
`BASE_TASK_CONFIG_PATH`) and creates RGB/depth sensors. It requires the episode's
scene asset and a loaded navigation mesh. Explicit recipe `hfov_deg` overrides
task sensor HFOV; otherwise task HFOV is used, falling back to checkpoint sector
geometry when absent. RGB and depth HFOV must agree; actual sensor geometry is
recorded in the run. Availability checks report missing files or
modules before launch; a successful check does not prove every scene or episode
will initialize. Runtime errors remain visible in the session.

## RxR guide English setup

The canonical RxR online evaluation uses the
[RxR-1600 dataset](../benchmarks/rxr_1600/README.md): a fixed custom subset of
1,600 `val_unseen` guide-English episodes selected for this task. It is not the
official full split. Scores for this subset and their evaluation conditions
are recorded in the separate [evaluation reference](../reproduce/rxr_1600.md).
Prepare its five annotation outputs under `prepared/rxr_1600`, then configure
your scene directory and separate Habitat interpreter in
[`configs/evaluation/rxr_1600.yaml`](../configs/evaluation/rxr_1600.yaml).
That file contains both the `online` runtime recipe and the `SIMULATOR` sensors.
Model, execution and output settings live in
[`cofl_s_rxr_online.yaml`](../configs/evaluation/cofl_s_rxr_online.yaml):

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

The runtime loads `selected_episodes.json.gz`, `selected_gt.json.gz` and
`selected_landmark_rxr.json`. It requires `oracle_source: landmark_rxr` with
`oracle_unavailable: error`; missing or unusable Landmark annotations fail
instead of falling back to timed or full instructions. Prepared episodes omit
only the unused `instruction.timed_instruction` field; original episode IDs
are preserved and CLI indices refer to the prepared `0..1599` sequence.

RxR uses 640 × 480 RGB/depth sensors, a 79° horizontal field of view and a
0.88 m sensor height. The model's decoder HFOV conditioning does not change
camera geometry. Supply the checkpoint and Matterport3D `.glb`/`.navmesh`
assets separately; see the [configuration guide](../configs/evaluation/README.md#rxr-online)
for paths, overrides and dependencies. The default result directory is
`artifacts/online/cofl-s-rxr-1600`; the evaluation YAML sets 16 environment
workers and a maximum inference batch size of 16.

### Legacy full-English Studio recipe

The separate local recipe, `local/rxr-online.yaml`, selects **guide, English,
val_unseen**: 3,669 episodes across 11 scenes from the 11,006-episode guide file.
`language_filter: en` includes both `en-IN` and `en-US`. This selection matches
the SecVLA RxR guide-English recipe; it does not evaluate the other languages or
the follower role. Episode indices refer to the filtered English sequence.
This recipe remains available for Studio and is not the canonical RxR-1600
evaluation configuration above.

For another installation, save a recipe with these paths adjusted to your assets:

```yaml
online:
  habitat_config: /data/vlnce/habitat_extensions/config/rxr_vlnce_english_task.yaml
  dataset_data_path: /data/RxR_VLNCE_v0/{split}/{split}_guide.json.gz
  gt_path: /data/RxR_VLNCE_v0/{split}/{split}_guide_gt.json.gz
  scenes_dir: /data/scene_datasets
  habitat_python: /opt/habitat-env/bin/python
  split: val_unseen
  language_filter: en
  instruction_mode: oracle
  oracle_unavailable: full_instruction
  landmark_rxr_json: /data/Landmark-RxR/LandmarkRxR_{split}.json
  progress_lookahead: 6
  hfov_deg: 79.0
  ndtw_fdtw: true
```

Use the concrete **guide GT** file for this guide-only evaluation. The RxR
Habitat task supplies 640 by 480 RGB/depth sensors, a 79° horizontal field of
view and a 0.88 m sensor height. Keep those task settings when moving the recipe;
the inference rule that clamps decoder HFOV conditioning does not change camera
geometry. Landmark-RxR annotations are matched by instruction ID. Oracle mode
uses their sub-instructions when usable and falls back to the episode's native
timed instructions when the external annotation is missing or degenerate.

This RxR recipe explicitly sets `oracle_unavailable: full_instruction`, matching
SecVLA's `get_instruction` fallback when neither source supplies usable
sub-instructions. The local preflight found 3,351 episodes using Landmark-RxR
edge annotations, 193 using native node annotations and 125 requiring the full
episode instruction. All 3,669 episodes remain in the evaluation denominator;
the 125 are full-instruction runs, not oracle sub-instruction runs. Each such
episode records `instruction_fallback` with its provenance. Other recipes retain
the default `oracle_unavailable: error` unless they explicitly select this option.

The local Studio catalog already registers R2R and RxR together. Launch it with:

```bash
uv run --locked cofl app --config local/studio-preview.yaml
```

Choose **2D egocentric → VLN Benchmark → RxR · guide English · closed-loop VLN**,
then select the policy and an episode. An already running Studio can connect
`local/rxr-online.yaml` through **Open benchmark**. For your own startup catalog,
add a sibling entry under `benchmarks`; the path resolves relative to that catalog:

```yaml
benchmarks:
  rxr-online:
    label: RxR · guide English · closed-loop VLN
    path: rxr-online.yaml
    domain_id: ego2d
```

R2R and RxR share `cofl-inference-v1`, benchmark protocol v6 and the same
controller, clocks and limits. The local recipes use a 5 Hz planner, 50 Hz
controller, 30 Hz sensors, 100 Hz plant, 120 s limit and 500 planner-step limit.
App runs retain the 30 Hz scheduled sensor renders; browser state still polls
every 350 ms, so this setting does not imply a 30 FPS browser view.

## Headless benchmark runs and resume

For a packaged headless installation, use `pip install 'cofl-navigation[online]'`
(or `pip install '.[online]'` from this checkout). This extra includes native
policy inference, YAML recipes and result locking without the Studio web stack.
The documented source `uv sync --locked` development environment already
includes these dependencies through the app extra; no additional sync is needed.

Use `--config` to keep the evaluation settings in one YAML. For R2R, first
prepare the [R2R-1200 data](../benchmarks/r2r_1200/README.md) under
`prepared/r2r_1200`, then set your scene and Habitat interpreter paths in
[`configs/evaluation/r2r_1200.yaml`](../configs/evaluation/r2r_1200.yaml).
The canonical evaluation configuration is
[`configs/evaluation/cofl_s_r2r_online.yaml`](../configs/evaluation/cofl_s_r2r_online.yaml):

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_r2r_online.yaml
```

This configuration defines the checkpoint, execution parameters and result
output. Its `benchmark: r2r_1200.yaml` recipe supplies data paths, scene and
interpreter paths, oracle options and simulator sensors. It runs the fixed
1,200-episode cohort and writes to `artifacts/online/cofl-s-r2r-1200`. Data
preparation produces annotation files only; evaluation owns the runtime setup.

For RxR, first prepare [RxR-1600](../benchmarks/rxr_1600/README.md) under
`prepared/rxr_1600` and configure its `configs/evaluation/rxr_1600.yaml` runtime
recipe. Its model evaluation command is:

```bash
uv run --locked python -m cofl.online --config configs/evaluation/cofl_s_rxr_online.yaml
```

It evaluates 1,600 guide-English episodes with a strict Landmark oracle and
writes to `artifacts/online/cofl-s-rxr-1600`. It does not use the R2R-1200
preparation or subset. See [RxR setup](#rxr-guide-english-setup) for its assets,
native camera and the separate legacy Studio recipe, and the
[configuration index](../configs/evaluation/README.md) for online versus offline
recipes.

For a separate experiment, create an evaluation YAML under
`configs/evaluation/`, for example:

```yaml
benchmark: r2r_1200.yaml
checkpoint: ../../checkpoints/cofl-s/best.ckpt
output: ../../artifacts/online/run-001
count: 10
workers: 1
inference_batch_size: 1
```

Evaluation YAML paths resolve relative to that file; CLI paths resolve from the
working directory. The referenced benchmark recipe keeps its own relative-path
base and must include `gt_path`. If data preparation used a custom directory,
update its episode, GT and annotation paths in `r2r_1200.yaml` or `rxr_1600.yaml`.
Prepared data contain no
evaluation configuration. `parameters` accepts an inline `RunParameters`
mapping or a path to a JSON mapping; omitted fields use the shared defaults.
`episode: [0, 5]` selects explicit dataset indices instead of start/stride/count.
Unknown configuration keys and invalid values are rejected before policy loading.

Explicit CLI options override YAML settings regardless of argument order. For
example, append `--count 10` for a short run, then repeat without that override
and append `--resume` to extend to the configured cohort. `--no-resume` overrides a
configured `resume: true`. A CLI `--episode` list or `--parameters` file replaces
the corresponding YAML value entirely. Resume still requires matching evaluation
inputs and protocol; use a new output directory when changing those settings.

Batch execution groups the **already selected, uncompleted** episodes by scene.
Selection uses indices in the configured input dataset: `--count 10` chooses the
same ten episodes before grouping, and stored episode identities do not change.
For prepared R2R-1200 these are indices `0..1199`; for RxR-1600 they are
`0..1599`. Original episode IDs are preserved, with source-index correspondence
in `episode_index_map.json`. The
first encountered scene runs first when `workers: 1`. Compatible episodes reuse
the Habitat worker, renderer and navmesh; each start resets the pose, sensor transforms, simulation
clock, commands and reference trajectory. A scene/configuration change or failed
worker forces a fresh initialization. Use `--no-group-by-scene` or
`--no-reuse-scene` to disable these execution optimizations for comparison.

For concurrent environment evaluation with one shared policy, increase `workers` in
YAML or use `--workers`:

```bash
uv run --locked python -m cofl.online \
  --config configs/evaluation/cofl_s_r2r_online.yaml \
  --workers 2 --inference-batch-size 2 --worker-threads 1 --batch-wait-ms 2 \
  --output artifacts/online/cofl-s-r2r-1200-workers2
```

The R2R evaluation configuration sets workers and batch size to 1, so the example
overrides both for a batching comparison. `workers` defaults to 1 in the CLI.
With more than one worker, `group_by_scene` must be
true. The parent assigns one unstarted episode at a time. It first chooses a
scene that no other worker is currently processing, preferring that worker's
previous scene when available and otherwise the scene with the most pending
episodes. If all pending scenes already have workers, it can assign another
episode from the same scene: it reuses the worker's previous scene when possible,
otherwise choosing the largest pending-episode count per assigned worker
(including the new worker). Ties follow the selected cohort's first scene order.
This starts workers on distinct scenes when possible and lets idle workers help
with a long scene after finishing a shorter one. Episodes remain indivisible;
each runs the same closed-loop runner, and no episode is assigned twice.
Up to `min(workers, remaining episodes)` processes start, even if only one scene
remains. Each worker owns an independent Habitat simulator and reuses it between
compatible episodes; two workers loading the same `.glb` have independent state.
One policy is loaded in the parent process; its inference thread combines ready
observations from different workers into a model batch and returns each
prediction to its requesting environment. The parent also owns the output lock
and commits completed results through one result store. Completion order can
differ from dataset order; episode identities and the selected cohort stay the same.

`--inference-batch-size` (YAML `inference_batch_size`) sets the maximum inference
batch size. Its default `null` uses the effective environment-worker count, and
an explicit size is capped at that count. `--batch-wait-ms` (YAML `batch_wait_ms`,
default 2.0) sets the extra batching window for a partially filled batch; it must
be finite and nonnegative. A full batch dispatches immediately. The inference
thread dispatches a partial batch when that window expires, so a slow or finished
environment cannot hold up all other workers. Requests arriving during inference
queue for a subsequent batch. Waiting behind an active batch adds separate
latency. Execution diagnostics report observed batch sizes, request counts and
batch-wait measurements.

The shared policy uses the configured `--device`; every Habitat simulator uses
the benchmark recipe's `gpu_id`. GPU placement is not assigned automatically per
worker. Model weights are shared through the parent, while additional simulators
and larger inference batches need extra GPU and host memory. `--worker-threads`
(YAML `worker_threads`, default 1) bounds CPU numerical-library threads per
spawned environment worker. Compare 1, 2 and then 4 workers on the same cohort
while monitoring memory, throughput and actual batch sizes. Shared GPU work,
episode-length imbalance, batching delay, startup and result writing limit scaling; no
measured speedup is implied by these settings.

Batch shape can change floating-point accumulation relative to single-observation
inference. Compare fixed-frame trajectories and actions within stated numerical
tolerances, then compare complete closed-loop outcomes on the same episodes.
Changing worker or batch counts does not guarantee bitwise-identical trajectories.

Worker count, thread limits and batch settings are execution settings. They can change when
resuming an existing compatible v6 output with `--resume`, after its previous
process has released the output lock. A running process keeps its original
settings and scheduler until restarted. The scheduler is recorded in each
episode's execution metadata, outside the fixed navigation protocol. Keep the checkpoint, input data, inference rules and controller
parameters unchanged for a throughput comparison. Studio retains its existing
single-session behavior; these worker options apply to the headless CLI.

Completed episodes are committed as they arrive. Ordinary episode errors retain
the existing failed-result semantics. Worker initialization failures, memory
exhaustion and process crashes interrupt the run instead; uncommitted episodes
remain eligible for `--resume`. Reduce `--workers` or `--inference-batch-size`
after a capacity failure.
Ctrl-C and SIGTERM also clean up the CLI's scene workers and their Habitat processes.

`executions.jsonl` records shared-model batch counts, the actual batch-size
histogram and queue timings for each completed invocation. Step logs include
`inference_batch`: its `roundtrip_ms` includes queueing, transport, neural
inference and that worker's CPU integration. The existing `inference_ms` excludes
queueing and transport and includes the full batch's neural service time plus
local integration; dividing it by batch size would not describe request latency.
Parallel per-request times overlap and must not be added to estimate wall time.

Batch shape changes floating-point accumulation. In the real shared-model smoke,
batch size 1 reproduced the scalar fixed-frame predictions exactly, while larger
batches introduced small field differences that subsequently changed one closed-loop
episode's stopping step. Therefore the scalar/per-worker-model equivalence result
does not apply to dynamic batching. Use a separate output for a formal batched
evaluation and compare navigation outcomes as well as throughput; compatible
resume preserves records but does not establish numerical equivalence between
different execution modes.

In headless benchmark evaluation, the worker renders only the final scheduled
sensor sample that each planner interval consumes. It previews the simulation
clock and renders at that sample's original pose; sensor rates and observation timing stay
the same, including when sensor and planner frequencies do not divide evenly.
The default 30 Hz sensors / 5 Hz planner therefore need about one RGB-D render
per planner interval instead of six.

App sessions, including visualized benchmarks, retain every scheduled sensor
render (`render_all_sensor_frames=True`); interactive workers also always retain
them. With the default sensor settings this is 30 renders per simulated second.
This is separate from browser frame delivery: the current App polls session
state every 350 ms, and observations are published at planner boundaries. A
steady 30 FPS browser view additionally requires independent frame delivery and
simulation/render scheduling; retaining sensor renders alone does not provide it.

Every observation is encoded once, then the decoder materializes the complete
**100 by 100 polar velocity grid** in FP32. Rows span normalized radius `[0, 1]`;
columns span normalized angle `[-1, 1]`, including both endpoints. Grid values
remain body-frame forward/left vectors. Integration samples the raster with
bilinear interpolation, border padding and `align_corners=True`; it makes no
per-step model queries or GPU transfers. This is independent of the Cartesian
BEV grid used for dataset supervision and field scoring.

`parameters.field_grid_size` defaults to 100 and is independent of
`parameters.policy_steps`. `parameters.field_query_chunk_size` defaults to 4096,
so the 10,000 points are queried in bounded batches and transferred together.
The CPU uses the shared time-scaled sector Euler integration in the field dtype and
projects each update into the real camera sector for all configured steps.
Ground origin offset and decoder-only HFOV conditioning follow the four
`cofl-inference-v1` rules above; action-head STOP remains a controller decision.
`inference_backend: auto` and `eager`
both use batched dense-grid inference; the legacy `cuda_graph` selection falls
back to that same grid path, never to per-step model queries. The effective grid
settings and backend are recorded in each result.

Protocol v6 records all four shared inference rules. Earlier v4 point-query
and v5 raster-only results cannot be resumed into this run. The example
R2R evaluation YAML uses `artifacts/online/cofl-s-r2r-1200` to keep new results
separate from the historical inference-v1 run.

The CLI uses the same benchmark runner as Studio and evaluates all selected
episodes. Use repeated `--episode INDEX` for individual dataset indices, or
`--start`, `--stride` and `--count` for a batch. For example:

```bash
uv run --locked python -m cofl.online \
  --benchmark benchmarks/ground-online.yaml \
  --checkpoint checkpoints/cofl-s.ckpt \
  --start 0 --count 20 \
  --device cuda:0 \
  --output artifacts/online/run-001
```

Omit `--count` to evaluate the remaining episodes. The default instruction mode
comes from the recipe (`oracle` by default);
`--instruction-mode full_instruction` explicitly evaluates full episode
instructions instead. These
are different evaluation conditions and are recorded in the result protocol.
`--parameters run-parameters.json` accepts fields from
[`RunParameters`](../src/cofl/online/dynamics.py).

Model replanning defaults to 5 Hz (one prediction every 0.2 simulated seconds)
in both the CLI and Studio. To select another rate, put, for example,
`{"planner_hz": 10.0}` in `run-parameters.json` and pass it with `--parameters`.
The controller, sensors and plant keep their independent rates. `policy_dt`
controls field integration and does not need to change with `planner_hz`.

A new output directory must be empty. Repeat the command with `--resume` to skip
committed episodes and continue; the checkpoint, recipe, dataset, official GT
files, external instruction sources, parameters and protocol must match the saved manifest.
The Habitat task configuration is also hashed. Changing the episode selection
is allowed, so a run can grow from a small subset to a larger evaluation. Checkpoint identity includes resolved path, size and
modification time; recipes, data and annotation sources additionally use SHA256.
Protocol v6 requires a new output directory. Earlier v5 runs used raster
integration without all four shared rules; v4 used point queries, and v2/v3
also used different metric recipes. They cannot be resumed or merged into the
same directory.

[`ResultStore`](../src/cofl/online/store.py) streams planner steps to
`episodes/<identity>/steps.jsonl`. It atomically commits that directory together
with `episode.json`. The identity includes dataset identity, episode ID and
dataset index, so IDs that repeat across indices or datasets do not collide.
An interrupted, uncommitted episode can be retried. Execution exceptions are
recorded as `failed`; ordinary navigation failure is a completed episode with
unsuccessful metrics. Resume skips both kinds of committed record.

`summary.json` and `summary.csv` are rebuilt from **all** committed episodes,
including earlier invocations and failures, regardless of terminal output.
The JSON summary retains metadata and metrics, with an `episode_file` relative
link to each full result. Dense executed/reference trajectories remain in that
episode's `episode.json` instead of being duplicated into every summary update.
Resuming older runs rebuilds the compact summary without modifying their full
episode files or manifest.
`success_rate` includes execution failures in its denominator; `metric_means`
averages each metric over the available values, and `metric_counts` makes those
sample counts explicit. Undefined/non-finite metrics are serialized as `null`,
listed in `metrics.undefined_metrics`, and excluded from their valid-value
counts. Do not treat a metric mean with missing failed-episode values as an
all-attempt success rate. Resume also repairs a summary update
interrupted after an episode commit. Only one writer can own an output directory.

The manifest identifies inputs for compatibility; it is not a hash archive of
every scene mesh or the entire software environment.

Studio keeps bounded telemetry and can export its current run snapshot. It does
not preserve a complete frame archive; use the CLI for the full step log.

## Shared protocol and comparison scope

The authoritative controller, clock, STOP arbitration and metric settings live
in [`online/dynamics.py`](../src/cofl/online/dynamics.py) and
[`online/runner.py`](../src/cofl/online/runner.py). `runner.protocol(...)` records
the effective values in every result; Studio receives default parameters through
its mode descriptor rather than maintaining a separate parameter table.

| Setting | Default / interpretation |
| --- | --- |
| Benchmark protocol | `cofl-s-vln-benchmark-v6`; shared rules `cofl-inference-v1` |
| Metric recipe | `vlnce_official_core_v1` |
| Interactive protocol | `cofl-s-interactive-v3`; live instruction commands and no benchmark metrics |
| Clock | Plant 100 Hz, controller 50 Hz, sensor 30 Hz, planner 5 Hz; first events at simulated time zero |
| Limits | 120 simulated seconds or 500 planner predictions; per command in interactive mode |
| Velocity field | 100×100 polar grid; batched decoder queries followed by CPU bilinear interpolation |
| Field integration | 100 time-scaled steps, bias 0.2/power 10; boundary projection continues all steps; `policy_dt` defaults to `1 / policy_steps` |
| Ground start / model HFOV | Half-cell origin offset; decoder conditioning 79–90°, with actual camera geometry retained |
| Controller | Pure pursuit, 0.5 m lookahead, at most 0.5 m/s and π/3 rad/s; rotate in place above 30° |
| STOP | Action-head threshold 0; endpoint threshold 0.2 m; pool increment 0.5 and half-increment decay; drain disabled |
| Oscillation | Window 4; forward probability upper bound 1.0 for alternating-turn rescue |
| Core metric sampling | Complete XYZ plant path including the terminal position; nDTW removes consecutive duplicate positions |
| Extra metric sampling | Planner-tick XZ poses and controller interval samples; cSPL uses the complete executed XZ path |
| Success | Explicit policy STOP and final navmesh distance strictly < 3 m; timeout/max_steps are unsuccessful |

Benchmark instruction progress is privileged: `oracle` uses ground-truth route
annotations and the robot pose, with a monotonic reference edge/node selector.
The default progress lookahead is 6. With `oracle_source: auto`, matching
Landmark-RxR annotations take priority, followed by native timed sentence
annotations, then FGR2R chunk/edge
annotations. Provenance and the active segment are recorded. Missing or
ambiguous matches do not silently become full instructions. Choose
`instruction_mode: full_instruction` to use full text throughout, or explicitly
set `oracle_unavailable: full_instruction` to retain episodes without usable
sub-instructions, as in the legacy full-English RxR Studio recipe. The canonical
RxR-1600 configuration sets `oracle_source: landmark_rxr` and requires a usable
Landmark annotation, with no fallback. Full-instruction fallback episodes record
`instruction_fallback` and `uses_gt_progress: false`; batch summaries include
`instruction_source_counts` for completed episodes.

NE, OS, SR and SPL require a navmesh geodesic callback. They do not fall back to
straight-line distance or reference-route length. An unreachable goal has
infinite geodesic distance and cannot pass the success test; exported non-finite
NE values use the `null` and undefined-metric convention above. NE and SR use the
actual final dense plant pose, including the last executed segment before a
timeout or step limit. OS scans the complete dense XYZ path, deduplicating exact
repeated poses and stopping once success is found; it does not subsample the
path.

The core definitions follow [Habitat-Lab 0.1.7 Success/SPL/DistanceToGoal](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py)
and [VLN-CE NDTW/SDTW/OracleSuccess](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/measures.py):

| Metric | Definition |
| --- | --- |
| NE | Final navmesh geodesic distance to the goal, in metres |
| OS | Any executed position has navmesh goal distance strictly < 3 m; no STOP required |
| SR | Explicit policy STOP with final navmesh goal distance strictly < 3 m |
| SPL | `SR * shortest / max(shortest, executed_xyz_length)`; shortest is the start-to-goal navmesh distance |
| `path_length_m` | Sum of Euclidean XYZ displacements over the complete executed path |
| nDTW | `exp(-DTW(executed_xyz, official_gt_locations) / (3 * len(official_gt_locations)))` |
| SDTW | `SR * nDTW` |

The old uppercase `NDTW` field is removed. The old lowercase `nDTW` field
contained a success-weighted custom score; its new meaning is standard nDTW.
Consecutive repeated executed positions are discarded only for DTW, as in the
official implementation; GT locations retain their original sampling. Core
metrics consume simulator plant positions, independently of planner sampling.
This continuous environment has 100 Hz plant steps: it does not discretize
motion to the official agent's 0.25 m actions. Control, instruction assistance,
step budgets and trajectory sampling must still be reported when comparing
experiments; aligned metric definitions alone do not establish benchmark parity.

Additional metrics remain available: **HS (Heading Smoothness)** replaces SM,
and **BSR (Blocked-Step Rate)** replaces StR. HS keeps the existing XZ
heading-change formula. BSR is the fraction of planner intervals containing
at least one blocked plant step, not the fraction of all plant steps blocked.
VC, EF, optional HA and custom CLS keep their previous definitions. CLS retains
its legacy hard-threshold XZ coverage and exponential length penalty; it is
not the official CLS formula. The custom `cSPL` retains XZ reference-route
length in its numerator and full executed XZ length in its denominator,
while sharing the new strict STOP/geodesic SR rule. See the
[`core`](../src/cofl/online/metrics.py) and
[`additional`](../src/cofl/online/continuous_metrics.py) implementations.

At the default 5 Hz, the 500-prediction limit corresponds to approximately
100 simulated seconds. STOP accumulation and oscillation windows count planner
predictions, so changing the planner rate also changes their duration in seconds.

Control and STOP defaults come from the actual SecVLA benchmark CLI and runner,
whose README and class defaults differed in several places. CoFL uses a 5 Hz
planner by default, centralizes these settings and fixes the old batch
resume/summary behavior. Protocol v4 replaces the former inclusive threshold,
timeout success, XZ SPL and custom DTW with the core definitions above.
Those metric definitions remain in v6, which additionally fixes the common
inference rules. Existing v2–v5 scores cannot be merged into a v6 run; GT sources,
metric recipe and inference rules are part of resume compatibility checks.
CoFL-S still encodes a **single RGB-D frame** at each replan; the former SecVLA model could use
RGB memory. Sharing evaluation mechanics does not establish numerical
reproduction of SecVLA model results. Tiny integration checkpoints and simulator
smoke tests do not establish trained navigation quality or a completed benchmark.

Both egocentric tasks share the domain API prefix `/api/v1/domains/ego2d`.
The former `/api/v1/modes/ego2d-playground` prefix remains a compatibility alias
and is omitted from the generated OpenAPI schema. New clients should use the
domain prefix:


| Method and suffix | Purpose |
| --- | --- |
| `GET /benchmarks` | Recipe/runtime availability, split and episode counts |
| `GET /benchmarks/{benchmark_id}/episodes` | Episode page with `offset` and `limit` |
| `GET /benchmarks/{benchmark_id}/scenes` | Available scene IDs and suggested spawn poses |
| `POST /sessions` | Start `mode: benchmark` with an episode index, or `mode: interactive` with a scene/optional pose/instruction |
| `GET /sessions/active` | Recover the active session across both tasks, or `null` |
| `GET /sessions/{session_id}?after_step=N` | Current observation, instruction state, metrics and recent steps after `N` |
| `POST /sessions/{session_id}/commands` | Submit an interactive instruction |
| `POST /sessions/{session_id}/pause` / `resume` | Freeze or continue interactive motion |
| `POST /sessions/{session_id}/reset` | Request an asynchronous reset with optional pose and instruction |
| `POST /sessions/{session_id}/stop` | End either kind of session |

See [`app/online.py`](../src/cofl/app/online.py) for schemas, HTTP bounds and mode
IDs. Reset acknowledgement changes `epoch`; `command_state.reset_error` reports
failure. `first_available_step` marks the retained telemetry window, and clients
must not assume a response contains the complete episode. The shared static
`/api/v1/predict` endpoint rejects session modes.

## Architecture and extension points

The interface uses React, Radix Themes, TanStack Query, React Three Fiber, Drei,
and Three.js. FastAPI/Pydantic define the versioned API; `openapi-typescript` and
`openapi-fetch` provide generated frontend types and a typed client. Model
loading, preprocessing, coordinate conversions, field metrics, and rollout
integration reuse CoFL's existing public implementations.

The main boundaries are:

- `app/contracts.py`: domain/task descriptors and the static 2D transport schemas.
- `app/adapters.py`: injectable inference and dataset service protocols.
- `app/server.py`: resource resolution, thread-pool dispatch, and mode routing.
- `app/modes.py`: built-in domains, discoverable tasks, and optional dedicated
  API routers.
- `app/resources.py`: local filesystem discovery, stable resource IDs, and
  same-origin checks for resource browsing and registration.
- `app/datasets.py`: read-only native dataset/collection adapter with bounded
  reader caches.
- `app/inference.py`: lazy model loading and serialized encode/query operations.
- `app/online.py`: typed HTTP adapter for the egocentric session API.
- `online/`: shared CLI/session runner, policy adapter, clock/controller,
  metrics, and isolated Habitat worker.
- `app/src/`: frontend views, renderer registration, and generated API client.

`GET /api/v1/resources/browse` and `POST /api/v1/resources` share the typed
`model`/`dataset`/`benchmark` resource contract and accept `domain_id`.
`DomainDescriptor.resource_kinds` controls the available resource tabs. File
browsing filters extensions and folder shapes; it does not open every checkpoint
to infer a method. Model/dataset registration validates through the existing
policy loader or native reader before publishing verified metadata to the
session catalog. Benchmark registration validates the recipe; runtime and asset
availability are reported separately. Configured resources use the same
registration path when activated.

Paths are resolved before deduplication, including symbolic links. Reopening a
dataset can update its split. Failed registration leaves existing catalog
entries intact. To keep one loaded policy in memory, switching checkpoints
releases the previous model first; after a failed load, the previous registered
policy can reload on the next prediction.

Local resource registration is an optional adapter capability, defined by
separate protocols in `app/adapters.py`. A remote/custom inference or dataset
adapter can implement registration itself; adapters without it remain usable
for prediction and return a structured 501 for local registration requests.

The static inference service caches one loaded policy and one observation's
encoded context and **100 by 100 integration field**. Its cache key hashes
actual pixels, instruction, geometry, depth and depth validity. Moving the start
or changing the integration step limit reuses both the encoding and field.
Display `grid_size` controls sampled arrows independently; supervised preview
metrics still use exact decoder queries at their selected cells. Integration
uses CPU bilinear interpolation of the cached field, without further decoder
queries. An observation change clears the field cache; a model switch releases
the previous model, context and field references. A
reentrant lock serializes requests sharing the model; dataset readers have
their own lock and bounded caches. Blocking work runs outside the ASGI event
loop. The API discards disconnected requests before queued inference starts;
the browser also discards results invalidated by a new camera, scene, floor,
instruction, or checkpoint. Starting an online session releases the static
model cache; an execution gate prevents concurrent static inference/model
loading while the session owns the policy. Cancellation retains session
ownership until the policy and simulator close. Shutdown clears the services.

A new 3D domain can own its coordinate and session contracts without changing
the static 2D prediction endpoint:

```python
from fastapi import APIRouter
from pydantic import BaseModel

from cofl.app import create_app
from cofl.app.contracts import DomainDescriptor, Finite, ModeDescriptor
from cofl.app.modes import StudioMode

router = APIRouter()


class Session3D(BaseModel):
    id: str
    position_m: tuple[Finite, Finite, Finite]


@router.get("/sessions", response_model=list[Session3D])
def list_sessions():
    # Replace with this domain's own session adapter.
    return []


app = create_app(
    domains=[DomainDescriptor(
        id="spatial3d",
        title="3D navigation",
        description="World-space XYZ navigation",
        methods=["example-3d"],
        profiles=["cartesian_xyz_v1"],
        resource_kinds=["model"],
    )],
    modes=[StudioMode(
        descriptor=ModeDescriptor(
            id="spatial3d-playground",
            domain_id="spatial3d",
            task="playground",
            execution="session",
            title="Playground",
            description="Sessions with world-space poses and 3D trajectories",
            renderer="spatial3d",
            capabilities=["sessions", "world_xyz"],
        ),
        router=router,
    )],
)
```

This adds a `spatial3d` domain and
`/api/v1/modes/spatial3d-playground/sessions`, while retaining the built-in
domains and tasks. Both domain IDs and mode IDs must be unique. Add a matching
`spatial3d` renderer in
[`app/src/modes/registry.ts`](../app/src/modes/registry.ts), plus the domain's
policy/resource and simulator adapters. The example registers an empty session
listing; it does not implement a 3D policy or simulator.

`ModeDescriptor` declares `domain_id`, `task`, and `execution` independently of
the renderer. Session modes leave `StudioMode.static_input` unset and use their
own router. Static image/dataset modes explicitly set that adapter field.
`Point2`, `PredictionRequest`, `PredictionResult`, and `DatasetView` are contracts
for the existing **2D** adapters; do not pad XYZ points into them or generalize
the shared `/predict` endpoint to carry unrelated session protocols. New 3D
schemas can be exposed through the same OpenAPI generation flow.

## Frontend development and validation

Run the backend and Vite in separate terminals from the repository:

```bash
uv run --locked cofl app --api-only --no-browser
```

```bash
cd app
npm ci
npm run dev
```

Vite proxies `/api` to the default backend port, 8787. After changing the Python
contracts, regenerate the checked-in schema/types:

```bash
uv run --locked cofl app --export-openapi app/openapi.json
cd app
npm run generate:api
```

The full internal Python and Playwright suites are not included in the public
release. Internal inference tests create tiny real SigLIP assets and portable
checkpoints locally; they do not download weights. They exercise both field profiles through the
public model interfaces and HTTP contracts, including masking, metric units,
prompt changes, invalid outputs, content-based cache invalidation, concurrent
starts, and lifecycle cleanup. The public frontend includes the standalone
unit tests in `app/src/` and provides `npm test` and `npm run build`.

Run the frontend unit tests and build from `app/`:

```bash
npm ci
npm test
npm run build
```

The internal Playwright suite starts its own local server with tiny, explicitly
labelled test weights and temporary native datasets. It checks real scene inference, clean
camera capture, start-only encoding reuse, switching floors, stale responses,
BEV validation semantics, domain/resource filtering and selection memory,
resource registration, and a narrow viewport. Online browser tests use explicit
HTTP fixtures to check cross-view session restoration, initial observations,
commands, pause/resume, reset epochs, command errors, benchmark handoff and
completion, plus explicit CPU confirmation after GPU failures and device sharing
across workspaces, and explicit shutdown acknowledgement, retries and stopped
polling; they do not run Habitat. These weights and fixtures test the integration, not trained
navigation quality.
