# CoFL inference rules

`cofl-inference-v1` defines neural field inference for both CoFL profiles. App
prediction, standalone checkpoint rollout, offline image navigation and online
CoFL-S planning use the same grid and integration implementation. Rules and
defaults live in `src/cofl/runtime/inference_rules.py`; the implementation uses
`runtime/field_grid.py` and `runtime/rollout.py`. Offline image rollout retains
its public import from `cofl.evaluation.rollout`.

The shared rules incorporate the existing CoFL raster integration and the
SecVLA sector decoder's grid sampling, time schedule, camera conditioning and
boundary projection. They also adopt the SecVLA App/QA reference-start offset.
These are inference rules, not an optional fast path.

## Complete field before integration

Encode each observation and instruction once, prepare each decoder layer's
attention keys and values once, then predict every vector on a complete grid.
The standard grid is 100 by 100; decoder queries use chunks of 4096 by default.
The full grid is transferred to CPU once. Euler integration only samples that
grid and never queries the decoder.

| Profile | Grid columns | Grid rows | Vector components |
| --- | --- | --- | --- |
| CoFL `image_field_v1` | image right, `[0, 1]` | image down, `[0, 1]` | right/down, image fractions per policy time |
| CoFL-S `ground_sector_v1` | normalized angle, `[-1, 1]` | normalized radius, `[0, 1]` | body forward/left, normalized by the checkpoint's metric scale |

Both grid axes include their endpoints. Sampling uses bilinear interpolation,
border padding and `align_corners=True`. The ground lattice comes from the
existing `CoFLSFieldDecoder.build_sector_query_lattice` helper. Grid resolution
and Euler step count are independent parameters; a custom resolution changes
the numerical field approximation and is recorded in the inference metadata.

An unchanged App observation can reuse its encoded context and full field when
the trajectory start or step settings change. Observation, instruction, depth,
geometry or model changes invalidate that cache. A new online observation
produces a new context and field.

## Time schedule and boundary projection

For `T` Euler steps, precompute the inclusive schedule `t = linspace(0, 1, T)`.
The default is `T=100`, `policy_dt=0.01`, power `10`:

```text
denominator[k] = (1 - t[k]) + bias * t[k]**power
dt_eff[k] = policy_dt / denominator[k]
candidate = position + dt_eff[k] * velocity(position)
position = project_into_profile_domain(candidate)
```

| Profile | Default bias | Projection after each update |
| --- | --- | --- |
| CoFL image | `0.5` | Clamp right/down to `[0, 1]` |
| CoFL-S ground | `0.2` | Clamp metric forward to `[0, r_max]` and left to `[-r_max, r_max]`, then clamp normalized radius to `[0, 1]` and normalized angle to `[-1, 1]` |

Ground vectors are scaled to metres and integrated in Cartesian forward/left
coordinates. Convert the projected polar position back to Cartesian with the
actual camera geometry; polar query coordinates are never incremented by
Cartesian velocity components.

The integrator continues through all `T` steps after a boundary projection or
a zero vector. It returns the initial point and `T` updated points. A constant
zero field therefore produces a repeated start, rather than an early stop.
Nonfinite input or predicted values are errors. The batch implementation
preallocates trajectories and computes the schedule once per rollout batch.

Policy time is a curve-integration parameter, not simulator seconds. The time
schedule reparameterizes integration of the static predicted field; it does
not make the decoder time-conditioned or change the stored training targets.
Explicit trajectory configuration values are recorded, including overrides
to the step count, increment and image offline schedule.

## Ground camera conditioning and start

The model's HFOV conditioning is clamped to its trained band of 79–90 degrees
at neural inference call sites. Physical coordinate conversion continues to
use the actual camera HFOV. Neither sample geometry nor checkpoint buffers
are overwritten by this conditioning clamp. Training decoder queries retain
the raw sample HFOV.

For an `N`-row ground grid, the default normalized start is:

```text
theta_start = 0
radius_start = max(0.001, 0.5 / (N - 1))
```

The radial floor avoids the exact-origin row used by the legacy exported
field. An explicitly supplied valid ground start receives the half-cell floor
`0.5 / (N - 1)` while retaining its requested direction. At the standard
`N=100` and a 5 metre
radius this starts approximately 2.525 centimetres forward on the centerline.
CoFL image inference uses its requested start; offline navigation supplies
the reference trajectory's first point.

## Scope, actions and scoring

The mandatory ground behaviors are the scheduled integration, boundary
projection, model-only HFOV clamp and radial start floor. Together with full
grid inference they define the versioned rollout. SecVLA's optional
`auto_truncate` postprocessing for persistent low speed or low progress remains
disabled. Online action-head STOP and oscillation recovery belong to the
controller and are separate from trajectory integration.

Offline field metrics still query the selected native supervision cells
directly. Those queries use inference HFOV conditioning, but do not need to
construct a navigation grid or integrate a trajectory. Action-only scoring
only runs the action head. Grid generation for labels, dataset conversion,
and scoring already supplied ground-truth arrays are not neural inference
and do not silently apply model conditioning or rewrite stored labels.

App sensor rendering remains controlled independently from inference. The App
preserves the configured 30 Hz sensor rendering; headless evaluation may skip
intermediate frames that would be overwritten before the next planner
observation. Browser transport and polling impose their own display rate.

## Protocol identity

Offline evaluation records `offline_policy_v2`, online benchmark records
`cofl-s-vln-benchmark-v6`, and interactive online sessions record
`cofl-s-interactive-v3`. Their metadata includes `cofl-inference-v1` and the
resolved numerical rules. Offline `protocol.json` stores these rules directly;
`recipe.json` also records which selected tasks actually query fields, roll
out trajectories or score actions. Rule metadata participates in the protocol
fingerprint, including field-only CoFL-S runs whose HFOV conditioning changed.

Results made under earlier protocols require a separate output directory and
must not be combined with these results. Process restarts are required to
load changed inference code. See [benchmark configuration and metrics](benchmarks.md)
and [App and online configuration](app.md).
