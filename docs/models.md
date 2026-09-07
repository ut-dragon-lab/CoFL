# Models and coordinate contracts

CoFL and CoFL-S predict continuous fields by multiplying a normalized direction
by a nonnegative magnitude. They share an observation encoder and use distinct
query decoders.

| Contract | CoFL | CoFL-S |
| --- | --- | --- |
| Observation | BEV RGB and instruction | One egocentric RGB-D frame and instruction |
| Query `[B,N,2]` | Image `(x,y)` in `[0,1]^2` | Polar `(theta_norm,r_norm)` in `[-1,1] × [0,1]` |
| Output `[B,N,2]` | Image-coordinate vector | Canonical forward/left vector |
| Query embedding | Fourier features of `(x,y)` | Fourier features of `(theta_norm,r_norm,theta_norm * hfov_rad / pi)` |
| Field decoder | Pre-normalized attention and final feature normalization | Pre-normalized attention and final feature normalization |
| Additional prediction | — | STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT logits |

## Field decoders

The decoder-only interface requires PyTorch and does not load pretrained assets:

```python
import math
import torch
from cofl.models import CoFLFieldDecoder, CoFLSFieldDecoder

image_decoder = CoFLFieldDecoder(
    d_model=32, hidden_dim=128, num_heads=4, num_layers=2,
).eval()
sector_decoder = CoFLSFieldDecoder(
    d_model=32, hidden_dim=128, num_heads=4, num_layers=2,
    bev_x_max=5.0, hfov_rad=math.pi / 2, v_norm=5.0,
).eval()
context = torch.randn(2, 8, 32)
image_queries = torch.rand(2, 64, 2)
sector_queries = image_queries.clone()
sector_queries[..., 0] = 2 * sector_queries[..., 0] - 1

with torch.inference_mode():
    image_field = image_decoder(image_queries, context)
    ego_field = sector_decoder(
        sector_queries, context,
        hfov_rad=torch.tensor([math.radians(79), math.radians(90)]),
    )
```

These random tensors illustrate shapes. Navigation requires a trained encoder
and policy. Context can be encoded once and reused across independent query
chunks with the default decoder. CoFL's optional `need_self_attn=True` introduces
query interactions; `mag_enable=False` selects its direction-only ablation.

CoFL-S maps queries to physical points using:

```text
theta = theta_norm * hfov_rad / 2
range_m = r_norm * bev_x_max
forward = range_m * cos(theta)
left = range_m * sin(theta)
```

Vectors retain Cartesian forward/left components. `bev_x_max`, `hfov_rad` and
`v_norm` are persistent decoder buffers. Forward accepts scalar or per-sample
HFOV. Standard inference calls `clamp_hfov_for_model` to restrict conditioning
to `[79°,90°]`; training and direct decoder calls retain their supplied HFOV.
Physical conversion uses actual camera HFOV. The complete grid, time schedule,
projection and start rules are defined in [inference rules](inference.md).

Field units follow the annotation recipe. For distance-based CoFL-S labels,
magnitude multiplied by `v_norm` represents remaining distance in metres.
Trajectory integration and the controller determine physical motion; fields are
not direct actuator velocities. See [dataset-format.md](dataset-format.md) and
[inference.md](inference.md) for data and rollout contracts.

## Shared observation encoder

Both policies construct `ConditionEncoder`:

```text
RGB features → optional depth fusion → visual projection
pooled text + token features → text projection
pre-normalized visual/language fusion → final LayerNorm → context
```

Depth is its optional architectural branch. `ConditionEncoder` and `CoFLPolicy`
default to `use_depth=False`; `CoFLSPolicy` defaults to `True`. The RGB ablation
uses `encoder_kwargs={"use_depth": False}`. Pooled text, pre-normalized
`VisionLanguageFusion` and final `ctx_norm` are always present.

`EncoderOutput` is a named tuple:

| Field | Shape | Meaning |
| --- | --- | --- |
| `context` | `[B,P,d_model]` | Fused features consumed by prediction heads |
| `vision_tokens` | `[B,P,backbone_dim]` | Visual features after optional depth fusion, before projection |
| `language_tokens` | `[B,L+1,d_model]` | Projected text with pooled prefix and `L` input positions |

Use named fields or unpack all three values. `policy.encode(...)` returns only
context. The default 64 input positions yield 65 text features;
`encoder.total_text_length` includes the pooled prefix. Injected backbones must
return pooled-first text features shaped `[B,L+1,backbone_dim]`.

Each batch item contains one image and instruction. Tensor images have shape
`[B,3,H,W]`; nested image sequences are rejected. Datasets can retain complete
episode sequences while training selects each annotation's observation.

## Preprocessing and masks

The [locked uv environment](../README.md#installation) includes SigLIP's runtime.
Run Python examples with `uv run --locked python` from the repository root
(use the installation guide's CPU group flags for CPU use). The backbone's
`preprocess(images=..., texts=...)` returns pixel values, token IDs and the
tokenizer's attention mask. Raw images and flat batches are supported. Tensor
inputs to `policy.encode` must already be preprocessed; provide their tokenizer
mask and move all tensors to the policy device.

Tokenizer masks apply in visual-language fusion. SigLIP retains its pretrained
attention over the fixed padded sequence and final-position pooling. The encoder
prepends a valid pooled-token mask bit. Real EOS tokens remain valid even if
EOS and padding share an ID. Masks are never inferred from IDs; omitting a mask
for tensor inputs treats every text feature as valid. An explicit mask passed
directly to `backbone.encode_text` controls that call's internal attention.

`VisionLanguageFusion.text_attention_mask` is `[B,L+1]` with `True` meaning valid.
Shape mismatches and samples without valid text features are rejected.

CoFL-S depth is `[B,1,H,W]`, normalized as `depth_m / depth_zmax`. Compute its
validity mask before normalization; resize both to the processor's image
resolution. Depth and visual patch counts must match. Without an explicit mask,
finite normalized depth in `(0,1]` is valid. The [training configuration](training.md)
controls interpolation, depth limits and augmentation.

`local_files_only=True` requires local pretrained assets; `siglip_model_name`
can name a local directory. Importing `cofl.models` does not load assets or
access the network.

The backbone uses the `SiglipProcessor`, `SiglipVisionModel` and `SiglipTextModel`
interfaces verified with Transformers 4.57.1. It reads patch size from the vision
configuration. `unfreeze_last_n_layers=0` freezes both pretrained encoders, `-1`
unfreezes all their parameters, and a positive value unfreezes that many final
encoder blocks in each modality. Values exceeding either encoder's depth are
rejected.

## Policy interface

`CoFLPolicy` and `CoFLSPolicy` expose `encode`, `query` and `forward`. CoFL-S also
exposes `action_logits(context)`, ordered STOP, MOVE_FORWARD, TURN_LEFT,
TURN_RIGHT. `NavigationActionHead` consumes fused context;
`use_action_head=False` disables it.

```python
from cofl.models import CoFLSPolicy

policy = CoFLSPolicy(
    encoder_kwargs={
        "siglip_model_name": "google/siglip2-base-patch16-224",
        "local_files_only": False,
        "use_depth": True,
    },
    decoder_kwargs={"bev_x_max": 5.0, "hfov_rad": 1.5707963267948966, "v_norm": 5.0},
).eval()

# With preprocessed tensors and normalized depth on the policy device:
context = policy.encode(
    pixel_values, input_ids, attention_mask,
    depth=depth, depth_valid_mask=depth_valid_mask,
)
query_context = policy.prepare_query_context(context)
field = policy.query(queries, query_context)
action_logits = policy.action_logits(context)
```

Construction loads pretrained SigLIP weights and initializes task-specific
layers. Evaluation requires trained weights and matching configuration and
preprocessing. Constructors accept `encoder_kwargs`/`decoder_kwargs`, or injected
`encoder=`/`decoder=` modules with matching context widths. The training package
handles run checkpoints and resume.

Tests cover numerical decoder references, coordinate bases, per-sample HFOV,
state restoration, projection/depth gradients, single-frame inputs and text-mask
isolation without pretrained downloads. Navigation evaluation also depends on
the data recipe, training settings, integrator and controller.

## Self-contained checkpoint assets

`cofl.training.policy.load_policy` reconstructs SigLIP from the configs and
processor/tokenizer bundle inside the checkpoint, then strictly restores the
complete state dictionary. Both frozen SigLIP towers and task parameters are
stored. Loading does not fetch pretrained weights or consult the model path
used for initial training. The installed CoFL vision runtime is required.
