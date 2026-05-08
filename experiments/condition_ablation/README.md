# Condition Ablation

Single-variable ablations on the FLAME conditioning pathway feeding the gen
UNet. All arms inherit from
[`loki/configs/base.yaml`](../../loki/configs/base.yaml) — same
seed, dataset, SD 2.1 init, optimizer, schedule, batch size, n_steps — and
differ in exactly one property of the conditioning, by design.

The arms split into two groups:

- **Rasterized arms** (`no_posenc`, `no_deform`) — vary *which channels* of
  the canonical 45-channel rasterized `spatial_cond` the model sees. The
  conditioning representation (pixel-space, ref-camera-aligned) is held
  fixed; only the per-channel content is ablated.
- **Representation arm** (`flame_vector`) — varies the conditioning
  *representation itself*, replacing the rasterized 45ch tensor with a
  spatially-broadcast projection of raw FLAME parameters. Tests §4.3.

> **Paper note — what this folder is for.** Loki's canonical recipe
> hangs its motion-conditioning on a FLAME mesh that is rasterized in the
> *reference's* camera / crop box. That rasterization lives in the same
> pixel frame the model is denoising into, so every pixel of `spatial_cond`
> aligns with the corresponding pixel of the generation target. The
> rasterized arms below remove one half of that 45-channel signal so the
> contribution of each half can be read off directly. The representation
> arm steps out of pixel space entirely so the value of *being in pixel
> space* can be read off too.

## Arms

| Arm | FLAME conditioning | `condition_channels` | Entry |
|---|---|---|---|
| **no_posenc** | 42ch pos_enc dropped; 3ch deform kept | 3  | [`run_no_posenc.py`](run_no_posenc.py) |
| **no_deform** | 3ch deform dropped; 42ch pos_enc kept. No substitute channel — identity reaches the model via the ref UNet, not via the cond tensor. | 42 | [`run_no_deform.py`](run_no_deform.py) |
| **flame_vector** | No rasterization. Raw 77-dim FLAME params (expr+rot+neck_rot+jaw_rot+eye_rot) → MLP → spatially-broadcast to (64, 64, 320). Encoder degenerates to a single zero-init Conv stem. | 320 | [`run_flame_vector.py`](run_flame_vector.py) |

Each arm's conditioning implementation is a single standalone module under
its arm subfolder (`<arm>/conditioning.py`), imported via the arm's config
`cond_stage_config.target`. The baseline module
[`loki/conditioning/conditioning.py::SpatialConditioning`](../../loki/conditioning/conditioning.py)
is untouched — arms swap the cond_stage class, they don't mutate it.

### Intended comparisons

- `no_posenc` vs `loki_baseline` → **pos_enc contribution** (only the
  42ch positional encoding is dropped).
- `no_deform` vs `loki_baseline` → **aligned deformation contribution**
  (only the 3ch deform channels are dropped, no substitute). Identity
  reaches both arms through the ref UNet, so the cond tensor doesn't need
  to carry identity information.
- `flame_vector` vs `loki_baseline` → **pixel-space representation
  contribution** (the rasterization itself is dropped; the same parametric
  motion information is delivered as a spatially-constant tile instead).
  The §4.3 falsification test: if the rasterized arm wins, the win is
  attributable to the spatial inductive bias, not to the parametric content.

## What stays constant across arms (the ablation invariants)

- **Same seed.** `cfg.seed: 42` is inherited from base. `run_training` calls
  `pl.seed_everything(cfg.seed, workers=True)`.
- **Same data & data order.** `DataLoader(shuffle=True)` with deterministic
  worker seeds → identical batch order across arms.
- **Same ref frame sampling.** `TalkingHeadDataset.ref_sampling_seed=0`
  XOR'd with sample index.
- **Same SD 2.1 init.** `data/models/v2-1_512-ema-pruned.ckpt` loaded via
  `RefFeatureExtractor.load_sd21_into_ref`. Ref UNet and gen UNet start
  from identical weights in every arm.
- **Same learning rate, optimizer, schedule, batch size, n_steps.**
- **Determinism.** `deterministic=True` on `pl.Trainer`,
  `torch.backends.cudnn.enabled = False`.

What **differs** per arm is exactly one of the conditioning pathways, by
design. For `no_posenc` the ConditioningEncoder's first conv is 3 input
channels instead of 45; for `no_deform` it's 42; for `flame_vector` the
ConditioningEncoder's role itself is the variable under test (see below).
Everything else downstream is identical.

### Load-bearing invariant: one ConditioningEncoder architecture for the rasterized arms

The [`ConditioningEncoder`](../../loki/model/conditioning_encoder.py)
is the SD-style conv stack that downsamples `spatial_cond` from 512×512 to
the UNet's 64×64 latent resolution and emits `model_channels=320` feature
maps added to the first UNet feature map. **Its architecture must be
identical across every rasterized arm** (`loki_baseline`, `no_posenc`,
`no_deform`) — the per-arm `conditioning.py` modules only decide *what
tensor* gets fed in; they must never change the encoder itself. Concretely:

- `cond_input_resolution: 512`, `cond_latent_resolution: 64`, and
  `cond_stage_channels: [64, 128, 256, 320]` are inherited from base.yaml
  and **must not be overridden in any rasterized arm config**.
- The **only** encoder parameter that differs across rasterized arms is
  `condition_channels`, which sets the width of the stem `Conv3×3`'s
  `in_channels`. The stem is ≤ 30k params out of a 1B+ model; the shape of
  the per-pixel learning stack downstream is identical across arms.

This keeps the rasterized-arm comparison honest: the only variable a
`no_posenc` / `no_deform` checkpoint sees vs. baseline is the *content* of
`spatial_cond`, not the capacity of the module learning to consume it.

#### Why `flame_vector` is exempt

The encoder's whole job is to *consume a 512×512 image*. In the
`flame_vector` arm there is no image to consume — the conditioning module
emits a tensor that is already at the latent resolution (64×64) and
already at `model_channels` width (320), produced by an MLP and a spatial
broadcast. Keeping the 512×512 → 64×64 downsampling stack around just to
preserve the invariant would be running 3 stages of strided convs on a
spatially-constant input, which is wasted compute and obscures the
variable the arm is testing. So `flame_vector` overrides
`cond_input_resolution`, `cond_latent_resolution`, and `cond_stage_channels`
to collapse the encoder to a single zero-init Conv3×3 stem at 64×64. The
zero-init out-conv is what carries the step-0 invariant — the conditioning
contributes exactly zero at training start, same as the rasterized arms.

## Per-arm details

### no_posenc — deform-only FLAME conditioning

`spatial_cond` = 3-channel per-vertex expression deformation rasterized in
ref space (same path as the baseline's last three channels). The 42-channel
positional encoding of vert positions is dropped entirely. See
[`no_posenc/conditioning.py`](no_posenc/conditioning.py).

Hypothesis: quality drops substantially. The deformation map encodes
*per-vertex* expression offsets relative to the mean FLAME shape, but
**no head pose and no global geometry** — it does not tell the model where
in the frame the face should be, which way it is facing, or how large it
is. Without pos_enc the conditioning signal is a blobby expression-only
map floating in an otherwise structureless 512×512 canvas.

### no_deform — pos_enc only (deform map dropped, no substitute)

`spatial_cond` = 42-channel sinusoidal pos_enc of rasterized FLAME vertex
positions. The 3ch per-vertex expression deformation is dropped entirely;
nothing is substituted in its place. `condition_channels` = 42. See
[`no_deform/conditioning.py`](no_deform/conditioning.py).

Why no substitute channel: identity is already carried by the frozen
reference UNet (K/V feature injection into every self-attention block),
so the cond tensor doesn't need to encode identity. Pasting driver-video
pixels into the freed 3 channels would conflate the deform ablation with a
spatial-misalignment confound (driver video lives in driver-crop space,
not ref-crop space). This arm answers "how much does the deform map
contribute on top of pos_enc?" cleanly.

Hypothesis: between `no_posenc` (deform only, no pose / geometry) and the
baseline (full FLAME), this arm isolates the value of the **aligned
expression deformation** specifically. If it approaches baseline quality,
the deform channels are mostly informational redundancy on top of pos_enc
+ the ref UNet's identity features. If it drops noticeably, the aligned
deformation map is doing real work for expression fidelity.

### flame_vector — raw FLAME parameters, no rasterization

`spatial_cond` is built by projecting per-frame raw FLAME motion parameters
(77-dim: `expr|rot|neck_rot|jaw_rot|eye_rot`) through a shared MLP to the
UNet's `model_channels` (320), then **spatially broadcasting** the
resulting per-frame vector across the 64×64 latent grid. Every spatial
position carries the same 320-dim feature for a given frame, so the
conditioning has *zero spatial structure* by construction. The dataset
emits a new `driver_flame_params` tensor under `hint` (gated by
`emit_flame_params: true`) so the rasterized arms remain byte-identical.
See [`flame_vector/conditioning.py`](flame_vector/conditioning.py).

The spatial-broadcast trick follows the
**Spatial Broadcast Decoder** (Watters et al., 2019) and **FiLM** (Perez
et al., 2018) — the standard mechanism for fusing a global vector into a
conv feature map. The arm therefore does not strawman vector conditioning;
it uses the established trick.

Hypothesis: this arm degrades on both head pose and expression — visibly
worse than `no_posenc` and `no_deform`. The mechanism (per §4.3): the
rasterized baseline hands the diffusion model a free spatial mapping (the
value at pixel (i, j) corresponds to the face point projecting to (i, j)),
while the broadcast tile contains no information about *where on the face*
each FLAME coefficient applies. The downstream conv stack and self-attention
have to recover that mapping from a feature map that doesn't carry it. If
this arm degrades while `no_deform` (which keeps pos_enc, the rasterized
spatial-mapping channels) holds up better, the §4.3 claim is supported:
the win is attributable to the spatial inductive bias, not to the
parametric content of the conditioning.

## Launch

```bash
conda activate loki
PYTHONPATH=. python experiments/condition_ablation/run_no_posenc.py     --gpus 0 1 2 3
PYTHONPATH=. python experiments/condition_ablation/run_no_deform.py     --gpus 0 1 2 3
PYTHONPATH=. python experiments/condition_ablation/run_flame_vector.py  --gpus 0 1 2 3
```

Each arm runs the canonical 30k-step schedule. Single-GPU is fine; DDP
matches the baseline's default launch.

## Evaluating the arms

Each checkpoint is a drop-in for the existing
[`experiments/loki_eval/`](../loki_eval/) pipeline. The
evaluator reads `cond_stage_config.target` from the resolved config and
dispatches to the correct conditioning class automatically.

## Output layout

```
outputs/
├── loki_baseline/run_<ts>/                # canonical
└── condition_ablation/
    ├── no_posenc/run_<ts>/
    │   ├── config_resolved.yaml
    │   ├── checkpoints/
    │   ├── logs/
    │   └── visualizations/
    ├── no_deform/run_<ts>/...
    └── flame_vector/run_<ts>/...
```

## Structure

```
experiments/condition_ablation/
├── README.md
├── run_no_posenc.py
├── run_no_deform.py
├── run_flame_vector.py
├── no_posenc/
│   ├── config.yaml
│   └── conditioning.py        # PosEncOnlyConditioning  (42ch rasterized)
├── no_deform/
│   ├── config.yaml
│   └── conditioning.py        # DeformOnlyConditioning  (3ch  rasterized)
└── flame_vector/
    ├── config.yaml
    └── conditioning.py        # FlameVectorConditioning (320ch broadcast)
```

Each arm's `conditioning.py` exposes a single `nn.Module` class with the
same `forward(batch: dict) -> {"spatial_cond": tensor}` contract as the
baseline `SpatialConditioning`, so the UNet plumbing doesn't need to know
which arm is active.
