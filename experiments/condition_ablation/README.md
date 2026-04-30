# Condition Ablation

Single-variable ablations on the FLAME spatial-conditioning pathway feeding
the gen UNet. Both arms inherit unchanged from
[`marionette/configs/base.yaml`](../../marionette/configs/base.yaml) — same
seed, dataset, SD 2.1 init, optimizer, schedule, batch size, n_steps — and
differ only in *which* channels of the 45-channel `spatial_cond` tensor
the model sees.

> **Paper note — what this folder is for.** Marionette's canonical recipe
> hangs its motion-conditioning on a FLAME mesh that is rasterized in the
> *reference's* camera / crop box. That rasterization lives in the same
> pixel frame the model is denoising into, so every pixel of `spatial_cond`
> aligns with the corresponding pixel of the generation target. The arms
> below remove one half of that 45-channel signal so the contribution of
> each half can be read off directly.

## Arms

| Arm | FLAME conditioning | `condition_channels` | Entry |
|---|---|---|---|
| **no_posenc** | 42ch pos_enc dropped; 3ch deform kept | 3  | [`run_no_posenc.py`](run_no_posenc.py) |
| **no_deform** | 3ch deform dropped; 42ch pos_enc kept. No substitute channel — identity reaches the model via the ref UNet, not via the cond tensor. | 42 | [`run_no_deform.py`](run_no_deform.py) |

Each arm's conditioning implementation is a single standalone module under
its arm subfolder (`<arm>/conditioning.py`), imported via the arm's config
`cond_stage_config.target`. The baseline module
[`marionette/conditioning/conditioning.py::SpatialConditioning`](../../marionette/conditioning/conditioning.py)
is untouched — arms swap the cond_stage class, they don't mutate it.

### Intended comparisons

- `no_posenc` vs `marionette_baseline` → **pos_enc contribution** (only the
  42ch positional encoding is dropped).
- `no_deform` vs `marionette_baseline` → **aligned deformation contribution**
  (only the 3ch deform channels are dropped, no substitute). Identity
  reaches both arms through the ref UNet, so the cond tensor doesn't need
  to carry identity information.

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
channels instead of 45; for `no_deform` it's 42. Everything downstream is
identical.

### Load-bearing invariant: one ConditioningEncoder architecture for all arms

The [`ConditioningEncoder`](../../marionette/model/conditioning_encoder.py)
is the SD-style conv stack that downsamples `spatial_cond` from 512×512 to
the UNet's 64×64 latent resolution and emits `model_channels=320` feature
maps added to the first UNet feature map. **Its architecture must be
identical across every arm in this folder** — the per-arm `conditioning.py`
modules only decide *what tensor* gets fed in; they must never change the
encoder itself. Concretely:

- `cond_input_resolution: 512`, `cond_latent_resolution: 64`, and
  `cond_stage_channels: [64, 128, 256, 320]` are inherited from base.yaml
  and **are not overridden in any arm config**. Do not override them.
- The **only** encoder parameter that differs across arms is
  `condition_channels`, which sets the width of the stem `Conv3×3`'s
  `in_channels`. The stem is ≤ 30k params out of a 1B+ model; the shape of
  the per-pixel learning stack downstream is identical across arms.

This keeps the ablation honest: the only variable a `no_posenc` /
`no_deform` checkpoint sees vs. baseline is the *content* of `spatial_cond`,
not the capacity of the module learning to consume it.

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

## Launch

```bash
conda activate marionette
PYTHONPATH=. python experiments/condition_ablation/run_no_posenc.py  --gpus 0 1 2 3
PYTHONPATH=. python experiments/condition_ablation/run_no_deform.py  --gpus 0 1 2 3
```

Each arm runs the canonical 30k-step schedule. Single-GPU is fine; DDP
matches the baseline's default launch.

## Evaluating the arms

Each checkpoint is a drop-in for the existing
[`experiments/marionette_eval/`](../marionette_eval/) pipeline. The
evaluator reads `cond_stage_config.target` from the resolved config and
dispatches to the correct conditioning class automatically.

## Output layout

```
outputs/
├── marionette_baseline/run_<ts>/                # canonical
└── condition_ablation/
    ├── no_posenc/run_<ts>/
    │   ├── config_resolved.yaml
    │   ├── checkpoints/
    │   ├── logs/
    │   └── visualizations/
    └── no_deform/run_<ts>/...
```

## Structure

```
experiments/condition_ablation/
├── README.md
├── run_no_posenc.py
├── run_no_deform.py
├── no_posenc/
│   ├── config.yaml
│   └── conditioning.py        # DeformOnlyConditioning (3ch)
└── no_deform/
    ├── config.yaml
    └── conditioning.py        # PosEncOnlyConditioning (42ch)
```

Each arm's `conditioning.py` exposes a single `nn.Module` class with the
same `forward(batch: dict) -> {"spatial_cond": tensor}` contract as the
baseline `SpatialConditioning`, so the UNet plumbing doesn't need to know
which arm is active.
