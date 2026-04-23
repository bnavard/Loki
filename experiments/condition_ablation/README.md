# Condition Ablation

A matrix of controlled ablations, all against the same `marionette_baseline`
recipe, designed to isolate the contribution of each conditioning pathway
feeding the generation UNet. Everything here is a single-variable ablation:
seed, dataset, SD 2.1 init, learning rate, optimizer, schedule, batch size,
and all other hyperparameters are inherited unchanged from
[`marionette/configs/base.yaml`](../../marionette/configs/base.yaml).

> **Paper note — what this folder is for.** Marionette's canonical recipe
> hangs its motion-conditioning on a FLAME mesh that is rasterized in the
> *reference's* camera / crop box. That rasterization lives in the same
> pixel frame the model is denoising into, so every pixel of the
> `spatial_cond` tensor aligns with the corresponding pixel of the
> generation target. The arms below remove or replace pieces of that
> pathway to quantify how much each piece is actually doing.
>
> The arms that substitute "natural driver video" for part of the FLAME
> channels feed the driver's *own* face-cropped frames as conditioning —
> i.e., pixels in the **driver's** pixel space, not the reference's. We
> expect these arms to underperform precisely because the conditioning no
> longer aligns spatially with the target; this is the controlled way to
> show that the alignment, not the presence of motion information, is what
> makes the FLAME recipe work.

## Arms

| Arm | What varies vs. baseline | `condition_channels` | Entry |
|---|---|---|---|
| **audio_off** | wav2vec2 cross-attention disabled; FLAME conditioning unchanged. | 45 | [`run_audio_off.py`](run_audio_off.py) |
| **no_flame** | Full 45ch FLAME `spatial_cond` replaced by driver's 3ch face-cropped video. Pos_enc + deform are both dropped. | 3 | [`run_no_flame.py`](run_no_flame.py) |
| **no_posenc** | 42ch positional encoding of vert positions dropped. Keep only 3ch deform. | 3 | [`run_no_posenc.py`](run_no_posenc.py) |
| **no_deform** | 3ch deform replaced by driver's 3ch face-cropped video. Pos_enc kept. | 45 | [`run_no_deform.py`](run_no_deform.py) |

The audio-on arm is already trained — it's the existing `marionette_baseline`
checkpoint. No separate "audio_on" config is duplicated here to avoid drift.

Each arm's conditioning implementation is a single standalone module under
its arm subfolder (`<arm>/conditioning.py`), imported via the arm's config
`cond_stage_config.target`. The original baseline module
[`marionette/conditioning/conditioning.py::SpatialConditioning`](../../marionette/conditioning/conditioning.py)
is untouched — arms swap the cond_stage class, they don't mutate it.

## What stays constant across arms (the ablation invariants)

- **Same seed.** `cfg.seed: 42` is inherited from base. `run_training` calls
  `pl.seed_everything(cfg.seed, workers=True)`.
- **Same data & data order.** `DataLoader(shuffle=True)` with deterministic
  worker seeds → identical batch order across arms. The dataset emits an
  extra `driver_video` key per batch; arms that don't use it ignore the key
  silently (shared storage with `target_video[1:]`, zero memory cost).
- **Same ref frame sampling.** `TalkingHeadDataset.ref_sampling_seed=0`
  XOR'd with sample index.
- **Same SD 2.1 init.** `data/models/v2-1_512-ema-pruned.ckpt` loaded via
  `RefFeatureExtractor.load_sd21_into_ref`. Ref UNet and gen UNet start from
  identical weights in every arm.
- **Same learning rate, optimizer, schedule, batch size, n_steps.**
- **Determinism.** `deterministic=True` on `pl.Trainer`,
  `torch.backends.cudnn.enabled = False`.

What **differs** per arm is exactly one of the conditioning pathways, by
design. For audio_off: the audio encoder is not instantiated and every
transformer block skips its cross-attention pass. For no_flame / no_posenc:
the ConditioningEncoder's first conv is 3 input channels instead of 45
(smaller first layer; everything downstream is identical). For no_deform:
the 45-channel cond tensor is the same width but its last 3 channels are
driver-video pixels instead of rasterized deformation.

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

This keeps the ablation honest: the only variable a no_flame / no_posenc
checkpoint sees vs. baseline is the *content* of `spatial_cond`, not the
capacity of the module learning to consume it.

## Per-arm details

### audio_off — audio cross-attention off

Canonical FLAME conditioning preserved; only the wav2vec2 audio pathway is
removed. See [`audio_off/config.yaml`](audio_off/config.yaml). The overlay
[`marionette/configs/overlays/audio/off.yaml`](../../marionette/configs/overlays/audio/off.yaml)
sets `audio_encoder_config: null` AND `use_audio_context: false`;
`MarionetteDiffusion.__init__` validates these two flags are consistent at
construction time.

Hypothesis: audio contributes non-trivially to lip-sync fidelity (LSE-D /
LSE-C) but is near-zero for identity preservation and overall pixel
quality. The `experiments/evaluation_metrics/` SyncNet pipeline is the
direct way to test this.

### no_flame — driver video instead of the entire FLAME conditioning

`spatial_cond` becomes literally the driver's face-cropped 3-channel video
(in `[-1, 1]`), read from `hint["driver_video"]`. No rasterization, no
pos_enc, no deformation map. See
[`no_flame/conditioning.py`](no_flame/conditioning.py) — it's a ~40-line
pass-through module.

Hypothesis: the model cannot recover baseline quality. The conditioning
signal lives in the **driver's pixel space**; the denoising target lives in
the **reference's pixel space**. Even at same-identity training, same-clip
windows mean the ref and the target sit at different head poses, so the
conditioning and target pixels do not line up 1-to-1. At cross-identity
inference the mismatch is twice as bad — the driver's face and the
reference's face occupy different positions, sizes, and orientations within
their respective crop boxes. The model has no way to use the conditioning
as spatially aligned supervision.

This arm is the direct counterfactual to the paper's claim that the
FLAME-rasterization-in-ref-space invariant is what makes the recipe work.

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

### no_deform — pos_enc + driver video (replacing the deform map)

`spatial_cond` = `[pos_enc_42ch, driver_video_3ch]` concatenated along the
channel axis, total 45 channels (UNet unchanged). The positional encoding
still carries head pose + geometry, aligned to the reference's crop. The
driver video provides motion information, but in the driver's own pixel
space (not aligned to the reference). See
[`no_deform/conditioning.py`](no_deform/conditioning.py).

Hypothesis: between `no_flame` (nothing FLAME) and the baseline (everything
FLAME), this arm isolates the value of **aligned expression deformation**
specifically. If it approaches baseline quality, the deform channels are
mostly informational redundancy on top of pos_enc. If it drops noticeably,
the aligned deformation map is doing real work.

## Launch

```bash
conda activate marionette
PYTHONPATH=. python experiments/condition_ablation/run_audio_off.py  --gpus 0 1 2 3
PYTHONPATH=. python experiments/condition_ablation/run_no_flame.py   --gpus 0 1 2 3
PYTHONPATH=. python experiments/condition_ablation/run_no_posenc.py  --gpus 0 1 2 3
PYTHONPATH=. python experiments/condition_ablation/run_no_deform.py  --gpus 0 1 2 3
```

Each arm runs the canonical 30k-step schedule. Single-GPU is fine; DDP
matches the baseline's default launch.

## Evaluating the arms

Each checkpoint is a drop-in for the existing
[`experiments/marionette_eval/`](../marionette_eval/) pipeline. The
evaluator reads `cond_stage_config.target` from the resolved config and
dispatches to the correct conditioning class automatically; the dataset
already emits `driver_video`, which the eval inference path (`Evaluator.run_one`)
also constructs at inference time for cross-identity pairs via
`prepare_driver_frames`.

## Output layout

```
outputs/
├── marionette_baseline/run_<ts>/                # audio-on, full FLAME — canonical
└── condition_ablation/
    ├── audio_off/run_<ts>/
    │   ├── config_resolved.yaml
    │   ├── checkpoints/
    │   ├── logs/
    │   └── visualizations/
    ├── no_flame/run_<ts>/...
    ├── no_posenc/run_<ts>/...
    └── no_deform/run_<ts>/...
```

## Structure

```
experiments/condition_ablation/
├── README.md
├── run_audio_off.py
├── run_no_flame.py
├── run_no_posenc.py
├── run_no_deform.py
├── audio_off/
│   └── config.yaml
├── no_flame/
│   ├── config.yaml
│   └── conditioning.py        # NaturalVideoConditioning (3ch)
├── no_posenc/
│   ├── config.yaml
│   └── conditioning.py        # DeformOnlyConditioning (3ch)
└── no_deform/
    ├── config.yaml
    └── conditioning.py        # PosEncPlusVideoConditioning (45ch)
```

Each arm's `conditioning.py` exposes a single `nn.Module` class with the
same `forward(batch: dict) -> {"spatial_cond": tensor}` contract as the
baseline `SpatialConditioning`, so the UNet plumbing doesn't need to know
which arm is active.
