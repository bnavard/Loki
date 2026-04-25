# Marionette Evaluation

Visual-only evaluation of a Marionette checkpoint on the validation set. Two
runners, one file per mode:

| Script | What it does | Output count |
|---|---|---|
| [`run_cross_identity.py`](run_cross_identity.py) | Every usable YouTube identity appears once as reference and once as driver (a derangement — `ref_identity ≠ driver_identity`). | N = number of usable identities. |
| [`run_same_identity.py`](run_same_identity.py)   | Every usable YouTube identity contributes `samples_per_identity` self-reconstructions inside one of its own clips. | N × samples_per_identity. |

Both runners share [`evaluator.py`](evaluator.py) for the inference path and
[`pairing.py`](pairing.py) for the sample-list construction. This phase saves
panels + muxed mp4s only — numerical metrics come next.

## Quick start

```bash
conda activate marionette

# Cross-identity (recommended first)
PYTHONPATH=. python experiments/marionette_eval/run_cross_identity.py \
    --config     experiments/marionette_eval/configs/cross_identity.yaml \
    --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

# Same-identity
PYTHONPATH=. python experiments/marionette_eval/run_same_identity.py \
    --config     experiments/marionette_eval/configs/same_identity.yaml \
    --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt
```

`--output_dir`, `--cfg_scale`, `--seed`, `--device` are optional overrides;
everything else lives in the YAML so runs are reproducible from the config
alone.

## What the config controls

```yaml
base: marionette/configs/base.yaml    # loads val_dataset paths, model/unet/VAE
output_dir: outputs/marionette_eval/<mode>
checkpoint: null                      # or pass via --checkpoint

seed: 42                              # seeds pairing, ref-frame + driver-start draws, torch RNG

inference:
  n_frames:     16                    # must match UNet time_steps in base.yaml
  cfg_scale:    2.0                   # classifier-free guidance scale at sample time
  n_ddim_steps: 50

# same_identity.yaml only:
eval:
  samples_per_identity: 2
  min_ref_driver_gap:   16            # ref stays ≥ this many frames outside target window
```

The validation clip list, FLAME tracking root, video root, and audio root all
come from `val_dataset.params` in [marionette/configs/base.yaml](../../marionette/configs/base.yaml)
— edit there, not here, if the data paths move.

## Sampling

Identity is the prefix of a clip ID before `_NA_`. For the current val set
([data/derived/val_clips.json](../../data/derived/val_clips.json)): 773 clips,
125 unique YouTube identities.

### Cross-identity (derangement)

1. Group val clips by YouTube ID.
2. Drop identities whose clips are all shorter than `n_frames`.
3. Shuffle the identity list under `seed` and draw a permutation until it has
   no fixed point (a derangement). This guarantees `ref_identity ≠
   driver_identity` for every pair and that each identity appears once as
   ref and once as driver.
4. Per pair: draw `ref_clip` and `driver_clip` uniformly from each identity's
   clip list. Draw `ref_frame_idx ∈ [0, ref_clip_len)` and
   `driver_start_idx ∈ [0, driver_clip_len - n_frames]`.

### Same-identity (per-identity, windowed)

1. Keep only identities with at least one clip of length
   ≥ `n_frames + 2 * min_ref_driver_gap` (there has to be room for the ref
   outside the target window).
2. Per identity, `samples_per_identity` times: draw a clip uniformly, draw
   `driver_start_idx` uniformly, then draw `ref_frame_idx` uniformly from
   frames outside `[driver_start - min_gap, driver_start + n_frames + min_gap)`.

All randomness is funnelled through a single `np.random.default_rng(seed)`,
so the same config + seed reproduces the same schedule.

## Inference path (per sample)

This is what `evaluator.Evaluator.run_one` does, and it mirrors
[marionette/generate.py](../../marionette/generate.py) modulo the driver-start
offset:

1. Load `ref_fit` and `driver_fit` from `fit.npz`.
2. `prepare_reference(ref_fit, ref_frame_idx, …)` → face-cropped ref image
   (512×512, `[-1, 1]`) + the crop_box that defines ref pixel space.
3. `retarget_driver_verts(ref_fit, driver_fit, crop_box, n_frames, …,
   driver_start=driver_start_idx)` → `(T, V, 3)` NDC verts and `(T, V, 3)`
   expression offsets, computed as `β_ref + ψ_driver[t] + θ_driver[t]` under
   the reference's camera.
4. Instantiate the active cond_stage module via
   `instantiate_from_config(cfg.model.params.cond_stage_config)` — this
   resolves to the baseline's `SpatialConditioning` for a baseline
   checkpoint, or to one of the per-arm modules under
   [`experiments/condition_ablation/`](../condition_ablation/) for an
   ablation checkpoint. Run it on the hint dict to get
   `spatial_cond (1, T, H, W, C)`. `C` is 45 for the baseline; ablation
   arms emit different widths (3 or 42).
5. VAE-encode the ref image → `ref_z (1, 4, h, w)`. This is the sole identity
   signal; `RefFeatureExtractor` consumes it inside `sample_video`.
6. If the model's audio encoder is present (default on this checkpoint), load
   the **driver's** wav, slice `[driver_start_idx, driver_start_idx + n_frames)`,
   build centered ±`audio_context_frames` windows, and encode them. Otherwise
   skip audio entirely. The runtime check is `model.audio_encoder is not None`
   — when the architecture drops audio, this code keeps working unchanged.
7. Build `c_uncond` as zero-filled tensors for every key in `c_cond`.
8. `model.sample_video(...)` — DDIM with classifier-free guidance; returns
   `(T, 4, h, w)` latents.
9. VAE-decode to `(T, 3, 512, 512)` and write:
   - `panel.png` — 4-row grid: Reference (static) | Driver Video |
     `<cond preview>` | Generated. The third row's slice + label come from
     the active cond_stage's `VIZ_SLICE` + `VIZ_LABEL` class attrs (e.g.
     baseline → `(42, 45)` ⇒ `"Driver Deform"`).
   - `panel.mp4` — the same panel stacked vertically, with driver audio
     muxed when the model has an audio encoder.

Per-frame PNGs are intentionally **not** saved (phase-2 metrics read from
`panel.mp4`).

## Output layout

```
outputs/marionette_eval/cross_identity/run_<timestamp>/
├── config_resolved.yaml
└── samples/
    └── NNN_ref-<YT_A>__drv-<YT_B>/
        ├── panel.png
        └── panel.mp4

outputs/marionette_eval/same_identity/run_<timestamp>/
├── config_resolved.yaml
└── samples/
    └── NNN_<YT>_<k>/
        ├── panel.png
        └── panel.mp4
```

## Audio-optional note

The evaluator reads `model.audio_encoder is not None` at runtime. If a future
training config sets `audio_encoder_config: null` (or removes audio from the
architecture entirely), the eval scripts will skip the wav load + the audio
encoder call and pass `None` as the audio context to `sample_video`. No
change needed here.
