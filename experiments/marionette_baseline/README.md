# Marionette Baseline

Full training run against the canonical recipe
([`marionette/configs/base.yaml`](../../marionette/configs/base.yaml)) — no
overlays applied. Keep this experiment as the "point of truth" for what the
model does out of the box.

## Recipe in one line

Same-identity self-supervised video diffusion: 45-channel FLAME spatial
conditioning (`pos_enc + driver_deform`), wav2vec2 audio cross-attention, SD
2.1 generation UNet with 3D spatiotemporal attention, and a frozen SD 2.1
reference UNet whose per-layer self-attention features are injected as K/V
tokens into the gen UNet. T=16 target frames per forward pass; the reference
does not occupy a slot in the output.

## Running

```bash
conda activate marionette

# Single GPU
PYTHONPATH=. python experiments/marionette_baseline/run.py

# Multi-GPU (DDP)
PYTHONPATH=. python experiments/marionette_baseline/run.py --gpus 0 1 2 3 4 5 6 7

# Resume from a checkpoint
PYTHONPATH=. python experiments/marionette_baseline/run.py \
    --resume outputs/marionette_baseline/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt
```

## Outputs

```
outputs/marionette_baseline/run_<timestamp>/
├── config_resolved.yaml                           # snapshot at run start
├── log.txt                                        # mirrored stdout / stderr (rank 0; install_log_tee)
├── checkpoints/
│   ├── th-<step>.ckpt                             # every save_every_n_steps (periodic)
│   └── th-best-<step>-<val_loss>.ckpt             # top-1 by val/loss
├── logs/                                          # TensorBoard
└── visualizations/
    └── step_<step>/
        ├── sample_NN.png                          # 4-row grid (Reference | Ground Truth | <cond preview> | Generated)
        └── sample_NN.mp4                          # same rows, with driver audio muxed in
```

Row 3 of the panel ("`<cond preview>`") is named by the active cond_stage
module's `VIZ_LABEL` class attr — `"Driver Deform"` for the baseline, but
`"Driver Video"` / `"Pos Enc"` etc. for the
[condition_ablation](../condition_ablation/) arms. The viz code is
arm-agnostic; the cond module owns its own row label + slice range.

## What runs periodically

Three independent periodic things fire during training:

### 1. Validation loop (Lightning)

- **When:** at the end of every training epoch (`val_check_interval=1.0`).
- **What:** `validation_step` runs `model.get_input(..., force_conditional=True)`
  (no CFG dropout) and computes the ε-MSE loss identical to training.
- **Metrics logged** (via `log_dict(sync_dist=True)` → averaged across ranks):
  - `val/loss_simple` — uniform ε-MSE across all T target slots (no slot masking; the reference lives in the separate ref UNet, not in the loss tensor).
  - `val/loss_vlb` — VLB-weighted version.
  - `val/loss` — combined (what `best_ckpt` monitors).
- **Budget:** at most `n_val_batches` batches per call (default 20).

### 2. `VisualizationCallback`

- **When:** every `val_every_n_steps` training steps (default 3000).
- **Who:** **every rank participates** — the work is sharded across ranks
  by sample (rank `r` of world size `W` handles
  `n_vis_samples // W` samples, with the remainder distributed to the
  first `n_vis_samples % W` ranks). Bracketing
  `torch.distributed.barrier()` calls keep the cluster aligned: every rank
  arrives at the viz callback together, fast ranks wait at the post-barrier
  for slow ones before resuming training so no rank races into the next
  step's all-reduce.
- **What:** for the run's `n_vis_samples` val samples (rotated through the
  val set across firings), each rank runs `MarionetteDiffusion.sample_video`
  on its slice with `vis_ddim_steps` denoising steps + classifier-free
  guidance at `cfg.inference.cfg_scale`, then writes:
  - **PNG grid** — 4 rows × up to 8 frames. Row labels are vertical
    (rotated 90° CCW) in a 70-px strip on the left so they don't clip into
    the frame content.
    - `Reference` — decoded reference latent, broadcast across time (static row).
    - `Ground Truth` — decoded target latents.
    - **`<cond preview>`** — a 3-channel slice of `spatial_cond` decided by
      the cond_stage module's `VIZ_SLICE` + `VIZ_LABEL` class attrs.
      Baseline → `(42, 45)` ⇒ `"Driver Deform"`. Condition-ablation arms
      override per arm.
    - `Generated` — DDIM+CFG-sampled frames from `sample_video`.
  - **MP4** — the same 4 rows stacked vertically, with the driver's audio
    muxed in by ffmpeg if the model's audio encoder is active (silent mp4
    otherwise).
  - **TensorBoard image** — rank 0 only logs to `vis/sample_<abs_idx>` —
    TB event files aren't safe under multi-rank concurrent writes; PNG and
    mp4 on disk are the full set across ranks.

File names use absolute sample indices (`sample_00`..`sample_{N-1}`) so
the union across ranks is a contiguous run with no collisions.

### 3. Checkpoints (`ModelCheckpoint`)

- `periodic_ckpt` — saves `th-<step:06d>.ckpt` every `save_every_n_steps`
  train steps, `save_top_k=-1` (keep all).
- `best_ckpt` — saves `th-best-<step>-<val_loss>.ckpt` whenever `val/loss`
  improves, `save_top_k=1`.

## Knobs worth checking in `base.yaml`

- `n_steps` (currently `30000`) — total training steps.
- `val_every_n_steps` (currently `3000`) — viz callback cadence. Drop only
  if you don't mind the visualization run dominating wall-clock between
  steps.
- `save_every_n_steps` (currently `10000`) — periodic checkpoint interval
  (in addition to the `best_ckpt` callback that fires on every val/loss
  improvement).
- `n_vis_samples` (currently `8`) — total samples per visualization fire.
  Sharded across ranks at run time; each rank handles `n // world_size`
  samples (remainder distributed to low-index ranks).
- `gpu_batch_size` × `virtual_batch_size` — controls gradient accumulation
  (`accumulate_grad_batches = virtual_batch_size // gpu_batch_size`).
