# Warp-Conditioned Baseline

Full training run against the canonical v3 recipe
([`marionette/configs/base.yaml`](../../marionette/configs/base.yaml)) — no
overlays applied. Keep this experiment as the "point of truth" for what the
v3 model does out of the box.

## Recipe in one line

Same-identity self-supervised video diffusion with 49-channel spatial
conditioning (`pos_enc + driver_deform + warped_ref + ref_mask`) + wav2vec2
audio cross-attention, SD 2.1 UNet + 3D attention, 16-frame windows, ref frame
at slot 0.

## Running

```bash
conda activate marionette

# Single GPU
PYTHONPATH=. python experiments/warp_baseline/run.py

# Multi-GPU (DDP)
PYTHONPATH=. python experiments/warp_baseline/run.py --gpus 0 1 2 3 4 5 6 7

# Resume from a checkpoint
PYTHONPATH=. python experiments/warp_baseline/run.py \
    --resume outputs/warp_baseline/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt
```

## Outputs

```
outputs/warp_baseline/run_<timestamp>/
├── config_resolved.yaml                           # snapshot at run start
├── checkpoints/
│   ├── th-<step>.ckpt                             # every save_every_n_steps (periodic)
│   └── th-best-<step>-<val_loss>.ckpt             # top-1 by val/loss
├── logs/                                          # TensorBoard
└── visualizations/
    └── step_<step>/
        ├── sample_NN.png                          # 4-row grid (GT | deform | warped_ref | generated)
        └── sample_NN.mp4                          # same rows, with driver audio muxed in
```

## What runs periodically

Three independent periodic things fire during training:

### 1. Validation loop (Lightning)

- **When:** at the end of every training epoch (`val_check_interval=1.0`).
- **What:** `validation_step` runs `model.get_input(..., force_conditional=True)`
  (no CFG dropout) and computes the ε-MSE loss identical to training.
- **Metrics logged** (via `log_dict(sync_dist=True)` → averaged across ranks):
  - `val/loss_simple` — uniform ε-MSE, masked to non-reference slots.
  - `val/loss_vlb` — VLB-weighted version.
  - `val/loss` — combined (what `best_ckpt` monitors).
- **Budget:** at most `n_val_batches` batches per call (default 20).

### 2. `VisualizationCallback`

- **When:** every `val_every_n_steps` training steps (default 2000; set in
  `base.yaml` currently to 1 for debug).
- **Who:** rank 0 only, wrapped in a `torch.distributed.barrier()` pair so the
  other ranks wait through the long sampling phase (otherwise NCCL watchdog
  times out).
- **What:** for `n_vis_samples` val samples (rotates through the val set
  between firings), runs `SlidingWindowSampler` with `vis_ddim_steps`
  denoising steps, then saves:
  - **PNG grid** — 4 rows × 8 frames, labeled:
    - `Ground Truth` — decoded target latents.
    - `Driver Deform` — `spatial_cond[..., 42:45]` (the 3-channel per-vertex
      deformation map rasterized from the driver's FLAME).
    - `Warped Ref` — `spatial_cond[..., 45:48]` (the 3-channel backward-warped
      reference — this is the core v3 signal).
    - `Generated` — DDIM-sampled frames, red border on the reference slot.
  - **MP4** — the same 4 rows stacked vertically, with the driver's audio
    muxed in by ffmpeg (no audio → silent mp4).
  - **TensorBoard image** — the grid, logged under `vis/sample_<N>`.

### 3. Checkpoints (`ModelCheckpoint`)

- `periodic_ckpt` — saves `th-<step:06d>.ckpt` every `save_every_n_steps`
  train steps, `save_top_k=-1` (keep all).
- `best_ckpt` — saves `th-best-<step>-<val_loss>.ckpt` whenever `val/loss`
  improves, `save_top_k=1`.

## Knobs worth checking in `base.yaml`

- `n_steps` (currently 5000) — total training steps. Raise for a real run.
- `val_every_n_steps` (currently 1 — debug setting) — bump to 1000–2000 for
  production so the viz callback doesn't dominate wall-clock.
- `save_every_n_steps` (currently 250) — periodic checkpoint interval.
- `gpu_batch_size` × `virtual_batch_size` — controls gradient accumulation
  (`accumulate_grad_batches = virtual_batch_size // gpu_batch_size`).
