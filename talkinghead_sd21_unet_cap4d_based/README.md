# Talking-Head Video Diffusion (SD 2.1 UNet)

A latent video diffusion model for talking-head generation. Given a reference portrait, a driving video (FLAME expression tracking), and audio, the model generates temporally coherent talking-head video. Built on Stable Diffusion 2.1's UNet extended with 3D spatiotemporal attention, adapted from CAP4D (CVPR 2025).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Conditioning Signals](#conditioning-signals)
- [Expression-Weighted Loss](#expression-weighted-loss)
- [Experiments](#experiments)
- [Training](#training)
- [Inference](#inference)
- [Codebase Structure](#codebase-structure)

## Architecture Overview

The model operates in SD 2.1's VAE latent space (4 channels, 8× spatial downsampling). It generates T=16 consecutive frames per forward pass via a UNet whose 2D self-attention is replaced with 3D spatiotemporal attention — convolutions remain per-frame, only attention connects frames temporally.

Reference and generated frames are concatenated along the time axis `[1 ref | 15 generated]`. The 3D attention operates across all 16 frames, allowing generated frames to attend to the reference for identity preservation. For reference frames (ref_mask=1), the UNet bypasses learned prediction and outputs the known noise residual `x - z_input`, so the reference passes through unchanged while still participating in attention. Loss is masked to non-reference frames only.

## Conditioning Signals

| Signal | Mechanism | Channels/Dim | Description |
|---|---|---|---|
| **FLAME expression map** | Spatial addition to UNet first feature map | 46ch (42 pos enc + 3 deform + 1 ref mask) | Rasterized FLAME mesh vertices + expression offsets via PyTorch3D |
| **Audio** | Cross-attention in every transformer block | 1024 (wav2vec2-base → linear projection) | Per-frame audio tokens from ±2 context frames |
| **Reference frame** | Identity passthrough via z_input + ref_mask | 4ch VAE latent | GT latent for frame 0, zeroed for generated frames |

The 46-channel conditioning tensor is projected to 320 channels via a learned linear layer and added to the UNet's first feature map. Audio tokens serve as keys/values in cross-attention. `context_dim=1024` to match SD 2.1's pretrained cross-attention dimensions (wav2vec2's 768-dim output is projected to 1024).

## Expression-Weighted Loss

The diffusion loss is weighted per-pixel by expression deformation magnitude. Pixels where the face is actively deforming (mouth opening, brow raising) receive higher loss weight, forcing the denoiser to prioritize face dynamics:

```
weight = 1.0 + alpha * normalize(deformation_magnitude)
loss = weighted_mean(MSE(noise_pred, noise) * weight)
```

Static regions (background, forehead) keep baseline weight 1.0 — they're never suppressed, only active regions are amplified. `expr_weight_alpha=0` disables this (uniform loss).

The expression deformation magnitudes come from `expr_weight_map` (always computed from the full 46-channel expression field, even in ablation configs where the UNet doesn't see the expression map as conditioning).

## Experiments

Five configs explore different combinations of spatial conditioning and loss weighting:

| # | Config | UNet spatial conditioning | Loss weighting |
|---|---|---|---|
| 1 | `full_cond_weighted_loss.yaml` | 46ch (pos enc + deform + ref mask) | Weighted (α=5.0) |
| 2 | `full_cond_uniform_loss.yaml` | 46ch | Uniform (α=0.0) |
| 3 | `deform_only_weighted_loss.yaml` | 4ch (deform + ref mask only) | Weighted (α=5.0) |
| 4 | `no_expr_uniform_loss.yaml` | 1ch (ref mask only) | Uniform (α=0.0) |
| 5 | `no_expr_weighted_loss.yaml` | 1ch (ref mask only) | Weighted (α=5.0) |

Key comparisons: 1 vs 2 (effect of loss weighting), 1 vs 3 (is positional encoding necessary?), 1 vs 4 (overall FLAME contribution), 4 vs 5 (loss weighting without spatial conditioning), 3 vs 4 (minimal deformation signal vs none).

## Training

```bash
conda activate cap4d_env
export PYTHONPATH=/data/pouyan/baseline/repository/cap4d:$PYTHONPATH

python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/full_cond_weighted_loss.yaml \
    --gpus 0 1 2 3
```

The dataset splits each clip into non-overlapping 16-frame windows (deterministic). DDP with `find_unused_parameters=True` (frozen VAE and conditioning module parameters don't contribute to the training loss). Timestamped run directories with config snapshots, TensorBoard logging, and visualization grids (GT / Expression Map / Generated).

## Inference

```bash
python talkinghead_sd21_unet_cap4d_based/generate.py \
    --checkpoint outputs/full_cond_weighted_loss/run_YYYYMMDD/th-best.ckpt \
    --config talkinghead_sd21_unet_cap4d_based/configs/full_cond_weighted_loss.yaml \
    --ref_data /path/to/subject/ \
    --driving_fit /path/to/driving/fit.npz \
    --audio /path/to/audio.wav
```

Identity (shape, camera, appearance) from the reference subject, expressions (`expr`, `eye_rot`) from the driving video, audio from the driving audio file. The THSampler implements sliding-window DDIM for long video generation.

## Codebase Structure

```
talkinghead_sd21_unet_cap4d_based/
├── conditioning/
│   ├── th_conditioning.py    # FLAME → 46ch spatial conditioning (pos enc + deform + ref mask)
│   └── mesh2img.py           # PyTorch3D mesh rasterizer
├── model/
│   ├── th_diffusion.py       # Diffusion training loop (expression-weighted loss, CFG)
│   ├── th_unet.py            # SD 2.1 UNet + 3D attention + audio cross-attention
│   ├── th_sampler.py         # Sliding-window DDIM sampler
│   ├── audio_encoder.py      # wav2vec2 → per-frame audio tokens
│   ├── attention.py          # SpatioTemporalTransformer blocks
│   └── utils.py              # Noise schedule utilities
├── flame/
│   ├── flame.py              # FLAME 3DMM mesh computation
│   └── mouth.py              # Inner mouth vertices
├── data/
│   ├── video_dataset.py      # Dataset (non-overlapping 16-frame windows)
│   └── utils.py              # Image loading, cropping, vertex projection
├── configs/                   # 5 experiment configs
├── tests/                     # Unit + integration tests
├── train.py                   # Training entry point
└── generate.py                # Inference entry point
```
