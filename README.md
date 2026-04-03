# Expressive Talking Head

A modular framework for audio-driven talking-head video generation, built on top of [CAP4D](https://github.com/felixtaubner/CAP4D) (CVPR 2025). The system generates temporally coherent talking-head videos from a reference portrait image, a driving video (providing facial expressions via FLAME 3DMM tracking), and an audio track. It combines a Stable Diffusion 2.1 UNet extended with 3D spatiotemporal attention for video generation, wav2vec2 audio cross-attention for lip sync, and FLAME expression maps for spatial facial control.

The repository also includes a text-to-expression-field pipeline that fine-tunes Wan2.2-T2V (via LoRA) to synthesize FLAME expression dense fields directly from text descriptions, enabling a fully text-driven generation pathway without requiring a driving video.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
  - [talkinghead_sd21_unet_cap4d_based](#talkinghead_sd21_unet_cap4d_based)
  - [talkinghead_wan22_animate_14b](#talkinghead_wan22_animate_14b)
  - [text_to_expr_field](#text_to_expr_field)
- [Shared Dependencies](#shared-dependencies)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Acknowledgements](#acknowledgements)

## Overview

The framework consists of three independent but composable modules:

1. **Talking-Head Video Diffusion** (`talkinghead_sd21_unet_cap4d_based/`) — The core rendering pipeline. A latent video diffusion model (SD 2.1 UNet + 3D attention) generates talking-head videos conditioned on FLAME expression maps (spatial addition), audio (wav2vec2 cross-attention), and a reference frame (identity passthrough). Includes expression-weighted loss, ablation configs, and a sliding-window DDIM sampler for long video inference.

2. **Wan2.2-Animate Integration** (`talkinghead_wan22_animate_14b/`) — Exploration of replacing the SD 2.1 UNet with Wan2.2-Animate-14B DiT, combining Wan2.2's I2V (CLIP + VAE reference frame), FaceAdapter (expression conditioning), and S2V (audio injection via AdaIN). Work in progress.

3. **Text-to-Expression Field** (`text_to_expr_field/`) — Removes the driving video dependency by training a generative model (Wan2.2-T2V-A14B, LoRA fine-tuned) to synthesize 45-channel FLAME expression dense fields from text captions. The generated field is a drop-in replacement for the FLAME tracking output consumed by the rendering UNet.

## Repository Structure

```
.
├── talkinghead_sd21_unet_cap4d_based/   # Core talking-head rendering (SD 2.1 UNet)
│   ├── conditioning/                     # FLAME → spatial conditioning maps
│   ├── model/                            # UNet, diffusion, sampler, audio encoder
│   ├── flame/                            # FLAME 3DMM mesh computation
│   ├── data/                             # Dataset, ID lists
│   ├── utils/                            # Background utilities
│   ├── configs/                          # Experiment configs (5 variants)
│   ├── tests/                            # Unit + integration tests
│   ├── train.py                          # Training entry point
│   ├── generate.py                       # Inference entry point
│   └── README.md                         # Full documentation
│
├── talkinghead_wan22_animate_14b/        # Wan2.2-Animate exploration (WIP)
│   ├── wan/                              # Wan2.2 source (cloned, gitignored)
│   ├── model/                            # Combined model wrapper
│   ├── configs/                          # Config files
│   └── README.md
│
├── text_to_expr_field/                   # Text → expression field generation
│   ├── scripts/                          # Preprocessing, caching, training, inference
│   ├── src/                              # Dataset, reshaping utilities
│   ├── configs/                          # Training configs
│   ├── setup_exprmap_env.sh              # Conda env setup
│   └── README.md                         # Full documentation
│
├── controlnet/                           # Stable Diffusion LDM utilities (from CAP4D)
├── data/assets/flame/                    # FLAME model assets (mesh, blendshapes)
├── instructions/                         # Design documents and notes
│   ├── cap4d-summary.md
│   └── text-to-deformation-map-instructions.md
├── requirements.txt                      # Base dependencies
└── README.md                             # This file
```

## Modules

### talkinghead_sd21_unet_cap4d_based

The main talking-head video generation pipeline. Adapts CAP4D's Morphable Multi-View Latent Diffusion Model for temporal video generation with audio conditioning.

**Key features:**
- 3D spatiotemporal attention (SD 2.1 UNet inflated to video)
- 46-channel FLAME expression map conditioning (spatial addition)
- wav2vec2 audio cross-attention in every transformer block
- Expression-weighted diffusion loss
- Stochastic I/O sampling with reference frame passthrough
- 5 experiment configs for ablation studies

See [`talkinghead_sd21_unet_cap4d_based/README.md`](talkinghead_sd21_unet_cap4d_based/README.md) for full documentation.

### talkinghead_wan22_animate_14b

Exploration of using Wan2.2-Animate-14B as the backbone, combining its I2V mechanism (CLIP + VAE reference frame), FaceAdapter (expression conditioning via cross-attention), and S2V audio injection (AdaIN). This module is a work in progress.

See [`talkinghead_wan22_animate_14b/`](talkinghead_wan22_animate_14b/) for current state.

### text_to_expr_field

Generates FLAME expression dense fields from text descriptions using a LoRA-fine-tuned Wan2.2-T2V model. Removes the need for a driving video at inference time.

**Pipeline:**
1. Generate text captions (Whisper ASR + Qwen2-Audio prosody)
2. Build training manifest
3. Cache VAE latents and text embeddings (DDP across GPUs)
4. LoRA fine-tune Wan2.2-T2V on expression field ↔ caption pairs
5. Inference: text prompt → expression dense field → rendering UNet

See [`text_to_expr_field/README.md`](text_to_expr_field/README.md) for full documentation.

## Shared Dependencies

The `controlnet/` directory contains Stable Diffusion LDM utilities inherited from the CAP4D codebase. It provides `LatentDiffusion`, `UNetModel`, `AutoencoderKL`, and related classes used by `talkinghead_sd21_unet_cap4d_based/`.

The `data/assets/flame/` directory contains FLAME 3DMM model files (mesh topology, blendshapes, vertex indices) required for expression map rasterization.

## Installation

```bash
# Create conda environment
conda create -n cap4d_env python=3.10 -y
conda activate cap4d_env

# PyTorch 2.4.1 + CUDA 12.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# PyTorch3D
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html

# Dependencies
pip install -r requirements.txt
```

For the text-to-expression-field pipeline, see [`text_to_expr_field/setup_exprmap_env.sh`](text_to_expr_field/setup_exprmap_env.sh).

## Data Layout

Training data is expected at (gitignored, not included in the repository):

```
data/
├── talkvid/
│   ├── talkvid/{clip_id}.mp4     # Source videos
│   └── audio/{clip_id}.wav       # 16kHz mono audio
├── flowface/
│   └── {clip_id}/
│       ├── fit.npz               # FLAME tracking parameters
│       ├── images/cam0/*.jpg     # Extracted frames
│       └── bg/cam0/*.png         # Foreground masks
├── derived/                       # Generated by preprocessing scripts
│   ├── captions/{clip_id}.json
│   ├── vae_latent_cache/{clip_id}.pt
│   ├── text_embed_cache/{clip_id}.pt
│   └── manifest.json
└── assets/flame/                  # FLAME model files (included in repo)
```

## Acknowledgements

This project builds on:
- [CAP4D](https://github.com/felixtaubner/CAP4D) — Creating Animatable 4D Portrait Avatars (CVPR 2025)
- [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) — Latent diffusion backbone
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) — Video diffusion transformer
- [FLAME](https://flame.is.tue.mpg.de/) — 3D morphable face model
