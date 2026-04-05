# Expressive Talking Head

A modular framework for talking-head video generation from audio and facial expression controls. Built on [CAP4D](https://github.com/felixtaubner/CAP4D) (CVPR 2025), the system combines a Stable Diffusion 2.1 UNet (extended with 3D spatiotemporal attention) for video rendering, wav2vec2 audio cross-attention for lip sync, and FLAME 3DMM expression maps for spatial facial control.

A separate text-to-expression-field pipeline trains Wan2.2-T2V (via LoRA) to synthesize FLAME expression dense fields directly from text descriptions, enabling text-only generation without a driving video.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Acknowledgements](#acknowledgements)

## Repository Structure

```
.
├── talkinghead_sd21_unet_cap4d_based/   # Talking-head rendering (SD 2.1 UNet + 3D attention)
├── text_to_expr_field/                  # Text → expression field generation (Wan2.2 DiT + LoRA)
├── controlnet/                          # SD LDM utilities (inherited from CAP4D)
├── data/assets/flame/                   # FLAME model files (mesh, blendshapes)
├── instructions/                        # Design documents
└── README.md
```

## Modules

### talkinghead_sd21_unet_cap4d_based

Talking-head video rendering. Three conditioning signals: FLAME expression maps (46ch spatial addition), wav2vec2 audio (cross-attention), reference frame (identity passthrough). Expression-weighted diffusion loss amplifies gradients on high-deformation face regions. Five experiment configs for ablation studies.

See [`talkinghead_sd21_unet_cap4d_based/README.md`](talkinghead_sd21_unet_cap4d_based/README.md).

### text_to_expr_field

Generates 45-channel FLAME expression dense fields from text captions via LoRA-fine-tuned Wan2.2-T2V-A14B. The 45-channel field is reshaped into 15 three-channel pseudo-video groups for VAE compatibility. VAE latents and UMT5 text embeddings are precomputed and cached to disk for training throughput.

See [`text_to_expr_field/README.md`](text_to_expr_field/README.md).

## Installation

```bash
conda create -n cap4d_env python=3.10 -y
conda activate cap4d_env

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html

pip install -r requirements.txt
```

## Data Layout

```
data/
├── talkvid/talkvid/{clip_id}.mp4      # source videos
├── talkvid/audio/{clip_id}.wav        # 16kHz mono audio
├── flowface/{clip_id}/fit.npz         # FLAME tracking parameters
├── flowface/{clip_id}/images/cam0/    # extracted frames
├── derived/captions/                  # text captions
├── derived/vae_latent_cache/          # precomputed VAE latents
├── derived/prompt_latent_cache/       # precomputed text embeddings
└── assets/flame/                      # FLAME model files (in repo)
```

## Acknowledgements

- [CAP4D](https://github.com/felixtaubner/CAP4D) — Creating Animatable 4D Portrait Avatars (CVPR 2025)
- [Stable Diffusion 2.1](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) — Video diffusion transformer
- [FLAME](https://flame.is.tue.mpg.de/) — 3D morphable face model
