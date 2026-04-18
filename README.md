# Expressive Talking Head

A modular framework for talking-head video generation from audio and facial expression controls. Combines a Stable Diffusion 2.1 UNet (extended with 3D spatiotemporal attention) for video rendering, wav2vec2 audio cross-attention for lip sync, and FLAME 3DMM expression maps for spatial facial control.

The two training pipelines are complementary:
- `marigold_training/` trains a per-frame generator that produces FLAME deformation maps from natural face frames (Marigold-style, SD3.5 Medium).
- `marionette/` trains the video diffusion model that consumes those expression maps (generated or ground-truth) along with audio and a reference frame to render the final talking-head video.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Acknowledgements](#acknowledgements)

## Repository Structure

```
.
├── marionette/                          # Talking-head video rendering (SD 2.1 UNet + 3D attention)
├── marigold_training/                   # Marigold-style face → deformation map (SD3.5 DiT)
├── experiments/                         # Ablation study organisation (launchers, configs, eval metrics)
├── scripts/                             # Full data-prep pipeline: download → preprocess → caption → cache → manifest
├── generate_exp_map/                    # FLAME tracking pipeline (pixel3dmm → fit.npz)
├── ldm_base/                            # Vendored Stable Diffusion LDM utilities
├── data/assets/flame/                   # FLAME model files (mesh, blendshapes)
├── data/models/                         # SD 2.1 init checkpoint (gitignored)
├── instructions/                        # Design documents + paper context
└── README.md
```

## Modules

### marionette

Talking-head video rendering. Four conditioning signals (each independently toggleable): FLAME expression map (46ch / 4ch / 1ch configurable), wav2vec2 audio cross-attention, 6DRepNet head pose embedding added to the timestep, and reference-frame identity passthrough. Expression-weighted diffusion loss amplifies gradients on high-deformation face regions. Configs are composed from a single `base.yaml` plus overlays — no duplicated concrete configs. Evaluation uses cross-identity driving by default (target's reference frame, driver's expression + audio) so models are judged on following external cues rather than trivial reconstruction.

See [`marionette/README.md`](marionette/README.md).

### marigold_training

Marigold-style deformation map generation. Given a natural face frame, generates the corresponding FLAME deformation map. Adapts the Marigold depth estimation approach (Ke et al., CVPR 2024) using SD3.5 Medium: the DiT's input layer is doubled from 16 to 32 channels to accept concatenated [noisy_target | clean_conditioning] latents. Full fine-tuning with null text conditioning.

See [`marigold_training/README.md`](marigold_training/README.md).

### experiments

Ablation study organisation. Each sub-folder targets a single axis of variation and holds thin experiment configs (base + overlay references) plus a `run.py` that calls `marionette.train.run_training()` directly. Current studies: `ablate_expr_source/` (gt_full / gt_baseline / marigold / driving_video), `ablate_audio/` (audio cross-attention on/off), `ablate_conditioning/` (channel budget matrix), `ablate_loss_weighting/` (uniform vs expression-weighted loss). Quantitative metrics (lip sync via SyncNet) live in `evaluation_metrics/`.

See [`experiments/README.md`](experiments/README.md).

### scripts

Full data-preparation pipeline, organised by stage: `download/` (YouTube clip scraping), `preprocess/` (face crop + resample), `caption/` (Whisper + Qwen2-Audio prosody), `cache/` (expression field + Marigold deformation tensors), `manifest/` (train/val split with identity-based 98/2 split to prevent identity leakage), `tools/` (devops utilities).

See [`scripts/README.md`](scripts/README.md).

## Installation

### `marionette` — main training and inference environment

Used by `marionette`, `marigold_training`, and `scripts`.

```bash
conda create -n marionette python=3.10 -y
conda activate marionette

# PyTorch 2.4.1 + CUDA 12.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# PyTorch3D 0.7.8 (prebuilt wheel for Python 3.10, CUDA 12.1, PyTorch 2.4.1)
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html

# Diffusion and training
pip install diffusers==0.33.0 transformers peft accelerate safetensors

# General
pip install einops opencv-python scipy matplotlib pyyaml sentencepiece tqdm
```

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10 | |
| PyTorch | 2.4.1+cu121 | GPU compute |
| PyTorch3D | 0.7.8 | FLAME mesh rasterization (expression field computation) |
| diffusers | 0.33.0 | SD3.5 diffusion pipeline |
| transformers | 4.49.0 | UMT5 text encoder, CLIP, Whisper |
| peft | 0.18.1 | LoRA fine-tuning |
| accelerate | 1.13.0 | Multi-GPU training (DDP) |
| safetensors | 0.7.0 | Model weight loading |
| einops | 0.8.2 | Tensor reshaping |
| sentencepiece | 0.2.1 | T5 tokenizer |

### `expmapgen` — FLAME tracking environment

Used by `generate_exp_map` for computing `fit.npz` from videos via [pixel3dmm](https://github.com/SimonGiebenhain/pixel3dmm). See [`generate_exp_map/README.md`](generate_exp_map/README.md) for full setup instructions.

| Package | Version | Purpose |
|---|---|---|
| Python | 3.9 | pixel3dmm compatibility |
| PyTorch | 2.0.1+cu118 | GPU compute |
| PyTorch3D | 0.7.4 | Differentiable mesh rendering |
| nvdiffrast | latest | Differentiable rasterization |
| insightface | 0.7.3 | Face detection (MICA) |
| facer | latest | Face semantic segmentation |

## Data Layout

```
data/
├── assets/flame/                              # FLAME model files (tracked in repo)
├── weights/{l2cs,mmdm,rvm,syncnet}            # pretrained auxiliary weights (tracked except *.pth)
├── models/v2-1_512-ema-pruned.ckpt            # SD 2.1 init checkpoint (gitignored)
├── talkvid/talkvid/{clip_id}.mp4              # source videos (symlink, gitignored)
├── talkvid/audio/{clip_id}.wav                # 16 kHz mono audio (symlink, gitignored)
├── flame_tracking/
│   ├── flowface/{clip_id}/fit.npz             # FLAME tracking parameters (fit.npz, images, bg)
│   ├── preprocessing/{clip_id}/p3dmm/
│   │   ├── normals/*.png                      # pixel3dmm predicted surface normals
│   │   └── uv_map/*.png                       # pixel3dmm UV coordinate maps
│   └── tracking/{clip_id}_nV1_noPho_uv2000.0_n1000.0/
│       ├── checkpoint/*.frame                 # per-frame tracking states
│       ├── mesh/*.ply, flowface_mesh/*.ply    # fitted meshes
│       └── result.mp4                         # overlay visualization
└── derived/
    ├── captions/{clip_id}.json                # Whisper + Qwen2-Audio prosody
    ├── expression_field/{clip_id}/            # GT rasterized 45-channel expression fields
    ├── marigold_deform/{clip_id}/             # Marigold-generated deformation mp4s
    ├── manifest.json                          # clip_id + paths + frame counts
    ├── train_clips.json, val_clips.json       # identity-based split (98/2)
    └── (run-time filtered clip lists under outputs/<experiment>/filtered_clips/)
```

## Acknowledgements

- [CAP4D](https://github.com/felixtaubner/CAP4D) — Creating Animatable 4D Portrait Avatars (CVPR 2025)
- [Stable Diffusion 2.1](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2)
- [Marigold](https://github.com/prs-eth/Marigold) — Repurposing diffusion for monocular depth (CVPR 2024)
- [FLAME](https://flame.is.tue.mpg.de/) — 3D morphable face model
