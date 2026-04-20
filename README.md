# Marionette — Warp-Conditioned Talking-Head Video Diffusion

A latent video diffusion model for identity-preserving talking-head generation.
Given a reference portrait and a driver clip, the model produces video of the
reference identity performing the driver's expression and head pose,
synchronised to the driver's audio.

The core signal is a **backward-warped reference**: under FLAME's
identity-agnostic topology, we drive the reference's mesh with the driver's
expression + head pose, rasterize the reference's per-vertex NDC coordinates
through that mesh, and `grid_sample` the reference image over the resulting
UV map. The warp is identity-locked where the mesh explains the pixels (skin,
jaw) and has characteristic artifacts where it doesn't (eye interior, mouth
interior, hair) — the UNet learns to inpaint those artifacts while keeping
the warp's identity prior intact.

## Repository Structure

```
.
├── marionette/                       # the video diffusion model (training + inference)
│   ├── configs/base.yaml             # canonical config
│   ├── conditioning/conditioning.py  # SpatialConditioning — 49ch (pos_enc + deform + warp + ref_mask)
│   ├── model/{diffusion,unet,sampler,conditioning_encoder,audio_encoder,attention}.py
│   ├── data/{video_dataset,types}.py
│   ├── utils/                        # audio / viz / video_io / image_ops / verts
│   ├── flame/                        # FLAME 3DMM
│   ├── retargeting.py                # FLAME retargeting helpers (shared inference + eval)
│   ├── train.py                      # training orchestrator
│   └── generate.py                   # inference orchestrator (same- or cross-identity)
├── experiments/
│   └── evaluation_metrics/           # SyncNet lip-sync evaluation
├── scripts/                          # data-prep pipeline: download → preprocess → manifest
├── generate_exp_map/                 # FLAME tracking (pixel3dmm) — upstream, produces fit.npz
├── ldm_base/                         # vendored SD 2.1 LDM utilities
├── data/                             # data / models / derived (mostly .gitignored symlinks)
├── instructions/                     # design docs + paper context
└── outputs/                          # training runs (gitignored)
```

See [marionette/README.md](marionette/README.md) for architecture details,
training, and inference usage.

## Installation

Tested on Linux + CUDA 12.1 with a single conda env `marionette`:

```bash
conda create -n marionette python=3.10 -y
conda activate marionette
pip install -r requirements.txt
```

Heavy dependencies worth flagging:
- **pytorch3d** — used by `SpatialConditioning` for GPU mesh rasterization +
  UV lookup. Must be built against the torch version you install.
- **decord** — video reading. Lazy-imported in `marionette/utils/video_io.py`
  to avoid CUDA-init ordering issues.

## Data Layout

Training reads from:

```
data/
├── flame_tracking/flowface/{clip_id}/fit.npz     # per-frame FLAME params
├── talkvid/talkvid/{clip_id}.mp4                 # face-cropped 512×512 @ 25fps
├── talkvid/audio/{clip_id}.wav                   # 16 kHz mono driver audio
├── derived/
│   ├── manifest.json                             # output of scripts/manifest/build_manifest.py
│   ├── train_clips.json
│   └── val_clips.json                            # output of scripts/manifest/partition_dataset.py
├── models/v2-1_512-ema-pruned.ckpt               # SD 2.1 init
└── assets/flame/                                 # FLAME model files (mesh + blendshapes)
```

To produce `fit.npz` from raw videos, see [generate_exp_map/README.md](generate_exp_map/README.md)
(pixel3dmm-based pipeline, separate env).

## Acknowledgements

This codebase inherits from [CAP4D](https://github.com/felixtaubner/cap4d)
(CVPR 2025) — we adopt their SD 2.1 UNet + 3D attention scaffold, FLAME
rasterization utilities, and sliding-window DDIM sampler. The warp-conditioning
idea and the v3 reorganisation are specific to this branch.
