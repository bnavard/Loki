# Marionette — Identity-Preserving Talking-Head Video Diffusion

A latent video diffusion model for identity-preserving talking-head generation.
Given a reference portrait and a driver clip, the model produces video of the
reference identity performing the driver's expression and head pose,
synchronised to the driver's audio.

Identity is preserved by a **reference UNet** (AnimateAnyone / ReferenceNet
pattern). A frozen SD 2.1 UNet runs once on the VAE-encoded reference frame
and we cache the input to each self-attention block. Those per-layer features
are then injected as additional K/V tokens into the generation UNet's
corresponding self-attention layers, so every generated frame attends to
both its own tokens and the reference's — rich, multi-scale identity
conditioning that does not need a hand-designed warp. Motion is specified
separately by a 45-channel FLAME conditioning map (sinusoidal pos_enc of
rasterized vertex positions + per-vertex expression deformation).

## Repository Structure

```
.
├── marionette/                       # the video diffusion model (training + inference)
│   ├── configs/base.yaml             # canonical config
│   ├── conditioning/conditioning.py  # SpatialConditioning — 45ch (pos_enc + driver_deform)
│   ├── model/{diffusion,unet,ref_unet,conditioning_encoder,audio_encoder,attention}.py
│   ├── data/{video_dataset,types}.py
│   ├── utils/                        # audio / viz / video_io / image_ops / verts
│   ├── flame/                        # FLAME 3DMM
│   ├── retargeting.py                # FLAME retargeting helpers (shared inference + eval)
│   ├── train.py                      # training orchestrator
│   └── generate.py                   # inference orchestrator (same- or cross-identity)
├── experiments/
│   ├── marionette_baseline/          # canonical full-stack training run
│   ├── condition_ablation/           # audio_off / no_flame / no_posenc / no_deform arms
│   ├── marionette_eval/              # cross + same identity eval against a checkpoint
│   ├── sota_comparison/              # SadTalker / AniTalker / EchoMimic / HunyuanPortrait / X-Portrait wrappers
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
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html
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
(CVPR 2025) — we adopt their SD 2.1 UNet + 3D attention scaffold and FLAME
rasterization utilities. The reference-UNet identity pathway follows
[AnimateAnyone](https://arxiv.org/abs/2311.17117) (Hu et al., 2024). The
reorganisation and the combined ReferenceNet + FLAME + audio recipe are
specific to this branch.
