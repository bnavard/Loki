# Loki — Identity-Preserving Talking-Head Video Diffusion

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
├── loki/                       # the video diffusion model (training + inference)
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
│   ├── loki_train/          # canonical full-stack training run
│   ├── condition_ablation/           # audio_off / no_flame / no_posenc / no_deform arms
│   ├── loki_eval/              # cross + same identity eval against a checkpoint
│   ├── sota_comparison/              # SadTalker / AniTalker / EchoMimic / HunyuanPortrait / X-Portrait wrappers
│   └── evaluation_metrics/           # SyncNet lip-sync evaluation
├── scripts/
│   ├── download/                     # YouTube clip scraping (yt-dlp)
│   ├── preprocess/                   # face crop + 512×512 / 25fps resample
│   ├── manifest/                     # build_manifest + identity-based train/val split
│   └── paper_viz/                    # paper figure rendering (comparison, teaser, driver-map, FLAME substitution)
├── generate_exp_map/                 # FLAME tracking (pixel3dmm) — upstream, produces fit.npz
├── ldm_base/                         # vendored SD 2.1 LDM utilities
├── data/                             # data / models / derived (mostly .gitignored symlinks)
└── outputs/                          # training runs (gitignored)
```

See [loki/README.md](loki/README.md) for architecture details and per-module documentation.

## Installation

Tested on Linux + CUDA 12.1 with a single conda env `loki`:

```bash
conda create -n loki python=3.10 -y
conda activate loki
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html
```

Heavy dependencies worth flagging:
- **pytorch3d** — used by `SpatialConditioning` for GPU mesh rasterization +
  UV lookup. Must be built against the torch version you install.
- **decord** — video reading. Lazy-imported in `loki/utils/video_io.py`
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

## Training

Same-identity self-supervised: each sample's reference frame, target window, and driver motion all come from the same clip — the reference is sampled from a different position in the clip than the target window, so the model learns an identity prior that generalises across expression/pose mismatch (which is what cross-identity inference presents at test time). Loss is uniform ε-MSE over the T target slots; the reference lives in the frozen ref UNet, not in the output tensor.

```bash
conda activate loki

# Single GPU
PYTHONPATH=. python experiments/loki_train/run.py

# Multi-GPU (DDP)
PYTHONPATH=. python experiments/loki_train/run.py --gpus 0 1 2 3

# Resume from a checkpoint
PYTHONPATH=. python experiments/loki_train/run.py \
    --resume outputs/loki_train/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt
```

Defaults from [loki/configs/base.yaml](loki/configs/base.yaml): `gpu_batch_size=2`, `virtual_batch_size=2`, `n_frames=16`, `n_steps=30000`, `val_every_n_steps=3000`, `save_every_n_steps=10000`, SD 2.1 init from `data/models/v2-1_512-ema-pruned.ckpt`.

Each run writes to `outputs/loki_train/run_<timestamp>/` with:

```
outputs/loki_train/run_<timestamp>/
├── config_resolved.yaml         # snapshot at run start
├── log.txt                      # mirrored stdout/stderr (rank 0)
├── checkpoints/
│   ├── th-<step>.ckpt           # every save_every_n_steps (periodic)
│   └── th-best-<step>-<val>.ckpt  # top-1 by val/loss
├── logs/                        # TensorBoard
└── visualizations/
    └── step_<step>/
        ├── sample_NN.png        # 4-row grid (Reference | GT | <cond preview> | Generated)
        └── sample_NN.mp4        # same rows, silent
```

For ablation arms (FLAME conditioning variants), see [experiments/condition_ablation/](experiments/condition_ablation/).

## Inference

```bash
PYTHONPATH=. python loki/generate.py \
    --checkpoint  outputs/loki_train/run_<ts>/checkpoints/<ckpt>.ckpt \
    --config      loki/configs/base.yaml \
    --ref_clip    <reference_clip_id> \
    --ref_frame   0 \
    --driver_clip <driver_clip_id> \
    --output_dir  outputs/generated/
```

- **Same-identity reconstruction**: `--ref_clip == --driver_clip`.
- **Cross-identity retargeting**: different clips. `generate.py` builds per-frame verts from `β_ref + ψ_driver[t] + θ_driver[t]` under the ref's camera, so the FLAME conditioning carries the driver's motion in the reference's shape and camera frame regardless of identity. Identity flows through the ref UNet; motion flows through `spatial_cond`.

Both clip IDs must have:

- `data/flame_tracking/flowface/{clip_id}/fit.npz` — FLAME params
- `data/talkvid/talkvid/{clip_id}.mp4` — face-cropped 512×512 / 25fps video


For benchmark-scale evaluation against checkpoints (cross + same-identity protocols, panel.mp4 outputs), see [experiments/loki_eval/](experiments/loki_eval/) and [experiments/evaluation_metrics/](experiments/evaluation_metrics/).

## Acknowledgements

This codebase builds on:

- **[CAP4D](https://github.com/felixtaubner/cap4d/)** (Taubner et al., CVPR 2025) — SD 2.1 UNet + 3D attention scaffold, FLAME rasterization utilities, and the FLAME skinner we extend in [loki/flame/flame.py](loki/flame/flame.py).
- **[pixel3dmm](https://github.com/SimonGiebenhain/pixel3dmm)** (Giebenhain et al.) — vendored under [generate_exp_map/pixel3dmm/](generate_exp_map/pixel3dmm/) for FLAME tracking on raw videos.
- **[FLAME](https://flame.is.tue.mpg.de/)** (Li et al., 2017) — the underlying 3DMM head model.
- **[AnimateAnyone](https://arxiv.org/abs/2311.17117)** (Hu et al., 2024) — the reference-UNet identity pathway pattern.

The reorganisation and the combined ReferenceNet + FLAME + audio recipe are specific to this branch.
