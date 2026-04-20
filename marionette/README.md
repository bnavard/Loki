# Marionette — Warp-Conditioned Talking-Head Video Diffusion

A latent video diffusion model for identity-preserving talking-head generation.
Given a reference portrait and a driver clip, it produces video of the
reference identity performing the driver's expression and head pose,
synchronized to the driver's audio.

## Core idea

FLAME's topology is identity-agnostic (same V, same face indices for everyone).
Under the reference's camera and shape `β_ref`, applying the driver's
expression `ψ` and head pose `θ` yields a mesh that lives in the reference's
pixel space *and* follows the driver's motion.

Rasterizing that mesh with the reference's per-vertex NDC coordinates as a
property gives a per-pixel UV lookup into the reference image. A `grid_sample`
over that lookup produces a **backward-warped reference** — an identity-locked
"pseudo frame" of the reference doing the driver's performance. The warp is
perfect where the mesh explains the pixels (skin, jaw) and has characteristic
artifacts where it doesn't (eye interior, mouth interior, glasses, hair). The
UNet learns to inpaint the artifact regions.

## Architecture

Operates in SD 2.1's VAE latent space (4 channels, 8× spatial downsampling).
Generates T=16 consecutive frames per forward pass via a UNet whose 2D
self-attention is replaced with 3D spatiotemporal attention. The window is
`[1 ref | 15 generated]`; ref frames pass through unchanged via a
`z_input + ref_mask` bypass, while generated frames attend to the ref via 3D
attention. Loss is masked to non-reference frames.

## Conditioning

| Signal | Mechanism | Shape |
|---|---|---|
| **Spatial conditioning** | Conv encoder (`ConditioningEncoder`, 512→64, zero-init final) → added to first UNet feature map | `(B, T, 512, 512, 49)` |
| **Audio** | Cross-attention context in every transformer block | wav2vec2-base → 1024-dim, ±2-frame context |
| **Reference frame** | `z_input + ref_mask` passthrough | VAE latent (4 ch, 64×64) |

**The 49-channel `spatial_cond` layout:**

- `[0:42]` **pos_enc** — 42ch sinusoidal positional encoding of rasterized FLAME vertex positions.
- `[42:45]` **driver_deform** — 3ch per-vertex expression deformation (rasterized alongside pos_enc in one pass).
- `[45:48]` **warped_ref** — 3ch backward-warped reference image (`grid_sample` on the rasterized UV map).
- `[48]` **ref_mask** — 1ch reference-slot indicator.

`SpatialConditioning.forward` produces all 49 channels on GPU in one rasterization
pass + one `grid_sample`. The null token for CFG is built by zero-filling the
output dict in `MarionetteDiffusion.get_input` — the conditioning module itself has
no CFG logic.

## Training

Strictly same-identity: target, driver, reference — all from the same clip and
window. Frame 0 of the window is the reference (passes through via
`z_input + ref_mask`), frames 1..T-1 are generation targets (`ε-MSE` loss
masked to these slots). The warp in this regime is a same-identity ref→target
pull; artifacts come from the natural mismatch between the reference frame's
expression/pose and each target frame's expression/pose — which is exactly the
artifact distribution cross-identity inference produces.

```bash
conda activate marionette
PYTHONPATH=. python marionette/train.py \
    --config marionette/configs/base.yaml \
    --gpus 0 1 2 3
```

Defaults (`base.yaml`): `gpu_batch_size=4`, `n_frames=16`, 5k steps, SD 2.1
init from `data/models/v2-1_512-ema-pruned.ckpt`. Outputs land under
`outputs/marionette_v3/run_<timestamp>/`.

## Inference

```bash
PYTHONPATH=. python marionette/generate.py \
    --checkpoint  outputs/<run>/<ckpt>.ckpt \
    --config      marionette/configs/base.yaml \
    --ref_clip    <reference_clip_id> \
    --ref_frame   0 \
    --driver_clip <driver_clip_id> \
    --output_dir  outputs/generated/
```

- Same-identity reconstruction: set `--ref_clip == --driver_clip`.
- Cross-identity retargeting: set them to different clips. `generate.py` builds
  per-frame verts from `β_ref + ψ_driver[t] + θ_driver[t]` under the ref's
  camera, keeping the warp a consistent ref→target pull regardless of
  identity.

Both clip IDs must have `fit.npz` at `data/flame_tracking/flowface/{id}/fit.npz`
and a video at `data/talkvid/talkvid/{id}.mp4`. Driver's audio (`data/talkvid/audio/{driver}.wav`)
is used if present.

## Codebase

```
marionette/
├── configs/
│   ├── base.yaml                  # canonical config
│   └── overlays/audio/off.yaml    # disable audio cross-attention
├── config_utils.py                # load_experiment_config: base + overlay merge
├── conditioning/
│   ├── conditioning.py            # SpatialConditioning — 49ch spatial_cond (pos_enc + deform + warp + ref_mask)
│   └── mesh2img.py                # pytorch3d PropRenderer
├── model/
│   ├── diffusion.py               # MarionetteDiffusion — LDM + CFG + ref-masked loss
│   ├── unet.py                    # MarionetteUNet — SD 2.1 UNet + 3D attention + audio x-attn
│   ├── sampler.py                 # SlidingWindowSampler — sliding-window DDIM
│   ├── conditioning_encoder.py    # Conv stack 512 → 64 (zero-init final), additive to first UNet feature map
│   ├── audio_encoder.py           # wav2vec2 → per-frame audio tokens
│   ├── attention.py               # SpatioTemporalTransformer blocks
│   └── utils.py                   # noise schedule helpers
├── flame/
│   ├── flame.py                   # FLAME 3DMM mesh computation
│   └── mouth.py                   # inner mouth vertices
├── data/
│   ├── video_dataset.py           # TalkingHeadDataset (same-identity, n-frame windows)
│   └── types.py                   # SampleDict / HintDict / ControlDict TypedDicts
├── utils/                         # single import surface: `from marionette.utils import ...`
│   ├── audio.py                   # audio window loading (shared dataset + inference)
│   ├── viz.py                     # VisualizationCallback + grid / video helpers
│   ├── video_io.py                # load_frame, FrameReader
│   ├── image_ops.py               # crop_image, rescale_image
│   └── verts.py                   # verts_to_pytorch3d, get_bbox_from_verts, get_square_bbox
├── retargeting.py                 # FLAME retargeting helpers (shared inference + eval)
├── train.py                       # training orchestrator
└── generate.py                    # inference orchestrator (same- or cross-identity)
```
