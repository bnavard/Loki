# Loki — Identity-Preserving Talking-Head Video Diffusion

A latent video diffusion model for identity-preserving talking-head generation.
Given a reference portrait and a driver clip, it produces video of the
reference identity performing the driver's expression and head pose,
synchronized to the driver's audio.

## Core idea

Identity is preserved via a **reference UNet** (the AnimateAnyone /
ReferenceNet pattern). A frozen copy of SD 2.1's UNet runs once on the
VAE-encoded reference frame; forward hooks on each `BasicTransformerBlock`'s
pre-attention LayerNorm capture the input to every self-attention block.
Those `(B, HW_k, D_k)` per-layer features are then injected as additional
K/V tokens into the matching self-attention block of the generation UNet,
so every gen-frame query attends to both its own tokens and the reference's
at the same resolution.

Motion is specified separately — 45-channel FLAME spatial conditioning
(sinusoidal positional encoding of rasterized vertex positions +
per-vertex expression deformation) is consumed by a small conv encoder and
added once to the first gen-UNet feature map.

## Architecture

Operates in SD 2.1's VAE latent space (4 channels, 8× spatial downsampling).
Generates T consecutive frames per forward pass (T=16 default) via a UNet
whose 2D self-attention is replaced with 3D spatiotemporal attention for the
inner stages and whose per-block self-attention is extended to accept ref
K/V tokens.

The reference does **not** occupy a slot in the gen tensor. It lives only in
the frozen reference UNet — the gen tensor is `(B, T, 4, h, w)` with T pure
target frames, and identity flows exclusively through the ref-attention
injection. Loss is a uniform ε-MSE across all T gen slots (no masking).

## Conditioning

| Signal | Mechanism | Shape |
|---|---|---|
| **Reference identity** | Frozen SD 2.1 UNet → per-layer self-attn K/V injection into gen UNet | `(B, 4, h, w)` ref latent → list of `(B, HW_k, D_k)` features |
| **Spatial conditioning** | Conv encoder (`ConditioningEncoder`, 512→64, zero-init final) → added to first UNet feature map | `(B, T, 512, 512, 45)` |
| **Audio** | Cross-attention context in every transformer block | wav2vec2-base → 1024-dim, ±2-frame context |

**The 45-channel `spatial_cond` layout:**

- `[0:42]` **pos_enc** — 42ch sinusoidal positional encoding of rasterized FLAME vertex positions (target-camera NDC).
- `[42:45]` **driver_deform** — 3ch per-vertex expression deformation (rasterized alongside pos_enc in one pass).

`SpatialConditioning.forward` produces all 45 channels on GPU in one
rasterization pass. The null token for CFG is built by zero-filling the
output dict in `LokiDiffusion.get_input` — the conditioning module
itself has no CFG logic.

## Training

Strictly same-identity self-supervised: one clip supplies the reference, the
target window, and the driver motion. Each sample carries `T+1` frames —
slot 0 is a reference frame drawn from an independently sampled position
in the clip (seeded separately from the target window), and slots 1..T are
the target window. VAE-encode all T+1, feed slot 0 to the ref UNet, and
compute ε-MSE loss uniformly over slots 1..T.

Because the reference is sampled from a different position in the same clip
than the target window, the model is forced to learn an identity prior that
generalises across expression and pose mismatch — which is exactly the
distribution cross-identity inference presents at test time.

```bash
conda activate loki

# Recommended: through the experiment runner (loads base + overlays, sets output_dir)
PYTHONPATH=. python experiments/loki_baseline/run.py --gpus 0 1 2 3

# Ad-hoc: directly against the base config
PYTHONPATH=. python loki/train.py \
    --config loki/configs/base.yaml \
    --output_dir outputs/ad_hoc \
    --gpus 0 1 2 3
```

Defaults (`base.yaml`): `gpu_batch_size=2`, `virtual_batch_size=2`,
`n_frames=16`, `n_steps=30000`, `val_every_n_steps=3000`,
`save_every_n_steps=10000`, SD 2.1 init from
`data/models/v2-1_512-ema-pruned.ckpt`. The experiment runner writes to
`outputs/<experiment>/run_<timestamp>/`.

## Inference

```bash
PYTHONPATH=. python loki/generate.py \
    --checkpoint  outputs/<run>/<ckpt>.ckpt \
    --config      loki/configs/base.yaml \
    --ref_clip    <reference_clip_id> \
    --ref_frame   0 \
    --driver_clip <driver_clip_id> \
    --output_dir  outputs/generated/
```

- Same-identity reconstruction: set `--ref_clip == --driver_clip`.
- Cross-identity retargeting: set them to different clips. `generate.py` builds
  per-frame verts from `β_ref + ψ_driver[t] + θ_driver[t]` under the ref's
  camera, so the FLAME conditioning carries the driver's motion in the
  reference's shape and camera frame regardless of identity. Identity flows
  through the ref UNet; motion flows through spatial_cond.

Both clip IDs must have `fit.npz` at `data/flame_tracking/flowface/{id}/fit.npz`
and a video at `data/talkvid/talkvid/{id}.mp4`. Driver's audio (`data/talkvid/audio/{driver}.wav`)
is used if present.

## Codebase

```
loki/
├── configs/
│   ├── base.yaml                  # canonical config
│   └── overlays/audio/off.yaml    # disable audio cross-attention
├── config_utils.py                # load_experiment_config: base + overlay merge
├── conditioning/
│   ├── conditioning.py            # SpatialConditioning — 45ch spatial_cond (pos_enc + driver_deform)
│   └── mesh2img.py                # pytorch3d PropRenderer
├── model/
│   ├── diffusion.py               # LokiDiffusion — LDM + CFG + sample_video DDIM
│   ├── unet.py                    # LokiUNet — SD 2.1 UNet + 3D attention + audio x-attn + ref-K/V injection
│   ├── ref_unet.py                # RefFeatureExtractor — frozen SD 2.1 UNet + per-layer self-attn hooks + null-prompt buffer
│   ├── conditioning_encoder.py    # Conv stack 512 → 64 (zero-init final), additive to first UNet feature map
│   ├── audio_encoder.py           # wav2vec2 → per-frame audio tokens
│   ├── attention.py               # SpatioTemporalTransformer + AttentionModule (accepts ref K/V)
│   └── utils.py                   # noise schedule helpers
├── flame/
│   ├── flame.py                   # FLAME 3DMM mesh computation
│   └── mouth.py                   # inner mouth vertices
├── data/
│   └── video_dataset.py           # TalkingHeadDataset (same-identity, slot-0 ref + T target frames)
├── utils/                         # single import surface: `from loki.utils import ...`
│   ├── audio.py                   # audio window loading (shared dataset + inference)
│   ├── viz.py                     # VisualizationCallback (rank-sharded) + grid / video helpers
│   ├── video_io.py                # load_frame, FrameReader
│   ├── image_ops.py               # crop_image, rescale_image
│   ├── verts.py                   # verts_to_pytorch3d, get_bbox_from_verts, get_square_bbox
│   └── log_tee.py                 # install_log_tee → mirrors stdout/stderr to run_<ts>/log.txt
├── retargeting.py                 # FLAME retargeting helpers (shared inference + eval)
├── train.py                       # training orchestrator
└── generate.py                    # inference orchestrator (same- or cross-identity)
```

