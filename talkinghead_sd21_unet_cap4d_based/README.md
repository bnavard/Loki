# Talking-Head Video Diffusion

A video diffusion model for talking-head generation, built on top of the [CAP4D](https://github.com/felixtaubner/CAP4D) architecture (CVPR 2025). Given a reference portrait image, a driving video (providing facial expressions via FLAME tracking), and an audio track, the model generates temporally coherent talking-head video. The system reuses CAP4D's latent diffusion backbone — a Stable Diffusion 2.1 UNet extended with 3D spatiotemporal attention — and adds audio cross-attention conditioning via a wav2vec2 encoder. Expression control comes from FLAME 3DMM parameters rasterized into dense spatial conditioning maps, following the same mesh-to-image pipeline used in CAP4D's Morphable Multi-View Latent Diffusion Model (MMDM).

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Architecture Overview](#architecture-overview)
  - [Conditioning Pipeline](#conditioning-pipeline)
  - [Audio Encoding](#audio-encoding)
  - [UNet and Diffusion Model](#unet-and-diffusion-model)
  - [Stochastic I/O Sampling](#stochastic-io-sampling)
  - [Expression-Weighted Loss](#expression-weighted-loss)
- [Training](#training)
- [Inference](#inference)
- [Ablation Experiments](#ablation-experiments)
- [Codebase Structure](#codebase-structure)
- [Testing](#testing)

## Installation

The codebase requires Python 3.10, PyTorch 2.4.1 with CUDA 12.1, and PyTorch3D 0.7.8.

```bash
# Create and activate conda environment
conda create -n cap4d_env python=3.10 -y
conda activate cap4d_env

# PyTorch 2.4.1 + CUDA 12.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# PyTorch3D (prebuilt wheel for py3.10 / CUDA 12.1 / PyTorch 2.4.1)
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html

# Remaining dependencies
pip install -r requirements.txt
```

**requirements.txt** (key packages):

| Package | Version |
|---|---|
| torch | 2.4.1+cu121 |
| pytorch3d | 0.7.8 |
| xformers | 0.0.28.post1 |
| transformers | 5.3.0 |
| pytorch-lightning | 2.6.1 |
| einops | 0.8.2 |
| omegaconf | 2.3.0 |
| opencv-python | 4.13.0.92 |
| soundfile | 0.13.1 |
| scipy | 1.13.1 |
| tensorboard | 2.20.0 |
| numpy | 2.2.6 |

You also need the Stable Diffusion 2.1 checkpoint for weight initialization:

```bash
mkdir -p models
# Download v2-1_512-ema-pruned.ckpt to models/
```

## Data Preparation

The training data is organized across three directories, linked by a shared clip ID:

```
video_root / {id}.mp4           # Source video
audio_root / {id}.wav           # 16 kHz mono audio
flame_root / {id} / fit.npz     # FLAME tracking output
```

Each `fit.npz` contains the following keys from FLAME tracking:

| Key | Shape | Description |
|---|---|---|
| `shape` | `(150,)` | Identity shape parameters (shared across frames) |
| `expr` | `(N, 65)` | Per-frame expression blend shape weights |
| `rot` | `(N, 3)` | Per-frame head rotation (axis-angle) |
| `tra` | `(N, 3)` | Per-frame head translation |
| `eye_rot` | `(N, 3)` | Per-frame eye rotation (axis-angle) |
| `fx, fy, cx, cy` | `(1, 1)` | Camera intrinsics |
| `extr` | `(1, 4, 4)` | Camera extrinsics |

A plain text file lists which clip IDs to use, one per line. Split this into train and validation sets:

```
talkinghead_sd21_unet_cap4d_based/data/train_ids.txt
talkinghead_sd21_unet_cap4d_based/data/val_ids.txt
```

Update the paths in `talkinghead_sd21_unet_cap4d_based/configs/talking_head.yaml` to point to your data.

## Architecture Overview

The model is a latent video diffusion model that operates in Stable Diffusion's VAE latent space (4 channels, 8x spatial downsampling). It generates T=16 consecutive video frames per forward pass. Two conditioning signals guide generation: FLAME expression maps provide explicit spatial control over facial geometry, while audio features provide speech content through cross-attention.

### Conditioning Pipeline

**`talkinghead_sd21_unet_cap4d_based/conditioning/th_conditioning.py`** converts raw FLAME mesh parameters into a dense spatial conditioning tensor of shape `(B, T, H, W, 46)`:

- **42 channels**: Sinusoidal Fourier positional encoding of the 3D vertex positions. Each of the 3 spatial coordinates (x, y, z) is expanded into 14 channels (7 frequency bands x sin/cos), capturing both coarse face structure and fine spatial detail.
- **3 channels**: Expression deformation map — per-vertex displacement from the neutral face to the current expression, rasterized onto the image grid.
- **1 channel**: Reference mask — binary flag indicating which frames are identity references (1) vs. frames to generate (0).

The rasterization step is handled by **`talkinghead_sd21_unet_cap4d_based/conditioning/mesh2img.py`**, which uses PyTorch3D to project the FLAME mesh triangles onto a 2D grid and interpolate per-vertex attributes via barycentric coordinates. This bridges the gap between FLAME's sparse 3D mesh vertices and the 2D spatial grid the UNet operates on.

The 46-channel conditioning tensor is projected to 320 channels via a learned linear layer and added to the UNet's first feature map (spatial addition, not concatenation).

Ray directions can optionally be enabled (+3 channels) for camera-aware generation, but are off by default.

### Audio Encoding

**`talkinghead_sd21_unet_cap4d_based/model/audio_encoder.py`** encodes raw 16 kHz waveform windows into per-frame context tokens:

1. A wav2vec2-base backbone (frozen during training) processes each frame's audio window through 7 CNN layers and 12 transformer layers, producing a sequence of 768-dimensional tokens.
2. A learned linear projection maps the backbone output to the UNet's context dimension.
3. Output shape: `(B, T, num_tokens, 768)` — each frame gets its own set of audio tokens.

These tokens are passed as the `context` argument to every SpatioTemporalTransformer block in the UNet, where they serve as keys and values in cross-attention. This allows each spatial position in the generated frame to attend to the relevant audio content — for example, lip pixels attend to phoneme tokens to determine mouth shape.

### UNet and Diffusion Model

**`talkinghead_sd21_unet_cap4d_based/model/th_unet.py`** is adapted from CAP4D's MMDMUnetModel. The key modifications:

- **Cross-attention enabled**: The original CAP4D UNet had `use_context = False` and asserted `context == None`. Both are changed to allow audio tokens to flow through every transformer block's cross-attention path.
- **Reference frame passthrough**: For frames marked as references (ref_mask=1), the UNet outputs `x - z_input` directly — the known noise residual — bypassing the learned prediction. This lets reference frames participate in 3D attention (providing identity features to generated frames) without wasting model capacity.

**`talkinghead_sd21_unet_cap4d_based/model/th_diffusion.py`** wraps the UNet in a latent diffusion training loop:

- Images are encoded to latents via the frozen SD 2.1 VAE.
- FLAME conditioning is computed and injected spatially.
- Audio is encoded and injected via cross-attention.
- Classifier-free guidance (CFG) training randomly drops conditioning with configurable probability.
- Loss is masked to non-reference frames using `torch.logical_not(ref_mask)`.

### Stochastic I/O Sampling

During both training and inference, reference and generated frames are concatenated along the time axis and processed together by the UNet: `[1 ref | T-1 generated]`. The 3D spatiotemporal attention operates across all T frames, allowing generated frames to directly attend to the reference frame's features for identity preservation (face shape, skin texture, lighting).

At inference time, **`talkinghead_sd21_unet_cap4d_based/model/th_sampler.py`** implements a sliding-window DDIM sampler that processes long videos in overlapping chunks of V frames (1 reference + V-1 generated), stepping through the full sequence.

### Expression-Weighted Loss

The standard diffusion loss treats every pixel equally — a pixel of static background contributes the same gradient as a pixel of a rapidly-moving lip. But the expression deformation map from FLAME tells us exactly which regions are actively deforming between frames.

We exploit this by weighting the per-pixel diffusion loss proportionally to the expression deformation magnitude:

```
expr_magnitude = expression_deformation_map.norm(dim=channel)  # (B, T, H, W)
expr_weight = 1.0 + alpha * normalize(expr_magnitude)          # higher where face moves
loss = weighted_mean(MSE(noise_pred, noise) * expr_weight)
```

Pixels where the mouth is opening, eyebrows are raising, or the jaw is moving receive up to `(1 + alpha)` times the base loss weight. Static regions (forehead, ears, background) receive the baseline weight of 1.0 — they're never suppressed, only the active regions are amplified.

The deformation magnitude comes directly from channels 42:45 of the conditioning tensor (`pos_enc`), which encode per-vertex displacement from the neutral face. These are already computed by THConditioning at the latent resolution (64×64), so no extra rasterization is needed.

Configure via:
```yaml
# In model params
expr_weight_alpha: 5.0    # 0 = uniform (disabled), 5 = active regions get up to 6x weight
```


## Training

```bash
conda activate cap4d_env
export PYTHONPATH=$(realpath "./"):$PYTHONPATH

python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/talking_head.yaml \
    --gpus 0
```

Multi-GPU training:

```bash
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/talking_head.yaml \
    --gpus 0 1 2 3
```

Resume from checkpoint:

```bash
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/talking_head.yaml \
    --gpus 0 \
    --resume outputs/talkinghead/th-step=010000.ckpt
```

Key training parameters (in `talking_head.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | 1e-4 | AdamW learning rate |
| `gpu_batch_size` | 8 | Batch size per GPU |
| `virtual_batch_size` | 8 | Set equal to gpu_batch_size to disable gradient accumulation |
| `n_steps` | 200,000 | Total training steps |
| `n_frames` | 16 | Video frames per training sample |
| `val_every_n_steps` | 250 | Validation and visualization frequency |
| `save_every_n_steps` | 250 | Checkpoint save frequency |
| `expr_weight_alpha` | 5.0 | Expression-weighted loss amplification (0 = uniform) |

Each run creates a timestamped directory under the output path (e.g. `outputs/talkinghead/run_20260327_143022/`) containing checkpoints, visualizations, TensorBoard logs, and a copy of the config file used.

Visualizations are saved as both image grids (`sample_XX.png`) with three labeled rows (Ground Truth / Expression Map / Generated) and as videos (`sample_XX.mp4`) with embedded audio.

Monitor training via TensorBoard:

```bash
tensorboard --logdir outputs/talkinghead/run_YYYYMMDD_HHMMSS/logs
```

## Inference

Inference follows CAP4D's reference/driving split: identity (shape, camera, appearance) comes from a reference subject, while facial expressions come from a separate driving video.

```bash
python talkinghead_sd21_unet_cap4d_based/generate.py \
    --checkpoint  outputs/talkinghead/run_YYYYMMDD/th-best.ckpt \
    --config      talkinghead_sd21_unet_cap4d_based/configs/talking_head.yaml \
    --ref_data    /path/to/reference_subject/ \
    --ref_frame   0 \
    --driving_fit /path/to/driving/fit.npz \
    --audio       /path/to/audio.wav \
    --output_dir  outputs/generated/ \
    --bg_mask_dir data/flowface/{clip_id}/bg/cam0
```

| Argument | Description |
|---|---|
| `--ref_data` | Directory containing the reference subject's `fit.npz` and `images/` |
| `--ref_frame` | Which frame to use as the identity reference (default: 0) |
| `--driving_fit` | `fit.npz` from the driving video — only `expr` and `eye_rot` are used |
| `--audio` | Driving audio file (16 kHz WAV) |
| `--n_frames` | Total frames to generate (default: 64) |
| `--cfg_scale` | Classifier-free guidance scale (default: 2.0) |
| `--n_ddim_steps` | DDIM denoising steps (default: 50) |
| `--bg_mask_dir` | Path to per-frame foreground masks for background stabilization (optional) |
| `--feather_radius` | Gaussian blur radius for soft mask edges (default: 5, 0 = hard) |

The output is saved as individual PNG frames in `{output_dir}/frames/`. When `--bg_mask_dir` is provided, each generated frame is composited with a clean background plate built from the reference video, ensuring a stable, jitter-free background.

## Experiments

Five configurations explore different combinations of spatial conditioning and loss weighting. All experiments share the same audio cross-attention and reference frame conditioning — only the spatial conditioning input to the UNet and the loss weighting strategy vary.

| # | Config | UNet spatial conditioning | Loss weighting | Purpose |
|---|---|---|---|---|
| 1 | `full_cond_weighted_loss.yaml` | 46ch (42 pos enc + 3 deform + 1 ref mask) | Weighted (`alpha=5.0`) | Full model — all signals |
| 2 | `full_cond_uniform_loss.yaml` | 46ch (42 pos enc + 3 deform + 1 ref mask) | Uniform (`alpha=0.0`) | Baseline to measure effect of expression-weighted loss |
| 3 | `deform_only_weighted_loss.yaml` | 4ch (3 deform + 1 ref mask) | Weighted (`alpha=5.0`) | Test if deformation heatmap alone (no vertex positions) suffices |
| 4 | `no_expr_uniform_loss.yaml` | 1ch (ref mask only) | Uniform (`alpha=0.0`) | Isolate FLAME conditioning contribution |
| 5 | `no_expr_weighted_loss.yaml` | 1ch (ref mask only) | Weighted (`alpha=5.0`) | Test if loss weighting alone compensates for missing conditioning |

**Terminology:**
- **Positional encoding (42ch)**: Sinusoidal Fourier features of FLAME vertex positions — encodes WHERE the face is in space (geometry and pose).
- **Deformation (3ch)**: Per-vertex displacement from the neutral face — encodes HOW the face is moving (mouth opening, brow raising, jaw dropping). This is the rasterized expression heatmap.
- **Expression-weighted loss**: Amplifies the diffusion loss on pixels where deformation is large, so the denoiser spends more capacity on face dynamics.

**Run commands:**

```bash
conda activate cap4d_env
export PYTHONPATH=/data/pouyan/baseline/repository/cap4d:$PYTHONPATH

# 1. Full conditioning + weighted loss
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/full_cond_weighted_loss.yaml \
    --gpus 0 1 2 3 --output_dir outputs/full_cond_weighted_loss

# 2. Full conditioning + uniform loss
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/full_cond_uniform_loss.yaml \
    --gpus 0 1 2 3 --output_dir outputs/full_cond_uniform_loss

# 3. Deformation only + weighted loss
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/deform_only_weighted_loss.yaml \
    --gpus 0 1 2 3 --output_dir outputs/deform_only_weighted_loss

# 4. No FLAME conditioning + uniform loss
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/no_expr_uniform_loss.yaml \
    --gpus 0 1 2 3 --output_dir outputs/no_expr_uniform_loss

# 5. No FLAME conditioning + weighted loss
python talkinghead_sd21_unet_cap4d_based/train.py \
    --config talkinghead_sd21_unet_cap4d_based/configs/no_expr_weighted_loss.yaml \
    --gpus 0 1 2 3 --output_dir outputs/no_expr_weighted_loss
```

**Key comparisons:**
- **1 vs 2**: Does expression-weighted loss improve face dynamics with full conditioning?
- **1 vs 3**: Is vertex position encoding (42ch) necessary, or is the deformation heatmap (3ch) enough?
- **1 vs 4**: How much does FLAME spatial conditioning contribute overall?
- **4 vs 5**: Does expression-weighted loss help when the model has no spatial conditioning?
- **3 vs 4**: Is a minimal deformation heatmap better than no spatial signal at all?

## Codebase Structure

```
talkinghead_sd21_unet_cap4d_based/
├── conditioning/
│   ├── th_conditioning.py    # FLAME → spatial conditioning maps (46 channels)
│   └── mesh2img.py           # PyTorch3D mesh rasterizer (PropRenderer)
├── model/
│   ├── th_diffusion.py       # Latent diffusion training loop (expression-weighted loss, CFG, VAE)
│   ├── th_unet.py            # SD 2.1 UNet with 3D attention + audio cross-attn
│   ├── th_sampler.py         # Sliding-window DDIM sampler
│   ├── audio_encoder.py      # wav2vec2 backbone → per-frame audio tokens
│   ├── attention.py          # SpatioTemporalTransformer blocks
│   └── utils.py              # Noise schedule utilities
├── flame/
│   ├── flame.py              # FLAME 3DMM: shape/expression → mesh vertices
│   └── mouth.py              # Inner mouth vertex generation
├── data/
│   ├── video_dataset.py      # Training dataset (video + audio + FLAME)
│   ├── utils.py              # Image loading, cropping, vertex projection
│   ├── train_ids.txt         # Training clip IDs
│   └── val_ids.txt           # Validation clip IDs
├── utils/
│   └── background.py         # Background plate builder + compositing utilities
├── configs/
│   ├── full_cond_weighted_loss.yaml            # Full 46ch conditioning + weighted loss
│   ├── full_cond_uniform_loss.yaml            # Full 46ch conditioning + uniform loss
│   ├── deform_only_weighted_loss.yaml         # 4ch deformation only + weighted loss
│   ├── no_expr_uniform_loss.yaml              # 1ch ref mask only + uniform loss
│   └── no_expr_weighted_loss.yaml             # 1ch ref mask only + weighted loss
├── tests/
│   ├── README.md             # Test descriptions and usage
│   ├── test_pipeline.py      # Shape-only unit tests (10 tests, no real data)
│   ├── test_dataset.py       # Visual dataset test
│   └── test_training_integration.py  # End-to-end test with real data
├── train.py                  # Diffusion model training entry point
└── generate.py               # Inference entry point
```

## Testing

Run shape-only unit tests (no real data or checkpoints needed):

```bash
PYTHONPATH=. python talkinghead_sd21_unet_cap4d_based/tests/test_pipeline.py
```

Run integration tests with real data (requires 2 clips and SD 2.1 weights):

```bash
PYTHONPATH=. python talkinghead_sd21_unet_cap4d_based/tests/test_training_integration.py
```
