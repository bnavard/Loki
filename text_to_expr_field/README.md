# Text-to-Expression Dense Field Video Generation

Generate FLAME expression dense field videos from text descriptions, removing the need for a driving video in the talking-head generation pipeline.

The rendering UNet (`talkinghead_sd21_unet_cap4d_based/`) requires a 45-channel expression dense field extracted from a driving video via FLAME 3DMM tracking. This pipeline trains a generative model to synthesize that field directly from a text prompt. Combined with text-to-speech for audio, this enables a text-only interface: reference portrait + text prompt → talking-head video.

## Table of Contents

- [Architecture](#architecture)
- [Expression Dense Field Format](#expression-dense-field-format)
- [Channel Reshaping for VAE Compatibility](#channel-reshaping-for-vae-compatibility)
- [Training Modes](#training-modes)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Integration with the Rendering UNet](#integration-with-the-rendering-unet)
- [Codebase Structure](#codebase-structure)
- [Dependencies](#dependencies)

## Architecture

```
Text prompt
    │
    ▼
┌──────────────────┐
│  UMT5 Encoder    │  (frozen, precomputed + cached)
└─────┬────────────┘
      │ text embeddings
      ▼
┌────────────────────────────────────┐
│  Wan DiT Transformer               │  ← LoRA or full fine-tuning
│  Flow Matching (velocity prediction)│
└─────┬──────────────────────────────┘
      │ denoised latent
      ▼
┌──────────────────┐
│  Wan VAE Decoder │  (frozen)
└─────┬────────────┘
      │
      ▼
  Expression dense field [T, C, H, W]
```

Flow matching trains the model to predict the velocity `v = noise - clean` given the interpolant `x_t = (1-t) * clean + t * noise` at uniform `t ~ [0, 1]`. Latents are normalized with pretrained per-channel `latents_mean/std` so the DiT operates in its expected input distribution.

## Expression Dense Field Format

The 45-channel expression dense field is produced by the FLAME rasterization pipeline:

| Channels | Content | Description |
|---|---|---|
| 0:42 | Positional encoding | Sinusoidal Fourier features of rasterized 3D FLAME vertex positions (3 coords x 7 freq bands x sin/cos = 42ch). Encodes face geometry and pose. |
| 42:45 | Expression deformation | Per-vertex displacement (dx, dy, dz) from the neutral FLAME mesh. Encodes mouth opening, brow movement, jaw drop, etc. |

Rasterized at 512x512 via PyTorch3D barycentric interpolation.

For the **deformation-only** experiments, only channels 42:45 (3ch) are used. This is a direct 3-channel video — no channel reshaping needed.

## Channel Reshaping for VAE Compatibility

The Wan VAE accepts 3-channel video. For the full 45-channel expression field, the channels are split into 15 groups of 3, stacked temporally, and padded to 4k+1:

```
[T=80, 45, 512, 512]
  → split: [80, 15, 3, 512, 512]
  → stack: [1200, 3, 512, 512]
  → pad:   [1201, 3, 512, 512]  (4k+1 = 4x300+1)
  → VAE:   [16, T_latent, 64, 64]
```

For the 3ch deformation map, no reshaping is needed — it goes directly to the VAE as a standard video.

At inference, latents are denormalized and decoded through the VAE directly (bypassing the pipeline's default `val * 0.5 + 0.5` clamp which would destroy expression field values).

## Training Modes

The pipeline supports multiple training configurations via YAML configs:

| Config | Model | Data | Fine-tuning | Description |
|---|---|---|---|---|
| `train_config.yaml` | Wan2.2-T2V-A14B (14B) | Cached 45ch latents | LoRA | Full expression field, cached VAE latents |
| `train_config_deform.yaml` | Wan2.2-T2V-A14B (14B) | Cached 3ch latents | LoRA | Deformation-only, cached VAE latents |
| `train_config_1b_deform.yaml` | Wan2.1-T2V-1.3B | On-the-fly 3ch | Full | Deformation-only, on-the-fly VAE encoding |

**Cached mode**: VAE latents and text embeddings are precomputed to disk. Neither the VAE nor text encoder is loaded during training — all GPU memory goes to the DiT.

**On-the-fly mode**: Expression fields are computed from `fit.npz` via FLAME + PyTorch3D rasterization and VAE-encoded live. Requires `num_workers=0` (CUDA in dataset). Used with smaller models (1.3B) where the VAE fits alongside the transformer.

## Data Preparation

Data preprocessing (caption generation, manifest building, VAE latent caching, text embedding caching) is handled by the shared `caching/` module at the repo root. See the [caching README](../caching/README.md) for details.

```bash
# 1. Generate captions
PYTHONPATH=. python caching/scripts/generate_captions.py --gpu 0 --num_gpus 8

# 2. Build manifest
python caching/scripts/build_manifest.py

# 3. Cache VAE latents (for cached training mode)
PYTHONPATH=. torchrun --nproc_per_node=4 caching/scripts/cache_vae_latents.py

# 4. Cache text embeddings
PYTHONPATH=. torchrun --nproc_per_node=4 caching/scripts/cache_text_embeddings.py
```

## Training

```bash
# Cached latent training (14B, LoRA):
PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
    text_to_expr_field/scripts/train.py \
    --config text_to_expr_field/configs/train_config.yaml

# On-the-fly deform training (1.3B, full fine-tuning):
PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
    text_to_expr_field/scripts/train.py \
    --config text_to_expr_field/configs/train_config_1b_deform.yaml
```

Key config parameters:

| Parameter | Description |
|---|---|
| `model_id` | HuggingFace model ID (e.g. `Wan-AI/Wan2.2-T2V-A14B-Diffusers`) |
| `use_lora` | `true` for LoRA, `false` for full fine-tuning |
| `mode` | `expr_field` (45ch) or `deform` (3ch) |
| `on_the_fly` | `true` to compute + VAE-encode live, `false` to load cached latents |
| `target_latent_T` | Direct latent temporal window size (overrides the 45ch formula) |
| `cfg_dropout` | Text conditioning dropout rate for classifier-free guidance |

## Inference

Supports multi-GPU parallelism via torchrun. Each GPU processes a disjoint subset of prompts.

```bash
# Single GPU:
PYTHONPATH=. python text_to_expr_field/scripts/inference.py \
    --prompts text_to_expr_field/configs/eval_prompts.json \
    --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final

# Multi-GPU:
PYTHONPATH=. torchrun --nproc_per_node=4 \
    text_to_expr_field/scripts/inference.py \
    --prompts text_to_expr_field/configs/eval_prompts.json \
    --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final

# Deformation-only inference:
PYTHONPATH=. python text_to_expr_field/scripts/inference.py \
    --prompts text_to_expr_field/configs/eval_prompts.json \
    --checkpoint outputs/text_to_deform_1b/run_YYYYMMDD/step_000250 \
    --mode deform --target_real_frames 84 --num_frames 81
```

Output per prompt: saved tensor (`.pt`), `prompt.txt`, and visualization videos (deformation map, positional encoding bands, combined grid). Results are saved alongside the checkpoint at `{checkpoint}/inference/{prompt_id}/`.

### Visualize Ground Truth

Compare generated outputs against real FLAME-tracked expression fields:

```bash
PYTHONPATH=. python text_to_expr_field/scripts/visualize_ground_truth.py \
    --clip_id 39Y_gFC9SmY_NA_1123.760_1128.801 \
    --num_frames 24 --output_dir outputs/ground_truth
```

## Integration with the Rendering UNet

The generated expression field is a drop-in replacement for FLAME tracking output:

1. Generate expression field from text → `[T, 45, H, W]`
2. Generate audio from text via TTS
3. Feed both into the rendering UNet (`talkinghead_sd21_unet_cap4d_based/`)

No modifications to the rendering UNet are needed.

## Codebase Structure

```
text_to_expr_field/
├── scripts/
│   ├── train.py                      # Training loop (LoRA or full fine-tuning)
│   ├── inference.py                  # Multi-GPU inference from text prompts
│   └── visualize_ground_truth.py     # Visualize real FLAME expression fields
├── src/
│   ├── data/
│   │   ├── base_dataset.py           # Base class: manifest + text embed loading
│   │   ├── cached_dataset.py         # Loads precomputed VAE latents, temporal slicing
│   │   ├── onthefly_dataset.py       # FLAME → rasterize → VAE encode live
│   │   └── collate.py                # Variable-length text embedding padding
│   ├── model/
│   │   ├── pipeline.py               # Load pipelines for training and inference
│   │   ├── lora.py                   # LoRA and full fine-tuning setup
│   │   └── checkpoint.py             # Checkpoint saving
│   ├── vis/
│   │   ├── video.py                  # normalize_to_uint8, save_video
│   │   └── expr_field.py             # visualize_expr_field, visualize_deform
│   └── utils/
│       └── reshape.py                # Pseudo-video ↔ expression field reshaping
└── configs/
    ├── train_config.yaml             # 14B, LoRA, cached 45ch
    ├── train_config_deform.yaml      # 14B, LoRA, cached 3ch deform
    ├── train_config_1b_deform.yaml   # 1.3B, full fine-tuning, on-the-fly 3ch
    ├── eval_prompts.json             # 10 OOD evaluation prompts
    └── eval_prompts_single.json      # Single training prompt for quick testing
```

Derived data:

```
data/derived/
├── captions/{clip_id}.json           # text captions
├── vae_latent_cache/{clip_id}.pt     # precomputed VAE latents (45ch)
├── vae_latent_cache_deform/{clip_id}.pt  # precomputed VAE latents (3ch deform)
├── prompt_latent_cache/{clip_id}.pt  # precomputed UMT5 text embeddings
└── manifest.json                     # training manifest
```

## Dependencies

**`cap4d_env`** (PyTorch 2.4.1, PyTorch3D, diffusers):
- Required for FLAME rasterization (data preparation, on-the-fly training, ground truth visualization)
- Required for text embedding caching (loads UMT5)

```
diffusers>=0.33.0
peft>=0.10
accelerate>=0.30
safetensors
pyyaml
einops
opencv-python
pytorch3d
```
