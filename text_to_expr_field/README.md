# Text-to-Expression Dense Field Video Generation

Generate FLAME expression dense field videos from text descriptions, removing the need for a driving video in the talking-head generation pipeline.

The existing rendering UNet (`talkinghead_sd21_unet_cap4d_based/`) requires a 45-channel expression dense field extracted from a real driving video via FLAME 3DMM tracking. This pipeline replaces that dependency by training a generative model (Wan2.2-T2V-A14B) to synthesize the expression field directly from a text prompt describing what a person says and how they say it. Combined with text-to-speech (TTS) for audio, this enables a text-only interface: reference portrait + text prompt → talking-head video.

## Table of Contents

- [Architecture](#architecture)
- [Expression Dense Field Format](#expression-dense-field-format)
- [Channel Reshaping for VAE Compatibility](#channel-reshaping-for-vae-compatibility)
- [Data Preparation](#data-preparation)
  - [Step 1: Generate Text Captions](#step-1-generate-text-captions)
  - [Step 2: Build Training Manifest](#step-2-build-training-manifest)
  - [Step 3: Cache VAE Latents](#step-3-cache-vae-latents)
  - [Step 4: Cache Text Embeddings](#step-4-cache-text-embeddings)
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
│  UMT5 Encoder    │  (frozen, Wan2.2's native text encoder)
└─────┬────────────┘
      │ text embeddings
      ▼
┌────────────────────────────────────┐
│  Wan2.2 DiT MoE (A14B)            │  ← LoRA fine-tuned on expression field videos
│  - transformer   (high-noise exp.) │
│  - transformer_2 (low-noise exp.)  │
│  (Flow Matching)                   │
└─────┬──────────────────────────────┘
      │ denoised latent
      ▼
┌──────────────────┐
│  Wan2.2 VAE      │  (frozen, 3ch in / 3ch out, z_dim=16)
│  Decoder         │  decode 15 sub-videos independently
└─────┬────────────┘
      │ 15 × [T=80, 3, 512, 512]
      ▼
  Reassemble → [T=80, 45, 512, 512]
  Expression dense field video
```

**Frozen:** VAE (encoder + decoder), UMT5 text encoder.
**Trained:** LoRA adapters on both DiT experts (`transformer` and `transformer_2`).

## Expression Dense Field Format

The 45-channel expression dense field is produced by the existing FLAME rasterization pipeline in `talkinghead_sd21_unet_cap4d_based/`:

| Channels | Content | Description |
|---|---|---|
| 0:42 | Positional encoding | Sinusoidal Fourier features of rasterized 3D vertex positions (3 coords × 7 freq bands × sin/cos = 42ch). Encodes WHERE the face is. |
| 42:45 | Expression deformation | Per-vertex displacement (Δx, Δy, Δz) from the neutral FLAME mesh, rasterized onto the 2D grid. Encodes HOW the face is moving. |

These are rasterized onto a 512×512 grid via PyTorch3D barycentric interpolation. The relevant code path:

1. `talkinghead_sd21_unet_cap4d_based/flame/flame.py` → `compute_flame()` — FLAME params → vertex positions + deformation offsets
2. `talkinghead_sd21_unet_cap4d_based/conditioning/mesh2img.py` → `PropRenderer.render()` — rasterize vertices onto 2D grid
3. `talkinghead_sd21_unet_cap4d_based/conditioning/th_conditioning.py` → `THConditioning.forward()` — positional encoding + assembly into 45ch tensor

## Channel Reshaping for VAE Compatibility

The Wan2.2 VAE accepts 3-channel RGB video and outputs 3-channel video. Our expression field has 45 channels. We reshape by splitting into 15 groups of 3 channels:

**Encoding (data preparation):**
```
[T=80, 45, 512, 512]
  → split into 15 groups: [80, 15, 3, 512, 512]
  → stack temporally:     [1200, 3, 512, 512]
  → pad to 4k+1:          [1201, 3, 512, 512]
  → VAE encode:           latent tensor
```

**Decoding (inference):**
```
latent tensor
  → VAE decode:           [1201, 3, 512, 512]
  → drop padding:         [1200, 3, 512, 512]
  → reshape:              [80, 15, 3, 512, 512]
  → merge channels:       [80, 45, 512, 512]
```

Groups 0-13 correspond to positional encoding channels (42ch), group 14 is the deformation field (3ch). The Wan2.2 VAE reconstructs each group with acceptable quality — minor loss on high-frequency positional encoding bands is tolerable for downstream rendering.

The VAE can handle 1201 frames at 512×512 in a single forward pass (~20GB peak GPU memory on H200).

## Data Preparation

### Source Data Layout

```
data/
├── talkvid/
│   ├── talkvid/{clip_id}.mp4     # 8313 source videos (~5s each, ~80+ frames)
│   └── audio/{clip_id}.wav       # 16kHz mono audio
└── flowface/
    └── {clip_id}/
        ├── fit.npz               # FLAME tracking parameters
        ├── images/cam0/*.jpg     # Extracted frames
        └── bg/cam0/*.png         # Foreground masks
```

### Step 1: Generate Text Captions

Combines ASR transcription (Whisper large-v3) with prosody description (Qwen2-Audio) into structured captions describing both what is said and how it is delivered.

```bash
conda activate cap4d_env
export PYTHONPATH=/data/pouyan/baseline/repository/cap4d:$PYTHONPATH

# Full pipeline (ASR + prosody):
python text_to_expr_field/scripts/generate_captions.py --gpu 0 --num_gpus 8

# ASR only (faster, skip prosody):
python text_to_expr_field/scripts/generate_captions.py --gpu 0 --num_gpus 8 --asr_only

# Test on one clip:
python text_to_expr_field/scripts/generate_captions.py --test

# Output: data/derived/captions/{clip_id}.json
```

Each caption JSON contains:
```json
{
  "clip_id": "abc123",
  "transcription": "I can't believe you did that",
  "prosody": "Calm, moderate-paced delivery with steady intonation...",
  "caption": "A person says: 'I can't believe you did that' Calm, moderate-paced delivery..."
}
```

### Step 2: Build Training Manifest

Validates that each clip has both FLAME tracking and a caption, then produces a manifest:

```bash
python text_to_expr_field/scripts/build_manifest.py
# Output: data/derived/manifest.json
```

### Step 3: Cache VAE Latents

Precomputes and caches VAE latents for all clips using DDP. For each clip, computes the 45-channel expression field on the fly from `fit.npz` (always the first 80 frames, deterministic), reshapes into a pseudo-video, encodes through the frozen Wan2.2 VAE, and saves the latent to disk.

This eliminates the VAE encoding bottleneck during training — the training loop loads precomputed latents directly.

Since the dataset is fully deterministic (always `start=0`, `target_frames=80`), the cached latents are guaranteed to match what the training loop would compute, making them safe to use as a drop-in replacement.

```bash
# Single GPU:
PYTHONPATH=. python text_to_expr_field/scripts/cache_vae_latents.py

# Multi-GPU DDP (4 GPUs):
PYTHONPATH=. torchrun --nproc_per_node=4 text_to_expr_field/scripts/cache_vae_latents.py

# 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 text_to_expr_field/scripts/cache_vae_latents.py

# Test on one batch:
PYTHONPATH=. python text_to_expr_field/scripts/cache_vae_latents.py --test

# Output: data/derived/vae_latent_cache/{clip_id}.pt
```

Each cached `.pt` file contains:
```python
{"latent": tensor, "num_expr_frames": 80}
```

The DDP script uses `DistributedSampler` to shard clips across GPUs with no manual interleaving needed. Already-cached clips are automatically skipped, so interrupted runs can be resumed.

### Step 4: Cache Text Embeddings

Precomputes UMT5-XXL text embeddings for all captions. This frees ~13B params of GPU memory during training since the text encoder no longer needs to be loaded.

```bash
# Single GPU:
PYTHONPATH=. python text_to_expr_field/scripts/cache_text_embeddings.py

# Multi-GPU DDP:
PYTHONPATH=. torchrun --nproc_per_node=4 text_to_expr_field/scripts/cache_text_embeddings.py

# Test:
PYTHONPATH=. python text_to_expr_field/scripts/cache_text_embeddings.py --test

# Output: data/derived/text_embed_cache/{clip_id}.pt
```

Each cached `.pt` file contains:
```python
{"text_embed": tensor, "caption": "A person says: ..."}
```

Text encoding is fast (batch_size=32 by default), so caching all ~7k clips takes only a few minutes.

## Training

Fine-tunes Wan2.2-T2V-A14B with LoRA on the expression field dataset. Both VAE latents and text embeddings are loaded from precomputed caches — the VAE encoder and UMT5 text encoder are never loaded during training, freeing significant GPU memory for the DiT.

```bash
# Install dependencies (if not already):
pip install peft pyyaml diffusers>=0.34.0

# Single GPU:
PYTHONPATH=. python text_to_expr_field/scripts/train.py \
    --config text_to_expr_field/configs/train_config.yaml

# 8x H200 distributed:
PYTHONPATH=. accelerate launch --num_processes 8 --mixed_precision bf16 \
    text_to_expr_field/scripts/train.py \
    --config text_to_expr_field/configs/train_config.yaml
```

Key training parameters (in `configs/train_config.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `model_id` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Base model |
| `lora_rank` | 128 | LoRA rank (reduce to 64 if overfitting) |
| `lora_alpha` | 128 | LoRA scaling factor |
| `batch_size` | 1 | Per GPU (MoE 14B + large latents are memory-heavy) |
| `gradient_accumulation` | 4 | Effective batch = batch_size × num_gpus × accum |
| `lr` | 1e-5 | AdamW learning rate |
| `warmup_steps` | 500 | Cosine schedule with linear warmup |
| `max_steps` | 20,000 | Total training steps |
| `cfg_dropout` | 0.1 | Drop text conditioning 10% for CFG |
| `mixed_precision` | bf16 | Native on H200 |

The dataset loads precomputed VAE latents from `data/derived/vae_latent_cache/` and text embeddings from `data/derived/text_embed_cache/`. With `cached_only: true`, only clips with cached latents are included — no on-the-fly computation needed. Run `cache_vae_latents.py` and `cache_text_embeddings.py` before training.

Each clip is one training sample. Clips with ≥80 frames are included (truncated to the first 80 frames). Fully deterministic — same data every epoch.

LoRA checkpoints (both experts) are saved every 2,000 steps. Each run creates a timestamped directory with a copy of the config.

## Inference

```bash
PYTHONPATH=. python text_to_expr_field/scripts/inference.py \
    --prompt "A person says: 'Hello, welcome to the presentation.' Warm tone, moderate pace." \
    --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final \
    --output outputs/generated_expr_field.pt
```

| Argument | Default | Description |
|---|---|---|
| `--prompt` | (required) | Text caption describing speech + delivery |
| `--checkpoint` | (required) | Path to LoRA checkpoint directory |
| `--guidance_scale` | 7.5 | CFG scale |
| `--num_inference_steps` | 50 | Denoising steps |
| `--height` / `--width` | 512 | Output resolution |
| `--seed` | 42 | Random seed for reproducibility |

Output: `[80, 45, 512, 512]` expression dense field tensor + deformation visualizations.

## Integration with the Rendering UNet

The generated expression field is in the exact format expected by the rendering UNet. To produce the final talking-head video:

1. Generate expression field from text: `text → [80, 45, 512, 512]`
2. Generate audio from text via TTS (e.g., Bark, XTTS)
3. Feed both into the rendering UNet:
   - The 45-channel field serves as `pos_enc` in `THConditioning.forward()`, concatenated with a reference mask channel (added separately) to form the full 46-channel conditioning input
   - Audio is encoded by wav2vec2 and injected via cross-attention

No modifications to the rendering UNet are needed — the generated field is a drop-in replacement for the FLAME tracking output.

## Codebase Structure

```
text_to_expr_field/
├── scripts/
│   ├── generate_captions.py          # Step 1: Whisper ASR + Qwen2-Audio prosody
│   ├── build_manifest.py             # Step 2: Validate data + build manifest
│   ├── cache_vae_latents.py          # Step 3: DDP precompute VAE latents for all clips
│   ├── cache_text_embeddings.py      # Step 4: DDP precompute UMT5 text embeddings
│   ├── train.py                      # LoRA fine-tuning of Wan2.2-T2V-A14B
│   └── inference.py                  # Generate expression fields from text
├── src/
│   ├── dataset.py                    # Training dataset (cached latents or on-the-fly computation)
│   └── utils.py                      # Channel reshaping utilities (pseudo-video ↔ expr field)
├── configs/
│   └── train_config.yaml             # Training hyperparameters
├── setup_exprmap_env.sh              # Conda environment setup script
└── README.md
```

Derived data produced by preprocessing:

```
data/derived/
├── captions/{clip_id}.json           # Structured text captions (Step 1)
├── vae_latent_cache/{clip_id}.pt     # Precomputed VAE latents (Step 3)
├── text_embed_cache/{clip_id}.pt    # Precomputed UMT5 text embeddings (Step 4)
└── manifest.json                     # Training manifest (Step 2)
```

## Dependencies

Two conda environments are used:

**`cap4d_env`** — for VAE latent caching and expression field computation (requires PyTorch3D):
```
torch==2.4.1+cu121
pytorch3d==0.7.8
diffusers>=0.33.0
transformers>=4.40
soundfile
```

**`exprmap`** — for caption generation with VideoLLaMA2 (optional, setup via `setup_exprmap_env.sh`):
```
torch==2.4.1+cu121
flash-attn==2.5.8
videollama2 (audio_visual branch)
transformers>=4.42
```

**For training** (either env, needs peft + accelerate):
```
diffusers>=0.34.0       # Wan2.2-A14B MoE support (transformer + transformer_2)
peft>=0.10              # LoRA
accelerate>=0.30        # Distributed training
pyyaml
```

## Future: Phase 2 — Audio Cross-Attention

Phase 1 (this pipeline) uses text-only conditioning. Phase 2 will add audio cross-attention:

1. Frozen wav2vec2 encoder extracts per-frame audio tokens
2. Learned projection maps wav2vec2 → DiT hidden dimension
3. Cross-attention layers inserted in each DiT block (zero-initialized)
4. Training: GT audio from dataset; Inference: TTS-generated audio
5. Independent CFG dropout for text (10%) and audio (10%)
