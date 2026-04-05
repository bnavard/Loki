# Text-to-Expression Dense Field Video Generation

Generate FLAME expression dense field videos from text descriptions, removing the need for a driving video in the talking-head generation pipeline.

The rendering UNet (`talkinghead_sd21_unet_cap4d_based/`) requires a 45-channel expression dense field extracted from a driving video via FLAME 3DMM tracking. This pipeline trains a generative model to synthesize that field directly from a text prompt. Combined with text-to-speech for audio, this enables a text-only interface: reference portrait + text prompt → talking-head video.

The pipeline uses Wan2.2-T2V-A14B (14B DiT) fine-tuned with LoRA via flow matching on paired (caption, expression field) data. Text embeddings and VAE latents are precomputed and cached to disk to maximize training throughput.

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
│  UMT5 Encoder    │  (frozen, precomputed + cached)
└─────┬────────────┘
      │ text embeddings (loaded from cache)
      ▼
┌────────────────────────────────────┐
│  Wan2.2 DiT (T2V-A14B, 14B)        │  ← LoRA fine-tuned
│  Flow Matching loss                │
└─────┬──────────────────────────────┘
      │ denoised latent
      ▼
┌──────────────────┐
│  Wan2.2 VAE      │  (frozen, precomputed + cached)
│  Decoder         │
└─────┬────────────┘
      │
      ▼
  Reassemble → [T, 45, H, W] expression dense field
```

Both VAE latents and UMT5 text embeddings are precomputed and cached to disk. During training, neither the VAE encoder nor the text encoder is loaded — the training loop only loads the DiT transformer + LoRA adapters, maximizing GPU memory for the model.

## Expression Dense Field Format

The 45-channel expression dense field is produced by the FLAME rasterization pipeline:

| Channels | Content | Description |
|---|---|---|
| 0:42 | Positional encoding | Sinusoidal Fourier features of rasterized 3D FLAME vertex positions (3 coords × 7 freq bands × sin/cos = 42ch). Encodes face geometry and pose. |
| 42:45 | Expression deformation | Per-vertex displacement (Δx, Δy, Δz) from the neutral FLAME mesh. Encodes mouth opening, brow movement, jaw drop, etc. |

Rasterized at 512×512 via PyTorch3D barycentric interpolation.

## Channel Reshaping for VAE Compatibility

The Wan2.2 VAE accepts 3-channel video. The 45-channel expression field is split into 15 groups of 3 channels, stacked temporally, and padded to satisfy the VAE's 4k+1 frame requirement:

```
[T=80, 45, 512, 512]
  → split: [80, 15, 3, 512, 512]
  → stack: [1200, 3, 512, 512]
  → pad:   [1201, 3, 512, 512]  (4k+1 = 4×300+1)
  → VAE:   latent tensor
```

At inference, the reverse: VAE decode → drop padding → reshape → [T, 45, H, W].

## Data Preparation

### Source Data Layout

```
data/
├── talkvid/
│   ├── talkvid/{clip_id}.mp4     # source videos
│   └── audio/{clip_id}.wav       # 16kHz mono audio
└── flowface/
    └── {clip_id}/
        ├── fit.npz               # FLAME tracking parameters
        └── images/cam0/*.jpg     # extracted frames
```

### Step 1: Generate Text Captions

Combines Whisper large-v3 ASR transcription with Qwen2-Audio prosody description. The prompt instructs the model to describe vocal delivery style (tone, pace, emphasis) without transcribing words.

```bash
conda activate cap4d_env
export PYTHONPATH=/data/pouyan/baseline/repository/cap4d:$PYTHONPATH

# Parallel across GPUs:
python text_to_expr_field/scripts/generate_captions.py --gpu 0 --num_gpus 8

# Output: data/derived/captions/{clip_id}.json
```

### Step 2: Build Training Manifest

Validates that each clip has both FLAME tracking data and a caption.

```bash
python text_to_expr_field/scripts/build_manifest.py
# Output: data/derived/manifest.json
```

### Step 3: Cache VAE Latents

Encodes expression fields through the Wan2.2 VAE using DDP. For each clip, computes the 45-channel expression field on the fly from `fit.npz` (first 80 frames, deterministic), reshapes into a pseudo-video, and saves the latent tensor. The `DistributedSampler` shards clips across GPUs automatically.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 text_to_expr_field/scripts/cache_vae_latents.py
# Output: data/derived/vae_latent_cache/{clip_id}.pt
```

### Step 4: Cache Text Embeddings

Encodes captions through the frozen UMT5-XXL text encoder. This frees ~13B params of GPU memory during training.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 text_to_expr_field/scripts/cache_text_embeddings.py
# Output: data/derived/prompt_latent_cache/{clip_id}.pt
```

## Training

Fine-tunes Wan2.2-T2V-A14B with LoRA using flow matching loss. Both VAE latents and text embeddings are loaded from precomputed caches — neither the VAE encoder nor UMT5 text encoder is loaded during training.

Each training step:
1. Loads a cached VAE latent and randomly slices a temporal window (`target_real_frames` setting)
2. Normalizes the latent with pretrained `latents_mean` / `latents_std`
3. Adds noise via flow matching interpolation
4. Runs the DiT forward pass with cached text embeddings
5. Computes MSE velocity loss in float32

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch \
    --num_processes 4 --mixed_precision bf16 \
    text_to_expr_field/scripts/train.py \
    --config text_to_expr_field/configs/train_config.yaml
```

Key parameters (`train_config.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `model_id` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Base model |
| `lora_rank` | 64 | LoRA rank |
| `target_real_frames` | 24 | Frames per training window (random temporal slice) |
| `gradient_checkpointing` | true | Recompute activations in backward pass |
| `cfg_dropout` | 0.1 | Drop text conditioning 10% for CFG |
| `cached_only` | true | Only train on clips with cached latents |

## Inference

```bash
PYTHONPATH=. python text_to_expr_field/scripts/inference.py \
    --prompt "A person says: 'Hello, welcome.' Warm tone, moderate pace." \
    --checkpoint outputs/text_to_expr_field/run_YYYYMMDD/final \
    --output outputs/generated_expr_field.pt
```

Output: expression dense field tensor + deformation visualizations.

## Integration with the Rendering UNet

The generated expression field is a drop-in replacement for FLAME tracking output. To produce the final talking-head video:

1. Generate expression field from text → `[T, 45, H, W]`
2. Generate audio from text via TTS
3. Feed both into the rendering UNet (`talkinghead_sd21_unet_cap4d_based/`)

No modifications to the rendering UNet are needed.

## Codebase Structure

```
text_to_expr_field/
├── scripts/
│   ├── generate_captions.py          # Whisper ASR + Qwen2-Audio prosody → captions
│   ├── build_manifest.py             # Validate data + build manifest.json
│   ├── cache_vae_latents.py          # DDP: expression field → VAE latent → disk
│   ├── cache_text_embeddings.py      # DDP: caption → UMT5 embedding → disk
│   ├── train.py                      # LoRA fine-tuning (14B T2V-A14B)
│   ├── train_ti2v.py                 # LoRA fine-tuning (TI2V variant)
│   └── inference.py                  # Generate expression fields from text
├── src/
│   ├── dataset.py                    # Training dataset (cached latents + text embeddings)
│   └── utils.py                      # Channel reshaping utilities
├── configs/
│   ├── train_config.yaml             # T2V-A14B training config
│   └── train_config_ti2v.yaml        # TI2V variant config
└── README.md
```

Derived data:

```
data/derived/
├── captions/{clip_id}.json           # text captions
├── vae_latent_cache/{clip_id}.pt     # precomputed VAE latents
├── prompt_latent_cache/{clip_id}.pt  # precomputed UMT5 text embeddings
└── manifest.json                     # training manifest
```

## Dependencies

**`cap4d_env`** (PyTorch 2.4.1, PyTorch3D, diffusers 0.33):
- Required for VAE latent caching (uses PyTorch3D for FLAME rasterization)
- Required for text embedding caching (loads UMT5)

**For training** (same env or separate):
```
diffusers>=0.33.0
peft>=0.10
accelerate>=0.30
bitsandbytes  (optional, for 8-bit Adam)
pyyaml
```
