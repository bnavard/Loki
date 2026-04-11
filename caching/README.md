# Data Caching

Precomputes and caches expensive data artifacts (VAE latents, text embeddings, expression fields) to disk for fast training. All scripts support multi-GPU parallelism via DDP or manual GPU sharding.

For data preparation scripts (caption generation, manifest building, train/val splitting), see [`scripts/README.md`](../scripts/README.md).

## Scripts

### 1. Cache VAE Latents

Encodes expression fields through the frozen Wan VAE using DDP. Computes the 45-channel expression field on the fly from `fit.npz`, reshapes into pseudo-video, and saves the latent tensor.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 caching/scripts/cache_vae_latents.py
# Output: data/derived/vae_latent_cache/{clip_id}.pt
```

### 2. Cache Text Embeddings

Encodes captions through the frozen UMT5-XXL text encoder. Frees ~13B params of GPU memory during training.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 caching/scripts/cache_text_embeddings.py
# Output: data/derived/prompt_latent_cache/{clip_id}.pt
```

### 3. Preprocess TalkVid Data

Takes raw downloaded clips (from `download_clips.py`), randomly samples a 5-second segment from each, detects the speaker's face with InsightFace, computes a stable head-centered crop, and outputs 512x512 / 25fps video + 16kHz mono WAV audio matching the existing talkvid layout.

```bash
PYTHONPATH=. python caching/scripts/preprocess_talkvid_data.py

# Custom input directory:
PYTHONPATH=. python caching/scripts/preprocess_talkvid_data.py --input_dir data/additional_data

# Output:
#   data/talkvid/talkvid/{clip_id}.mp4   (512x512, 25fps)
#   data/talkvid/audio/{clip_id}.wav     (16kHz, mono, PCM s16le)
```

### 4. Cache Expression Fields

Computes the full 45-channel expression field from fit.npz for every clip. Saves the tensor, a deformation map video, and per-frame deformation images.

```bash
# Parallel across 4 GPUs:
PYTHONPATH=. python caching/scripts/cache_expression_fields.py --gpu 0 --num_gpus 4

# Output per clip:
#   data/derived/expression_field/{clip_id}/expr_field.pt    (45ch tensor)
#   data/derived/expression_field/{clip_id}/deformation.mp4  (channels 42:45 video)
#   data/derived/expression_field/{clip_id}/deform_rgb/      (per-frame PNGs)
```

## Structure

```
caching/
├── scripts/
│   ├── preprocess_talkvid_data.py    # Face crop + resample downloaded clips → talkvid format
│   ├── cache_vae_latents.py          # DDP: expression field → VAE latent → disk
│   ├── cache_text_embeddings.py      # DDP: caption → UMT5 embedding → disk
│   ├── cache_expression_fields.py    # Expression field + deformation visualization
│   └── _expr_field_dataset.py        # Minimal dataset for VAE caching
└── README.md
```

## Output Layout

```
data/derived/
├── vae_latent_cache/{clip_id}.pt                  # precomputed VAE latents (45ch)
├── vae_latent_cache_deform/{clip_id}.pt           # precomputed VAE latents (3ch deform)
├── prompt_latent_cache/{clip_id}.pt               # precomputed UMT5 text embeddings
├── expression_field/{clip_id}/expr_field.pt       # full 45ch expression field
├── expression_field/{clip_id}/deformation.mp4     # deformation map video
└── expression_field/{clip_id}/deform_rgb/*.png    # per-frame deformation images
```
