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

### 3. Cache Expression Fields

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
