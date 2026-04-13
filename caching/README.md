# Data Caching

Precomputes and caches expensive data artifacts (expression fields, deformation predictions) to disk for fast training. Scripts support multi-GPU parallelism via DDP or manual GPU sharding.

For data preparation scripts (caption generation, manifest building, train/val splitting), see [`scripts/README.md`](../scripts/README.md).

## Scripts

### 1. Preprocess TalkVid Data

Takes raw downloaded clips (from `download_clips.py`), randomly samples a 5-second segment from each, detects the speaker's face with InsightFace, computes a stable head-centered crop, and outputs 512x512 / 25fps video + 16kHz mono WAV audio matching the existing talkvid layout.

```bash
PYTHONPATH=. python caching/scripts/preprocess_talkvid_data.py

# Custom input directory:
PYTHONPATH=. python caching/scripts/preprocess_talkvid_data.py --input_dir data/additional_data

# Output:
#   data/talkvid/talkvid/{clip_id}.mp4   (512x512, 25fps)
#   data/talkvid/audio/{clip_id}.wav     (16kHz, mono, PCM s16le)
```

### 2. Cache Expression Fields (ground-truth)

Computes the full 45-channel expression field from `fit.npz` for every clip using the FLAME-rasterization pipeline. Saves the tensor, a deformation map video, and per-frame deformation images. These are the **ground-truth** expression maps used as spatial conditioning (or as Stage 1 targets for the Marigold generator).

```bash
# Parallel across 4 GPUs:
PYTHONPATH=. python caching/scripts/cache_expression_fields.py --gpu 0 --num_gpus 4

# Output per clip:
#   data/derived/expression_field/{clip_id}/expr_field.pt    (45ch tensor)
#   data/derived/expression_field/{clip_id}/deformation.mp4  (channels 42:45 video)
#   data/derived/expression_field/{clip_id}/deform_rgb/      (per-frame PNGs)
```

### 3. Cache Marigold-Generated Deformations

Runs the trained Marigold SD3.5 model over every frame of every clip to produce **generated** deformation map videos. Use these as spatial conditioning in Marionette to compare against the ground-truth deformations.

```bash
PYTHONPATH=. python caching/scripts/cache_marigold_deform.py \
    --checkpoint <path/to/marigold_checkpoint> \
    --output_dir data/derived/marigold_deform

# Output per clip:
#   data/derived/marigold_deform/{clip_id}/deformation.mp4   (visualization)
#   data/derived/marigold_deform/{clip_id}/deform_field.pt   (optional, disabled by default)
```

## Structure

```
caching/
├── scripts/
│   ├── preprocess_talkvid_data.py    # Face crop + resample downloaded clips
│   ├── cache_expression_fields.py    # GT expression field + deformation visualization
│   ├── cache_marigold_deform.py      # Marigold-predicted deformation maps
│   ├── _expr_field_dataset.py        # Helper dataset
│   └── archive/                      # Deprecated scripts (Wan-based pipeline)
└── README.md
```

## Output Layout

```
data/derived/
├── expression_field/{clip_id}/expr_field.pt      # GT 45ch expression field
├── expression_field/{clip_id}/deformation.mp4    # GT deformation map video
├── expression_field/{clip_id}/deform_rgb/*.png   # GT per-frame deformation images
├── marigold_deform/{clip_id}/deformation.mp4     # Marigold-predicted deformation video
└── marigold_deform/{clip_id}/deform_field.pt     # Marigold-predicted deformation tensor (optional)
```