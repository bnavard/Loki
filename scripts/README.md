# Data Preparation Pipeline

All scripts needed to go from raw YouTube URLs to the cached tensors that the
Marionette and Marigold trainers consume. Each stage lives in its own
subfolder and is expected to run in the order below.

## Pipeline order

```
  download/      →  preprocess/      →  [FLAME tracking]   →  caption/        →  cache/                    →  manifest/
  raw clips      →  talkvid layout   →  data/flowface/     →  prosody text   →  GT + Marigold tensors     →  train/val split
```

FLAME tracking lives in [`../generate_exp_map/`](../generate_exp_map/) and is
not part of `scripts/` — it has its own setup and pixel3dmm dependency.

## Subfolders

### `download/` — YouTube clip scraping

yt-dlp-based downloader for the TalkVid dataset. Pulls ~8k 5-second talking-head
clips from the URLs in `talkvid_data.json`. Has built-in rate-limit handling
(exponential backoff, per-URL cooldowns).

See [`download/README.md`](download/README.md).

### `preprocess/` — face-centered crop + resample

For each raw downloaded mp4: detect the primary speaker with InsightFace,
compute a stable head-centered crop, and resample to 512x512 / 25fps video +
16kHz mono audio. Output matches the talkvid layout used by downstream training.

```bash
PYTHONPATH=. python scripts/preprocess/preprocess_talkvid_data.py

# Custom input directory:
PYTHONPATH=. python scripts/preprocess/preprocess_talkvid_data.py --input_dir data/additional_data
```

### `caption/` — prosody text captions

Combines Whisper large-v3 ASR transcription with Qwen2-Audio prosody description.
Each caption follows the format:
`A person says: '<transcription>' <prosody description>`.

```bash
# Parallel across GPUs:
PYTHONPATH=. python scripts/caption/generate_captions.py --gpu 0 --num_gpus 8

# ASR only (skip Qwen2-Audio):
PYTHONPATH=. python scripts/caption/generate_captions.py --asr_only

# Output: data/derived/captions/{clip_id}.json
```

### `cache/` — precomputed tensors (modular, one subfolder per signal)

Each cache signal is its own self-contained module with its own entry point,
README, and (where needed) dependencies.

**Ground-truth expression fields** (`cache/expression_field/`) — 45ch FLAME
rasterization per clip. Runs in the `marionette` env.

```bash
PYTHONPATH=. python scripts/cache/expression_field/cache.py --gpu 0 --num_gpus 4
# Output: data/derived/expression_field/{clip_id}/{expr_field.pt, deformation.mp4, deform_rgb/}
```

**Marigold-generated deformations** (`cache/marigold_deform/`) — runs the
trained SD3.5 Marigold model over every frame of every clip. Required before
running the `ablate_expr_source` study in Marigold mode. Runs in the
`marionette` env.

```bash
PYTHONPATH=. python scripts/cache/marigold_deform/cache.py \
    --checkpoint <path/to/marigold_checkpoint> \
    --output_dir data/derived/marigold_deform
# Output: data/derived/marigold_deform/{clip_id}/deformation.mp4
```

> Per-pixel 3D face surface position is already available via the FLAME
> rasterized positional encoding (channels 0–41 of the 46ch GT expression
> map), and per-pixel UV coordinates via
> `data/flame_tracking/preprocessing/{clip_id}/p3dmm/uv_map/` (output of
> pixel3dmm's UV head). No separate PRNet pipeline is needed.

### `manifest/` — training manifest + train/val split

```bash
# Validate each clip has FLAME tracking + caption, count frames
python scripts/manifest/build_manifest.py
# Output: data/derived/manifest.json

# Identity-based 98/2 split (all clips from same speaker go to same split)
python scripts/manifest/partition_dataset.py
# Output: data/derived/train_clips.json, data/derived/val_clips.json
```

Clip IDs are parsed to extract the YouTube video ID across several naming
conventions (`ID_NA_timestamp`, `language_X_videovideo{VID}_scene*`, etc.)
so that identity leakage is prevented across train/val.

### `tools/` — devops utilities

Miscellaneous infrastructure scripts not part of the training pipeline:

- `parallel_push155.sh` — chunked parallel scp transfer between GPU boxes.

## Output layout

```
data/derived/
├── captions/{clip_id}.json                      # prosody captions
├── expression_field/{clip_id}/                  # GT rasterized
│   ├── expr_field.pt
│   ├── deformation.mp4
│   └── deform_rgb/*.png
├── marigold_deform/{clip_id}/                   # Marigold-predicted
│   └── deformation.mp4
├── manifest.json
├── train_clips.json
└── val_clips.json
```

## Structure

```
scripts/
├── README.md
├── download/                            # YouTube scraping
├── preprocess/                          # face crop + resample
├── caption/                             # Whisper + Qwen2-Audio
├── cache/                               # precomputed tensors
│   ├── expression_field/                # GT FLAME rasterization (45ch)
│   └── marigold_deform/                 # Marigold-predicted deformation mp4s
├── manifest/                            # validation, train/val split
└── tools/                               # devops utilities
```