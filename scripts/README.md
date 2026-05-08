# Data Preparation Pipeline

Scripts that take raw YouTube URLs to the training-ready data layout consumed
by `loki/`. Run in the order shown.

```
  download/       →  preprocess/       →  [FLAME tracking]    →  manifest/
  raw clips       →  talkvid layout    →  data/flowface/      →  train / val split
```

FLAME tracking lives in [`../generate_exp_map/`](../generate_exp_map/) — it has
its own env and pixel3dmm dependency and is not part of `scripts/`.

## Subfolders

### `download/` — YouTube clip scraping

yt-dlp-based downloader for the TalkVid dataset. Pulls ~8k 5-second
talking-head clips from the URLs in `talkvid_data.json`. Built-in rate-limit
handling (exponential backoff, per-URL cooldowns).

See [`download/README.md`](download/README.md).

### `preprocess/` — face-centered crop + resample

For each raw mp4: detect the primary speaker with InsightFace, compute a
stable head-centered crop, and resample to 512×512 / 25 fps video + 16 kHz
mono audio. Output matches the talkvid layout used by downstream training.

```bash
PYTHONPATH=. python scripts/preprocess/preprocess_talkvid_data.py
```

### `manifest/` — training manifest + train/val split

```bash
# Validate each clip has FLAME tracking, count frames.
python scripts/manifest/build_manifest.py
# Output: data/derived/manifest.json

# Identity-based 98/2 split (all clips from same speaker go to same split).
python scripts/manifest/partition_dataset.py
# Output: data/derived/train_clips.json, data/derived/val_clips.json
```

Clip IDs are parsed to extract the YouTube video ID across several naming
conventions (`ID_NA_timestamp`, `language_X_videovideo{VID}_scene*`, etc.)
so identity leakage is prevented across train/val.

### `paper_viz/` — paper figure rendering

Standalone matplotlib scripts that render the paper figures from already-computed
eval outputs (no training, no inference). Each script is independent — invoke directly:

- `build_comparison_figure.py` — Loki vs. SOTA grid (driver / Loki / 5 baselines × frames).
- `flame_substitution_diagram.py` — cross-identity FLAME-substitution method diagram.
- `generate_driver_map_example.py` — single-clip driver-map decomposition (RGB / wireframe / mesh / Δ_expr / posenc).
- `paper_teaser_figure.py` — Loki-only cross-identity teaser (two ref/driver pairs).

```bash
PYTHONPATH=. python scripts/paper_viz/build_comparison_figure.py --help
```

## Output layout

```
data/derived/
├── manifest.json
├── train_clips.json
└── val_clips.json
```

## Structure

```
scripts/
├── README.md
├── download/            # YouTube scraping
├── preprocess/          # face crop + resample
├── manifest/            # build_manifest + partition_dataset
└── paper_viz/           # paper figure rendering
```
