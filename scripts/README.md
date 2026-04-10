# Data Preparation Scripts

Shared scripts for preparing and organizing the dataset used across all training pipelines.

## Scripts

### 1. Generate Text Captions

Combines Whisper large-v3 ASR transcription with Qwen2-Audio prosody description. Each caption follows the format: `A person says: '<transcription>' <prosody description>`.

```bash
# Parallel across GPUs:
PYTHONPATH=. python scripts/generate_captions.py --gpu 0 --num_gpus 8

# ASR only (skip Qwen2-Audio):
PYTHONPATH=. python scripts/generate_captions.py --asr_only

# Output: data/derived/captions/{clip_id}.json
```

### 2. Build Training Manifest

Validates that each clip has FLAME tracking + caption, counts frames.

```bash
python scripts/build_manifest.py
# Output: data/derived/manifest.json
```

### 3. Partition Dataset (Train/Val Split)

Splits the dataset by unique video identity — 98% train, 2% val. All clips from the same person go to the same split to prevent identity leakage.

```bash
python scripts/partition_dataset.py
# Output: data/derived/train_clips.json, data/derived/val_clips.json
```

Clip IDs are parsed to extract the YouTube video ID regardless of format (`ID_NA_timestamp`, `language_X_videovideo{VID}_scene*`, `gender_X_videovideo{VID}_scene*`, etc.).

## Structure

```
scripts/
├── generate_captions.py       # Whisper ASR + Qwen2-Audio prosody → captions
├── build_manifest.py          # Validate data + build manifest.json
├── partition_dataset.py       # Train/val split by identity
└── README.md
```

## Output Layout

```
data/derived/
├── captions/{clip_id}.json    # text captions
├── manifest.json              # full training manifest
├── train_clips.json           # training split clip IDs
└── val_clips.json             # validation split clip IDs
```
