# Evaluation Metrics

Quantitative metrics for evaluating Marionette's generated talking-head videos.
All scripts are reusable across experiments.

## Lip Sync (LSE-D / LSE-C)

Measures audio-visual synchronisation using the pretrained SyncNet v2 model
(Chung & Zisserman, 2016). Three metrics are reported:

| Metric | Meaning | Better |
|---|---|---|
| **LSE-D** | Min avg L2 distance between audio and video embeddings across temporal offsets | Lower |
| **LSE-C** | Confidence = median distance − min distance (how distinguishable the best offset is) | Higher |
| **AV Offset** | Best-matching temporal offset in frames | Closer to 0 |

### Setup

```bash
# Install dependency
pip install python-speech-features

# Download pretrained SyncNet v2 weights (~55 MB)
mkdir -p data/weights/syncnet
wget -O data/weights/syncnet/syncnet_v2.model \
    https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model
```

### Single video

```bash
PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
    --video path/to/generated.mp4 \
    --audio path/to/source_audio.wav
```

### Batch mode (all videos in a directory)

```bash
PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
    --video_dir outputs/ablate_audio/with_audio/generated/ \
    --audio_dir data/talkvid/audio/ \
    --output results/with_audio_lse.json
```

Video files are matched to audio files by stem name (e.g. `CLIP_ID.mp4` ↔
`CLIP_ID.wav`).

### Comparing audio ablation variants

```bash
# With audio cross-attention
PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
    --video_dir outputs/ablate_audio/with_audio/generated/ \
    --audio_dir data/talkvid/audio/ \
    --output results/with_audio_lse.json

# Without audio cross-attention
PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
    --video_dir outputs/ablate_audio/no_audio/generated/ \
    --audio_dir data/talkvid/audio/ \
    --output results/no_audio_lse.json

# Then diff the aggregate mean_lse_d and mean_lse_c in the two JSONs.
```

### Pipeline

1. **Video**: decode frames → resize to 224×224 → sliding windows of 5 frames
2. **Audio**: load WAV → compute 13-coefficient MFCC at 100 Hz → sliding windows
   of 20 MFCC frames (aligned to 5 video frames at 25 FPS)
3. **Embeddings**: both streams through pretrained SyncNet v2 → 1024-dim each
4. **Metric**: L2 distance between video and audio embeddings at each temporal
   offset (±15 frames), then min (LSE-D) and median−min (LSE-C)

### Structure

```
experiments/evaluation_metrics/
├── README.md
├── compute_lip_sync.py           # CLI entry point
└── syncnet/
    ├── __init__.py
    ├── model.py                  # SyncNet v2 architecture + weight loading
    └── preprocess.py             # MFCC + frame extraction + window alignment
```
