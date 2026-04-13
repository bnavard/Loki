# TalkVid Data Download

Downloads talking-head video clips from YouTube using the [TalkVid](https://github.com/FreedomIntelligence/TalkVid) dataset. TalkVid provides ~100k annotated video segments of people speaking to camera, with per-clip metadata (start/end timestamps, language, gender, age).

## Setup

```bash
pip install -r scripts/download/requirements.txt
```

Requires `ffmpeg` on PATH (used by yt-dlp for segment cutting).

## Pipeline

### Step 1: Download the TalkVid metadata

Download the original segment annotations from HuggingFace:

```bash
wget https://huggingface.co/datasets/FreedomIntelligence/TalkVid/resolve/main/data/filtered_video_clips.json
```

### Step 2: Filter for URL coverage (optional)

The full dataset has many segments per video. To maximize the number of unique videos downloaded (and reduce redundancy), filter to at most N segments per URL:

```bash
python scripts/download/filter_max_per_url.py \
    --input filtered_video_clips.json \
    --output scripts/download/talkvid_data.json \
    --max-per-url 2
```

The included `talkvid_data.json` was generated this way with `--max-per-url 2`.

### Step 3: Download clips

```bash
python scripts/download/download_clips.py \
    --input scripts/download/talkvid_data.json \
    --output data/talkvid/talkvid

# With browser cookies (recommended to avoid YouTube rate limits):
python scripts/download/download_clips.py \
    --input scripts/download/talkvid_data.json \
    --output data/talkvid/talkvid \
    --browser chrome

# Test with a small batch:
python scripts/download/download_clips.py \
    --input scripts/download/talkvid_data.json \
    --output data/talkvid/talkvid \
    --limit 10
```

The download script:
- Groups segments by YouTube URL and downloads all segments from the same video in a single yt-dlp call
- Saves per-clip JSON logs for resume — already-downloaded clips are skipped on re-run
- Detects YouTube rate limiting (HTTP 429, bot checks) and applies exponential backoff
- Pauses between URLs with random jitter and periodic batch cooldowns
- Probes each URL before downloading to skip permanently unavailable videos
- Saves video (mp4), audio (m4a), subtitles, and description files

### Rate-limit mitigation parameters

| Parameter | Default | Description |
|---|---|---|
| `--delay` | 5.0s | Base wait between URL downloads |
| `--jitter` | 3.0s | Random +/- added to delay |
| `--batch-size` | 50 | Pause after this many URLs |
| `--cooldown` | 120s | Sleep duration between batches |
| `--max-backoff` | 600s | Max exponential backoff on rate limit |

## Output Structure

```
data/talkvid/talkvid/
├── {video_id}/
│   ├── {video_id}_001_{start}_{end}.mp4    # video clip
│   ├── {video_id}_001_{start}_{end}.m4a    # audio clip
│   ├── {video_id}.description              # video description
│   └── {video_id}.en.vtt                   # subtitles (if available)
├── json_logs/
│   └── {video_id}_{start}_{end}.json       # per-clip download log (for resume)
└── logs/
    ├── failed_urls.txt                     # permanently unavailable URLs
    └── failed_segments.txt                 # failed/skipped segments
```

## File Structure

```
scripts/download/
├── download_clips.py        # Main download script (yt-dlp + rate-limit handling)
├── filter_max_per_url.py    # Filter to N segments per URL for coverage
├── talkvid_data.json        # Pre-filtered dataset (max 2 segments per URL, gitignored)
├── requirements.txt         # Python dependencies
└── README.md
```
