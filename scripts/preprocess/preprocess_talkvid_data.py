"""
Preprocess downloaded TalkVid clips into training-ready format.

For each mp4 in the download directory (output of download_clips.py):
  1. Randomly sample a 5-second segment from the clip.
  2. Detect the speaker's face with InsightFace (RetinaFace).
  3. Compute a stable square crop centered on the face (same logic as the
     original TalkVid preprocessing: median x-center, topmost head + padding,
     face_height * CROP_SCALE).
  4. Crop, resize to 512x512, re-encode at 25fps.
  5. Extract audio as 16kHz mono PCM WAV (matching training pipeline).

Output layout (matches existing talkvid structure):
  - data/talkvid/talkvid/{clip_id}.mp4   (512x512, 25fps, ~5s)
  - data/talkvid/audio/{clip_id}.wav     (16kHz, mono, PCM s16le)

The clip_id is the source filename stem, e.g. {video_id}_NA_{start}_{end}.

Clips shorter than 5 seconds or with no detectable face are skipped.
Already-processed clips are skipped (resume-safe).

Usage:
    cd <repo_root>

    PYTHONPATH=. python scripts/preprocess/preprocess_talkvid_data.py

    # Test on a few clips:
    PYTHONPATH=. python scripts/preprocess/preprocess_talkvid_data.py --limit 10
"""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

CLIP_DURATION     = 5.0      # seconds to sample from each source clip
TARGET_FPS        = 25       # output video fps
OUTPUT_SIZE       = 512      # output resolution (square)
AUDIO_SAMPLE_RATE = 16000    # output audio sample rate (Hz)

SAMPLE_EVERY      = 10       # face detection sampling interval (frames within 5s window)
CROP_SCALE        = 1.6      # crop_size = face_height * CROP_SCALE
HEAD_PAD          = 0.15     # fractional padding above detected head top
MIN_DET_CONF      = 0.5      # minimum InsightFace detection confidence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/additional_data",
                   help="Download directory from download_clips.py")
    p.add_argument("--video_out", default="data/talkvid/talkvid",
                   help="Output directory for 25fps video clips")
    p.add_argument("--audio_out", default="data/talkvid/audio",
                   help="Output directory for audio files")
    p.add_argument("--duration", type=float, default=CLIP_DURATION,
                   help="Duration of sampled clip in seconds (default: 5.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many clips (for testing)")
    return p.parse_args()


# ============================================================================
# Face detection (InsightFace)
# ============================================================================

_face_app = None


def get_detector(gpu_ctx_id):
    """Lazy-load InsightFace FaceAnalysis for the given CUDA context."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=gpu_ctx_id, det_size=(640, 640))
    return _face_app


def detect_face(frame_bgr, app):
    """Return (x1, y1, x2, y2) of the largest confident face, or None."""
    faces = app.get(frame_bgr)
    if not faces:
        return None
    valid = [f for f in faces if f.det_score >= MIN_DET_CONF]
    if not valid:
        return None
    best = max(valid, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    b = best.bbox
    return int(b[0]), int(b[1]), int(b[2]), int(b[3])


# ============================================================================
# Stable crop computation (same logic as original TalkVid preprocessing)
# ============================================================================

def compute_stable_crop(boxes, frame_h, frame_w):
    """
    From sampled face boxes, compute a single stable square crop:
      - Horizontally centered on the speaker (median x-center)
      - Top-aligned to topmost detected head + HEAD_PAD above
      - Size = median(face_height) * CROP_SCALE
    Returns (crop_x1, crop_y1, crop_size) or None.
    """
    if not boxes:
        return None

    arr = np.array(boxes, dtype=float)
    face_heights = arr[:, 3] - arr[:, 1]
    x_centers = (arr[:, 0] + arr[:, 2]) / 2.0

    crop_size = int(np.median(face_heights) * CROP_SCALE)
    crop_size = max(crop_size, 256)
    crop_size = min(crop_size, min(frame_h, frame_w))

    head_pad = int(crop_size * HEAD_PAD)
    x_center = int(np.median(x_centers))
    y_top_head = int(np.percentile(arr[:, 1], 10))

    crop_x1 = x_center - crop_size // 2
    crop_y1 = y_top_head - head_pad

    crop_x1 = max(0, min(crop_x1, frame_w - crop_size))
    crop_y1 = max(0, min(crop_y1, frame_h - crop_size))

    return crop_x1, crop_y1, crop_size


# ============================================================================
# Helpers
# ============================================================================

def sanitize_clip_id(clip_id):
    """Strip leading dashes from clip_id to avoid filenames like '--foo.mp4'."""
    return clip_id.lstrip("-")


def get_video_duration(path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


# ============================================================================
# Single-clip processing
# ============================================================================

def process_clip(src_path, video_out_dir, audio_out_dir, duration, seed, app):
    """
    Process one clip:
      1. Pick a random 5s window
      2. Detect faces in that window to compute crop
      3. ffmpeg: crop + resize + 25fps -> video
      4. ffmpeg: extract 16kHz mono WAV -> audio
    Returns (clip_id, status_string).
    """
    clip_id = sanitize_clip_id(Path(src_path).stem)
    video_out = Path(video_out_dir) / f"{clip_id}.mp4"
    audio_out = Path(audio_out_dir) / f"{clip_id}.wav"

    # Skip if already processed
    if video_out.exists() and audio_out.exists():
        return clip_id, "skipped"

    # Get source duration
    src_duration = get_video_duration(src_path)
    if src_duration < duration:
        return clip_id, f"too_short ({src_duration:.1f}s)"

    # Deterministic random start time (seeded by clip_id for reproducibility)
    rng = random.Random(hash(clip_id) + seed)
    max_start = src_duration - duration
    start_time = rng.uniform(0, max_start)

    # Open video and seek to start of the 5s window for face detection
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return clip_id, "cannot_open"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)
    window_frames = int(duration * fps)

    # Sample frames within the 5s window for face detection
    sample_indices = list(range(0, window_frames, SAMPLE_EVERY))
    if not sample_indices:
        sample_indices = [0]

    face_boxes = []
    for offset in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
        ret, frame = cap.read()
        if not ret:
            continue
        box = detect_face(frame, app)
        if box is not None:
            face_boxes.append(box)

    cap.release()

    if not face_boxes:
        return clip_id, "no_face_detected"

    detection_rate = len(face_boxes) / len(sample_indices)
    if detection_rate < 0.2:
        return clip_id, f"low_detection_rate ({detection_rate:.0%})"

    # Compute stable crop
    crop = compute_stable_crop(face_boxes, frame_h, frame_w)
    if crop is None:
        return clip_id, "invalid_crop"

    crop_x1, crop_y1, crop_size = crop

    # ffmpeg: seek -> crop -> resize -> 25fps video (no audio)
    vf = (f"crop={crop_size}:{crop_size}:{crop_x1}:{crop_y1},"
          f"scale={OUTPUT_SIZE}:{OUTPUT_SIZE},"
          f"fps={TARGET_FPS}")

    video_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_time:.3f}",
        "-i", str(src_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(video_out),
    ]

    # ffmpeg: seek -> extract audio as 16kHz mono PCM WAV
    audio_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_time:.3f}",
        "-i", str(src_path),
        "-t", f"{duration:.3f}",
        "-vn",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(audio_out),
    ]

    try:
        subprocess.run(video_cmd, check=True, capture_output=True)
        subprocess.run(audio_cmd, check=True, capture_output=True)
        return clip_id, "ok"
    except subprocess.CalledProcessError as e:
        video_out.unlink(missing_ok=True)
        audio_out.unlink(missing_ok=True)
        stderr = e.stderr.decode() if e.stderr else str(e)
        return clip_id, f"ffmpeg_error: {stderr[:200]}"


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    video_out_dir = Path(args.video_out)
    audio_out_dir = Path(args.audio_out)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # Find all mp4 files
    all_mp4s = sorted(input_dir.rglob("*.mp4"))
    print(f"Found {len(all_mp4s)} mp4 files in {input_dir}")

    if args.limit:
        all_mp4s = all_mp4s[:args.limit]
        print(f"Limited to {len(all_mp4s)} clips")

    if not all_mp4s:
        print("Nothing to do.")
        return

    # Initialize face detector
    app = get_detector(0)

    n_ok = n_skip = n_fail = 0
    failures = []

    for src_path in tqdm(all_mp4s, desc="Processing"):
        clip_id, status = process_clip(
            str(src_path), str(video_out_dir), str(audio_out_dir),
            args.duration, args.seed, app,
        )
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skip += 1
        else:
            n_fail += 1
            failures.append((clip_id, status))

    print(f"\nDone. OK: {n_ok} | Skipped: {n_skip} | Failed: {n_fail}")
    print(f"Video output: {video_out_dir}")
    print(f"Audio output: {audio_out_dir}")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for clip_id, reason in failures[:20]:
            print(f"  {clip_id}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
