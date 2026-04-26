"""
Compute LSE-D (Lip Sync Error — Distance) and LSE-C (Lip Sync Error —
Confidence) for generated talking-head videos using the pretrained SyncNet v2.

Metrics (Prajwal et al., "A Lip Sync Expert Is All You Need", 2020):
  LSE-D: minimum average L2 distance between audio and video embeddings
         across temporal offsets. LOWER = better sync.
  LSE-C: difference between median and minimum distance across offsets.
         HIGHER = better sync (the correct alignment is clearly distinguishable).
  AV Offset: temporal offset in frames at which the best sync occurs.
         CLOSER TO 0 = better.

Usage (from repo root):

    # Single video:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
        --video outputs/marionette_baseline/run_<ts>/visualizations/step_<step>/sample_00.mp4 \
        --audio data/talkvid/audio/CLIP_ID.wav \
        --syncnet_weights data/weights/syncnet/syncnet_v2.model

    # Batch mode — evaluate all .mp4 files in a directory, paired with audio:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
        --video_dir outputs/marionette_baseline/run_<ts>/visualizations/step_<step>/ \
        --audio_dir data/talkvid/audio/ \
        --syncnet_weights data/weights/syncnet/syncnet_v2.model \
        --output results_lip_sync.json

    # Compare audio-on (marionette_baseline) vs audio-off
    # (condition_ablation/audio_off):
    PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
        --video_dir outputs/marionette_baseline/run_<ts>/visualizations/step_<step>/ \
        --audio_dir data/talkvid/audio/ \
        --syncnet_weights data/weights/syncnet/syncnet_v2.model \
        --output audio_on_lse.json

    PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
        --video_dir outputs/condition_ablation/audio_off/run_<ts>/visualizations/step_<step>/ \
        --audio_dir data/talkvid/audio/ \
        --syncnet_weights data/weights/syncnet/syncnet_v2.model \
        --output audio_off_lse.json

    # Or evaluate a SOTA baseline's panels — same script, point at the
    # baseline's samples/ tree:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_lip_sync.py \
        --video_dir outputs/sota_comparison/sadtalker/talkvid/cross_identity/run_<ts>/samples/ \
        --audio_dir data/talkvid/audio/ \
        --syncnet_weights data/weights/syncnet/syncnet_v2.model \
        --output sadtalker_talkvid_cross_lse.json

Pretrained weights download:
    mkdir -p data/weights/syncnet
    wget -O data/weights/syncnet/syncnet_v2.model \
        https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.evaluation_metrics.syncnet.model import SyncNetV2, load_syncnet_v2
from experiments.evaluation_metrics.syncnet.preprocess import (
    extract_video_frames,
    compute_mfcc,
    build_syncnet_windows,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: SyncNetV2,
    video_frames: np.ndarray,
    mfcc: np.ndarray,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract SyncNet embeddings for all aligned windows.

    Returns:
        video_embeddings: (N_windows, 1024)
        audio_embeddings: (N_windows, 1024)
    """
    video_embs = []
    audio_embs = []

    batch_v = []
    batch_a = []

    for v_win, a_win in build_syncnet_windows(video_frames, mfcc):
        batch_v.append(v_win)
        batch_a.append(a_win)

        if len(batch_v) == batch_size:
            v_tensor = torch.tensor(np.stack(batch_v), dtype=torch.float32, device=device)
            a_tensor = torch.tensor(np.stack(batch_a), dtype=torch.float32, device=device)
            video_embs.append(model.forward_video(v_tensor).cpu())
            audio_embs.append(model.forward_audio(a_tensor).cpu())
            batch_v.clear()
            batch_a.clear()

    # Flush remaining
    if batch_v:
        v_tensor = torch.tensor(np.stack(batch_v), dtype=torch.float32, device=device)
        a_tensor = torch.tensor(np.stack(batch_a), dtype=torch.float32, device=device)
        video_embs.append(model.forward_video(v_tensor).cpu())
        audio_embs.append(model.forward_audio(a_tensor).cpu())

    if not video_embs:
        return torch.zeros(0, 1024), torch.zeros(0, 1024)

    return torch.cat(video_embs, dim=0), torch.cat(audio_embs, dim=0)


def compute_lse_metrics(
    video_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    vshift: int = 15,
) -> dict:
    """Compute LSE-D, LSE-C, and AV offset from pre-extracted embeddings.

    Args:
        video_embs: (N, 1024)
        audio_embs: (N, 1024)
        vshift: number of frames to search in each direction (default 15 → 31 offsets)

    Returns:
        dict with keys: lse_d, lse_c, av_offset, n_windows
    """
    n = video_embs.shape[0]
    if n < 1:
        return {"lse_d": float("nan"), "lse_c": float("nan"),
                "av_offset": 0, "n_windows": 0}

    win_size = 2 * vshift + 1

    # Pad audio embeddings to allow temporal shifting
    audio_padded = F.pad(audio_embs, (0, 0, vshift, vshift))  # (N + 2*vshift, 1024)

    # Compute L2 distance at each offset for each window position
    dists = []
    for i in range(n):
        v_repeated = video_embs[i:i + 1].expand(win_size, -1)      # (win_size, 1024)
        a_shifted = audio_padded[i:i + win_size]                     # (win_size, 1024)
        d = F.pairwise_distance(v_repeated, a_shifted)               # (win_size,)
        dists.append(d)

    dists = torch.stack(dists, dim=1)     # (win_size, N)
    mean_dist = dists.mean(dim=1)         # (win_size,) — average distance at each offset

    min_val, min_idx = mean_dist.min(dim=0)
    median_val = mean_dist.median()

    lse_d = min_val.item()
    lse_c = (median_val - min_val).item()
    av_offset = (vshift - min_idx).item()

    return {
        "lse_d": round(lse_d, 4),
        "lse_c": round(lse_c, 4),
        "av_offset": int(av_offset),
        "n_windows": n,
    }


def evaluate_single(
    model: SyncNetV2,
    video_path: str,
    audio_path: str,
    vshift: int = 15,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Full pipeline for one (video, audio) pair."""
    video_frames = extract_video_frames(video_path, target_size=224)
    mfcc = compute_mfcc(audio_path)

    video_embs, audio_embs = extract_embeddings(
        model, video_frames, mfcc, batch_size=batch_size, device=device,
    )

    metrics = compute_lse_metrics(video_embs, audio_embs, vshift=vshift)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute LSE-D / LSE-C lip-sync metrics")

    # Single-pair mode
    p.add_argument("--video", default=None, help="Path to a single generated video")
    p.add_argument("--audio", default=None, help="Path to the corresponding audio file")

    # Batch mode
    p.add_argument("--video_dir", default=None,
                   help="Directory of generated .mp4 videos. Each video is "
                        "matched to an audio file with the same stem in --audio_dir.")
    p.add_argument("--audio_dir", default=None,
                   help="Directory of .wav audio files (matched by stem to video files)")

    # Model
    p.add_argument("--syncnet_weights", default="data/weights/syncnet/syncnet_v2.model",
                   help="Path to syncnet_v2.model pretrained weights")

    # Output
    p.add_argument("--output", default=None,
                   help="JSON path to write per-video and aggregate metrics")

    # Params
    p.add_argument("--vshift", type=int, default=15,
                   help="Temporal offset search range in frames (default 15 → 31 offsets)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")

    return p.parse_args()


def main():
    args = parse_args()

    # Validate args
    if args.video is None and args.video_dir is None:
        raise ValueError("Provide either --video (single) or --video_dir (batch)")

    # Load model
    weights = Path(args.syncnet_weights)
    if not weights.exists():
        print(f"Pretrained weights not found at {weights}.")
        print("Download with:")
        print(f"  mkdir -p {weights.parent}")
        print(f"  wget -O {weights} "
              "https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model")
        return
    model = load_syncnet_v2(str(weights), device=args.device)
    print(f"[syncnet] Loaded weights from {weights}")

    # Build (video, audio) pairs
    pairs = []
    if args.video is not None:
        if args.audio is None:
            raise ValueError("--audio is required with --video")
        pairs.append((args.video, args.audio))
    else:
        video_dir = Path(args.video_dir)
        audio_dir = Path(args.audio_dir) if args.audio_dir else None
        if audio_dir is None:
            raise ValueError("--audio_dir is required with --video_dir")
        for vf in sorted(video_dir.glob("*.mp4")):
            af = audio_dir / f"{vf.stem}.wav"
            if af.exists():
                pairs.append((str(vf), str(af)))
            else:
                print(f"[skip] No audio match for {vf.name}")

    if not pairs:
        print("No (video, audio) pairs found.")
        return

    print(f"[eval] {len(pairs)} pairs, vshift={args.vshift}")

    # Evaluate
    results = []
    all_lse_d = []
    all_lse_c = []
    all_offset = []

    for i, (vp, ap) in enumerate(pairs):
        try:
            m = evaluate_single(
                model, vp, ap,
                vshift=args.vshift, batch_size=args.batch_size, device=args.device,
            )
            m["video"] = str(vp)
            m["audio"] = str(ap)
            results.append(m)

            if not np.isnan(m["lse_d"]):
                all_lse_d.append(m["lse_d"])
                all_lse_c.append(m["lse_c"])
                all_offset.append(m["av_offset"])

            print(f"  [{i + 1}/{len(pairs)}] {Path(vp).name}  "
                  f"LSE-D={m['lse_d']:.4f}  LSE-C={m['lse_c']:.4f}  "
                  f"offset={m['av_offset']}  windows={m['n_windows']}")
        except Exception as e:
            print(f"  [{i + 1}/{len(pairs)}] {Path(vp).name}  FAILED: {e}")
            results.append({"video": str(vp), "audio": str(ap), "error": str(e)})

    # Aggregate
    agg = {}
    if all_lse_d:
        agg = {
            "mean_lse_d":   round(np.mean(all_lse_d), 4),
            "std_lse_d":    round(np.std(all_lse_d), 4),
            "mean_lse_c":   round(np.mean(all_lse_c), 4),
            "std_lse_c":    round(np.std(all_lse_c), 4),
            "mean_offset":  round(np.mean(all_offset), 2),
            "n_evaluated":  len(all_lse_d),
            "n_total":      len(pairs),
        }

    print()
    print("=" * 60)
    print("Aggregate:")
    print(f"  LSE-D:     {agg.get('mean_lse_d', 'N/A')} +/- {agg.get('std_lse_d', 'N/A')}   (lower = better)")
    print(f"  LSE-C:     {agg.get('mean_lse_c', 'N/A')} +/- {agg.get('std_lse_c', 'N/A')}   (higher = better)")
    print(f"  AV Offset: {agg.get('mean_offset', 'N/A')} frames")
    print(f"  Evaluated: {agg.get('n_evaluated', 0)} / {agg.get('n_total', 0)}")

    # Save
    if args.output:
        out = {"aggregate": agg, "per_video": results}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
