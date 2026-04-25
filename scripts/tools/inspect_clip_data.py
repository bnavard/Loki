r"""
Validate every data file the TalkingHeadDataset will touch, with a per-file
timeout so corrupt videos that hang decord (a likely cause of NCCL timeouts
during DDP training) are caught instead of stalling the inspector.

Checks per clip_id:
  1. data/flame_tracking/flowface/{clip}/fit.npz   — load with np.load
  2. data/talkvid/talkvid/{clip}.mp4               — open + decode 4 sample frames
  3. data/talkvid/audio/{clip}.wav                 — load full waveform

Each check runs in a worker process; the pool kills the worker if it exceeds
`--timeout` seconds. Bad clips are reported by reason and written to a JSON
file for later use (e.g. removing them from the filtered clip lists).

Usage (from repo root):

    PYTHONPATH=. python scripts/tools/inspect_clip_data.py \
        --clip_lists data/derived/train_clips.json data/derived/val_clips.json \
        --workers 16 \
        --timeout 30 \
        --output data/derived/bad_clips.json
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def _check_clip(args):
    """Return (clip_id, status, details).

    status: 'ok' or 'bad'
    details: dict with per-file results when status == 'bad'
    """
    clip_id, paths, n_sample_frames, check_audio = args
    failures = {}

    # 1. FLAME fit
    fit_path = paths["flame_root"] / clip_id / "fit.npz"
    try:
        if not fit_path.exists():
            failures["fit_npz"] = f"missing: {fit_path}"
        else:
            import numpy as np
            data = dict(np.load(str(fit_path)))
            for key in ("expr", "rot", "tra", "shape"):
                if key not in data:
                    failures["fit_npz"] = f"missing key {key!r}"
                    break
            else:
                if data["expr"].shape[0] < 1:
                    failures["fit_npz"] = "empty expr array"
    except Exception as e:
        failures["fit_npz"] = f"{type(e).__name__}: {e}"

    # 2. Source video
    video_path = paths["video_root"] / f"{clip_id}.mp4"
    try:
        if not video_path.exists():
            failures["video_mp4"] = f"missing: {video_path}"
        else:
            from decord import VideoReader
            vr = VideoReader(str(video_path))
            n = len(vr)
            if n < 1:
                failures["video_mp4"] = "0 frames"
            else:
                indices = [int(i) for i in [0, n // 3, 2 * n // 3, n - 1]]
                for idx in indices:
                    f = vr[idx]
                    if f is None:
                        failures["video_mp4"] = f"None frame at idx {idx}"
                        break
    except Exception as e:
        failures["video_mp4"] = f"{type(e).__name__}: {e}"

    # 3. Audio
    if check_audio:
        audio_path = paths["audio_root"] / f"{clip_id}.wav"
        try:
            if not audio_path.exists():
                failures["audio_wav"] = f"missing: {audio_path}"
            else:
                import soundfile as sf
                arr, _ = sf.read(str(audio_path), dtype="float32", always_2d=False)
                if arr.size == 0:
                    failures["audio_wav"] = "empty audio"
        except Exception as e:
            failures["audio_wav"] = f"{type(e).__name__}: {e}"

    if failures:
        return clip_id, "bad", failures
    return clip_id, "ok", {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clip_lists", nargs="+", required=True,
        help="One or more JSON files containing clip_id lists to inspect.",
    )
    p.add_argument("--video_root", default="data/talkvid/talkvid")
    p.add_argument("--audio_root", default="data/talkvid/audio")
    p.add_argument("--flame_root", default="data/flame_tracking/flowface")
    p.add_argument(
        "--no_check_audio", action="store_true",
        help="Skip audio validation (audio files are large; saves time).",
    )
    p.add_argument(
        "--n_sample_frames", type=int, default=4,
        help="Number of frames to decode per video (head / two middle / tail).",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Parallel worker processes. Higher = faster but more I/O contention.",
    )
    p.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-clip timeout in seconds. A hung clip beyond this is flagged "
             "as 'bad: timeout' and the worker is killed + replaced.",
    )
    p.add_argument(
        "--output", default=None,
        help="JSON path to write the list of bad clips and their failure reasons. "
             "If omitted, only prints the summary.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    seen = set()
    clip_ids = []
    for path in args.clip_lists:
        with open(path) as f:
            ids = json.load(f)
        new = [c for c in ids if c not in seen]
        for c in new:
            seen.add(c)
        clip_ids.extend(new)
        print(f"[load] {path}: {len(ids)} clips ({len(new)} new)")

    print(f"[scan] inspecting {len(clip_ids)} unique clips with {args.workers} workers, "
          f"per-clip timeout = {args.timeout}s")
    print(f"[scan] checks: fit + source mp4"
          f"{' + audio' if not args.no_check_audio else ''}")

    paths = {
        "video_root": Path(args.video_root),
        "audio_root": Path(args.audio_root),
        "flame_root": Path(args.flame_root),
    }

    work_items = [
        (cid, paths, args.n_sample_frames, not args.no_check_audio)
        for cid in clip_ids
    ]

    bad: dict[str, dict] = {}
    timeouts: list[str] = []
    n_ok = 0
    started_at = time.monotonic()
    last_report = started_at

    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_check_clip, w): w[0] for w in work_items}
        for fut in cf.as_completed(futures):
            cid = futures[fut]
            try:
                _, status, details = fut.result(timeout=args.timeout)
                if status == "ok":
                    n_ok += 1
                else:
                    bad[cid] = details
            except cf.TimeoutError:
                timeouts.append(cid)
                bad[cid] = {"timeout": f">{args.timeout}s"}
                fut.cancel()
            except Exception as e:
                bad[cid] = {"worker_error": f"{type(e).__name__}: {e}"}

            done = n_ok + len(bad)
            now = time.monotonic()
            if now - last_report >= 5.0 or done == len(clip_ids):
                rate = done / max(now - started_at, 1e-3)
                eta = (len(clip_ids) - done) / max(rate, 1e-3)
                print(f"[progress] {done}/{len(clip_ids)} done  "
                      f"({n_ok} ok, {len(bad)} bad)  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s",
                      flush=True)
                last_report = now

    print()
    print("=" * 72)
    print(f"Scanned {len(clip_ids)} clips in {time.monotonic() - started_at:.1f}s")
    print(f"  OK:       {n_ok}")
    print(f"  Bad:      {len(bad)}")
    print(f"  Timeouts: {len(timeouts)}")
    print()

    if bad:
        by_reason = defaultdict(list)
        for cid, details in bad.items():
            for fkey in details.keys():
                by_reason[fkey].append(cid)
        print("Failures grouped by file:")
        for reason, ids in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{reason}]  {len(ids)} clips")
            for cid in ids[:5]:
                print(f"      {cid}: {bad[cid][reason]}")
            if len(ids) > 5:
                print(f"      ... and {len(ids) - 5} more")
        print()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "n_total":     len(clip_ids),
                    "n_ok":        n_ok,
                    "n_bad":       len(bad),
                    "n_timeout":   len(timeouts),
                    "bad":         bad,
                    "timeout_ids": timeouts,
                },
                f, indent=2,
            )
        print(f"Wrote bad-clip report to {args.output}")

    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
