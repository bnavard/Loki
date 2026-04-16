"""
Build training manifest from FLAME tracking data.

Scans all clips that have a fit.npz in the FLAME tracking directory and
produces a manifest JSON listing each clip_id, its audio file path, and
the number of frames.

Output: data/derived/manifest.json

Usage:
    cd <repo_root>
    python scripts/manifest/build_manifest.py
"""

import json
from pathlib import Path

import numpy as np

FLAME_ROOT = Path("data/flame_tracking/flowface")
AUDIO_DIR = Path("data/talkvid/audio")
MANIFEST_PATH = Path("data/derived/manifest.json")


def main():
    all_clips = sorted([
        d.name for d in FLAME_ROOT.iterdir()
        if d.is_dir() and (d / "fit.npz").exists()
    ])
    print(f"Total clips with fit.npz: {len(all_clips)}")

    manifest = []
    n_missing_audio = 0

    for clip_id in all_clips:
        audio_path = AUDIO_DIR / f"{clip_id}.wav"

        if not audio_path.exists():
            n_missing_audio += 1
            continue

        fit = np.load(str(FLAME_ROOT / clip_id / "fit.npz"))
        num_frames = fit["expr"].shape[0]

        manifest.append({
            "clip_id": clip_id,
            "audio_file": str(audio_path),
            "num_frames": int(num_frames),
        })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"  Valid clips:     {len(manifest)}")
    print(f"  Missing audio:   {n_missing_audio}")

    if manifest:
        durations = [e["num_frames"] / 25.0 for e in manifest]
        print(f"  Avg duration:    {sum(durations) / len(durations):.1f}s")
        print(f"  Total duration:  {sum(durations) / 3600:.1f}h")


if __name__ == "__main__":
    main()
