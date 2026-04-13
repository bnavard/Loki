"""
Step 1.3: Build training manifest from derived data.

Scans all clips that have both FLAME tracking (fit.npz) and captions,
and produces a manifest JSON. Expression fields are computed on the fly
during training, so only captions need to be precomputed.

Output: data/derived/manifest.json

Usage:
    cd <repo_root>
    python scripts/manifest/build_manifest.py
"""

import json
from pathlib import Path

import numpy as np

FLOWFACE_DIR = Path("data/flowface")
AUDIO_DIR = Path("data/talkvid/audio")
CAPTIONS_DIR = Path("data/derived/captions")
MANIFEST_PATH = Path("data/derived/manifest.json")


def main():
    all_flowface = sorted([
        d.name for d in FLOWFACE_DIR.iterdir()
        if d.is_dir() and (d / "fit.npz").exists()
    ])
    print(f"Total clips with fit.npz: {len(all_flowface)}")

    manifest = []
    n_missing_audio = 0
    n_missing_caption = 0

    for clip_id in all_flowface:
        audio_path = AUDIO_DIR / f"{clip_id}.wav"
        caption_path = CAPTIONS_DIR / f"{clip_id}.json"

        if not audio_path.exists():
            n_missing_audio += 1
            continue
        if not caption_path.exists():
            n_missing_caption += 1
            continue

        fit = np.load(str(FLOWFACE_DIR / clip_id / "fit.npz"))
        num_frames = fit["expr"].shape[0]

        manifest.append({
            "clip_id": clip_id,
            "caption_file": str(caption_path),
            "audio_file": str(audio_path),
            "num_frames": int(num_frames),
        })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"  Valid clips:     {len(manifest)}")
    print(f"  Missing audio:   {n_missing_audio}")
    print(f"  Missing caption: {n_missing_caption}")

    if manifest:
        durations = [e["num_frames"] / 25.0 for e in manifest]
        print(f"  Avg duration:    {sum(durations) / len(durations):.1f}s")
        print(f"  Total duration:  {sum(durations) / 3600:.1f}h")


if __name__ == "__main__":
    main()
