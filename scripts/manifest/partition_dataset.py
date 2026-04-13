"""
Partition the dataset into train/val splits by unique video identity.

Extracts the YouTube video ID from each clip_id, groups all clips from the
same video together, then assigns 98% of identities to train and 2% to val.
All clips from a given identity go to the same split — no identity leakage.

Clip ID formats handled:
  - VIDEOID_NA_start_end           (e.g. 39Y_gFC9SmY_NA_1123.760_1128.801)
  - prefix_videovideo{VID}_scene*  (e.g. language_German_videovideowNgpzygdGeU_scene2_scene2)

Output:
  data/derived/train_clips.json    — list of clip_ids for training
  data/derived/val_clips.json      — list of clip_ids for validation

Usage:
    cd <repo_root>
    python scripts/manifest/partition_dataset.py
    python scripts/manifest/partition_dataset.py --val_ratio 0.05  # 5% validation
    python scripts/manifest/partition_dataset.py --seed 42
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def extract_video_id(clip_id):
    """
    Extract the YouTube video ID from any clip_id format.

    Handles:
      - VIDEOID_NA_start_end
      - language_X_videovideo{VID}_sceneN_sceneM
      - gender_X_videovideo{VID}_sceneN_sceneM
      - age_X_videovideo{VID}_sceneN_sceneM
    """
    if "_NA_" in clip_id:
        return clip_id.split("_NA_")[0]
    m = re.search(r"videovideo(.+?)_scene", clip_id)
    if m:
        return m.group(1)
    return clip_id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/derived/manifest.json")
    p.add_argument("--output_dir", default="data/derived")
    p.add_argument("--val_ratio", type=float, default=0.02, help="Fraction of identities for validation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Group clips by video identity
    id_to_clips = defaultdict(list)
    for entry in manifest:
        vid_id = extract_video_id(entry["clip_id"])
        id_to_clips[vid_id].append(entry["clip_id"])

    all_ids = sorted(id_to_clips.keys())
    n_val = max(1, round(len(all_ids) * args.val_ratio))

    # Deterministic shuffle
    random.seed(args.seed)
    random.shuffle(all_ids)

    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    train_clips = []
    val_clips = []
    for vid_id, clips in id_to_clips.items():
        if vid_id in val_ids:
            val_clips.extend(clips)
        else:
            train_clips.extend(clips)

    train_clips.sort()
    val_clips.sort()

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_clips.json", "w") as f:
        json.dump(train_clips, f, indent=2)
    with open(output_dir / "val_clips.json", "w") as f:
        json.dump(val_clips, f, indent=2)

    print(f"Manifest: {len(manifest)} clips, {len(all_ids)} unique identities")
    print(f"Train: {len(train_ids)} identities, {len(train_clips)} clips")
    print(f"Val:   {len(val_ids)} identities, {len(val_clips)} clips")
    print(f"Saved: {output_dir}/train_clips.json, {output_dir}/val_clips.json")

    # Show val identities for reference
    print(f"\nVal identities ({len(val_ids)}):")
    for vid_id in sorted(val_ids):
        print(f"  {vid_id} ({len(id_to_clips[vid_id])} clips)")


if __name__ == "__main__":
    main()
