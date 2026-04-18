"""
Entry script for the `ablate_audio` experiment.

Two variants, both using Marigold-generated deformation (4ch) with uniform loss:

  with_audio  — wav2vec2 cross-attention ON  (full Marionette pipeline)
  no_audio    — wav2vec2 cross-attention OFF (spatial deformation only)

Tests whether explicit audio conditioning improves lip sync and expressiveness
beyond what the Marigold deformation signal already captures.

IMPORTANT — fair comparison via clip filtering:
    Uses the same Marigold-cache-based filtering as ablate_expr_source: the
    train/val splits are built from clips that have a cached deformation.mp4,
    split by identity (90/10) so no identity leaks across splits.

Usage (from repo root):

    PYTHONPATH=. python experiments/ablate_audio/run.py with_audio
    PYTHONPATH=. python experiments/ablate_audio/run.py no_audio
    PYTHONPATH=. python experiments/ablate_audio/run.py all

Outputs land in `outputs/ablate_audio/<variant>/run_<timestamp>/`.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from marionette.config_utils import load_experiment_config
from marionette.train import run_training


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "ablate_audio"

MARIGOLD_DEFORM_ROOT = REPO_ROOT / "data" / "derived" / "marigold_deform"

VARIANTS = ("with_audio", "no_audio")


def is_rank_zero() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ---------------------------------------------------------------------------
# Clip filtering (same logic as ablate_expr_source — identity-based split)
# ---------------------------------------------------------------------------

def _extract_video_id(clip_id: str) -> str:
    if "_NA_" in clip_id:
        return clip_id.split("_NA_")[0]
    m = re.search(r"videovideo(.+?)_scene", clip_id)
    return m.group(1) if m else clip_id


def get_marigold_cached_clip_ids(min_size_bytes: int = 1024) -> set[str]:
    if not MARIGOLD_DEFORM_ROOT.exists():
        return set()
    out = set()
    for d in MARIGOLD_DEFORM_ROOT.iterdir():
        if not d.is_dir():
            continue
        mp4 = d / "deformation.mp4"
        if mp4.exists() and mp4.stat().st_size >= min_size_bytes:
            out.add(d.name)
    return out


def write_filtered_clip_lists(
    cached_ids: set[str], val_ratio: float = 0.1, seed: int = 42,
) -> tuple[str, str]:
    import random

    filtered_dir = OUT_ROOT / "filtered_clips"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(filtered_dir / "train_clips_filtered.json")
    val_path = str(filtered_dir / "val_clips_filtered.json")

    id_to_clips: dict[str, list[str]] = defaultdict(list)
    for clip_id in sorted(cached_ids):
        id_to_clips[_extract_video_id(clip_id)].append(clip_id)

    identities = sorted(id_to_clips.keys())
    rng = random.Random(seed)
    rng.shuffle(identities)

    n_val_ids = max(1, int(len(identities) * val_ratio))
    val_ids = set(identities[:n_val_ids])
    train_ids = set(identities[n_val_ids:])

    train_clips = sorted(c for vid in train_ids for c in id_to_clips[vid])
    val_clips = sorted(c for vid in val_ids for c in id_to_clips[vid])

    with open(train_path, "w") as f:
        json.dump(train_clips, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_clips, f, indent=2)

    print(f"[filter] Marigold cache has {len(cached_ids)} clips across "
          f"{len(identities)} identities")
    print(f"[filter] Train: {len(train_clips)} clips / {len(train_ids)} identities")
    print(f"[filter] Val:   {len(val_clips)} clips / {len(val_ids)} identities")
    print(f"[filter] Saved to {filtered_dir}")

    return train_path, val_path


# ---------------------------------------------------------------------------
# Symlink helper
# ---------------------------------------------------------------------------

def ensure_outputs_symlink() -> None:
    link = HERE / "outputs"
    if link.exists() or link.is_symlink():
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    target = Path("..") / ".." / "outputs" / "ablate_audio"
    link.symlink_to(target, target_is_directory=True)
    print(f"[symlink] {link} -> {target}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(variant: str, train_clips: str, val_clips: str,
        resume: str | None = None, gpus: tuple[int, ...] = (0,)) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")

    cfg_path = HERE / "configs" / f"{variant}.yaml"
    cfg = load_experiment_config(cfg_path)

    cfg.train_dataset.params.clip_list_path = train_clips
    cfg.val_dataset.params.clip_list_path = val_clips

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_symlink()

    print(f"[run] variant={variant}  output={out_dir}")
    run_training(cfg=cfg, output_dir=str(out_dir), resume=resume, gpus=gpus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=VARIANTS + ("all",))
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to resume from (applies to the first variant).")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    filtered_dir = OUT_ROOT / "filtered_clips"
    train_clips = str(filtered_dir / "train_clips_filtered.json")
    val_clips = str(filtered_dir / "val_clips_filtered.json")

    if args.resume and Path(train_clips).exists() and Path(val_clips).exists():
        if is_rank_zero():
            print(f"[filter] Resuming — reusing existing clip lists from {filtered_dir}")
    elif is_rank_zero():
        cached_ids = get_marigold_cached_clip_ids()
        if not cached_ids:
            raise RuntimeError(
                f"No Marigold-cached clips found in {MARIGOLD_DEFORM_ROOT}. "
                f"Run scripts/cache/marigold_deform/cache.py first."
            )
        train_clips, val_clips = write_filtered_clip_lists(cached_ids)
    else:
        import time
        for _ in range(120):
            if Path(train_clips).exists() and Path(val_clips).exists():
                break
            time.sleep(0.5)

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for i, v in enumerate(variants):
        run(v, train_clips, val_clips,
            resume=(args.resume if i == 0 else None),
            gpus=tuple(args.gpus))


if __name__ == "__main__":
    main()
