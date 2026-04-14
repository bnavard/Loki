"""
Entry script for the `ablate_expr_source` experiment.

Two variants: `gt_baseline` and `marigold`. Both use the same deform-only
architecture and uniform loss — the only difference is where the 3ch
deformation map comes from (FLAME rasterization vs Marigold-generated mp4).

IMPORTANT — fair comparison via clip filtering:
    Not all training clips have Marigold-cached deformations yet
    (data/derived/marigold_deform/ may be a subset of the full manifest).
    To ensure both variants train on exactly the same data, this script
    intersects the train/val clip lists with the clips that have a
    cached `deformation.mp4` and writes the filtered lists to the
    experiment output directory. Both variants use these filtered lists.

Usage (from repo root):

    PYTHONPATH=. python experiments/ablate_expr_source/run.py gt_baseline
    PYTHONPATH=. python experiments/ablate_expr_source/run.py marigold
    PYTHONPATH=. python experiments/ablate_expr_source/run.py both    # runs in sequence

Outputs land in `outputs/ablate_expr_source/<variant>/run_<timestamp>/`, and a
lazy symlink at `experiments/ablate_expr_source/outputs` points there on first
run so you can inspect artifacts without leaving the experiment folder.
"""

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from marionette.config_utils import load_experiment_config
from marionette.train import run_training


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "ablate_expr_source"

MARIGOLD_DEFORM_ROOT = REPO_ROOT / "data" / "derived" / "marigold_deform"

VARIANTS = ("gt_baseline", "marigold")


def is_rank_zero() -> bool:
    """True when running outside DDP or on the rank-0 process."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ---------------------------------------------------------------------------
# Clip filtering
# ---------------------------------------------------------------------------

def get_marigold_cached_clip_ids() -> set[str]:
    """Return the set of clip_ids that have a cached deformation.mp4."""
    if not MARIGOLD_DEFORM_ROOT.exists():
        return set()
    return {
        d.name
        for d in MARIGOLD_DEFORM_ROOT.iterdir()
        if d.is_dir() and (d / "deformation.mp4").exists()
    }


def filter_clip_list(clip_list_path: str, allowed: set[str]) -> list[str]:
    """Load a JSON clip list and return only the IDs present in `allowed`."""
    with open(clip_list_path) as f:
        all_ids = json.load(f)
    return [cid for cid in all_ids if cid in allowed]


def write_filtered_clip_lists(
    cached_ids: set[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[str, str]:
    """Build train/val splits from the set of Marigold-cached clip IDs.

    Instead of filtering the original project-wide train/val JSON files (which
    may not overlap with the cached subset at all), we treat the cached clips
    as the whole dataset and split them directly:

        train = first (1 - val_ratio) fraction
        val   = remaining val_ratio fraction

    The split is seeded and deterministic. Both filtered lists are written to
    `outputs/ablate_expr_source/filtered_clips/`.

    Returns (train_filtered_path, val_filtered_path).
    """
    import random

    filtered_dir = OUT_ROOT / "filtered_clips"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(filtered_dir / "train_clips_filtered.json")
    val_path = str(filtered_dir / "val_clips_filtered.json")

    all_clips = sorted(cached_ids)
    rng = random.Random(seed)
    rng.shuffle(all_clips)

    n_val = max(1, int(len(all_clips) * val_ratio))
    val_clips = all_clips[:n_val]
    train_clips = all_clips[n_val:]

    with open(train_path, "w") as f:
        json.dump(train_clips, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_clips, f, indent=2)

    print(f"[filter] Marigold cache has {len(all_clips)} clips total")
    print(f"[filter] Train: {len(train_clips)} clips ({100 * (1 - val_ratio):.0f}%)")
    print(f"[filter] Val:   {len(val_clips)} clips ({100 * val_ratio:.0f}%)")
    print(f"[filter] Saved to {filtered_dir}")

    return train_path, val_path


# ---------------------------------------------------------------------------
# Symlink helper
# ---------------------------------------------------------------------------

def ensure_outputs_symlink() -> None:
    """Create the `experiments/ablate_expr_source/outputs` symlink lazily."""
    link = HERE / "outputs"
    if link.exists() or link.is_symlink():
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Relative symlink so the link stays valid if the repo moves.
    target = Path("..") / ".." / "outputs" / "ablate_expr_source"
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

    # Override clip lists so both variants train on the same filtered subset.
    cfg.train_dataset.params.clip_list_path = train_clips
    cfg.val_dataset.params.clip_list_path = val_clips

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_symlink()

    print(f"[run] variant={variant}  output={out_dir}")
    run_training(cfg=cfg, output_dir=str(out_dir), resume=resume, gpus=gpus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=VARIANTS + ("both",))
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to resume from (applies to the first variant).")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    # Filter train/val clips to the intersection with the Marigold cache.
    # This is done once and reused for all variants so both see exactly the
    # same data — a prerequisite for a fair source-ablation comparison.
    #
    # In DDP only rank 0 writes the filtered lists; other ranks wait for
    # the files to appear before proceeding.
    filtered_dir = OUT_ROOT / "filtered_clips"
    train_clips = str(filtered_dir / "train_clips_filtered.json")
    val_clips = str(filtered_dir / "val_clips_filtered.json")

    if is_rank_zero():
        cached_ids = get_marigold_cached_clip_ids()
        if not cached_ids:
            raise RuntimeError(
                f"No Marigold-cached clips found in {MARIGOLD_DEFORM_ROOT}. "
                f"Run scripts/cache/cache_marigold_deform.py first."
            )
        train_clips, val_clips = write_filtered_clip_lists(cached_ids)
    else:
        # Wait for rank 0 to finish writing the filtered lists.
        import time
        for _ in range(120):
            if Path(train_clips).exists() and Path(val_clips).exists():
                break
            time.sleep(0.5)

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    for i, v in enumerate(variants):
        run(v, train_clips, val_clips,
            resume=(args.resume if i == 0 else None),
            gpus=tuple(args.gpus))


if __name__ == "__main__":
    main()