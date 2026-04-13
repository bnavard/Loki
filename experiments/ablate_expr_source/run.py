"""
Entry script for the `ablate_expr_source` experiment.

Two variants: `gt_baseline` and `marigold`. Both use the same deform-only
architecture and uniform loss — the only difference is where the 3ch
deformation map comes from (FLAME rasterization vs Marigold-generated mp4).

Usage (from repo root):

    PYTHONPATH=. python experiments/ablate_expr_source/run.py gt_baseline
    PYTHONPATH=. python experiments/ablate_expr_source/run.py marigold
    PYTHONPATH=. python experiments/ablate_expr_source/run.py both    # runs in sequence

Outputs land in `outputs/ablate_expr_source/<variant>/run_<timestamp>/`, and a
lazy symlink at `experiments/ablate_expr_source/outputs` points there on first
run so you can inspect artifacts without leaving the experiment folder.
"""

import argparse
from pathlib import Path

from marionette.config_utils import load_experiment_config
from marionette.train import run_training


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "ablate_expr_source"

VARIANTS = ("gt_baseline", "marigold")


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


def run(variant: str, resume: str | None = None, gpus: tuple[int, ...] = (0,)) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")

    cfg_path = HERE / "configs" / f"{variant}.yaml"
    cfg = load_experiment_config(cfg_path)

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_symlink()

    print(f"[run] variant={variant}  output={out_dir}")
    run_training(cfg=cfg, output_dir=str(out_dir), resume=resume, gpus=gpus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=VARIANTS + ("both",))
    ap.add_argument("--resume", default=None, help="Checkpoint to resume from (applies to the first variant).")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    for i, v in enumerate(variants):
        run(v, resume=(args.resume if i == 0 else None), gpus=tuple(args.gpus))


if __name__ == "__main__":
    main()