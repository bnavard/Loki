r"""
Entry script for the Marionette baseline — a full training run against the
canonical recipe (`marionette/configs/base.yaml`).

Usage (from repo root):

    # Single GPU
    PYTHONPATH=. python experiments/marionette_baseline/run.py

    # Multi-GPU (DDP)
    PYTHONPATH=. python experiments/marionette_baseline/run.py --gpus 0 1 2 3

    # Resume from a checkpoint
    PYTHONPATH=. python experiments/marionette_baseline/run.py \
        --resume outputs/marionette_baseline/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt

Outputs:
    outputs/marionette_baseline/run_<timestamp>/
        config_resolved.yaml
        checkpoints/{th-<step>.ckpt, th-best-<step>-<val_loss>.ckpt}
        logs/ (TensorBoard)
        visualizations/step_<step>/*.{png,mp4}
"""
from __future__ import annotations

import argparse
from pathlib import Path

from marionette.config_utils import load_experiment_config
from marionette.train import run_training


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "configs" / "baseline.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs="+", type=int, default=[0],
                   help="GPU indices to train on.")
    p.add_argument("--resume", default=None,
                   help="Checkpoint to resume from (reuses its run directory).")
    p.add_argument("--output_dir", default=None,
                   help="Override output_dir from the config.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(CONFIG_PATH)
    run_training(cfg, output_dir=args.output_dir, resume=args.resume, gpus=tuple(args.gpus))


if __name__ == "__main__":
    main()
