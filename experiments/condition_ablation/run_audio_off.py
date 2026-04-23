r"""
Entry script for the audio ablation — trains the "audio off" arm of a
controlled ablation against `marionette_baseline` (the audio-on arm).

Both arms are launched fresh from the same base config, the same seed, and
the same SD 2.1 init. The only independent variable is whether the audio
cross-attention pathway is active. After training, compare lip-sync metrics
(LSE-D / LSE-C) between the two checkpoints on matching validation pairs
to quantify the contribution of audio conditioning.

Usage (from repo root):

    # Single GPU
    PYTHONPATH=. python experiments/ablate_audio/run.py

    # Multi-GPU (DDP)
    PYTHONPATH=. python experiments/ablate_audio/run.py --gpus 0 1 2 3

    # Resume from a checkpoint
    PYTHONPATH=. python experiments/ablate_audio/run.py \
        --resume outputs/ablate_audio/audio_off/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt

Outputs:
    outputs/ablate_audio/audio_off/run_<timestamp>/
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
CONFIG_PATH = HERE / "configs" / "audio_off.yaml"


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
