r"""
Entry script for the "audio off" arm of the conditioning ablation suite —
trains a twin of `marionette_baseline` with the wav2vec2 cross-attention
pathway disabled.

See `experiments/condition_ablation/README.md` for the full matrix. The
audio-on counterpart is the existing `marionette_baseline` checkpoint;
nothing new to launch for that arm.

Usage (from repo root):

    # Single GPU
    PYTHONPATH=. python experiments/condition_ablation/run_audio_off.py

    # Multi-GPU (DDP)
    PYTHONPATH=. python experiments/condition_ablation/run_audio_off.py --gpus 0 1 2 3

    # Resume from a checkpoint
    PYTHONPATH=. python experiments/condition_ablation/run_audio_off.py \
        --resume outputs/condition_ablation/audio_off/run_YYYYmmdd_HHMMSS/checkpoints/th-<step>.ckpt

Outputs:
    outputs/condition_ablation/audio_off/run_<timestamp>/
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
CONFIG_PATH = HERE / "audio_off" / "config.yaml"


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
