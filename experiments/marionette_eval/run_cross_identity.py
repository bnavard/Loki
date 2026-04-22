"""
Cross-identity evaluation runner.

Every usable YouTube identity in the validation set appears exactly once as
the reference and exactly once as the driver (a derangement), so each
identity's features are evaluated both as the identity anchor (via the frozen
reference UNet) and as the motion source (via the FLAME retargeting +
optional audio). Produces N panels, N = number of usable identities.

Usage (from repo root):

    conda activate marionette
    PYTHONPATH=. python experiments/marionette_eval/run_cross_identity.py \\
        --config     experiments/marionette_eval/configs/cross_identity.yaml \\
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

    # Override output_dir / cfg_scale / seed on the fly:
    PYTHONPATH=. python experiments/marionette_eval/run_cross_identity.py \\
        --config      experiments/marionette_eval/configs/cross_identity.yaml \\
        --checkpoint  outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt \\
        --output_dir  outputs/marionette_eval/cross_identity_cfg3 \\
        --cfg_scale   3.0

Outputs:
    outputs/marionette_eval/cross_identity/run_<ts>/
        config_resolved.yaml
        samples/<NNN>_ref-<YT_A>__drv-<YT_B>/panel.{png,mp4}
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from experiments.marionette_eval.evaluator import Evaluator, EvaluatorPaths
from experiments.marionette_eval.pairing import build_cross_identity_samples
from marionette.config_utils import load_experiment_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True,
                   help="Experiment YAML (base + overlays + eval knobs).")
    p.add_argument("--checkpoint", default=None,
                   help="Override `checkpoint` from the config.")
    p.add_argument("--output_dir", default=None,
                   help="Override `output_dir` from the config.")
    p.add_argument("--cfg_scale",  type=float, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_experiment_config(args.config)

    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("`checkpoint` must be provided via config or --checkpoint.")
    output_root = Path(args.output_dir or cfg.output_dir)
    seed        = args.seed       if args.seed       is not None else int(cfg.seed)
    cfg_scale   = args.cfg_scale  if args.cfg_scale  is not None else float(cfg.inference.cfg_scale)
    n_ddim_steps = int(cfg.inference.get("n_ddim_steps", 50))
    n_frames     = int(cfg.inference.n_frames)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    val_clips_path = Path(cfg.val_dataset.params.clip_list_path)
    with open(val_clips_path) as f:
        val_clips = json.load(f)

    paths = EvaluatorPaths(
        flame_root=Path(cfg.val_dataset.params.flame_root),
        video_root=Path(cfg.val_dataset.params.video_root),
        audio_root=Path(cfg.val_dataset.params.audio_root),
    )
    samples, stats = build_cross_identity_samples(
        val_clips, paths.flame_root, n_frames, seed=seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    print(f"[cross-identity eval] run dir: {run_dir}")
    print(f"  checkpoint:        {checkpoint}")
    print(f"  seed:              {seed}")
    print(f"  n_frames:          {n_frames}")
    print(f"  cfg_scale:         {cfg_scale}")
    print(f"  n_ddim_steps:      {n_ddim_steps}")
    print(f"  val clips:         {len(val_clips)}")
    print(f"  identities (all):  {stats['n_total_identities']}")
    print(f"  identities (usable): {stats['n_usable_identities']}")
    print(f"  dropped clips (too short for n_frames={n_frames}): "
          f"{stats['n_dropped_short_clips']}")
    print(f"  samples to generate: {stats['n_samples']}")

    evaluator = Evaluator(
        cfg=cfg, checkpoint=checkpoint, paths=paths,
        n_frames=n_frames, cfg_scale=cfg_scale, n_ddim_steps=n_ddim_steps,
        device=torch.device(args.device),
    )
    print(f"  audio encoder:     {'enabled' if evaluator.has_audio else 'disabled'}")

    for i, s in enumerate(samples):
        tag = f"{i:03d}_ref-{s.ref_identity}__drv-{s.driver_identity}"
        out_dir = run_dir / "samples" / tag
        title = (
            f"Cross-Identity [{i+1}/{len(samples)}] | "
            f"ref={s.ref_clip[:24]} (f{s.ref_frame_idx}) → "
            f"drv={s.driver_clip[:24]} (f{s.driver_start_idx}:{s.driver_start_idx + n_frames})"
        )
        print(f"[{i+1}/{len(samples)}] {tag}")
        print(f"  ref:    {s.ref_clip}  frame={s.ref_frame_idx}/{s.ref_clip_len}")
        print(f"  driver: {s.driver_clip}  start={s.driver_start_idx}/"
              f"{s.driver_clip_len - n_frames}")
        evaluator.run_one(
            ref_clip=s.ref_clip, driver_clip=s.driver_clip,
            ref_frame_idx=s.ref_frame_idx, driver_start_idx=s.driver_start_idx,
            out_dir=out_dir, title=title,
        )

    print(f"[cross-identity eval] done. {len(samples)} panels → {run_dir}")


if __name__ == "__main__":
    main()
