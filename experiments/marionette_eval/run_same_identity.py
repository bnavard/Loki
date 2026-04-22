"""
Same-identity evaluation runner.

Every usable YouTube identity contributes `samples_per_identity` reconstructions
against its own motion: one clip is drawn uniformly per sample, then an
independent `ref_frame_idx` and `driver_start_idx` are drawn inside that clip
under a minimum-gap constraint (`min_ref_driver_gap`) to keep the ref frame
well outside the target window — mirroring the training-time convention.

Usage (from repo root):

    conda activate marionette
    PYTHONPATH=. python experiments/marionette_eval/run_same_identity.py \\
        --config     experiments/marionette_eval/configs/same_identity.yaml \\
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

Outputs:
    outputs/marionette_eval/same_identity/run_<ts>/
        config_resolved.yaml
        samples/<NNN>_<YT>_<k>/panel.{png,mp4}
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from experiments.marionette_eval.evaluator import Evaluator, EvaluatorPaths
from experiments.marionette_eval.pairing import build_same_identity_samples
from marionette.config_utils import load_experiment_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", default=None)
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
    seed         = args.seed      if args.seed      is not None else int(cfg.seed)
    cfg_scale    = args.cfg_scale if args.cfg_scale is not None else float(cfg.inference.cfg_scale)
    n_ddim_steps = int(cfg.inference.get("n_ddim_steps", 50))
    n_frames     = int(cfg.inference.n_frames)

    samples_per_identity = int(cfg.eval.samples_per_identity)
    min_ref_driver_gap   = int(cfg.eval.min_ref_driver_gap)

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
    samples, stats = build_same_identity_samples(
        val_clips, paths.flame_root,
        n_frames=n_frames,
        samples_per_identity=samples_per_identity,
        min_ref_driver_gap=min_ref_driver_gap,
        seed=seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    print(f"[same-identity eval] run dir: {run_dir}")
    print(f"  checkpoint:             {checkpoint}")
    print(f"  seed:                   {seed}")
    print(f"  n_frames:               {n_frames}")
    print(f"  cfg_scale:              {cfg_scale}")
    print(f"  n_ddim_steps:           {n_ddim_steps}")
    print(f"  samples_per_identity:   {samples_per_identity}")
    print(f"  min_ref_driver_gap:     {min_ref_driver_gap}")
    print(f"  val clips:              {len(val_clips)}")
    print(f"  identities (all):       {stats['n_total_identities']}")
    print(f"  identities (usable):    {stats['n_usable_identities']}")
    print(f"  identities (dropped):   {stats['n_dropped_identities']} "
          f"(no clip ≥ {n_frames + 2 * min_ref_driver_gap} frames)")
    print(f"  samples to generate:    {stats['n_samples']}")

    evaluator = Evaluator(
        cfg=cfg, checkpoint=checkpoint, paths=paths,
        n_frames=n_frames, cfg_scale=cfg_scale, n_ddim_steps=n_ddim_steps,
        device=torch.device(args.device),
    )
    print(f"  audio encoder:          {'enabled' if evaluator.has_audio else 'disabled'}")

    # Per-identity counter for the k-th sample tag within that identity.
    per_id_k: dict[str, int] = {}
    for i, s in enumerate(samples):
        k = per_id_k.get(s.identity, 0)
        per_id_k[s.identity] = k + 1
        tag = f"{i:03d}_{s.identity}_{k}"
        out_dir = run_dir / "samples" / tag
        title = (
            f"Same-Identity [{i+1}/{len(samples)}] | "
            f"clip={s.clip[:24]} | ref=f{s.ref_frame_idx} "
            f"target=[{s.driver_start_idx}:{s.driver_start_idx + n_frames})"
        )
        print(f"[{i+1}/{len(samples)}] {tag}")
        print(f"  clip:   {s.clip} (len={s.clip_len})")
        print(f"  ref:    f{s.ref_frame_idx}   target: f{s.driver_start_idx}..f{s.driver_start_idx + n_frames - 1}")
        evaluator.run_one(
            ref_clip=s.clip, driver_clip=s.clip,
            ref_frame_idx=s.ref_frame_idx, driver_start_idx=s.driver_start_idx,
            out_dir=out_dir, title=title,
        )

    print(f"[same-identity eval] done. {len(samples)} panels → {run_dir}")


if __name__ == "__main__":
    main()
