r"""
Marionette evaluation runner — produces the same identity-pair-aligned panel
output a SOTA wrapper would, against a Marionette checkpoint instead of an
external baseline.

Aligned with `experiments/sota_comparison/<baseline>/run_inference.py`:
  * Same pair list source (`load_by_dataset(args.dataset)` → curated
    `experiments/sota_comparison/manifests/<dataset>.json`).
  * Same `EvalSample` shape and same `(protocol, seed, sample_id)` ref-frame
    selection policy — `id_0457_id_0009` here refers to the same identity
    pair as in any SOTA baseline's output tree.
  * Same `samples/<sample_id>/panel.{png,mp4}` on-disk layout.

Key differences from a SOTA wrapper:
  * Marionette runs in-process — no `conda run` shell-out, so we don't loop
    a subprocess per sample. The `Evaluator` loads the model + cond_stage
    module once at startup and amortizes that cost across every sample.
  * Both `--dataset talkvid` and `--dataset hdtf` are supported. The
    per-dataset FLAME tracking root is read from `cfg.flame_roots[<dataset>]`
    (TalkVid under `data/flame_tracking/`, HDTF under
    `data/benchmark/hdtf/flame_tracking/`). Each clip needs `fit.npz` at
    `<flame_root>/<clip_id>/fit.npz` — generate via `generate_exp_map/`
    if it's missing for a dataset.
  * Marionette generates `cfg.inference.n_frames` frames per panel
    (default 16 = 0.64 s at 25 fps). SOTA panels are 5 s. The
    `<sample_id>` folder name aligns; `panel.mp4` durations don't.
    `--clip_duration_s` here only controls the `build_samples` clip-length
    filter; it does NOT change Marionette's actual generation length.

Usage (from repo root, `marionette` env). Build the curated manifest once
via `experiments/sota_comparison/dataset/build_manifest.py --dataset talkvid`
before running these.

    # TalkVid — same-identity reconstruction (one panel per identity)
    PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
        --dataset talkvid \
        --protocol same_identity_reconstruction \
        --n_samples 125 \
        --clip_duration_s 5.0 \
        --seed 42 \
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

    # TalkVid — cross-identity (one panel per derangement pair)
    PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
        --dataset talkvid \
        --protocol cross_identity \
        --n_samples 125 \
        --clip_duration_s 5.0 \
        --seed 42 \
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

    # HDTF — same-identity reconstruction (one panel per identity).
    # Requires `fit.npz` under data/benchmark/hdtf/flame_tracking/flowface/<clip_id>/
    # — see the README's HDTF prereq block for the one-time generation command.
    PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --n_samples 212 \
        --clip_duration_s 3.0 \
        --seed 42 \
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

    # HDTF — cross-identity (one panel per derangement pair). HDTF clips are
    # ~3.24 s so use a 3.0 s filter (TalkVid's 5.0 s default would drop them).
    PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
        --dataset hdtf \
        --protocol cross_identity \
        --n_samples 212 \
        --clip_duration_s 3.0 \
        --seed 42 \
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt

    # Override CFG scale / DDIM steps on the fly:
    PYTHONPATH=. python experiments/marionette_eval/run_inference.py \
        --dataset talkvid --protocol cross_identity \
        --n_samples 125 --clip_duration_s 5.0 --seed 42 \
        --checkpoint outputs/marionette_baseline/run_<ts>/checkpoints/<ckpt>.ckpt \
        --cfg_scale 3.0 --n_ddim_steps 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from experiments.marionette_eval.adapter import Evaluator, MarionetteEvalArgs
from experiments.sota_comparison.dataset.benchmark_manifest import load_by_dataset
from experiments.sota_comparison.dataset.pairing             import build_samples
from marionette.config_utils import load_experiment_config


HERE        = Path(__file__).resolve().parent
DEFAULT_CFG = HERE / "configs" / "eval.yaml"
DEFAULT_OUT = Path("outputs/marionette_eval")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Marionette inference over a benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / protocol — match SOTA wrappers' shape.
    p.add_argument("--dataset",         default="talkvid", choices=["talkvid", "hdtf"],
                   help="Both datasets need FLAME tracking on disk; the per-dataset "
                        "`flame_root` is read from `cfg.flame_roots[<dataset>]`.")
    p.add_argument("--protocol",        default="cross_identity",
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--n_samples",       type=int,   default=125)
    p.add_argument("--clip_duration_s", type=float, default=5.0,
                   help="Used by `build_samples` to filter eligible clips. "
                        "Does NOT change Marionette's generation length — "
                        "that's `cfg.inference.n_frames` (16 by default).")
    p.add_argument("--seed",            type=int,   default=42)

    # Marionette inference knobs (override what's in eval.yaml's `inference`)
    p.add_argument("--n_frames",     type=int,   default=None,
                   help="Override `cfg.inference.n_frames`. Must match the "
                        "UNet's `time_steps` — the model is configured for a "
                        "specific frame count at training time.")
    p.add_argument("--cfg_scale",    type=float, default=None,
                   help="Override `cfg.inference.cfg_scale`.")
    p.add_argument("--n_ddim_steps", type=int,   default=None,
                   help="Override `cfg.inference.n_ddim_steps`.")

    # Plumbing
    p.add_argument("--config",     type=Path, default=DEFAULT_CFG,
                   help="Experiment config — base + inference knobs + flame_root.")
    p.add_argument("--checkpoint", default=None,
                   help="Override `cfg.checkpoint`.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help=f"Root for outputs. Default: "
                        f"{DEFAULT_OUT}/<protocol>/run_<ts>/")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--n_take",     type=int, default=None,
                   help="Cap pair list to this many samples (debug).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_experiment_config(args.config)

    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint is None:
        raise SystemExit(
            "`checkpoint` must be provided via config or --checkpoint."
        )

    n_frames     = args.n_frames     if args.n_frames     is not None else int(cfg.inference.n_frames)
    cfg_scale    = args.cfg_scale    if args.cfg_scale    is not None else float(cfg.inference.cfg_scale)
    n_ddim_steps = args.n_ddim_steps if args.n_ddim_steps is not None else int(cfg.inference.get("n_ddim_steps", 50))
    seed         = int(args.seed if args.seed is not None else cfg.seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Resolve output directory (timestamped per run, mirrors SOTA wrappers).
    if args.output_dir is None:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(cfg.output_dir) if "output_dir" in cfg else DEFAULT_OUT
        out  = root / args.protocol / f"run_{ts}"
    else:
        out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Curated manifest (UID-based identity pool shared with every SOTA wrapper).
    clips, manifest_meta = load_by_dataset(args.dataset)
    print(f"[marionette_eval] {args.dataset} manifest: {len(clips)} identities "
          f"(cap={manifest_meta['n_samples_cap']}, seed={manifest_meta['seed']})")

    samples = build_samples(
        protocol        = args.protocol,
        clips           = clips,
        n_samples       = args.n_samples,
        clip_duration_s = args.clip_duration_s,
        seed            = seed,
    )
    if args.n_take is not None:
        samples = samples[: args.n_take]
    print(f"[marionette_eval] {args.protocol}: {len(samples)} samples")

    # Snapshot the resolved config + CLI args for reproducibility.
    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    # Two snapshots at the run root:
    #   * `config_resolved.yaml` — full OmegaConf dump (model architecture etc.).
    #   * `config_resolved.json` — flattened CLI args + run metadata.
    #     The metrics runner reads this to recover dataset/protocol; same
    #     filename every SOTA wrapper writes, so a single loader handles all.
    OmegaConf.save(cfg, out / "config_resolved.yaml")
    (out / "config_resolved.json").write_text(json.dumps({
        **vars(args),
        "config":     str(args.config),
        "checkpoint": str(checkpoint),
        "output_dir": str(out),
        "git_rev":    git_rev,
    }, indent=2, default=str))

    # Build evaluator (loads the model + cond_stage + FLAME skinner once).
    # Per-dataset flame_root: TalkVid lives under `data/flame_tracking/` (the
    # training-time layout); HDTF lives under `data/benchmark/hdtf/flame_tracking/`.
    if "flame_roots" in cfg and args.dataset in cfg.flame_roots:
        flame_root = Path(cfg.flame_roots[args.dataset])
    else:
        flame_root = Path(cfg.val_dataset.params.flame_root)
    evaluator = Evaluator(
        cfg        = cfg,
        checkpoint = checkpoint,
        flame_root = flame_root,
        device     = torch.device(args.device),
        args       = MarionetteEvalArgs(
            n_frames     = n_frames,
            cfg_scale    = cfg_scale,
            n_ddim_steps = n_ddim_steps,
        ),
    )
    print(f"[marionette_eval] checkpoint:  {checkpoint}")
    print(f"[marionette_eval] n_frames:    {n_frames}")
    print(f"[marionette_eval] cfg_scale:   {cfg_scale}")
    print(f"[marionette_eval] DDIM steps:  {n_ddim_steps}")
    print(f"[marionette_eval] audio path:  {'enabled' if evaluator.has_audio else 'disabled'}")

    # One seeded RNG drives ref-frame draws across the whole run.
    # Same seed schedule as every SOTA wrapper, so `(protocol, seed,
    # sample_id)` picks the same ref frame across baselines.
    rng = np.random.default_rng(seed)

    failed: list[tuple[str, str]] = []
    for sample in tqdm(samples, desc="marionette_eval"):
        ref_frame_idx = int(rng.integers(0, sample.ref_clip.n_frames))
        title = (
            f"[{sample.sample_id}] "
            f"{'Same' if args.protocol == 'same_identity_reconstruction' else 'Cross'}-Identity "
            f"| ref={sample.ref_clip.clip_id[:24]} (f{ref_frame_idx}) "
            f"→ drv={sample.driver_clip.clip_id[:24]}"
        )
        try:
            evaluator.run_one(
                sample        = sample,
                ref_frame_idx = ref_frame_idx,
                output_dir    = out,
                title         = title,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((sample.sample_id, f"{type(e).__name__}: {e}"))

    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=2))
        print(f"[marionette_eval] {len(failed)} samples failed — see failed.json")
    print(f"[marionette_eval] done. {out}")


if __name__ == "__main__":
    main()
