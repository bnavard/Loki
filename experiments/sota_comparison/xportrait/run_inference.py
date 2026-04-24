r"""
X-Portrait evaluation runner.

Builds a deterministic list of `EvalSample`s from the chosen dataset +
protocol, then invokes X-Portrait's inference one sample at a time via the
adapter. Output layout mirrors `marionette_eval/` and the other SOTA
baselines.

Protocol semantics (motion-driven, NOT audio-driven):
  * same_identity_reconstruction  — ref frame and driver video from the
                                    same clip. Self-reconstruction check.
  * cross_identity                — ref frame from identity A, driver video
                                    from identity B ≠ A (derangement pair).
                                    Output: A's face doing B's motion.

Usage (from repo root, `marionette` env — NOT `xportrait`; the subprocess
hops into the xportrait env itself). Build the curated manifest for each
dataset once via `dataset/build_manifest.py` before running these.

    # HDTF — same-identity reconstruction
    #   HDTF's mirror is pre-chunked to ~3.24 s, so pass --clip_duration_s 3.0.
    PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --n_samples 346 \
        --clip_duration_s 3.0 \
        --seed 42

    # HDTF — cross-identity
    PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
        --dataset hdtf \
        --protocol cross_identity \
        --n_samples 200 \
        --clip_duration_s 3.0 \
        --seed 42

    # TalkVid — same-identity reconstruction
    PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
        --dataset talkvid \
        --protocol same_identity_reconstruction \
        --n_samples 125 \
        --clip_duration_s 5.0 \
        --seed 42

    # TalkVid — cross-identity
    PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
        --dataset talkvid \
        --protocol cross_identity \
        --n_samples 125 \
        --clip_duration_s 5.0 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.sota_comparison.dataset.benchmark_manifest import load_by_dataset
from experiments.sota_comparison.dataset.pairing             import build_samples
from experiments.sota_comparison.xportrait.adapter           import (
    DEFAULT_CKPT, DEFAULT_MODEL_CONFIG, XPortraitArgs, run_one,
)


HERE          = Path(__file__).resolve().parent
DEFAULT_IMPL  = HERE / "impl"
DEFAULT_OUT   = Path("outputs/sota_comparison/xportrait")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run X-Portrait inference over a benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / protocol
    p.add_argument("--dataset",         default="talkvid", choices=["hdtf", "talkvid"])
    p.add_argument("--protocol",        default="cross_identity",
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--n_samples",       type=int,   default=125)
    p.add_argument("--clip_duration_s", type=float, default=5.0,
                   help="Driver video is ffmpeg-trimmed to this many seconds. "
                        "5.0 fits TalkVid; HDTF's mirror is ~3.24 s so pass 3.0.")
    p.add_argument("--seed",            type=int,   default=42)

    # X-Portrait knobs (→ XPortraitArgs; defaults match upstream's demo script)
    p.add_argument("--uc_scale",    type=int, default=5,
                   help="Unconditional guidance scale (CFG-like).")
    p.add_argument("--ddim_steps",  type=int, default=30)
    p.add_argument("--num_mix",     type=int, default=4,
                   help="Overlap frames for prompt-travelling inference.")
    p.add_argument("--xp_seed",     type=int, default=999,
                   help="Internal seed passed to X-Portrait (independent of "
                        "the runner's --seed which drives pair list + frame "
                        "selection).")
    p.add_argument("--best_frame",  type=int, default=-1,
                   help="-1 → auto-detect via face-alignment (recommended "
                        "for batch). Any non-negative value forces that "
                        "exact driver frame index across every sample.")

    # Plumbing
    p.add_argument("--impl_dir",   type=Path, default=DEFAULT_IMPL,
                   help="Path to the cloned X-Portrait repo (gitignored).")
    p.add_argument("--conda_env",  default="xportrait",
                   help="Conda env holding X-Portrait's torch 2.0.1+cu118 stack.")
    p.add_argument("--ckpt_rel",   type=Path, default=DEFAULT_CKPT,
                   help="Checkpoint path relative to impl_dir. Default is "
                        "upstream's expected filename under `checkpoint/`.")
    p.add_argument("--model_config", type=Path, default=DEFAULT_MODEL_CONFIG,
                   help="Model config path relative to impl_dir.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help=f"Root for outputs. Default: "
                        f"{DEFAULT_OUT}/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--n_take",     type=int, default=None,
                   help="Cap the pair list to this many samples (debug). "
                        "Independent of --n_samples, applied post-build.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.impl_dir.is_dir() or not (args.impl_dir / "core" / "test_xportrait.py").exists():
        raise SystemExit(
            f"X-Portrait repo not found at {args.impl_dir}. "
            f"See experiments/sota_comparison/xportrait/README.md for the "
            f"clone + weights setup."
        )

    ckpt_abs = args.impl_dir / args.ckpt_rel
    if not ckpt_abs.is_file():
        raise SystemExit(
            f"X-Portrait checkpoint not found at {ckpt_abs}. "
            f"Run `experiments/sota_comparison/xportrait/setup_env.sh` or "
            f"download it manually (see the baseline README)."
        )

    # Resolve output directory (timestamped per run for reproducibility).
    if args.output_dir is None:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = DEFAULT_OUT / args.dataset / args.protocol / f"run_{ts}"
    else:
        out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)

    # Load curated benchmark manifest.
    clips, manifest_meta = load_by_dataset(args.dataset)
    print(f"[xportrait] {args.dataset} manifest: {len(clips)} identities "
          f"(cap={manifest_meta['n_samples_cap']}, seed={manifest_meta['seed']})")

    samples = build_samples(
        protocol        = args.protocol,
        clips           = clips,
        n_samples       = args.n_samples,
        clip_duration_s = args.clip_duration_s,
        seed            = args.seed,
    )
    if args.n_take is not None:
        samples = samples[: args.n_take]
    print(f"[xportrait] {args.protocol}: {len(samples)} samples")

    # One seeded RNG for ref-frame draws across the whole run. Shared seeding
    # policy with SadTalker / HunyuanPortrait — same `(protocol, seed,
    # sample_id)` picks the same ref frame across every baseline.
    rng = np.random.default_rng(args.seed)

    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    (out / "config_resolved.json").write_text(json.dumps({
        **vars(args),
        "impl_dir":   str(args.impl_dir),
        "output_dir": str(out),
        "git_rev":    git_rev,
    }, indent=2, default=str))

    xp_args = XPortraitArgs(
        uc_scale    = args.uc_scale,
        ddim_steps  = args.ddim_steps,
        num_mix     = args.num_mix,
        seed        = args.xp_seed,
        best_frame  = args.best_frame,
    )

    failed: list[tuple[str, str]] = []
    for sample in tqdm(samples, desc="xportrait"):
        ref_frame_idx = int(rng.integers(0, sample.ref_clip.n_frames))
        try:
            run_one(
                sample        = sample,
                ref_frame_idx = ref_frame_idx,
                impl_dir      = args.impl_dir,
                output_dir    = out,
                scratch       = scratch,
                conda_env     = args.conda_env,
                ckpt_rel      = args.ckpt_rel,
                model_config  = args.model_config,
                args          = xp_args,
            )
        except subprocess.CalledProcessError as e:
            failed.append((sample.sample_id, f"subprocess exit {e.returncode}"))
        except Exception as e:
            failed.append((sample.sample_id, f"{type(e).__name__}: {e}"))

    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=2))
        print(f"[xportrait] {len(failed)} samples failed — see failed.json")
    print(f"[xportrait] done. {out}")


if __name__ == "__main__":
    main()
