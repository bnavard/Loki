r"""
HunyuanPortrait evaluation runner.

Builds a deterministic list of `EvalSample`s from the chosen dataset +
protocol, then invokes HunyuanPortrait's inference one sample at a time via
the adapter. Output layout mirrors `marionette_eval/` and the other SOTA
baselines.

Protocol semantics (motion-driven, NOT audio-driven):
  * same_identity_reconstruction  — ref frame and driver video come from
                                    the same clip. A self-reconstruction
                                    sanity check.
  * cross_identity                — ref frame from identity A, driver video
                                    from identity B ≠ A (derangement pair).
                                    Output: A's face doing B's motion.

Usage (from repo root, `marionette` env — NOT `hunyuan_portrait`; the
subprocess hops into the hunyuan_portrait env itself). Build the curated
manifest for each dataset once via `dataset/build_manifest.py` before
running these.

    # HDTF — same-identity reconstruction
    #   HDTF's mirror is pre-chunked to ~3.24 s, so pass --clip_duration_s 3.0.
    PYTHONPATH=. python experiments/sota_comparison/hunyuan_portrait/run_inference.py \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --n_samples 346 \
        --clip_duration_s 3.0 \
        --seed 42

    # HDTF — cross-identity
    PYTHONPATH=. python experiments/sota_comparison/hunyuan_portrait/run_inference.py \
        --dataset hdtf \
        --protocol cross_identity \
        --n_samples 200 \
        --clip_duration_s 3.0 \
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
from experiments.sota_comparison.hunyuan_portrait.adapter    import (
    HunyuanPortraitArgs, run_one,
)


HERE          = Path(__file__).resolve().parent
DEFAULT_IMPL  = HERE / "impl"
DEFAULT_OUT   = Path("outputs/sota_comparison/hunyuan_portrait")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run HunyuanPortrait inference over a benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / protocol
    p.add_argument("--dataset",         default="hdtf", choices=["hdtf"])
    p.add_argument("--protocol",        default="cross_identity",
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--n_samples",       type=int,   default=212)
    p.add_argument("--clip_duration_s", type=float, default=3.0,
                   help="Driver video is ffmpeg-trimmed to this many seconds. "
                        "HDTF's mirror is pre-chunked to ~3.24 s so 3.0 keeps "
                        "the full 81-frame window in scope of build_samples's "
                        "length filter.")
    p.add_argument("--seed",            type=int,   default=42)

    # HunyuanPortrait knobs (→ HunyuanPortraitArgs; mirror upstream yaml)
    p.add_argument("--num_inference_steps", type=int,   default=25)
    p.add_argument("--motion_bucket_id",    type=int,   default=0)
    p.add_argument("--n_sample_frames",     type=int,   default=25,
                   help="Inner pipeline frame batch (memory-bound).")
    p.add_argument("--no_arcface",          action="store_true",
                   help="Disable ArcFace identity conditioning (speeds up, "
                        "slight drop in identity fidelity).")
    p.add_argument("--min_appearance_guidance", type=float, default=2.0)
    p.add_argument("--max_appearance_guidance", type=float, default=2.0)
    p.add_argument("--min_motion_guidance",     type=float, default=2.0)
    p.add_argument("--max_motion_guidance",     type=float, default=2.0)

    # Plumbing
    p.add_argument("--impl_dir",   type=Path, default=DEFAULT_IMPL,
                   help="Path to the cloned HunyuanPortrait repo (gitignored).")
    p.add_argument("--conda_env",  default="hunyuan_portrait",
                   help="Conda env holding HunyuanPortrait's torch stack.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help=f"Root for outputs. Default: "
                        f"{DEFAULT_OUT}/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--n_take",     type=int, default=None,
                   help="Cap the pair list to this many samples (debug). "
                        "Independent of --n_samples, applied post-build.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.impl_dir.is_dir() or not (args.impl_dir / "inference.py").exists():
        raise SystemExit(
            f"HunyuanPortrait repo not found at {args.impl_dir}. "
            f"See experiments/sota_comparison/hunyuan_portrait/README.md for "
            f"the clone + weights setup."
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

    # Load the curated benchmark manifest — one clip per identity with stable
    # `id_XXXX` UIDs, built by `dataset/build_manifest.py` and committed to
    # git under `experiments/sota_comparison/manifests/<dataset>.json`.
    clips, manifest_meta = load_by_dataset(args.dataset)
    print(f"[hunyuan_portrait] {args.dataset} manifest: {len(clips)} identities "
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
    print(f"[hunyuan_portrait] {args.protocol}: {len(samples)} samples")

    # One seeded RNG for ref-frame draws across the whole run. Matches
    # SadTalker's policy, so a given `(protocol, seed, sample_id)` tuple
    # picks the same ref frame across every baseline driven from the same
    # manifest — keeps cross-baseline comparison aligned frame-for-frame.
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

    hp_args = HunyuanPortraitArgs(
        num_inference_steps      = args.num_inference_steps,
        motion_bucket_id         = args.motion_bucket_id,
        n_sample_frames          = args.n_sample_frames,
        use_arcface              = not args.no_arcface,
        min_appearance_guidance  = args.min_appearance_guidance,
        max_appearance_guidance  = args.max_appearance_guidance,
        min_motion_guidance      = args.min_motion_guidance,
        max_motion_guidance      = args.max_motion_guidance,
    )

    failed: list[tuple[str, str]] = []
    for sample in tqdm(samples, desc="hunyuan_portrait"):
        ref_frame_idx = int(rng.integers(0, sample.ref_clip.n_frames))
        try:
            run_one(
                sample        = sample,
                ref_frame_idx = ref_frame_idx,
                impl_dir      = args.impl_dir,
                output_dir    = out,
                scratch       = scratch,
                conda_env     = args.conda_env,
                args          = hp_args,
            )
        except subprocess.CalledProcessError as e:
            failed.append((sample.sample_id, f"subprocess exit {e.returncode}"))
        except Exception as e:
            failed.append((sample.sample_id, f"{type(e).__name__}: {e}"))

    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=2))
        print(f"[hunyuan_portrait] {len(failed)} samples failed — see failed.json")
    print(f"[hunyuan_portrait] done. {out}")


if __name__ == "__main__":
    main()
