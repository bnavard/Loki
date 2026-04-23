"""
SadTalker evaluation runner.

Builds a deterministic list of `EvalSample`s from the chosen dataset +
protocol, then invokes SadTalker's inference one sample at a time via the
adapter. Output layout mirrors `marionette_eval/`:

    outputs/sota_comparison/sadtalker/<dataset>/<protocol>/run_<ts>/
    ├── config_resolved.json     # full args + git rev
    ├── scratch/                 # per-sample working dir (source.png, audio.wav)
    └── samples/<sample_id>/panel.mp4

Usage (from repo root):

    conda activate marionette    # NOT sadtalker — this runner only orchestrates;
                                 # the subprocess hops to the sadtalker env itself.

    # Same-identity reconstruction (matches SadTalker's paper protocol, except
    # clip_duration_s is 3.0 instead of 8.0 because our HDTF mirror is
    # pre-chunked into 3.24-s segments — see README).
    PYTHONPATH=. python experiments/sota_comparison/sadtalker/run_inference.py \\
        --dataset hdtf \\
        --protocol same_identity_reconstruction \\
        --n_samples 346 \\
        --clip_duration_s 3.0 \\
        --seed 42

    # Cross-identity voice transfer (speaker A's face driven by speaker B's
    # audio — complements the paper protocol).
    PYTHONPATH=. python experiments/sota_comparison/sadtalker/run_inference.py \\
        --dataset hdtf \\
        --protocol cross_identity \\
        --n_samples 200 \\
        --clip_duration_s 3.0 \\
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

from experiments.sota_comparison.dataset.hdtf    import HDTFDataset
from experiments.sota_comparison.dataset.pairing import build_samples
from experiments.sota_comparison.sadtalker.adapter import SadTalkerArgs, run_one


HERE          = Path(__file__).resolve().parent
DEFAULT_IMPL  = HERE / "impl"
DEFAULT_OUT   = Path("outputs/sota_comparison/sadtalker")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SadTalker inference over a benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / protocol
    p.add_argument("--dataset",         default="hdtf", choices=["hdtf"])
    p.add_argument("--protocol",        default="same_identity_reconstruction",
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--n_samples",       type=int,   default=346)
    p.add_argument("--clip_duration_s", type=float, default=3.0)
    p.add_argument("--seed",            type=int,   default=42)

    # SadTalker knobs (→ SadTalkerArgs)
    p.add_argument("--size",       type=int, default=512, choices=[256, 512])
    p.add_argument("--preprocess", default="crop",
                   choices=["crop", "extcrop", "resize", "full", "extfull"])
    p.add_argument("--pose_style", type=int, default=0,
                   help="0..45 — learned speaker-style bucket; "
                        "only affects head pose, not lip sync.")
    p.add_argument("--enhancer",   default=None,
                   choices=[None, "gfpgan", "RestoreFormer"])
    p.add_argument("--still",      action="store_true",
                   help="Reduce head motion (paper-style output).")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Facerender batch size; memory-bound.")

    # Plumbing
    p.add_argument("--impl_dir",   type=Path, default=DEFAULT_IMPL,
                   help="Path to the cloned SadTalker repo (gitignored).")
    p.add_argument("--conda_env",  default="sadtalker",
                   help="Conda env holding SadTalker's torch 2.1+cu121 stack.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help=f"Root for outputs. Default: "
                        f"{DEFAULT_OUT}/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--n_take",     type=int, default=None,
                   help="Cap the pair list to this many samples (debug). "
                        "Independent of --n_samples, applied post-build.")
    return p.parse_args()


_DATASET_REGISTRY = {"hdtf": HDTFDataset}


def main():
    args = parse_args()

    if not args.impl_dir.is_dir() or not (args.impl_dir / "inference.py").exists():
        raise SystemExit(
            f"SadTalker repo not found at {args.impl_dir}. "
            f"See experiments/sota_comparison/sadtalker/README.md for the "
            f"clone + weights setup."
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

    # Build the pair list.
    ds    = _DATASET_REGISTRY[args.dataset]()
    clips = ds.load()
    print(f"[sadtalker] {ds.name} manifest: {len(clips)} clips")

    samples = build_samples(
        protocol        = args.protocol,
        clips           = clips,
        n_samples       = args.n_samples,
        clip_duration_s = args.clip_duration_s,
        seed            = args.seed,
    )
    if args.n_take is not None:
        samples = samples[: args.n_take]
    print(f"[sadtalker] {args.protocol}: {len(samples)} samples")

    # One seeded RNG for ref-frame draws across the whole run. Same top-level
    # --seed reproduces the same frame choices.
    rng = np.random.default_rng(args.seed)

    # Capture the full config (+ git rev) so a run is reproducible from the
    # on-disk record alone.
    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    (out / "config_resolved.json").write_text(json.dumps({
        **vars(args),
        "impl_dir":   str(args.impl_dir),
        "output_dir": str(out),
        "git_rev":    git_rev,
    }, indent=2, default=str))

    st_args = SadTalkerArgs(
        size       = args.size,
        preprocess = args.preprocess,
        pose_style = args.pose_style,
        enhancer   = args.enhancer,
        still      = args.still,
        batch_size = args.batch_size,
    )

    failed: list[tuple[str, str]] = []
    for sample in tqdm(samples, desc="sadtalker"):
        ref_frame_idx = int(rng.integers(0, sample.ref_clip.n_frames))
        try:
            run_one(
                sample        = sample,
                ref_frame_idx = ref_frame_idx,
                impl_dir      = args.impl_dir,
                output_dir    = out,
                scratch       = scratch,
                conda_env     = args.conda_env,
                args          = st_args,
            )
        except subprocess.CalledProcessError as e:
            failed.append((sample.sample_id, f"subprocess exit {e.returncode}"))
        except Exception as e:
            failed.append((sample.sample_id, f"{type(e).__name__}: {e}"))

    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=2))
        print(f"[sadtalker] {len(failed)} samples failed — see failed.json")
    print(f"[sadtalker] done. {out}")


if __name__ == "__main__":
    main()
