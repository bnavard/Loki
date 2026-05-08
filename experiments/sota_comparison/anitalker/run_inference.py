r"""
AniTalker evaluation runner.

Builds a deterministic list of `EvalSample`s from the chosen dataset +
protocol, then invokes AniTalker's inference one sample at a time via the
adapter. Output layout mirrors `loki_eval/` and the other SOTA
baselines.

Protocol semantics (AniTalker is audio-driven, like SadTalker):
  * same_identity_reconstruction  — ref image and driver audio from the
                                    same clip (self-reconstruction).
  * cross_identity                — ref image from identity A, driver audio
                                    from identity B ≠ A. Output: A's face
                                    lip-synced to B's speech.

Usage (from repo root, `loki` env — NOT `anitalker`; the subprocess
hops into the anitalker env itself). Build the curated manifest for each
dataset once via `dataset/build_manifest.py` before running these.

    # HDTF — same-identity reconstruction
    PYTHONPATH=. python experiments/sota_comparison/anitalker/run_inference.py \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --n_samples 346 \
        --clip_duration_s 3.0 \
        --seed 42

    # HDTF — cross-identity
    PYTHONPATH=. python experiments/sota_comparison/anitalker/run_inference.py \
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
from experiments.sota_comparison.anitalker.adapter           import (
    DEFAULT_STAGE1_CKPT, DEFAULT_STAGE2_CKPT, AniTalkerArgs, run_one,
)


HERE          = Path(__file__).resolve().parent
DEFAULT_IMPL  = HERE / "impl"
DEFAULT_OUT   = Path("outputs/sota_comparison/anitalker")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run AniTalker inference over a benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / protocol
    p.add_argument("--dataset",         default="hdtf", choices=["hdtf"])
    p.add_argument("--protocol",        default="cross_identity",
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--n_samples",       type=int,   default=212)
    p.add_argument("--clip_duration_s", type=float, default=3.0,
                   help="Driver audio is ffmpeg-trimmed to this many seconds. "
                        "HDTF's mirror is pre-chunked to ~3.24 s.")
    p.add_argument("--seed",            type=int,   default=42)

    # AniTalker knobs (→ AniTalkerArgs; defaults match upstream's demo)
    p.add_argument("--step_T",         type=int, default=50,
                   help="Diffusion denoising steps. Upstream default.")
    p.add_argument("--anitalker_seed", type=int, default=0,
                   help="AniTalker's internal seed (independent of the "
                        "runner's --seed which drives pair-list + ref-frame).")
    p.add_argument("--no_face_sr",     action="store_true",
                   help="Disable GFPGAN 256→512 face super-resolution. "
                        "Default leaves SR on so AniTalker's output surface "
                        "matches every other baseline (512×512).")
    p.add_argument("--motion_dim",     type=int, default=20)
    p.add_argument("--decoder_layers", type=int, default=2)

    # Plumbing
    p.add_argument("--impl_dir",   type=Path, default=DEFAULT_IMPL,
                   help="Path to the cloned AniTalker repo (gitignored).")
    p.add_argument("--conda_env",  default="anitalker",
                   help="Conda env holding AniTalker's torch 2.0.1+cu118 stack.")
    p.add_argument("--stage1_ckpt", type=Path, default=DEFAULT_STAGE1_CKPT,
                   help="Stage-1 ckpt path relative to impl_dir.")
    p.add_argument("--stage2_ckpt", type=Path, default=DEFAULT_STAGE2_CKPT,
                   help="Stage-2 ckpt path relative to impl_dir. Default is "
                        "the audio-only hubert variant (the one upstream "
                        "recommends for highest quality).")
    p.add_argument("--output_dir", type=Path, default=None,
                   help=f"Root for outputs. Default: "
                        f"{DEFAULT_OUT}/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--n_take",     type=int, default=None,
                   help="Cap the pair list to this many samples (debug). "
                        "Independent of --n_samples, applied post-build.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.impl_dir.is_dir() or not (args.impl_dir / "code" / "demo.py").exists():
        raise SystemExit(
            f"AniTalker repo not found at {args.impl_dir}. "
            f"See experiments/sota_comparison/anitalker/README.md for the "
            f"clone + weights setup."
        )

    for ckpt in (args.stage1_ckpt, args.stage2_ckpt):
        abs_ckpt = args.impl_dir / ckpt
        if not abs_ckpt.is_file():
            raise SystemExit(
                f"AniTalker checkpoint missing: {abs_ckpt}. "
                f"Run `experiments/sota_comparison/anitalker/setup_env.sh` "
                f"or download the `taocode/anitalker_ckpts` HuggingFace repo "
                f"into `{args.impl_dir}/ckpts/`."
            )
    hubert_dir = args.impl_dir / "ckpts" / "chinese-hubert-large"
    if not hubert_dir.is_dir():
        raise SystemExit(
            f"HuBERT model directory missing: {hubert_dir}. "
            f"Needed for on-the-fly feature extraction — see setup_env.sh."
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

    clips, manifest_meta = load_by_dataset(args.dataset)
    print(f"[anitalker] {args.dataset} manifest: {len(clips)} identities "
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
    print(f"[anitalker] {args.protocol}: {len(samples)} samples")

    # Shared seeding policy with the other runners.
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

    at_args = AniTalkerArgs(
        step_T         = args.step_T,
        seed           = args.anitalker_seed,
        face_sr        = not args.no_face_sr,
        motion_dim     = args.motion_dim,
        decoder_layers = args.decoder_layers,
    )

    failed: list[tuple[str, str]] = []
    for sample in tqdm(samples, desc="anitalker"):
        ref_frame_idx = int(rng.integers(0, sample.ref_clip.n_frames))
        try:
            run_one(
                sample        = sample,
                ref_frame_idx = ref_frame_idx,
                impl_dir      = args.impl_dir,
                output_dir    = out,
                scratch       = scratch,
                conda_env     = args.conda_env,
                stage1_ckpt   = args.stage1_ckpt,
                stage2_ckpt   = args.stage2_ckpt,
                args          = at_args,
            )
        except subprocess.CalledProcessError as e:
            failed.append((sample.sample_id, f"subprocess exit {e.returncode}"))
        except Exception as e:
            failed.append((sample.sample_id, f"{type(e).__name__}: {e}"))

    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=2))
        print(f"[anitalker] {len(failed)} samples failed — see failed.json")
    print(f"[anitalker] done. {out}")


if __name__ == "__main__":
    main()
