r"""
EchoMimic evaluation runner.

Builds a deterministic list of `EvalSample`s from the chosen dataset +
protocol, then invokes EchoMimic's `infer_audio2vid.py` one sample at a
time via the adapter. Output layout mirrors `marionette_eval/` and the
other SOTA baselines.

Protocol semantics (audio-driven, like SadTalker / AniTalker):
  * same_identity_reconstruction  — ref image and driver audio from the
                                    same clip (self-reconstruction).
  * cross_identity                — ref image from identity A, driver
                                    audio from identity B ≠ A. Output:
                                    A's face lip-synced to B's speech.

Usage (from repo root, `marionette` env — NOT `echomimic`; the subprocess
hops into the echomimic env itself). Build the curated manifest for each
dataset once via `dataset/build_manifest.py` before running these.

    # HDTF — same-identity reconstruction
    PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --n_samples 346 \
        --clip_duration_s 3.0 \
        --seed 42

    # HDTF — cross-identity
    PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \
        --dataset hdtf \
        --protocol cross_identity \
        --n_samples 200 \
        --clip_duration_s 3.0 \
        --seed 42

    # TalkVid — same-identity reconstruction
    PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \
        --dataset talkvid \
        --protocol same_identity_reconstruction \
        --n_samples 125 \
        --clip_duration_s 5.0 \
        --seed 42

    # TalkVid — cross-identity
    PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \
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
from experiments.sota_comparison.echomimic.adapter           import (
    DEFAULT_UPSTREAM_CONFIG, EchoMimicArgs, run_one,
)


HERE          = Path(__file__).resolve().parent
DEFAULT_IMPL  = HERE / "impl"
DEFAULT_OUT   = Path("outputs/sota_comparison/echomimic")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run EchoMimic inference over a benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / protocol
    p.add_argument("--dataset",         default="talkvid", choices=["hdtf", "talkvid"])
    p.add_argument("--protocol",        default="cross_identity",
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--n_samples",       type=int,   default=125)
    p.add_argument("--clip_duration_s", type=float, default=5.0,
                   help="Driver audio is ffmpeg-trimmed to this many seconds. "
                        "5.0 fits TalkVid; HDTF mirror is ~3.24 s → 3.0.")
    p.add_argument("--seed",            type=int,   default=42)

    # EchoMimic knobs (→ EchoMimicArgs; mirror upstream's argparse)
    p.add_argument("--width",  "-W", type=int, default=512)
    p.add_argument("--height", "-H", type=int, default=512)
    p.add_argument("--fps",         type=int,   default=25,
                   help="Output video fps. 25 matches TalkVid + HDTF native.")
    p.add_argument("--cfg",         type=float, default=2.5,
                   help="Classifier-free guidance scale.")
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--echomimic_seed", type=int, default=420,
                   help="EchoMimic's internal seed (independent of the "
                        "runner's --seed which drives pair list + ref-frame).")
    p.add_argument("--sample_rate", type=int,   default=16000)
    p.add_argument("--context_frames",  type=int, default=12)
    p.add_argument("--context_overlap", type=int, default=3)
    p.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    p.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)

    # Plumbing
    p.add_argument("--impl_dir",   type=Path, default=DEFAULT_IMPL,
                   help="Path to the cloned EchoMimic repo (gitignored).")
    p.add_argument("--conda_env",  default="echomimic",
                   help="Conda env holding EchoMimic's torch 2.1+cu121 stack.")
    p.add_argument("--upstream_config", type=Path, default=DEFAULT_UPSTREAM_CONFIG,
                   help="Path to upstream's animation.yaml relative to impl_dir. "
                        "Our adapter reads this and writes a per-sample patched "
                        "copy with the test_cases mapping replaced.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help=f"Root for outputs. Default: "
                        f"{DEFAULT_OUT}/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--n_take",     type=int, default=None,
                   help="Cap pair list to this many samples (debug).")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.impl_dir.is_dir() or not (args.impl_dir / "infer_audio2vid.py").exists():
        raise SystemExit(
            f"EchoMimic repo not found at {args.impl_dir}. "
            f"See experiments/sota_comparison/echomimic/README.md for the "
            f"clone + weights setup."
        )

    upstream_cfg = args.impl_dir / args.upstream_config
    if not upstream_cfg.is_file():
        raise SystemExit(
            f"Upstream config not found at {upstream_cfg}. "
            f"Did the clone complete? Default expected: "
            f"{args.impl_dir}/configs/prompts/animation.yaml"
        )

    # Spot-check the four main ckpts so we fail fast on a half-done download
    # rather than deep inside the subprocess's model loader.
    ckpts = [
        "pretrained_weights/denoising_unet.pth",
        "pretrained_weights/reference_unet.pth",
        "pretrained_weights/motion_module.pth",
        "pretrained_weights/face_locator.pth",
        "pretrained_weights/audio_processor/whisper_tiny.pt",
    ]
    for rel in ckpts:
        if not (args.impl_dir / rel).is_file():
            raise SystemExit(
                f"EchoMimic checkpoint missing: {args.impl_dir / rel}\n"
                f"Run `experiments/sota_comparison/echomimic/setup_env.sh` "
                f"to download the audio-only ckpt subset (~10 GB)."
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
    print(f"[echomimic] {args.dataset} manifest: {len(clips)} identities "
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
    print(f"[echomimic] {args.protocol}: {len(samples)} samples")

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

    em_args = EchoMimicArgs(
        width                   = args.width,
        height                  = args.height,
        fps                     = args.fps,
        cfg                     = args.cfg,
        steps                   = args.steps,
        seed                    = args.echomimic_seed,
        sample_rate             = args.sample_rate,
        context_frames          = args.context_frames,
        context_overlap         = args.context_overlap,
        facemusk_dilation_ratio = args.facemusk_dilation_ratio,
        facecrop_dilation_ratio = args.facecrop_dilation_ratio,
    )

    failed: list[tuple[str, str]] = []
    for sample in tqdm(samples, desc="echomimic"):
        ref_frame_idx = int(rng.integers(0, sample.ref_clip.n_frames))
        try:
            run_one(
                sample          = sample,
                ref_frame_idx   = ref_frame_idx,
                impl_dir        = args.impl_dir,
                output_dir      = out,
                scratch         = scratch,
                conda_env       = args.conda_env,
                upstream_config = args.upstream_config,
                args            = em_args,
            )
        except subprocess.CalledProcessError as e:
            failed.append((sample.sample_id, f"subprocess exit {e.returncode}"))
        except Exception as e:
            failed.append((sample.sample_id, f"{type(e).__name__}: {e}"))

    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=2))
        print(f"[echomimic] {len(failed)} samples failed — see failed.json")
    print(f"[echomimic] done. {out}")


if __name__ == "__main__":
    main()
