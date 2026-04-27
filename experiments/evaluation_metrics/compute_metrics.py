r"""CLI entry point for the metrics package.

Auto-detects protocol + dataset from `<run_dir>/config_resolved.json` and routes
the right metric set:
  * `same_identity_reconstruction`: PSNR / SSIM / LPIPS / LMD-F / LMD-M  (+ FVD)
  * `cross_identity`: ID similarity (ArcFace cosine)                      (+ FVD)

Outputs land at `<run_dir>/metrics.jsonl` (per-sample) and
`<run_dir>/metrics_summary.json` (aggregates + FVD).

Usage
-----

    # Evaluate one run dir:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir outputs/sota_comparison/sadtalker/talkvid/same_identity_reconstruction/run_<ts>/

    # Skip FVD (e.g. while iterating on per-sample metrics):
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir outputs/sota_comparison/xportrait/talkvid/cross_identity/run_<ts>/ \
        --skip-fvd

    # Both backbones for FVD (default is videomae only — converges on
    # smaller samples than I3D, which wants ≥ 2k clips):
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir outputs/sota_comparison/hunyuan_portrait/talkvid/same_identity_reconstruction/run_<ts>/ \
        --fvd-models videomae i3d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.evaluation_metrics.metrics.evaluator import EvalConfig, evaluate


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute evaluation metrics for one run dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run dir, e.g. outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--fps",            type=int, default=25,
                   help="Target fps for paired-metric frame alignment.")
    p.add_argument("--resolution",     type=int, default=512,
                   help="Resolution at which paired metrics are computed.")
    p.add_argument("--fvd-models",     nargs="+", default=["videomae"],
                   choices=["videomae", "i3d"],
                   help="FVD backbones to run. Default: videomae only.")
    p.add_argument("--fvd-seq-len",    type=int, default=16)
    p.add_argument("--lpips-chunk",    type=int, default=64,
                   help="Inner-batch size for LPIPS — drop if you OOM.")
    p.add_argument("--skip-fvd",       action="store_true",
                   help="Skip the distribution-level FVD step.")
    p.add_argument("--no-face-crop", dest="face_crop", action="store_false",
                   help="Disable face-cropping. Default: enabled — both pred "
                        "and GT are independently cropped to a tight square "
                        "around the detected face (margin × bbox), then "
                        "resized to --resolution. PSNR / SSIM / LPIPS / LMD "
                        "thus measure pure face-region quality. Disable for "
                        "raw-framing numbers.")
    p.add_argument("--face-crop-margin", type=float, default=1.3,
                   help="Padding multiplier around the raw RetinaFace bbox.")
    p.add_argument("--n-frames", type=int, default=16,
                   help="Cap pred (and therefore GT) to this many frames so "
                        "SOTA's 75–125 frame outputs are scored on the same "
                        "temporal coverage as Marionette's 16-frame panel. "
                        "Set to 0 (or any non-positive int) for tool-native "
                        "length. Default 16.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write metrics.jsonl + metrics_summary.json + "
                        "the FVD staging tree. Default: inside the run dir "
                        "(--run-dir). Set this to redirect artifacts away "
                        "from the inference run dir — e.g. for Marionette "
                        "evaluation, point at "
                        "outputs/test_metric/metrics/marionette/<dataset>/<protocol>/ "
                        "so outputs/marionette_eval/<run_dir>/ stays clean.")
    p.add_argument("--fvd-only", action="store_true",
                   help="Skip the per-sample loop. Load the existing "
                        "metrics_summary.json, compute only FVD on top, "
                        "merge it in, and rewrite. Use when the per-sample "
                        "work is already done and you're adding FVD after "
                        "the fact. Errors if no existing summary is found.")
    p.set_defaults(face_crop=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = EvalConfig(
        fps              = args.fps,
        resolution       = args.resolution,
        device           = args.device,
        fvd_models       = args.fvd_models,
        fvd_seq_len      = args.fvd_seq_len,
        lpips_chunk_size = args.lpips_chunk,
        face_crop        = args.face_crop,
        n_frames         = args.n_frames if args.n_frames > 0 else None,
        face_crop_margin = args.face_crop_margin,
    )
    summary = evaluate(args.run_dir, cfg,
                       skip_fvd=args.skip_fvd,
                       output_dir=args.output_dir,
                       fvd_only=args.fvd_only)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
