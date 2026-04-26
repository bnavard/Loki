r"""CLI entry point for the metrics package.

Auto-detects protocol + dataset from `<run_dir>/run_args.json` and routes
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
    )
    summary = evaluate(args.run_dir, cfg, skip_fvd=args.skip_fvd)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
