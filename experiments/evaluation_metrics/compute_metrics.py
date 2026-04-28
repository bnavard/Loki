r"""CLI entry point for the metrics package.

Auto-detects protocol + dataset from `<run_dir>/config_resolved.json`
and computes the metric set appropriate for that protocol:

  * `same_identity_reconstruction`:
        pixel       (PSNR / SSIM / LPIPS)
        lmd         (LMD-F / LMD-M)
        head_rot    (geodesic angular distance over FLAME `rot · neck_rot`
                     deltas, vs driver=GT)
        expression  (FLAME deformation-map L2, pose-disentangled vs driver=GT)
        fvd         (videomae backbone, distribution-level)
  * `cross_identity`:
        head_rot    (vs driver clip's FLAME fit)
        expression  (vs driver clip's FLAME fit)
        id          (ArcFace cosine vs ref-clip prior)

Outputs at `<output_dir>/metrics.jsonl` (per-sample) and
`<output_dir>/metrics_summary.json` (aggregates + fvd).

Metric-mode semantics (`--metrics MODE`)
----------------------------------------
* `auto`  (default) — load existing summary, compute only the groups
                      whose headline metric isn't already there.
                      Existing per-sample fields in metrics.jsonl are
                      preserved by merging.
* `all`              — recompute every group available for the protocol;
                      every existing field is overwritten.
* explicit list      — comma-separated, e.g. `--metrics head_rot,fvd`.
                      Recompute only those, overwrite their fields, leave
                      every other group's existing fields alone.

Usage
-----

    # First-pass evaluation (auto = full sweep on a fresh dir):
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir outputs/sota_comparison/sadtalker/talkvid/same_identity_reconstruction/run_<ts>/

    # Add head_rot + expression to a run that already has pixel/lmd/fvd:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir <…> --metrics head_rot,expression

    # Full overwrite:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir <…> --metrics all

    # Both FVD backbones (default is videomae only):
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir <…> --metrics fvd --fvd-models videomae i3d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.evaluation_metrics.metrics.evaluator import (
    EvalConfig, GROUPS_BY_PROTOCOL, evaluate,
)


def _parse_metrics_mode(raw: str):
    """`raw` is `auto`, `all`, or a comma-separated list of group names.
    Returns `'auto' | 'all' | set[str]`."""
    raw = raw.strip().lower()
    if raw in ("auto", "all"):
        return raw
    items = {tok.strip() for tok in raw.split(",") if tok.strip()}
    if not items:
        raise argparse.ArgumentTypeError("`--metrics` is empty")
    valid = set().union(*GROUPS_BY_PROTOCOL.values())
    unknown = items - valid
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown metric group(s): {sorted(unknown)}. "
            f"Valid: {sorted(valid)} | auto | all"
        )
    return items


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute evaluation metrics for one run dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run dir, e.g. outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>/")
    p.add_argument("--metrics", default="auto", type=_parse_metrics_mode,
                   help="`auto` (default — top up missing groups), `all` "
                        "(full overwrite), or comma-separated group names "
                        "from {pixel, lmd, head_rot, expression, id, fvd}.")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--fps",              type=int, default=25,
                   help="Target fps for paired-metric frame alignment.")
    p.add_argument("--resolution",       type=int, default=512,
                   help="Resolution at which paired metrics are computed.")
    p.add_argument("--fvd-models",       nargs="+", default=["videomae"],
                   choices=["videomae", "i3d"],
                   help="FVD backbones to run. Default: videomae only.")
    p.add_argument("--fvd-seq-len",      type=int, default=16)
    p.add_argument("--lpips-chunk",      type=int, default=64,
                   help="Inner-batch size for LPIPS — drop if you OOM.")
    p.add_argument("--no-face-crop", dest="face_crop", action="store_false",
                   help="Disable face-cropping. Default: enabled — pred "
                        "and target are independently cropped to a tight "
                        "square around the detected face (1.3× bbox), then "
                        "resized to --resolution.")
    p.add_argument("--face-crop-margin", type=float, default=1.3,
                   help="Padding multiplier around the raw RetinaFace bbox.")
    p.add_argument("--n-frames",         type=int, default=16,
                   help="Cap pred (and therefore GT/driver) to this many "
                        "frames so SOTA's 75–125 frame outputs are scored "
                        "on the same temporal coverage as Marionette's "
                        "16-frame panel. Set to 0 for tool-native length.")
    p.add_argument("--output-dir",       type=Path, default=None,
                   help="Where to write metrics.jsonl + metrics_summary.json + "
                        "(transient) FVD staging. Default: inside the run dir.")
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
        metrics_mode     = args.metrics,
    )
    summary = evaluate(args.run_dir, cfg, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
