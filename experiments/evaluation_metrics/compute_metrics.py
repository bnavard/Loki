r"""CLI entry point for the metrics package.

Auto-detects protocol from `<run_dir>/config_resolved.json` and computes
the metric set appropriate for that protocol:

  * `same_identity_reconstruction`:
        head_rot    (FLAME-native geodesic angular distance over
                     `rot · neck_rot` deltas, vs the GT clip's fit)
        expression  (FLAME deformation-map L1, pose-disentangled,
                     vs the GT clip's fit)
  * `cross_identity`:
        head_rot    (vs driver clip's FLAME fit)
        expression  (vs driver clip's FLAME fit)
        id          (ArcFace cosine vs ref-clip prior)

Outputs at `<output_dir>/metrics.jsonl` (per-sample) and
`<output_dir>/metrics_summary.json` (aggregates).

Metric-mode semantics (`--metrics MODE`)
----------------------------------------
* `auto`  (default) — load existing summary, compute only the groups
                      whose headline metric isn't already there.
                      Existing per-sample fields in metrics.jsonl are
                      preserved by merging.
* `all`              — recompute every group available for the protocol;
                      every existing field is overwritten.
* explicit list      — comma-separated, e.g. `--metrics head_rot`.
                      Recompute only those, overwrite their fields, leave
                      every other group's existing fields alone.

Usage
-----

    # First-pass evaluation (auto = full sweep on a fresh dir):
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir outputs/sota_comparison/sadtalker/hdtf/same_identity_reconstruction/run_<ts>/

    # Add head_rot to a run that already has expression:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir <…> --metrics head_rot

    # Full overwrite:
    PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
        --run-dir <…> --metrics all
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
                        "from {head_rot, expression, id}.")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--fps",        type=int, default=25,
                   help="Target fps for ArcFace ref-clip frame alignment.")
    p.add_argument("--resolution", type=int, default=512,
                   help="Resolution at which the FLAME-rasterised expression "
                        "metric and the ArcFace-prior video are processed.")
    p.add_argument("--n-frames",   type=int, default=16,
                   help="Cap pred to this many frames so SOTA's 75–125-frame "
                        "outputs are scored on the same temporal coverage as "
                        "Loki's 16-frame panel. Set to 0 for tool-native length.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write metrics.jsonl + metrics_summary.json. "
                        "Default: inside the run dir.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = EvalConfig(
        fps          = args.fps,
        resolution   = args.resolution,
        device       = args.device,
        n_frames     = args.n_frames if args.n_frames > 0 else None,
        metrics_mode = args.metrics,
    )
    summary = evaluate(args.run_dir, cfg, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
