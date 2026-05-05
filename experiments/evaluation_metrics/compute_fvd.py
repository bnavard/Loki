r"""Driver for FVD — distribution-level visual quality on HDTF same-identity.

Why a separate driver: FVD is a single number per (bucket, dataset,
protocol). It can't decompose into a per-sample row, so it doesn't
fit `compute_metrics.py`'s per-sample evaluator pattern. This script
walks one or more bucket run dirs, computes FVD on each, and folds
the scalar back into the same
`outputs/test_metric/metrics/<bucket>/<dataset>/<protocol>/metrics_summary.json`
that `compute_metrics.py` writes — so the central comparison table
picks the FVD column up uniformly with the per-sample headlines.

Scope:
  * HDTF only (the only dataset for which we report FVD in the paper).
  * `same_identity_reconstruction` only — for cross-identity there is
    no aligned GT distribution to compare against.

What gets written:
  `metrics.fvd_videomae` for the default VideoMAE-v2 backbone, plus
  `metrics.fvd_i3d` if `--i3d` is passed. Both record `n_real`,
  `n_fake`, and `model` alongside the scalar so reviewers can sanity
  check sample counts.

Real-side reuse: the GT distribution is shared across every bucket
(it's just the HDTF GT clips). We compute real-side stats once at
the start of the run and reuse them for every fake bucket so the
2-GB VideoMAE forward pass over GT doesn't repeat per bucket.

Usage (from repo root):

    # All buckets, VideoMAE backbone (default).
    PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py

    # All buckets, both backbones (~2x compute on the fake side).
    PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py --i3d

    # One specific bucket.
    PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py \
        --run-dir outputs/marionette_eval/hdtf/same_identity_reconstruction/run_<ts>/

    # Re-use a previously cached GT-side stats blob (skip the 2-GB pass).
    PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py \
        --gt-stats-cache /tmp/fvd_gt_stats.npz

Pre-requisites: run `bash experiments/evaluation_metrics/setup_fvd.sh`
once to install `cdfvd`, apply the upstream-URL patch, and pre-stage
the VideoMAE-v2 SSv2-finetuned checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


OUT_ROOT          = Path("outputs/test_metric/metrics")
MARIONETTE_ROOT   = Path("outputs/marionette_eval")
SOTA_ROOT         = Path("outputs/sota_comparison")
ABLATION_ROOT     = Path("outputs/condition_ablation_eval")


def _bucket_for_run_dir(run_dir: Path) -> str:
    """Mirror `run_eval_metrics.sh`'s bucket-derivation rules so the
    metrics-tree path matches up exactly:
        marionette_eval/...                  → "marionette"
        sota_comparison/<baseline>/...       → "<baseline>"
        condition_ablation_eval/<arm>/...    → "marionette_<arm>_abl"
    """
    parts = run_dir.parts
    if "marionette_eval" in parts:
        return "marionette"
    if "sota_comparison" in parts:
        return parts[parts.index("sota_comparison") + 1]
    if "condition_ablation_eval" in parts:
        arm = parts[parts.index("condition_ablation_eval") + 1]
        return f"marionette_{arm}_abl"
    raise ValueError(f"cannot infer bucket from {run_dir}")


def _discover_run_dirs() -> list[Path]:
    """Walk the three roots for HDTF same-identity-reconstruction runs.
    When a (bucket, protocol) has multiple `run_<ts>/` dirs, keep only
    the latest (timestamps sort lexicographically). Without this, two
    runs of the same arm would both write FVD into the same
    `metrics_summary.json` and the second silently overwrite the first."""
    candidates: list[Path] = []
    for d in MARIONETTE_ROOT.glob("hdtf/same_identity_reconstruction/run_*/"):
        if d.is_dir(): candidates.append(d)
    for d in SOTA_ROOT.glob("*/hdtf/same_identity_reconstruction/run_*/"):
        if d.is_dir(): candidates.append(d)
    for d in ABLATION_ROOT.glob("*/hdtf/same_identity_reconstruction/run_*/"):
        if d.is_dir(): candidates.append(d)

    # Bucket → latest run dir (lexicographic max on the timestamp suffix).
    latest: dict[str, Path] = {}
    for rd in candidates:
        b = _bucket_for_run_dir(rd)
        prev = latest.get(b)
        if prev is None or rd.name > prev.name:
            latest[b] = rd
    return [latest[k] for k in sorted(latest)]


def _build_pred_dir(run_dir: Path, dest: Path) -> int:
    """Symlink every `samples/<sample_id>/panel.mp4` under `run_dir` into
    `dest`, named after the sample_id. Returns the number of links.
    `cdfvd.load_videos(data_type='video_folder')` walks `dest` non-
    recursively and reads each .mp4 as one clip."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for sample_dir in sorted((run_dir / "samples").iterdir()):
        if not sample_dir.is_dir(): continue
        panel = sample_dir / "panel.mp4"
        if not panel.is_file(): continue
        link = dest / f"{sample_dir.name}.mp4"
        link.symlink_to(panel.resolve())
        n += 1
    return n


def _build_gt_dir(meta, dest: Path) -> int:
    """Symlink the unique GT video for each sample (same-id ⇒
    ref_clip['video_path']) into `dest`. Returns the number of links."""
    from experiments.evaluation_metrics.metrics.io import iter_samples
    dest.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    for sample in iter_samples(meta):
        gt_path = Path(sample.ref_clip["video_path"]).resolve()
        if str(gt_path) in seen: continue
        seen.add(str(gt_path))
        link = dest / f"{sample.ref_clip['clip_id']}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(gt_path)
        n += 1
    return n


def _summary_path_for_run(run_dir: Path) -> Path:
    bucket = _bucket_for_run_dir(run_dir)
    # `<bucket>/hdtf/same_identity_reconstruction/metrics_summary.json`
    return (OUT_ROOT / bucket / "hdtf" / "same_identity_reconstruction"
            / "metrics_summary.json")


def _merge_into_summary(summary_path: Path, key: str, payload: dict) -> None:
    """Read `metrics_summary.json` (if present), set
    `summary['metrics'][key] = payload`, write back. Creates parents and
    minimal scaffolding if the file doesn't exist yet."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = (json.loads(summary_path.read_text())
                     if summary_path.is_file() else {})
    metrics_block = summary.setdefault("metrics", {})
    metrics_block[key] = payload
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _evaluate_bucket(
    fvd_evaluator,
    run_dir:       Path,
    pred_dir:      Path,
    sequence_length: int,
    resolution:    int,
    backbone:      str,
) -> dict:
    """Run the fake-side stats + Fréchet-from-stats step for one bucket.
    Real-side stats are assumed to already be loaded into `fvd_evaluator`."""
    # `FeatureStats` is the accumulator cdfvd's `compute_fake_stats`
    # appends features into; cdfvd's __init__ creates one in
    # `self.fake_stats`. To run multiple buckets in one process we have
    # to reset the fake-side container between buckets — but we have to
    # rebuild a fresh `FeatureStats` (mirroring cdfvd/fvd.py:68), not
    # set it to None, or the next `compute_fake_stats` dies on
    # `self.fake_stats.max_items` access.
    from cdfvd.utils.metric_utils import FeatureStats
    fvd_evaluator.evaluator.fake_stats = FeatureStats(
        max_items=None,            # `n_fake='full'` matches our FVD ctor
        capture_mean_cov=True,
        capture_all=False,         # `compute_feats=False` matches our FVD ctor
    )

    fake_loader = fvd_evaluator.evaluator.load_videos(
        str(pred_dir), data_type="video_folder",
        resolution=resolution, sequence_length=sequence_length,
    )
    n_fake = (len(fake_loader.dataset)
              if hasattr(fake_loader, "dataset") else -1)
    fvd_evaluator.evaluator.compute_fake_stats(fake_loader)
    score = float(fvd_evaluator.evaluator.compute_fvd_from_stats())
    return {
        "fvd":    score,
        "model":  backbone,
        "n_real": int(getattr(fvd_evaluator.evaluator, "n_real_used", -1)),
        "n_fake": int(n_fake),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute FVD for HDTF same-identity-reconstruction runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Single run dir to evaluate. If omitted, walks all "
                        "discovered HDTF same-id run dirs.")
    p.add_argument("--i3d", action="store_true",
                   help="Also compute FVD with the I3D backbone "
                        "(default is VideoMAE-v2 only).")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--resolution", type=int, default=224,
                   help="Spatial resolution fed into the FVD backbone.")
    p.add_argument("--sequence-length", type=int, default=16,
                   help="Per-clip frame count fed into the FVD backbone.")
    args = p.parse_args()

    from experiments.evaluation_metrics.metrics.io import load_run_metadata
    from experiments.evaluation_metrics.metrics.src.fvd import FVD

    run_dirs = ([args.run_dir] if args.run_dir is not None
                else _discover_run_dirs())
    if not run_dirs:
        print("[fvd] no HDTF same-identity-reconstruction runs found.")
        return 0

    print(f"[fvd] evaluating {len(run_dirs)} run dir(s):")
    for d in run_dirs:
        print(f"        {d}")

    backbones = ["videomae"] + (["i3d"] if args.i3d else [])

    with tempfile.TemporaryDirectory(prefix="fvd_") as tmp:
        tmp = Path(tmp)
        # GT side is the same for every bucket — symlink once from any run.
        gt_dir = tmp / "gt"
        meta_first = load_run_metadata(run_dirs[0])
        n_gt = _build_gt_dir(meta_first, gt_dir)
        print(f"[fvd] GT clips staged: {n_gt} (under {gt_dir})")

        for backbone in backbones:
            print(f"\n[fvd] backbone={backbone}")
            fvd = FVD(model=backbone, resolution=args.resolution,
                      sequence_length=args.sequence_length, device=args.device)

            real_loader = fvd.evaluator.load_videos(
                str(gt_dir), data_type="video_folder",
                resolution=args.resolution, sequence_length=args.sequence_length,
            )
            fvd.evaluator.compute_real_stats(real_loader)
            # `compute_real_stats` doesn't set this; we record it for the
            # summary payload.
            fvd.evaluator.n_real_used = (
                len(real_loader.dataset)
                if hasattr(real_loader, "dataset") else n_gt
            )

            for rd in run_dirs:
                bucket = _bucket_for_run_dir(rd)
                pred_dir = tmp / f"pred_{bucket}_{backbone}"
                n_fake = _build_pred_dir(rd, pred_dir)
                if n_fake == 0:
                    print(f"[fvd]   [SKIP] {bucket}: no panel.mp4 found")
                    continue
                payload = _evaluate_bucket(
                    fvd, rd, pred_dir,
                    sequence_length = args.sequence_length,
                    resolution      = args.resolution,
                    backbone        = backbone,
                )
                summary_key = f"fvd_{backbone}"
                summary_path = _summary_path_for_run(rd)
                _merge_into_summary(summary_path, summary_key, payload)
                print(f"[fvd]   [OK] {bucket}: {summary_key}={payload['fvd']:.2f} "
                      f"(n_real={payload['n_real']}, n_fake={payload['n_fake']}) "
                      f"→ {summary_path}")

    print("\n[fvd] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())