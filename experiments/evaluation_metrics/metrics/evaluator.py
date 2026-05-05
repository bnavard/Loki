"""Unified protocol-aware evaluator with metric-group abstraction.

Walks one `<run_dir>` produced by the SOTA-comparison or marionette-eval
runners and routes each prediction through the metric set appropriate
to its protocol:

| Protocol                          | Metric groups                                       |
|-----------------------------------|-----------------------------------------------------|
| `same_identity_reconstruction`    | head_rot, expression, psnr, ssim, lpips             |
| `cross_identity`                  | head_rot, expression, id                            |

The pixel-aligned groups (`psnr`, `ssim`, `lpips`) are HDTF-only — they
require pred and GT to be pixel-aligned frame-for-frame, which is only
meaningful on the same-identity protocol; and HDTF is the only dataset
on which we report them. They are silently skipped for any other
(dataset, protocol) combination by the HDTF gate inside `_resolve_groups`.

FVD is **distribution-level** and lives outside this evaluator — see
`compute_fvd.py` for the separate driver. It can't decompose to a
per-sample row, so it doesn't fit the per-sample pattern this module
implements.

Per-sample metrics land at `<output_dir>/metrics.jsonl` (one JSON row per
sample); aggregates at `<output_dir>/metrics_summary.json`.

Metric modes (`cfg.metrics_mode`)
---------------------------------
* `"auto"` (default) — load the existing summary and compute only the
  groups whose headline metric isn't in it. Existing per-sample fields in
  `metrics.jsonl` are preserved by merging — only newly-computed group
  fields are overwritten.
* `"all"` — recompute every group available for the protocol, overwriting
  every existing field.
* explicit `set[str]`, e.g. `{"head_rot", "expression"}` — recompute only
  those groups, overwrite their fields, leave others alone.

Group → headline metric mapping (`GROUP_HEADLINE_METRIC`) drives the
`auto`-mode missing-detection check.

Per-frame handling
------------------
Head-rot / expression read FLAME parameters from `fit.npz`; a sample
contributes when both pred and target fits exist with ≥2 frames, and
`*_track_rate = T_used / n_frames` records how much of the requested
window was usable.

Identity (cross-id only) needs per-frame ArcFace detection: a frame
contributes only when both the ref-clip prior and the pred frame embed
successfully. Per-sample number is the mean over hits;
`id_detect_rate` records failure density.

Run-level aggregation
---------------------
All three metrics use a **weighted** mean — each sample's track rate
(head_rot, expression) or detect rate (id) is the weight, so a sample
whose number was computed on 5/16 frames contributes proportionally less
than one computed on 16/16.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from .io import (
    DEFAULT_FPS, DEFAULT_RESOLUTION,
    RunMetadata, load_run_metadata, iter_samples, load_video,
)


# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------


GROUPS_BY_PROTOCOL: dict[str, list[str]] = {
    "same_identity_reconstruction": [
        "head_rot", "expression", "psnr", "ssim", "lpips",
    ],
    "cross_identity":               ["head_rot", "expression", "id"],
}

# Pixel-aligned groups are reported on HDTF only; on any other dataset
# they are silently dropped from the resolved set.
HDTF_ONLY_GROUPS: set[str] = {"psnr", "ssim", "lpips"}

# Detect "missing in summary" via these headline keys.
GROUP_HEADLINE_METRIC: dict[str, str] = {
    "head_rot":   "head_rot_dist",
    "expression": "expression_l1",
    "id":         "id_cosine",
    "psnr":       "psnr_db",
    "ssim":       "ssim",
    "lpips":      "lpips",
}

# Run-level aggregation: which metrics use a weighted mean and which
# per-sample list supplies the weights. The three pixel-aligned metrics
# share `_pixel_weights` (`pixel_track_rate`), since they all read the
# same pred / GT video and are limited by the same per-sample frame
# coverage.
WEIGHTED_METRICS: dict[str, str] = {
    "id_cosine":     "_id_weights",
    "head_rot_dist": "_head_rot_weights",
    "expression_l1": "_expression_weights",
    "psnr_db":       "_pixel_weights",
    "ssim":          "_pixel_weights",
    "lpips":         "_pixel_weights",
}


def _group_present(group: str, summary: dict) -> bool:
    """Whether `summary` already carries the headline value for `group`."""
    if not summary:
        return False
    return GROUP_HEADLINE_METRIC[group] in summary.get("metrics", {})


def _resolve_groups(
    mode: Union[str, set[str]],
    summary: dict,
    protocol: str,
    dataset: Optional[str] = None,
) -> set[str]:
    """Translate the user's `metrics_mode` into a concrete set of groups
    to compute, filtered by what's available for the protocol and by
    the HDTF-only gate for pixel-aligned groups."""
    available = set(GROUPS_BY_PROTOCOL[protocol])
    if dataset != "hdtf":
        available = available - HDTF_ONLY_GROUPS
    if mode == "all":
        return available
    if isinstance(mode, set):
        unknown = mode - available
        if unknown:
            print(f"[evaluate] groups {sorted(unknown)} not applicable to "
                  f"protocol={protocol} dataset={dataset}; ignored.")
        return mode & available
    if mode == "auto":
        return {g for g in available if not _group_present(g, summary)}
    raise ValueError(f"unknown metrics_mode: {mode!r}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    fps:        int = DEFAULT_FPS
    resolution: int = DEFAULT_RESOLUTION
    device:     str = "cuda"
    n_frames:   Optional[int] = 16

    # Metric mode: "auto" (default), "all", or a set[str] of group names.
    metrics_mode: Union[str, set[str]] = "auto"


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------


def _read_existing_rows(metrics_path: Path) -> dict[str, dict]:
    """Read prior `metrics.jsonl` (if present) into a dict keyed by
    sample_id, so we can merge new fields per sample without losing
    fields produced by earlier runs."""
    if not metrics_path.is_file():
        return {}
    out = {}
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("sample_id")
        if sid:
            out[sid] = row
    return out


def _derive_bucket(run_dir: Path) -> str:
    """Recover the FLAME-tracking-preds subdirectory name from a run-dir path.
    Used to look up cached predicted FLAME fits under
    `data/flame_tracking/preds/<bucket>/<dataset>/<protocol>/<sample_id>/fit.npz`.
    `outputs/marionette_eval/...`                  → "marionette".
    `outputs/sota_comparison/<baseline>/...`       → "<baseline>".
    `outputs/condition_ablation_eval/<arm>/...`    → "<arm>"
        (no_posenc / no_deform / flame_vector). FLAME tracking writes to
        `data/flame_tracking/preds/<arm>/...`, so the arm name is the
        right pred-subdir even though the metrics-tree bucket for
        ablations is `marionette_<arm>_abl` (set in run_eval_metrics.sh)."""
    parts = run_dir.parts
    if "marionette_eval" in parts:
        return "marionette"
    if "sota_comparison" in parts:
        return parts[parts.index("sota_comparison") + 1]
    if "condition_ablation_eval" in parts:
        return parts[parts.index("condition_ablation_eval") + 1]
    raise ValueError(f"cannot infer bucket from run_dir={run_dir}")


def _gt_fit_root() -> Path:
    """Where ground-truth FLAME fits live."""
    return Path("data/benchmark/hdtf/flame_tracking/flowface")


def _pred_fit_path(run_dir: Path, dataset: str, protocol: str,
                   sample_id: str) -> Path:
    return (Path("data/flame_tracking/preds") / _derive_bucket(run_dir)
            / dataset / protocol / sample_id / "fit.npz")


def _compute_expression_pair(pred_fit_path: Path, target_fit_path: Path,
                             expr_metric, n_frames: int):
    """Returns (l1, weight). Weight = T_used / n_frames in [0, 1]; l1 is
    None and weight is 0 when either fit file is missing or empty."""
    if not pred_fit_path.is_file() or not target_fit_path.is_file():
        return None, 0.0
    pred_fit   = dict(np.load(str(pred_fit_path)))
    target_fit = dict(np.load(str(target_fit_path)))
    if pred_fit["expr"].shape[0] == 0 or target_fit["expr"].shape[0] == 0:
        return None, 0.0
    out = expr_metric.compute_pair(pred_fit, target_fit, n_frames=n_frames)
    weight = out["n_frames"] / max(n_frames, 1)
    return out["l1"], float(weight)


def _compute_pixel_pair(
    pred_path: Path,
    gt_path:   Path,
    n_frames:  int,
    fps:       int,
    resolution: int,
    psnr_fn,
    ssim_fn,
    lpips_fn,
) -> dict:
    """Run the three pixel-aligned metrics on one pred / GT pair, sharing
    a single video decode. Any of the `*_fn` may be `None` if the caller
    isn't computing that metric.

    Returns
    -------
    {
      "psnr_db":          float | None,
      "ssim":             float | None,
      "lpips":            float | None,
      "pixel_track_rate": float,        # T_used / n_frames
      "n_frames":         int,
    }
    """
    if not pred_path.is_file() or not gt_path.is_file():
        return {"psnr_db": None, "ssim": None, "lpips": None,
                "pixel_track_rate": 0.0, "n_frames": 0}

    pred = load_video(pred_path, fps, resolution, max_frames=n_frames)
    gt   = load_video(gt_path,   fps, resolution, max_frames=n_frames)
    T = int(min(pred.shape[0], gt.shape[0]))
    if T < 1:
        return {"psnr_db": None, "ssim": None, "lpips": None,
                "pixel_track_rate": 0.0, "n_frames": 0}
    pred = pred[:T].unsqueeze(0)   # (1, T, 3, H, W)
    gt   = gt  [:T].unsqueeze(0)

    out: dict = {"pixel_track_rate": float(T) / max(n_frames, 1), "n_frames": T}
    out["psnr_db"] = float(psnr_fn(pred, gt)[0].item()) if psnr_fn else None
    out["ssim"]    = float(ssim_fn(pred, gt)[0].item()) if ssim_fn else None
    # LPIPS is a class instance with __call__, not a free function — but it
    # consumes the same shape contract.
    out["lpips"]   = float(lpips_fn(pred, gt)[0].item()) if lpips_fn else None
    return out


def _compute_head_rot_pair(pred_fit_path: Path, target_fit_path: Path,
                           n_frames: int):
    """Returns (dist, weight). Weight = T_used / n_frames in [0, 1]; dist
    is None and weight is 0 when either fit is missing or has < 2
    frames. dist is the geodesic angular distance between pred's and
    target's frame-0-anchored pose-trajectory deltas, in degrees,
    averaged over the available frames — see head_rot.py for details."""
    if not pred_fit_path.is_file() or not target_fit_path.is_file():
        return None, 0.0
    pred_fit   = dict(np.load(str(pred_fit_path)))
    target_fit = dict(np.load(str(target_fit_path)))
    from .src.head_rot import compute_head_rot_pair
    out = compute_head_rot_pair(pred_fit, target_fit, n_frames=n_frames)
    return out["dist"], float(out["track_rate"])


# ---------------------------------------------------------------------------
# Same-identity per-sample loop
# ---------------------------------------------------------------------------


def _eval_same_identity(
    meta:          RunMetadata,
    cfg:           EvalConfig,
    metrics_path:  Path,
    groups:        set[str],
    existing_rows: dict[str, dict],
) -> dict[str, list]:
    """Same-identity: head_rot and expression read FLAME fits (no video
    decode); psnr/ssim/lpips read pred and GT videos and compute
    pixel-aligned scores via a shared decode. The pixel groups are
    HDTF-only — the gate in `_resolve_groups` strips them on any other
    dataset before this loop runs."""
    expr_metric = None
    if "expression" in groups:
        from .src.expression import ExpressionDeformationDiff
        expr_metric = ExpressionDeformationDiff(
            image_size=cfg.resolution, device=cfg.device,
        )
    # Lazy-load LPIPS so head_rot-only / expression-only invocations
    # don't pay the AlexNet load cost.
    lpips_metric = None
    if "lpips" in groups:
        from .src.lpips import LPIPSMetric
        lpips_metric = LPIPSMetric(net="alex", device=cfg.device)
    psnr_fn = ssim_fn = None
    if "psnr" in groups:
        from .src.psnr import psnr_video
        psnr_fn = psnr_video
    if "ssim" in groups:
        from .src.ssim import ssim_video
        ssim_fn = ssim_video
    pixel_groups_active = bool({"psnr", "ssim", "lpips"} & groups)

    results: dict[str, list] = {}
    if "head_rot" in groups:
        results.update({
            "head_rot_dist":       [],
            "head_rot_track_rate": [],
            "_head_rot_weights":   [],
        })
    if "expression" in groups:
        results.update({
            "expression_l1":       [],
            "_expression_weights": [],
        })
    if pixel_groups_active:
        results["_pixel_weights"] = []
    if "psnr"  in groups: results["psnr_db"] = []
    if "ssim"  in groups: results["ssim"]    = []
    if "lpips" in groups: results["lpips"]   = []

    out_rows: list[dict] = []
    desc = f"same-id ({','.join(sorted(groups))})"

    for sample in tqdm(list(iter_samples(meta)), desc=desc):
        existing = existing_rows.get(sample.sample_id, {})
        new_fields: dict = {"sample_id": sample.sample_id}

        if "head_rot" in groups:
            target_fit = _gt_fit_root() / sample.ref_clip["clip_id"] / "fit.npz"
            pred_fit   = _pred_fit_path(meta.run_dir, meta.dataset, meta.protocol,
                                        sample.sample_id)
            dist, w = _compute_head_rot_pair(pred_fit, target_fit,
                                             n_frames=cfg.n_frames or 16)
            if dist is not None:
                results["head_rot_dist"]    .append(dist)
                results["_head_rot_weights"].append(w)
            results["head_rot_track_rate"].append(w)
            new_fields["head_rot_dist"]       = dist
            new_fields["head_rot_track_rate"] = w

        if "expression" in groups:
            target_fit = _gt_fit_root() / sample.ref_clip["clip_id"] / "fit.npz"
            pred_fit   = _pred_fit_path(meta.run_dir, meta.dataset, meta.protocol,
                                        sample.sample_id)
            l1, w = _compute_expression_pair(
                pred_fit, target_fit, expr_metric,
                n_frames=cfg.n_frames or 16,
            )
            if l1 is not None:
                results["expression_l1"]      .append(l1)
                results["_expression_weights"].append(w)
            new_fields["expression_l1"] = l1

        if pixel_groups_active:
            # Same-identity ⇒ ref_clip == driver_clip; either points at
            # the GT video. One decode, three metrics share it.
            gt_path = Path(sample.ref_clip["video_path"])
            pix = _compute_pixel_pair(
                pred_path  = sample.pred_path,
                gt_path    = gt_path,
                n_frames   = cfg.n_frames or 16,
                fps        = cfg.fps,
                resolution = cfg.resolution,
                psnr_fn    = psnr_fn,
                ssim_fn    = ssim_fn,
                lpips_fn   = lpips_metric,
            )
            w_pix = float(pix["pixel_track_rate"])
            for k_grp, k_field in (("psnr", "psnr_db"),
                                    ("ssim", "ssim"),
                                    ("lpips", "lpips")):
                if k_grp in groups:
                    new_fields[k_field] = pix[k_field]
                    if pix[k_field] is not None:
                        results[k_field].append(pix[k_field])
            new_fields["pixel_track_rate"] = w_pix
            # The shared weights list has one entry per *successful*
            # pixel evaluation (T_used >= 1). All three metrics are
            # produced together or not at all, so a single weights list
            # serves all three aggregates.
            if pix["n_frames"] >= 1:
                results["_pixel_weights"].append(w_pix)

        merged = {**existing, **new_fields}
        merged.pop("skipped", None)
        out_rows.append(merged)

    metrics_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    return results


# ---------------------------------------------------------------------------
# Cross-identity per-sample loop
# ---------------------------------------------------------------------------


def _eval_cross_identity(
    meta:          RunMetadata,
    cfg:           EvalConfig,
    metrics_path:  Path,
    groups:        set[str],
    existing_rows: dict[str, dict],
) -> dict[str, list]:
    """For cross-id: id_cosine uses the ref clip's video (identity prior);
    head_rot and expression read FLAME fits (no video / face crop needed)."""
    id_metric = None
    if "id" in groups:
        from .src.id_sim import IDSimilarity
        id_metric = IDSimilarity(device=cfg.device)
    expr_metric = None
    if "expression" in groups:
        from .src.expression import ExpressionDeformationDiff
        expr_metric = ExpressionDeformationDiff(
            image_size=cfg.resolution, device=cfg.device,
        )

    results: dict[str, list] = {}
    if "id" in groups:
        results.update({"id_cosine": [], "id_detect_rate": [], "_id_weights": []})
    if "head_rot" in groups:
        results.update({
            "head_rot_dist":       [],
            "head_rot_track_rate": [],
            "_head_rot_weights":   [],
        })
    if "expression" in groups:
        results.update({
            "expression_l1":       [],
            "_expression_weights": [],
        })

    prior_cache: dict = {}
    out_rows: list[dict] = []
    desc = f"cross-id ({','.join(sorted(groups))})"

    for sample in tqdm(list(iter_samples(meta)), desc=desc):
        existing = existing_rows.get(sample.sample_id, {})
        new_fields: dict = {
            "sample_id":  sample.sample_id,
            "ref_uid":    sample.ref_clip   ["uid"],
            "driver_uid": sample.driver_clip["uid"],
        }

        # ---- ID similarity: ref-clip prior + ArcFace on pred frames. ----
        if "id" in groups:
            ref_id = sample.ref_clip["uid"]
            if ref_id not in prior_cache:
                ref_video = load_video(
                    Path(sample.ref_clip["video_path"]), cfg.fps, cfg.resolution,
                )
                prior_cache[ref_id] = id_metric.build_identity_prior(ref_video)
            prior = prior_cache[ref_id]
            if prior is None:
                # Couldn't build identity prior — record skip-style data
                # for `id` only; head_rot / expression can still try below.
                new_fields["id_cosine"]      = None
                new_fields["id_detect_rate"] = 0.0
            else:
                gen = load_video(sample.pred_path, cfg.fps, cfg.resolution,
                                 max_frames=cfg.n_frames)
                cos, det = id_metric.score(gen, prior)
                cos_val  = None if np.isnan(cos) else float(cos)
                new_fields["id_cosine"]      = cos_val
                new_fields["id_detect_rate"] = float(det)
                results["id_detect_rate"].append(float(det))
                if cos_val is not None:
                    results["id_cosine"]   .append(cos_val)
                    results["_id_weights"] .append(float(det))

        # ---- Head rotation: pred fit vs driver fit (motion source). ----
        if "head_rot" in groups:
            target_fit = _gt_fit_root() / sample.driver_clip["clip_id"] / "fit.npz"
            pred_fit   = _pred_fit_path(meta.run_dir, meta.dataset, meta.protocol,
                                        sample.sample_id)
            dist, w = _compute_head_rot_pair(pred_fit, target_fit,
                                             n_frames=cfg.n_frames or 16)
            if dist is not None:
                results["head_rot_dist"]    .append(dist)
                results["_head_rot_weights"].append(w)
            results["head_rot_track_rate"].append(w)
            new_fields["head_rot_dist"]       = dist
            new_fields["head_rot_track_rate"] = w

        # ---- Expression: pred fit vs driver fit (motion source). ----
        if "expression" in groups:
            target_fit = _gt_fit_root() / sample.driver_clip["clip_id"] / "fit.npz"
            pred_fit   = _pred_fit_path(meta.run_dir, meta.dataset, meta.protocol,
                                        sample.sample_id)
            l1, w = _compute_expression_pair(
                pred_fit, target_fit, expr_metric,
                n_frames=cfg.n_frames or 16,
            )
            if l1 is not None:
                results["expression_l1"]      .append(l1)
                results["_expression_weights"].append(w)
            new_fields["expression_l1"] = l1

        merged = {**existing, **new_fields}
        # Stale skip flag should drop only if we successfully computed
        # something *new* for this sample. For cross-id, all three groups
        # can independently fail; we only clear the flag if at least one
        # succeeded.
        if (
            ("id" in groups and merged.get("id_cosine") is not None)
            or ("head_rot" in groups
                and merged.get("head_rot_dist") is not None)
            or ("expression" in groups
                and merged.get("expression_l1") is not None)
        ):
            merged.pop("skipped", None)
        out_rows.append(merged)

    metrics_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(per_sample: dict[str, list]) -> dict[str, dict[str, float]]:
    """Per-metric aggregation. All three headline metrics use a weighted
    mean with each sample's detect / track rate as the weight."""
    aggregates: dict[str, dict[str, float]] = {}
    for name, vals in per_sample.items():
        if name.startswith("_"):
            continue                # internal weight tracker, not aggregated directly
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        if name in WEIGHTED_METRICS:
            w_key   = WEIGHTED_METRICS[name]
            weights = np.array(per_sample.get(w_key, []), dtype=np.float64)
            if weights.size == arr.size and weights.sum() > 0:
                w_mean = float((weights * arr).sum() / weights.sum())
                if arr.size > 1:
                    w_var = ((weights * (arr - w_mean) ** 2).sum() / weights.sum())
                    w_std = float(np.sqrt(w_var))
                else:
                    w_std = 0.0
                aggregates[name] = {
                    "mean":        w_mean,
                    "std":         w_std,
                    "n":           int(arr.size),
                    "weighted_by": w_key.lstrip("_"),
                }
                continue
        aggregates[name] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
            "n":    int(arr.size),
        }
    return aggregates


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(
    run_dir:    Path,
    cfg:        Optional[EvalConfig] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Evaluate one run dir, computing metric groups according to
    `cfg.metrics_mode` (default `"auto"` — only groups missing from
    the existing summary). Writes:

      * `<output_dir>/metrics.jsonl`         — one row per sample (merged
                                                with prior rows when not
                                                fully recomputing)
      * `<output_dir>/metrics_summary.json`  — aggregates

    `output_dir` lets callers redirect every metric artifact away from
    the inference run dir. When None, artifacts go inside `meta.run_dir`.

    Returns the summary dict (also written to disk).
    """
    cfg  = cfg or EvalConfig()
    meta = load_run_metadata(run_dir)

    if output_dir is None:
        out_dir = meta.run_dir
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "metrics_summary.json"

    existing_summary = (
        json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    )
    groups = _resolve_groups(cfg.metrics_mode, existing_summary,
                             meta.protocol, meta.dataset)

    if not groups:
        print(f"[evaluate] {meta.run_dir} — nothing to compute "
              f"(all groups present, mode={cfg.metrics_mode!r}).")
        return existing_summary

    print(f"[evaluate] {meta.run_dir} — computing groups: {sorted(groups)}")

    # `all` mode means full overwrite; ignore any existing per-sample rows.
    existing_rows = ({} if cfg.metrics_mode == "all"
                     else _read_existing_rows(metrics_path))

    if meta.protocol == "same_identity_reconstruction":
        per_sample = _eval_same_identity(meta, cfg, metrics_path,
                                         groups, existing_rows)
    elif meta.protocol == "cross_identity":
        per_sample = _eval_cross_identity(meta, cfg, metrics_path,
                                          groups, existing_rows)
    else:
        raise ValueError(f"Unknown protocol: {meta.protocol}")

    new_aggregates = _aggregate(per_sample)

    # Merge aggregates: keep existing for groups not recomputed; overwrite
    # for newly-computed metrics. `all` mode produces every metric, so
    # every existing aggregate gets overwritten naturally.
    aggregates = (dict(existing_summary.get("metrics", {}))
                  if cfg.metrics_mode != "all" else {})
    aggregates.update(new_aggregates)

    summary: dict = {
        "run_dir":   str(meta.run_dir),
        "dataset":   meta.dataset,
        "protocol":  meta.protocol,
        "n_samples": len(list(iter_samples(meta))),
        "metrics":   aggregates,
    }

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
