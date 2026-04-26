"""Unified protocol-aware evaluator.

Walks one `<run_dir>` produced by the SOTA-comparison or marionette-eval
runners (`outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>/`
or `outputs/marionette_eval/<protocol>/run_<ts>/`) and routes each
prediction through the metric set appropriate to its protocol:

| Protocol                          | Per-sample metrics             | Distribution metrics |
|-----------------------------------|--------------------------------|----------------------|
| `same_identity_reconstruction`    | PSNR, SSIM, LPIPS, LMD-F, LMD-M | FVD                  |
| `cross_identity`                  | ID similarity (ArcFace cosine)  | FVD                  |

Per-sample metrics are emitted as one JSON line per sample to
`<run_dir>/metrics.jsonl`. Distribution metrics + aggregates are written to
`<run_dir>/metrics_summary.json`.

FVD is sample-size sensitive: I3D wants ≥ 2k clips for stability. The
talking-head benchmarks here are 125 / 212 clips, so the summary tags
`low_sample = True` whenever the count is below the threshold.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .io import (
    DEFAULT_FPS, DEFAULT_RESOLUTION,
    RunMetadata, SamplePair,
    load_run_metadata, iter_samples, load_video, truncate_to_match,
)
from .psnr import psnr_video
from .ssim import ssim_video


FVD_LOW_SAMPLE_THRESHOLD = 2000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    fps:              int  = DEFAULT_FPS
    resolution:       int  = DEFAULT_RESOLUTION
    device:           str  = "cuda"
    fvd_models:       list[str] = field(default_factory=lambda: ["videomae"])
    fvd_seq_len:      int  = 16
    fvd_resolution:   int  = 224
    lmd_normalize:    bool = True
    lpips_chunk_size: int  = 64


# ---------------------------------------------------------------------------
# Per-sample loops
# ---------------------------------------------------------------------------


def _eval_same_identity(
    meta:         RunMetadata,
    cfg:          EvalConfig,
    metrics_path: Path,
) -> dict[str, list[float]]:
    """PSNR / SSIM / LPIPS / LMD per sample. One JSONL line written per sample.
    Returns a dict of per-metric lists for aggregation."""
    from .lpips_metric import LPIPSMetric
    from .lmd          import LMD

    lpips_m = LPIPSMetric(net="alex", device=cfg.device, chunk_size=cfg.lpips_chunk_size)
    lmd_m   = LMD(normalize_by_iod=cfg.lmd_normalize)

    results: dict[str, list[float]] = {
        "psnr":  [], "ssim":  [], "lpips":  [],
        "lmd_f": [], "lmd_m": [], "lmd_detect_rate": [],
    }

    with metrics_path.open("w") as f:
        for sample in tqdm(list(iter_samples(meta)), desc="same-id metrics"):
            pred = load_video(sample.pred_path,     cfg.fps, cfg.resolution)
            ref  = load_video(sample.gt_video_path, cfg.fps, cfg.resolution,
                              max_frames=pred.shape[0])
            pred, ref = truncate_to_match(pred, ref)
            pred_b = pred.unsqueeze(0).to(cfg.device)
            ref_b  = ref.unsqueeze(0).to(cfg.device)

            row = {
                "sample_id": sample.sample_id,
                "n_frames":  int(pred.shape[0]),
                "psnr":  float(psnr_video(pred_b, ref_b).item()),
                "ssim":  float(ssim_video(pred_b, ref_b).item()),
                "lpips": float(lpips_m(pred_b, ref_b).item()),
            }
            lmd = lmd_m(pred.unsqueeze(0), ref.unsqueeze(0))
            row["lmd_f"]            = float(lmd["lmd_f"].item())
            row["lmd_m"]            = float(lmd["lmd_m"].item())
            row["lmd_detect_rate"]  = float(lmd["detect_rate"].item())

            for k in results:
                v = row.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    results[k].append(v)
            f.write(json.dumps(row) + "\n")

    lmd_m.close()
    return results


def _eval_cross_identity(
    meta:         RunMetadata,
    cfg:          EvalConfig,
    metrics_path: Path,
) -> dict[str, list[float]]:
    """ID similarity (ArcFace cosine) per sample. The identity prior comes
    from the *ref* clip's frames — averaged ArcFace embedding, L2-normalized.
    Per-clip identity priors are cached so a clip that appears as ref for
    multiple cross-id pairs is only embedded once."""
    from .id_sim import IDSimilarity

    id_metric = IDSimilarity(device=cfg.device)
    prior_cache: dict[str, np.ndarray] = {}

    results: dict[str, list[float]] = {"id_cosine": [], "id_detect_rate": []}

    with metrics_path.open("w") as f:
        for sample in tqdm(list(iter_samples(meta)), desc="cross-id metrics"):
            ref_id = sample.ref_clip["uid"]
            if ref_id not in prior_cache:
                ref_video = load_video(
                    Path(sample.ref_clip["video_path"]), cfg.fps, cfg.resolution,
                )
                prior = id_metric.build_identity_prior(ref_video)
                if prior is None:
                    prior_cache[ref_id] = None  # type: ignore[assignment]
                    f.write(json.dumps({
                        "sample_id":    sample.sample_id,
                        "id_cosine":    None,
                        "id_detect_rate": 0.0,
                        "note":         "ArcFace failed on every ref frame",
                    }) + "\n")
                    continue
                prior_cache[ref_id] = prior
            prior = prior_cache[ref_id]
            if prior is None:
                continue   # ref-side detection had failed earlier; row already logged

            gen = load_video(sample.pred_path, cfg.fps, cfg.resolution)
            cos, det = id_metric.score(gen, prior)

            row = {
                "sample_id":      sample.sample_id,
                "ref_uid":        sample.ref_clip["uid"],
                "driver_uid":     sample.driver_clip["uid"],
                "n_frames":       int(gen.shape[0]),
                "id_cosine":      None if np.isnan(cos) else float(cos),
                "id_detect_rate": float(det),
            }
            if row["id_cosine"] is not None:
                results["id_cosine"].append(row["id_cosine"])
            results["id_detect_rate"].append(float(det))
            f.write(json.dumps(row) + "\n")

    return results


# ---------------------------------------------------------------------------
# FVD (distribution metric — runs once per protocol)
# ---------------------------------------------------------------------------


def _stage_fvd_dirs(meta: RunMetadata, fvd_root: Path) -> tuple[Path, Path, int]:
    """Build `pred_dir/` (one symlink per sample's panel.mp4) and `ref_dir/`
    (one symlink per ref clip's source video). Returns the dirs + N.

    Symlinks rather than copies — the source files can be hundreds of MB.
    Names are `<sample_id>.mp4` so cdfvd's video-folder loader can pair
    them positionally."""
    pred_dir = fvd_root / "pred"
    ref_dir  = fvd_root / "ref"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir (parents=True, exist_ok=True)

    n = 0
    for sample in iter_samples(meta):
        pred_link = pred_dir / f"{sample.sample_id}.mp4"
        ref_link  = ref_dir  / f"{sample.sample_id}.mp4"
        for link, src in [(pred_link, sample.pred_path),
                          (ref_link,  Path(sample.ref_clip["video_path"]).resolve())]:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
        n += 1
    return pred_dir, ref_dir, n


def _compute_fvd(meta: RunMetadata, cfg: EvalConfig) -> dict[str, float | bool]:
    """Run every backbone in `cfg.fvd_models` once. Stages a temp symlink
    tree under `<run_dir>/_fvd/`."""
    from .fvd import FVD

    fvd_root = meta.run_dir / "_fvd"
    pred_dir, ref_dir, n = _stage_fvd_dirs(meta, fvd_root)
    out: dict[str, float | bool] = {
        "n_clips":      n,
        "low_sample":   n < FVD_LOW_SAMPLE_THRESHOLD,
    }
    for backbone in cfg.fvd_models:
        fvd = FVD(
            model           = backbone,
            resolution      = cfg.fvd_resolution,
            sequence_length = cfg.fvd_seq_len,
            device          = cfg.device,
        )
        out[f"fvd_{backbone}"] = fvd.compute(pred_dir, ref_dir)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(
    run_dir: Path,
    cfg:     Optional[EvalConfig] = None,
    skip_fvd: bool = False,
) -> dict:
    """Evaluate one run dir end-to-end. Writes:
      * `<run_dir>/metrics.jsonl`         — one row per sample
      * `<run_dir>/metrics_summary.json`  — aggregates + FVD

    Returns the summary dict (also written to disk).
    """
    cfg  = cfg or EvalConfig()
    meta = load_run_metadata(run_dir)

    metrics_path = meta.run_dir / "metrics.jsonl"
    summary_path = meta.run_dir / "metrics_summary.json"

    if meta.protocol == "same_identity_reconstruction":
        per_sample = _eval_same_identity(meta, cfg, metrics_path)
    elif meta.protocol == "cross_identity":
        per_sample = _eval_cross_identity(meta, cfg, metrics_path)
    else:
        raise ValueError(f"Unknown protocol: {meta.protocol}")

    # Per-metric mean / std / n.
    aggregates: dict[str, dict[str, float]] = {}
    for name, vals in per_sample.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        aggregates[name] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
            "n":    int(arr.size),
        }

    summary: dict = {
        "run_dir":   str(meta.run_dir),
        "dataset":   meta.dataset,
        "protocol":  meta.protocol,
        "n_samples": len(list(iter_samples(meta))),
        "metrics":   aggregates,
    }

    if not skip_fvd:
        summary["fvd"] = _compute_fvd(meta, cfg)

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
