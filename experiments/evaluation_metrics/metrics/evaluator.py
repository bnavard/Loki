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

Face-region cropping (same-identity)
------------------------------------
PSNR / SSIM / LPIPS / LMD are computed on a tight face-only square,
not on the raw pred / GT framings. For each video independently:
detect a face bbox, expand by `cfg.face_crop_margin` (default 1.3×),
square it on the face center, and resize to `cfg.resolution`. The same
routine is applied to both pred and GT, so the metric isolates pure
face-region quality — no framing asymmetry, no scale drift from
tool-specific crop conventions, no background bleed-in. This matches
what talking-head papers report.

Disable via `face_crop=False` on `EvalConfig` (or `--no-face-crop` on
the CLI) for raw-framing numbers.

Per-frame handling
------------------
PSNR / SSIM / LPIPS are pure pixel/feature operations and are computed on
**every** paired frame — broken pred frames produce bad numbers and
naturally penalize the tool, which is what we want for fair comparison.

LMD-F / LMD-M and ID similarity *do* require face detection: a frame
contributes only when MediaPipe (LMD) or ArcFace (ID similarity) detects
a face. The per-sample number is the mean over those hits, and a
per-sample `lmd_detect_rate` / `id_detect_rate` is recorded so failure
density is visible.

Run-level aggregation
---------------------
Pixel metrics use a plain arithmetic mean across samples. Landmark-style
metrics (LMD-F, LMD-M, id_cosine) use a **weighted** mean with each
sample's detect rate as the weight:

    weighted_mean(LMD) = Σᵢ rateᵢ · lmdᵢ / Σᵢ rateᵢ

Samples where many frames failed detection contribute proportionally less
to the run-level score, so a tool that breaks half its frames doesn't
get credit for the easy ones it landed cleanly.

Sample-level skip
-----------------
Logged to `metrics.jsonl` with a `skipped` field:
  * `face_detection_failed_clip` — InsightFace couldn't detect a face on
    the first 10 probe frames of pred or GT, so a per-pair crop can't be
    derived. The sample is excluded from every metric and from FVD staging.
  * (cross-identity only) `ref_arcface_failed` — ArcFace failed on every
    frame of the reference clip; no identity prior to compare against.

Per-frame failures within a sample do **not** trigger a skip — pixel
metrics still compute, landmark metrics record `null` and a 0 detect rate.

FVD on small samples
--------------------
The talking-head benchmarks here are 125 (TalkVid) / 212 (HDTF) clips.
I3D FVD is widely held to need ≥ 2k clips for stability, so the summary
tags `low_sample = True` whenever the count is below the threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .io import (
    DEFAULT_FACE_CROP_MARGIN, DEFAULT_FPS, DEFAULT_RESOLUTION,
    RunMetadata, SamplePair,
    detect_face_bbox_xyxy, face_crop_around_detection, face_crop_video,
    load_run_metadata, iter_samples, load_video, truncate_to_match,
)
from .psnr import psnr_video
from .ssim import ssim_video


FVD_LOW_SAMPLE_THRESHOLD = 2000

# Run-level aggregation: which metrics use a weighted mean and which
# per-sample list supplies the weights. Weighted-mean keeps detect rate
# from being silently lost when we average a sample with `lmd_f=0.025`
# computed on 30% of its frames against another at the same `lmd_f` on
# 100% of frames.
WEIGHTED_METRICS: dict[str, str] = {
    "lmd_f":     "_lmd_weights",
    "lmd_m":     "_lmd_weights",
    "id_cosine": "_id_weights",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    fps:              int   = DEFAULT_FPS
    resolution:       int   = DEFAULT_RESOLUTION
    device:           str   = "cuda"
    fvd_models:       list[str] = field(default_factory=lambda: ["videomae"])
    fvd_seq_len:      int   = 16
    fvd_resolution:   int   = 224
    lmd_normalize:    bool  = True
    lpips_chunk_size: int   = 64
    face_crop:        bool  = True   # crop pred AND GT to face-only squares
    face_crop_margin: float = DEFAULT_FACE_CROP_MARGIN
    # Cap pred (and therefore GT) to N frames so SOTA's 75–125 frame
    # outputs are scored on the same temporal coverage as Marionette's
    # 16-frame panel. Default 16 = `cfg.inference.n_frames` in
    # `marionette/configs/base.yaml`. Set to None for tool-native length.
    n_frames:         Optional[int] = 16


# ---------------------------------------------------------------------------
# Shared face detector (built once per run)
# ---------------------------------------------------------------------------


def _build_face_detector(device: str):
    """Multi-scale RetinaFace detector callable: `detect(img_bgr) -> list`.

    InsightFace's RetinaFace is configured at a fixed `det_size` per
    instance, and its anchor tuning means a face that fills 90% of the
    frame (e.g. SadTalker's tight 512×512 pred crop) is *too big* for the
    largest anchors at `det_size=(640, 640)` and slips through. Native
    GT clips at 1024×1024 with the face at ~30% of frame, on the other
    hand, work fine at 640.

    We instantiate two detectors (640 and 320) and fall back from the
    larger to the smaller on no-hit. Two ONNX sessions, ~80 MB extra —
    much cheaper than re-`prepare()`-ing on every call.

    Returns a `detect(img_bgr) -> list` callable (matches `app.get()`).
    """
    from insightface.app import FaceAnalysis
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda") else ["CPUExecutionProvider"]
    )
    ctx_id = 0 if device.startswith("cuda") else -1
    apps = []
    for det_size in ((640, 640), (320, 320)):
        app = FaceAnalysis(name="buffalo_l", providers=providers,
                           allowed_modules=["detection"])
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        apps.append(app)

    def detect(img_bgr):
        for app in apps:
            faces = app.get(img_bgr)
            if faces:
                return faces
        return []
    return detect


# ---------------------------------------------------------------------------
# Same-identity per-sample loop
# ---------------------------------------------------------------------------


def _eval_same_identity(
    meta:         RunMetadata,
    cfg:          EvalConfig,
    metrics_path: Path,
) -> dict[str, list[float]]:
    """Per-sample PSNR / SSIM / LPIPS (no gating) and LMD-F / LMD-M
    (per-frame MediaPipe detection gate). Writes one JSONL line per sample
    to `metrics_path`. Returns per-metric lists; the special key
    `_lmd_weights` carries the per-sample lmd_detect_rate to be used as
    the weight for `lmd_f` and `lmd_m` in the run-level weighted mean."""
    from .lmd          import LMD, landmark_distance_pair
    from .lpips_metric import LPIPSMetric

    lpips_m   = LPIPSMetric(net="alex", device=cfg.device, chunk_size=cfg.lpips_chunk_size)
    lmd_obj   = LMD(normalize_by_iod=cfg.lmd_normalize)
    detect_fn = _build_face_detector(cfg.device) if cfg.face_crop else None

    # `_lmd_weights` is the per-sample lmd_detect_rate paired with each
    # entry of lmd_f / lmd_m, used as the weighted-mean weight in
    # `evaluate()`. It only carries entries for samples that produced a
    # non-null lmd_f (i.e. at least one detected frame).
    results: dict[str, list[float]] = {
        "psnr":  [], "ssim":  [], "lpips":  [],
        "lmd_f": [], "lmd_m": [],
        "lmd_detect_rate": [],
        "_lmd_weights":    [],
    }

    samples = list(iter_samples(meta))
    skipped_face_crop = 0

    with metrics_path.open("w") as f:
        for sample in tqdm(samples, desc="same-id metrics"):
            # Load both at native resolution; the face-crop step does the
            # final 512×512 resize. Skipping the early resize keeps the
            # detected face bbox at full source-pixel precision.
            pred = load_video(sample.pred_path,     cfg.fps, resolution=None,
                              max_frames=cfg.n_frames)
            ref  = load_video(sample.gt_video_path, cfg.fps, resolution=None,
                              max_frames=pred.shape[0])

            # 1. Clip-level face crop applied to BOTH pred and GT
            #    independently — same routine, same margin. After this both
            #    videos are at `cfg.resolution` with the face filling
            #    roughly the same fraction.
            if cfg.face_crop:
                pred_cropped = face_crop_around_detection(
                    pred, detect_fn, margin=cfg.face_crop_margin,
                    target_resolution=cfg.resolution,
                )
                ref_cropped = face_crop_around_detection(
                    ref, detect_fn, margin=cfg.face_crop_margin,
                    target_resolution=cfg.resolution,
                )
                if pred_cropped is None or ref_cropped is None:
                    skipped_face_crop += 1
                    f.write(json.dumps({
                        "sample_id": sample.sample_id,
                        "skipped":   "face_detection_failed_clip",
                    }) + "\n")
                    continue
                pred, ref = pred_cropped, ref_cropped
            else:
                pred = torch.nn.functional.interpolate(
                    pred, size=(cfg.resolution, cfg.resolution),
                    mode="bilinear", align_corners=False,
                )
                ref = torch.nn.functional.interpolate(
                    ref, size=(cfg.resolution, cfg.resolution),
                    mode="bilinear", align_corners=False,
                )
            pred, ref = truncate_to_match(pred, ref)
            T = pred.shape[0]

            # 2. Pixel metrics on ALL paired frames — a broken pred frame
            #    has pixels too, so PSNR / SSIM / LPIPS get to penalize it
            #    naturally. No detection gate here.
            pred_d = pred.unsqueeze(0).to(cfg.device)
            ref_d  = ref .unsqueeze(0).to(cfg.device)
            psnr  = float(psnr_video(pred_d, ref_d).item())
            ssim  = float(ssim_video(pred_d, ref_d).item())
            lpips = float(lpips_m   (pred_d, ref_d).item())

            # 3. LMD-F / LMD-M with per-frame MediaPipe gating: a frame
            #    contributes only when both pred and GT return landmarks.
            #    Per-sample number is the unweighted mean over hit frames;
            #    the per-sample `lmd_detect_rate` is reported alongside
            #    and used as the run-level weight.
            lmd_f_per, lmd_m_per = [], []
            for t in range(T):
                p_uint8 = (pred[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                g_uint8 = (ref [t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pl = lmd_obj.extract(p_uint8)
                gl = lmd_obj.extract(g_uint8)
                if pl is None or gl is None:
                    continue
                lf, lm = landmark_distance_pair(pl, gl, normalize_by_iod=cfg.lmd_normalize)
                lmd_f_per.append(lf)
                lmd_m_per.append(lm)

            n_hits          = len(lmd_f_per)
            lmd_detect_rate = n_hits / T
            if n_hits > 0:
                lmd_f = float(np.mean(lmd_f_per))
                lmd_m = float(np.mean(lmd_m_per))
            else:
                lmd_f = None
                lmd_m = None

            row = {
                "sample_id":       sample.sample_id,
                "n_frames":        T,
                "psnr":            psnr,
                "ssim":            ssim,
                "lpips":           lpips,
                "lmd_f":           lmd_f,
                "lmd_m":           lmd_m,
                "lmd_detect_rate": lmd_detect_rate,
            }

            results["psnr"]            .append(psnr)
            results["ssim"]            .append(ssim)
            results["lpips"]           .append(lpips)
            results["lmd_detect_rate"] .append(lmd_detect_rate)
            if lmd_f is not None:
                results["lmd_f"]       .append(lmd_f)
                results["lmd_m"]       .append(lmd_m)
                results["_lmd_weights"].append(lmd_detect_rate)

            f.write(json.dumps(row) + "\n")

    lmd_obj.close()
    if skipped_face_crop:
        print(f"[metrics] {skipped_face_crop} samples skipped: clip-level face detection failed")
    return results


# ---------------------------------------------------------------------------
# Cross-identity per-sample loop (unchanged — ArcFace handles its own
# alignment internally, so framing mismatch isn't a metric concern here)
# ---------------------------------------------------------------------------


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

    # `_id_weights` is the per-sample id_detect_rate paired with each
    # entry of id_cosine, used as the weighted-mean weight in `evaluate()`.
    results: dict[str, list[float]] = {
        "id_cosine":      [],
        "id_detect_rate": [],
        "_id_weights":    [],
    }

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
                        "skipped":      "ref_arcface_failed",
                    }) + "\n")
                    continue
                prior_cache[ref_id] = prior
            prior = prior_cache[ref_id]
            if prior is None:
                continue   # ref-side detection had failed earlier; row already logged

            gen = load_video(sample.pred_path, cfg.fps, cfg.resolution,
                             max_frames=cfg.n_frames)
            cos, det = id_metric.score(gen, prior)

            row = {
                "sample_id":      sample.sample_id,
                "ref_uid":        sample.ref_clip   ["uid"],
                "driver_uid":     sample.driver_clip["uid"],
                "n_frames":       int(gen.shape[0]),
                "id_cosine":      None if np.isnan(cos) else float(cos),
                "id_detect_rate": float(det),
            }
            results["id_detect_rate"].append(float(det))
            if row["id_cosine"] is not None:
                results["id_cosine"]   .append(row["id_cosine"])
                results["_id_weights"] .append(float(det))
            f.write(json.dumps(row) + "\n")

    return results


# ---------------------------------------------------------------------------
# FVD staging (face-cropped on-disk mp4s)
# ---------------------------------------------------------------------------


def _save_video_mp4(video: torch.Tensor, path: Path, fps: int) -> None:
    """Write `(T, 3, H, W)` float32 in `[0, 1]` to `path` as H.264 mp4."""
    import imageio.v2 as iio_v2
    arr = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    writer = iio_v2.get_writer(str(path), fps=fps, codec="libx264",
                               quality=8, macro_block_size=1)
    try:
        for frame in arr:
            writer.append_data(frame)
    finally:
        writer.close()


def _stage_fvd_dirs(meta: RunMetadata, fvd_root: Path, cfg: EvalConfig,
                    detect_fn=None) -> tuple[Path, Path, int, int]:
    """Stage `<fvd_root>/{pred,ref}/` for cdfvd's video-folder loader.

    With `cfg.face_crop` on (the default), **both** pred and GT are
    loaded, face-cropped via the same `face_crop_around_detection`
    routine used for per-sample metrics, and re-encoded as fresh mp4s.
    FVD then compares the same face-region distribution on both sides.

    With `cfg.face_crop` off, pred is symlinked from each sample's
    `panel.mp4` and GT is symlinked from the manifest path (variable
    resolution; framing mismatch will bias the FVD number).

    Samples whose face detection fails on either side are skipped from
    FVD staging entirely. Returns `(pred_dir, ref_dir, n_staged, n_skipped)`.
    """
    pred_dir = fvd_root / "pred"
    ref_dir  = fvd_root / "ref"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir (parents=True, exist_ok=True)

    n_staged, n_skipped = 0, 0

    for sample in tqdm(list(iter_samples(meta)), desc="fvd staging"):
        pred_dst = pred_dir / f"{sample.sample_id}.mp4"
        ref_dst  = ref_dir  / f"{sample.sample_id}.mp4"

        if cfg.face_crop and detect_fn is not None:
            # Cache hit: face-cropped mp4s already on disk from a prior run.
            if (pred_dst.exists() and not pred_dst.is_symlink()
                    and ref_dst.exists() and not ref_dst.is_symlink()):
                n_staged += 1
                continue
            for p in (pred_dst, ref_dst):
                if p.is_symlink() or p.exists():
                    p.unlink()
            pred_video = load_video(sample.pred_path, cfg.fps, resolution=None)
            ref_video  = load_video(Path(sample.ref_clip["video_path"]),
                                    cfg.fps, resolution=None,
                                    max_frames=pred_video.shape[0])
            pred_crop = face_crop_around_detection(
                pred_video, detect_fn, margin=cfg.face_crop_margin,
                target_resolution=cfg.resolution,
            )
            ref_crop = face_crop_around_detection(
                ref_video, detect_fn, margin=cfg.face_crop_margin,
                target_resolution=cfg.resolution,
            )
            if pred_crop is None or ref_crop is None:
                n_skipped += 1
                continue
            _save_video_mp4(pred_crop, pred_dst, fps=cfg.fps)
            _save_video_mp4(ref_crop,  ref_dst,  fps=cfg.fps)
        else:
            # Symlink raw mp4s (variable resolution).
            for dst, src in [(pred_dst, sample.pred_path),
                             (ref_dst,  Path(sample.ref_clip["video_path"]))]:
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())

        n_staged += 1

    return pred_dir, ref_dir, n_staged, n_skipped


def _compute_fvd(meta: RunMetadata, cfg: EvalConfig) -> dict[str, float | bool | int]:
    """Run every backbone in `cfg.fvd_models` once. Stages a temp tree
    under `<run_dir>/_fvd/` (face-cropped mp4s for both pred and GT
    when `cfg.face_crop` is on)."""
    from .fvd import FVD

    detect_fn = _build_face_detector(cfg.device) if cfg.face_crop else None
    fvd_root = meta.run_dir / "_fvd"
    pred_dir, ref_dir, n, n_skipped = _stage_fvd_dirs(meta, fvd_root, cfg, detect_fn=detect_fn)
    out: dict[str, float | bool | int] = {
        "n_clips":     n,
        "n_skipped":   n_skipped,
        "low_sample":  n < FVD_LOW_SAMPLE_THRESHOLD,
        "face_crop": cfg.face_crop,
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
    run_dir:  Path,
    cfg:      Optional[EvalConfig] = None,
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

    # Per-metric mean / std / n. Landmark-style metrics (LMD-F, LMD-M,
    # id_cosine) use a weighted mean with the per-sample detect rate as
    # the weight; everything else is a plain arithmetic mean.
    aggregates: dict[str, dict[str, float]] = {}
    for name, vals in per_sample.items():
        if name.startswith("_"):
            continue                                # internal weight tracker
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

    summary: dict = {
        "run_dir":      str(meta.run_dir),
        "dataset":      meta.dataset,
        "protocol":     meta.protocol,
        "n_samples":    len(list(iter_samples(meta))),
        "face_crop": cfg.face_crop,
        "metrics":      aggregates,
    }

    if not skip_fvd:
        summary["fvd"] = _compute_fvd(meta, cfg)

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
