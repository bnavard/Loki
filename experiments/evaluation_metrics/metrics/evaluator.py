"""Unified protocol-aware evaluator with metric-group abstraction.

Walks one `<run_dir>` produced by the SOTA-comparison or marionette-eval
runners and routes each prediction through the metric set appropriate
to its protocol:

| Protocol                          | Metric groups                       |
|-----------------------------------|-------------------------------------|
| `same_identity_reconstruction`    | pixel, lmd, head_orientation, fvd   |
| `cross_identity`                  | head_orientation, id                |

Per-sample metrics land at `<output_dir>/metrics.jsonl` (one JSON row per
sample); aggregates + FVD at `<output_dir>/metrics_summary.json`.

Metric modes (`cfg.metrics_mode`)
---------------------------------
* `"auto"` (default) — load the existing summary and compute only the
  groups whose headline metric isn't in it. Existing per-sample fields in
  `metrics.jsonl` are preserved by merging — only newly-computed group
  fields are overwritten.
* `"all"` — recompute every group available for the protocol, overwriting
  every existing field.
* explicit `set[str]`, e.g. `{"head_orientation", "fvd"}` — recompute only
  those groups, overwrite their fields, leave others alone.

Group → headline metric mapping (`GROUP_HEADLINE_METRIC`) drives the
`auto`-mode missing-detection check. FVD is special-cased: detected by
the presence of `summary["fvd"]`, not a key under `summary["metrics"]`.

Face-region cropping (used by pixel / lmd / head_orientation)
-------------------------------------------------------------
Each video is independently cropped to a face-only square via
`face_crop_around_detection` (1.3× margin → 512×512). Both pred and
target (= driver clip) go through the same routine, so the metric
isolates pure face-region quality with no framing asymmetry. Disable
via `cfg.face_crop=False`.

Per-frame handling
------------------
Pixel metrics (PSNR/SSIM/LPIPS) run on **every** paired frame — no
detection gate; broken pred frames produce bad numbers and naturally
penalize the tool.

Landmark / orientation / identity metrics need per-frame face / pose /
arcface detection: a frame contributes only when both sides detect.
Per-sample number is the mean over hits; per-sample `*_detect_rate` is
recorded so failure density is visible.

Run-level aggregation
---------------------
Pixel metrics use a plain arithmetic mean across samples. Detection-
gated metrics (lmd_f, lmd_m, head_orientation_*_l1, id_cosine) use a
**weighted** mean with each sample's detect rate as the weight, so
a sample whose number was computed on 5/16 frames contributes
proportionally less than one computed on 16/16.

FVD on small samples
--------------------
The talking-head benchmarks here are 125 (TalkVid) / 212 (HDTF) clips.
I3D / VideoMAE-2 FVD nominally wants ≥ 2k clips for stability, so the
summary tags `low_sample = True` whenever the count is below that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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


# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------


GROUPS_BY_PROTOCOL: dict[str, list[str]] = {
    "same_identity_reconstruction": ["pixel", "lmd", "head_orientation", "fvd"],
    "cross_identity":               ["head_orientation", "id"],
}

# Detect "missing in summary" via these headline keys. fvd is special-
# cased — it lives under summary["fvd"], not summary["metrics"].
GROUP_HEADLINE_METRIC: dict[str, str] = {
    "pixel":            "psnr",
    "lmd":              "lmd_f",
    "head_orientation": "head_orientation_axis_mean_l1",
    "id":               "id_cosine",
}

# Run-level aggregation: which metrics use a weighted mean and which
# per-sample list supplies the weights.
WEIGHTED_METRICS: dict[str, str] = {
    "lmd_f":                          "_lmd_weights",
    "lmd_m":                          "_lmd_weights",
    "id_cosine":                      "_id_weights",
    "head_orientation_yaw_l1":        "_pose_weights",
    "head_orientation_pitch_l1":      "_pose_weights",
    "head_orientation_roll_l1":       "_pose_weights",
    "head_orientation_axis_mean_l1":  "_pose_weights",
}


def _group_present(group: str, summary: dict) -> bool:
    """Whether `summary` already carries the headline value for `group`."""
    if not summary:
        return False
    if group == "fvd":
        return "fvd" in summary
    return GROUP_HEADLINE_METRIC[group] in summary.get("metrics", {})


def _resolve_groups(
    mode: Union[str, set[str]],
    summary: dict,
    protocol: str,
) -> set[str]:
    """Translate the user's `metrics_mode` into a concrete set of groups
    to compute, filtered by what's available for the protocol."""
    available = set(GROUPS_BY_PROTOCOL[protocol])
    if mode == "all":
        return available
    if isinstance(mode, set):
        unknown = mode - available
        if unknown:
            print(f"[evaluate] groups {sorted(unknown)} not applicable to "
                  f"protocol={protocol}; ignored.")
        return mode & available
    if mode == "auto":
        return {g for g in available if not _group_present(g, summary)}
    raise ValueError(f"unknown metrics_mode: {mode!r}")


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
    face_crop:        bool  = True
    face_crop_margin: float = DEFAULT_FACE_CROP_MARGIN
    n_frames:         Optional[int] = 16

    # Metric mode: "auto" (default), "all", or a set[str] of group names.
    metrics_mode:     Union[str, set[str]] = "auto"


# ---------------------------------------------------------------------------
# Shared face detector (built once per run)
# ---------------------------------------------------------------------------


def _build_face_detector(device: str):
    """Multi-scale RetinaFace detector callable: `detect(img_bgr) -> list`.
    InsightFace's anchor tuning means a face filling 90% of the frame
    slips past det_size=640; the 320 fallback catches that case. Two
    ONNX sessions, ~80 MB extra."""
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
# Per-sample helpers
# ---------------------------------------------------------------------------


def _frame_to_uint8(t_chw: torch.Tensor) -> np.ndarray:
    """`(3, H, W)` float32 in `[0, 1]` → `(H, W, 3)` uint8 RGB."""
    return (t_chw.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


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


def _compute_head_orientation_pair(pred, target, hoe, T):
    """Per-frame |yaw|, |pitch|, |roll| (degrees) over the `T` paired
    frames where 6DRepNet produced a pose for both sides. Returns
    `(yaw_l1, pitch_l1, roll_l1, axis_mean_l1, detect_rate)` with all
    four l1 values being `None` if no frame produced a hit."""
    yaw_per, pitch_per, roll_per = [], [], []
    for t in range(T):
        p_u8 = _frame_to_uint8(pred[t])
        g_u8 = _frame_to_uint8(target[t])
        p_pose = hoe.extract(p_u8)
        g_pose = hoe.extract(g_u8)
        if p_pose is None or g_pose is None:
            continue
        yaw_per.  append(abs(p_pose[0] - g_pose[0]))
        pitch_per.append(abs(p_pose[1] - g_pose[1]))
        roll_per. append(abs(p_pose[2] - g_pose[2]))
    n_hits = len(yaw_per)
    detect_rate = (n_hits / T) if T > 0 else 0.0
    if n_hits == 0:
        return None, None, None, None, detect_rate
    yaw_l1   = float(np.mean(yaw_per))
    pitch_l1 = float(np.mean(pitch_per))
    roll_l1  = float(np.mean(roll_per))
    axis_l1  = (yaw_l1 + pitch_l1 + roll_l1) / 3.0
    return yaw_l1, pitch_l1, roll_l1, axis_l1, detect_rate


def _resize_to_square(video, resolution):
    return torch.nn.functional.interpolate(
        video, size=(resolution, resolution),
        mode="bilinear", align_corners=False,
    )


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
    """Process every sample, computing only the requested `groups`. For
    fields outside `groups`, values from `existing_rows[sample_id]` are
    preserved when we rewrite `metrics.jsonl`. Returns per-metric value
    lists for newly-computed groups (used for run-level aggregation).
    """
    needs_face_crop = bool({"pixel", "lmd", "head_orientation"} & groups)

    lpips_m  = lmd_obj = hoe = None
    detect_fn = None
    if "pixel" in groups:
        from .lpips_metric import LPIPSMetric
        lpips_m = LPIPSMetric(net="alex", device=cfg.device,
                              chunk_size=cfg.lpips_chunk_size)
    if "lmd" in groups:
        from .lmd import LMD, landmark_distance_pair  # noqa: F401  (used below)
        lmd_obj = LMD(normalize_by_iod=cfg.lmd_normalize)
    if "head_orientation" in groups:
        from .head_orientation import HeadOrientationEstimator
        hoe = HeadOrientationEstimator(device=cfg.device)
    if needs_face_crop and cfg.face_crop:
        detect_fn = _build_face_detector(cfg.device)

    results: dict[str, list] = {}
    if "pixel" in groups:
        results.update({"psnr": [], "ssim": [], "lpips": []})
    if "lmd" in groups:
        results.update({"lmd_f": [], "lmd_m": [], "lmd_detect_rate": [],
                        "_lmd_weights": []})
    if "head_orientation" in groups:
        results.update({
            "head_orientation_yaw_l1":       [],
            "head_orientation_pitch_l1":     [],
            "head_orientation_roll_l1":      [],
            "head_orientation_axis_mean_l1": [],
            "head_orientation_detect_rate":  [],
            "_pose_weights":                 [],
        })

    samples = list(iter_samples(meta))
    out_rows: list[dict] = []
    skipped_face_crop = 0

    desc = f"same-id ({','.join(sorted(groups))})"
    for sample in tqdm(samples, desc=desc):
        existing = existing_rows.get(sample.sample_id, {})
        new_fields: dict = {"sample_id": sample.sample_id}

        if needs_face_crop:
            pred = load_video(sample.pred_path,     cfg.fps, resolution=None,
                              max_frames=cfg.n_frames)
            ref  = load_video(sample.gt_video_path, cfg.fps, resolution=None,
                              max_frames=pred.shape[0])

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
                    # Merge with existing — drop computed groups' old values
                    # since we just failed to recompute, but keep groups we
                    # never touched (e.g. fvd doesn't apply here).
                    merged = dict(existing)
                    merged["sample_id"] = sample.sample_id
                    merged["skipped"]   = "face_detection_failed_clip"
                    out_rows.append(merged)
                    continue
                pred, ref = pred_cropped, ref_cropped
            else:
                pred = _resize_to_square(pred, cfg.resolution)
                ref  = _resize_to_square(ref,  cfg.resolution)
            pred, ref = truncate_to_match(pred, ref)
            T = pred.shape[0]
            new_fields["n_frames"] = T

        if "pixel" in groups:
            pred_d = pred.unsqueeze(0).to(cfg.device)
            ref_d  = ref .unsqueeze(0).to(cfg.device)
            psnr  = float(psnr_video(pred_d, ref_d).item())
            ssim  = float(ssim_video(pred_d, ref_d).item())
            lpips = float(lpips_m   (pred_d, ref_d).item())
            new_fields["psnr"]  = psnr
            new_fields["ssim"]  = ssim
            new_fields["lpips"] = lpips
            results["psnr"] .append(psnr)
            results["ssim"] .append(ssim)
            results["lpips"].append(lpips)

        if "lmd" in groups:
            from .lmd import landmark_distance_pair
            lmd_f_per, lmd_m_per = [], []
            for t in range(T):
                pl = lmd_obj.extract(_frame_to_uint8(pred[t]))
                gl = lmd_obj.extract(_frame_to_uint8(ref [t]))
                if pl is None or gl is None:
                    continue
                lf, lm = landmark_distance_pair(
                    pl, gl, normalize_by_iod=cfg.lmd_normalize,
                )
                lmd_f_per.append(lf)
                lmd_m_per.append(lm)
            n_hits = len(lmd_f_per)
            ldr    = (n_hits / T) if T > 0 else 0.0
            if n_hits > 0:
                lmd_f = float(np.mean(lmd_f_per))
                lmd_m = float(np.mean(lmd_m_per))
                results["lmd_f"]       .append(lmd_f)
                results["lmd_m"]       .append(lmd_m)
                results["_lmd_weights"].append(ldr)
            else:
                lmd_f = lmd_m = None
            results["lmd_detect_rate"].append(ldr)
            new_fields["lmd_f"]            = lmd_f
            new_fields["lmd_m"]            = lmd_m
            new_fields["lmd_detect_rate"]  = ldr

        if "head_orientation" in groups:
            yaw_l1, pitch_l1, roll_l1, axis_l1, pdr = _compute_head_orientation_pair(
                pred, ref, hoe, T,
            )
            if yaw_l1 is not None:
                results["head_orientation_yaw_l1"]      .append(yaw_l1)
                results["head_orientation_pitch_l1"]    .append(pitch_l1)
                results["head_orientation_roll_l1"]     .append(roll_l1)
                results["head_orientation_axis_mean_l1"].append(axis_l1)
                results["_pose_weights"]                .append(pdr)
            results["head_orientation_detect_rate"].append(pdr)
            new_fields["head_orientation_yaw_l1"]        = yaw_l1
            new_fields["head_orientation_pitch_l1"]      = pitch_l1
            new_fields["head_orientation_roll_l1"]       = roll_l1
            new_fields["head_orientation_axis_mean_l1"]  = axis_l1
            new_fields["head_orientation_detect_rate"]   = pdr

        # Merge: existing ∪ new (new overrides). If we successfully computed
        # any group, drop a stale `skipped` flag.
        merged = {**existing, **new_fields}
        merged.pop("skipped", None)
        out_rows.append(merged)

    if lmd_obj is not None:
        lmd_obj.close()

    metrics_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")

    if skipped_face_crop:
        print(f"[metrics] {skipped_face_crop} samples skipped: clip-level face detection failed")
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
    """For cross-id: id_cosine uses the ref clip's video (identity prior),
    head_orientation uses the driver clip's video (motion source).
    Different videos, different face crops; gated independently per group."""
    id_metric = hoe = None
    detect_fn = None
    if "id" in groups:
        from .id_sim import IDSimilarity
        id_metric = IDSimilarity(device=cfg.device)
    if "head_orientation" in groups:
        from .head_orientation import HeadOrientationEstimator
        hoe = HeadOrientationEstimator(device=cfg.device)
        if cfg.face_crop:
            detect_fn = _build_face_detector(cfg.device)

    results: dict[str, list] = {}
    if "id" in groups:
        results.update({"id_cosine": [], "id_detect_rate": [], "_id_weights": []})
    if "head_orientation" in groups:
        results.update({
            "head_orientation_yaw_l1":       [],
            "head_orientation_pitch_l1":     [],
            "head_orientation_roll_l1":      [],
            "head_orientation_axis_mean_l1": [],
            "head_orientation_detect_rate":  [],
            "_pose_weights":                 [],
        })

    prior_cache: dict = {}
    out_rows: list[dict] = []
    skipped_face_crop = 0
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
                # for `id` only; head_orientation can still try below.
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

        # ---- Head orientation: pred vs driver clip's video. ----
        if "head_orientation" in groups:
            pred = load_video(sample.pred_path, cfg.fps, resolution=None,
                              max_frames=cfg.n_frames)
            target = load_video(
                Path(sample.driver_clip["video_path"]), cfg.fps,
                resolution=None, max_frames=pred.shape[0],
            )
            if cfg.face_crop:
                pred_cropped = face_crop_around_detection(
                    pred, detect_fn, margin=cfg.face_crop_margin,
                    target_resolution=cfg.resolution,
                )
                target_cropped = face_crop_around_detection(
                    target, detect_fn, margin=cfg.face_crop_margin,
                    target_resolution=cfg.resolution,
                )
                if pred_cropped is None or target_cropped is None:
                    skipped_face_crop += 1
                    pdr = 0.0
                    yaw_l1 = pitch_l1 = roll_l1 = axis_l1 = None
                else:
                    pred, target = pred_cropped, target_cropped
                    pred, target = truncate_to_match(pred, target)
                    T = pred.shape[0]
                    yaw_l1, pitch_l1, roll_l1, axis_l1, pdr = (
                        _compute_head_orientation_pair(pred, target, hoe, T)
                    )
            else:
                pred   = _resize_to_square(pred,   cfg.resolution)
                target = _resize_to_square(target, cfg.resolution)
                pred, target = truncate_to_match(pred, target)
                T = pred.shape[0]
                yaw_l1, pitch_l1, roll_l1, axis_l1, pdr = (
                    _compute_head_orientation_pair(pred, target, hoe, T)
                )

            if yaw_l1 is not None:
                results["head_orientation_yaw_l1"]      .append(yaw_l1)
                results["head_orientation_pitch_l1"]    .append(pitch_l1)
                results["head_orientation_roll_l1"]     .append(roll_l1)
                results["head_orientation_axis_mean_l1"].append(axis_l1)
                results["_pose_weights"]                .append(pdr)
            results["head_orientation_detect_rate"].append(pdr)
            new_fields["head_orientation_yaw_l1"]       = yaw_l1
            new_fields["head_orientation_pitch_l1"]     = pitch_l1
            new_fields["head_orientation_roll_l1"]      = roll_l1
            new_fields["head_orientation_axis_mean_l1"] = axis_l1
            new_fields["head_orientation_detect_rate"]  = pdr

        merged = {**existing, **new_fields}
        # Stale skip flag should drop only if we successfully computed
        # something *new* for this sample. For cross-id, both id and
        # head_orientation can independently fail; we only clear the
        # flag if at least one group succeeded.
        if (
            ("id" in groups and merged.get("id_cosine") is not None)
            or ("head_orientation" in groups
                and merged.get("head_orientation_yaw_l1") is not None)
        ):
            merged.pop("skipped", None)
        out_rows.append(merged)

    metrics_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")

    if skipped_face_crop:
        print(f"[metrics] {skipped_face_crop} samples: head-orientation face crop failed")
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

    Every staged mp4 is truncated to `cfg.fvd_seq_len` (= 16) frames on
    both pred and GT sides. This collapses cdfvd's random-clip-sampling
    asymmetry — its `VideoDataset` would otherwise pick a random
    16-frame chunk per video, and SOTA's 75–125-frame panels yield
    mid-clip windows while Marionette's 16-frame panel is forced to
    frames 0–15. Capping both sides to 16 frames means cdfvd has
    exactly one possible chunk, and every baseline scores on the same
    first-16-frame window.

    With `cfg.face_crop` on, both sides are face-cropped via the same
    routine used for per-sample metrics and re-encoded. With `cfg.face_crop`
    off, raw mp4s are symlinked (cdfvd's `sequence_length=16` still
    truncates at decode time).

    Samples whose face detection fails on either side are skipped from
    FVD staging entirely.
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
            for p in (pred_dst, ref_dst):
                if p.is_symlink() or p.exists():
                    p.unlink()
            pred_video = load_video(sample.pred_path, cfg.fps, resolution=None,
                                    max_frames=cfg.fvd_seq_len)
            ref_video  = load_video(Path(sample.ref_clip["video_path"]),
                                    cfg.fps, resolution=None,
                                    max_frames=cfg.fvd_seq_len)
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
            for dst, src in [(pred_dst, sample.pred_path),
                             (ref_dst,  Path(sample.ref_clip["video_path"]))]:
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())

        n_staged += 1

    return pred_dir, ref_dir, n_staged, n_skipped


def _compute_fvd(meta: RunMetadata, cfg: EvalConfig,
                 staging_root: Path) -> dict[str, float | bool | int]:
    """Run every backbone in `cfg.fvd_models` once. Stages a temp tree
    under `<staging_root>/_fvd/` (face-cropped 16-frame mp4s for both
    pred and GT when `cfg.face_crop` is on), then removes it after the
    backbone forward passes finish."""
    import shutil
    from .fvd import FVD

    detect_fn = _build_face_detector(cfg.device) if cfg.face_crop else None
    fvd_root  = staging_root / "_fvd"
    try:
        pred_dir, ref_dir, n, n_skipped = _stage_fvd_dirs(
            meta, fvd_root, cfg, detect_fn=detect_fn,
        )
        out: dict[str, float | bool | int] = {
            "n_clips":    n,
            "n_skipped":  n_skipped,
            "low_sample": n < FVD_LOW_SAMPLE_THRESHOLD,
            "face_crop":  cfg.face_crop,
        }
        for backbone in cfg.fvd_models:
            fvd = FVD(
                model           = backbone,
                resolution      = cfg.fvd_resolution,
                sequence_length = cfg.fvd_seq_len,
                device          = cfg.device,
            )
            out[f"fvd_{backbone}"] = fvd.compute(pred_dir, ref_dir)
    finally:
        shutil.rmtree(fvd_root, ignore_errors=True)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(per_sample: dict[str, list]) -> dict[str, dict[str, float]]:
    """Per-metric aggregation. Detection-gated metrics
    (lmd_*, id_cosine, head_orientation_*_l1) use a weighted mean with
    each sample's detect rate as the weight; everything else is a plain
    arithmetic mean."""
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
      * `<output_dir>/metrics_summary.json`  — aggregates + FVD

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
    groups = _resolve_groups(cfg.metrics_mode, existing_summary, meta.protocol)

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
        "run_dir":      str(meta.run_dir),
        "dataset":      meta.dataset,
        "protocol":     meta.protocol,
        "n_samples":    len(list(iter_samples(meta))),
        "face_crop":    cfg.face_crop,
        "metrics":      aggregates,
    }

    if "fvd" in groups:
        summary["fvd"] = _compute_fvd(meta, cfg, staging_root=out_dir)
    elif "fvd" in existing_summary and cfg.metrics_mode != "all":
        summary["fvd"] = existing_summary["fvd"]

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
