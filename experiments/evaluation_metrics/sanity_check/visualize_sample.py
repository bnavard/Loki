r"""Render per-frame metric overlays for one sample as an mp4, plus a
per-video metric-curves PNG.

Each visualizable metric gets an overlay so you can verify by eye that the
numbers correspond to what the model actually produced:

  * **LMD-F / LMD-M** (same-id) — 478 MediaPipe FaceMesh landmarks rendered
    per frame on both pred and GT (red on pred, green on GT). Lip
    landmarks (the LMD-M subset) are drawn slightly larger.
  * **PSNR / SSIM / LPIPS** (same-id) — per-frame values overlaid as a
    text bar on the pred frame, plus a temporal curve plot saved as a
    sibling PNG so you can see the whole-video trend.
  * **ID similarity** (cross-id) — InsightFace's largest detected face
    bbox drawn on every pred frame, with the per-frame cosine score
    against the ref clip's averaged identity prior. The reference face
    thumbnail (one detected ref frame) is shown alongside so the cosine
    has a visual anchor.

FVD is a distribution-level metric (no per-frame visual sense) and isn't
overlaid here.

This is a *visual debugging tool* — it does not write `metrics.jsonl` or
the run-level summary. For aggregate numbers, use
`compute_metrics.py --run-dir <run_dir>`.

Usage
-----

    # Auto-pick the first sample in the run:
    PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_sample.py \
        --run-dir outputs/sota_comparison/sadtalker/talkvid/same_identity_reconstruction/run_<ts>/

    # Or name a specific sample:
    PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_sample.py \
        --run-dir outputs/sota_comparison/xportrait/talkvid/cross_identity/run_<ts>/ \
        --sample-id id_0042_id_0099

Output
------

  <sample_dir>/metrics_overlay.mp4   — pred ‖ GT (or pred ‖ ref-thumbnail)
                                       with metric overlays per frame.
  <sample_dir>/metrics_curves.png    — same-id only: per-frame PSNR / SSIM /
                                       LPIPS curves over the whole video.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio_v2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.evaluation_metrics.metrics.io import (
    DEFAULT_FACE_CROP_MARGIN, DEFAULT_FPS, DEFAULT_RESOLUTION,
    face_crop_around_detection,
    iter_samples, load_run_metadata, load_video, truncate_to_match,
)
from experiments.evaluation_metrics.metrics.lmd import (
    LMD, LEFT_EYE_OUTER, MOUTH_LANDMARKS, RIGHT_EYE_OUTER,
)
from experiments.evaluation_metrics.metrics.lpips_metric import LPIPSMetric
from experiments.evaluation_metrics.metrics.psnr import psnr_video
from experiments.evaluation_metrics.metrics.ssim import ssim_video


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize per-frame metric overlays for one sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",   type=Path, required=True)
    p.add_argument("--sample-id", type=str,  default=None,
                   help="Specific sample. Default: first sample in the run.")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--out-mp4",   type=Path, default=None,
                   help="Override output mp4 path. Default: "
                        "<sample_dir>/metrics_overlay.mp4")
    p.add_argument("--out-png",   type=Path, default=None,
                   help="Override curves PNG path. Default: "
                        "<sample_dir>/metrics_curves.png (same-id only).")
    p.add_argument("--no-face-crop", dest="face_crop", action="store_false",
                   help="Disable face-cropping (default: enabled).")
    p.set_defaults(face_crop=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _font(size: int = 14, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_landmarks(img_uint8: np.ndarray, lms: np.ndarray | None,
                    color: tuple[int, int, int]) -> np.ndarray:
    """Overlay 478 landmarks on a single frame. Mouth + eye-corner landmarks
    are drawn larger so LMD-M (mouth subset) and the IOD anchors are
    visible against the dense face point cloud."""
    img = Image.fromarray(img_uint8.copy())
    if lms is not None:
        draw = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(lms):
            r = 2 if i in MOUTH_LANDMARKS or i in (LEFT_EYE_OUTER, RIGHT_EYE_OUTER) else 1
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return np.asarray(img)


def _draw_bbox(img_uint8: np.ndarray, bbox: np.ndarray | None,
               color: tuple[int, int, int], width: int = 3) -> np.ndarray:
    """Overlay an `[x1, y1, x2, y2]` bbox."""
    img = Image.fromarray(img_uint8.copy())
    if bbox is not None:
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    return np.asarray(img)


def _text_strip(text: str, width: int, height: int = 36,
                bg=(20, 20, 20), fg=(255, 255, 255), pad=10) -> np.ndarray:
    strip = Image.new("RGB", (width, height), bg)
    ImageDraw.Draw(strip).text((pad, height // 2 - 9), text, fill=fg, font=_font(14, bold=True))
    return np.asarray(strip)


def _vstack(*arrs: np.ndarray) -> np.ndarray:
    return np.concatenate(arrs, axis=0)


def _hstack(*arrs: np.ndarray) -> np.ndarray:
    return np.concatenate(arrs, axis=1)


# ---------------------------------------------------------------------------
# Same-identity overlay
# ---------------------------------------------------------------------------


def _per_frame_psnr_ssim(pred: torch.Tensor, ref: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame PSNR / SSIM (`(T,)` arrays)."""
    T = pred.shape[0]
    ps, ss = np.zeros(T), np.zeros(T)
    for t in range(T):
        ps[t] = psnr_video(pred[t:t+1].unsqueeze(0), ref[t:t+1].unsqueeze(0)).item()
        ss[t] = ssim_video(pred[t:t+1].unsqueeze(0), ref[t:t+1].unsqueeze(0)).item()
    return ps, ss


def _per_frame_lpips(metric: LPIPSMetric, pred: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    T = pred.shape[0]
    out = np.zeros(T)
    for t in range(T):
        out[t] = metric(pred[t:t+1].unsqueeze(0), ref[t:t+1].unsqueeze(0)).item()
    return out


def _curves_png(psnr: np.ndarray, ssim: np.ndarray, lpips: np.ndarray,
                width: int = 1024, height: int = 240) -> np.ndarray:
    """Three stacked tracks rendered without matplotlib (no extra dep)."""
    panel = Image.new("RGB", (width, height), (240, 240, 240))
    draw  = ImageDraw.Draw(panel)
    track_h = height // 3
    series_list = [("PSNR (dB)", psnr,  (220,  60,  60)),
                   ("SSIM",       ssim, ( 50, 130, 220)),
                   ("LPIPS",     lpips, ( 80, 180,  80))]
    T = len(psnr)
    for i, (name, values, color) in enumerate(series_list):
        y_top    = i * track_h
        y_bottom = y_top + track_h - 4
        draw.rectangle((0, y_top, width - 1, y_bottom), outline=(180, 180, 180))
        v_min, v_max = float(values.min()), float(values.max())
        span = (v_max - v_min) or 1e-8
        pts = []
        for t in range(T):
            x = int(t / max(1, T - 1) * (width - 1))
            y = y_bottom - int((values[t] - v_min) / span * (y_bottom - y_top - 4))
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        draw.text((6, y_top + 2),
                  f"{name}  [{v_min:.2f} → {v_max:.2f}]  mean={values.mean():.3f}",
                  fill=(20, 20, 20), font=_font(12, bold=True))
    return np.asarray(panel)


def _build_same_id_overlay(
    sample, pred: torch.Tensor, ref: torch.Tensor,
    psnr_t: np.ndarray, ssim_t: np.ndarray, lpips_t: np.ndarray,
    out_mp4: Path,
) -> None:
    """Per-frame side-by-side: pred (red landmarks) ‖ GT (green landmarks),
    with per-frame metric values stamped on top."""
    T   = pred.shape[0]
    res = pred.shape[-1]
    lmd = LMD()
    print(f"[viz] writing same-id overlay → {out_mp4}")
    writer = iio_v2.get_writer(str(out_mp4), fps=DEFAULT_FPS, codec="libx264",
                               quality=8, macro_block_size=1)
    try:
        # Header strip stays the same width as the side-by-side composite.
        header_text = f"{sample.sample_id}  —  generated (red lms) ‖ ground truth (green lms)"
        for t in range(T):
            p_uint8 = (pred[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            r_uint8 = (ref [t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            lp = lmd._extract(p_uint8)
            lr = lmd._extract(r_uint8)
            p_drawn = _draw_landmarks(p_uint8, lp, color=(255,  64,  64))
            r_drawn = _draw_landmarks(r_uint8, lr, color=( 64, 220,  64))

            row     = _hstack(p_drawn, r_drawn)              # (H, 2W, 3)
            header  = _text_strip(header_text, width=2 * res, height=28)
            metrics = _text_strip(
                f"frame {t:3d}/{T-1:3d}    "
                f"PSNR {psnr_t[t]:6.2f} dB    "
                f"SSIM {ssim_t[t]:.3f}    "
                f"LPIPS {lpips_t[t]:.3f}",
                width=2 * res, height=32, bg=(40, 40, 40),
            )
            frame = _vstack(header, row, metrics)
            writer.append_data(frame)
    finally:
        writer.close()
    lmd.close()


# ---------------------------------------------------------------------------
# Cross-identity overlay (ID similarity)
# ---------------------------------------------------------------------------


def _build_cross_id_overlay(sample, pred: torch.Tensor, device: str,
                            out_mp4: Path) -> None:
    """Per-frame pred (with ArcFace bbox + cosine score) ‖ ref-clip mean
    face thumbnail (with bbox). Cosine is computed against an averaged
    identity prior built from the ref clip."""
    from experiments.evaluation_metrics.metrics.id_sim import IDSimilarity
    id_metric = IDSimilarity(device=device)

    print("[viz] loading reference clip for identity prior…")
    ref_clip_video = load_video(
        Path(sample.ref_clip["video_path"]), DEFAULT_FPS, DEFAULT_RESOLUTION,
    )
    prior = id_metric.build_identity_prior(ref_clip_video)
    if prior is None:
        raise SystemExit(
            "[viz] could not build identity prior — ArcFace failed on every "
            "frame of the reference clip. Visualization aborted."
        )

    # Pick a representative ref frame: first one with a successful detection.
    ref_thumb = None
    ref_thumb_bbox = None
    for t in range(ref_clip_video.shape[0]):
        img = (ref_clip_video[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        bgr = img[..., ::-1].copy()
        faces = id_metric.app.get(bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            ref_thumb = img
            ref_thumb_bbox = face.bbox
            break
    if ref_thumb is None:
        ref_thumb = (ref_clip_video[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ref_thumb_bbox = None

    T   = pred.shape[0]
    res = pred.shape[-1]

    # Pre-render the static ref column once with the bbox baked in.
    ref_col = _draw_bbox(ref_thumb, ref_thumb_bbox, color=(64, 220, 64), width=3)

    print(f"[viz] writing cross-id overlay → {out_mp4}")
    cosines: list[float] = []
    writer = iio_v2.get_writer(str(out_mp4), fps=DEFAULT_FPS, codec="libx264",
                               quality=8, macro_block_size=1)
    try:
        header_text = (
            f"{sample.sample_id}  —  generated (red bbox + cosine vs ref prior)"
            f"  ‖  reference {sample.ref_clip['uid']}"
        )
        for t in range(T):
            img  = (pred[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            bgr  = img[..., ::-1].copy()
            faces = id_metric.app.get(bgr)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                cos  = float(face.normed_embedding @ prior)
                bbox = face.bbox
            else:
                cos, bbox = float("nan"), None
            cosines.append(cos)

            pred_drawn = _draw_bbox(img, bbox, color=(255, 64, 64), width=3)
            row        = _hstack(pred_drawn, ref_col)
            header     = _text_strip(header_text, width=2 * res, height=28)
            cos_txt    = "n/a" if np.isnan(cos) else f"{cos:.3f}"
            running    = np.nanmean(cosines)
            metrics    = _text_strip(
                f"frame {t:3d}/{T-1:3d}    cosine {cos_txt}    "
                f"running mean {running:.3f}",
                width=2 * res, height=32, bg=(40, 40, 40),
            )
            frame = _vstack(header, row, metrics)
            writer.append_data(frame)
    finally:
        writer.close()

    n_hits = sum(1 for c in cosines if not np.isnan(c))
    print(f"[viz] cross-id mean cosine: {np.nanmean(cosines):.3f}  "
          f"(detect rate {n_hits}/{T})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    meta = load_run_metadata(args.run_dir)
    print(f"[viz] dataset={meta.dataset} protocol={meta.protocol}")

    samples = list(iter_samples(meta))
    if not samples:
        raise SystemExit(f"No samples found under {args.run_dir}/samples/")
    sample = (samples[0] if args.sample_id is None
              else next((s for s in samples if s.sample_id == args.sample_id), None))
    if sample is None:
        raise SystemExit(f"Sample `{args.sample_id}` not in {args.run_dir}.")
    print(f"[viz] sample={sample.sample_id}")

    pred = load_video(sample.pred_path, DEFAULT_FPS, DEFAULT_RESOLUTION)
    print(f"[viz] pred shape: {tuple(pred.shape)}")

    out_mp4 = args.out_mp4 or (sample.pred_path.parent / "metrics_overlay.mp4")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if meta.protocol == "same_identity_reconstruction":
        # Reload pred at native resolution for the face-detection step;
        # the crop helper does the final resize itself.
        pred_native = load_video(sample.pred_path, DEFAULT_FPS, resolution=None)
        ref         = load_video(sample.gt_video_path, DEFAULT_FPS, resolution=None,
                                 max_frames=pred_native.shape[0])
        if args.face_crop:
            from experiments.evaluation_metrics.metrics.evaluator import (
                _build_face_detector,
            )
            print("[viz] face-cropping pred and GT independently…")
            detect = _build_face_detector(args.device)
            pred = face_crop_around_detection(
                pred_native, detect, margin=DEFAULT_FACE_CROP_MARGIN,
                target_resolution=DEFAULT_RESOLUTION,
            )
            ref = face_crop_around_detection(
                ref, detect, margin=DEFAULT_FACE_CROP_MARGIN,
                target_resolution=DEFAULT_RESOLUTION,
            )
            if pred is None or ref is None:
                raise SystemExit(
                    "[viz] clip-level face detection failed on pred or GT — "
                    "rerun with --no-face-crop for raw-framing visuals."
                )
        else:
            pred = torch.nn.functional.interpolate(
                pred_native, size=(DEFAULT_RESOLUTION, DEFAULT_RESOLUTION),
                mode="bilinear", align_corners=False,
            )
            ref = torch.nn.functional.interpolate(
                ref, size=(DEFAULT_RESOLUTION, DEFAULT_RESOLUTION),
                mode="bilinear", align_corners=False,
            )
        pred, ref = truncate_to_match(pred, ref)
        T = pred.shape[0]
        print(f"[viz] paired T={T}")

        print("[viz] computing per-frame PSNR / SSIM…")
        psnr_t, ssim_t = _per_frame_psnr_ssim(pred, ref)
        print("[viz] computing per-frame LPIPS…")
        lpips_metric = LPIPSMetric(net="alex", device=args.device, chunk_size=16)
        lpips_t = _per_frame_lpips(lpips_metric, pred, ref)

        _build_same_id_overlay(sample, pred, ref, psnr_t, ssim_t, lpips_t, out_mp4)

        out_png = args.out_png or (sample.pred_path.parent / "metrics_curves.png")
        Image.fromarray(_curves_png(psnr_t, ssim_t, lpips_t,
                                    width=2 * pred.shape[-1])).save(out_png)
        print(f"[viz] curves     → {out_png}")
        print(f"[viz] mean PSNR={psnr_t.mean():.2f}dB  "
              f"SSIM={ssim_t.mean():.3f}  LPIPS={lpips_t.mean():.3f}")

    elif meta.protocol == "cross_identity":
        _build_cross_id_overlay(sample, pred, args.device, out_mp4)

    else:
        raise SystemExit(f"Unknown protocol: {meta.protocol}")

    print(f"[viz] overlay mp4 → {out_mp4}")


if __name__ == "__main__":
    main()
