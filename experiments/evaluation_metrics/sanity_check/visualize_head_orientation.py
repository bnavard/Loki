r"""Render a side-by-side head-orientation overlay for one (baseline, sample) pair.

"Head orientation" = rigid head rotation extracted by 6DRepNet (yaw,
pitch, roll). We name it "orientation" rather than "pose" to avoid
collision with FLAME's `θ` (jaw + neck + head pose parameters).

Layout per frame (top → bottom):
    [text strip: pred (yaw/pitch/roll), target (y/p/r), per-axis L1 in deg]
    [pred face crop with rotated XYZ axis  |  target face crop with rotated XYZ axis]
    [title strip: baseline / dataset / protocol / sample_id]

The target is the **driving** signal:
  * `same_identity_reconstruction` → driver = ref = GT (single source clip per sample).
  * `cross_identity` → driver = `sample.driver_clip.video_path` (different identity).

Both pred and target are face-cropped via the same `face_crop_around_detection`
the rest of the metric suite uses (1.3× margin → 512×512), then run through
6DRepNet for `(yaw, pitch, roll)` per frame. The mp4 has the axis basis
drawn on each face so a viewer can eyeball whether the numbers track what's
actually happening.

Per-sample numerical summary (yaw/pitch/roll L1 in degrees,
axis_mean_l1, detect_rate) is written next to the mp4 as the same-named
`.json`.

Usage
-----

    PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_head_orientation.py \
        --baseline marionette \
        --dataset talkvid \
        --protocol cross_identity \
        --sample-id id_0042_id_0099

    PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_head_orientation.py \
        --baseline xportrait \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --sample-id id_0042 \
        --out-mp4 /tmp/head_orientation_xport_hdtf_0042.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import imageio.v2 as iio_v2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.evaluation_metrics.metrics.evaluator import _build_face_detector
from experiments.evaluation_metrics.metrics.head_orientation import (
    HeadOrientationEstimator, draw_axis,
)
from experiments.evaluation_metrics.metrics.io import (
    DEFAULT_FACE_CROP_MARGIN, DEFAULT_FPS, DEFAULT_RESOLUTION,
    face_crop_around_detection, load_video,
)


SOTA_ROOT       = Path("outputs/sota_comparison")
MARIONETTE_ROOT = Path("outputs/marionette_eval")
MANIFEST_DIR    = Path("experiments/sota_comparison/manifests")
N_FRAMES        = 16   # match cfg.inference.n_frames so we score the same window everywhere


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Side-by-side head-orientation overlay for one (baseline, sample) pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline",  required=True,
                   help="`marionette` or a SOTA name (sadtalker / anitalker / …).")
    p.add_argument("--dataset",   required=True, choices=["talkvid", "hdtf"])
    p.add_argument("--protocol",  required=True,
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--sample-id", required=True,
                   help="UID-based id, e.g. `id_0042` (same-id) or "
                        "`id_0042_id_0099` (cross-id).")
    p.add_argument("--out-mp4",   type=Path, default=None,
                   help="Override output mp4 path. Default: "
                        "outputs/test_metric/head_orientation_sanity/<bucket>/<dataset>/<protocol>/<sid>/overlay.mp4")
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _load_manifest(dataset: str) -> dict:
    return {c["uid"]: c for c in
            json.loads((MANIFEST_DIR / f"{dataset}.json").read_text())["clips"]}


def _split_sample_id(sample_id: str, protocol: str) -> tuple[str, str]:
    if protocol == "same_identity_reconstruction":
        return sample_id, sample_id
    if "_id_" not in sample_id:
        raise ValueError(f"cross_identity sample_id `{sample_id}` lacks `_id_` separator")
    ref, drv = sample_id.split("_id_", 1)
    return ref, f"id_{drv}"


def _latest_run(parent: Path) -> Optional[Path]:
    runs = sorted([d for d in parent.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def _resolve_pred_path(baseline: str, dataset: str, protocol: str, sample_id: str) -> Optional[Path]:
    if baseline == "marionette":
        run = _latest_run(MARIONETTE_ROOT / dataset / protocol)
    else:
        run = _latest_run(SOTA_ROOT / baseline / dataset / protocol)
    if run is None:
        return None
    p = run / "samples" / sample_id / "panel.mp4"
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# Drawing helpers (shared style with visualize_sample.py)
# ---------------------------------------------------------------------------


def _font(size: int = 14, bold: bool = False) -> ImageFont.ImageFont:
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _text_strip(text: str, width: int, height: int = 36,
                bg=(20, 20, 20), fg=(255, 255, 255), pad: int = 10) -> np.ndarray:
    strip = Image.new("RGB", (width, height), bg)
    ImageDraw.Draw(strip).text((pad, height // 2 - 9), text, fill=fg, font=_font(13, bold=True))
    return np.asarray(strip)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    manifest = _load_manifest(args.dataset)
    ref_uid, drv_uid = _split_sample_id(args.sample_id, args.protocol)
    target_uid = drv_uid                              # driver for both protocols (= ref for same-id)
    if target_uid not in manifest:
        raise SystemExit(f"target uid {target_uid} not in {args.dataset} manifest")

    pred_path = _resolve_pred_path(args.baseline, args.dataset, args.protocol, args.sample_id)
    if pred_path is None:
        raise SystemExit(
            f"no panel.mp4 for {args.baseline}/{args.dataset}/{args.protocol}/{args.sample_id}"
        )
    target_path = Path(manifest[target_uid]["video_path"])

    out_mp4 = args.out_mp4 or Path(
        f"outputs/test_metric/head_orientation_sanity/{args.baseline}/{args.dataset}/"
        f"{args.protocol}/{args.sample_id}/overlay.mp4"
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f"[viz] {args.baseline}/{args.dataset}/{args.protocol}/{args.sample_id}")
    print(f"[viz] pred   = {pred_path}")
    print(f"[viz] target = {target_path}  (uid={target_uid})")

    # Load both at native resolution; the face-crop step does the final 512×512 resize.
    pred   = load_video(pred_path,   fps=DEFAULT_FPS, resolution=None, max_frames=N_FRAMES)
    target = load_video(target_path, fps=DEFAULT_FPS, resolution=None, max_frames=N_FRAMES)

    print("[viz] face-cropping pred and target…")
    detect = _build_face_detector(args.device)
    pred_crop = face_crop_around_detection(
        pred, detect, margin=DEFAULT_FACE_CROP_MARGIN,
        target_resolution=DEFAULT_RESOLUTION,
    )
    target_crop = face_crop_around_detection(
        target, detect, margin=DEFAULT_FACE_CROP_MARGIN,
        target_resolution=DEFAULT_RESOLUTION,
    )
    if pred_crop is None or target_crop is None:
        raise SystemExit(
            "[viz] clip-level face detection failed on pred or target — cannot run head orientation."
        )

    print("[viz] estimating head orientation with 6DRepNet…")
    hoe = HeadOrientationEstimator(device=args.device)

    T   = min(pred_crop.shape[0], target_crop.shape[0])
    res = pred_crop.shape[-1]
    per_frame: list[Optional[dict]] = []
    pred_frames:   list[np.ndarray] = []
    target_frames: list[np.ndarray] = []

    for t in range(T):
        p_u8 = (pred_crop  [t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        g_u8 = (target_crop[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        p_pose = hoe.extract(p_u8)
        g_pose = hoe.extract(g_u8)
        if p_pose is None or g_pose is None:
            per_frame.append(None)
            pred_frames  .append(p_u8)
            target_frames.append(g_u8)
            continue
        # Axis size proportional to the cropped resolution (512 → ~120 px arrow).
        size = res * 0.23
        pred_frames  .append(draw_axis(p_u8, *p_pose, tdx=res / 2, tdy=res / 2, size=size))
        target_frames.append(draw_axis(g_u8, *g_pose, tdx=res / 2, tdy=res / 2, size=size))
        per_frame.append({
            "pred":        {"yaw": p_pose[0], "pitch": p_pose[1], "roll": p_pose[2]},
            "target":      {"yaw": g_pose[0], "pitch": g_pose[1], "roll": g_pose[2]},
            "axis_l1_err": {
                "yaw":   abs(p_pose[0] - g_pose[0]),
                "pitch": abs(p_pose[1] - g_pose[1]),
                "roll":  abs(p_pose[2] - g_pose[2]),
            },
        })

    # ------------- render side-by-side mp4 -------------
    print(f"[viz] writing {out_mp4}")
    writer = iio_v2.get_writer(str(out_mp4), fps=DEFAULT_FPS, codec="libx264",
                               quality=8, macro_block_size=1)
    try:
        header_text = (f"{args.baseline}  /  {args.dataset}  /  {args.protocol}  /  "
                       f"{args.sample_id}     left = generated     right = driving (uid={target_uid})")
        for t in range(T):
            row    = np.concatenate([pred_frames[t], target_frames[t]], axis=1)
            header = _text_strip(header_text, width=2 * res, height=28)
            if per_frame[t] is None:
                metrics_text = f"frame {t:2d}/{T-1:2d}    head-orientation detection failed"
            else:
                e   = per_frame[t]
                p_p = e["pred"];   t_p = e["target"];   l1 = e["axis_l1_err"]
                metrics_text = (
                    f"frame {t:2d}/{T-1:2d}    "
                    f"pred (y/p/r) {p_p['yaw']:+6.1f}° / {p_p['pitch']:+6.1f}° / {p_p['roll']:+6.1f}°    "
                    f"target {t_p['yaw']:+6.1f}° / {t_p['pitch']:+6.1f}° / {t_p['roll']:+6.1f}°    "
                    f"|err| y/p/r {l1['yaw']:5.1f}° / {l1['pitch']:5.1f}° / {l1['roll']:5.1f}°"
                )
            metrics_strip = _text_strip(metrics_text, width=2 * res, height=32, bg=(40, 40, 40))
            # Per-frame numbers on TOP (eye-level, easy to track frame-by-frame);
            # static title on the BOTTOM (sample id / paths don't change).
            writer.append_data(np.concatenate([metrics_strip, row, header], axis=0))
    finally:
        writer.close()

    # ------------- aggregate + json -------------
    valid = [e for e in per_frame if e is not None]
    if not valid:
        agg = {"n_valid_frames": 0, "n_frames": T, "detect_rate": 0.0}
    else:
        ys = np.array([e["axis_l1_err"]["yaw"]   for e in valid])
        ps = np.array([e["axis_l1_err"]["pitch"] for e in valid])
        rs = np.array([e["axis_l1_err"]["roll"]  for e in valid])
        agg = {
            "yaw_l1":         float(ys.mean()),
            "pitch_l1":       float(ps.mean()),
            "roll_l1":        float(rs.mean()),
            "axis_mean_l1":   float((ys + ps + rs).mean() / 3),
            "n_valid_frames": len(valid),
            "n_frames":       T,
            "detect_rate":    len(valid) / T,
        }

    out_json = out_mp4.with_suffix(".json")
    out_json.write_text(json.dumps({
        "baseline":    args.baseline,
        "dataset":     args.dataset,
        "protocol":    args.protocol,
        "sample_id":   args.sample_id,
        "ref_uid":     ref_uid,
        "target_uid":  target_uid,
        "pred_path":   str(pred_path),
        "target_path": str(target_path),
        "summary":     agg,
        "per_frame":   per_frame,
    }, indent=2))

    print(f"[viz] mp4  → {out_mp4}")
    print(f"[viz] json → {out_json}")
    print(f"[viz] summary: {agg}")


if __name__ == "__main__":
    main()
