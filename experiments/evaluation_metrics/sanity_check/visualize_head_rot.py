r"""Render a side-by-side head-rotation overlay for one (baseline, sample) pair.

Head rotation is read directly off the FLAME fit (``rot · neck_rot``) — no
external head-pose estimator. The metric reduces to the geodesic angular
distance between **frame-0-anchored** delta rotations of pred and target
clips (see ``metrics/src/head_rot.py`` for why we use deltas).

Layout per frame (top → bottom):
    [text strip: per-frame geodesic distance + cumulative mean]
    [pred face crop with axes  |  target face crop with axes]
    [title strip: baseline / dataset / protocol / sample_id]

The target is the **driving** signal:
  * ``same_identity_reconstruction`` → driver = ref = GT.
  * ``cross_identity`` → driver = ``sample.driver_clip.video_path``.

Per-sample numerical summary (``head_rot_dist`` in degrees, track_rate)
is written next to the mp4 as the same-named ``.json``.

Usage
-----

    PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_head_rot.py \
        --baseline marionette \
        --dataset hdtf \
        --protocol cross_identity \
        --sample-id id_0042_id_0099

    PYTHONPATH=. python experiments/evaluation_metrics/sanity_check/visualize_head_rot.py \
        --baseline xportrait \
        --dataset hdtf \
        --protocol same_identity_reconstruction \
        --sample-id id_0042 \
        --out-mp4 /tmp/head_rot_xport_hdtf_0042.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rsc

from experiments.evaluation_metrics.metrics.evaluator import (
    _derive_bucket, _gt_fit_root, _pred_fit_path,
)
from experiments.evaluation_metrics.metrics.src.head_rot import (
    _head_R, _quat_angular_dist_deg, head_axes_in_image,
)
from marionette.flame.flame import OPENCV2PYTORCH3D, CAP4DFlameSkinner, compute_flame


SOTA_ROOT       = Path("outputs/sota_comparison")
MARIONETTE_ROOT = Path("outputs/marionette_eval")
MANIFEST_DIR    = Path("experiments/sota_comparison/manifests")
HEAD_VERT_PATH  = "data/assets/flame/head_vertices.txt"
N_FRAMES        = 16
RES             = 512
M_CV            = OPENCV2PYTORCH3D[:3, :3].numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Side-by-side head-rotation overlay for one (baseline, sample) pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline",  required=True)
    p.add_argument("--dataset",   required=True, choices=["hdtf"])
    p.add_argument("--protocol",  required=True,
                   choices=["same_identity_reconstruction", "cross_identity"])
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out-mp4",   type=Path, default=None,
                   help="Override output mp4 path. Default: "
                        "outputs/test_metric/head_rot_sanity/<bucket>/<dataset>/<protocol>/<sid>/overlay.mp4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _load_manifest(dataset: str) -> dict:
    return {c["uid"]: c for c in
            json.loads((MANIFEST_DIR / f"{dataset}.json").read_text())["clips"]}


def _split_sample_id(sample_id: str, protocol: str) -> tuple[str, str]:
    if protocol == "same_identity_reconstruction":
        return sample_id, sample_id
    if "_id_" not in sample_id:
        raise ValueError(f"cross_identity sample_id `{sample_id}` lacks `_id_`")
    ref, drv = sample_id.split("_id_", 1)
    return ref, f"id_{drv}"


def _latest_run(parent: Path) -> Optional[Path]:
    runs = sorted([d for d in parent.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def _resolve_pred_video(baseline: str, dataset: str, protocol: str,
                        sample_id: str) -> tuple[Optional[Path], Optional[Path]]:
    parent = (MARIONETTE_ROOT / dataset / protocol if baseline == "marionette"
              else SOTA_ROOT / baseline / dataset / protocol)
    run = _latest_run(parent)
    if run is None:
        return None, None
    return run, run / "samples" / sample_id / "panel.mp4"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _square_head_bbox(verts_2d: np.ndarray, head_vert_ids: np.ndarray,
                      margin: float = 1.4) -> tuple[int, int, int, int]:
    pts = verts_2d[head_vert_ids][:, :2]
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side   = max(x1 - x0, y1 - y0) * margin
    half   = side / 2
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def _read_frame(video_path: Path, t: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {t} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crop_to_square(img: np.ndarray, box, target_res: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    pad_x0, pad_y0 = max(0, -x0), max(0, -y0)
    pad_x1, pad_y1 = max(0, x1 - w), max(0, y1 - h)
    if any((pad_x0, pad_y0, pad_x1, pad_y1)):
        img = cv2.copyMakeBorder(img, pad_y0, pad_y1, pad_x0, pad_x1,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        x0 += pad_x0; x1 += pad_x0; y0 += pad_y0; y1 += pad_y0
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (target_res, target_res), interpolation=cv2.INTER_AREA)


def _draw_axes(img: np.ndarray, R_cv: np.ndarray,
               tdx: int, tdy: int, size: int = 110) -> np.ndarray:
    """Draw the head-frame X/Y/Z axes by projecting their unit tips through
    the OpenCV-frame rotation matrix R_cv. Red=X (head's right), Green=Y
    (head's down), Blue=Z (head's forward)."""
    out = img.copy()
    for col, color in zip((0, 1, 2),
                          ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        x = int(size * R_cv[0, col] + tdx)
        y = int(size * R_cv[1, col] + tdy)
        cv2.line(out, (tdx, tdy), (x, y), color, 4, cv2.LINE_AA)
    return out


def _font(size: int = 14, bold: bool = False) -> ImageFont.ImageFont:
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _text_strip(text: str, width: int, height: int = 32,
                bg=(20, 20, 20), fg=(255, 255, 255), pad: int = 10) -> np.ndarray:
    strip = Image.new("RGB", (width, height), bg)
    ImageDraw.Draw(strip).text((pad, height // 2 - 9), text,
                                fill=fg, font=_font(13, bold=True))
    return np.asarray(strip)


def _per_frame_fit(fit: dict, t: int, cam_id: int = 0) -> dict:
    item = {
        "shape":   fit["shape"],
        "expr":    fit["expr"][[t]],
        "rot":     fit["rot"][[t]],
        "tra":     fit["tra"][[t]],
        "eye_rot": fit["eye_rot"][[t]],
        "fx":      fit["fx"][[cam_id]], "fy": fit["fy"][[cam_id]],
        "cx":      fit["cx"][[cam_id]], "cy": fit["cy"][[cam_id]],
        "extr":    fit["extr"][[cam_id]],
    }
    if "neck_rot" in fit: item["neck_rot"] = fit["neck_rot"][[t]]
    if "jaw_rot"  in fit: item["jaw_rot"]  = fit["jaw_rot"][[t]]
    return item


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    manifest = _load_manifest(args.dataset)
    ref_uid, drv_uid = _split_sample_id(args.sample_id, args.protocol)
    target_uid = drv_uid
    if target_uid not in manifest:
        raise SystemExit(f"target uid {target_uid} not in {args.dataset} manifest")
    target_clip_id = manifest[target_uid]["clip_id"]

    run_dir, pred_video = _resolve_pred_video(
        args.baseline, args.dataset, args.protocol, args.sample_id,
    )
    if pred_video is None or not pred_video.is_file():
        raise SystemExit(
            f"no panel.mp4 for {args.baseline}/{args.dataset}/{args.protocol}/{args.sample_id}"
        )
    target_video = Path(manifest[target_uid]["video_path"])

    pred_fit_path = _pred_fit_path(run_dir, args.dataset, args.protocol,
                                   args.sample_id)
    target_fit_path = _gt_fit_root() / target_clip_id / "fit.npz"
    if not pred_fit_path.is_file():
        raise SystemExit(f"pred FLAME fit missing: {pred_fit_path}")
    if not target_fit_path.is_file():
        raise SystemExit(f"target FLAME fit missing: {target_fit_path}")

    out_mp4 = args.out_mp4 or Path(
        f"outputs/test_metric/head_rot_sanity/{args.baseline}/{args.dataset}/"
        f"{args.protocol}/{args.sample_id}/overlay.mp4"
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f"[viz] {args.baseline}/{args.dataset}/{args.protocol}/{args.sample_id}")
    print(f"[viz] pred fit   = {pred_fit_path}")
    print(f"[viz] target fit = {target_fit_path}")

    pred_fit   = dict(np.load(str(pred_fit_path)))
    target_fit = dict(np.load(str(target_fit_path)))
    T = min(N_FRAMES, pred_fit["expr"].shape[0], target_fit["expr"].shape[0])
    if T < 2:
        raise SystemExit(f"need ≥ 2 frames in both fits; got T={T}")

    skinner = CAP4DFlameSkinner(add_mouth=True, n_shape_params=150,
                                n_expr_params=65).eval()
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    # Per-frame R_head, frame-0-anchored deltas, geodesic distances.
    neck_p = pred_fit  .get("neck_rot", np.zeros_like(pred_fit  ["rot"]))
    neck_t = target_fit.get("neck_rot", np.zeros_like(target_fit["rot"]))
    R_pred = [_head_R(pred_fit  ["rot"][t], neck_p[t]) for t in range(T)]
    R_tgt  = [_head_R(target_fit["rot"][t], neck_t[t]) for t in range(T)]
    dR_pred = [R_pred[t] @ R_pred[0].T for t in range(T)]
    dR_tgt  = [R_tgt [t] @ R_tgt [0].T for t in range(T)]
    geo = [_quat_angular_dist_deg(dR_pred[t], dR_tgt[t]) for t in range(T)]
    cum_mean = np.cumsum(geo) / (np.arange(T) + 1)

    print(f"[viz] writing {out_mp4}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, 8, (RES * 2, RES + 60))

    header = _text_strip(
        f"{args.baseline} / {args.dataset} / {args.protocol} / {args.sample_id}    "
        f"left=pred  right=target (uid={target_uid})",
        width=RES * 2, height=28,
    )

    for t in range(T):
        # FLAME bbox per side per frame.
        item_pred   = _per_frame_fit(pred_fit, t)
        item_target = _per_frame_fit(target_fit, t)
        verts_pred = compute_flame(skinner, item_pred  )["verts_2d"][0, 0]
        verts_tgt  = compute_flame(skinner, item_target)["verts_2d"][0, 0]
        bbox_p = _square_head_bbox(verts_pred, head_vert_ids)
        bbox_t = _square_head_bbox(verts_tgt,  head_vert_ids)

        crop_p = _crop_to_square(_read_frame(pred_video,   t), bbox_p, RES)
        crop_t = _crop_to_square(_read_frame(target_video, t), bbox_t, RES)

        # Axes from absolute head rotation (so they visually follow the head),
        # in OpenCV image frame (M_CV · R_p3d · M_CV).
        R_p_cv = M_CV @ R_pred[t] @ M_CV
        R_t_cv = M_CV @ R_tgt [t] @ M_CV
        crop_p = _draw_axes(crop_p, R_p_cv, tdx=RES // 2, tdy=RES // 2)
        crop_t = _draw_axes(crop_t, R_t_cv, tdx=RES // 2, tdy=RES // 2)

        row = np.concatenate([crop_p, crop_t], axis=1)
        metrics_text = (
            f"frame {t:2d}/{T-1:2d}    "
            f"geodesic = {geo[t]:5.2f}°    "
            f"running mean = {cum_mean[t]:5.2f}°"
        )
        metrics_strip = _text_strip(metrics_text, width=RES * 2, height=32,
                                    bg=(40, 40, 40))
        panel = np.concatenate([metrics_strip, row], axis=0)
        # Append static title strip below the panel.
        full = np.concatenate([panel, header], axis=0)
        writer.write(cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

    writer.release()

    summary = {
        "head_rot_dist":       float(np.mean(geo)),
        "head_rot_track_rate": float(T) / float(N_FRAMES),
        "n_frames":            T,
    }
    out_json = out_mp4.with_suffix(".json")
    out_json.write_text(json.dumps({
        "baseline":         args.baseline,
        "dataset":          args.dataset,
        "protocol":         args.protocol,
        "sample_id":        args.sample_id,
        "ref_uid":          ref_uid,
        "target_uid":       target_uid,
        "pred_fit":         str(pred_fit_path),
        "target_fit":       str(target_fit_path),
        "summary":          summary,
        "geodesic_per_frame": [float(x) for x in geo],
    }, indent=2))

    print(f"[viz] mp4  → {out_mp4}")
    print(f"[viz] json → {out_json}")
    print(f"[viz] summary: {summary}")


if __name__ == "__main__":
    main()
