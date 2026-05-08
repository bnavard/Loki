"""Visualise the FLAME expression-deformation swap behind the
``expression_l1`` metric.

Renders a 3×3 mp4 panel for one (target_fit, pred_fit) pair:

    Row 1 (real video frames)    : target | pred           | (—)
    Row 2 (rasterised mesh)      : target | pred (own pose)| swap (target-pose, pred-expr)
    Row 3 (rasterised expression): target | pred (own pose)| swap (target-pose, pred-expr)

Where the per-cell renderings come from:
    target           — render target_fit verbatim.
    pred (own pose)  — render pred_fit verbatim. Different identity, different pose,
                       different expression. Provided for visual context.
    swap             — render a fit that combines target's pose / shape / camera /
                       neck with pred's (expr, eye_rot, jaw_rot). The metric
                       ``expression_l1`` scores this against the target render —
                       both share image-space pose by construction, so the only
                       thing that can drive a per-pixel residual is expression.

Row 3 col 3 (the swap deform map with a heatmap of |target − swap| painted on
it) is exactly what ``expression_l1`` measures. Row 2 col 3 is the same swap
but in the pos_enc (mesh-position) space — should look almost identical to
row 2 col 1, since the swap shares target's pose; if it doesn't, the swap
plumbing is broken.

Usage (from repo root)::

    PYTHONPATH=. /venv/expmapgen/bin/python \
        experiments/evaluation_metrics/sanity_check/visualize_expression.py \
        --target-fit   data/benchmark/hdtf/flame_tracking/flowface/<clip_id>/fit.npz \
        --pred-fit     data/flame_tracking/preds/<baseline>/<dataset>/<protocol>/<sample_id>/fit.npz \
        --target-video data/benchmark/hdtf/clips/<clip_id>.mp4 \
        --pred-video   outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>/samples/<sample_id>/panel.mp4 \
        --out-mp4      outputs/test_metric/expr_sanity_swap/<baseline>_<sample_id>.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from loki.conditioning.conditioning import SpatialConditioning
from loki.flame.flame import CAP4DFlameSkinner, compute_flame
from loki.utils import (
    crop_image, get_bbox_from_verts, load_frame, rescale_image,
    verts_to_pytorch3d,
)

HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


# ---------------------------------------------------------------------------
# Per-frame fit slicing + the swap operation that defines the metric.
# Mirrors `metrics/src/expression.py` so what's drawn matches what's scored.
# ---------------------------------------------------------------------------


def _per_frame_fit(fit: dict, t: int, cam_id: int = 0) -> dict:
    """Slice a single-frame fit dict from a multi-frame fit.npz."""
    item = {
        "shape":   fit["shape"],
        "expr":    fit["expr"][[t]],
        "rot":     fit["rot"][[t]],
        "tra":     fit["tra"][[t]],
        "eye_rot": fit["eye_rot"][[t]],
        "fx":      fit["fx"][[cam_id]],
        "fy":      fit["fy"][[cam_id]],
        "cx":      fit["cx"][[cam_id]],
        "cy":      fit["cy"][[cam_id]],
        "extr":    fit["extr"][[cam_id]],
    }
    if "neck_rot" in fit:
        item["neck_rot"] = fit["neck_rot"][[t]]
    if "jaw_rot" in fit:
        item["jaw_rot"] = fit["jaw_rot"][[t]]
    return item


def _swap_expression(target_item: dict, pred_item: dict) -> dict:
    """Build the swap fit: target's pose / shape / camera / neck + pred's
    (expr, eye_rot, jaw_rot). Identical to ``metrics/src/expression.py``'s
    ``_swap_expression`` — kept here so the visualisation matches what the
    metric scores."""
    out = {**target_item}
    out["expr"]    = pred_item["expr"]
    out["eye_rot"] = pred_item["eye_rot"]
    if "jaw_rot" in pred_item:
        out["jaw_rot"] = pred_item["jaw_rot"]
    return out


# ---------------------------------------------------------------------------
# Rasterisation + image utilities
# ---------------------------------------------------------------------------


def _rasterize(
    cond: SpatialConditioning,
    skinner: CAP4DFlameSkinner,
    item: dict,
    crop_box,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """compute_flame → NDC verts → rasterizer.

    Returns:
        pos_enc_rgb : (H, W, 3) uint8   — per-channel normalised, for display.
        deform_rgb  : (H, W, 3) uint8   — per-channel normalised, for display.
        pos_enc_raw : (H, W, 3) float32 — masked raw values, for L2 metric.
        deform_raw  : (H, W, 3) float32 — masked raw values, for L2 metric.
        mask_bool   : (H, W) bool       — on-mesh indicator.

    The `*_raw` channels are the same units the actual `expression_l1`
    metric works on (deform offsets pre-divided by `std_expr_deformation`
    inside `_rasterize_conditioning`), so an L2 computed on them equals
    the metric value for that frame.
    """
    out = compute_flame(skinner, item)
    verts_2d  = out["verts_2d"][0, 0].copy()
    offsets   = out["offsets_3d"][0]
    verts_ndc = verts_to_pytorch3d(verts_2d, np.array(crop_box))

    verts_t   = torch.from_numpy(verts_ndc).float()[None].to(device)
    offsets_t = torch.from_numpy(offsets).float()[None].to(device)

    pos_enc_input, deform_map, mask = cond._rasterize_conditioning(
        verts_t, offsets_t,
    )
    pos_enc_raw = (pos_enc_input * mask).cpu().numpy()[0]
    deform_raw  = (deform_map   * mask).cpu().numpy()[0]
    mask_bool   = mask.cpu().numpy()[0, ..., 0] > 0
    return _to_rgb(pos_enc_raw), _to_rgb(deform_raw), pos_enc_raw, deform_raw, mask_bool


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Map a 3-channel float feature map into a viewable uint8 RGB image,
    independently per-channel min-max normalised across the on-mesh
    pixels (so the visualisation isn't dominated by background zeros)."""
    out = np.zeros_like(arr, dtype=np.float32)
    nonzero = np.any(arr != 0, axis=-1)
    if nonzero.any():
        for c in range(3):
            ch = arr[..., c]
            vals = ch[nonzero]
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo > 1e-8:
                out[..., c] = np.where(nonzero, (ch - lo) / (hi - lo), 0.0)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def _diff_overlay(swap: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Alpha-blend a JET heatmap of |swap − target| onto the swap render.
    Output shows the swap with red/yellow blotches wherever it differs from
    the target — direct visualisation of where the expression mismatch
    lives on the face. Per-frame max-normalised, so the colour scale
    always uses the full dynamic range; absolute magnitude lives in the
    label number."""
    diff = np.abs(swap.astype(np.float32) - target.astype(np.float32)).mean(axis=-1)
    if diff.max() > 1e-6:
        diff_u8 = (diff / diff.max() * 255).clip(0, 255).astype(np.uint8)
    else:
        diff_u8 = np.zeros_like(diff, dtype=np.uint8)
    heat = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    # Only overlay where the swap render has support (non-zero pixels), so
    # we don't paint heat onto the empty background.
    mask   = np.any(swap != 0, axis=-1, keepdims=True).astype(np.float32)
    weight = 0.55 * mask
    return (swap.astype(np.float32) * (1 - weight)
            + heat.astype(np.float32) * weight).clip(0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _masked_l1(a: np.ndarray, b: np.ndarray, mask_bool: np.ndarray) -> float:
    """Per-pixel mean-absolute-deviation across the 3 channels, then mean
    over on-mesh pixels. Identical to `metrics/src/expression.py::_masked_l1`,
    so a value computed here matches the real `expression_l1` metric for that
    frame when called on (target_def_raw, swap_def_raw, target_mask)."""
    per_pixel = np.abs(a - b).mean(axis=-1)
    if not mask_bool.any():
        return 0.0
    return float(per_pixel[mask_bool].mean())


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _infer_video_path(fit_path: Path) -> Path | None:
    """Convention: <dataset_root>/flame_tracking/<...>/<clip_id>/fit.npz pairs
    with <dataset_root>/_eval_inputs/<clip_id>.mp4. Returns None if not
    found — caller should pass --target-video / --pred-video explicitly
    when the convention doesn't match (e.g. for the pred-fit tree)."""
    clip_id = fit_path.parent.name
    p = fit_path
    while p.parent != p:
        p = p.parent
        cand = p / "_eval_inputs" / f"{clip_id}.mp4"
        if cand.is_file():
            return cand
    return None


def _load_video_frame(
    video_path: Path, t: int, fit: dict, head_vert_ids: np.ndarray,
    skinner: CAP4DFlameSkinner, resolution: int,
) -> np.ndarray:
    """Read frame `t` from `video_path` and crop / rescale to match the
    rendered mesh — same head-bbox-from-verts logic the dataset uses."""
    item = _per_frame_fit(fit, t)
    verts_2d = compute_flame(skinner, item)["verts_2d"][0, 0]
    crop_box = get_bbox_from_verts(verts_2d.copy(), head_vert_ids)
    img = load_frame(video_path, t)
    img = crop_image(img, crop_box, bg_value=255)
    img = rescale_image(img, resolution)
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise the FLAME expression swap behind expression_l1 "
                    "(renders a 3×3 mp4 for one (target_fit, pred_fit) pair).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target-fit", "--fit-a", type=Path, required=True,
                   help="Target fit.npz — supplies pose / shape / camera / "
                        "neck. Same role as the target_fit in expression_l1.")
    p.add_argument("--pred-fit",   "--fit-b", type=Path, required=True,
                   help="Pred fit.npz — supplies the (expr, eye_rot, jaw_rot) "
                        "that gets substituted into target's pose for the "
                        "swap render.")
    p.add_argument("--target-video", "--video-a", type=Path, default=None,
                   help="Source mp4 for target-fit (the driver / GT clip). "
                        "Default: inferred from the fit path.")
    p.add_argument("--pred-video",   "--video-b", type=Path, default=None,
                   help="Source mp4 for pred-fit (the baseline's generation). "
                        "Default: inferred — usually you'll want to pass it "
                        "explicitly since pred fits live under "
                        "data/flame_tracking/preds/ which doesn't follow the "
                        "_eval_inputs/ convention.")
    p.add_argument("--out-mp4",    type=Path, required=True)
    p.add_argument("--n-frames",   type=int,  default=16)
    p.add_argument("--resolution", type=int,  default=512)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--fps",        type=int,  default=8)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    target_fit = dict(np.load(str(args.target_fit)))
    pred_fit   = dict(np.load(str(args.pred_fit)))
    n = min(args.n_frames, target_fit["expr"].shape[0], pred_fit["expr"].shape[0])

    # Skinner stays on CPU (compute_flame builds CPU tensors). Only the
    # rasterizer needs GPU.
    skinner = CAP4DFlameSkinner(
        add_mouth=True, n_shape_params=150, n_expr_params=65,
    ).eval()
    cond = SpatialConditioning(image_size=args.resolution).to(args.device).eval()
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    target_video = args.target_video or _infer_video_path(args.target_fit)
    pred_video   = args.pred_video   or _infer_video_path(args.pred_fit)
    if target_video is None or pred_video is None:
        raise SystemExit(
            f"could not infer source videos. target_video={target_video}, "
            f"pred_video={pred_video}. Pass --target-video / --pred-video "
            "explicitly."
        )

    R = args.resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.out_mp4), fourcc, args.fps, (R * 3, R * 3),
    )

    print(f"[viz] target = {args.target_fit.parent.name}")
    print(f"[viz] pred   = {args.pred_fit.parent.name}")
    print(f"[viz] {n} frames  →  {args.out_mp4}")
    print(f"[viz] target_video={target_video}")
    print(f"[viz] pred_video  ={pred_video}")

    # Placeholder for the (—) cell at row 1 col 3 (swap is synthetic, no source video).
    blank = np.full((R, R, 3), 64, dtype=np.uint8)
    cv2.putText(blank, "(swap: synthetic — no source video)",
                (16, R // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)

    for t in range(n):
        item_target = _per_frame_fit(target_fit, t)
        item_pred   = _per_frame_fit(pred_fit, t)
        item_swap   = _swap_expression(item_target, item_pred)

        # Crop boxes: target and swap share target's pose, so use target's
        # bbox; pred (own pose) has its own bbox.
        verts_target = compute_flame(skinner, item_target)["verts_2d"][0, 0]
        verts_pred   = compute_flame(skinner, item_pred  )["verts_2d"][0, 0]
        crop_target  = get_bbox_from_verts(verts_target.copy(), head_vert_ids)
        crop_pred    = get_bbox_from_verts(verts_pred.copy(),   head_vert_ids)

        pos_target, def_target, pos_target_raw, def_target_raw, mask_target = \
            _rasterize(cond, skinner, item_target, crop_target, args.device)
        pos_pred, def_pred, _, _, _ = \
            _rasterize(cond, skinner, item_pred, crop_pred, args.device)
        pos_swap, def_swap, pos_swap_raw, def_swap_raw, _ = \
            _rasterize(cond, skinner, item_swap, crop_target, args.device)

        # Real per-frame metric values, computed identically to
        # metrics/src/expression.py — per-pixel mean-absolute-deviation
        # across the 3 channels, then mean over target's on-mesh pixels.
        # The deform-channel L1 is exactly `expression_l1` for this frame;
        # the pos-enc L1 is a sanity check that should sit ≈ 0 (the swap
        # shares target's pose, so the rasterised mesh positions should
        # match almost exactly).
        l1_pos = _masked_l1(pos_swap_raw, pos_target_raw, mask_target)
        l1_def = _masked_l1(def_swap_raw, def_target_raw, mask_target)

        vid_target = _load_video_frame(target_video, t, target_fit, head_vert_ids, skinner, R)
        vid_pred   = _load_video_frame(pred_video,   t, pred_fit,   head_vert_ids, skinner, R)

        pos_swap_diff = _diff_overlay(pos_swap, pos_target)
        def_swap_diff = _diff_overlay(def_swap, def_target)

        row0 = np.concatenate([
            _label(vid_target, f"t={t}  Target video (driver / GT)"),
            _label(vid_pred,   "Pred video (baseline output)"),
            _label(blank,      "—"),
        ], axis=1)
        row1 = np.concatenate([
            _label(pos_target,    "Target mesh (pos_enc)"),
            _label(pos_pred,      "Pred mesh (own pose)"),
            _label(pos_swap_diff, f"Swap mesh + |target-swap| heat   pos_l1={l1_pos:.4f}"),
        ], axis=1)
        row2 = np.concatenate([
            _label(def_target,    "Target expression (deform)"),
            _label(def_pred,      "Pred expression (own pose)"),
            _label(def_swap_diff, f"Swap expression + heat   expression_l1={l1_def:.4f}"),
        ], axis=1)
        panel = np.concatenate([row0, row1, row2], axis=0)
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        print(f"  t={t:02d}  expression_l1={l1_def:.4f}   pos_l1={l1_pos:.4f}")

    writer.release()
    print(f"[viz] wrote {args.out_mp4}  ({args.out_mp4.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
