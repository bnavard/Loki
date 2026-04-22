"""
FLAME retargeting utilities — pure functions shared across inference,
evaluation, and future viz callbacks.

Core idea: FLAME's topology is identity-agnostic (same V, same face indices
across identities). Under the reference's camera and β_ref, applying the
driver's expression ψ and head pose θ yields a mesh that:
  - lives in the reference's pixel space (so the warp grid lookup is valid),
  - follows the driver's motion (so the generated video matches the driver).

Same-identity case (ref_clip == driver_clip) is a no-op: `retarget_driver_verts`
just runs the reference's own FLAME; the retargeted verts equal the reference's
own projected verts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from marionette.flame.flame import CAP4DFlameSkinner, compute_flame
from marionette.utils import (
    load_frame, crop_image, rescale_image, get_bbox_from_verts, verts_to_pytorch3d,
)


def prepare_reference(
    ref_fit: dict,
    ref_frame: int,
    video_path: Path,
    resolution: int,
    flame_skinner: CAP4DFlameSkinner,
    head_vert_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the reference frame image + compute its NDC verts + face crop box.

    This is the identity anchor used by the warp source: `ref_image` feeds
    grid_sample, `ref_verts_ndc` provides the per-vertex UV coordinates, and
    `crop_box` pins the coordinate system for driver-verts retargeting.

    Returns:
        ref_image     : (H, W, 3) float32 in [-1, 1]
        ref_verts_ndc : (V, 3) in pytorch3d NDC relative to crop_box
        crop_box      : np.ndarray [x0, y0, x1, y1]
    """
    fi = {
        "shape":   ref_fit["shape"],
        "expr":    ref_fit["expr"][[ref_frame]],
        "rot":     ref_fit["rot"][[ref_frame]],
        "tra":     ref_fit["tra"][[ref_frame]],
        "eye_rot": ref_fit["eye_rot"][[ref_frame]],
        "fx":      ref_fit["fx"][[0]], "fy": ref_fit["fy"][[0]],
        "cx":      ref_fit["cx"][[0]], "cy": ref_fit["cy"][[0]],
        "extr":    ref_fit["extr"][[0]],
    }
    if "jaw_rot" in ref_fit:
        fi["jaw_rot"] = ref_fit["jaw_rot"][[ref_frame]]

    fo = compute_flame(flame_skinner, fi)
    verts_2d = fo["verts_2d"][0, 0]
    crop_box = get_bbox_from_verts(verts_2d.copy(), head_vert_ids)

    img = load_frame(video_path, ref_frame)
    img = crop_image(img, crop_box, bg_value=255)
    img = rescale_image(img, resolution)
    img_norm = (img.astype(np.float32) / 127.5) - 1.0

    ref_verts_ndc = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))
    return img_norm, ref_verts_ndc, crop_box


def retarget_driver_verts(
    ref_fit: dict,
    driver_fit: dict,
    crop_box: np.ndarray,
    n_frames: int,
    flame_skinner: CAP4DFlameSkinner,
    driver_start: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame FLAME: β_ref + ψ_driver[t] + θ_driver[t] under the ref's camera.

    Same-identity case (ref_fit == driver_fit) is a no-op — returns the
    reference's own projected verts at each timestep.

    `driver_start` shifts which driver frame corresponds to gen slot 0; the
    window covered is `[driver_start, driver_start + n_frames)`, clamped at
    the clip's tail.

    Returns:
        verts_ndc : (T, V, 3) pytorch3d NDC relative to crop_box
        offsets   : (T, V, 3) per-vertex expression deformation
    """
    n_drv = driver_fit["expr"].shape[0]
    verts_list, offsets_list = [], []
    for t in range(n_frames):
        t_d = min(driver_start + t, n_drv - 1)
        fi = {
            "shape":   ref_fit["shape"],
            "expr":    driver_fit["expr"][[t_d]],
            "rot":     driver_fit["rot"][[t_d]],
            "tra":     driver_fit["tra"][[t_d]],
            "eye_rot": driver_fit["eye_rot"][[t_d]],
            "fx":      ref_fit["fx"][[0]], "fy": ref_fit["fy"][[0]],
            "cx":      ref_fit["cx"][[0]], "cy": ref_fit["cy"][[0]],
            "extr":    ref_fit["extr"][[0]],
        }
        if "jaw_rot" in driver_fit:
            fi["jaw_rot"] = driver_fit["jaw_rot"][[t_d]]
        fo = compute_flame(flame_skinner, fi)
        v = verts_to_pytorch3d(fo["verts_2d"][0, 0].copy(), np.array(crop_box))
        verts_list.append(v.astype(np.float32))
        offsets_list.append(fo["offsets_3d"][0].astype(np.float32))

    return (
        np.stack(verts_list,   axis=0),
        np.stack(offsets_list, axis=0),
    )


def prepare_driver_frames(
    driver_fit: dict,
    video_path: Path,
    n_frames: int,
    resolution: int,
    flame_skinner: CAP4DFlameSkinner,
    head_vert_ids: np.ndarray,
    driver_start: int = 0,
) -> np.ndarray:
    """Load the driver's own face-cropped video frames for visualization.

    Each frame is cropped by the DRIVER's FLAME (so the face stays centered in
    the driver's own pixel space), then resized to `resolution`. Intended for
    side-by-side "this is what the driver looked like" rows — not consumed by
    the diffusion model itself.

    `driver_start` shifts the window covered to `[driver_start,
    driver_start + n_frames)`; both fit indexing and `load_frame(video, t_d)`
    follow the same offset so the fit and the pixel frames stay in sync.

    Returns: (T, resolution, resolution, 3) uint8 RGB.
    """
    n_drv = driver_fit["expr"].shape[0]
    frames = []
    for t in range(n_frames):
        t_d = min(driver_start + t, n_drv - 1)
        fi = {
            "shape":   driver_fit["shape"],
            "expr":    driver_fit["expr"][[t_d]],
            "rot":     driver_fit["rot"][[t_d]],
            "tra":     driver_fit["tra"][[t_d]],
            "eye_rot": driver_fit["eye_rot"][[t_d]],
            "fx":      driver_fit["fx"][[0]], "fy": driver_fit["fy"][[0]],
            "cx":      driver_fit["cx"][[0]], "cy": driver_fit["cy"][[0]],
            "extr":    driver_fit["extr"][[0]],
        }
        if "jaw_rot" in driver_fit:
            fi["jaw_rot"] = driver_fit["jaw_rot"][[t_d]]
        fo = compute_flame(flame_skinner, fi)
        verts_2d = fo["verts_2d"][0, 0]
        crop_box = get_bbox_from_verts(verts_2d.copy(), head_vert_ids)

        img = load_frame(video_path, t_d)
        img = crop_image(img, crop_box, bg_value=255)
        img = rescale_image(img, resolution)
        frames.append(img.astype(np.uint8))

    return np.stack(frames, axis=0)
