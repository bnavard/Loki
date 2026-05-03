r"""FLAME parametric-substitution method-diagram figure.

Single horizontal figure illustrating the cross-identity conditioning
pipeline: take the reference's FLAME shape (β_ref) and camera, combine
with the driver's expression and pose (ψ_drv, θ_drv) at one timestep,
and read out the resulting rasterized retargeted Δ_expr that conditions
the diffusion UNet.

Layout (two horizontal strips stacked, one per ref/driver pair). Each
strip is a single row of 8 imagery panels:

    [ref] [ref+wf] [ref mesh] [Δ_ref] [drv] [drv+wf] [drv mesh] →arrow→ [Δ_retarget]

Reading left to right:
  0  reference RGB
  1  reference RGB + FLAME wireframe overlay (shows the fit's quality)
  2  shaded V(β_ref, ψ_ref, θ_ref) mesh in the *reference's own camera*
  3  Δ_expr rasterized from the reference frame's own params
  4  driver RGB
  5  driver RGB + FLAME wireframe overlay
  6  shaded V(β_drv, ψ_drv, θ_drv) mesh in the *driver's own camera*
  ↗  arrow with caption "β_ref + (ψ_drv, θ_drv)" (parametric substitution)
  7  retargeted Δ_expr from V(β_ref, ψ_drv, θ_drv, cam_ref)

All four Δ_expr panels share one signed diverging colormap + colorbar.

Usage (from repo root, marionette conda env):

    # Random pair selection (driver frames score-ranked by motion magnitude).
    PYTHONPATH=. python paper/flame_substitution_diagram.py --seed 7

    # Pin both pairs explicitly.
    PYTHONPATH=. python paper/flame_substitution_diagram.py \
        --pair_a_ref <ref_clip_id> --pair_a_driver <drv_clip_id> \
        --pair_b_ref <ref_clip_id> --pair_b_driver <drv_clip_id>

    # Render N variants for browsing — each invocation lands in its own
    # timestamped run subdir under outputs/paper_figures/flame_substitution/.
    PYTHONPATH=. python paper/flame_substitution_diagram.py --n_figures 5 --seed 7
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import re

from marionette.conditioning.conditioning import SpatialConditioning
from marionette.conditioning.mesh2img import PropRenderer
from marionette.flame.flame import CAP4DFlameSkinner, compute_flame
from marionette.retargeting import prepare_reference
from marionette.utils import (
    crop_image, get_bbox_from_verts, load_frame,
    rescale_image, verts_to_pytorch3d,
)


# ---------------------------------------------------------------------------
# Inlined helpers (previously imported from `paper/render_retargeting_figure.py`,
# which has been removed; keeping these local makes the script self-contained).
# ---------------------------------------------------------------------------

HEAD_VERT_PATH      = "data/assets/flame/head_vertices.txt"
DEFAULT_FLAME_ROOT  = "data/flame_tracking/flowface"
DEFAULT_VIDEO_ROOT  = "data/talkvid/talkvid"


def load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


def safe_name(s: str, max_len: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)[:max_len]


def discover_clips(flame_root: Path, video_root: Path) -> list[str]:
    """Return clip IDs with both a fit.npz under `flame_root` and an mp4
    under `video_root`. Sorted for stable iteration."""
    ids = []
    for child in flame_root.iterdir():
        if not child.is_dir():
            continue
        if not (child / "fit.npz").is_file():
            continue
        if not (video_root / f"{child.name}.mp4").is_file():
            continue
        ids.append(child.name)
    ids.sort()
    return ids


def _flame_inputs(fit: dict, t: int) -> dict:
    """Per-timestep FLAME input dict in the form `compute_flame` expects."""
    fi = {
        "shape":   fit["shape"],
        "expr":    fit["expr"][[t]],
        "rot":     fit["rot"][[t]],
        "tra":     fit["tra"][[t]],
        "eye_rot": fit["eye_rot"][[t]],
        "fx":      fit["fx"][[0]],
        "fy":      fit["fy"][[0]],
        "cx":      fit["cx"][[0]],
        "cy":      fit["cy"][[0]],
        "extr":    fit["extr"][[0]],
    }
    if "jaw_rot" in fit:
        fi["jaw_rot"] = fit["jaw_rot"][[t]]
    return fi


def load_driver_frame(
    driver_fit: dict, t: int, video_path: Path, resolution: int,
    flame_skinner: CAP4DFlameSkinner, head_vert_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + crop one driver RGB frame at time `t` under the driver's own
    FLAME geometry; return cropped image plus per-frame NDC verts and
    per-vertex deformation offsets."""
    fi = _flame_inputs(driver_fit, t)
    fo = compute_flame(flame_skinner, fi)
    verts_2d = fo["verts_2d"][0, 0]
    offsets  = fo["offsets_3d"][0]
    crop_box = get_bbox_from_verts(verts_2d.copy(), head_vert_ids)

    img = load_frame(video_path, t)
    img = crop_image(img, crop_box, bg_value=255)
    img = rescale_image(img, resolution).astype(np.uint8)

    verts_ndc = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))
    return img, verts_ndc.astype(np.float32), offsets.astype(np.float32)


def retargeted_verts_offsets(
    ref_fit: dict, driver_fit: dict, t: int,
    ref_crop_box: np.ndarray, flame_skinner: CAP4DFlameSkinner,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FLAME with REF identity (β_ref, camera_ref) under the DRIVER's
    motion (ψ, θ) at time `t`; return per-frame NDC verts (in REF crop
    space) + per-vertex deformation offsets."""
    fi = _flame_inputs(driver_fit, t)
    fi["shape"] = ref_fit["shape"]
    for k in ("fx", "fy", "cx", "cy", "extr"):
        fi[k] = ref_fit[k][[0]]

    fo = compute_flame(flame_skinner, fi)
    verts_2d  = fo["verts_2d"][0, 0]
    offsets   = fo["offsets_3d"][0]
    verts_ndc = verts_to_pytorch3d(verts_2d.copy(), np.array(ref_crop_box))
    return verts_ndc.astype(np.float32), offsets.astype(np.float32)


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Wireframe overlay (cells (a-bottom), (b-bottom)).
WIREFRAME_COLOR = (0.40, 0.75, 0.95)   # light blue
WIREFRAME_ALPHA = 0.35
WIREFRAME_LW    = 0.18                 # very thin — 5K faces × 3 edges crowds fast

# Shaded-mesh render (cell (c-top)).
# Mesh tone is chosen to read clearly against the light-gray background:
# diffuse + ambient summed at peak ≈ 0.85, well below bg=0.94, so highlights
# don't blend out.
MESH_BG_RGB    = (0.94, 0.94, 0.94)     # slightly lighter than #E0E0E0 for stronger contrast
MESH_DIFFUSE   = (0.55, 0.55, 0.60)
MESH_AMBIENT   = (0.30, 0.30, 0.34)
MESH_SPECULAR  = (0.10, 0.10, 0.10)
LIGHT_DIRECTION = (0.0, 0.5, -1.0)      # front-up in pytorch3d cam space

# Δ_expr false color. We display the per-vertex magnitude ||Δ||₂ with a
# diverging colormap (smoother gradient than the perceptually-uniform
# magma); off-mesh background is white so the head silhouette pops.
DEFORM_CMAP_NAME = "Spectral_r"
DEFORM_BG_RGB    = (1.0, 1.0, 1.0)

# Arrow annotation between cols (b) and (c).
ARROW_COLOR  = "#222222"
ARROW_LABEL  = (r"$\beta_{\mathrm{ref}}\, +$" "\n"
                r"$(\psi_{\mathrm{drv}},\, \theta_{\mathrm{drv}})$")
ARROW_HINT   = "parametric\nsubstitution"
ARROW_LW     = 5.0          # line thickness; "width" of the arrow itself

# Font sizes — sized so labels stay legible at typical NeurIPS column
# widths (figure embeds at ~half-page). Long captions are split onto two
# lines via `\n` in COL_LABELS so they don't run wider than the panels.
CAPTION_FONTSIZE       = 22
ARROW_LABEL_FONTSIZE   = 20
ARROW_HINT_FONTSIZE    = 17
COLORBAR_TICK_FONTSIZE = 18
COLORBAR_TITLE_FONTSIZE = 20

# Column captions printed under the strip — one per GridSpec column slot
# (arrow column has empty label).
COL_LABELS = [
    "Reference",                                                          # 0
    "FLAME mesh\n"                                                        # 1
    r"$V(\beta_{\mathrm{ref}}, \psi_{\mathrm{ref}}, \theta_{\mathrm{ref}})$",
    "Driver",                                                             # 2
    "FLAME mesh\n"                                                        # 3
    r"$V(\beta_{\mathrm{drv}}, \psi_{\mathrm{drv}}, \theta_{\mathrm{drv}})$",
    r"$\Delta_{\mathrm{expr}}$ (driver)",                                 # 4
    "",                                                                    # 5  arrow column
    r"$\Delta_{\mathrm{expr}}$ retargeted" "\n"                           # 6
    r"$V(\beta_{\mathrm{ref}}, \psi_{\mathrm{drv}}, \theta_{\mathrm{drv}}, \mathrm{cam}_{\mathrm{ref}})$",
]


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

@dataclass
class PairSpec:
    ref_clip:     str
    ref_frame:    int
    driver_clip:  str
    driver_frame: int


def _motion_score(fit: dict) -> np.ndarray:
    """Per-frame heuristic 'how much motion' — used to pick driver frames
    that are pose-turned and expressive. Returns array of length T."""
    rot_mag  = np.linalg.norm(fit["rot"],  axis=-1)            # (T,)  axis-angle norm
    expr_mag = np.linalg.norm(fit["expr"], axis=-1)            # (T,)
    # Z-score within the clip + sum, so we don't rank clips by overall scale.
    def _z(x):
        s = x.std() + 1e-8
        return (x - x.mean()) / s
    return _z(rot_mag) + _z(expr_mag)


def _best_driver_frame(fit: dict) -> int:
    """Pick the highest-motion frame within the middle 80% of the clip."""
    n = fit["expr"].shape[0]
    lo, hi = max(1, int(n * 0.10)), max(2, int(n * 0.90))
    score = _motion_score(fit)
    return int(lo + np.argmax(score[lo:hi]))


def _expressive_ref_frame(fit: dict, rng: np.random.Generator) -> int:
    """Sample from the top quartile (by motion score) of the middle 80% of
    the clip. Bias toward expressive ref frames so Δ_ref is non-trivial and
    the spatial relocation of the deformation under retargeting is visible
    side-by-side with Δ_retargeted; some randomness retained so multiple
    runs don't collapse to the same ref pick."""
    n = fit["expr"].shape[0]
    lo, hi = max(0, int(n * 0.10)), max(1, int(n * 0.90))
    if hi <= lo:
        return 0
    score = _motion_score(fit)[lo:hi]
    cutoff = np.percentile(score, 75)
    eligible = np.where(score >= cutoff)[0]
    return int(lo + rng.choice(eligible))


def sample_pair(
    rng: np.random.Generator,
    clip_pool: list[str],
    flame_root: Path,
) -> PairSpec:
    ref, drv = rng.choice(clip_pool, size=2, replace=False)
    ref, drv = str(ref), str(drv)
    ref_fit = load_fit(flame_root / ref / "fit.npz")
    drv_fit = load_fit(flame_root / drv / "fit.npz")
    return PairSpec(
        ref_clip     = ref,
        ref_frame    = _expressive_ref_frame(ref_fit, rng),
        driver_clip  = drv,
        driver_frame = _best_driver_frame(drv_fit),
    )


# ---------------------------------------------------------------------------
# Substituted mesh — shaded render (column (c) top)
# ---------------------------------------------------------------------------

def _build_phong_renderer(image_size: int, device: torch.device) -> PropRenderer:
    """Reuse the repo's PropRenderer for the underlying rasterization. We
    rasterize per-vertex world-space normals into the image, then apply a
    single-directional-light Phong shade in pixel space — sidesteps any
    OpenCV↔pytorch3d camera-convention mismatch since the rasterizer is
    the same code path SpatialConditioning uses."""
    return PropRenderer().to(device).eval()


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-vertex normals via area-weighted face normals. `verts` (V, 3),
    `faces` (F, 3). Returns (V, 3) unit-length on the device of `verts`."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)            # (F, 3) area-weighted
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    vn = vn / (vn.norm(dim=-1, keepdim=True) + 1e-8)
    return vn


def render_shaded_mesh(
    verts_ndc:   np.ndarray,        # (V, 3) substituted, ref-camera NDC
    verts_3d_cv: np.ndarray,        # (V, 3) substituted, OpenCV-cam coords (for normals)
    faces:       torch.Tensor,      # (F, 3) long
    image_size:  int,
    device:      torch.device,
    renderer:    PropRenderer,
) -> np.ndarray:
    """Render the substituted FLAME mesh on a neutral-gray background using
    a single directional light, returning `(image_size, image_size, 3)` uint8.

    Approach:
      1. PropRenderer rasterizes vertex *normals* (computed in cam space)
         as a 3-channel property → per-pixel normal map.
      2. Phong shading in pixel space: ambient + diffuse·max(0, n · l).
      3. Compose with the bg color via the on-mesh mask."""
    verts_t = torch.from_numpy(verts_ndc).float().to(device).unsqueeze(0)   # (1, V, 3)
    verts_cv_t = torch.from_numpy(verts_3d_cv).float().to(device)
    faces_t = faces.to(device)
    v_normals = _vertex_normals(verts_cv_t, faces_t).unsqueeze(0)           # (1, V, 3)

    pose_map, mask = renderer.render(
        verts_t, (image_size, image_size), prop=v_normals,
    )
    # pose_map is (1, H, W, 6) = rasterized verts (3) + rasterized normals (3).
    normals = pose_map[0, ..., 3:6]                                          # (H, W, 3)
    mask    = mask[0, ..., 0] > 0                                            # (H, W)

    n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    light = torch.tensor(LIGHT_DIRECTION, device=device, dtype=n.dtype)
    light = -light / light.norm()                                            # direction *to* light
    diff = (n * light).sum(dim=-1).clamp(min=0.0)                            # (H, W)

    ambient = torch.tensor(MESH_AMBIENT, device=device, dtype=n.dtype).view(1, 1, 3)
    diffuse = torch.tensor(MESH_DIFFUSE, device=device, dtype=n.dtype).view(1, 1, 3)
    shaded  = ambient + diffuse * diff.unsqueeze(-1)
    shaded  = shaded.clamp(0.0, 1.0)

    bg = torch.tensor(MESH_BG_RGB, device=device, dtype=n.dtype).view(1, 1, 3)
    out = torch.where(mask.unsqueeze(-1), shaded, bg)
    return (out.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Wireframe overlay (cells (a-bottom), (b-bottom))
# ---------------------------------------------------------------------------

def _verts_pixels(verts_2d_orig: np.ndarray, crop_box: np.ndarray, image_size: int):
    """Map original-frame pixel (x, y) → cropped-and-resized pixel (x', y')."""
    x0, y0, x1, y1 = crop_box
    sx = image_size / float(x1 - x0)
    sy = image_size / float(y1 - y0)
    out = verts_2d_orig.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - x0) * sx
    out[:, 1] = (out[:, 1] - y0) * sy
    return out[:, :2]


def draw_wireframe_overlay(
    ax, rgb: np.ndarray, verts_pixels: np.ndarray, faces_np: np.ndarray,
) -> None:
    """Show `rgb` and overlay FLAME triangle edges as a translucent
    LineCollection. `verts_pixels` is `(V, 2)` in image pixels; `faces_np`
    is `(F, 3)` int."""
    ax.imshow(rgb)
    # Build an (E, 2, 2) array of segment endpoints for matplotlib LineCollection.
    # Each triangle yields 3 edges; many shared, but matplotlib draws the union
    # cheaply as a single artist.
    e = np.concatenate([
        np.stack([verts_pixels[faces_np[:, 0]], verts_pixels[faces_np[:, 1]]], axis=1),
        np.stack([verts_pixels[faces_np[:, 1]], verts_pixels[faces_np[:, 2]]], axis=1),
        np.stack([verts_pixels[faces_np[:, 2]], verts_pixels[faces_np[:, 0]]], axis=1),
    ], axis=0)
    lc = LineCollection(
        e, colors=[(*WIREFRAME_COLOR, WIREFRAME_ALPHA)] * len(e),
        linewidths=WIREFRAME_LW,
    )
    ax.add_collection(lc)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)


# ---------------------------------------------------------------------------
# Δ_expr false color (cells (c-bottom), (d-bottom))
# ---------------------------------------------------------------------------

def deform_to_magnitude_rgb(
    deform: np.ndarray, mask: np.ndarray, vmax: float, cmap,
) -> np.ndarray:
    """Map ||Δ||₂ at each pixel through `cmap`, normalized to [0, vmax].
    Off-mesh pixels are filled with `DEFORM_BG_RGB` (white)."""
    mag = np.linalg.norm(deform, axis=-1)                        # (H, W)
    norm = np.clip(mag / max(vmax, 1e-8), 0.0, 1.0)
    rgb = cmap(norm)[..., :3].astype(np.float32)                 # drop alpha
    rgb[~mask] = np.array(DEFORM_BG_RGB, dtype=np.float32)
    return rgb


def _compute_vmax(deform_maps: list[np.ndarray], q: float = 99.5) -> float:
    """99.5th percentile of per-pixel ||Δ||₂ across the supplied maps."""
    mags = [np.linalg.norm(d, axis=-1).reshape(-1) for d in deform_maps]
    pooled = np.concatenate(mags)
    return float(np.percentile(pooled, q) + 1e-8)


# ---------------------------------------------------------------------------
# Substituted-side raster: pos_enc + deform + mask
# ---------------------------------------------------------------------------

def rasterize_substituted(
    cond_module:   SpatialConditioning,
    verts_ndc:     np.ndarray,                   # (V, 3) substituted, ref-camera NDC
    offsets:       np.ndarray,                   # (V, 3) substituted, per-vertex Δ
    device:        torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one SpatialConditioning pass over a single substituted timestep.
    Returns (pos_enc_42ch, deform_3ch, mask_HxW), all numpy."""
    v = torch.from_numpy(verts_ndc).float().to(device).unsqueeze(0).unsqueeze(0)
    o = torch.from_numpy(offsets  ).float().to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = cond_module({"driver_verts": v, "driver_deform": o})
    spatial = out["spatial_cond"][0, 0].cpu().numpy()                # (H, W, 45)
    pos_enc = spatial[..., :42]
    deform  = spatial[..., 42:45]
    # On-mesh mask: pos_enc was multiplied by the mask in the module, so any
    # nonzero pos_enc location is on-mesh. Cheaper than re-running the raster.
    mask = (pos_enc != 0).any(axis=-1)
    return pos_enc, deform, mask


def pos_enc_low_freq_grayscale(pos_enc: np.ndarray) -> np.ndarray:
    """Channel 0 of the pos-enc tensor = sin(2^0 · x). Map to grayscale RGB
    in [0, 1] with white background where the head mask is empty."""
    ch = pos_enc[..., 0]                                              # (H, W) in [-1, 1]
    mask = (pos_enc != 0).any(axis=-1)
    g = (ch + 1.0) / 2.0                                              # [0, 1]
    g[~mask] = 1.0                                                    # white bg
    rgb = np.stack([g, g, g], axis=-1)
    return rgb


# ---------------------------------------------------------------------------
# Per-pair gather
# ---------------------------------------------------------------------------

@dataclass
class PairBundle:
    """One pair's imagery for the single-row figure:
        ref RGB | ref mesh | drv RGB | drv mesh | Δ_drv | →arrow→ | Δ_retarget
    (wireframe and Δ_ref panels were removed in earlier layout passes.)"""
    pair: PairSpec
    # Reference side (no Δ_ref).
    ref_rgb:         np.ndarray
    ref_mesh_shaded: np.ndarray
    # Driver side — its own deformation map is rasterized in driver-camera NDC.
    drv_rgb:         np.ndarray
    drv_mesh_shaded: np.ndarray
    drv_deform:      np.ndarray            # (H, W, 3) raw
    drv_deform_mask: np.ndarray
    # Retargeted (substituted) Δ_expr.
    retarget_deform:      np.ndarray
    retarget_deform_mask: np.ndarray


def _ndc_verts(verts_2d_orig: np.ndarray, crop_box: np.ndarray) -> np.ndarray:
    """Map original-frame (x, y) verts → pytorch3d NDC under `crop_box`."""
    return verts_to_pytorch3d(verts_2d_orig.copy(), np.array(crop_box)).astype(np.float32)


def gather_pair(
    spec: PairSpec,
    flame_root: Path,
    video_root: Path,
    flame_skinner: CAP4DFlameSkinner,
    head_vert_ids: np.ndarray,
    cond_module: SpatialConditioning,
    phong_renderer: PropRenderer,
    image_size: int,
    device: torch.device,
) -> PairBundle:
    ref_fit = load_fit(flame_root / spec.ref_clip    / "fit.npz")
    drv_fit = load_fit(flame_root / spec.driver_clip / "fit.npz")

    # --- Reference side: RGB, wireframe verts, shaded mesh, Δ_expr ---
    ref_video = video_root / f"{spec.ref_clip}.mp4"
    ref_img_norm, _, ref_crop_box = prepare_reference(
        ref_fit, spec.ref_frame, ref_video,
        image_size, flame_skinner, head_vert_ids,
    )
    ref_rgb = ((ref_img_norm + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    fo_ref = compute_flame(flame_skinner, _flame_inputs(ref_fit, spec.ref_frame))
    ref_verts_2d_orig = fo_ref["verts_2d"][0, 0]
    ref_verts_3d_cv   = fo_ref["verts_3d_cv"][0]
    ref_verts_ndc     = _ndc_verts(ref_verts_2d_orig, ref_crop_box)

    ref_mesh_shaded = render_shaded_mesh(
        ref_verts_ndc, ref_verts_3d_cv, flame_skinner.template_faces,
        image_size, device, phong_renderer,
    )

    # --- Driver side: RGB (own crop), wireframe verts, shaded mesh ---
    drv_video = video_root / f"{spec.driver_clip}.mp4"
    drv_rgb_arr, _, _ = load_driver_frame(
        drv_fit, spec.driver_frame, drv_video,
        image_size, flame_skinner, head_vert_ids,
    )
    fo_drv = compute_flame(flame_skinner, _flame_inputs(drv_fit, spec.driver_frame))
    drv_verts_2d_orig = fo_drv["verts_2d"][0, 0]
    drv_verts_3d_cv   = fo_drv["verts_3d_cv"][0]
    drv_offsets       = fo_drv["offsets_3d"][0]
    drv_crop_box      = get_bbox_from_verts(drv_verts_2d_orig.copy(), head_vert_ids)
    drv_verts_ndc     = _ndc_verts(drv_verts_2d_orig, drv_crop_box)

    drv_mesh_shaded = render_shaded_mesh(
        drv_verts_ndc, drv_verts_3d_cv, flame_skinner.template_faces,
        image_size, device, phong_renderer,
    )
    _, drv_deform, drv_deform_mask = rasterize_substituted(
        cond_module, drv_verts_ndc, drv_offsets, device,
    )

    # --- Retargeted: substitute β_ref + (ψ_drv, θ_drv) under cam_ref ---
    ret_verts, ret_offsets = retargeted_verts_offsets(
        ref_fit, drv_fit, spec.driver_frame, ref_crop_box, flame_skinner,
    )
    _, retarget_deform, retarget_mask = rasterize_substituted(
        cond_module, ret_verts, ret_offsets, device,
    )

    return PairBundle(
        pair                 = spec,
        ref_rgb              = ref_rgb,
        ref_mesh_shaded      = ref_mesh_shaded,
        drv_rgb              = drv_rgb_arr,
        drv_mesh_shaded      = drv_mesh_shaded,
        drv_deform           = drv_deform,
        drv_deform_mask      = drv_deform_mask,
        retarget_deform      = retarget_deform,
        retarget_deform_mask = retarget_mask,
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def _hide(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_arrow_column(fig, ax):
    """Draw a thick rightward arrow with the substitution equation above
    and 'parametric substitution' below."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _hide(ax)
    ax.set_facecolor("none")
    ax.annotate(
        "", xy=(0.97, 0.5), xytext=(0.03, 0.5),
        arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.6",
                        color=ARROW_COLOR, lw=ARROW_LW,
                        shrinkA=0, shrinkB=0),
    )
    ax.text(0.5, 0.62, ARROW_LABEL, ha="center", va="bottom",
            fontsize=ARROW_LABEL_FONTSIZE, color=ARROW_COLOR)
    ax.text(0.5, 0.38, ARROW_HINT,  ha="center", va="top",
            fontsize=ARROW_HINT_FONTSIZE, color=ARROW_COLOR)


def compose_figure(pair: PairBundle, out_path: Path) -> None:
    """Single-row figure for one ref/driver pair:

        [ref] [ref mesh] [drv] [drv mesh] [Δ_drv] →arrow→ [Δ_retarget]

    No text annotations — the user overlays them manually. The two
    Δ_expr panels share an unsigned magnitude scale + colorbar."""
    vmax = _compute_vmax([pair.drv_deform, pair.retarget_deform], q=99.5)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap(DEFORM_CMAP_NAME)

    # GridSpec column slots:
    #   0 ref RGB | 1 ref mesh | 2 drv RGB | 3 drv mesh | 4 Δ_drv
    #   5 arrow   | 6 Δ retargeted
    # Plus one short caption row at the bottom.
    n_cols = 7
    # Arrow col widened so the equation + "parametric substitution" hint
    # don't overflow at the larger fontsizes.
    width_ratios  = [2.0, 2.0, 2.0, 2.0, 2.0, 1.8, 2.0]
    # Caption row tall enough for two-line labels at CAPTION_FONTSIZE.
    height_ratios = [2.0, 0.85]
    fig_w = sum(width_ratios) * 1.55
    fig_h = sum(height_ratios) * 1.85
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        2, n_cols, figure=fig,
        width_ratios=width_ratios, height_ratios=height_ratios,
        wspace=0.04, hspace=0.06,
        left=0.012, right=0.93, top=0.985, bottom=0.03,
    )

    # (0) ref RGB
    ax = fig.add_subplot(gs[0, 0]); _hide(ax); ax.imshow(pair.ref_rgb)
    # (1) ref shaded mesh
    ax = fig.add_subplot(gs[0, 1]); _hide(ax); ax.imshow(pair.ref_mesh_shaded)
    # (2) driver RGB
    ax = fig.add_subplot(gs[0, 2]); _hide(ax); ax.imshow(pair.drv_rgb)
    # (3) driver shaded mesh
    ax = fig.add_subplot(gs[0, 3]); _hide(ax); ax.imshow(pair.drv_mesh_shaded)
    # (4) driver Δ_expr magnitude
    ax = fig.add_subplot(gs[0, 4]); _hide(ax)
    ax.imshow(deform_to_magnitude_rgb(pair.drv_deform, pair.drv_deform_mask,
                                      vmax, cmap))
    # (5) substitution arrow (no text; user annotates manually)
    ax_arrow = fig.add_subplot(gs[0, 5]); _draw_arrow_column(fig, ax_arrow)
    # (6) retargeted Δ_expr magnitude
    ax = fig.add_subplot(gs[0, 6]); _hide(ax)
    ax.imshow(deform_to_magnitude_rgb(pair.retarget_deform, pair.retarget_deform_mask,
                                      vmax, cmap))

    # --- column captions row ---
    for c, label in enumerate(COL_LABELS):
        if not label:
            continue
        ax = fig.add_subplot(gs[1, c]); _hide(ax)
        ax.set_facecolor("none")
        ax.text(0.5, 0.85, label, ha="center", va="top",
                fontsize=CAPTION_FONTSIZE, linespacing=1.15)

    # --- shared colorbar — same colormap as the panels, range [0, vmax] ---
    cax = fig.add_axes([0.938, 0.32, 0.014, 0.60])
    sm  = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, vmax / 2, vmax])
    cbar.set_ticklabels([f"{0:.2g}", f"{vmax / 2:.2g}", f"{vmax:.2g}"])
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# Cache the FLAME face indices once per process (built lazily).
_FACES_NP_CACHE: list[np.ndarray] = []
def _faces_np_cache() -> np.ndarray:
    return _FACES_NP_CACHE[0] if _FACES_NP_CACHE else None


def _set_faces(faces: torch.Tensor) -> None:
    _FACES_NP_CACHE.clear()
    _FACES_NP_CACHE.append(faces.cpu().numpy())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render the FLAME parametric-substitution method diagram "
                    "(two stacked ref/driver pairs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_figures",   type=int, default=1,
                   help="Number of figures to render in random mode (default 1).")
    p.add_argument("--seed",        type=int, default=None,
                   help="Optional seed for reproducible draws. Default: time-based.")
    p.add_argument("--ref_clip",     default=None,
                   help="Pin the reference clip ID. Pair with --driver_clip "
                        "for explicit mode (renders one figure, ignores --n_figures).")
    p.add_argument("--ref_frame",    type=int, default=None)
    p.add_argument("--driver_clip",  default=None,
                   help="Pin the driver clip ID.")
    p.add_argument("--driver_frame", type=int, default=None)
    p.add_argument("--flame_root", default=DEFAULT_FLAME_ROOT)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--out_root",   type=Path,
                   default=Path("outputs/paper_figures/flame_substitution"))
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def _explicit_pair_or_none(args) -> Optional[PairSpec]:
    if args.ref_clip is None and args.driver_clip is None:
        return None
    if args.ref_clip is None or args.driver_clip is None:
        raise ValueError("Provide BOTH --ref_clip and --driver_clip, or neither.")
    return PairSpec(
        ref_clip     = args.ref_clip,
        ref_frame    = args.ref_frame    or 0,
        driver_clip  = args.driver_clip,
        driver_frame = args.driver_frame or 0,
    )


def _safe_filename(spec: PairSpec, idx: Optional[int], pad: int) -> str:
    stem = f"{safe_name(spec.ref_clip)}__{safe_name(spec.driver_clip)}"
    if idx is None:
        return f"{stem}.png"
    return f"{idx:0{pad}d}_{stem}.png"


def main():
    args = parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    flame_root = Path(args.flame_root)
    video_root = Path(args.video_root)

    # Skinner stays on CPU — `compute_flame` wraps inputs with `torch.tensor(...)`
    # without moving them, so a GPU-resident skinner would mismatch device.
    flame_skinner = CAP4DFlameSkinner(
        add_mouth=True, n_shape_params=150, n_expr_params=65,
    ).eval()
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)
    _set_faces(flame_skinner.template_faces)

    cond_module = SpatialConditioning(
        image_size=args.resolution,
        positional_channels=42,
        positional_multiplier=1.0,
    ).to(device).eval()

    phong = _build_phong_renderer(args.resolution, device)

    # --- pair selection: explicit > random ---
    explicit_pair = _explicit_pair_or_none(args)
    explicit = explicit_pair is not None

    seed = args.seed if args.seed is not None else int(time.time())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = args.out_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fig] writing to {run_dir}")
    print(f"[fig] seed={seed}  (re-runnable with --seed {seed})")

    if not explicit:
        clip_pool = discover_clips(flame_root, video_root)
        if len(clip_pool) < 2:
            raise SystemExit(
                f"Need ≥ 2 clips with both fit.npz and .mp4; found {len(clip_pool)}."
            )
        print(f"[fig] random mode: pool of {len(clip_pool)} clips, "
              f"{args.n_figures} figure(s)")

    picks_log: list[dict] = []
    n_failed = 0
    n_total  = 1 if explicit else args.n_figures
    n_pad    = max(2, len(str(n_total)))

    for i in range(n_total):
        if explicit:
            spec = explicit_pair
            idx_token = None
        else:
            rng = np.random.default_rng(seed + i)
            spec = sample_pair(rng, clip_pool, flame_root)
            idx_token = i + 1

        out_path = run_dir / _safe_filename(spec, idx_token, n_pad)
        try:
            bundle = gather_pair(spec, flame_root, video_root, flame_skinner,
                                 head_vert_ids, cond_module, phong,
                                 args.resolution, device)
            compose_figure(bundle, out_path)
            picks_log.append({
                "index":  (i + 1) if not explicit else 1,
                "pair":   spec.__dict__,
                "out_png": str(out_path),
            })
            print(f"  rendered  ref={spec.ref_clip}→drv={spec.driver_clip}  "
                  f"→ {out_path.name}")
        except Exception as e:
            n_failed += 1
            picks_log.append({
                "index":  (i + 1) if not explicit else 1,
                "pair":   spec.__dict__,
                "out_png": None,
                "error":   f"{type(e).__name__}: {e}",
            })
            print(f"[fig] FAILED on ref={spec.ref_clip}→drv={spec.driver_clip}: "
                  f"{type(e).__name__}: {e}")

    n_ok = sum(1 for p in picks_log if p.get("out_png"))
    manifest = {
        "seed":        seed,
        "seed_source": ("user --seed" if args.seed is not None else "wall-clock"),
        "timestamp":   timestamp,
        "mode":        "explicit" if explicit else "random",
        "n_requested": n_total,
        "n_rendered":  n_ok,
        "n_failed":    n_total - n_ok,
        "args": {
            "flame_root": str(flame_root),
            "video_root": str(video_root),
            "resolution": args.resolution,
            "device":     str(device),
        },
        "picks":       picks_log,
    }
    (run_dir / "_index.json").write_text(json.dumps(manifest, indent=2))

    print(f"[fig] done: {n_ok}/{n_total} figures saved to {run_dir}/"
          + (f"  ({n_failed} failed)" if n_failed else ""))


if __name__ == "__main__":
    main()
