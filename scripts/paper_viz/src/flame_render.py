"""Shared helpers for FLAME-based paper figures.

Used by `flame_substitution_diagram.py` and `generate_driver_map_example.py`.
Anything that touches FLAME tracking, mesh rasterization, or the
`SpatialConditioning` driver-map outputs lives here so both figures stay
visually consistent (same wireframe color, same mesh shading, same
Δ_expr colormap).

Not a standalone script — invoked via:
    from src.flame_render import ...
inside the figure-rendering scripts in `scripts/paper_viz/`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from matplotlib.collections import LineCollection

from loki.conditioning.conditioning import SpatialConditioning
from loki.conditioning.mesh2img import PropRenderer


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

HEAD_VERT_PATH      = "data/assets/flame/head_vertices.txt"
DEFAULT_FLAME_ROOT  = "data/flame_tracking/flowface"
DEFAULT_VIDEO_ROOT  = "data/talkvid/talkvid"


# ---------------------------------------------------------------------------
# Visual constants — keep figures consistent across the paper.
# ---------------------------------------------------------------------------

# Wireframe overlay (very thin: 5K faces × 3 edges crowds fast).
WIREFRAME_COLOR = (0.40, 0.75, 0.95)
WIREFRAME_ALPHA = 0.35
WIREFRAME_LW    = 0.18

# Phong-shaded mesh tone — diffuse + ambient peaks ≈ 0.85, well below the
# 0.94 background, so highlights don't blend out.
MESH_BG_RGB     = (0.94, 0.94, 0.94)
MESH_DIFFUSE    = (0.55, 0.55, 0.60)
MESH_AMBIENT    = (0.30, 0.30, 0.34)
LIGHT_DIRECTION = (0.0, 0.5, -1.0)        # front-up in pytorch3d cam space

# Δ_expr false color: per-pixel ||Δ||₂ through Spectral_r diverging cmap,
# white off-mesh background.
DEFORM_CMAP_NAME = "Spectral_r"
DEFORM_BG_RGB    = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Clip discovery + FLAME-fit IO
# ---------------------------------------------------------------------------

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


def flame_inputs(fit: dict, t: int) -> dict:
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


def motion_score(fit: dict) -> np.ndarray:
    """Per-frame z-scored rotation + expression magnitude. Used to pick
    expressive driver / reference frames."""
    rot_mag  = np.linalg.norm(fit["rot"],  axis=-1)
    expr_mag = np.linalg.norm(fit["expr"], axis=-1)
    def _z(x):
        s = x.std() + 1e-8
        return (x - x.mean()) / s
    return _z(rot_mag) + _z(expr_mag)


# ---------------------------------------------------------------------------
# Vertex projection helpers
# ---------------------------------------------------------------------------

def verts_pixels(verts_2d_orig: np.ndarray, crop_box: np.ndarray, image_size: int):
    """Map original-frame pixel (x, y) → cropped-and-resized pixel (x', y')."""
    x0, y0, x1, y1 = crop_box
    sx = image_size / float(x1 - x0)
    sy = image_size / float(y1 - y0)
    out = verts_2d_orig.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - x0) * sx
    out[:, 1] = (out[:, 1] - y0) * sy
    return out[:, :2]


def ndc_verts(verts_2d_orig: np.ndarray, crop_box: np.ndarray) -> np.ndarray:
    """Map original-frame (x, y) verts → pytorch3d NDC under `crop_box`."""
    from loki.utils import verts_to_pytorch3d
    return verts_to_pytorch3d(verts_2d_orig.copy(), np.array(crop_box)).astype(np.float32)


# ---------------------------------------------------------------------------
# Phong-shaded mesh
# ---------------------------------------------------------------------------

def build_renderer(device: torch.device) -> PropRenderer:
    """PropRenderer rasterizes vertex normals as a 3-channel property; we
    Phong-shade in pixel space rather than via pytorch3d's lighting so we
    sidestep any OpenCV↔pytorch3d camera-convention mismatch — the
    rasterizer is the same code path SpatialConditioning uses."""
    return PropRenderer().to(device).eval()


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-vertex normals via area-weighted face normals. `verts` (V, 3),
    `faces` (F, 3). Returns (V, 3) unit-length on the device of `verts`."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return vn / (vn.norm(dim=-1, keepdim=True) + 1e-8)


def render_shaded_mesh(
    verts_ndc:   np.ndarray,        # (V, 3) ref-camera NDC
    verts_3d_cv: np.ndarray,        # (V, 3) OpenCV-cam coords (for normals)
    faces:       torch.Tensor,      # (F, 3) long
    image_size:  int,
    device:      torch.device,
    renderer:    PropRenderer,
) -> np.ndarray:
    """Render the FLAME mesh on a neutral-gray background using a single
    directional light. Returns `(image_size, image_size, 3)` uint8.

    Approach: rasterize vertex *normals* (computed in cam space) as a
    3-channel property → per-pixel normal map → Phong shade in pixel space
    → compose with bg via the on-mesh mask.
    """
    verts_t   = torch.from_numpy(verts_ndc).float().to(device).unsqueeze(0)
    verts_cv  = torch.from_numpy(verts_3d_cv).float().to(device)
    faces_t   = faces.to(device)
    v_normals = vertex_normals(verts_cv, faces_t).unsqueeze(0)

    pose_map, mask = renderer.render(verts_t, (image_size, image_size), prop=v_normals)
    # pose_map is (1, H, W, 6) = rasterized verts (3) + rasterized normals (3).
    normals = pose_map[0, ..., 3:6]
    mask    = mask[0, ..., 0] > 0

    n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    light = torch.tensor(LIGHT_DIRECTION, device=device, dtype=n.dtype)
    light = -light / light.norm()                       # direction *to* light
    diff = (n * light).sum(dim=-1).clamp(min=0.0)

    ambient = torch.tensor(MESH_AMBIENT, device=device, dtype=n.dtype).view(1, 1, 3)
    diffuse = torch.tensor(MESH_DIFFUSE, device=device, dtype=n.dtype).view(1, 1, 3)
    shaded  = (ambient + diffuse * diff.unsqueeze(-1)).clamp(0.0, 1.0)

    bg = torch.tensor(MESH_BG_RGB, device=device, dtype=n.dtype).view(1, 1, 3)
    out = torch.where(mask.unsqueeze(-1), shaded, bg)
    return (out.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Wireframe overlay
# ---------------------------------------------------------------------------

def draw_wireframe_overlay(
    ax, rgb: np.ndarray, verts_pixels_xy: np.ndarray, faces_np: np.ndarray,
) -> None:
    """Show `rgb` and overlay FLAME triangle edges as a translucent
    LineCollection. `verts_pixels_xy` is `(V, 2)` in image pixels;
    `faces_np` is `(F, 3)` int."""
    ax.imshow(rgb)
    e = np.concatenate([
        np.stack([verts_pixels_xy[faces_np[:, 0]], verts_pixels_xy[faces_np[:, 1]]], axis=1),
        np.stack([verts_pixels_xy[faces_np[:, 1]], verts_pixels_xy[faces_np[:, 2]]], axis=1),
        np.stack([verts_pixels_xy[faces_np[:, 2]], verts_pixels_xy[faces_np[:, 0]]], axis=1),
    ], axis=0)
    lc = LineCollection(
        e, colors=[(*WIREFRAME_COLOR, WIREFRAME_ALPHA)] * len(e),
        linewidths=WIREFRAME_LW,
    )
    ax.add_collection(lc)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)


# ---------------------------------------------------------------------------
# SpatialConditioning rasterization
# ---------------------------------------------------------------------------

def rasterize_conditioning(
    cond_module: SpatialConditioning,
    verts_ndc:   np.ndarray,
    offsets:     np.ndarray,
    device:      torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one SpatialConditioning pass over a single timestep. Works for
    both same-identity (driver verts under driver camera) and substituted
    (β_ref + driver motion under ref camera) — the call site just passes
    whichever (verts_ndc, offsets) it wants rasterized.

    Returns (pos_enc_42ch, deform_3ch, mask_HxW), all numpy.
    """
    v = torch.from_numpy(verts_ndc).float().to(device).unsqueeze(0).unsqueeze(0)
    o = torch.from_numpy(offsets  ).float().to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = cond_module({"driver_verts": v, "driver_deform": o})
    spatial = out["spatial_cond"][0, 0].cpu().numpy()                # (H, W, 45)
    pos_enc = spatial[..., :42]
    deform  = spatial[..., 42:45]
    # On-mesh mask: pos_enc was multiplied by the mask in the module, so any
    # nonzero pos_enc location is on-mesh — cheaper than re-running the raster.
    mask = (pos_enc != 0).any(axis=-1)
    return pos_enc, deform, mask


# ---------------------------------------------------------------------------
# Channel visualizations
# ---------------------------------------------------------------------------

def deform_to_magnitude_rgb(
    deform: np.ndarray, mask: np.ndarray, vmax: float, cmap,
) -> np.ndarray:
    """Map ||Δ||₂ at each pixel through `cmap`, normalized to [0, vmax].
    Off-mesh pixels are filled with `DEFORM_BG_RGB` (white)."""
    mag = np.linalg.norm(deform, axis=-1)
    norm = np.clip(mag / max(vmax, 1e-8), 0.0, 1.0)
    rgb = cmap(norm)[..., :3].astype(np.float32)
    rgb[~mask] = np.array(DEFORM_BG_RGB, dtype=np.float32)
    return rgb


def pos_enc_channel_to_rgb(
    pos_enc: np.ndarray, channel: int, mask: np.ndarray,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Single pos-enc channel as grayscale RGB; values are sin/cos in
    [-1, 1] mapped to [0, 1]; off-mesh = `bg` (white by default)."""
    ch = pos_enc[..., channel]
    g = ((ch + 1.0) / 2.0).astype(np.float32)
    g[~mask] = bg[0]
    return np.stack([g, g, g], axis=-1)


def compute_vmax(deform_maps: list[np.ndarray], q: float = 99.5) -> float:
    """`q`-th percentile of per-pixel ||Δ||₂ across the supplied maps."""
    mags = [np.linalg.norm(d, axis=-1).reshape(-1) for d in deform_maps]
    pooled = np.concatenate(mags)
    return float(np.percentile(pooled, q) + 1e-8)


# ---------------------------------------------------------------------------
# Generic matplotlib helper
# ---------------------------------------------------------------------------

def hide_axis(ax) -> None:
    """Hide ticks AND spines. Used by every figure that places imagery
    flush against neighbors."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
