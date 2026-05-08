r"""Driver-map example figure (single subject across N expressive frames).

Shows how the diffusion UNet's "driver map" looks across a sequence of
expressive frames from ONE clip (same identity throughout — driver =
reference). Each column is a frame; each row is one visualization layer
of the driver map:

    row 1  Clip RGB
    row 2  FLAME wireframe overlay
    row 3  Phong-shaded FLAME mesh in the clip's own camera
    row 4  Δ_expr  (||Δ||₂ magnitude, Spectral_r, matches
                    paper/flame_substitution_diagram.py)
    row 5  Positional encoding   sin(2^0 · x)   (low frequency)
    row 6  Positional encoding   sin(2^6 · x)   (high frequency)

A shared colorbar on the right shows the Δ_expr magnitude range
(0 → vmax, 99.5th percentile of ||Δ||₂ across all picked frames).

Usage (from repo root, loki conda env):

    # Default: 1 random clip, 4 expressive frames as columns.
    PYTHONPATH=. python paper/generate_driver_map_example.py --seed 7

    # More columns, more variants.
    PYTHONPATH=. python paper/generate_driver_map_example.py \
        --n_frames 6 --n_figures 5 --seed 7

    # Pin a specific clip.
    PYTHONPATH=. python paper/generate_driver_map_example.py \
        --clip <clip_id>

    # Pin specific frame indices within a clip.
    PYTHONPATH=. python paper/generate_driver_map_example.py \
        --clip <clip_id> --frames 12 47 80 120
"""
from __future__ import annotations

import argparse
import json
import re
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

from loki.conditioning.conditioning import SpatialConditioning
from loki.conditioning.mesh2img import PropRenderer
from loki.flame.flame import CAP4DFlameSkinner, compute_flame
from loki.retargeting import prepare_reference
from loki.utils import (
    crop_image, get_bbox_from_verts, load_frame,
    rescale_image, verts_to_pytorch3d,
)


# ---------------------------------------------------------------------------
# Inlined helpers (self-contained — see header of
# `paper/flame_substitution_diagram.py` for the same set; intentionally not
# imported across paper/ files to avoid namespace-package fragility).
# ---------------------------------------------------------------------------

HEAD_VERT_PATH      = "data/assets/flame/head_vertices.txt"
DEFAULT_FLAME_ROOT  = "data/flame_tracking/flowface"
DEFAULT_VIDEO_ROOT  = "data/talkvid/talkvid"


def load_fit(path: Path) -> dict:
    return {k: v for k, v in np.load(str(path)).items()}


def safe_name(s: str, max_len: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)[:max_len]


def discover_clips(flame_root: Path, video_root: Path) -> list[str]:
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


def _motion_score(fit: dict) -> np.ndarray:
    """Per-frame z-scored rotation + expression magnitude. Same scoring
    as the substitution diagram."""
    rot_mag  = np.linalg.norm(fit["rot"],  axis=-1)
    expr_mag = np.linalg.norm(fit["expr"], axis=-1)
    def _z(x):
        s = x.std() + 1e-8
        return (x - x.mean()) / s
    return _z(rot_mag) + _z(expr_mag)


def pick_expressive_frames(
    fit: dict, n: int, rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """Pick `n` frames from the middle 80% of the clip with high motion
    score. Frames are sorted in temporal order so the columns read as a
    progression. We cluster picks across the full eligible range (rather
    than crowding around the global maximum) by:
      1. Splitting the eligible window into `n` equal-sized segments.
      2. Picking the highest-motion frame within each segment.
    This guarantees temporal variety even when the clip has one strongly
    dominant motion peak."""
    n_total = fit["expr"].shape[0]
    lo, hi = max(0, int(n_total * 0.10)), max(1, int(n_total * 0.90))
    if hi - lo < n:
        # Clip too short to space frames out — fall back to evenly spaced.
        return np.linspace(lo, hi - 1, n, dtype=int).tolist()

    score = _motion_score(fit)
    seg_edges = np.linspace(lo, hi, n + 1, dtype=int)
    picks = []
    for i in range(n):
        a, b = seg_edges[i], seg_edges[i + 1]
        seg_score = score[a:b]
        picks.append(int(a + np.argmax(seg_score)))
    return picks


# ---------------------------------------------------------------------------
# Visual constants — match `paper/flame_substitution_diagram.py`.
# ---------------------------------------------------------------------------

# Wireframe overlay
WIREFRAME_COLOR = (0.40, 0.75, 0.95)
WIREFRAME_ALPHA = 0.35
WIREFRAME_LW    = 0.18

# Phong-shaded mesh
MESH_BG_RGB     = (0.94, 0.94, 0.94)
MESH_DIFFUSE    = (0.55, 0.55, 0.60)
MESH_AMBIENT    = (0.30, 0.30, 0.34)
LIGHT_DIRECTION = (0.0, 0.5, -1.0)

# Δ_expr: ||Δ||₂ magnitude with Spectral_r diverging cmap, white off-mesh
# background. Same choices as flame_substitution_diagram.py so figures
# are visually consistent across the paper.
DEFORM_CMAP_NAME = "Spectral_r"
DEFORM_BG_RGB    = (1.0, 1.0, 1.0)

# Pos-enc slices — sin(2^0 · x) and sin(2^6 · x), grayscale, white bg.
POS_ENC_LOW_CH  = 0
POS_ENC_HIGH_CH = 6
POS_ENC_BG_RGB  = (1.0, 1.0, 1.0)

# Font sizes — sized so labels stay legible at NeurIPS column widths
# (figure embeds at ~half-page). Two-word labels are split onto two lines
# so the left margin stays narrow.
ROW_LABEL_FONTSIZE     = 30
COLORBAR_TICK_FONTSIZE = 24

ROW_LABELS = [
    "Clip\nRGB",
    "FLAME\nwireframe",
    "FLAME\nmesh",
    r"$\Delta_{\mathrm{expr}}$",
    r"Pos-enc"   "\n"   r"$\sin(2^0\!\cdot x)$",
    r"Pos-enc"   "\n"   r"$\sin(2^6\!\cdot x)$",
]


# ---------------------------------------------------------------------------
# Phong-shaded mesh (row 3)
# ---------------------------------------------------------------------------

def _build_renderer(device: torch.device) -> PropRenderer:
    return PropRenderer().to(device).eval()


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return vn / (vn.norm(dim=-1, keepdim=True) + 1e-8)


def render_shaded_mesh(
    verts_ndc:   np.ndarray,
    verts_3d_cv: np.ndarray,
    faces:       torch.Tensor,
    image_size:  int,
    device:      torch.device,
    renderer:    PropRenderer,
) -> np.ndarray:
    """Phong shade in pixel space using rasterized per-vertex normals."""
    verts_t   = torch.from_numpy(verts_ndc).float().to(device).unsqueeze(0)
    verts_cv  = torch.from_numpy(verts_3d_cv).float().to(device)
    faces_t   = faces.to(device)
    v_normals = _vertex_normals(verts_cv, faces_t).unsqueeze(0)

    pose_map, mask = renderer.render(verts_t, (image_size, image_size), prop=v_normals)
    normals = pose_map[0, ..., 3:6]
    mask    = mask[0, ..., 0] > 0

    n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    light = torch.tensor(LIGHT_DIRECTION, device=device, dtype=n.dtype)
    light = -light / light.norm()
    diff = (n * light).sum(dim=-1).clamp(min=0.0)

    ambient = torch.tensor(MESH_AMBIENT, device=device, dtype=n.dtype).view(1, 1, 3)
    diffuse = torch.tensor(MESH_DIFFUSE, device=device, dtype=n.dtype).view(1, 1, 3)
    shaded  = (ambient + diffuse * diff.unsqueeze(-1)).clamp(0.0, 1.0)

    bg = torch.tensor(MESH_BG_RGB, device=device, dtype=n.dtype).view(1, 1, 3)
    out = torch.where(mask.unsqueeze(-1), shaded, bg)
    return (out.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Wireframe overlay (row 2)
# ---------------------------------------------------------------------------

def _verts_pixels(verts_2d_orig: np.ndarray, crop_box: np.ndarray, image_size: int):
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
    ax.imshow(rgb)
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


def _ndc_verts(verts_2d_orig: np.ndarray, crop_box: np.ndarray) -> np.ndarray:
    return verts_to_pytorch3d(verts_2d_orig.copy(), np.array(crop_box)).astype(np.float32)


# ---------------------------------------------------------------------------
# SpatialConditioning rasterization (rows 4–6)
# ---------------------------------------------------------------------------

def rasterize_same_identity(
    cond_module: SpatialConditioning,
    verts_ndc:   np.ndarray,
    offsets:     np.ndarray,
    device:      torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single SpatialConditioning pass. Returns (pos_enc, deform, mask)."""
    v = torch.from_numpy(verts_ndc).float().to(device).unsqueeze(0).unsqueeze(0)
    o = torch.from_numpy(offsets  ).float().to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = cond_module({"driver_verts": v, "driver_deform": o})
    spatial = out["spatial_cond"][0, 0].cpu().numpy()
    pos_enc = spatial[..., :42]
    deform  = spatial[..., 42:45]
    mask    = (pos_enc != 0).any(axis=-1)
    return pos_enc, deform, mask


# ---------------------------------------------------------------------------
# Channel visualizations (match flame_substitution_diagram.py)
# ---------------------------------------------------------------------------

def deform_to_magnitude_rgb(
    deform: np.ndarray, mask: np.ndarray, vmax: float, cmap,
) -> np.ndarray:
    """Map ||Δ||₂ at each pixel through `cmap`, normalized to [0, vmax].
    Off-mesh pixels filled with `DEFORM_BG_RGB` (white). Identical to the
    function of the same name in flame_substitution_diagram.py."""
    mag = np.linalg.norm(deform, axis=-1)
    norm = np.clip(mag / max(vmax, 1e-8), 0.0, 1.0)
    rgb = cmap(norm)[..., :3].astype(np.float32)
    rgb[~mask] = np.array(DEFORM_BG_RGB, dtype=np.float32)
    return rgb


def pos_enc_channel_to_rgb(
    pos_enc: np.ndarray, channel: int, mask: np.ndarray,
) -> np.ndarray:
    """Single pos-enc channel as grayscale RGB; values are sin/cos in
    [-1, 1] mapped to [0, 1]; off-mesh = white."""
    ch = pos_enc[..., channel]
    g = ((ch + 1.0) / 2.0).astype(np.float32)
    g[~mask] = POS_ENC_BG_RGB[0]
    return np.stack([g, g, g], axis=-1)


def _compute_vmax(deform_maps: list[np.ndarray], q: float = 99.5) -> float:
    """99.5th percentile of per-pixel ||Δ||₂ across all supplied maps."""
    mags = [np.linalg.norm(d, axis=-1).reshape(-1) for d in deform_maps]
    pooled = np.concatenate(mags)
    return float(np.percentile(pooled, q) + 1e-8)


# ---------------------------------------------------------------------------
# Per-column gather (one frame of one clip)
# ---------------------------------------------------------------------------

@dataclass
class FrameBundle:
    frame:        int
    rgb:          np.ndarray             # (H, W, 3) uint8
    verts_pixels: np.ndarray             # (V, 2)
    mesh_shaded:  np.ndarray             # (H, W, 3) uint8
    deform:       np.ndarray             # (H, W, 3) raw
    pos_enc:      np.ndarray             # (H, W, 42)
    mask:         np.ndarray             # (H, W) bool


def gather_frame(
    fit:           dict,
    frame:         int,
    video:         Path,
    flame_skinner: CAP4DFlameSkinner,
    head_vert_ids: np.ndarray,
    cond_module:   SpatialConditioning,
    renderer:      PropRenderer,
    image_size:    int,
    device:        torch.device,
) -> FrameBundle:
    img_norm, _, crop_box = prepare_reference(
        fit, frame, video, image_size, flame_skinner, head_vert_ids,
    )
    rgb = ((img_norm + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    fo = compute_flame(flame_skinner, _flame_inputs(fit, frame))
    verts_2d_orig = fo["verts_2d"][0, 0]
    verts_3d_cv   = fo["verts_3d_cv"][0]
    offsets       = fo["offsets_3d"][0]
    verts_pixels  = _verts_pixels(verts_2d_orig, crop_box, image_size)
    verts_ndc     = _ndc_verts(verts_2d_orig, crop_box)

    mesh_shaded = render_shaded_mesh(
        verts_ndc, verts_3d_cv, flame_skinner.template_faces,
        image_size, device, renderer,
    )
    pos_enc, deform, mask = rasterize_same_identity(
        cond_module, verts_ndc, offsets, device,
    )

    return FrameBundle(
        frame        = frame,
        rgb          = rgb,
        verts_pixels = verts_pixels,
        mesh_shaded  = mesh_shaded,
        deform       = deform,
        pos_enc      = pos_enc,
        mask         = mask,
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def _hide(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def compose_figure(
    frames: list[FrameBundle], faces_np: np.ndarray, out_path: Path,
) -> None:
    """6 visualization rows × N frame columns + one narrow row-label column
    on the left + colorbar on the right.

    GridSpec column layout:
      0     = row labels (narrow)
      1..N  = frame columns (one per FrameBundle)

    GridSpec row layout (each is one image row):
      0  Clip RGB
      1  FLAME wireframe overlay
      2  Phong mesh
      3  Δ_expr (Spectral_r magnitude, shared vmax across frames)
      4  Pos-enc low freq
      5  Pos-enc high freq"""
    n_frames = len(frames)
    n_rows   = 6
    label_col_ratio = 1.3      # wider — accommodates the bigger two-line labels
    width_ratios  = [label_col_ratio] + [2.0] * n_frames
    height_ratios = [2.0] * n_rows

    vmax = _compute_vmax([fb.deform for fb in frames], q=99.5)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap(DEFORM_CMAP_NAME)

    fig_w = sum(width_ratios)  * 1.55
    fig_h = sum(height_ratios) * 1.55
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_rows, 1 + n_frames, figure=fig,
        width_ratios=width_ratios, height_ratios=height_ratios,
        wspace=0.04, hspace=0.04,
        left=0.005, right=0.93, top=0.995, bottom=0.005,
    )

    # --- row labels (left column, every row) ---
    for r, label in enumerate(ROW_LABELS):
        ax = fig.add_subplot(gs[r, 0]); _hide(ax)
        ax.set_facecolor("none")
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=ROW_LABEL_FONTSIZE, linespacing=1.15,
                rotation=0)

    # --- imagery (one column per frame) ---
    for c, fb in enumerate(frames):
        col = 1 + c
        # Row 0: Clip RGB
        ax = fig.add_subplot(gs[0, col]); _hide(ax); ax.imshow(fb.rgb)
        # Row 1: Wireframe overlay
        ax = fig.add_subplot(gs[1, col]); _hide(ax)
        draw_wireframe_overlay(ax, fb.rgb, fb.verts_pixels, faces_np)
        # Row 2: Phong mesh
        ax = fig.add_subplot(gs[2, col]); _hide(ax); ax.imshow(fb.mesh_shaded)
        # Row 3: Δ_expr magnitude
        ax = fig.add_subplot(gs[3, col]); _hide(ax)
        ax.imshow(deform_to_magnitude_rgb(fb.deform, fb.mask, vmax, cmap))
        # Row 4: pos-enc low
        ax = fig.add_subplot(gs[4, col]); _hide(ax)
        ax.imshow(pos_enc_channel_to_rgb(fb.pos_enc, POS_ENC_LOW_CH, fb.mask))
        # Row 5: pos-enc high
        ax = fig.add_subplot(gs[5, col]); _hide(ax)
        ax.imshow(pos_enc_channel_to_rgb(fb.pos_enc, POS_ENC_HIGH_CH, fb.mask))

    # --- shared colorbar — Spectral_r magnitude, range [0, vmax] ---
    # Anchored to the Δ_expr row (row 3) on the right edge of the figure.
    cax_y_top   = 0.005 + (height_ratios[4] + height_ratios[5]) / sum(height_ratios) * 0.99
    cax_y_bot   = cax_y_top
    # Place colorbar vertically centred on the Δ_expr row of the figure.
    row_height_frac = 1.0 / n_rows
    delta_row_top    = 1.0 - 3 * row_height_frac      # top of row 3 (0-indexed from top)
    delta_row_bottom = 1.0 - 4 * row_height_frac      # bottom of row 3
    cax = fig.add_axes([
        0.94,
        delta_row_bottom + row_height_frac * 0.05,
        0.014,
        row_height_frac * 0.9,
    ])
    sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, vmax / 2, vmax])
    cbar.set_ticklabels([f"{0:.2g}", f"{vmax / 2:.2g}", f"{vmax:.2g}"])
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render the same-identity driver-map example figure: "
                    "ONE clip across N expressive frames as columns, six "
                    "visualization rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_figures", type=int, default=1,
                   help="Number of figures to render (each picks a fresh clip).")
    p.add_argument("--n_frames",  type=int, default=4,
                   help="Number of frame columns per figure (default 4).")
    p.add_argument("--seed",      type=int, default=None,
                   help="Optional seed for reproducible draws. Default: time-based.")
    p.add_argument("--clip",      default=None,
                   help="Pin a specific clip ID (overrides random sampling).")
    p.add_argument("--frames",    nargs="+", type=int, default=None,
                   help="Pin specific frame indices within --clip. Length "
                        "must equal --n_frames if both are given.")
    p.add_argument("--flame_root", default=DEFAULT_FLAME_ROOT)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--out_root",  type=Path,
                   default=Path("outputs/paper_figures/driver_map"))
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def _filename(clip: str, frames: list[int], idx: Optional[int], pad: int) -> str:
    frame_token = "-".join(str(f) for f in frames)
    stem = f"{safe_name(clip)}__f{frame_token}"
    return f"{stem}.png" if idx is None else f"{idx:0{pad}d}_{stem}.png"


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
    faces_np = flame_skinner.template_faces.cpu().numpy()

    cond_module = SpatialConditioning(
        image_size=args.resolution, positional_channels=42,
        positional_multiplier=1.0,
    ).to(device).eval()
    renderer = _build_renderer(device)

    explicit_clip = args.clip is not None
    seed = args.seed if args.seed is not None else int(time.time())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = args.out_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fig] writing to {run_dir}")
    print(f"[fig] seed={seed}  (re-runnable with --seed {seed})")

    if not explicit_clip:
        clip_pool = discover_clips(flame_root, video_root)
        if not clip_pool:
            raise SystemExit(
                f"No clips with both fit.npz and .mp4 found under "
                f"{flame_root} / {video_root}."
            )
        print(f"[fig] random mode: pool of {len(clip_pool)} clips, "
              f"{args.n_figures} figure(s) × {args.n_frames} frames")

    if args.frames is not None:
        if not explicit_clip:
            raise SystemExit("--frames requires --clip to be set.")
        if len(args.frames) != args.n_frames:
            print(f"[fig] note: --n_frames={args.n_frames} ignored; using "
                  f"{len(args.frames)} pinned frames.")

    picks_log: list[dict] = []
    n_failed = 0
    n_total  = 1 if explicit_clip else args.n_figures
    n_pad    = max(2, len(str(n_total)))

    for i in range(n_total):
        if explicit_clip:
            clip = args.clip
            idx_token = None
        else:
            rng_clip = np.random.default_rng(seed + i)
            clip = str(rng_clip.choice(clip_pool))
            idx_token = i + 1

        try:
            fit = load_fit(flame_root / clip / "fit.npz")
            if args.frames is not None and explicit_clip:
                frames = list(args.frames)
            else:
                rng_f = np.random.default_rng(seed + 1000 + i)
                frames = pick_expressive_frames(fit, args.n_frames, rng_f)

            video = video_root / f"{clip}.mp4"
            bundles = [
                gather_frame(fit, f, video, flame_skinner, head_vert_ids,
                             cond_module, renderer, args.resolution, device)
                for f in frames
            ]
            out_path = run_dir / _filename(clip, frames, idx_token, n_pad)
            compose_figure(bundles, faces_np, out_path)
            picks_log.append({
                "index":   (i + 1) if not explicit_clip else 1,
                "clip":    clip,
                "frames":  frames,
                "out_png": str(out_path),
            })
            print(f"  rendered  clip={clip}  frames={frames}  → {out_path.name}")
        except Exception as e:
            n_failed += 1
            picks_log.append({
                "index":   (i + 1) if not explicit_clip else 1,
                "clip":    clip,
                "frames":  None,
                "out_png": None,
                "error":   f"{type(e).__name__}: {e}",
            })
            print(f"[fig] FAILED on clip={clip}: {type(e).__name__}: {e}")

    n_ok = sum(1 for p in picks_log if p.get("out_png"))
    manifest = {
        "seed":        seed,
        "seed_source": ("user --seed" if args.seed is not None else "wall-clock"),
        "timestamp":   timestamp,
        "mode":        "explicit" if explicit_clip else "random",
        "n_requested": n_total,
        "n_rendered":  n_ok,
        "n_failed":    n_total - n_ok,
        "args": {
            "flame_root": str(flame_root),
            "video_root": str(video_root),
            "resolution": args.resolution,
            "n_frames":   args.n_frames,
            "device":     str(device),
        },
        "picks":       picks_log,
    }
    (run_dir / "_index.json").write_text(json.dumps(manifest, indent=2))

    print(f"[fig] done: {n_ok}/{n_total} figures saved to {run_dir}/"
          + (f"  ({n_failed} failed)" if n_failed else ""))


if __name__ == "__main__":
    main()
