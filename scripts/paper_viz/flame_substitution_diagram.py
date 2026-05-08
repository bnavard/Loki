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

Usage (from repo root, loki conda env):

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
from matplotlib.colors import Normalize

from loki.flame.flame import FlameSkinnerExtended, compute_flame
from loki.retargeting import prepare_reference
from loki.utils import (
    crop_image, get_bbox_from_verts, load_frame,
    rescale_image, verts_to_pytorch3d,
)

from src.flame_render import (
    DEFAULT_FLAME_ROOT, DEFAULT_VIDEO_ROOT, DEFORM_CMAP_NAME, HEAD_VERT_PATH,
    build_renderer, compute_vmax, deform_to_magnitude_rgb, discover_clips,
    flame_inputs, hide_axis, load_fit, motion_score, ndc_verts,
    rasterize_conditioning, render_shaded_mesh, safe_name,
)


# ---------------------------------------------------------------------------
# Substitution-specific FLAME helpers (per-driver / per-retarget projection)
# ---------------------------------------------------------------------------

def load_driver_frame(
    driver_fit: dict, t: int, video_path: Path, resolution: int,
    flame_skinner: FlameSkinnerExtended, head_vert_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + crop one driver RGB frame at time `t` under the driver's own
    FLAME geometry; return cropped image plus per-frame NDC verts and
    per-vertex deformation offsets."""
    fi = flame_inputs(driver_fit, t)
    fo = compute_flame(flame_skinner, fi)
    verts_2d = fo["verts_2d"][0, 0]
    offsets  = fo["offsets_3d"][0]
    crop_box = get_bbox_from_verts(verts_2d.copy(), head_vert_ids)

    img = load_frame(video_path, t)
    img = crop_image(img, crop_box, bg_value=255)
    img = rescale_image(img, resolution).astype(np.uint8)

    verts_3d = verts_to_pytorch3d(verts_2d.copy(), np.array(crop_box))
    return img, verts_3d.astype(np.float32), offsets.astype(np.float32)


def retargeted_verts_offsets(
    ref_fit: dict, driver_fit: dict, t: int,
    ref_crop_box: np.ndarray, flame_skinner: FlameSkinnerExtended,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FLAME with REF identity (β_ref, camera_ref) under the DRIVER's
    motion (ψ, θ) at time `t`; return per-frame NDC verts (in REF crop
    space) + per-vertex deformation offsets."""
    fi = flame_inputs(driver_fit, t)
    fi["shape"] = ref_fit["shape"]
    for k in ("fx", "fy", "cx", "cy", "extr"):
        fi[k] = ref_fit[k][[0]]

    fo = compute_flame(flame_skinner, fi)
    verts_2d   = fo["verts_2d"][0, 0]
    offsets    = fo["offsets_3d"][0]
    verts_3d   = verts_to_pytorch3d(verts_2d.copy(), np.array(ref_crop_box))
    return verts_3d.astype(np.float32), offsets.astype(np.float32)


# ---------------------------------------------------------------------------
# Substitution-specific visual constants (arrow column, captions)
# ---------------------------------------------------------------------------

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


def _best_driver_frame(fit: dict) -> int:
    """Pick the highest-motion frame within the middle 80% of the clip."""
    n = fit["expr"].shape[0]
    lo, hi = max(1, int(n * 0.10)), max(2, int(n * 0.90))
    score = motion_score(fit)
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
    score = motion_score(fit)[lo:hi]
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


def gather_pair(
    spec: PairSpec,
    flame_root: Path,
    video_root: Path,
    flame_skinner: FlameSkinnerExtended,
    head_vert_ids: np.ndarray,
    cond_module: SpatialConditioning,
    phong_renderer: PropRenderer,
    image_size: int,
    device: torch.device,
) -> PairBundle:
    ref_fit = load_fit(flame_root / spec.ref_clip    / "fit.npz")
    drv_fit = load_fit(flame_root / spec.driver_clip / "fit.npz")

    # --- Reference side: RGB, shaded mesh ---
    ref_video = video_root / f"{spec.ref_clip}.mp4"
    ref_img_norm, _, ref_crop_box = prepare_reference(
        ref_fit, spec.ref_frame, ref_video,
        image_size, flame_skinner, head_vert_ids,
    )
    ref_rgb = ((ref_img_norm + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    fo_ref = compute_flame(flame_skinner, flame_inputs(ref_fit, spec.ref_frame))
    ref_verts_2d_orig = fo_ref["verts_2d"][0, 0]
    ref_verts_3d_cv   = fo_ref["verts_3d_cv"][0]
    ref_verts_ndc     = ndc_verts(ref_verts_2d_orig, ref_crop_box)

    ref_mesh_shaded = render_shaded_mesh(
        ref_verts_ndc, ref_verts_3d_cv, flame_skinner.template_faces,
        image_size, device, phong_renderer,
    )

    # --- Driver side: RGB (own crop), shaded mesh, Δ_expr ---
    drv_video = video_root / f"{spec.driver_clip}.mp4"
    drv_rgb_arr, _, _ = load_driver_frame(
        drv_fit, spec.driver_frame, drv_video,
        image_size, flame_skinner, head_vert_ids,
    )
    fo_drv = compute_flame(flame_skinner, flame_inputs(drv_fit, spec.driver_frame))
    drv_verts_2d_orig = fo_drv["verts_2d"][0, 0]
    drv_verts_3d_cv   = fo_drv["verts_3d_cv"][0]
    drv_offsets       = fo_drv["offsets_3d"][0]
    drv_crop_box      = get_bbox_from_verts(drv_verts_2d_orig.copy(), head_vert_ids)
    drv_verts_ndc     = ndc_verts(drv_verts_2d_orig, drv_crop_box)

    drv_mesh_shaded = render_shaded_mesh(
        drv_verts_ndc, drv_verts_3d_cv, flame_skinner.template_faces,
        image_size, device, phong_renderer,
    )
    _, drv_deform, drv_deform_mask = rasterize_conditioning(
        cond_module, drv_verts_ndc, drv_offsets, device,
    )

    # --- Retargeted: substitute β_ref + (ψ_drv, θ_drv) under cam_ref ---
    ret_verts, ret_offsets = retargeted_verts_offsets(
        ref_fit, drv_fit, spec.driver_frame, ref_crop_box, flame_skinner,
    )
    _, retarget_deform, retarget_mask = rasterize_conditioning(
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

def _draw_arrow_column(fig, ax):
    """Draw a thick rightward arrow with the substitution equation above
    and 'parametric substitution' below."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    hide_axis(ax)
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
    vmax = compute_vmax([pair.drv_deform, pair.retarget_deform], q=99.5)
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
    ax = fig.add_subplot(gs[0, 0]); hide_axis(ax); ax.imshow(pair.ref_rgb)
    # (1) ref shaded mesh
    ax = fig.add_subplot(gs[0, 1]); hide_axis(ax); ax.imshow(pair.ref_mesh_shaded)
    # (2) driver RGB
    ax = fig.add_subplot(gs[0, 2]); hide_axis(ax); ax.imshow(pair.drv_rgb)
    # (3) driver shaded mesh
    ax = fig.add_subplot(gs[0, 3]); hide_axis(ax); ax.imshow(pair.drv_mesh_shaded)
    # (4) driver Δ_expr magnitude
    ax = fig.add_subplot(gs[0, 4]); hide_axis(ax)
    ax.imshow(deform_to_magnitude_rgb(pair.drv_deform, pair.drv_deform_mask,
                                      vmax, cmap))
    # (5) substitution arrow (no text; user annotates manually)
    ax_arrow = fig.add_subplot(gs[0, 5]); _draw_arrow_column(fig, ax_arrow)
    # (6) retargeted Δ_expr magnitude
    ax = fig.add_subplot(gs[0, 6]); hide_axis(ax)
    ax.imshow(deform_to_magnitude_rgb(pair.retarget_deform, pair.retarget_deform_mask,
                                      vmax, cmap))

    # --- column captions row ---
    for c, label in enumerate(COL_LABELS):
        if not label:
            continue
        ax = fig.add_subplot(gs[1, c]); hide_axis(ax)
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
    flame_skinner = FlameSkinnerExtended(
        add_mouth=True, n_shape_params=150, n_expr_params=65,
    ).eval()
    head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)
    _set_faces(flame_skinner.template_faces)

    cond_module = SpatialConditioning(
        image_size=args.resolution,
        positional_channels=42,
        positional_multiplier=1.0,
    ).to(device).eval()

    phong = build_renderer(device)

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
