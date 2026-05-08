r"""Render paper-ready Loki-vs-SOTA comparison figures.

Each figure has the layout:

                  ┌──────────────────────────────────────────────┐
                  │ d_0 d_3 d_7 d_15   Driving Video
                  │ m_0 m_3 m_7 m_15   Loki
  [Reference]     │ a_0 a_3 a_7 a_15   AniTalker
                  │ e_0 e_3 e_7 e_15   EchoMimic
                  │ h_0 h_3 h_7 h_15   HunyuanPortrait
                  │ s_0 s_3 s_7 s_15   SadTalker
                  │ x_0 x_3 x_7 x_15   X-Portrait
                  └──────────────────────────────────────────────┘

  - The reference column spans every row.
  - Each row shows 4 frames at fixed indices (1st, 4th, 8th, 16th).
  - SadTalker (or any other baseline) appears even when the chosen sample
    isn't in its outputs; its cells are grey-filled placeholders so the
    figure stays structurally complete (paper-ready).

Each invocation lands in its own timestamped folder under the
`sota_vs_loki_comparison` subtree, so re-rolls don't overwrite
earlier draws and this script's outputs stay separate from other paper
figures (e.g. the FLAME-retargeting figures):

    outputs/paper_figures/sota_vs_loki_comparison/run_<YYYYMMDD_HHMMSS>/
        ├── <dataset>__<protocol>__<sample_id>.png
        ├── ...
        └── _index.json     # records seed + per-figure provenance

Usage (from repo root, inside the `loki` conda env):

    PYTHONPATH=. python paper/build_comparison_figure.py
    PYTHONPATH=. python paper/build_comparison_figure.py --n-figures 10
    PYTHONPATH=. python paper/build_comparison_figure.py \
        --datasets hdtf --protocols same_identity_reconstruction
    PYTHONPATH=. python paper/build_comparison_figure.py --seed 1234
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

import torch.nn.functional as F

from experiments.evaluation_metrics.metrics.io import (
    DEFAULT_FPS, DEFAULT_RESOLUTION, load_video,
)


def _center_square_resize(video, resolution: int = DEFAULT_RESOLUTION):
    """Center-square crop `(T, 3, H, W)` then bilinear-resize to
    `(resolution, resolution)`. SOTA `panel.mp4` / `driver.mp4` are
    already face-cropped at the wrapper layer; raw HDTF source clips are
    landscape and need this. Lightweight stand-in for the InsightFace
    crop the metrics package used to ship — exact alignment doesn't
    matter for a figure, only that the face is visible and roughly
    centered."""
    h, w = video.shape[-2], video.shape[-1]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    video = video[..., y0:y0 + s, x0:x0 + s]
    if video.shape[-1] != resolution:
        video = F.interpolate(video, size=(resolution, resolution),
                              mode="bilinear", align_corners=False)
    return video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SOTA_BASELINES_ORDERED = [
    "anitalker",
    "echomimic",
    "hunyuan_portrait",
    "sadtalker",
    "xportrait",
]

DISPLAY_NAMES = {
    "driving":          "Driving\nVideo",
    "loki":       "Loki",
    "anitalker":        "AniTalker",
    "echomimic":        "EchoMimic",
    "hunyuan_portrait": "HunyuanPortrait",
    "sadtalker":        "SadTalker",
    "xportrait":        "X-Portrait",
}

ALL_DATASETS  = ["hdtf"]
ALL_PROTOCOLS = ["same_identity_reconstruction", "cross_identity"]

# Explicit frame indices (0-based) sampled per row: 1st, 4th, 8th, 16th.
SAMPLED_FRAME_INDICES = [0, 3, 7, 15]
N_FRAMES_PER_ROW      = len(SAMPLED_FRAME_INDICES)

SOTA_ROOT       = Path("outputs/sota_comparison")
LOKI_ROOT = Path("outputs/loki_eval")
MANIFEST_DIR    = Path("experiments/sota_comparison/manifests")

# Loki's `panel.mp4` is the generated 512×512 video directly
# (16 frames at 25 fps). Earlier versions emitted a 4-row composite
# panel; the current adapter writes only the generation.


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FigurePick:
    """One chosen (dataset, protocol, sample_id) combo + the run dirs that
    will supply the rows."""
    dataset:    str
    protocol:   str
    sample_id:  str
    ref_uid:    str
    drv_uid:    str
    ref_clip:   dict             # manifest entry for ref UID
    loki: Path             # path to Loki panel.mp4
    driver:     Path             # path to driver clip (any sample dir's driver.mp4 or source)
    sota:       dict[str, Optional[Path]]   # baseline -> panel.mp4 or None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _latest_run(parent: Path) -> Optional[Path]:
    runs = sorted([d for d in parent.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def _load_manifest(dataset: str) -> dict[str, dict]:
    p = MANIFEST_DIR / f"{dataset}.json"
    m = json.loads(p.read_text())
    return {c["uid"]: c for c in m["clips"]}


def _split_sample_id(sample_id: str, protocol: str) -> tuple[str, str]:
    if protocol == "same_identity_reconstruction":
        return sample_id, sample_id
    if "_id_" not in sample_id:
        raise ValueError(f"cross-id sample_id without `_id_` separator: {sample_id}")
    ref, drv = sample_id.split("_id_", 1)
    return ref, f"id_{drv}"


def discover_picks(
    datasets:  list[str],
    protocols: list[str],
) -> list[FigurePick]:
    """Walk Loki + every SOTA tree, build the pool of available
    (dataset, protocol, sample_id) combinations. A combo enters the pool
    iff Loki has a `panel.mp4` for it; SOTA presence is recorded
    per-baseline (missing rows render as placeholders)."""
    picks: list[FigurePick] = []
    for dataset in datasets:
        manifest = _load_manifest(dataset)

        for protocol in protocols:
            mario_run = _latest_run(LOKI_ROOT / dataset / protocol)
            if mario_run is None:
                continue
            mario_samples_root = mario_run / "samples"
            if not mario_samples_root.is_dir():
                continue

            sota_runs = {
                b: _latest_run(SOTA_ROOT / b / dataset / protocol)
                for b in SOTA_BASELINES_ORDERED
            }

            for sample_dir in sorted(mario_samples_root.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sid = sample_dir.name
                mario_panel = sample_dir / "panel.mp4"
                if not mario_panel.is_file():
                    continue

                ref_uid, drv_uid = _split_sample_id(sid, protocol)
                if ref_uid not in manifest or drv_uid not in manifest:
                    continue

                # Find any baseline that has a `driver.mp4` (the populate
                # script wrote it next to every panel.mp4 across SOTA).
                # Loki runs don't have it, so we borrow from
                # whichever SOTA tree has the sample.
                driver_path: Optional[Path] = None
                sota_panels: dict[str, Optional[Path]] = {}
                for b in SOTA_BASELINES_ORDERED:
                    run = sota_runs[b]
                    if run is None:
                        sota_panels[b] = None
                        continue
                    cand = run / "samples" / sid / "panel.mp4"
                    sota_panels[b] = cand if cand.is_file() else None
                    drv_cand = run / "samples" / sid / "driver.mp4"
                    if driver_path is None and drv_cand.is_file():
                        driver_path = drv_cand

                if driver_path is None:
                    # No SOTA tree has the driver yet — fall back to the
                    # source video.
                    driver_path = Path(manifest[drv_uid]["video_path"])

                picks.append(FigurePick(
                    dataset    = dataset,
                    protocol   = protocol,
                    sample_id  = sid,
                    ref_uid    = ref_uid,
                    drv_uid    = drv_uid,
                    ref_clip   = manifest[ref_uid],
                    loki = mario_panel,
                    driver     = driver_path,
                    sota       = sota_panels,
                ))
    return picks


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _video_to_uint8_frames(video_tensor) -> np.ndarray:
    """`(T, 3, H, W)` float32 in `[0, 1]` → `(T, H, W, 3)` uint8."""
    return (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def _stride_indices(T: int) -> list[int]:
    """Return the explicit `SAMPLED_FRAME_INDICES`, clamped to `T-1` for
    short clips."""
    if T <= 0:
        return []
    return [min(i, T - 1) for i in SAMPLED_FRAME_INDICES]


def _peek_loki_panel_length(path: Path) -> int:
    """How many time-axis frames Loki's panel.mp4 carries. The crop
    and stride for every other row are derived from this — every row in
    the final figure shares Loki's wall-clock coverage."""
    return load_video(path, fps=DEFAULT_FPS, resolution=None).shape[0]


def _load_and_sample(
    path:       Optional[Path],
    crop:       bool = False,
    max_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load a clip, trim to `max_frames`, and return `n` stride-sampled
    uint8 frames as `(n, H, W, 3)`. Returns None if `path` is missing.

    `max_frames` caps the time axis BEFORE the index pick, so SOTA
    panels (75–125 frames) get cut to Loki's panel length (16)
    first, then frames at the fixed `SAMPLED_FRAME_INDICES` are taken.
    Columns then align in wall-clock content across rows.

    `crop=True` applies a center-square crop + resize for raw landscape
    clips (HDTF source); SOTA panels and `driver.mp4` already arrive
    face-cropped near-square so the default skips it.
    """
    if path is None or not path.is_file():
        return None
    video = load_video(path, fps=DEFAULT_FPS, resolution=None,
                       max_frames=max_frames)
    if crop:
        video = _center_square_resize(video)
    indices = _stride_indices(video.shape[0])
    if not indices:
        return None
    return _video_to_uint8_frames(video[indices])


def _load_loki_generated(path: Optional[Path]) -> Optional[np.ndarray]:
    """Loki's `panel.mp4` is the 512×512 generated video as-is."""
    return _load_and_sample(path, crop=False)


def _load_reference_image(ref_clip: dict) -> Optional[np.ndarray]:
    """First center-square-cropped frame of the ref UID's source video,
    at 512×512."""
    src = Path(ref_clip["video_path"])
    if not src.is_file():
        return None
    video = load_video(src, fps=DEFAULT_FPS, resolution=None, max_frames=1)
    video = _center_square_resize(video)
    return _video_to_uint8_frames(video[:1])[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _strip_axis(ax, edgecolor="black", facecolor="white"):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.0)
        s.set_color(edgecolor)
    ax.set_facecolor(facecolor)
    ax.set_aspect("equal", adjustable="box")


def render_figure(pick: FigurePick, out_path: Path) -> None:
    """Render one figure for one pick and save to `out_path`."""
    n_rows  = 1 + 1 + len(SOTA_BASELINES_ORDERED)   # driving + loki + 5 SOTA

    # ------------- gather pixel data -------------
    # Cap every non-Loki row to Loki's actual panel length so
    # the columns align in wall-clock content. With Loki's default
    # `n_frames=16` at 25 fps, that's 0.64 s; SOTA panels (75–125 frames)
    # would otherwise span 3–5 s and the row contents wouldn't match
    # in time.
    T_mario = _peek_loki_panel_length(pick.loki)

    ref_image   = _load_reference_image(pick.ref_clip)
    driver_grid = _load_and_sample(pick.driver, crop=True, max_frames=T_mario)
    mario_grid  = _load_loki_generated(pick.loki)
    sota_grids  = {
        b: _load_and_sample(p, crop=False, max_frames=T_mario)
        for b, p in pick.sota.items()
    }

    row_data: list[tuple[str, Optional[np.ndarray]]] = [
        ("driving",    driver_grid),
        ("loki", mario_grid),
    ]
    for b in SOTA_BASELINES_ORDERED:
        row_data.append((b, sota_grids.get(b)))

    # ------------- layout -------------
    # Columns: [reference | gap | N frames | gap | label]
    width_ratios = [2.4, 0.25] + [1.0] * N_FRAMES_PER_ROW + [0.25, 0.85]
    fig_w = sum(width_ratios) * 0.85
    fig_h = n_rows * 1.0 * 0.85 * 1.02

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows, len(width_ratios),
        width_ratios=width_ratios,
        wspace=0.04, hspace=0.04,
        left=0.02, right=0.985, top=0.96, bottom=0.02,
    )

    # Reference column spans every row.
    ax_ref = fig.add_subplot(gs[:, 0])
    _strip_axis(ax_ref)
    if ref_image is not None:
        ax_ref.imshow(ref_image)
    ax_ref.set_title("Reference", fontsize=12, pad=6)

    for r, (name, grid) in enumerate(row_data):
        for c in range(N_FRAMES_PER_ROW):
            ax = fig.add_subplot(gs[r, 2 + c])
            if grid is not None:
                _strip_axis(ax)
                ax.imshow(grid[c])
            else:
                _strip_axis(ax, edgecolor="#bbbbbb", facecolor="#ececec")
        # Row label on the far right.
        ax_lbl = fig.add_subplot(gs[r, -1])
        ax_lbl.text(0.05, 0.5, DISPLAY_NAMES[name],
                    fontsize=11, ha="left", va="center")
        ax_lbl.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Render Loki-vs-SOTA comparison figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets",  nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS)
    p.add_argument("--protocols", nargs="+", default=ALL_PROTOCOLS,
                   choices=ALL_PROTOCOLS)
    p.add_argument("--n-figures", type=int, default=5,
                   help="How many random samples to render per invocation.")
    p.add_argument("--seed",      type=int, default=None,
                   help="Optional seed for reproducible draws. Default: time-based.")
    p.add_argument("--out-root",  type=Path,
                   default=Path("outputs/paper_figures/sota_vs_loki_comparison"),
                   help="Parent dir; the script writes to a timestamped "
                        "<out-root>/run_<YYYYMMDD_HHMMSS>/ subdir each invocation.")
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    rng  = np.random.default_rng(seed)

    pool = discover_picks(args.datasets, args.protocols)
    if not pool:
        raise SystemExit(
            f"No (dataset, protocol, sample_id) combinations found. Datasets: "
            f"{args.datasets}  protocols: {args.protocols}. Did Loki eval "
            f"and the SOTA backfill run?"
        )

    n = min(args.n_figures, len(pool))
    chosen_idx = rng.choice(len(pool), size=n, replace=False)
    chosen     = [pool[i] for i in chosen_idx]

    # Per-invocation timestamped subdir under --out-root.
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paper-fig] pool size: {len(pool)}  picking: {n}  seed: {seed}")
    print(f"[paper-fig] writing to: {out_dir}")

    index: list[dict] = []
    for pick in chosen:
        out_path = out_dir / f"{pick.dataset}__{pick.protocol}__{pick.sample_id}.png"
        print(f"  rendering  {pick.dataset}/{pick.protocol}/{pick.sample_id}  → {out_path.name}")
        render_figure(pick, out_path)
        index.append({
            "dataset":   pick.dataset,
            "protocol":  pick.protocol,
            "sample_id": pick.sample_id,
            "ref_uid":   pick.ref_uid,
            "drv_uid":   pick.drv_uid,
            "rows": {
                "driver":     str(pick.driver),
                "loki": str(pick.loki),
                **{b: (str(p) if p else None) for b, p in pick.sota.items()},
            },
            "out_png":   str(out_path),
        })

    (out_dir / "_index.json").write_text(json.dumps({
        "seed":      seed,
        "timestamp": ts,
        "datasets":  args.datasets,
        "protocols": args.protocols,
        "n_figures": n,
        "picks":     index,
    }, indent=2))
    print(f"[paper-fig] done. Index → {out_dir / '_index.json'}")


if __name__ == "__main__":
    main()
