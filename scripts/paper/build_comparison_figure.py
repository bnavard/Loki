r"""Render paper-ready Marionette-vs-SOTA comparison figures.

Each figure has the layout:

                  ┌──────────────────────────────────────────────┐
                  │ d_0 d_1 d_2 d_3 d_4 d_5 d_6 d_7   Driving Video
                  │ m_0 m_1 m_2 m_3 m_4 m_5 m_6 m_7   Marionette
  [Reference]     │ a_0 a_1 ...                       AniTalker
                  │ e_0 e_1 ...                       EchoMimic
                  │ h_0 h_1 ...                       HunyuanPortrait
                  │ s_0 s_1 ...                       SadTalker
                  │ x_0 x_1 ...                       X-Portrait
                  └──────────────────────────────────────────────┘

  - The reference column spans every row.
  - Each row shows 8 frames evenly spaced over its own time axis.
  - SadTalker (or any other baseline) appears even when the chosen sample
    isn't in its outputs; its cells are grey-filled placeholders so the
    figure stays structurally complete (paper-ready).

Each invocation lands in its own timestamped folder so re-rolls don't
overwrite earlier draws:

    outputs/paper_figures/run_<YYYYMMDD_HHMMSS>/
        ├── <dataset>__<protocol>__<sample_id>.png
        ├── ...
        └── _index.json     # records seed + per-figure provenance

Usage (from repo root):

    python scripts/paper/build_comparison_figure.py
    python scripts/paper/build_comparison_figure.py --n-figures 10
    python scripts/paper/build_comparison_figure.py \
        --datasets talkvid --protocols same_identity_reconstruction
    python scripts/paper/build_comparison_figure.py --seed 1234
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

from experiments.evaluation_metrics.metrics.evaluator import _build_face_detector
from experiments.evaluation_metrics.metrics.io import (
    DEFAULT_FACE_CROP_MARGIN, DEFAULT_FPS, DEFAULT_RESOLUTION,
    face_crop_around_detection, load_video,
)


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
    "marionette":       "Marionette",
    "anitalker":        "AniTalker",
    "echomimic":        "EchoMimic",
    "hunyuan_portrait": "HunyuanPortrait",
    "sadtalker":        "SadTalker",
    "xportrait":        "X-Portrait",
}

ALL_DATASETS  = ["talkvid", "hdtf"]
ALL_PROTOCOLS = ["same_identity_reconstruction", "cross_identity"]

N_FRAMES_PER_ROW = 8

SOTA_ROOT       = Path("outputs/sota_comparison")
MARIONETTE_ROOT = Path("outputs/marionette_eval")
MANIFEST_DIR    = Path("experiments/sota_comparison/manifests")

# Marionette's panel.mp4 is a 4-row composite (Reference, Driver Video,
# Cond preview, Generated) with a 70px-wide vertical label strip on the
# left and an optional ~30px title bar on top — see
# `marionette/utils/viz.py::save_video_with_audio` and
# `experiments/marionette_eval/adapter.py`. For the paper figure we want
# only the Generated row (the bottom one).
MARIONETTE_LABEL_STRIP_W = 70
MARIONETTE_ROW_H         = 512
MARIONETTE_N_ROWS        = 4
MARIONETTE_GEN_ROW_IDX   = 3   # 0-based: ref=0, driver=1, cond=2, gen=3


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
    marionette: Path             # path to Marionette panel.mp4
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
    """Walk Marionette + every SOTA tree, build the pool of available
    (dataset, protocol, sample_id) combinations. A combo enters the pool
    iff Marionette has a `panel.mp4` for it; SOTA presence is recorded
    per-baseline (missing rows render as placeholders)."""
    picks: list[FigurePick] = []
    for dataset in datasets:
        manifest = _load_manifest(dataset)

        for protocol in protocols:
            mario_run = _latest_run(MARIONETTE_ROOT / dataset / protocol)
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
                # Marionette runs don't have it, so we borrow from
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
                    marionette = mario_panel,
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


def _stride_indices(T: int, n: int = N_FRAMES_PER_ROW) -> list[int]:
    """Pick `n` frame indices with a constant stride starting at 0.

    `stride = max(1, T // n)`. Picks `[0, stride, 2·stride, …]` clamped
    to `T-1` for the very-short-clip case. This matches "every 2 frames
    out of 16" for Marionette's default panel and generalizes if the
    panel length changes (32 frames → stride 4, etc.).
    """
    if T <= 0:
        return []
    stride = max(1, T // n)
    return [min(i * stride, T - 1) for i in range(n)]


def _peek_marionette_panel_length(path: Path) -> int:
    """How many time-axis frames Marionette's panel.mp4 carries. The crop
    and stride for every other row are derived from this — every row in
    the final figure shares Marionette's wall-clock coverage."""
    return load_video(path, fps=DEFAULT_FPS, resolution=None).shape[0]


def _load_and_sample(
    path:       Optional[Path],
    detect_fn   = None,
    max_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load a clip, trim to `max_frames`, and return `n` stride-sampled
    uint8 frames as `(n, H, W, 3)`. Returns None if `path` is missing.

    `max_frames` caps the time axis BEFORE the stride pick, so SOTA
    panels (75–125 frames) get cut to Marionette's panel length first,
    then 8 frames are taken at stride 2. Columns then align in
    wall-clock content across rows.
    """
    if path is None or not path.is_file():
        return None
    video = load_video(path, fps=DEFAULT_FPS, resolution=None,
                       max_frames=max_frames)
    if detect_fn is not None:
        cropped = face_crop_around_detection(
            video, detect_fn, margin=DEFAULT_FACE_CROP_MARGIN,
            target_resolution=DEFAULT_RESOLUTION,
        )
        if cropped is not None:
            video = cropped
    indices = _stride_indices(video.shape[0])
    if not indices:
        return None
    return _video_to_uint8_frames(video[indices])


def _load_marionette_generated(path: Optional[Path]) -> Optional[np.ndarray]:
    """Load Marionette's 4-row composite `panel.mp4` and return only the
    Generated row's `n` stride-sampled frames as `(n, 512, 512, 3)` uint8.

    Geometry is deterministic (see module-level constants): a 70 px label
    strip on the left, optional title bar on top (height = `total_h −
    4·512`), and 4 rows of 512 px each in fixed order.
    """
    if path is None or not path.is_file():
        return None
    video = load_video(path, fps=DEFAULT_FPS, resolution=None)
    total_h = int(video.shape[-2])
    title_h = total_h - MARIONETTE_N_ROWS * MARIONETTE_ROW_H
    if title_h < 0:
        # Unexpected geometry (older / different panel layout). Fall back
        # to the whole frame so the figure still renders, but warn.
        print(f"[paper-fig] WARN: Marionette panel at {path} has unexpected "
              f"height {total_h} (< {MARIONETTE_N_ROWS * MARIONETTE_ROW_H}); "
              f"using full frame.")
        return _load_and_sample(path)

    y0 = title_h + MARIONETTE_GEN_ROW_IDX * MARIONETTE_ROW_H
    y1 = y0 + MARIONETTE_ROW_H
    x0 = MARIONETTE_LABEL_STRIP_W
    x1 = x0 + MARIONETTE_ROW_H
    gen = video[:, :, y0:y1, x0:x1]                  # (T, 3, 512, 512)

    indices = _stride_indices(gen.shape[0])
    if not indices:
        return None
    return _video_to_uint8_frames(gen[indices])


def _load_reference_image(ref_clip: dict, detect_fn) -> Optional[np.ndarray]:
    """First face-cropped frame of the ref UID's source video, at 512×512."""
    src = Path(ref_clip["video_path"])
    if not src.is_file():
        return None
    video = load_video(src, fps=DEFAULT_FPS, resolution=None, max_frames=10)
    cropped = face_crop_around_detection(
        video, detect_fn, margin=DEFAULT_FACE_CROP_MARGIN,
        target_resolution=DEFAULT_RESOLUTION,
    )
    if cropped is None:
        return None
    return _video_to_uint8_frames(cropped[:1])[0]


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


def render_figure(pick: FigurePick, detect_fn, out_path: Path) -> None:
    """Render one figure for one pick and save to `out_path`."""
    n_rows  = 1 + 1 + len(SOTA_BASELINES_ORDERED)   # driving + marionette + 5 SOTA

    # ------------- gather pixel data -------------
    # Cap every non-Marionette row to Marionette's actual panel length so
    # the columns align in wall-clock content. With Marionette's default
    # `n_frames=16` at 25 fps, that's 0.64 s; SOTA panels (75–125 frames)
    # would otherwise span 3–5 s and the row contents wouldn't match
    # in time.
    T_mario = _peek_marionette_panel_length(pick.marionette)

    ref_image   = _load_reference_image(pick.ref_clip, detect_fn)
    driver_grid = _load_and_sample(pick.driver,    detect_fn=detect_fn,
                                   max_frames=T_mario)
    mario_grid  = _load_marionette_generated(pick.marionette)
    sota_grids  = {
        b: _load_and_sample(p, detect_fn=None, max_frames=T_mario)
        for b, p in pick.sota.items()
    }

    row_data: list[tuple[str, Optional[np.ndarray]]] = [
        ("driving",    driver_grid),
        ("marionette", mario_grid),
    ]
    for b in SOTA_BASELINES_ORDERED:
        row_data.append((b, sota_grids.get(b)))

    # ------------- layout -------------
    # Columns: [reference | gap | 8 frames | gap | label]
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
        description="Render Marionette-vs-SOTA comparison figures.",
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
    p.add_argument("--out-root",  type=Path, default=Path("outputs/paper_figures"),
                   help="Parent dir; the script writes to a timestamped "
                        "<out-root>/run_<YYYYMMDD_HHMMSS>/ subdir each invocation.")
    p.add_argument("--device",    default="cuda",
                   help="Device for face-detection (driver + reference cropping).")
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    rng  = np.random.default_rng(seed)

    pool = discover_picks(args.datasets, args.protocols)
    if not pool:
        raise SystemExit(
            f"No (dataset, protocol, sample_id) combinations found. Datasets: "
            f"{args.datasets}  protocols: {args.protocols}. Did Marionette eval "
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

    detect_fn = _build_face_detector(args.device)

    index: list[dict] = []
    for pick in chosen:
        out_path = out_dir / f"{pick.dataset}__{pick.protocol}__{pick.sample_id}.png"
        print(f"  rendering  {pick.dataset}/{pick.protocol}/{pick.sample_id}  → {out_path.name}")
        render_figure(pick, detect_fn, out_path)
        index.append({
            "dataset":   pick.dataset,
            "protocol":  pick.protocol,
            "sample_id": pick.sample_id,
            "ref_uid":   pick.ref_uid,
            "drv_uid":   pick.drv_uid,
            "rows": {
                "driver":     str(pick.driver),
                "marionette": str(pick.marionette),
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
