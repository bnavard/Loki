r"""Marionette teaser figure — two cross-identity reenactment examples.

Marionette-only (no SOTA comparison). Each figure has two stacked blocks,
both showing cross-identity retargeting (different ref / driver pairs);
each block has a reference column on the left spanning two rows, with
driver frames on the upper row and Marionette generations on the lower:

    Block 1 — Cross-Identity Retargeting (pair A)
        ┌─────┬──────┬──────┬──────┬──────┐
        │     │ d_1  │ d_2  │ d_3  │ d_4  │   driver frames
        │ ref ├──────┼──────┼──────┼──────┤
        │     │ g_1  │ g_2  │ g_3  │ g_4  │   Marionette generated
        └─────┴──────┴──────┴──────┴──────┘

    Block 2 — Cross-Identity Retargeting (pair B)
        (same layout, a different ref / driver pair)

All three streams come from the Marionette eval run itself:
  * `panel.mp4`              — generated 16-frame video (the prediction)
  * `scratch/.../source.png` — face-cropped reference frame the model saw
  * `scratch/.../driver.mp4` — face-cropped driver clip the model saw

Reading the eval's own scratch artifacts (rather than re-cropping raw
source clips) is what makes the driver column actually correspond to the
frames Marionette generated against — same crop box, same 16 frames, same
fps. Re-cropping the source video would diverge on every variable-fps clip.

Each invocation lands in its own timestamped folder under
`outputs/paper_figures/teaser/run_<YYYYmmdd_HHMMSS>/` and writes an
`_index.json` recording the seed and per-figure provenance.

Usage (from repo root, inside the marionette conda env):

    # default — 1 figure, two random cross-identity picks across all datasets
    PYTHONPATH=. python paper/paper_teaser_figure.py

    # render 5 figures, talkvid only, fixed seed
    PYTHONPATH=. python paper/paper_teaser_figure.py \
        --datasets talkvid --n_figures 5 --seed 7

    # pin specific sample_ids (and the dataset they came from)
    PYTHONPATH=. python paper/paper_teaser_figure.py \
        --dataset_for_picks talkvid \
        --sample_a id_0001_id_0026 --sample_b id_0077_id_0069
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from experiments.evaluation_metrics.metrics.io import DEFAULT_FPS, load_video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DATASETS = ["hdtf", "talkvid"]
PROTOCOL     = "cross_identity"

# Frame indices sampled per row (1st, 4th, 8th, 16th of Marionette's window).
SAMPLED_FRAME_INDICES = [0, 3, 7, 15]
N_FRAMES_PER_ROW      = len(SAMPLED_FRAME_INDICES)

MARIONETTE_ROOT = Path("outputs/marionette_eval")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeaserPick:
    """One cross-identity Marionette eval sample. Every path points into
    the eval run itself — no cross-referencing source videos."""
    dataset:    str
    sample_id:  str
    ref_uid:    str
    drv_uid:    str
    source_png: Path                  # `<run>/scratch/<sid>/source.png` (ref)
    driver_mp4: Path                  # `<run>/scratch/<sid>/driver.mp4` (driver)
    panel_mp4:  Path                  # `<run>/samples/<sid>/panel.mp4`  (generation)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _latest_run(parent: Path) -> Optional[Path]:
    runs = sorted([d for d in parent.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def _split_sample_id(sample_id: str) -> tuple[str, str]:
    """Cross-identity sample_id = `<ref_uid>_id_<drv_uid_tail>`
    (e.g. `id_0001_id_0026`)."""
    if "_id_" not in sample_id:
        raise ValueError(f"cross-id sample_id missing `_id_` separator: {sample_id}")
    ref, drv_tail = sample_id.split("_id_", 1)
    return ref, f"id_{drv_tail}"


def discover_picks(datasets: list[str]) -> list[TeaserPick]:
    """Walk the latest Marionette cross-identity eval run per dataset and
    enumerate every sample that has a complete `panel.mp4` + scratch
    `source.png` + scratch `driver.mp4` triple."""
    pool: list[TeaserPick] = []
    for dataset in datasets:
        run = _latest_run(MARIONETTE_ROOT / dataset / PROTOCOL)
        if run is None:
            continue
        samples_root = run / "samples"
        scratch_root = run / "scratch"
        if not samples_root.is_dir() or not scratch_root.is_dir():
            continue

        for sample_dir in sorted(samples_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            sid         = sample_dir.name
            panel_mp4   = sample_dir / "panel.mp4"
            source_png  = scratch_root / sid / "source.png"
            driver_mp4  = scratch_root / sid / "driver.mp4"
            if not (panel_mp4.is_file() and source_png.is_file() and driver_mp4.is_file()):
                continue

            ref_uid, drv_uid = _split_sample_id(sid)
            pool.append(TeaserPick(
                dataset    = dataset,
                sample_id  = sid,
                ref_uid    = ref_uid,
                drv_uid    = drv_uid,
                source_png = source_png,
                driver_mp4 = driver_mp4,
                panel_mp4  = panel_mp4,
            ))
    return pool


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _video_to_uint8_frames(video_tensor) -> np.ndarray:
    """`(T, 3, H, W)` float32 in `[0, 1]` → `(T, H, W, 3)` uint8."""
    return (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def _stride_indices(T: int) -> list[int]:
    """`SAMPLED_FRAME_INDICES`, clamped to `T-1` for short clips."""
    if T <= 0:
        return []
    return [min(i, T - 1) for i in SAMPLED_FRAME_INDICES]


def _load_video_strided(path: Path) -> np.ndarray:
    """Load a 25fps face-cropped video from the eval scratch tree (or the
    Marionette panel) and return stride-sampled uint8 frames `(n, H, W, 3)`."""
    video = load_video(path, fps=DEFAULT_FPS, resolution=None)
    indices = _stride_indices(video.shape[0])
    return _video_to_uint8_frames(video[indices])


def _load_source_png(path: Path) -> np.ndarray:
    """Read the eval's pre-rendered reference frame as `(H, W, 3)` uint8."""
    img = iio.imread(path)
    return img[..., :3] if img.ndim == 3 and img.shape[-1] == 4 else img


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _hide_spines(ax, edgecolor="white", facecolor="white"):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(facecolor)
    ax.set_aspect("equal", adjustable="box")


def _draw_block(
    fig: plt.Figure,
    gs: gridspec.GridSpec,
    block_rows: tuple[int, int],
    ref_image:   np.ndarray,
    driver_grid: np.ndarray,
    mario_grid:  np.ndarray,
) -> None:
    """Render one (ref column + driver row + Marionette row) block."""
    r_drv, r_gen = block_rows

    ax_ref = fig.add_subplot(gs[r_drv:r_gen + 1, 0])
    _hide_spines(ax_ref)
    ax_ref.imshow(ref_image)

    for c in range(N_FRAMES_PER_ROW):
        ax_drv = fig.add_subplot(gs[r_drv, 1 + c])
        _hide_spines(ax_drv)
        ax_drv.imshow(driver_grid[c])

        ax_gen = fig.add_subplot(gs[r_gen, 1 + c])
        _hide_spines(ax_gen)
        ax_gen.imshow(mario_grid[c])


def render_figure(
    pick_a: TeaserPick,
    pick_b: TeaserPick,
    out_path: Path,
) -> None:
    """Render one teaser figure with two stacked cross-identity blocks."""
    def _gather(pick: TeaserPick):
        return (
            _load_source_png(pick.source_png),
            _load_video_strided(pick.driver_mp4),
            _load_video_strided(pick.panel_mp4),
        )

    a_ref, a_drv, a_gen = _gather(pick_a)
    b_ref, b_drv, b_gen = _gather(pick_b)

    n_rows  = 4
    n_cols  = 1 + N_FRAMES_PER_ROW
    width_ratios = [2.0] + [1.0] * N_FRAMES_PER_ROW

    fig_w = sum(width_ratios) * 1.2
    fig_h = n_rows * 1.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.04, hspace=0.04,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )

    _draw_block(fig, gs, (0, 1), a_ref, a_drv, a_gen)
    _draw_block(fig, gs, (2, 3), b_ref, b_drv, b_gen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render Marionette-only teaser figures (two stacked "
                    "cross-identity blocks per figure).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets",  nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS,
                   help="Datasets to source Marionette cross-identity samples from.")
    p.add_argument("--n_figures", type=int, default=1,
                   help="Number of figures to render in random mode (default 1).")
    p.add_argument("--seed",      type=int, default=None,
                   help="Optional seed for reproducible draws. Default: time-based.")
    p.add_argument("--sample_a",  default=None,
                   help="Pin block A's cross-identity sample_id (e.g. id_0001_id_0026).")
    p.add_argument("--sample_b",  default=None,
                   help="Pin block B's cross-identity sample_id (e.g. id_0077_id_0069).")
    p.add_argument("--dataset_for_picks", default=None, choices=ALL_DATASETS,
                   help="When using --sample_a / --sample_b, the dataset they "
                        "come from. Required in explicit mode if more than one "
                        "dataset is enabled.")
    p.add_argument("--out_root",  type=Path,
                   default=Path("outputs/paper_figures/teaser"),
                   help="Parent dir; the script writes to a timestamped "
                        "<out_root>/run_<YYYYmmdd_HHMMSS>/ subdir each invocation.")
    return p.parse_args()


def _validate_modes(args, pool: list[TeaserPick]) -> str:
    has_a = args.sample_a is not None
    has_b = args.sample_b is not None
    if has_a ^ has_b:
        raise ValueError(
            "Provide BOTH --sample_a and --sample_b for explicit mode, "
            "or neither for random mode."
        )
    if has_a and has_b:
        if args.n_figures != 1:
            print(f"[teaser] explicit pair given; ignoring --n_figures={args.n_figures}.")
        return "explicit"
    if len(pool) < 2:
        raise SystemExit(
            "Need at least 2 cross-identity samples in the pool. Check that "
            "Marionette eval ran cross_identity on the requested datasets, "
            "and that each sample has a panel.mp4 + scratch/source.png + "
            "scratch/driver.mp4 triple."
        )
    return "random"


def _pick_by_sample_id(pool: list[TeaserPick], sid: str,
                       dataset: Optional[str]) -> TeaserPick:
    matches = [pk for pk in pool if pk.sample_id == sid
               and (dataset is None or pk.dataset == dataset)]
    if not matches:
        raise SystemExit(
            f"sample_id {sid!r} (dataset={dataset!r}) not in the discovered pool. "
            f"Available examples: "
            f"{[(pk.dataset, pk.sample_id) for pk in pool[:5]]}..."
        )
    return matches[0]


def _safe_stem(s: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)
    return keep[:60]


def main():
    args = parse_args()

    pool = discover_picks(args.datasets)
    print(f"[teaser] cross-identity pool: {len(pool)}")

    mode = _validate_modes(args, pool)

    seed = args.seed if args.seed is not None else int(time.time())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = args.out_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[teaser] writing to {run_dir}")
    print(f"[teaser] seed={seed}  (re-runnable with --seed {seed})")

    picks_log: list[dict] = []
    n_failed = 0

    def _log_pick(idx: int, a: TeaserPick, b: TeaserPick,
                  out_path: Optional[Path], error: Optional[str]) -> None:
        entry = {
            "index":  idx,
            "block_a": {"dataset": a.dataset, "sample_id": a.sample_id,
                        "ref_uid": a.ref_uid, "drv_uid": a.drv_uid},
            "block_b": {"dataset": b.dataset, "sample_id": b.sample_id,
                        "ref_uid": b.ref_uid, "drv_uid": b.drv_uid},
            "out_png": str(out_path) if out_path is not None else None,
        }
        if error is not None:
            entry["error"] = error
        picks_log.append(entry)

    if mode == "explicit":
        if args.dataset_for_picks is None and len(args.datasets) > 1:
            raise SystemExit(
                "--dataset_for_picks is required in explicit mode when more "
                "than one dataset is enabled."
            )
        ds = args.dataset_for_picks or args.datasets[0]
        a = _pick_by_sample_id(pool, args.sample_a, ds)
        b = _pick_by_sample_id(pool, args.sample_b, ds)
        out_path = run_dir / f"a-{_safe_stem(a.sample_id)}__b-{_safe_stem(b.sample_id)}.png"
        try:
            render_figure(a, b, out_path)
            _log_pick(1, a, b, out_path, None)
            print(f"  rendered  a={a.sample_id}  b={b.sample_id}  → {out_path.name}")
        except Exception as e:
            n_failed += 1
            _log_pick(1, a, b, None, f"{type(e).__name__}: {e}")
            print(f"[teaser] FAILED: {type(e).__name__}: {e}")

    else:
        rng   = np.random.default_rng(seed)
        n_pad = max(3, len(str(args.n_figures)))

        for i in range(args.n_figures):
            # Two distinct picks per figure.
            i_a, i_b = rng.choice(len(pool), size=2, replace=False)
            a, b = pool[int(i_a)], pool[int(i_b)]
            out_path = run_dir / (
                f"{i+1:0{n_pad}d}_{a.dataset}-a-{_safe_stem(a.sample_id)}__"
                f"{b.dataset}-b-{_safe_stem(b.sample_id)}.png"
            )
            try:
                render_figure(a, b, out_path)
                _log_pick(i + 1, a, b, out_path, None)
                print(f"  rendered  [{a.dataset}] a={a.sample_id}  "
                      f"[{b.dataset}] b={b.sample_id}  → {out_path.name}")
            except Exception as e:
                n_failed += 1
                _log_pick(i + 1, a, b, None, f"{type(e).__name__}: {e}")
                print(f"[teaser] FAILED on a={a.sample_id} b={b.sample_id}: "
                      f"{type(e).__name__}: {e}")

    n_total = len(picks_log)
    n_ok    = sum(1 for p in picks_log if p.get("out_png"))
    manifest = {
        "seed":        seed,
        "seed_source": ("user --seed" if args.seed is not None else "wall-clock"),
        "timestamp":   timestamp,
        "mode":        mode,
        "protocol":    PROTOCOL,
        "n_requested": (1 if mode == "explicit" else args.n_figures),
        "n_rendered":  n_ok,
        "n_failed":    n_total - n_ok,
        "args": {
            "datasets": list(args.datasets),
            "out_root": str(args.out_root),
        },
        "picks":       picks_log,
    }
    manifest_path = run_dir / "_index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[teaser] done: {n_ok}/{(1 if mode == 'explicit' else args.n_figures)} "
          f"figures saved to {run_dir}/"
          + (f"  ({n_failed} failed)" if n_failed else ""))
    print(f"[teaser] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
