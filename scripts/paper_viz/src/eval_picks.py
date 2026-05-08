"""Shared helpers for paper figures that read from Loki / SOTA eval runs.

Used by `build_comparison_figure.py` and `paper_teaser_figure.py`. These
helpers walk `outputs/loki_eval/...` (and its sibling
`outputs/sota_comparison/...`) to discover available samples and to load
strided uint8 frames from `panel.mp4` / `driver.mp4`.

Not a standalone script — invoked via:
    from src.eval_picks import ...
inside the figure-rendering scripts in `scripts/paper_viz/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Path / sampling constants
# ---------------------------------------------------------------------------

LOKI_ROOT = Path("outputs/loki_eval")

# Frames sampled per row (1st, 4th, 8th, 16th of Loki's 16-frame window).
SAMPLED_FRAME_INDICES = [0, 3, 7, 15]
N_FRAMES_PER_ROW      = len(SAMPLED_FRAME_INDICES)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def latest_run(parent: Path) -> Optional[Path]:
    """Return the most recent `run_*` subdir of `parent`, or None."""
    runs = sorted([d for d in parent.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def split_sample_id(sample_id: str, protocol: str = "cross_identity") -> tuple[str, str]:
    """Recover (ref_uid, drv_uid) from a sample_id. Sample IDs are
    constructed by the eval harness:
      * `same_identity_reconstruction` → ref == drv == sample_id (no `_id_`)
      * `cross_identity`               → `<ref_uid>_id_<drv_uid_tail>`
                                         e.g. `id_0001_id_0026`.
    """
    if protocol == "same_identity_reconstruction":
        return sample_id, sample_id
    if "_id_" not in sample_id:
        raise ValueError(f"cross-id sample_id missing `_id_` separator: {sample_id}")
    ref, drv_tail = sample_id.split("_id_", 1)
    return ref, f"id_{drv_tail}"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def video_to_uint8_frames(video_tensor) -> np.ndarray:
    """`(T, 3, H, W)` float32 in `[0, 1]` → `(T, H, W, 3)` uint8."""
    return (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def stride_indices(T: int) -> list[int]:
    """`SAMPLED_FRAME_INDICES`, clamped to `T-1` for short clips."""
    if T <= 0:
        return []
    return [min(i, T - 1) for i in SAMPLED_FRAME_INDICES]
