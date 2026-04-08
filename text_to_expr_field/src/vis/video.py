"""Low-level video writing and normalization primitives."""

from pathlib import Path

import cv2
import numpy as np


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float array symmetrically around zero to [0, 255] uint8."""
    abs_max = np.abs(arr).max() + 1e-8
    return ((arr / abs_max + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


def save_video(frames: list, path, fps: int):
    """Save a list of uint8 BGR frames as an mp4 video."""
    path = Path(path)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
