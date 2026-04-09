"""Visualization for deformation maps."""

from pathlib import Path

import cv2
import numpy as np
import torch


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


def visualize_deform(deform_field: torch.Tensor, output_dir, fps: int, verbose: bool = True):
    """
    Save deformation map as a video.

    Args:
        deform_field: [T, 3, H, W] deformation tensor
        output_dir:   directory to save deformation.mp4
        fps:          video frame rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T = deform_field.shape[0]
    deform_np = deform_field.numpy()

    frames = []
    for t in range(T):
        deform = deform_np[t].transpose(1, 2, 0)
        vis = normalize_to_uint8(deform)
        frames.append(vis[..., ::-1])
    save_video(frames, output_dir / "deformation.mp4", fps)

    if verbose:
        print(f"  Video saved to {output_dir}/deformation.mp4")
