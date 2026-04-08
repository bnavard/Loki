"""Visualization for expression fields and deformation maps."""

from pathlib import Path

import numpy as np
import torch

from text_to_expr_field.src.vis.video import normalize_to_uint8, save_video


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


def visualize_expr_field(expr_field: torch.Tensor, output_dir, fps: int, verbose: bool = True):
    """
    Save visualization videos for all 45 expression field channels.

    The 45-channel expression field is organized as:
      - Channels 0-41: sinusoidal positional encoding (14 frequency bands x 3 xyz)
      - Channels 42-44: 3D deformation map (expression-driven vertex offsets)

    Saves:
      - deformation.mp4: 3-channel deformation map as RGB
      - pos_enc_band_{i:02d}.mp4: each positional encoding frequency band
      - combined.mp4: 5x3 grid video with all 15 channel groups
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T = expr_field.shape[0]
    H, W = expr_field.shape[2], expr_field.shape[3]
    expr_np = expr_field.numpy()

    # Deformation map (channels 42-44)
    deform_frames = []
    for t in range(T):
        deform = expr_np[t, 42:45].transpose(1, 2, 0)
        vis = normalize_to_uint8(deform)
        deform_frames.append(vis[..., ::-1])
    save_video(deform_frames, output_dir / "deformation.mp4", fps)

    # Positional encoding bands (14 bands x 3ch)
    for band_idx in range(14):
        ch_start = band_idx * 3
        band_frames = []
        for t in range(T):
            band = expr_np[t, ch_start:ch_start + 3].transpose(1, 2, 0)
            vis = normalize_to_uint8(band)
            band_frames.append(vis[..., ::-1])
        save_video(band_frames, output_dir / f"pos_enc_band_{band_idx:02d}.mp4", fps)

    # Combined grid: 5 columns x 3 rows
    cols, rows = 5, 3
    grid_h, grid_w = rows * H, cols * W

    combined_frames = []
    for t in range(T):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for group_idx in range(15):
            ch_start = group_idx * 3
            ch_data = expr_np[t, ch_start:ch_start + 3].transpose(1, 2, 0)
            vis = normalize_to_uint8(ch_data)
            row = group_idx // cols
            col = group_idx % cols
            y0, x0 = row * H, col * W
            grid[y0:y0 + H, x0:x0 + W] = vis[..., ::-1]
        combined_frames.append(grid)
    save_video(combined_frames, output_dir / "combined.mp4", fps)

    if verbose:
        print(f"  Videos saved to {output_dir}/")
