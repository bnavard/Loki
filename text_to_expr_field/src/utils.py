"""
Utility functions for the text-to-expression-field pipeline.

Handles reshaping between the 45-channel expression field format and
the 3-channel pseudo-video format used by the Wan2.2 VAE.
"""

import torch


def expr_field_to_pseudo_video(expr_field: torch.Tensor) -> torch.Tensor:
    """
    Reshape a 45-channel expression field into a pseudo-video for the VAE.

    The 45 channels are split into 15 groups of 3 channels each, then
    stacked along the temporal dimension. A padding frame is appended
    to satisfy the Wan2.2 VAE's 4k+1 frame requirement.

    Args:
        expr_field: [T, 45, H, W] expression dense field

    Returns:
        [T*15 + 1, 3, H, W] pseudo-video 
    """
    T, C, H, W = expr_field.shape
    assert C == 45, f"Expected 45 channels, got {C}"

    grouped = expr_field.reshape(T, 15, 3, H, W)
    pseudo = grouped.reshape(T * 15, 3, H, W)

    # Pad to 4k+1
    pad = pseudo[-1:].clone()
    pseudo = torch.cat([pseudo, pad], dim=0)

    return pseudo


def pseudo_video_to_expr_field(pseudo_video: torch.Tensor, num_frames: int = 16) -> torch.Tensor:
    """
    Reassemble a decoded pseudo-video back into a 45-channel expression field.

    Args:
        pseudo_video: [T*15 + 1, 3, H, W] or [T*15, 3, H, W]
        num_frames:   Number of original expression field frames (default 16)

    Returns:
        [T, 45, H, W] expression dense field
    """
    expected_with_pad = num_frames * 15 + 1
    expected_without_pad = num_frames * 15

    if pseudo_video.shape[0] == expected_with_pad:
        pseudo_video = pseudo_video[:expected_without_pad]
    elif pseudo_video.shape[0] != expected_without_pad:
        raise ValueError(
            f"Expected {expected_with_pad} or {expected_without_pad} frames, "
            f"got {pseudo_video.shape[0]}"
        )

    H, W = pseudo_video.shape[2], pseudo_video.shape[3]
    expr_field = pseudo_video.reshape(num_frames, 15, 3, H, W)
    expr_field = expr_field.reshape(num_frames, 45, H, W)

    return expr_field


def normalize_for_vae(expr_field: torch.Tensor) -> tuple:
    """
    Normalize expression field values to [-1, 1] for VAE input.

    Returns the normalized tensor and the (min, max) used for denormalization.
    """
    vmin = expr_field.min()
    vmax = expr_field.max()
    normalized = 2.0 * (expr_field - vmin) / (vmax - vmin + 1e-8) - 1.0
    return normalized, (vmin.item(), vmax.item())


def denormalize_from_vae(normalized: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """
    Reverse the normalization applied by normalize_for_vae.
    """
    return (normalized + 1.0) / 2.0 * (vmax - vmin) + vmin
