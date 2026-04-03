"""
Utility functions for the text-to-expression-field pipeline.

Handles reshaping between the 45-channel expression field format and
the 3-channel pseudo-video format used by the Wan2.2 VAE.
"""

import torch


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