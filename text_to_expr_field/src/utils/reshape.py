"""Channel reshaping between expression field formats and pseudo-video."""

import torch


def to_pseudo_video(field: torch.Tensor) -> torch.Tensor:
    """
    Convert an expression field to a 3ch pseudo-video padded to 4k+1.

    Handles both:
      - 45ch: [T, 45, H, W] → reshape to [T*15, 3, H, W] → pad to 4k+1
      - 3ch:  [T, 3, H, W]  → already a video, just pad to 4k+1
    """
    T, C, H, W = field.shape
    if C == 45:
        pseudo = field.reshape(T, 15, 3, H, W).reshape(T * 15, 3, H, W)
    elif C == 3:
        pseudo = field
    else:
        raise ValueError(f"Expected 45 or 3 channels, got {C}")

    # Pad to 4k+1
    target = 1 + 4 * (pseudo.shape[0] // 4)
    if target < pseudo.shape[0]:
        target += 4
    while pseudo.shape[0] < target:
        pseudo = torch.cat([pseudo, pseudo[-1:]], dim=0)

    return pseudo


def pseudo_video_to_expr_field(pseudo_video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Reassemble a decoded pseudo-video back into a 45-channel expression field.

    Args:
        pseudo_video: [T*15 + 1, 3, H, W] or [T*15, 3, H, W]
        num_frames:   Number of original expression field frames

    Returns:
        [num_frames, 45, H, W] expression dense field
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
    return pseudo_video.reshape(num_frames, 15, 3, H, W).reshape(num_frames, 45, H, W)
