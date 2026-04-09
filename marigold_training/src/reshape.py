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

    target = 1 + 4 * (pseudo.shape[0] // 4)
    if target < pseudo.shape[0]:
        target += 4
    while pseudo.shape[0] < target:
        pseudo = torch.cat([pseudo, pseudo[-1:]], dim=0)

    return pseudo
