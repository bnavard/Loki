"""Peak Signal-to-Noise Ratio — frame-level, averaged over time per video.

Thin wrapper around `torchmetrics.functional.image.peak_signal_noise_ratio`.
`data_range=1.0` because the IO convention is `[0, 1]` float32.
"""
from __future__ import annotations

import torch
from torchmetrics.functional.image import peak_signal_noise_ratio


def psnr_video(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred, ref: `(B, T, 3, H, W)` in `[0, 1]`.
    Returns:
        `(B,)` mean PSNR over frames per video, in dB.
    """
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs ref {tuple(ref.shape)}")
    B, T = pred.shape[:2]
    pred_flat = pred.reshape(B * T, *pred.shape[2:])
    ref_flat  = ref.reshape(B * T,  *ref.shape[2:])
    psnr_per_frame = peak_signal_noise_ratio(
        pred_flat, ref_flat, data_range=1.0, reduction="none", dim=(1, 2, 3),
    )  # (B*T,)
    return psnr_per_frame.view(B, T).mean(dim=1)
