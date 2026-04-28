"""Structural Similarity Index — frame-level, averaged over time per video.

Wang et al. 2004 settings (11×11 Gaussian, σ=1.5, K1=0.01, K2=0.03) — the
field implicitly assumes these defaults. Don't change them; results would
stop being comparable to published numbers.
"""
from __future__ import annotations

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def ssim_video(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred, ref: `(B, T, 3, H, W)` in `[0, 1]`.
    Returns:
        `(B,)` mean SSIM over frames per video.
    """
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs ref {tuple(ref.shape)}")
    B, T = pred.shape[:2]
    pred_flat = pred.reshape(B * T, *pred.shape[2:])
    ref_flat  = ref.reshape(B * T,  *ref.shape[2:])
    ssim_per_frame = structural_similarity_index_measure(
        pred_flat, ref_flat,
        data_range=1.0,
        gaussian_kernel=True, sigma=1.5, kernel_size=11,
        k1=0.01, k2=0.03,
        reduction="none",
    )  # (B*T,)
    return ssim_per_frame.view(B, T).mean(dim=1)
