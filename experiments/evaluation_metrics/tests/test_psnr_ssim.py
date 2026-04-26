"""PSNR / SSIM cross-check vs `skimage.metrics`.

Both are well-defined mathematical operations — torchmetrics and skimage
should agree to within numerical tolerance on identical inputs. If they
ever diverge that's a real bug (data-range mismatch, channel ordering,
SSIM kernel parameters), so the test is tight at `1e-3`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

skimage = pytest.importorskip("skimage.metrics")

from experiments.evaluation_metrics.metrics.psnr import psnr_video
from experiments.evaluation_metrics.metrics.ssim import ssim_video


def _random_video_pair(B: int = 2, T: int = 4, H: int = 64, W: int = 64,
                       seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    pred = rng.random((B, T, 3, H, W)).astype(np.float32)
    ref  = rng.random((B, T, 3, H, W)).astype(np.float32)
    return torch.from_numpy(pred), torch.from_numpy(ref)


def test_psnr_matches_skimage():
    pred, ref = _random_video_pair()
    ours = psnr_video(pred, ref)

    expected = []
    for b in range(pred.shape[0]):
        per_frame = []
        for t in range(pred.shape[1]):
            per_frame.append(skimage.peak_signal_noise_ratio(
                ref[b, t].permute(1, 2, 0).numpy(),
                pred[b, t].permute(1, 2, 0).numpy(),
                data_range=1.0,
            ))
        expected.append(float(np.mean(per_frame)))
    expected_t = torch.tensor(expected, dtype=ours.dtype)
    assert torch.allclose(ours, expected_t, atol=1e-3), (ours, expected_t)


def test_ssim_matches_skimage():
    pred, ref = _random_video_pair()
    ours = ssim_video(pred, ref)

    expected = []
    for b in range(pred.shape[0]):
        per_frame = []
        for t in range(pred.shape[1]):
            per_frame.append(skimage.structural_similarity(
                ref[b, t].permute(1, 2, 0).numpy(),
                pred[b, t].permute(1, 2, 0).numpy(),
                data_range=1.0,
                channel_axis=-1,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            ))
        expected.append(float(np.mean(per_frame)))
    expected_t = torch.tensor(expected, dtype=ours.dtype)
    # SSIM kernel-edge handling differs slightly between skimage and
    # torchmetrics — 5e-3 is a comfortable headroom.
    assert torch.allclose(ours, expected_t, atol=5e-3), (ours, expected_t)


def test_shape_mismatch_raises():
    pred = torch.zeros(1, 2, 3, 16, 16)
    ref  = torch.zeros(1, 3, 3, 16, 16)
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr_video(pred, ref)
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim_video(pred, ref)
