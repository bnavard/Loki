"""LPIPS identity sanity: `LPIPS(x, x)` should be ≈ 0 for any video.

The actual value is bounded by floating-point noise inside the network,
not zero, so the threshold is `1e-4`."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("lpips")

from experiments.evaluation_metrics.metrics.lpips_metric import LPIPSMetric


@pytest.fixture(scope="module")
def metric():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LPIPSMetric(net="alex", device=device, chunk_size=8)


def test_lpips_identity(metric):
    torch.manual_seed(0)
    x = torch.rand(1, 4, 3, 64, 64)   # (B, T, 3, H, W) in [0, 1]
    out = metric(x, x)
    assert out.shape == (1,)
    assert out.item() < 1e-4, f"expected ≈ 0, got {out.item()}"


def test_lpips_nonzero_on_difference(metric):
    torch.manual_seed(0)
    x = torch.rand(1, 4, 3, 64, 64)
    y = torch.rand(1, 4, 3, 64, 64)
    out = metric(x, y)
    assert out.shape == (1,)
    # On uniformly-random images the LPIPS distance is well above noise.
    assert out.item() > 0.1, f"expected > 0.1, got {out.item()}"
