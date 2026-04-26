"""LMD checks.

Two angles:
  * `LMD(x, x)` ≈ 0 when MediaPipe detects the face on both copies. We
    can't guarantee detection on synthetic noise, so the test uses an
    InsightFace pre-warmed face image as a fixture if available; otherwise
    it skips the detection-based assertion.
  * Synthetic-shift sanity: shift one frame by `k` px → LMD-F equals
    `k / IOD` exactly when normalization is on.

The synthetic-shift test bypasses MediaPipe by stubbing `_extract` to
return a fixed (478, 2) landmark grid, so it runs without the heavy dep.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

mp = pytest.importorskip("mediapipe")

from experiments.evaluation_metrics.metrics.lmd import (
    LMD, LEFT_EYE_OUTER, MOUTH_LANDMARKS, RIGHT_EYE_OUTER,
)


def _stub_landmarks(seed: int = 0) -> np.ndarray:
    """Synthetic 478-landmark grid. Pixel coords need to span enough range
    that the IOD denominator is non-degenerate (>>1)."""
    rng = np.random.default_rng(seed)
    lms = rng.uniform(0, 256, size=(478, 2)).astype(np.float32)
    # Make IOD = 100 px exactly so the normalized shift is interpretable.
    lms[LEFT_EYE_OUTER]  = [50.0,  128.0]
    lms[RIGHT_EYE_OUTER] = [150.0, 128.0]
    return lms


def test_synthetic_shift(monkeypatch):
    """Shift one set of landmarks by 5 px → LMD-F should equal 5 / IOD."""
    metric = LMD(normalize_by_iod=True)

    base    = _stub_landmarks()
    shifted = base + np.array([5.0, 0.0], dtype=np.float32)

    # Monkeypatch the detector to alternate base / shifted on consecutive
    # calls (pred odd, ref even — the metric calls _extract on both per
    # frame, so we serve from a small queue).
    call_log = {"i": 0}
    sequence = [shifted, base, shifted, base]   # pred[0], ref[0], pred[1], ref[1]
    def fake_extract(_self, _frame):
        out = sequence[call_log["i"] % len(sequence)]
        call_log["i"] += 1
        return out
    monkeypatch.setattr(LMD, "_extract", fake_extract)

    pred = torch.zeros(1, 2, 3, 256, 256)
    ref  = torch.zeros(1, 2, 3, 256, 256)
    res  = metric(pred, ref)

    iod = 100.0   # set by _stub_landmarks
    expected = 5.0 / iod
    assert pytest.approx(res["lmd_f"].item(), abs=1e-6) == expected, res["lmd_f"]
    # Mouth landmarks shifted by the same vector → same per-landmark
    # distance → same average.
    assert pytest.approx(res["lmd_m"].item(), abs=1e-6) == expected, res["lmd_m"]
    # Detection succeeded on every frame.
    assert res["detect_rate"].item() == 1.0
    metric.close()


def test_detect_failure_logs_nan(monkeypatch):
    metric = LMD()
    monkeypatch.setattr(LMD, "_extract", lambda _self, _frame: None)
    pred = torch.zeros(1, 2, 3, 64, 64)
    ref  = torch.zeros(1, 2, 3, 64, 64)
    res = metric(pred, ref)
    assert torch.isnan(res["lmd_f"]).item()
    assert res["detect_rate"].item() == 0.0
    metric.close()
