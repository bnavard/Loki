"""Head-orientation error — yaw / pitch / roll mismatch between pred and
target frames.

Naming note: we deliberately call this "head orientation" rather than
"head pose" to avoid collision with FLAME's `θ` (jaw + neck + head pose
parameters). FLAME pose is part of the expression-evaluation pipeline;
this metric is purely the rigid head rotation extracted by 6DRepNet.

Backbone is **6DRepNet** (Hempel et al. 2022 — `300W-LP → AFLW2000`
weights), wrapped via the `sixdrepnet` PyPI package. The package
auto-downloads its checkpoint to `~/.cache/torch/hub/` on first
construct (~150 MB); `setup_env.sh` step 3b2 triggers that download
ahead of time so the first metrics run doesn't block on it.

Per-frame extraction returns `(yaw, pitch, roll)` in degrees from a
face-cropped RGB uint8 image. The same `face_crop_around_detection`
routine used by every other metric in this package is applied to both
pred and target frames before pose extraction — pose models are
sensitive to crop framing and won't behave like their training
distribution if you feed them raw, unaligned video frames.

Per-frame error is **L1 (absolute) in degrees**, NOT squared. Reporting
in degrees is directly interpretable ("our model is off by 4° on
average in yaw"), whereas deg² values require mental sqrt. Per-sample
reduction is the arithmetic mean of per-frame absolute errors over the
frames where pose was extracted on **both** sides. Frames where either
side's RetinaFace pre-crop or 6DRepNet pose extraction failed are
skipped — `head_orientation_detect_rate` is reported alongside,
mirroring the LMD / id_cosine pattern.

This module also re-exports `sixdrepnet.utils.draw_axis` for the
side-by-side visualizer in `sanity_check/visualize_head_orientation.py`.
Drawing the rotated XYZ basis on pred and target faces is the most
defensible way to eyeball whether the numerical error matches what's
visually happening, especially for catastrophic head-orientation drift
on a few samples that would otherwise pull the aggregate around.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _make_estimator(device: str = "cuda"):
    """Construct the SixDRepNet model on `device`. Lazily-imported so the
    module can be `import head_orientation` cheaply (the model load is a
    few seconds + a one-time 150 MB download on first call)."""
    from sixdrepnet import SixDRepNet
    if device.startswith("cuda"):
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    else:
        gpu_id = -1
    try:
        return SixDRepNet(gpu_id=gpu_id)
    except TypeError:
        # Older sixdrepnet releases don't expose gpu_id; fall back.
        return SixDRepNet()


class HeadOrientationEstimator:
    """Thin wrapper around `sixdrepnet.SixDRepNet` exposing a per-frame
    extractor + per-pair L1 errors.

    One instance per process — like our other CV-network wrappers,
    construction loads weights and shouldn't be done in a tight loop.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model = _make_estimator(device)

    def extract(self, face_uint8: np.ndarray) -> Optional[tuple[float, float, float]]:
        """`face_uint8`: (H, W, 3) RGB uint8 face crop. Returns
        `(yaw, pitch, roll)` in degrees, or None if pose extraction
        fails. The 6DRepNet model itself doesn't have a "no-detection"
        signal — it always returns numbers — but a None gate here lets
        future detectors plug in without changing the metric loop."""
        try:
            pitch, yaw, roll = self._model.predict(face_uint8)
        except Exception:
            return None
        # `predict` returns 1-element ndarrays.
        return float(yaw[0]), float(pitch[0]), float(roll[0])

    @staticmethod
    def axis_l1_errors(pred: tuple[float, float, float],
                       target: tuple[float, float, float]) -> tuple[float, float, float]:
        """Per-axis absolute error in **degrees** for `(yaw, pitch, roll)`.
        6DRepNet's outputs are in `[-90°, +90°]` for each axis, so wrapping
        isn't needed for typical talking-head clips — frontal faces stay
        well within range."""
        return (
            abs(pred[0] - target[0]),
            abs(pred[1] - target[1]),
            abs(pred[2] - target[2]),
        )


# Re-export the canonical 6DRepNet axis-drawing helper. Used by
# `sanity_check/visualize_head_orientation.py`. Imported lazily because
# importing `sixdrepnet` triggers torch / cv2 / opencv-python init,
# which we don't want to pay for code paths that only need
# `axis_l1_errors`.
def draw_axis(img: np.ndarray, yaw: float, pitch: float, roll: float,
              tdx: float | None = None, tdy: float | None = None,
              size: float = 100.0) -> np.ndarray:
    """Overlay the rotated XYZ basis (red = X-right, green = Y-up,
    blue = Z-forward) at `(tdx, tdy)` on the image. Wraps
    `sixdrepnet.utils.draw_axis`. The wrapper writes in-place; we copy
    the buffer so callers aren't forced to."""
    from sixdrepnet.utils import draw_axis as _draw
    out = img.copy()
    _draw(out, yaw, pitch, roll, tdx=tdx, tdy=tdy, size=size)
    return out
