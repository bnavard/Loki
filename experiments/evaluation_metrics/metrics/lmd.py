"""Landmark Mean Distance — paired metric vs ground-truth video.

Reports two variants:
  * **LMD-F**: mean over all 478 face landmarks — penalizes pose/expression
    mismatch as well as lip motion. Useful when GT pose should be matched.
  * **LMD-M**: mean over the 22 lip-region landmarks (outer + inner lips,
    matching MediaPipe's canonical FaceMesh topology) — proxy for lip-sync
    quality.

Detector is **MediaPipe Tasks `FaceLandmarker`** (the v2 model with
`output_face_blendshapes` available, same 478-landmark topology as the
legacy attention-mesh FaceMesh that the older `mp.solutions.face_mesh`
API exposed). The legacy `mp.solutions` namespace was dropped in
mediapipe 0.10.x; this module uses the Tasks API and downloads the
`.task` model bundle once via `setup_env.sh`.

Per-frame distances are normalized by the **inter-ocular distance**
(MediaPipe landmarks 33 ↔ 263) measured on the **GT** frame, so the
metric is scale-invariant across resolutions. State this normalization
explicitly in any reported result; LMD comparisons across detectors or
normalization schemes are not meaningful.

Failure mode: if the detector fails on either pred or GT for a given
frame, that frame contributes nothing — the running average over hits is
what the final number reflects. `detect_rate` (hits / T) is reported
alongside; if it drops below ~0.95 the LMD numbers are biased toward
easier frames and should be flagged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Lip landmarks under MediaPipe's canonical FaceMesh topology.
# Outer lip (11 points) + inner lip (11 points) = 22.
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,   # outer
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,    # inner
]
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263


# Default cache location for the `.task` model bundle. Setup_env.sh writes
# it here; LMD reads it from the same path. Override with `LMD(model_path=...)`.
DEFAULT_MODEL_PATH = Path("data/weights/mediapipe/face_landmarker_v2_with_blendshapes.task")


class LMD:
    """One MediaPipe `FaceLandmarker` per process — the underlying graph
    is single-threaded and not safe to share across workers (silent
    corruption). Construct one per worker if parallelizing.

    The landmarker is lazily instantiated on first call so unit tests can
    patch `_extract` without touching the real model file.
    """

    def __init__(
        self,
        normalize_by_iod: bool = True,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self.normalize_by_iod = normalize_by_iod
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._landmarker = None   # lazily built

    # ---------------- backend (lazy) ----------------

    def _ensure_landmarker(self):
        if self._landmarker is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"FaceLandmarker model bundle not found at {self.model_path}. "
                f"Run `bash experiments/evaluation_metrics/setup_env.sh` to download it, "
                f"or pass `model_path=` to `LMD(...)`."
            )
        # Heavy imports stay inside the lazy block — keeps unit tests that
        # patch `_extract` runnable on hosts without mediapipe Tasks set up.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options                     = base_options,
            running_mode                     = mp_vision.RunningMode.IMAGE,
            num_faces                        = 1,
            min_face_detection_confidence    = 0.5,
            min_face_presence_confidence     = 0.5,
            output_face_blendshapes          = False,
            output_facial_transformation_matrixes = False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def _extract(self, frame_uint8: np.ndarray) -> np.ndarray | None:
        """`frame_uint8`: (H, W, 3) RGB uint8. Returns (478, 2) pixel coords or None."""
        import mediapipe as mp
        self._ensure_landmarker()
        H, W, _ = frame_uint8.shape
        # Tasks API expects an `mp.Image` wrapping a numpy buffer with
        # explicit format. SRGB matches our RGB float→uint8 pipeline.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_uint8)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        return np.array([[p.x * W, p.y * H] for p in lms], dtype=np.float32)

    # ---------------- public API ----------------

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pred, ref: `(B, T, 3, H, W)` in `[0, 1]`.
        Returns:
            dict with keys `lmd_f`, `lmd_m`, `detect_rate`, each `(B,)`.
            NaN entries mean the detector failed on every frame for that video.
        """
        if pred.shape != ref.shape:
            raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs ref {tuple(ref.shape)}")
        B, T = pred.shape[:2]

        out_f  = torch.full((B,), float("nan"))
        out_m  = torch.full((B,), float("nan"))
        detect = torch.zeros(B)

        for b in range(B):
            f_dists, m_dists, hits = [], [], 0
            for t in range(T):
                p = (pred[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                r = (ref [b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                lp = self._extract(p)
                lr = self._extract(r)
                if lp is None or lr is None:
                    continue
                hits += 1
                if self.normalize_by_iod:
                    iod = float(np.linalg.norm(lr[LEFT_EYE_OUTER] - lr[RIGHT_EYE_OUTER])) + 1e-8
                else:
                    iod = 1.0
                d = np.linalg.norm(lp - lr, axis=1) / iod  # (478,)
                f_dists.append(float(d.mean()))
                m_dists.append(float(d[MOUTH_LANDMARKS].mean()))
            if hits > 0:
                out_f[b]  = float(np.mean(f_dists))
                out_m[b]  = float(np.mean(m_dists))
                detect[b] = hits / T

        return {"lmd_f": out_f, "lmd_m": out_m, "detect_rate": detect}

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
