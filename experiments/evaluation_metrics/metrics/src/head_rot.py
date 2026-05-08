"""Head-orientation error — pose-trajectory mismatch between pred and target,
read directly off the FLAME tracking parameters.

For each clip we compose the visible head rotation from the FLAME fit::

    R_head[t] = R(rot[t]) · R(neck_rot[t])

(``rot`` is the global rigid rotation applied after LBS; ``neck_rot`` is the
neck-joint rotation applied within LBS — together they specify the head's
full visible orientation in the FLAME tracker's per-clip camera frame.)

Frame-0-anchored deltas
-----------------------

The metric compares **delta rotations** rather than absolute rotations::

    dR[t] = R_head[t] · R_head[0]^T

so that any constant per-clip offset between pred's and target's
independently-fit cameras drops out, and what we report is purely
pose-trajectory follow — the rotation each clip *induces* between t=0
and t. This also matches what portrait animation should be evaluated
on: the prediction's idle pose comes from the reference image, not the
driver, so absolute pose mismatch isn't a meaningful penalty.

Why frame-0 anchoring instead of inter-frame ``R[t] · R[t-1]^T``
---------------------------------------------------------------

An alternative would be inter-frame deltas — "how much the head rotated
this frame" — which also cancels constant offsets. We picked frame-0
anchoring (cumulative) deliberately:

* **Measures what we want.** Cumulative deltas penalise *trajectory
  divergence* — if pred drifts steadily off the driver's path, error
  grows over time. Inter-frame deltas let slow systematic drift go
  unpenalised: a pred adding 0.3° of yaw every frame would have
  inter-frame deltas matching the driver's at every step yet end the
  clip 4.8° off, which is exactly the failure mode portrait animation
  should not get a free pass on.
* **Better signal-to-noise.** At 25 fps the head moves ~1–2° per frame;
  the FLAME tracker carries ~0.5° per-frame noise. Inter-frame SNR is
  ~2–4×; cumulative deltas reach 10–20° over the 16-frame window so
  SNR is 20–40×.
* **Frame 0 is well-anchored.** For same-identity reconstruction, pred
  frame 0 ↔ GT frame 0 by construction. For cross-identity retargeting,
  pred frame 0 is generated from driver frame 0's motion, so the two
  are aligned at t=0 too. Not an arbitrary anchor.

Per-frame distance: geodesic via quaternion
-------------------------------------------

The two delta rotations are compared by the geodesic angular distance,
computed via the quaternion dot product::

    q_p = quat(dR_pred[t]);  q_t = quat(dR_target[t])
    θ[t] = 2 · arccos( clip( |q_p · q_t|, -1, 1 ) )

This is the smallest angle that takes one orientation to the other and is
the standard rotation-difference measure in robotics and graphics. We
deliberately do **not** report a per-axis (yaw / pitch / roll) breakdown:
direct Euler-angle comparison is unreliable — different conventions give
different numbers, wrapping at ±180° introduces discontinuities, and the
same rotation has multiple equivalent Euler triples near gimbal lock.
The single geodesic number above avoids all of those pitfalls.

Per-sample reduction is the mean of θ over the available frames
(frame 0 contributes 0 trivially).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def _head_R(rot_aa: np.ndarray, neck_aa: np.ndarray) -> np.ndarray:
    """Composition of FLAME global ``rot`` and ``neck_rot`` (both axis-angle)
    into a 3×3 rotation matrix. Frame doesn't matter for the metric (we
    only compare deltas to themselves), so we skip the OpenCV/pytorch3d
    basis change."""
    return (Rsc.from_rotvec(rot_aa).as_matrix()
            @ Rsc.from_rotvec(neck_aa).as_matrix())


def _quat_angular_dist_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angular distance in degrees between two rotations, via the
    quaternion dot product. Robust to the q/-q ambiguity via abs()."""
    q_a = Rsc.from_matrix(R_a).as_quat()
    q_b = Rsc.from_matrix(R_b).as_quat()
    d   = float(np.clip(abs(float(np.dot(q_a, q_b))), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(d)))


def compute_head_rot_pair(
    pred_fit: dict,
    target_fit: dict,
    n_frames: int,
) -> dict:
    """Per-frame geodesic angular distance between pred's and target's
    pose trajectories, averaged over ``n_frames`` (or the shorter of the
    two fit lengths). Returns ``None`` for ``dist`` when neither fit has
    at least 2 frames.

    Returns
    -------
    {
      "dist":       float | None,   # mean geodesic angular distance (degrees)
      "track_rate": float,          # T_used / n_frames in [0, 1]
      "n_frames":   int,
    }
    """
    T_pred   = pred_fit  ["expr"].shape[0]
    T_target = target_fit["expr"].shape[0]
    T = min(n_frames, T_pred, T_target)
    if T < 2:
        return {"dist": None, "track_rate": 0.0, "n_frames": T}

    neck_pred = (pred_fit  ["neck_rot"] if "neck_rot" in pred_fit
                 else np.zeros((T_pred,   3), dtype=np.float32))
    neck_tgt  = (target_fit["neck_rot"] if "neck_rot" in target_fit
                 else np.zeros((T_target, 3), dtype=np.float32))

    R_pred = [_head_R(pred_fit  ["rot"][t], neck_pred[t]) for t in range(T)]
    R_tgt  = [_head_R(target_fit["rot"][t], neck_tgt [t]) for t in range(T)]

    # Frame-0-anchored delta rotations. Cancels any constant per-clip
    # camera-fit offset between independently-tracked clips. See module
    # docstring for why this beats inter-frame deltas.
    R_pred_t0_inv = R_pred[0].T
    R_tgt_t0_inv  = R_tgt [0].T
    dR_pred = [R_pred[t] @ R_pred_t0_inv for t in range(T)]
    dR_tgt  = [R_tgt [t] @ R_tgt_t0_inv  for t in range(T)]

    geo = [_quat_angular_dist_deg(dR_pred[t], dR_tgt[t]) for t in range(T)]

    return {
        "dist":       float(np.mean(geo)),
        "track_rate": float(T) / float(max(n_frames, 1)),
        "n_frames":   T,
    }


def head_axes_in_image(rot_aa: np.ndarray, neck_aa: np.ndarray,
                       opencv2pytorch3d: np.ndarray) -> np.ndarray:
    """Return the 3 head-frame unit axes (X right, Y down, Z forward) as
    columns of a 3×3 matrix in the OpenCV image frame. Used by the
    visualization helper to draw axes consistent with how a viewer reads
    the rendered video.

    `opencv2pytorch3d` is the diag(1, -1, -1) basis change living at
    `loki.flame.flame.OPENCV2PYTORCH3D[:3, :3]`."""
    R_p3d = _head_R(rot_aa, neck_aa)
    M = opencv2pytorch3d
    return M @ R_p3d @ M    # M is its own inverse for diag(±1)
