"""Expression-error metric — pose-disentangled FLAME deformation-map L1.

Compares the rasterized expression-deformation map of a pred FLAME fit
against a target fit (= GT fit for same-identity reconstruction; = driver
fit for cross-identity retargeting), with pose, identity shape, and
camera held to the target's so the residual is attributable to expression
alone.

Mechanism: render the target's fit verbatim, and render a substituted fit
where pred's ``(expr, eye_rot, jaw_rot)`` are inserted into the target's
``(rot, tra, neck_rot, shape, camera)``. Both renders land on the same
image-space pixels by construction — identical mesh layout — so the
per-pixel difference between the two deformation maps is purely an
expression difference.

Pose error is measured separately by the head-rot metric (geodesic
angular distance over FLAME `rot · neck_rot`), so we deliberately do
not try to capture pose mismatch here.

L1 reduction: per-pixel mean-absolute-deviation across the 3 deform
channels, then mean over on-mesh pixels (mask-aware — background pixels
would otherwise dilute the score). L1 is preferred over L2 here because
its units are directly the average per-component deformation residual,
which is more interpretable than an RMSE-across-channels.
"""
from __future__ import annotations

import numpy as np
import torch

from loki.conditioning.conditioning import SpatialConditioning
from loki.flame.flame import FlameSkinnerExtended, compute_flame
from loki.utils import get_bbox_from_verts, verts_to_pytorch3d

HEAD_VERT_PATH = "data/assets/flame/head_vertices.txt"


def _per_frame_fit(fit: dict, t: int, cam_id: int = 0) -> dict:
    """Slice a single-frame fit dict from a multi-frame fit.npz."""
    item = {
        "shape":   fit["shape"],
        "expr":    fit["expr"][[t]],
        "rot":     fit["rot"][[t]],
        "tra":     fit["tra"][[t]],
        "eye_rot": fit["eye_rot"][[t]],
        "fx":      fit["fx"][[cam_id]],
        "fy":      fit["fy"][[cam_id]],
        "cx":      fit["cx"][[cam_id]],
        "cy":      fit["cy"][[cam_id]],
        "extr":    fit["extr"][[cam_id]],
    }
    if "neck_rot" in fit:
        item["neck_rot"] = fit["neck_rot"][[t]]
    if "jaw_rot" in fit:
        item["jaw_rot"] = fit["jaw_rot"][[t]]
    return item


def _swap_expression(target_item: dict, pred_item: dict) -> dict:
    """Return target_item with its (expr, eye_rot, jaw_rot) replaced by
    pred_item's. Pose / shape / camera / neck_rot stay target's, so the
    rendered mesh shares target's image-space layout — only expression
    coefficients can drive a residual."""
    out = {**target_item}
    out["expr"]    = pred_item["expr"]
    out["eye_rot"] = pred_item["eye_rot"]
    if "jaw_rot" in pred_item:
        out["jaw_rot"] = pred_item["jaw_rot"]
    return out


class ExpressionDeformationDiff:
    """Render the pose-substituted FLAME deformation map for a pred fit and
    compare it against the target's deformation map via mask-aware L2.

    Stateful: loads the FLAME skinner (CPU) + SpatialConditioning rasterizer
    (GPU) once on construction.
    """

    def __init__(self, image_size: int = 512, device: str = "cuda"):
        self.image_size = image_size
        self.device     = device

        # Skinner stays on CPU (matches video_dataset usage; compute_flame
        # returns CPU tensors). Only the rasterizer needs GPU.
        self.skinner = FlameSkinnerExtended(
            add_mouth=True, n_shape_params=150, n_expr_params=65,
        ).eval()
        self.cond = SpatialConditioning(image_size=image_size).to(device).eval()
        self.head_vert_ids = np.genfromtxt(HEAD_VERT_PATH).astype(int)

    @torch.no_grad()
    def _rasterize(self, item: dict, crop_box) -> tuple[np.ndarray, np.ndarray]:
        """compute_flame → NDC verts in `crop_box`'s frame → rasterizer.

        Returns (deform_map, mask) with shapes `(H, W, 3)` and `(H, W, 1)`.
        """
        out = compute_flame(self.skinner, item)
        verts_2d = out["verts_2d"][0, 0].copy()
        offsets  = out["offsets_3d"][0]
        verts_ndc = verts_to_pytorch3d(verts_2d, np.array(crop_box))

        verts_t   = torch.from_numpy(verts_ndc).float()[None].to(self.device)
        offsets_t = torch.from_numpy(offsets).float()[None].to(self.device)

        _, deform_map, mask = self.cond._rasterize_conditioning(verts_t, offsets_t)
        return deform_map.cpu().numpy()[0], mask.cpu().numpy()[0]

    @staticmethod
    def _masked_l1(pred_def: np.ndarray, tgt_def: np.ndarray,
                   mask_bool: np.ndarray) -> float:
        """Mean-absolute-deviation across the 3 deform channels per pixel,
        then mean over the on-mesh pixels (mask=True). Returns 0.0 if the
        mask is empty."""
        per_pixel = np.abs(pred_def - tgt_def).mean(axis=-1)
        if not mask_bool.any():
            return 0.0
        return float(per_pixel[mask_bool].mean())

    @torch.no_grad()
    def compute_pair(
        self,
        pred_fit: dict,
        target_fit: dict,
        n_frames: int,
    ) -> dict:
        """Compute the deformation-map L1 over the first `n_frames` of the
        pred / target pair.

        Returns ``{"l1": float, "n_frames": int}``.
        """
        T = min(n_frames, pred_fit["expr"].shape[0], target_fit["expr"].shape[0])
        l1s: list[float] = []

        for t in range(T):
            item_target = _per_frame_fit(target_fit, t)
            item_pred   = _per_frame_fit(pred_fit, t)
            item_swap   = _swap_expression(item_target, item_pred)

            verts_target = compute_flame(self.skinner, item_target)["verts_2d"][0, 0]
            crop_target  = get_bbox_from_verts(verts_target.copy(), self.head_vert_ids)

            def_target, mask_target = self._rasterize(item_target, crop_target)
            def_swap,   _           = self._rasterize(item_swap,   crop_target)

            l1s.append(self._masked_l1(def_swap, def_target,
                                       mask_target[..., 0] > 0))

        return {
            "l1":       float(np.mean(l1s)) if l1s else 0.0,
            "n_frames": T,
        }
