"""
Conditioning module for the `no_posenc` arm of condition_ablation.

Drops the 42ch sinusoidal positional encoding of rasterized FLAME vertex
positions and keeps only the 3ch per-vertex expression deformation map.
Emits a 3-channel `spatial_cond` tensor.

Purpose: test how much of the FLAME conditioning's usefulness comes from
the positional encoding (which carries head pose + geometry) vs. the pure
expression deformation (which carries "which blendshape, how much" but no
global mesh location). Expected to underperform the full 45ch recipe because
the model loses all explicit head-pose / geometry cues from the conditioning
channel — ref image + deform-only is not enough to place the face in the
frame; the network would have to infer pose from the ref image alone.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn

from marionette.conditioning.mesh2img import PropRenderer


class DeformOnlyConditioning(nn.Module):
    """Rasterize only the 3-channel per-vertex expression deformation.

    Consumes `hint["driver_verts"]` (for the triangle rasterization target)
    and `hint["driver_deform"]` (the prop being rasterized). Ignores
    `driver_video`. Emits `spatial_cond` at `(B, T, H, W, 3)`.
    """

    N_CHANNELS = 3

    # Viz contract: spatial_cond is the rasterized deform map. Whole tensor
    # is the preview.
    VIZ_SLICE: tuple[int, int] = (0, 3)
    VIZ_LABEL: str = "Driver Deform"

    def __init__(
        self,
        image_size: int = 512,
        std_expr_deformation: float = 0.0104,
        **_unused,
    ) -> None:
        # Absorb baseline params (positional_channels, positional_multiplier)
        # inherited through YAML merge — they don't apply to a deform-only
        # rasterization.
        super().__init__()
        self.image_size = image_size
        self.std_expr_deformation = std_expr_deformation
        # Eager construction — DDP broadcasts buffers at wrap time; lazy
        # registration leaves non-rank-0 ranks without the renderer's buffers
        # and silently hangs the first collective.
        self.renderer = PropRenderer()
        self.register_buffer("_device_anchor", torch.zeros(1))

    @property
    def n_conditioning_channels(self) -> int:
        return self.N_CHANNELS

    def forward(self, batch: dict) -> dict:
        device = self._device_anchor.device
        with torch.no_grad():
            driver_verts  = batch["driver_verts"].to(device)
            driver_deform = batch["driver_deform"].to(device)
            B, T = driver_verts.shape[:2]

            verts_flat   = einops.rearrange(driver_verts,  "b t n v -> (b t) n v")
            offsets_flat = einops.rearrange(driver_deform, "b t n v -> (b t) n v")
            offsets_flat = offsets_flat / self.std_expr_deformation

            # `_` is the 3ch vert-position prop we are deliberately discarding
            # here — that is what distinguishes this arm from the canonical
            # SpatialConditioning.
            pose_map, mask = self.renderer.render(
                verts_flat, (self.image_size, self.image_size),
                prop=offsets_flat,
            )
            _, deform_map = pose_map.split([3, 3], dim=-1)
            deform_map = deform_map * mask

            spatial_cond = einops.rearrange(deform_map, "(b t) h w c -> b t h w c", b=B)
        return {"spatial_cond": spatial_cond}
