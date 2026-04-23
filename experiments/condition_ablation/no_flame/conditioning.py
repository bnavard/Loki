"""
Conditioning module for the `no_flame` arm of condition_ablation.

Replaces the full 45-channel FLAME spatial conditioning (42ch pos_enc + 3ch
driver_deform) with the driver's **natural face-cropped video** as the sole
spatial-conditioning signal. Emits a 3-channel `spatial_cond` tensor equal
to the raw driver-video pixels (normalized to [-1, 1]), bypassing the
pytorch3d mesh rasterizer entirely.

Purpose: show that FLAME-based conditioning — rasterized into the *reference's*
camera/crop space so it spatially aligns with the generation target — is
strictly more useful than feeding the driver's own pixels (which live in the
driver's pixel space and carry no cross-identity alignment guarantee).

Key asymmetry vs. the canonical `SpatialConditioning`:

  * No rasterization. The module is a pure-tensor pass-through plus layout
    swap — `driver_video` comes in as `(B, T, H, W, 3)` in [-1, 1] and is
    emitted under the same `spatial_cond` key the UNet's conditioning encoder
    already expects.
  * Spatial frame of reference is the DRIVER's face crop (the same recipe
    used by `marionette.retargeting.prepare_driver_frames`), NOT the ref's.
    Expected consequence: the conditioning signal and the denoising target
    do not line up pixel-for-pixel across identities, forcing the UNet to
    learn a softer "whatever you see, map it" mapping instead of the crisp
    aligned supervision FLAME provides.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NaturalVideoConditioning(nn.Module):
    """Emit the driver's face-cropped video as `spatial_cond`. 3 channels.

    Consumes `hint["driver_video"]` (shape `(B, T, H, W, 3)`, values in
    [-1, 1]) and ignores `driver_verts` / `driver_deform` if present. No
    trainable parameters — the module only moves the tensor to its buffer
    device and returns it under the conditioning API's single expected key.
    """

    N_CHANNELS = 3

    def __init__(self, image_size: int = 512, **_unused) -> None:
        # `**_unused` absorbs params that the baseline SpatialConditioning
        # takes (positional_channels, positional_multiplier, …) so YAML merges
        # that inherit the base config's cond_stage_config.params don't have
        # to explicitly null them — the arm can just override `target`.
        super().__init__()
        self.image_size = image_size
        # Device-tracking buffer: `nn.Module` has no intrinsic device, and the
        # conditioning forward needs to land the output on the same device as
        # the UNet. Matching SpatialConditioning's `freqs`-as-anchor pattern.
        self.register_buffer("_device_anchor", torch.zeros(1))

    @property
    def n_conditioning_channels(self) -> int:
        return self.N_CHANNELS

    def forward(self, batch: dict) -> dict:
        with torch.no_grad():
            driver_video = batch["driver_video"].to(self._device_anchor.device)
            _, _, H, W, _ = driver_video.shape
            if H != self.image_size or W != self.image_size:
                raise ValueError(
                    f"NaturalVideoConditioning expects driver_video at "
                    f"{self.image_size}x{self.image_size}; got {H}x{W}."
                )
        return {"spatial_cond": driver_video}
