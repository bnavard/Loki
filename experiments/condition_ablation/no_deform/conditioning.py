"""
Conditioning module for the `no_deform` arm of condition_ablation.

Keeps the 42ch sinusoidal positional encoding of rasterized FLAME vertex
positions but **replaces** the 3ch per-vertex expression deformation with
the 3ch natural driver video. Total output width remains 45 channels, so
the ConditioningEncoder's first conv is unchanged vs. the baseline — the
only difference is *what* occupies the last three channels.

Purpose: isolate the value of the deformation map. The pos_enc alone gives
the model head pose + mesh geometry (coarse motion), but the deformation map
is what encodes per-vertex expression offsets (mouth-shape micro-motion,
blink, etc.). Replacing it with driver pixels lets the model see the driver's
face but in the DRIVER's spatial frame — spatially misaligned with the
reference identity whose face is being denoised. If the model can still hit
baseline quality, pos_enc is doing the heavy lifting; if quality drops, the
aligned deformation channel matters.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn

from marionette.conditioning.conditioning import PositionalEncoding
from marionette.conditioning.mesh2img import PropRenderer


class PosEncPlusVideoConditioning(nn.Module):
    """Emit `spatial_cond = [pos_enc_42ch, driver_video_3ch]` concatenated
    along the channel axis. Shape `(B, T, H, W, 45)`.

    Consumes `hint["driver_verts"]` (for pos_enc rasterization) and
    `hint["driver_video"]` (the replacement for the deform channels).
    Ignores `driver_deform`.
    """

    N_CHANNELS = 45

    def __init__(
        self,
        image_size: int = 512,
        positional_channels: int = 42,
        positional_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.positional_multiplier = positional_multiplier

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)
        self.renderer = PropRenderer()

    @property
    def n_conditioning_channels(self) -> int:
        return self.N_CHANNELS

    def forward(self, batch: dict) -> dict:
        device = self.pos_encoding.freqs.device
        with torch.no_grad():
            driver_verts = batch["driver_verts"].to(device)
            driver_video = batch["driver_video"].to(device)
            B, T = driver_verts.shape[:2]

            verts_flat = einops.rearrange(driver_verts, "b t n v -> (b t) n v")

            # `prop=None` → PropRenderer emits only its built-in 3-channel
            # vert-position prop; we skip the extra per-vertex offset prop
            # that the canonical SpatialConditioning would use here.
            pos_enc_input, mask = self.renderer.render(
                verts_flat, (self.image_size, self.image_size),
            )
            pos_enc_feat = self.pos_encoding(pos_enc_input * self.positional_multiplier)
            pos_enc_feat = pos_enc_feat * mask
            pos_enc_feat = einops.rearrange(
                pos_enc_feat, "(b t) h w c -> b t h w c", b=B,
            )

            # driver_video is already (B, T, H, W, 3), in [-1, 1], aligned to
            # the driver's own face crop. It is NOT masked to the FLAME mesh
            # — the whole frame is valid conditioning signal.
            spatial_cond = torch.cat([pos_enc_feat, driver_video], dim=-1)
        return {"spatial_cond": spatial_cond}
