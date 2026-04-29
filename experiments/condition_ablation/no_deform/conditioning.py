"""
Conditioning module for the `no_deform` arm of condition_ablation.

Keeps the 42ch sinusoidal positional encoding of rasterized FLAME vertex
positions and REMOVES the 3ch per-vertex expression deformation map
entirely — no replacement channel. The UNet's ConditioningEncoder drops
from 45 input channels to 42.

Purpose: isolate the value of the expression deformation map. Identity is
already injected independently through the frozen reference UNet (K/V
feature injection into every self-attention block), so the conditioning
tensor does not need to carry identity information. Head pose and mesh
geometry are carried by the 42ch pos_enc; the deform map is the only
signal in the baseline's 45ch cond that encodes per-vertex expression
offsets. If this arm approaches baseline quality, the deform map is
informationally redundant on top of pos_enc + the ref UNet's identity
features. If this arm drops noticeably, the deform map is doing real work
for expression fidelity.

No natural driver video is substituted in place of the deform channels —
that experiment was considered and dropped because pasting driver-space
pixels in as conditioning conflates two variables (the deform ablation
and the spatial-misalignment confound). The `no_flame` arm covers the
"natural video as conditioning" comparison cleanly.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn

from marionette.conditioning.conditioning import PositionalEncoding
from marionette.conditioning.mesh2img import PropRenderer


class PosEncOnlyConditioning(nn.Module):
    """Emit `spatial_cond` = 42-channel sinusoidal pos_enc of rasterized
    FLAME vertex positions. Shape `(B, T, H, W, 42)`.

    Consumes `hint["driver_verts"]` (for pos_enc rasterization) and ignores
    `driver_deform`.
    """

    N_CHANNELS = 42

    # Viz contract: the 42 pos_enc channels are laid out as
    # [0:14] = x (sin×7, cos×7), [14:28] = y, [28:42] = z. Slice [21:24]
    # picks y's cos at the three lowest frequencies — a clean horizontal-
    # stripe signature on the rasterized mesh region that communicates "the
    # conditioning is a positional encoding of the face mesh" without
    # pretending to be a natural-image preview.
    VIZ_SLICE: tuple[int, int] = (21, 24)
    VIZ_LABEL: str = "Pos Enc"

    def __init__(
        self,
        image_size: int = 512,
        positional_channels: int = 42,
        positional_multiplier: float = 1.0,
        **_unused,
    ) -> None:
        # Absorb baseline params (std_expr_deformation) inherited through
        # YAML merge — they don't apply to a pos-enc-only rasterization.
        super().__init__()
        self.image_size = image_size
        self.positional_multiplier = positional_multiplier

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)
        # Eager construction — DDP broadcasts buffers at wrap time; lazy
        # registration leaves non-rank-0 ranks without the renderer's
        # buffers and silently hangs the first collective.
        self.renderer = PropRenderer()

    @property
    def n_conditioning_channels(self) -> int:
        return self.N_CHANNELS

    def forward(self, batch: dict) -> dict:
        device = self.pos_encoding.freqs.device
        with torch.no_grad():
            driver_verts = batch["driver_verts"].to(device)
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

            spatial_cond = einops.rearrange(
                pos_enc_feat, "(b t) h w c -> b t h w c", b=B,
            )
        return {"spatial_cond": spatial_cond}
