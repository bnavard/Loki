"""
Spatial conditioning producer for the talking-head UNet.

Produces a 45-channel `spatial_cond` tensor at full image resolution:

    [0:42]  pos_enc  — 42ch sinusoidal positional encoding of rasterized FLAME
                       vertex positions (in target-camera NDC).
    [42:45] deform   — 3ch per-vertex expression deformation offsets, rasterized
                       alongside pos_enc in one render pass.

Identity information does not ride on spatial_cond — it flows through the
frozen reference UNet (`RefFeatureExtractor`), whose per-layer self-attention
inputs are injected as additional K/V tokens into the generation UNet. The
reference never occupies a slot in the gen tensor, so no ref-slot indicator
is needed here.

For same-identity training the rasterizer runs on target-window FLAME verts;
for cross-identity inference the driver's motion is retargeted onto the ref's
shape and camera (`marionette.retargeting`), and the same rasterization path
is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from marionette.conditioning.mesh2img import PropRenderer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding applied to 3-D vertex map coordinates."""

    def __init__(self, channels_per_dim: int):
        super().__init__()
        assert channels_per_dim % 2 == 0
        n_ch = channels_per_dim // 2
        freqs = 2.0 ** torch.linspace(0.0, n_ch - 1, steps=n_ch)
        self.register_buffer("freqs", freqs)

    def forward(self, tensor):
        """
        Args:
            tensor: (B, H, W, 3)  — normalised [x, y, z] coordinates
        Returns:
            (B, H, W, channels_per_dim * 3)
        """
        if tensor.ndim != 4:
            raise RuntimeError("Input tensor must be 4-D (B, H, W, C)")
        pos_xyz = tensor[..., None] * self.freqs[None, None, None, None]
        pos_emb = torch.cat([torch.sin(pos_xyz), torch.cos(pos_xyz)], dim=-1)
        return einops.rearrange(pos_emb, "b h w c f -> b h w (c f)")


class SpatialConditioning(nn.Module):
    """Produce the 45-channel FLAME conditioning tensor consumed by the UNet's
    ConditioningEncoder.

    Inputs (via `batch` dict, typically `batch_outer["hint"]`):
        driver_verts   : (B, T, V, 3) driver's FLAME verts in target-camera NDC
        driver_deform  : (B, T, V, 3) per-vertex expression deformation

    Returns dict with a single key: spatial_cond  — (B, T, H, W, 45).
    """

    N_CHANNELS = 45

    def __init__(
        self,
        image_size: int = 512,
        positional_channels: int = 42,
        positional_multiplier: float = 1.0,
        std_expr_deformation: float = 0.0104,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.positional_channels = positional_channels
        self.positional_multiplier = positional_multiplier
        self.std_expr_deformation = std_expr_deformation

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)

        # Eager construction. DDP's broadcast_module_states enumerates buffers
        # at wrap time; lazy registration (@property / first-call init) would
        # leave non-rank-0 ranks without the renderer's buffers and silently
        # hang the first collective. Always register in __init__.
        self.renderer = PropRenderer()

    @property
    def n_conditioning_channels(self) -> int:
        return self.N_CHANNELS

    def _rasterize_conditioning(
        self,
        verts_flat: torch.Tensor,    # (B*T, V, 3) NDC verts in target-camera space
        offsets_flat: torch.Tensor,  # (B*T, V, 3) per-vertex deformation offsets
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one pytorch3d rasterization pass over the target mesh with the
        deformation offsets appended to the default vert-position prop.

        Returns channels-last tensors at `image_size`:
            pos_enc_input : (B*T, H, W, 3)  rasterized vert positions
            deform_map    : (B*T, H, W, 3)  rasterized expression offsets
            mask          : (B*T, H, W, 1)  on-mesh indicator
        """
        img_size = self.image_size
        offsets_flat = offsets_flat / self.std_expr_deformation

        pose_map, mask = self.renderer.render(
            verts_flat, (img_size, img_size), prop=offsets_flat,
        )
        pos_enc_input, deform_map = pose_map.split([3, 3], dim=-1)
        return pos_enc_input, deform_map, mask

    def forward(self, batch: dict) -> dict:
        device = self.pos_encoding.freqs.device

        with torch.no_grad():
            driver_verts  = batch["driver_verts"].to(device)
            driver_deform = batch["driver_deform"].to(device)
            B, T = driver_verts.shape[:2]

            verts_flat   = einops.rearrange(driver_verts,  "b t n v -> (b t) n v")
            offsets_flat = einops.rearrange(driver_deform, "b t n v -> (b t) n v")

            pos_enc_input, deform_map, mask = self._rasterize_conditioning(
                verts_flat, offsets_flat,
            )
            pos_enc_feat = self.pos_encoding(pos_enc_input * self.positional_multiplier)

            pos_enc_feat = pos_enc_feat * mask
            deform_map   = deform_map   * mask

            spatial_cond = torch.cat([pos_enc_feat, deform_map], dim=-1)
            spatial_cond = einops.rearrange(spatial_cond, "(b t) h w c -> b t h w c", b=B)

        return {"spatial_cond": spatial_cond}
