"""
Spatial conditioning producer for the talking-head UNet.

Always produces a 49-channel conditioning tensor (`spatial_cond`) at full image
resolution:

    [0:42]  pos_enc      — 42ch sinusoidal positional encoding of FLAME vertex
                           positions (rasterized into the target's pixel space).
    [42:45] deform       — 3ch per-vertex expression deformation offsets
                           (rasterized alongside pos_enc in one render pass).
    [45:48] warped_ref   — 3ch backward-warped reference image: the reference
                           frame pulled into the target's pose + expression via
                           a grid_sample over the rasterized UV lookup map.
    [48]    ref_mask     — 1ch indicator (1 on the reference slot, 0 elsewhere).

The warp is identity-preserving where the FLAME mesh explains the pixels (skin,
jaw, cheeks, forehead) and has characteristic artifacts where it doesn't (eye
interior, mouth interior, glasses, hair). The UNet learns to inpaint the
artifact regions; the warp gives it a strong identity prior for the rest.

For same-identity training the driver's FLAME verts ARE the reference's verts
(same clip), so the UV lookup pulls clean reference pixels into themselves
across expressions and poses. For cross-identity inference the driver's motion
(ψ, θ) is applied to the reference's shape β under the reference's camera
(retargeting happens in `generate.py`) — the rasterizer still runs on per-frame
verts, and the warp stays a consistent ref→target pull.

CFG's "null" conditioning is built by zero-filling the output of this module
in the training loop (`MarionetteDiffusion.get_input`), not here. This module only
computes the real conditioning.
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
    """Produce the 49-channel spatial conditioning tensor consumed by the UNet's
    ConditioningEncoder.

    Inputs (via `batch` dict, typically `batch_outer["hint"]`):
        driver_verts   : (B, T, V, 3) driver's FLAME verts in target-camera NDC
        driver_deform  : (B, T, V, 3) per-vertex expression deformation
        ref_mask       : (B, T, 1, h_lat, w_lat) or (B, T, h_lat, w_lat)
        ref_image      : (B, 3, H, W) reference frame in [-1, 1] (static across T)
        ref_verts      : (B, V, 3) reference's FLAME verts in NDC (static across T)
        z              : (B, T, 4, h_lat, w_lat) GT latents (training only), optional.

    Returns dict with keys: spatial_cond, z_input, ref_mask.
    """

    N_CHANNELS = 49

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
        ref_xy_flat: torch.Tensor,   # (B*T, V, 2) reference NDC verts
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one fused pytorch3d rasterization pass over the target mesh, with
        prop = [offsets(3) | ref_xy(2)] appended to the default vert-position prop.

        Returns channels-last tensors at `image_size`:
            pos_enc_input : (B*T, H, W, 3)  rasterized vert positions
            deform_map    : (B*T, H, W, 3)  rasterized expression offsets
            uv_map        : (B*T, H, W, 2)  rasterized reference NDC coords
            mask          : (B*T, H, W, 1)  on-mesh indicator
        """
        img_size = self.image_size
        offsets_flat = offsets_flat / self.std_expr_deformation

        fused_prop = torch.cat([offsets_flat, ref_xy_flat], dim=-1)
        pose_map, mask = self.renderer.render(
            verts_flat, (img_size, img_size), prop=fused_prop,
        )
        pos_enc_input, deform_map, uv_map = pose_map.split([3, 3, 2], dim=-1)
        return pos_enc_input, deform_map, uv_map, mask

    def _warp_reference(
        self,
        ref_image_flat: torch.Tensor,   # (B*T, 3, H, W) reference image, [-1, 1]
        uv_map: torch.Tensor,           # (B*T, H, W, 2) rasterized ref NDC coords
    ) -> torch.Tensor:
        """Backward-warp the reference image into the target's pose + expression.

        Flips both axes to convert pytorch3d NDC (+x=left, +y=up) to grid_sample
        NDC (+x=right, +y=down). `padding_mode="border"` so off-mesh regions
        sample the nearest ref pixel rather than zeroing — the on-mesh `mask`
        is applied separately by the caller.

        Returns channels-last (B*T, H, W, 3).
        """
        warp_grid = -uv_map
        warped = F.grid_sample(
            ref_image_flat, warp_grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        return einops.rearrange(warped, "bt c h w -> bt h w c")

    def forward(self, batch: dict) -> dict:
        rm = batch["ref_mask"]
        ref_mask = rm[:, :, None] if rm.ndim == 4 else rm
        B, T = ref_mask.shape[:2]

        z_input = batch.get("z", None)
        device = self.pos_encoding.freqs.device
        ref_mask = ref_mask.to(device)

        img_size = self.image_size
        rm_chw = einops.rearrange(ref_mask, "b t c h w -> (b t) c h w")
        if rm_chw.shape[-1] != img_size:
            rm_chw = F.interpolate(rm_chw.float(), (img_size, img_size), mode="nearest")
        ref_mask_last = einops.rearrange(rm_chw, "(b t) c h w -> b t h w c", b=B)

        with torch.no_grad():
            driver_verts  = batch["driver_verts"].to(device)
            driver_deform = batch["driver_deform"].to(device)
            ref_verts     = batch["ref_verts"].to(device)
            ref_image     = batch["ref_image"].to(device)

            verts_flat     = einops.rearrange(driver_verts,  "b t n v -> (b t) n v")
            offsets_flat   = einops.rearrange(driver_deform, "b t n v -> (b t) n v")
            ref_xy_flat    = einops.repeat(ref_verts[..., :2], "b n v -> (b t) n v", t=T)
            ref_image_flat = einops.repeat(ref_image, "b c h w -> (b t) c h w", t=T)

            pos_enc_input, deform_map, uv_map, mask = self._rasterize_conditioning(
                verts_flat, offsets_flat, ref_xy_flat,
            )
            pos_enc_feat = self.pos_encoding(pos_enc_input * self.positional_multiplier)
            warped_ref   = self._warp_reference(ref_image_flat, uv_map)

            pos_enc_feat = pos_enc_feat * mask
            deform_map   = deform_map   * mask
            warped_ref   = warped_ref   * mask

            stacked = torch.cat([pos_enc_feat, deform_map, warped_ref], dim=-1)
            stacked = einops.rearrange(stacked, "(b t) h w c -> b t h w c", b=B)

        spatial_cond = torch.cat([stacked, ref_mask_last], dim=-1)
        return {"spatial_cond": spatial_cond, "z_input": z_input, "ref_mask": ref_mask}
