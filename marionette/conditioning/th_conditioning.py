"""
FLAME → spatial conditioning for the talking-head UNet.

Converts raw FLAME mesh parameters into dense spatial conditioning tensors by
rasterizing vertices onto a 2D grid via PyTorch3D and applying sinusoidal
positional encoding.

Channel layout (default: 46 channels):
  [0  :42] Sinusoidal Fourier positional encoding of 3D vertex positions
  [42 :45] Expression deformation (per-vertex Δx,Δy,Δz from neutral)
  [45]     Reference mask (1 = reference frame, 0 = frame to generate)

Modes:
  - Full (46ch): default, all spatial information
  - drop_expression_map=True (1ch): only ref_mask, for ablating FLAME conditioning
  - expr_deform_only=True (4ch): only deformation + ref_mask, drops positional encoding

The full 46ch expression field is always computed internally (stored as
expr_weight_map) so that expression-weighted loss can use it regardless of
which mode the UNet sees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

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


class THConditioning(nn.Module):
    """
    Produces spatial conditioning tensors from FLAME mesh data for the
    talking-head diffusion model.

    The returned dictionary always contains:
        pos_enc  : (B, T, H, W, C_cond)  — full conditioning feature map
        z_input  : (B, T, 4, H/8, W/8)   — VAE latents (reference frames only,
                                            zeros elsewhere); None if not provided
        ref_mask : (B, T, 1, H/8, W/8)   — 1 for reference frame slots
    """

    def __init__(
        self,
        image_size: int = 64,
        positional_channels: int = 42,
        positional_multiplier: float = 1.0,
        super_resolution: int = 2,
        use_ray_directions: bool = False,
        use_expr_deformation: bool = True,
        use_crop_mask: bool = False,
        std_expr_deformation: float = 0.0104,
        drop_expression_map: bool = False,   # ablation: removes all FLAME spatial conditioning,
                                              # UNet sees only ref_mask (1 channel)
        expr_deform_only: bool = False,      # experiment: replaces the full 46ch conditioning with
                                              # only the 3ch rasterized expression deformation +
                                              # 1ch ref_mask (4 channels total). Drops the 42ch
                                              # positional encoding of vertex positions, keeping
                                              # only the visual heatmap of face deformation.
    ) -> None:
        super().__init__()

        self.image_size = image_size
        assert super_resolution >= 1 and super_resolution % 1 == 0
        self.super_resolution = super_resolution
        self.positional_channels = positional_channels
        self.positional_multiplier = positional_multiplier
        self.use_ray_directions = use_ray_directions
        self.use_expr_deformation = use_expr_deformation
        self.std_expr_deformation = std_expr_deformation
        self.use_crop_mask = use_crop_mask
        self.drop_expression_map = drop_expression_map
        self.expr_deform_only = expr_deform_only

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)
        self._renderer = None  # lazy-initialized on first conditional forward

    @property
    def renderer(self):
        """Lazy-init PropRenderer so importing THConditioning never crashes
        if pytorch3d is missing or built against a different PyTorch version."""
        if self._renderer is None:
            from marionette.conditioning.mesh2img import PropRenderer
            device = self.pos_encoding.freqs.device
            self._renderer = PropRenderer().to(device)
        return self._renderer

    # ------------------------------------------------------------------
    # Channel count helper (useful for config validation)
    # ------------------------------------------------------------------
    @property
    def n_conditioning_channels(self) -> int:
        if self.drop_expression_map:
            return 1  # only ref_mask
        if self.expr_deform_only:
            return 4  # 3ch expression deformation + 1ch ref_mask
        n = self.positional_channels + 1  # pos enc + ref mask
        if self.use_expr_deformation:
            n += 3
        if self.use_ray_directions:
            n += 3
        if self.use_crop_mask:
            n += 1
        return n

    def forward(self, batch: dict, unconditional: bool = False) -> dict:
        """
        Args:
            batch: dict containing at minimum:
                verts_2d       (B, T, V, 2)  — projected 2-D vertices
                offsets_3d     (B, T, V, 3)  — expression deformation offsets
                reference_mask (B, T, 1, H, W) — 1 for reference frame slots
              optionally:
                ray_map        (B, T, 3, H, W) — camera ray directions
                out_crop_mask  (B, T, H, W)    — valid-crop mask
                z              (B, T, 4, h, w) — pre-encoded VAE latents
            unconditional: if True return all-zero conditioning (for CFG).

        Returns:
            dict with keys: pos_enc, z_input, ref_mask
        """
        verts    = batch["verts_2d"]          # (B, T, V, 2)
        offsets  = batch["offsets_3d"]        # (B, T, V, 3)
        # reference_mask may arrive as (B, T, H, W) or (B, T, 1, H, W).
        # Normalise to (B, T, 1, H, W) for downstream use.
        rm = batch["reference_mask"]
        ref_mask = rm[:, :, None] if rm.ndim == 4 else rm  # → (B, T, 1, H, W)
        B, T = verts.shape[:2]

        z_input = batch.get("z", None)

        img_size = self.image_size

        # Always compute the full expression map (needed for loss weighting
        # even when drop_expression_map=True for the UNet conditioning).
        # The full map is stored under "expr_weight_map" in the output dict.

        if unconditional:
            total_channels = self.n_conditioning_channels
            pose_pos_enc = torch.zeros(
                (B, T, img_size, img_size, total_channels), device=verts.device
            )
            if z_input is not None:
                z_input = z_input * 0.0
        else:
            with torch.no_grad():
                # Ensure tensors are on the same device as this module
                device = self.pos_encoding.freqs.device
                verts   = verts.to(device)
                offsets = offsets.to(device)
                ref_mask = ref_mask.to(device)

                verts_flat   = einops.rearrange(verts,   'b t n v -> (b t) n v')
                offsets_flat = einops.rearrange(offsets, 'b t n v -> (b t) n v')
                offsets_flat = offsets_flat / self.std_expr_deformation

                pose_map, mask = self.renderer.render(
                    verts_flat,
                    (img_size * self.super_resolution, img_size * self.super_resolution),
                    prop=offsets_flat if self.use_expr_deformation else None,
                )

                if self.use_expr_deformation:
                    pose_map, expr_offsets = pose_map.split([3, 3], dim=-1)

                pose_pos_enc = self.pos_encoding(pose_map * self.positional_multiplier)

                if self.use_expr_deformation:
                    pose_pos_enc = torch.cat([pose_pos_enc, expr_offsets], dim=-1)

                pose_pos_enc = pose_pos_enc * mask

                # Downscale if super-resolution was used
                pose_pos_enc = einops.rearrange(pose_pos_enc, 'bt h w c -> bt c h w')
                pose_pos_enc = F.interpolate(pose_pos_enc, (img_size, img_size), mode="area")
                pose_pos_enc = einops.rearrange(pose_pos_enc, '(b t) c h w -> b t h w c', b=B)

                if self.use_ray_directions:
                    ray_map = batch["ray_map"]                        # (B, T, 3, H, W)
                    ray_map = einops.rearrange(ray_map, 'b t c h w -> b t h w c')
                    pose_pos_enc = torch.cat([pose_pos_enc, ray_map], dim=-1)

                # Reference mask
                ref_mask_reshape = einops.rearrange(ref_mask, 'b t c h w -> b t h w c')
                pose_pos_enc = torch.cat([pose_pos_enc, ref_mask_reshape], dim=-1)

                if self.use_crop_mask:
                    crop_mask = batch["out_crop_mask"][..., None]     # (B, T, H, W, 1)
                    pose_pos_enc = torch.cat([pose_pos_enc, crop_mask], dim=-1)

        # Store the full expression map separately. This is always computed
        # regardless of drop_expression_map or expr_deform_only, so that
        # expression-weighted loss can optionally use it even when the UNet
        # receives reduced conditioning.
        expr_weight_map = pose_pos_enc  # (B, T, H, W, 46)

        # When ablating expression maps from UNet conditioning: the model
        # receives only the reference mask (1 channel) as spatial conditioning.
        # Audio cross-attention is unaffected.
        if self.drop_expression_map:
            ref_mask_reshape = einops.rearrange(ref_mask, 'b t c h w -> b t h w c')
            if unconditional:
                pose_pos_enc = torch.zeros_like(ref_mask_reshape)
            else:
                pose_pos_enc = ref_mask_reshape

        # Experiment: condition the UNet with only the 3ch rasterized expression
        # deformation heatmap + 1ch reference mask (4 channels total), dropping
        # the 42ch positional encoding of vertex positions.
        elif self.expr_deform_only:
            ref_mask_reshape = einops.rearrange(ref_mask, 'b t c h w -> b t h w c')
            if unconditional:
                pose_pos_enc = torch.zeros(
                    B, T, img_size, img_size, 4, device=ref_mask.device
                )
            else:
                expr_deform = pose_pos_enc[..., self.positional_channels:self.positional_channels + 3]
                pose_pos_enc = torch.cat([expr_deform, ref_mask_reshape], dim=-1)

        return {
            "pos_enc":         pose_pos_enc,     # (B, T, H, W, C_cond) or (B, T, H, W, 1)
            "expr_weight_map": expr_weight_map,  # (B, T, H, W, 46) — always available
            "z_input":         z_input,
            "ref_mask":        ref_mask,
        }

    def get_vis(self, enc: torch.Tensor) -> dict:
        """Return named slices of the conditioning tensor for visualisation."""
        vis = {}
        n_pos = self.positional_channels // 3

        for i in range(n_pos - 2, n_pos):
            vis[f"pose_map_{i}"] = enc[..., [i, i + n_pos, i + n_pos * 2]]

        counter = self.positional_channels

        if self.use_expr_deformation:
            vis["expr_disp"] = enc[..., counter:counter + 3]
            counter += 3

        if self.use_ray_directions:
            vis["ray_map"] = enc[..., counter:counter + 3]
            counter += 3

        vis["ref_mask"] = enc[..., [counter] * 3]
        counter += 1

        if self.use_crop_mask:
            vis["crop_mask"] = enc[..., [counter] * 3]

        return vis
