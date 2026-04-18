"""
Spatial conditioning producer for the talking-head UNet.

Three sources are supported:

  - `expression_source="gt"` (default): the 3ch expression deformation is
    rasterized from FLAME verts/offsets on the fly, alongside the 42ch
    sinusoidal positional encoding of vertex positions.

  - `expression_source="marigold"`: the 3ch expression deformation is taken
    directly from `batch["marigold_deform"]` (decoded from the cached
    `deformation.mp4` written by `scripts/cache/marigold_deform/cache.py`).
    FLAME rasterization is skipped entirely — the Marigold module only produces
    the deformation map, not the positional encoding channels — so the output
    is always 4 channels (3 deform + 1 ref_mask).

  - `expression_source="driving_video"`: the raw natural driving video frames,
    downsampled to latent resolution (64x64), are used as 3ch RGB spatial
    conditioning. No FLAME decomposition at all — the UNet sees the face
    appearance at low resolution instead of a structured deformation signal.
    Output is 4 channels (3 RGB + 1 ref_mask). This mode tests whether the
    FLAME decomposition adds value over plain video conditioning.

Channel layout by mode:

  source="gt"
  ├── default (full):        [0:42] pos_enc · [42:45] deform · [45] ref_mask    → 46
  ├── drop_expression_map:   [0] ref_mask                                        →  1
  └── expr_deform_only:      [0:3] deform · [3] ref_mask                         →  4

  source="marigold"
  └── always 4ch:            [0:3] marigold_deform · [3] ref_mask                →  4

  source="driving_video"
  └── always 4ch:            [0:3] RGB · [3] ref_mask                            →  4
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
    Produces spatial conditioning tensors for the talking-head diffusion model.

    Returns a dict containing:
        pos_enc          : (B, T, H, W, C_cond)  — UNet spatial conditioning
        expr_weight_map  : (B, T, H, W, Cw)      — signal used by expression-weighted
                                                    loss (46ch full map when
                                                    source="gt", 3ch deform when
                                                    source="marigold")
        z_input          : (B, T, 4, h, w) | None
        ref_mask         : (B, T, 1, H, W)
    """

    def __init__(
        self,
        image_size: int = 64,
        positional_channels: int = 42,
        positional_multiplier: float = 1.0,
        super_resolution: int = 2,
        use_expr_deformation: bool = True,
        std_expr_deformation: float = 0.0104,
        drop_expression_map: bool = False,   # ablation: UNet sees only ref_mask (1ch)
        expr_deform_only: bool = False,      # ablation: UNet sees 3ch deform + 1ch ref_mask (4ch)
        expression_source: str = "gt",       # "gt" | "marigold" | "driving_video"
    ) -> None:
        super().__init__()

        assert expression_source in ("gt", "marigold", "driving_video"), \
            f"expression_source must be 'gt', 'marigold', or 'driving_video', got {expression_source!r}"
        self.expression_source = expression_source

        self.image_size = image_size
        assert super_resolution >= 1 and super_resolution % 1 == 0
        self.super_resolution = super_resolution
        self.positional_channels = positional_channels
        self.positional_multiplier = positional_multiplier
        self.use_expr_deformation = use_expr_deformation
        self.std_expr_deformation = std_expr_deformation
        self.drop_expression_map = drop_expression_map
        self.expr_deform_only = expr_deform_only

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)
        self._renderer = None  # lazy-initialized on first conditional GT forward

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
        if self.expression_source in ("marigold", "driving_video"):
            return 4  # 3ch (deform or RGB) + 1ch ref_mask
        if self.expr_deform_only:
            return 4  # 3ch GT deform + 1ch ref_mask
        n = self.positional_channels + 1  # pos enc + ref mask
        if self.use_expr_deformation:
            n += 3
        return n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resize_deform_to_image_size(self, deform_btchw: torch.Tensor, B: int) -> torch.Tensor:
        """Resize a channels-first (B, T, 3, H_src, W_src) tensor to
        (B, T, image_size, image_size, 3) channels-last for concat with pos_enc."""
        deform_flat_bt_chw = einops.rearrange(deform_btchw, 'b t c h w -> (b t) c h w')
        deform_flat_bt_chw = F.interpolate(
            deform_flat_bt_chw, (self.image_size, self.image_size), mode="area"
        )
        return einops.rearrange(deform_flat_bt_chw, '(b t) c h w -> b t h w c', b=B)

    def _rasterize_gt(self, verts_flat, offsets_flat, B):
        """Run FLAME rasterization -> 45ch (pos_enc + deform) channels-last tensor.

        Returns: (B, T, image_size, image_size, 45) where the layout is
        [positional_encoding (42) | expression_deformation (3)].
        """
        img_size = self.image_size
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
        return einops.rearrange(pose_pos_enc, '(b t) c h w -> b t h w c', b=B)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict, unconditional: bool = False) -> dict:
        """
        Args:
            batch: dict containing at minimum:
                reference_mask  (B, T, 1, H, W) — 1 for reference frame slots
              when expression_source="gt":
                verts_2d        (B, T, V, 2)
                offsets_3d      (B, T, V, 3)
              when expression_source="marigold":
                marigold_deform (B, T, 3, H, W) — pre-generated deformation map
              when expression_source="driving_video":
                driving_video   (B, T, 3, H, W) — raw video frames at latent res, [-1, 1]
              optional:
                z               (B, T, 4, h, w) — pre-encoded VAE latents
            unconditional: if True return all-zero conditioning (for CFG).

        Returns:
            dict with keys: pos_enc, expr_weight_map, z_input, ref_mask
        """
        # reference_mask may arrive as (B, T, H, W) or (B, T, 1, H, W).
        rm = batch["reference_mask"]
        ref_mask = rm[:, :, None] if rm.ndim == 4 else rm  # → (B, T, 1, H, W)
        B, T = ref_mask.shape[:2]

        z_input = batch.get("z", None)

        img_size = self.image_size
        device = self.pos_encoding.freqs.device
        ref_mask = ref_mask.to(device)

        # Channels-last view of the reference mask for concatenation: (B, T, H, W, 1)
        ref_mask_last = einops.rearrange(ref_mask, 'b t c h w -> b t h w c')

        # -------------------------------------------------------------
        # Compute the full spatial conditioning tensor `pose_pos_enc`
        # -------------------------------------------------------------
        if unconditional:
            total_channels = self.n_conditioning_channels
            pose_pos_enc = torch.zeros(
                (B, T, img_size, img_size, total_channels), device=device
            )
            # expr_weight_map: match whatever the chosen source would produce.
            weight_ch = 46 if self.expression_source == "gt" else 3
            expr_weight_map = torch.zeros(
                (B, T, img_size, img_size, weight_ch), device=device
            )
            if z_input is not None:
                z_input = z_input * 0.0

        elif self.expression_source == "marigold":
            # Marigold produces only the 3ch deformation — no positional encoding.
            # UNet conditioning is always 4ch: [deform (3) | ref_mask (1)].
            marigold_deform = batch["marigold_deform"].to(device)            # (B, T, 3, H, W)
            deform_channels_last = self._resize_deform_to_image_size(
                marigold_deform, B
            )                                                                 # (B, T, H, W, 3)
            expr_weight_map = deform_channels_last                            # (B, T, H, W, 3)

            pose_pos_enc = torch.cat(
                [deform_channels_last, ref_mask_last], dim=-1
            )                                                                 # (B, T, H, W, 4)

        elif self.expression_source == "driving_video":
            # Raw driving video frames (already at latent resolution) as spatial
            # conditioning. Tests whether structured FLAME decomposition adds
            # value over simply showing the model the face at low resolution.
            # UNet conditioning is 4ch: [RGB (3) | ref_mask (1)].
            driving = batch["driving_video"].to(device)                       # (B, T, 3, H, W)
            driving_channels_last = self._resize_deform_to_image_size(
                driving, B
            )                                                                 # (B, T, H, W, 3)
            # No meaningful deformation-based weight map for driving video —
            # use the RGB frames themselves as the weight signal (uniform in
            # practice since there's no explicit deformation decomposition).
            expr_weight_map = driving_channels_last                           # (B, T, H, W, 3)

            pose_pos_enc = torch.cat(
                [driving_channels_last, ref_mask_last], dim=-1
            )                                                                 # (B, T, H, W, 4)

        else:
            # expression_source == "gt" — rasterize FLAME verts/offsets
            with torch.no_grad():
                verts    = batch["verts_2d"].to(device)
                offsets  = batch["offsets_3d"].to(device)

                verts_flat   = einops.rearrange(verts,   'b t n v -> (b t) n v')
                offsets_flat = einops.rearrange(offsets, 'b t n v -> (b t) n v')

                gt_45ch = self._rasterize_gt(
                    verts_flat, offsets_flat, B
                )                                                             # (B, T, H, W, 45)

                pose_pos_enc = torch.cat(
                    [gt_45ch, ref_mask_last], dim=-1
                )                                                             # (B, T, H, W, 46)

            # Full 46ch map is the weight signal for expression-weighted loss.
            expr_weight_map = pose_pos_enc

        # -------------------------------------------------------------
        # Apply ablation modes (drop / expr_deform_only)
        # -------------------------------------------------------------
        if self.drop_expression_map:
            # UNet sees only the reference mask (1 channel).
            pose_pos_enc = (
                torch.zeros_like(ref_mask_last) if unconditional else ref_mask_last
            )

        elif self.expr_deform_only and self.expression_source == "gt":
            # Replace the full 46ch with only [deform (3) | ref_mask (1)].
            # (In marigold mode this is already the natural output — no-op.)
            if unconditional:
                pose_pos_enc = torch.zeros(
                    B, T, img_size, img_size, 4, device=device
                )
            else:
                expr_deform = pose_pos_enc[
                    ..., self.positional_channels:self.positional_channels + 3
                ]
                pose_pos_enc = torch.cat([expr_deform, ref_mask_last], dim=-1)

        return {
            "pos_enc":         pose_pos_enc,     # (B, T, H, W, C_cond)
            "expr_weight_map": expr_weight_map,  # (B, T, H, W, 46 or 3)
            "z_input":         z_input,
            "ref_mask":        ref_mask,
        }

    def get_vis(self, enc: torch.Tensor) -> dict:
        """Return named slices of the full GT conditioning tensor for visualisation.

        Intended for use on the 46ch GT `expr_weight_map`. Not meaningful for the
        3ch Marigold weight map.
        """
        vis = {}
        n_pos = self.positional_channels // 3

        for i in range(n_pos - 2, n_pos):
            vis[f"pose_map_{i}"] = enc[..., [i, i + n_pos, i + n_pos * 2]]

        counter = self.positional_channels

        if self.use_expr_deformation:
            vis["expr_disp"] = enc[..., counter:counter + 3]
            counter += 3

        vis["ref_mask"] = enc[..., [counter] * 3]

        return vis