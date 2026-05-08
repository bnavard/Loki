"""
SD 2.1 UNet extended for talking-head video generation.

Two conditioning pathways into the UNet:

  * Spatial FLAME conditioning (`spatial_cond`, 45 channels: 42ch pos_enc +
    3ch driver_deform) enters through a learned `ConditioningEncoder` — a
    small SD-style conv stack that downsamples the full-resolution cond
    tensor to the UNet's latent resolution and emits `model_channels`
    feature maps that are added once to the first UNet feature map. The
    encoder's final Conv3×3 is zero-initialised so conditioning contributes
    zero at step 0.

  * Reference identity rides in via per-layer self-attention K/V injection.
    `forward` accepts a `ref_features` list, one `(B, HW_k, D_k)` tensor per
    self-attention block in forward-pass order, produced by a frozen
    `RefFeatureExtractor`. Every gen token's query attends to both its own
    tokens and the ref tokens at the same resolution. This is the
    ReferenceNet / AnimateAnyone pattern.

Input channels stay at 4 (SD 2.1's pretrained spec). Every gen slot runs the
full denoising path; identity flows exclusively through the ref-attention
injection.
"""

import torch
import einops

from ldm_base.ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm_base.ldm.modules.diffusionmodules.util import timestep_embedding

from loki.model.attention import SpatioTemporalTransformer
from loki.model.conditioning_encoder import ConditioningEncoder


class LokiUNet(UNetModel):
    """SD 2.1 UNet extended with 3D spatiotemporal attention, FLAME spatial
    conditioning, and reference-attention K/V injection for identity
    preservation."""

    def __init__(
        self,
        *args,
        time_steps: int,
        condition_channels: int = 45,
        model_channels: int = 320,
        image_size: int = 32,
        context_dim: int = 768,
        temporal_mode: str = "3d",
        cond_input_resolution: int = 512,
        cond_latent_resolution: int = 64,
        cond_stage_channels: tuple | list | None = None,
        **kwargs,
    ):
        assert temporal_mode in ["3d", "temporal"]
        self.temporal_mode = temporal_mode
        self.time_steps = time_steps

        super().__init__(
            *args,
            model_channels=model_channels,
            image_size=image_size,
            context_dim=context_dim,
            **kwargs,
        )

        self.cond_encoder = ConditioningEncoder(
            in_channels=condition_channels,
            model_channels=model_channels,
            input_resolution=cond_input_resolution,
            output_resolution=cond_latent_resolution,
            stage_channels=(
                tuple(cond_stage_channels) if cond_stage_channels is not None
                else None
            ),
        )

        # Cache the ordered list of SpatioTemporalTransformer blocks so the
        # forward pass can distribute `ref_features` to them by index without
        # walking the nested block structure each time.
        self._attn_blocks = [
            m for m in self.modules() if isinstance(m, SpatioTemporalTransformer)
        ]

    def create_attention_block(
        self,
        ch,
        mult,
        use_checkpoint,
        num_heads,
        dim_head,
        transformer_depth,
        context_dim,
        disable_self_attn,
        use_linear,
        use_new_attention_order,
        use_spatial_transformer,
    ):
        temporal_connection_type = (
            "temporal" if self.temporal_mode == "temporal"
            else ("3d" if mult >= 2 else "none")
        )
        return SpatioTemporalTransformer(
            ch, num_heads, dim_head,
            temporal_connection_type=temporal_connection_type,
            num_timesteps=self.time_steps,
        )

    def forward(self, x, timesteps=None, control=None,
                ref_features: list[torch.Tensor] | None = None, **kwargs):
        """
        Args:
            x            : (B, T, 4, H, W) noisy latents.
            timesteps    : (B, T) diffusion timesteps per frame.
            control      : dict with `spatial_cond` (B,T,H,W,45).
            ref_features : list of (B, HW_k, D_k) tensors captured from the
                           frozen reference UNet, one per self-attention
                           block in forward-pass order. Same length as
                           `self._attn_blocks`. Pass None to disable ref
                           injection (baseline behavior).
        """
        b_, _ = x.shape[:2]
        x_flat    = einops.rearrange(x,         "b t c h w -> (b t) c h w")
        timesteps = einops.rearrange(timesteps, "b t      -> (b t)")

        # Distribute ref_features to the matching attention blocks ahead of
        # the forward walk. The attribute is read by our patched block forward
        # (see below) and cleared after to avoid leaking state across calls.
        self._install_ref_features(ref_features)

        spatial_cond = einops.rearrange(control["spatial_cond"], "b t h w c -> (b t) c h w")
        cond_feature = self.cond_encoder(spatial_cond.type(self.dtype))

        try:
            hs = []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(self.dtype)
            emb = self.time_embed(t_emb)
            h = x_flat.type(self.dtype)

            for module in self.input_blocks:
                h = module(h, emb)
                if cond_feature is not None:
                    h += cond_feature
                    cond_feature = None
                hs.append(h)

            h = self.middle_block(h, emb)

            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb)

            h = self.out(h).type(x_flat.dtype)
        finally:
            self._clear_ref_features()

        return einops.rearrange(h, "(b t) c h w -> b t c h w", b=b_)

    def _install_ref_features(self, ref_features):
        """Attach each ref feature to its matching attention block as a
        transient attribute. `SpatioTemporalTransformer.forward` doesn't take
        `ref_kv` through `TimestepEmbedSequential`'s dispatcher, so we stash
        it on the block and have the block's forward pick it up."""
        if ref_features is None:
            for block in self._attn_blocks:
                block._ref_kv_feature = None
            return
        if len(ref_features) != len(self._attn_blocks):
            raise ValueError(
                f"ref_features has {len(ref_features)} entries but the UNet has "
                f"{len(self._attn_blocks)} self-attention blocks."
            )
        for block, feat in zip(self._attn_blocks, ref_features):
            block._ref_kv_feature = feat

    def _clear_ref_features(self):
        for block in self._attn_blocks:
            block._ref_kv_feature = None
