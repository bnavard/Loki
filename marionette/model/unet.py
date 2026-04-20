"""
SD 2.1 UNet extended for talking-head video generation.

Two conditioning pathways:

  * Spatial (49ch `spatial_cond` = pos_enc + driver_deform + warped_ref +
    ref_mask) enters through a small SD-style convolutional encoder
    (`ConditioningEncoder`) that takes the conditioning at full image
    resolution (default 512×512), downsamples it through three strided
    ResBlocks to the UNet's latent resolution (64×64), and emits
    `model_channels` feature maps that are added to the UNet's first feature
    map. The encoder's final Conv3×3 is zero-initialised so conditioning
    contributes zero at step 0 and the pretrained UNet starts from its
    known-good state. The encoder is trained jointly with the UNet.

  * Audio (wav2vec2 cross-attention): controlled by `use_audio_context`.
    When True, audio tokens are passed as the `context` argument to every
    transformer block. When False (audio-ablation), cross-attention is
    skipped entirely across the UNet.

Reference frames bypass denoising via the ref_mask + z_input passthrough
mechanism — frame 0 (ref_mask=1) outputs the known noise residual so the
reference passes through unchanged while still participating in 3D attention
for identity conditioning.
"""

import torch
import torch.nn as nn
import einops

from ldm_base.ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm_base.ldm.modules.diffusionmodules.util import timestep_embedding

from marionette.model.attention import SpatioTemporalTransformer
from marionette.model.conditioning_encoder import ConditioningEncoder


class MarionetteUNet(UNetModel):
    """
    Talking-head diffusion UNet.

    Components:
      - Cross-attention, toggled by `use_audio_context`: when True, audio features
        are passed as the `context` argument at every transformer block so each
        spatial position can attend to its frame's audio representation. When
        False, cross-attention is skipped throughout the UNet (audio ablation).
      - Spatial conditioning (49ch `spatial_cond` — FLAME pos_enc, driver
        expression deformation, warped reference, ref_mask) is injected through
        a learned `ConditioningEncoder` that takes the conditioning at full
        image resolution (default 512×512), downsamples it through three
        strided ResBlocks to the UNet's latent resolution (64×64), and produces
        `model_channels` feature maps that are added to the first UNet feature
        map. The encoder is trained jointly with the UNet; its final projection
        is zero-initialised.
      - Reference-masking logic: replace latent with GT where ref_mask=1, and
        pass those frames through unchanged at the output.

    Args:
        time_steps        : video window length T.
        condition_channels: channels in the spatial conditioning tensor (default 46).
        cond_input_resolution : spatial size of the incoming conditioning tensor
                            (default 512 — must match SpatialConditioning.image_size).
        cond_latent_resolution: target spatial size after the encoder (default 64
                            — must match the UNet's first feature map).
        cond_stage_channels: channel progression through the encoder's downsample
                            stages. Default `(64, 128, 256, model_channels)`
                            (3 downsample stages: 512 → 256 → 128 → 64).
        context_dim       : audio context feature dimension (default 768, matches
                            wav2vec2-base / HuBERT-base output).
        temporal_mode     : "3d" or "temporal" — controls spatio-temporal attention.
        use_audio_context : enable / disable audio cross-attention (default True).
    """

    def __init__(
        self,
        *args,
        time_steps: int,
        condition_channels: int = 46,
        model_channels: int = 320,
        image_size: int = 32,
        context_dim: int = 768,              # audio cross-attention dim
        temporal_mode: str = "3d",           # ["3d", "temporal"]
        use_audio_context: bool = True,      # False -> skip cross-attn entirely (no-audio ablation)
        cond_input_resolution: int = 512,
        cond_latent_resolution: int = 64,
        cond_stage_channels: tuple | list | None = None,
        **kwargs,
    ):
        assert temporal_mode in ["3d", "temporal"]
        self.temporal_mode = temporal_mode
        self.time_steps = time_steps
        self.use_context = use_audio_context  # drives cross-attention in every block

        super().__init__(
            *args,
            model_channels=model_channels,
            image_size=image_size,
            context_dim=context_dim,
            **kwargs,
        )

        # Learned conditioning encoder: full-res spatial conditioning →
        # `model_channels` feature maps at the UNet's latent resolution.
        # Zero-initialised at its final layer so training starts from the
        # pretrained UNet's behaviour (no conditioning contribution at step 0).
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
        if self.temporal_mode == "temporal":
            temporal_connection_type = "temporal"
        else:  # "3d"
            temporal_connection_type = "3d" if mult >= 2 else "none"

        return SpatioTemporalTransformer(
            ch,
            num_heads,
            dim_head,
            use_context=self.use_context,   # True → audio cross-attention enabled
            context_dim=context_dim,
            temporal_connection_type=temporal_connection_type,
            num_timesteps=self.time_steps,
        )

    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        """
        Args:
            x        : (B, T, 4, H, W)  noisy latents
            timesteps: (B, T)            diffusion timesteps per frame
            context  : unused (audio context is taken from control dict instead)
            control  : dict with keys:
                         spatial_cond  (B, T, H, W, C_cond)  spatial conditioning (49ch)
                         z_input       (B, T, 4, H, W)        GT latents (refs only)
                         ref_mask      (B, T, 1, H, W)        1=reference, 0=generated
                         audio_context (B, T, S, D)           audio features per frame
        Returns:
            (B, T, 4, H, W)  predicted noise (reference slots pass through unchanged)
        """
        z_input  = control["z_input"]   # (B, T, 4, H, W)
        ref_mask = control["ref_mask"]  # (B, T, 1, H, W)

        # Ground-truth noise target = (noisy - clean); used to passthrough at refs
        x_input = x - z_input

        ref_mask_inv = torch.logical_not(ref_mask)

        # Replace noisy latents with GT latents at reference frame slots
        x = z_input * ref_mask + x * ref_mask_inv

        # ------- flatten time dimension -------
        b_, t_ = x.shape[:2]
        x         = einops.rearrange(x,         'b t c h w -> (b t) c h w')
        timesteps = einops.rearrange(timesteps, 'b t      -> (b t)')

        # ------- spatial conditioning (conv encoder, add to first feature map) -------
        # Incoming spatial_cond is (B, T, H_hi, W_hi, 49) at full image resolution.
        # Rearrange to channels-first, run through the encoder which downsamples
        # to (B*T, model_ch, H_lat, W_lat) matching the UNet's first feature map.
        spatial_cond  = einops.rearrange(control["spatial_cond"], 'b t h w c -> (b t) c h w')
        spatial_cond  = spatial_cond.type(self.dtype)
        cond_feature  = self.cond_encoder(spatial_cond)   # (B*T, model_ch, H_lat, W_lat)

        # ------- audio context for cross-attention -------
        # Shape: (B, T, S, D) → (B*T, S, D).  Skipped entirely when audio is ablated.
        audio_ctx = None
        if self.use_context and "audio_context" in control and control["audio_context"] is not None:
            audio_ctx = einops.rearrange(
                control["audio_context"], 'b t s d -> (b t) s d'
            ).type(self.dtype)

        # ------- UNet forward -------
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(self.dtype)
        emb   = self.time_embed(t_emb)
        h     = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb, audio_ctx)   # audio_ctx passed as context
            if cond_feature is not None:
                h += cond_feature            # spatial conditioning injected once
                cond_feature = None
            hs.append(h)

        h = self.middle_block(h, emb, audio_ctx)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, audio_ctx)

        h = self.out(h).type(x.dtype)

        h = einops.rearrange(h, '(b t) c h w -> b t c h w', b=b_)

        # Pass reference frames through unchanged; return predicted noise elsewhere
        h = x_input * ref_mask + h * ref_mask_inv

        return h
