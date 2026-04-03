"""
Adapted from cap4d/mmdm/net/mmdm_unet.py.

Changes vs MMDMUnetModel:
  1. use_context = True  — cross-attention is enabled for audio conditioning.
  2. Removed `assert context == None`.
  3. audio_context extracted from control dict and reshaped to (B*T, S, D)
     before being passed as `context` to every UNet block.
  4. Import path updated: talkinghead.model.attention.
"""

import torch
import torch.nn as nn
import einops

from controlnet.ldm.modules.diffusionmodules.openaimodel import UNetModel
from controlnet.ldm.modules.diffusionmodules.util import zero_module, timestep_embedding

from talkinghead_sd21_unet_cap4d_based.model.attention import SpatioTemporalTransformer


class THUnetModel(UNetModel):
    """
    Talking-head diffusion UNet.

    Differences from the base CAP4D UNet (MMDMUnetModel):
      - Cross-attention is ENABLED (use_context=True). Audio features are passed
        as the `context` argument at every transformer block, allowing each
        spatial position to attend to its frame's audio representation.
      - Spatial conditioning (FLAME pose/expression maps) is still injected as a
        learned linear projection added to the first feature map — unchanged.
      - The reference-masking logic (replace latent with GT where ref_mask=1) is
        preserved identically.

    Args:
        time_steps        : video window length T.
        condition_channels: channels in the spatial conditioning tensor (default 46).
        context_dim       : audio context feature dimension (default 768, matches
                            wav2vec2-base / HuBERT-base output).
        temporal_mode     : "3d" or "temporal" — controls spatio-temporal attention.
    """

    def __init__(
        self,
        *args,
        time_steps: int,
        condition_channels: int = 46,
        model_channels: int = 320,
        image_size: int = 32,
        context_dim: int = 768,          # audio cross-attention dim
        temporal_mode: str = "3d",       # ["3d", "temporal"]
        **kwargs,
    ):
        assert temporal_mode in ["3d", "temporal"]
        self.temporal_mode = temporal_mode
        self.time_steps = time_steps
        self.use_context = True          # ← enabled for audio cross-attention

        super().__init__(
            *args,
            model_channels=model_channels,
            image_size=image_size,
            context_dim=context_dim,
            **kwargs,
        )

        # Linear projection: spatial conditioning channels → model channels
        # (added to the first UNet feature map; same mechanism as CAP4D)
        self.cond_linear = zero_module(nn.Linear(condition_channels, model_channels))

    # ------------------------------------------------------------------
    # Attention block factory  (called by parent UNetModel.__init__)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        """
        Args:
            x        : (B, T, 4, H, W)  noisy latents
            timesteps: (B, T)            diffusion timesteps per frame
            context  : unused (audio context is taken from control dict instead)
            control  : dict with keys:
                         pos_enc       (B, T, H, W, C_cond)  spatial conditioning
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

        # ------- spatial conditioning (linear proj + add to first feature map) -------
        pos_enc      = einops.rearrange(control["pos_enc"], 'b t h w c -> (b t) h w c')
        pos_enc      = pos_enc.type(self.dtype)
        pos_embedding = self.cond_linear(pos_enc).permute(0, 3, 1, 2)  # (B*T, model_ch, H, W)

        # ------- audio context for cross-attention -------
        # Shape: (B, T, S, D) → (B*T, S, D)
        audio_ctx = None
        if "audio_context" in control and control["audio_context"] is not None:
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
            if pos_embedding is not None:
                h += pos_embedding           # spatial conditioning injected once
                pos_embedding = None
            hs.append(h)

        h = self.middle_block(h, emb, audio_ctx)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, audio_ctx)

        h = self.out(h).type(x.dtype)

        # ------- unflatten time dimension -------
        h = einops.rearrange(h, '(b t) c h w -> b t c h w', b=b_)

        # Pass reference frames through unchanged; return predicted noise elsewhere
        h = x_input * ref_mask + h * ref_mask_inv

        return h
