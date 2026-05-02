"""
Conditioning module for the `flame_vector` arm of condition_ablation.

Feeds the diffusion model the raw FLAME motion parameters as a flat vector,
without rasterizing them into pixel space. This is the "earlier work" recipe
the paper contrasts against in §4.3 — the strawman the canonical 45-channel
spatial conditioning is meant to outperform.

Pipeline (per frame, shared MLP):

    driver_flame_params  (B, T, D)              ← from TalkingHeadDataset
                                                  (D = 77, fixed schema:
                                                   expr|rot|neck_rot|jaw_rot|eye_rot)
        │
        │  Linear(D → hidden) → SiLU → Linear(hidden → C)
        ▼
    v                    (B, T, C)              C = model_channels (320)
        │
        │  spatial broadcast (no learnable parameters)
        ▼
    spatial_cond         (B, T, H, W, C)        H = W = latent_resolution (64)

The broadcast tiles the per-frame vector across every spatial position, so
the resulting tensor has *zero spatial structure* by construction. This is
the load-bearing property of the ablation: the downstream gen UNet's conv
stack and self-attention cannot recover "where on the face" each FLAME
coefficient applies, because every pixel of the conditioning carries the
same value. The rasterized baseline hands the model that mapping for free
(value at pixel (i, j) corresponds to the face point projecting to (i, j));
this arm withholds it.

The spatial-broadcast trick is a standard mechanism for fusing global
vectors into convolutional feature maps. Two canonical references that
justify the contract used here:

  - Watters et al., "Spatial Broadcast Decoder: A Simple Architecture for
    Learning Disentangled Representations in VAEs" (2019). The decoder tiles
    a latent code across all spatial positions before the conv stack — the
    exact operation this module performs. Their ablation shows the broadcast
    decoder *outperforms* a deconv decoder on disentanglement, establishing
    that broadcasting a non-spatial vector into a conv feature map is a
    legitimate (and often preferable) form of conditioning.
  - Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
    (2018). A vector is projected to per-channel scale/bias and broadcast
    across H×W to modulate a feature map — same machinery, motivating that
    the spatial broadcast is the *strong* form of vector-only conditioning,
    not a strawman.

Citing these matters for the ablation argument: we are not handicapping the
vector path with a weak injection, we are using the established trick. If
the rasterized 45-channel arm wins anyway, the win is attributable to the
representation (pixel-space FLAME), not to the injection mechanism.

Output contract matches `SpatialConditioning.forward` so the UNet plumbing
is unchanged. The downstream `ConditioningEncoder` is configured (in
`config.yaml`) as a near-no-op at the latent resolution: a Conv3×3 stem at
64×64, one refine ResBlock, and a zero-initialised Conv3×3 out-projection.
The zero-init out-conv keeps the step-0 invariant the rasterized arm relies
on — the conditioning contributes exactly zero at training start, so the
pretrained UNet begins from its known-good state.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlameVectorConditioning(nn.Module):
    """Project per-frame raw FLAME params through a shared MLP and tile the
    result across the latent grid. Emits the same `(B, T, H, W, C)` tensor
    shape as the canonical `SpatialConditioning`, with C = model_channels.
    """

    # Viz contract: every spatial position carries the same vector for a given
    # frame, so any 3-channel slice renders as a per-frame solid-colour panel
    # — that's the visual signature of "no spatial structure," which is
    # exactly the property the ablation is testing. Slice [0:3] of the
    # post-MLP feature is as good as any.
    VIZ_SLICE: tuple[int, int] = (0, 3)
    VIZ_LABEL: str = "FLAME (broadcast)"

    def __init__(
        self,
        flame_params_dim: int = 77,
        model_channels: int = 320,
        latent_resolution: int = 64,
        hidden_dim: int = 320,
        **_unused,
    ) -> None:
        # `**_unused` absorbs baseline `cond_stage_config.params` keys
        # (image_size, positional_channels, ...) that survive the OmegaConf
        # deep-merge from base.yaml but don't apply to a vector-conditioning
        # path.
        super().__init__()
        self.flame_params_dim  = flame_params_dim
        self.model_channels    = model_channels
        self.latent_resolution = latent_resolution

        self.mlp = nn.Sequential(
            nn.Linear(flame_params_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_channels),
        )

    @property
    def n_conditioning_channels(self) -> int:
        return self.model_channels

    def forward(self, batch: dict) -> dict:
        device = next(self.parameters()).device

        flame = batch["driver_flame_params"].to(device)  # (B, T, D)
        if flame.shape[-1] != self.flame_params_dim:
            raise ValueError(
                f"FlameVectorConditioning expected last dim "
                f"{self.flame_params_dim}, got {flame.shape[-1]}. Check "
                f"FLAME_PARAMS_SCHEMA in marionette.data.video_dataset and "
                f"`flame_params_dim` in this arm's config."
            )

        v = self.mlp(flame)  # (B, T, C)

        H = W = self.latent_resolution
        spatial_cond = (
            v[:, :, None, None, :]
             .expand(-1, -1, H, W, -1)
             .contiguous()                                # (B, T, H, W, C)
        )
        return {"spatial_cond": spatial_cond}
