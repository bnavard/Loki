"""
Reference-feature extractor built on a frozen SD 2.1 UNet.

Runs a pretrained SD 2.1 UNet once per sample on the VAE-encoded reference
frame and captures the input tensor to each self-attention block (attn1).
These per-layer features are then injected as additional K/V tokens into the
main generation UNet's corresponding self-attention layers, following the
ReferenceNet pattern (Hu et al., 2024, "Animate Anyone: Consistent and
Controllable Image-to-Video Synthesis for Character Animation").

The ref UNet is held in eval mode with `requires_grad=False` on all params —
it serves as a static identity feature extractor anchored to SD 2.1's
pretrained representation space. No gradient flows through it; no activations
are stored for backprop. The only overhead at training is the single forward
pass per batch; at inference the pass runs once per sample and the features
are reused across every DDIM step.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ldm_base.ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm_base.ldm.modules.attention import SpatialTransformer
from ldm_base.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder


def _compute_null_context() -> torch.Tensor:
    """Encode the empty string via SD 2.1's OpenCLIP text encoder once and
    return the (1, 77, 1024) null-prompt embedding on CPU. The encoder is
    instantiated on GPU (if available) for a single forward, then deleted and
    the CUDA cache is flushed — we only need this static tensor to feed the
    frozen ref UNet's cross-attention layers (which expect a 1024-dim context).
    """
    enc_device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenOpenCLIPEmbedder(device=enc_device, freeze=True).eval()
    encoder.model.to(enc_device)
    with torch.no_grad():
        null_ctx = encoder.encode([""]).detach().cpu()
    del encoder
    if enc_device == "cuda":
        torch.cuda.empty_cache()
    return null_ctx


def _iter_self_attn_inputs(unet: UNetModel):
    """Yield the `norm1` modules of each BasicTransformerBlock in forward-pass
    order. Hooking these gives the input tensor that flows into the block's
    self-attention (attn1)."""
    for stage in (unet.input_blocks, (unet.middle_block,), unet.output_blocks):
        for block in stage:
            for sub in block:
                if isinstance(sub, SpatialTransformer):
                    for tblock in sub.transformer_blocks:
                        yield tblock.norm1


class RefFeatureExtractor(nn.Module):
    """Runs a frozen SD 2.1 UNet on the reference latent and returns the
    per-layer self-attention inputs as a list of (B, HW, D) tensors, ordered
    to match the corresponding self-attention blocks in the generation UNet.
    """

    def __init__(self, unet_config: dict):
        super().__init__()
        self.unet = UNetModel(**unet_config)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # SD 2.1's cross-attention layers expect a 1024-dim text context. We
        # always feed the ref UNet with the null prompt — identity features
        # should not be modulated by any textual signal. Precompute once and
        # travel with the module via a non-persistent buffer.
        self.register_buffer(
            "null_context", _compute_null_context(), persistent=False,
        )

        # Forward hooks capture norm1 outputs — the input to each attn1.
        # We register an index-aware hook per block so the returned list is
        # deterministically ordered for downstream injection.
        self._cache: List[Optional[torch.Tensor]] = []
        for idx, norm1 in enumerate(_iter_self_attn_inputs(self.unet)):
            norm1.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx: int):
        def _hook(_module, _input, output):
            while len(self._cache) <= idx:
                self._cache.append(None)
            self._cache[idx] = output.detach()
        return _hook

    @torch.no_grad()
    def forward(
        self,
        ref_z: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Extract per-layer self-attention inputs from a single ref latent.

        Args:
            ref_z:    (B, 4, h, w) VAE-encoded reference latent.
            timesteps: optional (B,) long tensor. Defaults to zeros — the ref
                       is clean (not noisy), so t=0 is the right conditioning.

        Returns:
            List of (B, HW_i, D_i) tensors, one per self-attention block, in
            forward-pass order. HW_i and D_i vary across UNet levels; the
            matching generation block at index `i` consumes the `i`-th entry.
        """
        self._cache = []
        b = ref_z.shape[0]
        if timesteps is None:
            timesteps = torch.zeros(b, device=ref_z.device, dtype=torch.long)
        context = self.null_context.expand(b, -1, -1)
        _ = self.unet(ref_z, timesteps=timesteps, context=context)
        return list(self._cache)

    @staticmethod
    def load_sd21_into_ref(
        state_dict: dict, prefix: str = "ref_extractor.unet.",
    ) -> dict:
        """Given a state_dict that contains SD 2.1's UNet weights under
        `model.diffusion_model.*`, return a new dict that ALSO carries those
        same weights under `{prefix}*` so a single `load_state_dict` call
        populates both the generation UNet (its original location) and the
        reference UNet (the new prefix).

        Pure function; the returned dict is a superset of the input. No-op
        when no SD 2.1 UNet keys are present.
        """
        out = dict(state_dict)
        gen_prefix = "model.diffusion_model."
        for key, val in state_dict.items():
            if key.startswith(gen_prefix):
                out[prefix + key[len(gen_prefix):]] = val
        return out
