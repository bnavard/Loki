"""
Marigold-style input layer modification for diffusion transformers.

Doubles the input projection's channel count so the transformer can accept
concatenated [noisy_target | clean_conditioning] latents. Uses Marigold's
weight duplication trick: clone weights, repeat along input channel dim,
halve to preserve activation magnitude.

Supports:
  - SD3Transformer2DModel: pos_embed.proj Conv2d(16 → 32, k=2, s=2)
  - WanTransformer3DModel: patch_embedding Conv3d(16 → 32, k=(1,2,2), s=(1,2,2))

Reference: Ke et al., "Repurposing Diffusion-Based Image Generators for
Monocular Depth Estimation" (CVPR 2024).
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def double_input_channels(transformer):
    """
    Double the transformer's input projection to accept 2x input channels.

    Detects the architecture automatically:
      - SD3Transformer2DModel: modifies pos_embed.proj (Conv2d)
      - WanTransformer3DModel: modifies patch_embedding (Conv3d)

    Following Marigold's _replace_unet_conv_in():
      1. Clone the original weight
      2. Repeat along input channel dim (dim=1)
      3. Scale by 0.5 to preserve activation magnitude
      4. Replace with new Conv of doubled input channels

    Args:
        transformer: SD3Transformer2DModel or WanTransformer3DModel

    Returns:
        transformer with modified input projection (in-place)
    """
    # Detect architecture
    if hasattr(transformer, "pos_embed") and hasattr(transformer.pos_embed, "proj"):
        # SD3Transformer2DModel: Conv2d at pos_embed.proj
        original_conv = transformer.pos_embed.proj
        conv_cls = nn.Conv2d
        repeat_dims = (1, 2, 1, 1)  # [out, in*2, kH, kW]
        attr_chain = ("pos_embed", "proj")
    elif hasattr(transformer, "patch_embedding"):
        # WanTransformer3DModel: Conv3d at patch_embedding
        original_conv = transformer.patch_embedding
        conv_cls = nn.Conv3d
        repeat_dims = (1, 2, 1, 1, 1)  # [out, in*2, kT, kH, kW]
        attr_chain = ("patch_embedding",)
    else:
        raise ValueError(
            f"Unknown transformer architecture: {type(transformer).__name__}. "
            f"Expected SD3Transformer2DModel or WanTransformer3DModel."
        )

    _weight = original_conv.weight.clone()
    _bias = original_conv.bias.clone() if original_conv.bias is not None else None

    # Repeat along input channel dim (dim=1) to double: 16 → 32
    _weight = _weight.repeat(repeat_dims)
    _weight *= 0.5  # half activation magnitude

    _new_conv = conv_cls(
        in_channels=original_conv.in_channels * 2,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
    )
    _new_conv.weight = Parameter(_weight)
    if _bias is not None:
        _new_conv.bias = Parameter(_bias)

    # Set the new conv on the transformer
    if len(attr_chain) == 1:
        setattr(transformer, attr_chain[0], _new_conv)
    else:
        parent = getattr(transformer, attr_chain[0])
        setattr(parent, attr_chain[1], _new_conv)

    # Update config
    transformer.config["in_channels"] = original_conv.in_channels * 2

    return transformer
