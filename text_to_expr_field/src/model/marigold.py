"""
Marigold-style input layer modification for Wan DiT.

Doubles the patch_embedding Conv3d input channels from 16 to 32
so the transformer can accept concatenated [noisy_target | clean_conditioning]
latents. Uses Marigold's weight duplication trick: clone weights, repeat
along input channel dim, halve to preserve activation magnitude.

Reference: Ke et al., "Repurposing Diffusion-Based Image Generators for
Monocular Depth Estimation" (CVPR 2024). Code adapted from:
https://github.com/prs-eth/Marigold/blob/main/src/trainer/marigold_depth_trainer.py
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def double_patch_embedding(transformer):
    """
    Replace the transformer's patch_embedding Conv3d to accept 2x input channels.

    Following Marigold's _replace_unet_conv_in():
      1. Clone the original weight [out_ch, 16, kT, kH, kW]
      2. Repeat along input channel dim: [out_ch, 32, kT, kH, kW]
      3. Scale by 0.5 to preserve activation magnitude at initialization
      4. Create new Conv3d(32, out_ch, ...) with these weights

    This is the ONLY architectural change. All other layers remain identical.

    Args:
        transformer: WanTransformer3DModel with a patch_embedding Conv3d

    Returns:
        transformer with modified patch_embedding (in-place)
    """
    original_conv = transformer.patch_embedding

    _weight = original_conv.weight.clone()  # [out_ch, 16, kT, kH, kW]
    _bias = original_conv.bias.clone() if original_conv.bias is not None else None

    # Repeat along input channel dim (dim=1) to double: 16 → 32
    _weight = _weight.repeat((1, 2, 1, 1, 1))  # [out_ch, 32, kT, kH, kW]
    _weight *= 0.5  # half activation magnitude

    _new_conv = nn.Conv3d(
        in_channels=original_conv.in_channels * 2,  # 32
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
    )
    _new_conv.weight = Parameter(_weight)
    if _bias is not None:
        _new_conv.bias = Parameter(_bias)

    transformer.patch_embedding = _new_conv

    # Update config so save/load knows about the new channel count
    transformer.config['in_channels'] = original_conv.in_channels * 2

    return transformer
