"""
Multi-resolution noise for diffusion model training.

Instead of plain Gaussian noise, generates a pyramid of noise at progressively
lower resolutions, summed together and normalized to unit variance. This gives
the noise correlated low-frequency structure, helping the model learn large-scale
spatial coherence (smooth deformation gradients, consistent face regions).

Applied in latent space: for 512px images with 8x VAE compression, the latents
are 64x64. The pyramid generates noise from 64x64 down to 1x1, with each level
weighted by strength^i.

When annealed, the multi-res strength scales with timestep: at high timesteps
(heavy noise) the full pyramid effect applies, at low timesteps (near-clean)
it fades toward standard Gaussian noise.

Adapted from Marigold (Ke et al., CVPR 2024):
  https://github.com/prs-eth/Marigold/blob/main/src/util/multi_res_noise.py

Note: Marigold used 768px (96x96 latents). We use 512px (64x64 latents).
The pyramid adapts automatically — fewer levels at lower resolution.
"""

import math
import torch


def multi_res_noise_like(
    x, strength=0.9, downscale_strategy="original", generator=None, device=None,
):
    """
    Generate multi-resolution noise matching the shape of x.

    Args:
        x:                  [B, C, H, W] tensor (latent space)
        strength:           decay per pyramid level. Scalar or [B]-shaped tensor
                           for per-sample annealing (strength * t / t_max).
        downscale_strategy: "original" (random 2-4x per level),
                           "every_layer" (fixed 2x),
                           "power_of_two", "random_step"
        generator:          optional torch.Generator for reproducibility
        device:             device for noise tensors

    Returns:
        [B, C, H, W] noise tensor normalized to ~unit variance
    """
    if torch.is_tensor(strength):
        strength = strength.reshape((-1, 1, 1, 1))

    b, c, w, h = x.shape

    if device is None:
        device = x.device

    up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(x.shape, device=device, generator=generator)

    if downscale_strategy == "original":
        for i in range(10):
            r = torch.rand(1, generator=generator, device=device) * 2 + 2
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break
    elif downscale_strategy == "every_layer":
        for i in range(int(math.log2(min(w, h)))):
            w, h = max(1, int(w / 2)), max(1, int(h / 2))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
    elif downscale_strategy == "power_of_two":
        for i in range(10):
            r = 2
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break
    elif downscale_strategy == "random_step":
        for i in range(10):
            r = torch.rand(1, generator=generator, device=device) * 2 + 2
            w, h = max(1, int(w / r)), max(1, int(h / r))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break
    else:
        raise ValueError(f"Unknown downscale strategy: {downscale_strategy}")

    noise = noise / noise.std()
    return noise
