# Ablate Expression-Weighted Diffusion Loss

Does weighting the per-pixel denoising loss by expression deformation
magnitude actually help, or does uniform loss converge to the same quality?

## The weighting formula

Implemented in `THDiffusion`:

```
weight = 1.0 + alpha * normalize(deformation_magnitude)
loss   = weighted_mean(MSE(noise_pred, noise) * weight)
```

`alpha` comes from `model.params.expr_weight_alpha`. `alpha=0` is uniform,
`alpha=5.0` is the standard weighted setting.

## How to compose each variant

| Variant       | Overlays                           | Comment                  |
|---------------|------------------------------------|--------------------------|
| Uniform loss  | *(base default, no overlays)*      | `alpha=0` in base.yaml   |
| Weighted loss | `overlays/loss/weighted.yaml`      | flips `alpha` to `5.0`   |

Compare each pair while holding conditioning constant. The full 2×2 matrix
with conditioning is documented in [`../ablate_conditioning/README.md`](../ablate_conditioning/README.md).