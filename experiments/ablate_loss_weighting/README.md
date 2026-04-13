# Ablate Expression-Weighted Diffusion Loss

Does weighting the per-pixel denoising loss by expression deformation
magnitude actually help, or does uniform loss converge to the same quality?

## Setup

The weighting formula (implemented in `THDiffusion`):

```
weight = 1.0 + alpha * normalize(deformation_magnitude)
loss   = weighted_mean(MSE(noise_pred, noise) * weight)
```

`alpha` is set via `expr_weight_alpha` in the model config. `alpha=0` yields
uniform loss; `alpha=5.0` is the default weighted setting.

## Config pairs to compare

| Conditioning               | Weighted (`alpha=5.0`)                | Uniform (`alpha=0`)                     |
|----------------------------|----------------------------------------|-----------------------------------------|
| Full 46ch                  | `full_cond_weighted_loss.yaml`         | `full_cond_uniform_loss.yaml`           |
| Ref-mask-only (1ch)        | `no_expr_weighted_loss.yaml`           | `no_expr_uniform_loss.yaml`             |

Compare each pair row-wise to isolate the loss-weighting effect from the
conditioning channel budget.