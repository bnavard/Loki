# Ablate Spatial Conditioning

Isolates the effect of varying the spatial conditioning channel budget while
keeping the loss weighting axis fixed.

The loss axis matters because **expression-weighted loss amplifies the diffusion
objective on deforming face regions** — it is a separate modeling choice from
the conditioning channels. When comparing conditioning variants, hold the loss
type constant.

## Conditioning × Loss matrix

| Conditioning                  | Weighted loss                              | Uniform loss                              |
|-------------------------------|--------------------------------------------|-------------------------------------------|
| Full (46ch pos + deform + ref) | `full_cond_weighted_loss.yaml`             | `full_cond_uniform_loss.yaml`             |
| Deform-only (4ch)              | `deform_only_weighted_loss.yaml`           | *missing — add if needed*                 |
| Ref-mask-only (1ch)            | `no_expr_weighted_loss.yaml`               | `no_expr_uniform_loss.yaml`               |

## Recommended sweeps

- **Row-wise (fix conditioning, vary loss):** measures the contribution of
  expression-weighted loss for a given conditioning regime.
- **Column-wise (fix loss, vary conditioning):** measures how much spatial
  information each channel budget carries for the denoiser.

All configs live in `marionette/configs/`. Architecture, dataset, and audio
conditioning are otherwise identical across the matrix.
