# Ablate Spatial Conditioning

Isolates the effect of varying the spatial conditioning channel budget while
holding loss weighting fixed. Expression-weighted loss is a separate modelling
choice (see `ablate_loss_weighting/`); when comparing conditioning variants,
hold loss type constant.

## How to compose each variant

| Conditioning              | Overlays                                                            |
|---------------------------|---------------------------------------------------------------------|
| Full (46ch)               | *(base default, no overlays)*                                        |
| Deform-only (4ch)         | `overlays/conditioning/deform_only.yaml`                             |
| Ref-mask-only (1ch)       | `overlays/conditioning/no_expr.yaml`                                 |

The base defaults to uniform loss (`alpha=0`). Add
`overlays/loss/weighted.yaml` to any of the above to get the weighted-loss
row of the matrix.

## Conditioning × Loss matrix

Hold one axis fixed at a time:

|                             | Uniform loss (base)                               | Weighted loss                                                          |
|-----------------------------|---------------------------------------------------|------------------------------------------------------------------------|
| Full (46ch)                 | `[]`                                              | `[loss/weighted]`                                                      |
| Deform-only (4ch)           | `[conditioning/deform_only]`                      | `[conditioning/deform_only, loss/weighted]`                            |
| Ref-mask-only (1ch)         | `[conditioning/no_expr]`                          | `[conditioning/no_expr, loss/weighted]`                                |

Each cell is the overlay list to drop into an experiment config. Author the
matrix by creating one YAML per cell you want to run under
`experiments/ablate_conditioning/configs/`.
