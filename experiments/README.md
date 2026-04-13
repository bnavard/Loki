# Marionette Ablation Experiments

This folder organizes ablation studies over the Marionette video diffusion
model. Each sub-folder targets a single axis of variation and holds the
configs, launcher scripts, and evaluation entry points for that study.

## Conventions

- Model architecture configs live in [`marionette/configs/`](../marionette/configs/).
  Experiment folders here only *reference* those configs via launch scripts —
  they do not duplicate model definitions.
- Each experiment folder contains:
  - `launch.sh` — runs the relevant training invocations in sequence or parallel.
  - `eval.py` (where applicable) — evaluates a trained checkpoint and reports
    the metrics relevant to the study axis.
  - `README.md` — states the question being asked and the expected signal.

## Studies

### [`ablate_conditioning/`](ablate_conditioning/)

**Question:** Which spatial conditioning signals actually contribute?
Compares full 46ch positional + deformation vs deformation-only (4ch) vs
ref-mask-only (1ch).

Uses the existing `full_cond_*`, `deform_only_*`, `no_expr_*` configs.

### [`ablate_audio/`](ablate_audio/)

**Question:** Does wav2vec2 cross-attention improve lip sync and
expressiveness, or is the spatial FLAME signal sufficient on its own?

Requires a config with `use_audio_context: false` and `audio_encoder_config: null`.
A template is provided in this folder.

### [`ablate_expr_source/`](ablate_expr_source/)

**Question:** How much reconstruction quality is lost when the deformation map
comes from the Marigold generator vs the ground-truth FLAME rasterization?

Runs the same architecture with `expression_source: "gt"` vs `"marigold"` in
both the dataset and the conditioning module. Requires pre-running
`caching/scripts/cache_marigold_deform.py` to populate
`data/derived/marigold_deform/`.

### [`ablate_loss_weighting/`](ablate_loss_weighting/)

**Question:** Does the expression-weighted loss actually help vs uniform loss?
Already covered by the existing `*_weighted_loss.yaml` vs `*_uniform_loss.yaml`
config pairs.