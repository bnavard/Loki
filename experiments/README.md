# Marionette Ablation Experiments

This folder organises ablation studies over the Marionette video diffusion
model. Each sub-folder targets a single axis of variation and holds the
experiment configs, entry script, and a lazy output symlink.

## How experiment configs compose

Every Marionette config is a **base + overlays** composition. The base config
at [`marionette/configs/base.yaml`](../marionette/configs/base.yaml) defines
defaults for all fields. Overlays under
[`marionette/configs/overlays/`](../marionette/configs/overlays/) carry only
the fields that change:

- `overlays/conditioning/` — `deform_only.yaml` (4ch), `no_expr.yaml` (1ch).
  The base ships the full 46ch conditioning.
- `overlays/loss/weighted.yaml` — turn on expression-weighted loss
  (`alpha=5.0`). The base ships uniform loss (`alpha=0`).
- `overlays/audio/off.yaml` — disable wav2vec2 cross-attention end-to-end.
- `overlays/expr_source/marigold.yaml` — swap FLAME rasterization for
  pre-generated Marigold deformations.

An experiment config declares a `base:` and a list of `overlays:` to apply, in
order. Later overlays win on conflicts. `marionette.config_utils.load_experiment_config`
resolves the chain.

Each experiment's `run.py` calls `marionette.train.run_training(cfg, output_dir=...)`
directly — no subprocess, no `python marionette/train.py --config ...`. The
training helper snapshots the fully-resolved config as `config_resolved.yaml`
in the run directory for reproducibility.

## Conventions

- **Outputs live at repo root**, under `outputs/<experiment_name>/<variant>/run_<timestamp>/`.
- Each experiment folder creates a **lazy symlink** at `experiments/<name>/outputs`
  pointing at its output root on first run.
- Architecture configs (`base.yaml`, overlays) live in `marionette/configs/`;
  experiment folders only hold the tiny composition YAMLs + `run.py`.

## Studies

### [`ablate_conditioning/`](ablate_conditioning/)

**Question:** Which spatial conditioning signals actually contribute?
Compares full 46ch vs deform-only (4ch) vs ref-mask-only (1ch) by swapping
`overlays/conditioning/*` entries.

### [`ablate_audio/`](ablate_audio/)

**Question:** Does wav2vec2 cross-attention improve lip sync and
expressiveness, or is the spatial FLAME signal sufficient on its own?
Flips `overlays/audio/off.yaml` on/off.

### [`ablate_expr_source/`](ablate_expr_source/) — **implemented**

**Question:** How much reconstruction quality is lost when the deformation map
comes from the Marigold generator vs the ground-truth FLAME rasterization?
Both variants use `overlays/conditioning/deform_only.yaml`; only the Marigold
variant adds `overlays/expr_source/marigold.yaml`. See its README for how to
run and what to measure.

### [`ablate_loss_weighting/`](ablate_loss_weighting/)

**Question:** Does the expression-weighted loss actually help vs uniform loss?
Flips `overlays/loss/weighted.yaml` on/off.
