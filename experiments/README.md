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
  pre-generated Marigold deformations (4ch deformation + ref_mask).
- `overlays/expr_source/driving_video.yaml` — use raw driver frames at latent
  resolution instead of any structured deformation signal (4ch RGB + ref_mask).
- `overlays/pose/on.yaml` — enable the 6DRepNet head pose encoder whose
  embedding is added to the UNet's timestep embedding.

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

**Question:** What spatial conditioning signal gives the video diffusion model
the most useful information about facial dynamics? Four variants:

| Variant | Channels | Signal |
|---|---|---|
| `gt_full` | 46 | Full FLAME (42ch pos enc + 3ch deform + 1ch ref_mask) |
| `gt_baseline` | 4 | FLAME deformation only (deform-only overlay) |
| `marigold` | 4 | Marigold-generated deformation |
| `driving_video` | 4 | Raw driver frames at 64×64 (unstructured) |

The three 4ch variants share UNet architecture for apples-to-apples
comparison. `gt_full` is the upper-bound reference.

### [`ablate_loss_weighting/`](ablate_loss_weighting/)

**Question:** Does the expression-weighted loss actually help vs uniform loss?
Flips `overlays/loss/weighted.yaml` on/off.

## Quantitative metrics

[`evaluation_metrics/`](evaluation_metrics/) — reusable metric implementations
that operate on trained-model outputs. Currently hosts lip-sync evaluation
(LSE-D, LSE-C, AV-Offset) via SyncNet. Used by `ablate_audio` to quantify
whether wav2vec2 cross-attention actually improves audio-visual
synchronisation over deformation-only conditioning.
