# Experiments

Evaluation utilities for Marionette. Ablation studies tied to removed
codepaths (marigold, weighted loss, conditioning ablations, pose encoder)
were cleared out on `marionette_v3`; they can be re-added if/when those
variants are reintroduced.

## Contents

### [`marionette_baseline/`](marionette_baseline/) — canonical training run
Point-of-truth training against [`marionette/configs/base.yaml`](../marionette/configs/base.yaml)
unchanged. The audio-on, full-FLAME reference every ablation arm is
compared against.

### [`condition_ablation/`](condition_ablation/) — conditioning-pathway ablations
Single-variable ablations on the conditioning inputs feeding the gen UNet:
audio on/off, FLAME-vs-natural-video, pos_enc-vs-deform. Each arm trains
a fresh 30k-step run from the same SD 2.1 init + seed as the baseline; the
only thing that differs per arm is which conditioning channels the model
sees. See the subfolder README for the full matrix.

### [`marionette_eval/`](marionette_eval/) — cross + same identity evaluation
Loads any baseline or ablation checkpoint and generates side-by-side
panel/mp4 outputs on the validation set. Same-identity reconstruction plus
a deranged-pair cross-identity sweep across all usable YouTube identities.

### [`evaluation_metrics/`](evaluation_metrics/) — lip-sync evaluation
SyncNet-based audio-visual sync metrics (LSE-D, LSE-C, AV offset) for
judging generated videos against their driver audio.

## Conventions

- Base training / inference configs live in `marionette/configs/`.
  Experiment-specific configs live inside each experiment subfolder.
- Outputs land at repo root under `outputs/<experiment>/…`.
