# Experiments

Evaluation utilities for Marionette. Ablation studies tied to removed
codepaths (marigold, weighted loss, conditioning ablations, pose encoder)
were cleared out on `marionette_v3`; they can be re-added if/when those
variants are reintroduced.

## Contents

### [`evaluation_metrics/`](evaluation_metrics/) — lip-sync evaluation

SyncNet-based audio-visual sync metrics (LSE-D, LSE-C, AV offset) for
judging generated videos against their driver audio. See the subfolder
README for setup + usage.

## Conventions

- Training / inference configs live in `marionette/configs/` — not here.
  This folder is for evaluation scripts only on the v3 branch.
- Outputs land at repo root under `outputs/<run>/…`; any experiment folder
  that writes output creates a lazy `outputs` symlink on first run.
