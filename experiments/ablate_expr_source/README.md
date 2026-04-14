# Ablate Expression Map Source (GT vs Marigold)

How much reconstruction quality does Marionette lose when the 3ch expression
deformation comes from the Marigold generator instead of the ground-truth
FLAME rasterization?

## Setup

Both variants use the same architecture (deform-only 4ch conditioning, uniform
diffusion loss). The only thing that changes is where the 3ch deformation map
comes from.

### Configs

```
experiments/ablate_expr_source/configs/
├── gt_baseline.yaml     # base + overlays/conditioning/deform_only
└── marigold.yaml        # base + overlays/conditioning/deform_only
                         #      + overlays/expr_source/marigold
```

Uniform loss is the base default, so no loss overlay is needed. The
`deform_only` overlay sets `expr_deform_only=true` and `condition_channels=4`.
The `expr_source/marigold` overlay flips `expression_source` to `"marigold"` in
both the dataset and the conditioning module, and points the dataset at
`data/derived/marigold_deform/`.

### Channel layout in Marigold mode

The Marigold module produces only the 3ch deformation — not the 42ch
positional encoding. Conditioning is therefore always 4ch
`(3 deform + 1 ref_mask)` in Marigold mode, regardless of other flags. The GT
baseline uses matching 4ch conditioning (`deform_only` overlay) so the
comparison is apples-to-apples.

## Prerequisites

Populate the Marigold deformation cache before running the `marigold` variant:

```bash
PYTHONPATH=. python scripts/cache/cache_marigold_deform.py \
    --checkpoint <path/to/marigold_checkpoint.pt> \
    --output_dir data/derived/marigold_deform
```

This writes `data/derived/marigold_deform/{clip_id}/deformation.mp4` for every
clip in the manifest. The dataset decodes the requested frames on the fly.

**The cache does not need to be complete.** `run.py` automatically intersects
the train/val clip lists with the set of clips that actually have a
`deformation.mp4` in the cache directory, and writes the filtered lists to
`outputs/ablate_expr_source/filtered_clips/`. Both variants (GT and Marigold)
train and evaluate on exactly this filtered subset, so the comparison is fair
even when only a fraction of the full dataset has been processed.

## Running

```bash
# From the repo root:
PYTHONPATH=. python experiments/ablate_expr_source/run.py gt_baseline
PYTHONPATH=. python experiments/ablate_expr_source/run.py marigold
PYTHONPATH=. python experiments/ablate_expr_source/run.py both   # runs in sequence
```

Outputs land under `outputs/ablate_expr_source/<variant>/run_<timestamp>/`.
A symlink `experiments/ablate_expr_source/outputs` is created lazily on first
run so the artifacts are browsable from within the experiment folder.

Each run directory contains `config_resolved.yaml` — the fully-merged config
that actually trained the checkpoint, captured for reproducibility.

## What to measure

- Visual reconstruction quality (FID, SSIM) on held-out val.
- Expression fidelity vs GT deformation ground truth.
- Temporal coherence — per-frame Marigold inference is expected to inherit
  coherence from the input video; quantify any introduced jitter.