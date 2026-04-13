# Ablate Expression Map Source (GT vs Marigold)

How much reconstruction quality does Marionette lose when the 3ch expression
deformation comes from the Marigold generator instead of the ground-truth
FLAME rasterization?

## Setup

Both runs use the same Marionette architecture — only the deformation-map
source changes. Two knobs must be set together, in both the dataset block
and the conditioning block of the YAML:

```yaml
train_dataset:
  params:
    expression_source: "marigold"                         # "gt" | "marigold"
    marigold_deform_root: "data/derived/marigold_deform"  # required for "marigold"

model:
  params:
    cond_stage_config:
      params:
        expression_source: "marigold"
```

## Prerequisites

Populate the Marigold deformation cache before training:

```bash
PYTHONPATH=. python caching/scripts/cache_marigold_deform.py \
    --checkpoint <marigold_checkpoint.pt> \
    --output_dir data/derived/marigold_deform
```

This writes `data/derived/marigold_deform/{clip_id}/deformation.mp4` for every
clip in the manifest. The dataset decodes the requested frames on the fly.

## Channel layout in Marigold mode

The Marigold module produces only the 3ch deformation — not the 42ch
positional encoding. Conditioning is therefore always 4ch
`(3 deform + 1 ref_mask)` in Marigold mode, regardless of other flags.

## What to measure

- Visual reconstruction quality (FID, SSIM) on held-out val.
- Expression fidelity vs GT deformation ground truth.
- Temporal coherence — per-frame Marigold inference is expected to inherit
  coherence from the input video; quantify any introduced jitter.