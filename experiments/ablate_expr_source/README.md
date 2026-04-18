# Ablate Expression Map Source

What spatial conditioning signal gives the video diffusion model the most
useful information about facial dynamics? This experiment compares four
sources, all with uniform diffusion loss:

| Variant | Channels | Conditioning signal |
|---|---|---|
| `gt_full` | 46 | Full FLAME: 42ch vertex positional encoding + 3ch deformation + 1ch ref_mask |
| `gt_baseline` | 4 | FLAME deformation only: 3ch deformation heatmap + 1ch ref_mask |
| `marigold` | 4 | Marigold-generated deformation: 3ch learned deformation + 1ch ref_mask |
| `driving_video` | 4 | Raw driving video at 64x64: 3ch RGB + 1ch ref_mask |

The three 4ch variants share the same UNet architecture (`deform_only`
overlay, `condition_channels=4`), making the comparison fair on channel budget.
`gt_full` (46ch) serves as the upper-bound reference — it gives the UNet both
WHERE the face is (vertex positions) and HOW it moves (deformation), while the
4ch variants only carry the motion signal.

The `driving_video` variant tests whether the structured FLAME decomposition
adds value over simply showing the model the face at low resolution. The
driving frames carry implicit motion and appearance but don't decompose
deformation magnitude the way the expression map does.

## Configs

```
experiments/ablate_expr_source/configs/
├── gt_full.yaml          # base (no overlays — 46ch is the default)
├── gt_baseline.yaml      # base + overlays/conditioning/deform_only
├── marigold.yaml         # base + overlays/conditioning/deform_only
│                         #      + overlays/expr_source/marigold
└── driving_video.yaml    # base + overlays/conditioning/deform_only
                          #      + overlays/expr_source/driving_video
```

## Prerequisites

The `marigold` variant requires pre-cached deformation videos:

```bash
PYTHONPATH=. python scripts/cache/marigold_deform/cache.py \
    --checkpoint <path/to/marigold_checkpoint.pt> \
    --output_dir data/derived/marigold_deform
```

The other three variants (`gt_full`, `gt_baseline`, `driving_video`) need only
the standard FLAME tracking data in `data/flowface/`.

## Fair comparison via clip filtering

Not all clips have Marigold-cached deformations. To ensure all four variants
train on exactly the same data, `run.py` builds train/val splits (90/10) from
only the clips that have a `deformation.mp4` in the Marigold cache directory.
The filtered lists are written to
`outputs/ablate_expr_source/filtered_clips/` and reused across all variants.

When resuming a run (`--resume`), the existing filtered lists are reused to
guarantee the same dataset that produced the checkpoint.

## Running

```bash
# From the repo root — single variant:
PYTHONPATH=. python experiments/ablate_expr_source/run.py gt_baseline
PYTHONPATH=. python experiments/ablate_expr_source/run.py gt_full
PYTHONPATH=. python experiments/ablate_expr_source/run.py marigold
PYTHONPATH=. python experiments/ablate_expr_source/run.py driving_video

# All four in sequence:
PYTHONPATH=. python experiments/ablate_expr_source/run.py all

# Multi-GPU:
PYTHONPATH=. python experiments/ablate_expr_source/run.py gt_baseline --gpus 0 1 2 3
```

Outputs land under `outputs/ablate_expr_source/<variant>/run_<timestamp>/`.
A symlink `experiments/ablate_expr_source/outputs` is created lazily on first
run. Each run directory contains `config_resolved.yaml` — the fully-merged
config captured for reproducibility.

## What to measure

- Visual reconstruction quality (FID, SSIM) on held-out val across all four.
- Expression fidelity: how well does each conditioning source preserve the
  target facial dynamics (mouth shape, brow motion, eye gaze)?
- Temporal coherence: does the driving-video variant produce smoother or
  jerkier output than the structured deformation variants?
- Ablation ranking: gt_full > gt_baseline > marigold > driving_video is the
  expected ordering. Deviations are the interesting finding.
