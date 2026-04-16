# Ablate Audio Cross-Attention

Does wav2vec2 cross-attention materially improve lip sync and expressiveness,
or is the Marigold spatial deformation signal alone sufficient?

Both variants use the paper's proposed spatial conditioning: Marigold-generated
deformation (4ch: 3ch deformation + 1ch ref_mask) with uniform loss. The only
axis that changes is audio cross-attention on/off.

## Configs

```
experiments/ablate_audio/configs/
├── with_audio.yaml    # base + deform_only + marigold (full Marionette pipeline)
└── no_audio.yaml      # base + deform_only + marigold + audio/off
```

The `audio/off` overlay sets both `audio_encoder_config: null` and
`unet_config.params.use_audio_context: false`. `THDiffusion` validates that
these two flags agree at construction time and raises a clear error on
mismatch.

## Prerequisites

Same as `ablate_expr_source/marigold` — the Marigold deformation cache must be
populated:

```bash
PYTHONPATH=. python scripts/cache/cache_marigold_deform.py \
    --checkpoint <path/to/marigold_checkpoint.pt> \
    --output_dir data/derived/marigold_deform
```

## Fair comparison via clip filtering

Uses the same identity-based filtering as `ablate_expr_source`: train/val
splits are built from the clips with cached Marigold deformations, split by
identity (90/10), no identity leakage.

## Running

```bash
# From repo root:
PYTHONPATH=. python experiments/ablate_audio/run.py with_audio
PYTHONPATH=. python experiments/ablate_audio/run.py no_audio
PYTHONPATH=. python experiments/ablate_audio/run.py all

# Multi-GPU:
PYTHONPATH=. python experiments/ablate_audio/run.py with_audio --gpus 0 1 2 3
```

Outputs land under `outputs/ablate_audio/<variant>/run_<timestamp>/`.
A symlink `experiments/ablate_audio/outputs` is created lazily on first run.

## What to measure

- **Lip sync** — SyncNet offset/confidence against the driver's audio. The
  `with_audio` variant should score higher since it directly conditions on
  the audio signal; the `no_audio` variant must infer lip shape entirely from
  the spatial deformation map.
- **Expression fidelity** — per-frame L1/SSIM of the generated mouth region
  against the driver's mouth. Audio may help resolve ambiguities in the
  deformation signal (e.g. distinguishing "p" from "b" shapes).
- **Temporal smoothness** — jitter metrics (inter-frame optical flow variance).
  Audio cross-attention provides temporal grounding; its absence may increase
  frame-to-frame flicker around the lips.
