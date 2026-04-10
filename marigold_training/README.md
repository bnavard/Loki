# Marigold-Style Deformation Map Generation

Generate FLAME deformation maps from natural talking-head face frames, adapting the Marigold depth estimation approach (Ke et al., CVPR 2024) to face deformation prediction.

## Motivation

Direct text-conditioned generation of deformation maps from scratch fails — the pretrained diffusion model retains a strong natural image bias. Marigold solved the analogous problem for depth estimation by conditioning the denoiser on the input RGB image via channel concatenation, using null text (unconditional generation).

We apply the same principle: given a natural face frame, generate the corresponding FLAME deformation map. The model learns a purely spatial mapping — no temporal modeling needed, since per-frame inference on video produces temporally coherent results when the input video is coherent.

## Architecture

**Base model:** SD3.5 Medium (`stabilityai/stable-diffusion-3.5-medium`) — a 24-layer MMDiT with ~1.5B parameters, using rectified flow.

**Input layer modification:** The transformer's `pos_embed.proj` Conv2d is doubled from 16 to 32 input channels using Marigold's weight duplication trick (clone weights, repeat along input channel dim, halve to preserve activation magnitude). At every denoising step, `[noisy_target_deform_latent, clean_natural_frame_latent]` are concatenated along the channel dimension.

**VAE:** SD3's AutoencoderKL with 16 latent channels, 8x spatial compression. Normalization: `latent = (raw - shift_factor) * scaling_factor` where `shift_factor=0.0609`, `scaling_factor=1.5305`.

**Text conditioning:** Null embeddings (unconditional), following the original Marigold paper. The model relies entirely on the visual conditioning from the input frame.

**Training:** Full fine-tuning of all transformer parameters. Rectified flow with velocity prediction.

```
Natural face frame                      Null text
    |                                      |
    v                                      v
[Frozen SD3 VAE Encoder]            [Zero embeddings]
    |                                      |
    | clean_latent [16ch]                  |
    |                                      |
    +--- concat ---+                       |
    |              |                       |
    v              v                       v
+----------------------------------------------+
|  SD3.5 MMDiT (24 layers, ~1.5B params)       |
|  pos_embed.proj: Conv2d(32, 1536, k=2, s=2)  |
|  Rectified flow velocity prediction          |
+----------------------------------------------+
    | predicted velocity (16ch)
    v
  Euler integration -> denoised target latent
    |
    v
[Frozen SD3 VAE Decoder]
    |
    v
  Deformation map [3, H, W]
```

## Training

```bash
cd <repo_root>

# Multi-GPU:
PYTHONPATH=. accelerate launch \
    --num_processes 4 --mixed_precision bf16 \
    marigold_training/scripts/train_spatial.py \
    --config marigold_training/configs/spatial_config.yaml

# Resume from checkpoint:
PYTHONPATH=. accelerate launch \
    --num_processes 4 --mixed_precision bf16 \
    marigold_training/scripts/train_spatial.py \
    --config marigold_training/configs/spatial_config.yaml \
    --resume outputs/marigold_spatial/run_YYYYMMDD/step_NNNNNN
```

| Parameter | Value | Notes |
|---|---|---|
| Model | SD3.5 Medium (~1.5B) | 24-layer MMDiT |
| Training pairs | ~900k | 7150 clips x ~127 frames, random frame sampling |
| Batch size | 32 per GPU | Image-level, no temporal dim |
| Learning rate | 3.5e-5 | IterExponential: warmup 100 → decay to 1% over 50k |
| Steps | 50,000 | |
| Text conditioning | Null | Unconditional (per original Marigold) |
| Fine-tuning | Full | All transformer parameters trainable |

**Checkpoints** save the full training state (model weights, optimizer, LR scheduler, RNG state) for exact resume. A learning rate plot and periodic multi-sample evaluation grids are saved to the run directory.

## Inference

Processes a video clip frame-by-frame. Each frame is independently denoised conditioned on the corresponding natural face frame.

```bash
PYTHONPATH=. python marigold_training/scripts/inference.py \
    --clip_id CLIP_ID \
    --checkpoint outputs/marigold_spatial/run_YYYYMMDD/step_NNNNNN
```

Output:
- `side_by_side.mp4` — input face | predicted deform | ground truth deform
- `predicted.mp4` — predicted deformation only
- `ground_truth.mp4` — ground truth deformation only

Uses 50-step Euler ODE integration from t=1 (noise) to t=0 (data). The clean natural frame latent is concatenated at every denoising step.

## Codebase Structure

```
marigold_training/
├── scripts/
│   ├── train_spatial.py          # Training loop (SD3.5 + Marigold conditioning)
│   └── inference.py              # Frame-by-frame inference on video clips
├── src/
│   ├── marigold_model.py         # double_input_channels (16→32ch, SD3 + Wan)
│   ├── frame_pair_dataset.py     # (natural_frame, deform_frame) pairs
│   ├── collate.py                # Batch collation
│   ├── checkpoint.py             # Full training state save/load
│   └── vis.py                    # Deformation visualization + eval grids
├── configs/
│   └── spatial_config.yaml       # SD3.5 Medium training hyperparameters
└── README.md
```

## References

- Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024) — [Paper](https://arxiv.org/abs/2312.02145), [Code](https://github.com/prs-eth/Marigold)
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (ICML 2024) — [Paper](https://arxiv.org/abs/2403.03206)
