# Marigold-Style Deformation Map Generation

Generate FLAME deformation map videos from natural talking-head videos using a two-stage training approach adapted from Marigold (Ke et al., CVPR 2024).

## Motivation

Direct end-to-end fine-tuning of video diffusion models to generate deformation maps fails — the pretrained model retains a strong natural video bias. Marigold solved the analogous problem in the image domain (RGB → depth) by conditioning the denoiser on the input image via channel concatenation.

**The additional challenge in video:** the model must simultaneously learn spatial mapping (face → deformation) and temporal coherence. This dual objective is too difficult in a single stage.

**Our solution (inspired by Wan VAE's staged inflation):** decompose training into two stages:

1. **Stage 1 (Spatial):** Run the DiT at T=1 (single frame) and train on ~900k image pairs. Temporal attention becomes a no-op. The model learns purely spatial mapping.
2. **Stage 2 (Temporal):** Load Stage 1 weights into the full video DiT and fine-tune on video pairs (T=81). The model only needs to learn temporal dynamics on top of an already-strong spatial prior.

## Architecture

The DiT's `patch_embedding` Conv3d is doubled from 16 to 32 input channels (Marigold's weight duplication trick). At every denoising step, `[noisy_target_deform_latent, clean_natural_video_latent]` are concatenated along the channel dimension. The natural video provides spatiotemporal anchoring. Noise is added only to the target.

Both videos are encoded through the same frozen Wan VAE with the same normalization statistics. Flow matching with velocity prediction: `v = noise - clean`, `x_t = (1-t)*clean + t*noise`.

## Stage 1: Spatial Training

```bash
PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
    marigold_training/scripts/train_spatial.py \
    --config marigold_training/configs/spatial_config.yaml
```

| Parameter | Value | Notes |
|---|---|---|
| Temporal frames | 1 | Single frame (deflated mode) |
| Training pairs | ~900k | 7150 clips x ~127 frames, random sampling |
| Batch size | 8 per GPU | T=1 is cheap |
| Learning rate | 1e-5 | |
| Steps | 50,000 | |

**Convergence criteria:** Given a face frame, the model generates a deformation map that is spatially correct (deformations in the right facial regions). Do not over-train — the goal is a strong spatial prior, not perfect per-frame accuracy.

## Stage 2: Temporal Training

```bash
# Edit temporal_config.yaml to set stage1_checkpoint first
PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
    marigold_training/scripts/train_temporal.py \
    --config marigold_training/configs/temporal_config.yaml
```

| Parameter | Value | Notes |
|---|---|---|
| Temporal frames | 81 | 4k+1 with k=20, latent T=21 |
| Training clips | ~7150 | Video pairs |
| Batch size | 1 per GPU | Video is memory-heavy |
| Learning rate | 3e-6 | Lower than Stage 1 to preserve spatial prior |
| Steps | 25,000 | |
| Prerequisite | Stage 1 checkpoint | Set `stage1_checkpoint` in config |

Stage 1 weights transfer directly — the Conv3d already has the correct 32-channel input shape. Temporal attention weights retain Wan's pretrained temporal priors (they were effectively frozen at T=1 in Stage 1).

## Inference

Works with checkpoints from either stage. Stage 1 checkpoints generate single frames, Stage 2 generates videos.

```bash
# Single prompt:
PYTHONPATH=. python marigold_training/scripts/inference.py \
    --clip_id CLIP_ID \
    --checkpoint outputs/marigold_temporal/run_YYYYMMDD/step_NNNNNN \
    --prompt "A person says: '...' The delivery is calm and measured."

# Batch:
PYTHONPATH=. python marigold_training/scripts/inference.py \
    --prompts marigold_training/configs/eval_prompts.json \
    --clip_id CLIP_ID \
    --checkpoint outputs/marigold_temporal/run_YYYYMMDD/step_NNNNNN
```

Uses Euler ODE integration from t=1 (noise) to t=0 (data). The clean natural video latent is re-concatenated at every step. Classifier-free guidance via `--guidance_scale`.

## Codebase Structure

```
marigold_training/
├── scripts/
│   ├── train_spatial.py          # Stage 1: single-frame training (T=1)
│   ├── train_temporal.py         # Stage 2: video training (T=81, loads Stage 1)
│   └── inference.py              # Euler sampling (works for both stages)
├── src/
│   ├── marigold_model.py         # double_patch_embedding (16→32ch)
│   ├── frame_pair_dataset.py     # Stage 1: (frame, deform_frame, text) triplets
│   ├── marigold_dataset.py       # Stage 2: (video, deform_video, text) triplets
│   ├── collate.py                # Variable-length text embedding padding
│   ├── checkpoint.py             # Checkpoint saving
│   ├── reshape.py                # Pseudo-video padding (4k+1)
│   └── vis.py                    # Deformation map visualization
├── configs/
│   ├── spatial_config.yaml       # Stage 1 hyperparameters
│   └── temporal_config.yaml      # Stage 2 hyperparameters
└── README.md
```

## References

- Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024) — [Paper](https://arxiv.org/abs/2312.02145), [Code](https://github.com/prs-eth/Marigold)
- Wan Technical Report (arxiv 2503.20314) — Progressive image-video joint training and VAE inflation strategy
