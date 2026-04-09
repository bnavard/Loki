# Marigold-Style Deformation Map Generation

Generate FLAME deformation map videos from natural talking-head videos, adapting the Marigold depth estimation approach (Ke et al., CVPR 2024) to video-to-video generation.

## Motivation

Direct text-to-deformation generation struggles because the pretrained video DiT retains a strong natural video prior. Instead of fighting that prior, we leverage it: the model receives the natural video as a conditioning signal and learns to produce the corresponding structured deformation map. This is the same insight behind Marigold, which repurposes a pretrained image diffusion model for depth estimation by conditioning on the input RGB image.

## Architecture

```
Natural video (RGB)                    Text prompt
    │                                      │
    ▼                                      ▼
┌──────────┐                        ┌──────────────┐
│ Wan VAE  │ (frozen)               │ UMT5 Encoder │ (cached)
│ Encoder  │                        └─────┬────────┘
└────┬─────┘                              │
     │ clean_latent [16ch]                │ text embeddings
     │                                    │
     ├─── concat ───┐                     │
     │              │                     │
     ▼              ▼                     ▼
┌────────────────────────────────────────────────┐
│  Wan DiT (1.3B) — patch_embedding doubled      │
│  Input: [noisy_target_16ch | clean_cond_16ch]  │
│  = 32 channels                                 │
│  Flow matching velocity prediction             │
└─────┬──────────────────────────────────────────┘
      │ predicted velocity (16ch)
      ▼
  Euler integration → denoised target latent
      │
      ▼
┌──────────┐
│ Wan VAE  │ (frozen)
│ Decoder  │
└────┬─────┘
     │
     ▼
  Deformation map video [T, 3, H, W]
```

### Key Design Decisions

**Input layer modification:** The transformer's `patch_embedding` Conv3d is doubled from 16 to 32 input channels. Following Marigold's `_replace_unet_conv_in()`, the original weights are cloned, repeated along the input channel dimension, and halved to preserve activation magnitude at initialization. This is the only architectural change.

**Channel concatenation:** At each training step, `[noisy_target_deform_latent, clean_natural_video_latent]` are concatenated along the channel dimension. The natural video latent is always clean — noise is only added to the target.

**Both videos use the same VAE:** Natural video and deformation map are both encoded through the same frozen Wan VAE with the same normalization statistics. Both are 3-channel videos at the same resolution.

**Flow matching:** Velocity prediction loss `v = noise - clean`, with interpolant `x_t = (1-t) * clean + t * noise`.

**Full fine-tuning:** All transformer parameters are trainable (not LoRA). The 1.3B model is small enough for this.

## Training

```bash
cd /data/pouyan/baseline/repository/cap4d

PYTHONPATH=. accelerate launch --num_processes 4 --mixed_precision bf16 \
    marigold_training/scripts/train.py \
    --config marigold_training/configs/train_config.yaml
```

Data is computed on the fly: for each clip, the dataset computes the 3ch deformation map from `fit.npz` via FLAME + PyTorch3D rasterization, and loads the corresponding natural video frames with the same face crop. Both are VAE-encoded in the training loop.

## Inference

```bash
# Single prompt:
PYTHONPATH=. python marigold_training/scripts/inference.py \
    --clip_id CLIP_ID \
    --checkpoint outputs/marigold_deform/run_YYYYMMDD/step_NNNNNN \
    --prompt "A person says: '...' The delivery is calm and measured."

# Batch from JSON:
PYTHONPATH=. python marigold_training/scripts/inference.py \
    --prompts marigold_training/configs/eval_prompts.json \
    --clip_id CLIP_ID \
    --checkpoint outputs/marigold_deform/run_YYYYMMDD/step_NNNNNN
```

Inference uses Euler ODE integration from t=1 (noise) to t=0 (data). At each step, the clean natural video latent is re-concatenated with the current noisy deformation latent. Classifier-free guidance is supported via `--guidance_scale`.

Output per prompt: predicted deformation video, ground truth deformation, and visualization mp4s.

## Codebase Structure

```
marigold_training/
├── scripts/
│   ├── train.py              # Training loop
│   └── inference.py          # Euler sampling with video conditioning
├── src/
│   ├── marigold_model.py     # double_patch_embedding (16→32ch)
│   ├── marigold_dataset.py   # (natural_video, deform_video, text) triplets
│   ├── collate.py            # Variable-length text embedding padding
│   ├── checkpoint.py         # Checkpoint saving
│   ├── reshape.py            # Pseudo-video padding (4k+1)
│   └── vis.py                # Deformation map visualization
├── configs/
│   └── train_config.yaml     # Wan2.1-T2V-1.3B, full fine-tuning
└── README.md
```

## References

- Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024) — [Paper](https://arxiv.org/abs/2312.02145), [Code](https://github.com/prs-eth/Marigold)
