# INSTRUCTIONS: Staged Marigold Training for Video → Expression Map Video

> **⚠️ SUPERSEDED (2026-04-13).** This document prescribes a two-stage plan on
> **Wan2.1-T2V-1.3B**. The actual implementation at `marigold_training/` uses
> **SD3.5 Medium** and only implements per-frame (Stage 1) training. Stage 2
> video inflation was skipped — per-frame inference on a temporally coherent
> natural video produces a coherent deformation-map video. Treat this file as
> historical context. For the current architecture, see `marigold_training/README.md`.

## Motivation

Our previous attempts to fine-tune pretrained video diffusion models end-to-end to generate FLAME expression dense field maps have failed. The models retain strong bias toward natural video generation — even with full parameter fine-tuning, the output is either natural-looking video or noise, never the structured expression maps we need.

The **Marigold** paper (Ke et al., CVPR 2024 Oral, Best Paper Candidate; https://arxiv.org/abs/2312.02145) solved an analogous problem in the **image** domain: repurposing Stable Diffusion 2 to generate depth maps, surface normals, and other non-natural structured outputs from natural image inputs. We adapt their approach to our video domain.

**However, directly applying Marigold to video end-to-end is significantly harder than in the image domain**, primarily due to temporal alignment — the model must simultaneously learn (1) the spatial mapping from natural appearance to deformation maps, and (2) temporal coherence of deformation dynamics. This dual objective is too difficult to optimize in a single stage.

**Our solution:** Inspired by the Wan VAE paper's three-stage training strategy — where a 2D image VAE is first trained, then "inflated" into a 3D causal VAE to provide a strong spatial compression prior — we adopt an analogous staged approach for the denoising model itself:

1. **Stage 1 (Deflation):** Run the Wan2.1-T2V-1.3B DiT as a single-frame (T=1) model and train it Marigold-style on **image pairs** (face frame → deformation map frame). This teaches the spatial mapping.
2. **Stage 2 (Inflation):** Transfer the Stage 1 spatial weights back into the full video DiT and fine-tune on **video pairs** (face video → deformation map video). The model now only needs to learn temporal dynamics on top of an already-strong spatial prior.

**Reference implementation:** https://github.com/prs-eth/Marigold
**Diffusers pipeline:** `https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py`
**Paper Section 3.2:** Adapted denoising U-Net, weight duplication formula for the input layer
**Marigold v1.1 (arxiv 2505.09358):** Zero-SNR and trailing timestep improvements

---

## Marigold's Core Technique (Recap)

Marigold starts from **Stable Diffusion 2** and adapts it:

1. **Both input and target are images processed through the same frozen VAE:**
   - Input RGB image `x` → VAE encoder → input latent `z^(x)` (4 channels)
   - Target depth/normals `d` → VAE encoder → target latent `z^(d)` (4 channels)

2. **Concatenate along the channel dimension at every denoising step:**
   - `z_input = cat(z^(d)_t, z^(x))` — noisy target latent + clean input latent → 8 channels

3. **Modify only the input layer of the U-Net:**
   - First conv layer expanded from 4 to 8 input channels
   - Pretrained weight tensor is **duplicated** and **divided by 2** (preserves activation magnitudes)
   - **No other layers are modified architecturally**

4. **Fine-tune only the U-Net.** VAE frozen. Text encoder frozen. Marigold passes **empty text embeddings**.

5. **Noise is added only to the target latent.** The input image latent is always clean.

6. **Inference:** Start from pure noise for the target latent. At each step, re-concatenate the clean input latent with the current noisy target latent, predict noise/velocity, step the scheduler.

---

## Why Two Stages?

| Single-stage (previous attempt) | Two-stage (this approach) |
|---|---|
| Model must learn spatial mapping AND temporal coherence simultaneously | Stage 1 isolates spatial mapping; Stage 2 adds temporal coherence |
| Video-level noise and denoising makes the optimization landscape harder | Image-level training is well-understood and stable (Marigold proved this) |
| ~7k video clips for training | Stage 1: ~900k image pairs (7150 clips × ~127 frames). Stage 2: 7150 video clips with a warm-started model |
| No prior for what deformation maps look like | Stage 2 starts from a model that already understands deformation map structure |

The Wan technical report confirms this philosophy works: they "first conduct pre-training on low-resolution images, followed by multi-stage joint optimization of images and videos." We apply the same principle — learn the easier task first (images), then extend to the harder task (videos).

---

## Model

**Wan2.1-T2V-1.3B** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) — 1.3B parameters, flow matching (velocity prediction).

Key specs:
- DiT architecture: 30 transformer layers, dim=1536, 12 attention heads, ffn_dim=8960
- `in_channels=16` (matching VAE `z_dim=16`)
- `patch_embedding`: Conv3d that maps from 16 latent channels to hidden dim
- VAE: `AutoencoderKLWan`, `z_dim=16`, compression 4×8×8 (T×H×W), frame count rule: `4k+1`

---

## Stage 1: Image-Level Marigold (Deflation)

### Concept

Run Wan2.1-T2V-1.3B with **T=1** (a single frame). The VAE's `4k+1` frame rule gives `T=1` for `k=0`. When the DiT receives a single temporal frame:
- Temporal attention becomes a no-op (self-attention over a sequence of length 1)
- 3D convolutions in the patch embedding operate on a single temporal slice
- The model effectively functions as a text-to-image diffusion model

This is the "deflation" — no architectural changes needed, just set the temporal dimension to 1.

### Architecture Modification (Marigold Input Doubling)

Following Marigold, we modify **only the input layer** of the DiT to accept the concatenated conditioning:

```python
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# --- Marigold's input layer replacement adapted for Wan's DiT ---
# Reference: Marigold's _replace_unet_conv_in() method

# Identify the patch embedding layer
# IMPORTANT: Verify the actual attribute name by inspecting:
#   for name, mod in transformer.named_modules():
#       if isinstance(mod, nn.Conv3d):
#           print(name, mod)

original_conv = transformer.patch_embedding  # Conv3d(16, hidden_dim, ...)
_weight = original_conv.weight.clone()       # [out_ch, 16, kT, kH, kW]
_bias = original_conv.bias.clone() if original_conv.bias is not None else None

# Repeat along input channel dim (dim=1) to double: 16 → 32
_weight = _weight.repeat((1, 2, 1, 1, 1))   # [out_ch, 32, kT, kH, kW]
_weight *= 0.5                                # halve to preserve activation magnitudes

# Create new Conv3d with doubled input channels
_new_patch_embed = nn.Conv3d(
    in_channels=original_conv.in_channels * 2,  # 32
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
)
_new_patch_embed.weight = Parameter(_weight)
if _bias is not None:
    _new_patch_embed.bias = Parameter(_bias)

transformer.patch_embedding = _new_patch_embed

# Update config to reflect new input channels
transformer.config['in_channels'] = original_conv.in_channels * 2
```

**This is the ONLY architectural change.** All other layers remain identical to the pretrained model.

### Training Data (Stage 1)

Each training sample is a **single-frame triplet** from the same clip:

- **`natural_frame`**: A single frame from the talking-head video `data/talkvid/talkvid/{clip_id}.mp4` — tensor `[3, 1, H, W]` in [-1, 1]
- **`target_frame`**: Corresponding single-frame deformation map `[3, 1, H, W]` in [-1, 1] — generated from the FLAME pipeline (`THConditioning.forward()`, channels `[42:45]` of `pos_enc`), same frame index as the natural frame
- **`text`**: Prosody caption from `data/derived/captions/{clip_id}.json`

**Dataset size:** 7150 clips × ~127 frames/clip ≈ **~900k image pairs**. This is substantially more data than Marigold used for depth estimation.

**Frame sampling strategy:** During training, randomly sample one frame index per clip per epoch. Over many epochs, all frames get covered. This is simpler and more memory-efficient than loading all frames upfront.

**Critical:** The natural frame and deformation map frame must be from the **same clip** and the **same frame index**.

### Training Loop (Stage 1)

```python
# === Setup ===
from diffusers import WanPipeline
import torch
import torch.nn.functional as F

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

transformer = pipe.transformer
vae = pipe.vae
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

# Get latent normalization stats
latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

# Freeze VAE and text encoder
vae.requires_grad_(False)
vae.eval()
text_encoder.requires_grad_(False)

# === Modify input layer (Marigold's trick) ===
# [Code from Architecture Modification section above]
# After this, transformer.patch_embedding accepts 32 channels instead of 16

# Move normalization stats to device
latents_mean = latents_mean.to(device, dtype=torch.bfloat16)
latents_std = latents_std.to(device, dtype=torch.bfloat16)

# === Training loop ===
for batch in dataloader:
    natural_frame = batch["natural_frame"]   # [B, 3, 1, H, W] single face frame, [-1, 1]
    target_frame = batch["target_frame"]     # [B, 3, 1, H, W] single deformation map, [-1, 1]
    text = batch["text"]                     # list of caption strings

    with torch.no_grad():
        # Encode text
        text_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        text_embeds = text_encoder(**text_inputs.to(device)).last_hidden_state

        # Encode natural frame through VAE (CONDITIONING — always clean)
        # VAE expects [B, C, T, H, W] — T=1 here
        natural_f32 = natural_frame.to(device, dtype=torch.float32)  # VAE needs float32
        natural_latent = vae.encode(natural_f32).latent_dist.mode()
        # Shape: [B, 16, 1, h, w] (T_latent=1 for single frame)

        # Encode target deformation map frame through VAE (TARGET — will be noised)
        target_f32 = target_frame.to(device, dtype=torch.float32)
        target_latent = vae.encode(target_f32).latent_dist.mode()
        # Shape: [B, 16, 1, h, w]

        # Normalize both latents with pretrained stats
        natural_latent = (natural_latent - latents_mean) / latents_std
        target_latent = (target_latent - latents_mean) / latents_std

    # Cast to training dtype
    natural_latent = natural_latent.to(dtype=torch.bfloat16)
    target_latent = target_latent.to(dtype=torch.bfloat16)

    # === Flow matching: noise the TARGET latent only ===
    B = target_latent.shape[0]
    noise = torch.randn_like(target_latent)
    t = torch.rand(B, device=device, dtype=torch.bfloat16)
    t_expand = t[:, None, None, None, None]

    noisy_target = (1 - t_expand) * target_latent + t_expand * noise
    target_velocity = noise - target_latent

    # === Marigold's concatenation ===
    # [noisy_target | clean_natural_frame] along channel dimension
    model_input = torch.cat([noisy_target, natural_latent], dim=1)
    # Shape: [B, 32, 1, h, w]

    # === CFG dropout: randomly drop text conditioning 10% ===
    drop_mask = torch.rand(B, device=device) < 0.1
    if drop_mask.any():
        null_embeds = torch.zeros_like(text_embeds)
        text_embeds = torch.where(
            drop_mask[:, None, None].expand_as(text_embeds),
            null_embeds, text_embeds,
        )

    # === Forward: DiT predicts velocity ===
    velocity_pred = transformer(
        model_input,
        timestep=t,
        encoder_hidden_states=text_embeds,
    ).sample

    # === Loss (fp32 for precision) ===
    loss = F.mse_loss(velocity_pred.float(), target_velocity.float())

    loss.backward()
    # ... gradient clipping, optimizer step, lr scheduler step, zero_grad
```

**Important notes for Stage 1:**
- The VAE with T=1 input: verify that `vae.encode()` accepts `[B, 3, 1, H, W]`. The Wan causal VAE should handle this since `4k+1` with `k=0` gives T=1, but test this explicitly.
- The DiT with T_latent=1: the temporal attention layers will compute self-attention over a single token in the temporal dimension — effectively a no-op. This is expected and correct.
- Resolution: **512×512** (matching the original MD's target).

### Stage 1 Convergence Criteria

Stage 1 is considered converged when:
- Given a natural face frame, the model generates a deformation map frame that is spatially correct (deformations in the right facial regions)
- Visual inspection: generated deformation maps look structurally similar to ground truth for held-out frames
- Quantitative: MSE between generated and GT deformation maps on a validation set decreases and plateaus

**Do not over-train Stage 1.** The goal is a strong spatial prior, not perfect per-frame accuracy. Over-training on single frames may cause the model to memorize frame-level details that hurt temporal generalization in Stage 2.

---

## Stage 2: Video-Level Fine-Tuning (Inflation)

### Concept

Transfer the Stage 1 weights into the full video model and fine-tune on video pairs. The key design decision is **how to inflate** the single-frame weights back into the video architecture.

### Weight Transfer Strategy

The Wan DiT architecture uses the same weights for both single-frame and multi-frame operation — the temporal attention and the temporal components of 3D convolutions simply have more tokens/frames to process. This means:

1. **All spatial weights transfer directly.** The self-attention Q/K/V projections, cross-attention layers, feed-forward layers, and layer norms are shared across spatial and temporal dimensions. The Stage 1 training updated these weights to understand deformation map structure — this transfers perfectly.

2. **Temporal attention weights retain their pretrained values.** During Stage 1 (T=1), temporal attention was a no-op, so these weights were not meaningfully updated. They retain Wan's pretrained temporal priors for natural video motion, which is a reasonable starting point for temporal coherence of deformation maps.

3. **The patch embedding (Conv3d) is already 3D.** In Stage 1 with T=1, the temporal kernel dimension was operating on a single slice. The learned spatial filter patterns transfer directly. For the temporal kernel weights, there are two options:

   **Option A (Simple — recommended first):** Load the Stage 1 checkpoint directly. The Conv3d weights from Stage 1 already have the correct shape (including the doubled 32-channel input from Marigold's trick). The temporal kernel weights were updated during Stage 1 training even though T=1, because gradients still flow through them. Start Stage 2 training from this checkpoint directly.

   **Option B (Wan-VAE-style inflation):** If Option A struggles with temporal coherence, apply the Wan VAE inflation trick: copy the Stage 1 spatial weights into the center temporal slice of fresh 3D kernels, zero-initialize the remaining temporal slices. This forces the model to start from pure spatial prediction and gradually learn temporal dynamics.

```python
# Option A: Direct loading (try first)
stage1_state_dict = torch.load("stage1_checkpoint.pt")
transformer.load_state_dict(stage1_state_dict, strict=True)
# Proceed directly to Stage 2 training on video pairs

# Option B: Explicit temporal inflation (if Option A struggles)
# Only needed for Conv3d layers if you want to reset temporal kernels
for name, module in transformer.named_modules():
    if isinstance(module, nn.Conv3d) and module.kernel_size[0] > 1:
        with torch.no_grad():
            # Copy Stage 1 weights into center temporal slice
            kT = module.kernel_size[0]
            center = kT // 2
            new_weight = torch.zeros_like(module.weight)
            new_weight[:, :, center, :, :] = module.weight[:, :, center, :, :]
            module.weight.copy_(new_weight)
```

### Training Data (Stage 2)

Each training sample is a **video triplet** from the same clip:

- **`natural_video`**: Talking-head video from `data/talkvid/talkvid/{clip_id}.mp4` — `[3, T, H, W]` in [-1, 1]
- **`target_video`**: Expression deformation map video `[3, T, H, W]` in [-1, 1] — generated from the FLAME pipeline, temporally aligned frame-by-frame
- **`text`**: Prosody caption from `data/derived/captions/{clip_id}.json`

**Frame count:** Must satisfy the `4k+1` rule. Recommended: **T=81 frames** (k=20, ~3.2s at 25fps). Latent temporal dimension after VAE: `(81-1)/4 + 1 = 21`.

**Critical:** Natural video and deformation map must be from the **same clip** and **temporally aligned** (frame N ↔ frame N).

### Training Loop (Stage 2)

The training loop is identical to Stage 1, except:
- Input tensors are `[B, 3, T, H, W]` with T=81 instead of T=1
- Latent tensors are `[B, 16, T_lat, h, w]` with T_lat=21 instead of 1
- Concatenated model input is `[B, 32, T_lat, h, w]`

```python
# === Stage 2 training loop (differences from Stage 1 highlighted) ===

for batch in dataloader:
    natural_video = batch["natural_video"]   # [B, 3, 81, H, W]  <-- video, not frame
    target_video = batch["target_video"]     # [B, 3, 81, H, W]
    text = batch["text"]

    with torch.no_grad():
        text_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        text_embeds = text_encoder(**text_inputs.to(device)).last_hidden_state

        natural_f32 = natural_video.to(device, dtype=torch.float32)
        natural_latent = vae.encode(natural_f32).latent_dist.mode()
        # Shape: [B, 16, 21, h, w]  <-- T_lat=21

        target_f32 = target_video.to(device, dtype=torch.float32)
        target_latent = vae.encode(target_f32).latent_dist.mode()

        natural_latent = (natural_latent - latents_mean) / latents_std
        target_latent = (target_latent - latents_mean) / latents_std

    natural_latent = natural_latent.to(dtype=torch.bfloat16)
    target_latent = target_latent.to(dtype=torch.bfloat16)

    # Flow matching (identical logic)
    B = target_latent.shape[0]
    noise = torch.randn_like(target_latent)
    t = torch.rand(B, device=device, dtype=torch.bfloat16)
    t_expand = t[:, None, None, None, None]

    noisy_target = (1 - t_expand) * target_latent + t_expand * noise
    target_velocity = noise - target_latent

    model_input = torch.cat([noisy_target, natural_latent], dim=1)
    # Shape: [B, 32, 21, h, w]

    # CFG dropout
    drop_mask = torch.rand(B, device=device) < 0.1
    if drop_mask.any():
        null_embeds = torch.zeros_like(text_embeds)
        text_embeds = torch.where(
            drop_mask[:, None, None].expand_as(text_embeds),
            null_embeds, text_embeds,
        )

    velocity_pred = transformer(
        model_input,
        timestep=t,
        encoder_hidden_states=text_embeds,
    ).sample

    loss = F.mse_loss(velocity_pred.float(), target_velocity.float())
    loss.backward()
    # ... gradient clipping, optimizer step, lr scheduler step, zero_grad
```

### Stage 2 Learning Rate

Use a **lower learning rate** than Stage 1 (e.g., 2-5× smaller). The spatial weights are already well-trained — a high learning rate risks catastrophically forgetting the spatial prior while learning temporal dynamics.

---

## Inference

At inference time, provide a natural talking-head video and a text description. The model generates the corresponding deformation map video.

```python
# 1. Encode the natural talking-head video (conditioning)
natural_5d = natural_video.unsqueeze(0).to(device, dtype=torch.float32)  # [1, 3, T, H, W]
natural_latent = vae.encode(natural_5d).latent_dist.mode()
natural_latent = (natural_latent - latents_mean) / latents_std
natural_latent = natural_latent.to(dtype=torch.bfloat16)

# 2. Encode text
text_embeds = encode_text(text_encoder, tokenizer, prompt)

# 3. Start from pure noise for the deformation map latent
x = torch.randn_like(natural_latent)  # same shape as natural_latent

# 4. Flow matching sampling: Euler integration from t=1 (noise) to t=0 (data)
timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

for i in range(num_inference_steps):
    t_current = timesteps[i]
    dt = timesteps[i + 1] - timesteps[i]  # negative

    # Marigold concatenation at every step
    model_input = torch.cat([x, natural_latent], dim=1)  # [1, 32, T_lat, h, w]

    # DiT predicts velocity
    velocity = transformer(
        model_input,
        timestep=t_current.expand(1),
        encoder_hidden_states=text_embeds,
    ).sample

    # Euler step
    x = x + velocity * dt

# 5. Denormalize
raw_latent = x * latents_std + latents_mean

# 6. Decode through VAE
deform_video = vae.decode(raw_latent.to(vae.dtype)).sample  # [1, 3, T, H, W]
```

**Classifier-Free Guidance at inference (optional):**
```python
vel_cond = transformer(model_input, t, text_embeds).sample
vel_uncond = transformer(model_input, t, null_embeds).sample
velocity = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
```

**Note:** The natural video latent is re-concatenated at **every** denoising step — it is never modified, exactly as Marigold does.

---

## Hyperparameters

### Stage 1 (Image-Level)

| Parameter | Value | Notes |
|---|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | |
| Temporal frames | 1 | Single frame (deflated mode) |
| Batch size per GPU | 8-16 | Much larger than video since T=1 |
| Gradient accumulation | 2-4 | |
| Learning rate | 1e-5 | |
| Optimizer | AdamW | Following Wan's training |
| Training steps | 20,000-50,000 | ~900k pairs, multiple epochs |
| CFG dropout | 0.1 | Drop text 10% |
| Mixed precision | bf16 | |
| Gradient clipping | 1.0 | |
| Resolution | 512×512 | |

### Stage 2 (Video-Level)

| Parameter | Value | Notes |
|---|---|---|
| Model | Stage 1 checkpoint | |
| Temporal frames | 81 | `4k+1` with k=20 |
| Batch size per GPU | 1-2 | Video is memory-heavy |
| Gradient accumulation | 4-8 | |
| Learning rate | 2e-6 to 5e-6 | Lower than Stage 1 to preserve spatial prior |
| Training steps | 10,000-25,000 | |
| CFG dropout | 0.1 | |
| Mixed precision | bf16 | |
| Gradient clipping | 1.0 | |
| Resolution | 512×512 | |

---

## What to Fine-Tune

Both stages: **full fine-tuning of the transformer** (all parameters updated by optimizer), exactly as Marigold fine-tunes the entire U-Net. VAE and text encoder are always frozen.

If full fine-tuning causes overfitting in Stage 2 (only ~7k video clips), consider LoRA on Stage 2 while keeping Stage 1 spatial weights frozen. But try full fine-tuning first.

Enable gradient checkpointing for Stage 2 to manage memory with video-length sequences.

---

## VAE Considerations

- Wan2.1 VAE: `z_dim=16`, `scale_factor_spatial=8`, `scale_factor_temporal=4`
- Frame count must satisfy `4k+1` rule: Stage 1 uses T=1 (k=0), Stage 2 uses T=81 (k=20)
- **VAE must run in float32** for encoding/decoding quality
- Normalize latents with `vae.config.latents_mean` and `vae.config.latents_std`
- Both natural video latent and deformation map latent use the **same normalization stats** (same VAE)
- **Verify T=1 encoding:** Test that `vae.encode(tensor_of_shape_[1, 3, 1, H, W])` works correctly before starting Stage 1 training

---

## Key Differences from Previous Single-Stage Approach

| Previous single-stage Marigold | This two-stage approach |
|---|---|
| Train directly on video pairs end-to-end | Stage 1: image pairs → Stage 2: video pairs |
| ~7k training samples | Stage 1: ~900k image pairs, Stage 2: ~7k video clips |
| Model must learn spatial + temporal simultaneously | Spatial mapping learned first, temporal dynamics added second |
| High risk of temporal misalignment dominating the loss | Stage 1 eliminates temporal dimension entirely |
| Single learning rate for all objectives | Stage-specific learning rates (higher for spatial, lower for temporal) |
| Inspired by Marigold only | Inspired by Marigold (spatial) + Wan VAE inflation (staged training) |

---

## Progressive Temporal Extension (Optional Stage 2 Variant)

If Stage 2 with T=81 is too aggressive a jump from T=1, consider progressive temporal extension:

1. **Stage 2a:** Fine-tune on T=5 (k=1, latent T=2). Very short clips — model learns basic frame-to-frame consistency.
2. **Stage 2b:** Fine-tune on T=17 (k=4, latent T=5). Model learns short-range temporal dynamics.
3. **Stage 2c:** Fine-tune on T=81 (k=20, latent T=21). Full target length.

Each sub-stage initializes from the previous checkpoint. This mirrors Wan's own progressive training strategy of "progressively upscaled data resolutions and extended temporal durations."

---

## Future Work

### Audio Cross-Attention Conditioning
After the visual-to-deformation mapping is stable, add audio conditioning via a cross-attention adapter. The two-stage approach makes this cleaner — the spatial mapping is already locked in, so the audio only needs to modulate temporal dynamics.

### TI2V-5B with Native Image Conditioning
Use **Wan2.2-TI2V-5B** which natively supports image + text → video. Instead of Marigold's channel-doubling trick, use the model's built-in image conditioning. Requires understanding TI2V-5B's internal 48-channel input structure.

### Text-Only Generation (No Driving Video)
The current approach requires a natural video at inference. To achieve text-only generation, a future stage would train a model to generate deformation maps from a single reference image + text, or from text alone.

---

## File Structure

```
marigold_training/
├── scripts/
│   ├── train_stage1_image.py          # Stage 1: image-level Marigold training
│   ├── train_stage2_video.py          # Stage 2: video-level fine-tuning
│   ├── inference_marigold.py          # Inference (works for both stages)
│   └── validate_vae_t1.py            # Verify VAE works with T=1 input
├── src/
│   ├── dataset_image_pairs.py         # Stage 1 dataset: (frame, deform_frame, text) triplets
│   ├── dataset_video_pairs.py         # Stage 2 dataset: (video, deform_video, text) triplets
│   ├── model_marigold.py              # Input layer modification + loading utilities
│   └── inflate_weights.py             # Stage 1 → Stage 2 weight transfer (Option B)
└── configs/
    ├── train_stage1_config.yaml       # Stage 1 hyperparameters
    └── train_stage2_config.yaml       # Stage 2 hyperparameters
```

---

## Reference

Study the original Marigold implementation before coding:
- **Training:** `https://github.com/prs-eth/Marigold` — study `train.py` and the trainer class
- **Diffusers pipeline:** `diffusers/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py` — inference loop with concatenation
- **Community pipeline:** `diffusers/examples/community/marigold_depth_estimation.py` — simpler reference showing the core loop
- **Paper Section 3.2:** The adapted denoising U-Net, weight duplication formula for the input layer
- **v1.1 (arxiv 2505.09358):** Zero-SNR and trailing timestep improvements
- **Wan technical report (arxiv 2503.20314):** Section on progressive image-video joint training strategy and VAE inflation

The core pattern: at every denoising step, concatenate `[noisy_target_latent, clean_conditioning_latent]` along the channel dimension and pass to the denoiser. The conditioning latent provides spatiotemporal anchoring. The model predicts the update for the target latent only.

## Training Data Paths

```
data/
├── talkvid/
│   ├── talkvid/
│   │   ├── {clip_id}.mp4              # Source video (8313 clips)
│   │   └── ...
│   └── audio/
│       ├── {clip_id}.wav              # 16kHz mono audio (8313 clips)
│       └── ...
├── flowface/
│   ├── {clip_id}/                     # Per-clip FLAME tracking output (7150 clips)
│   │   ├── fit.npz                    # FLAME parameters
│   │   ├── images/cam0/              # Extracted video frames
│   │   └── bg/cam0/                  # Foreground masks
│   └── ...
└── derived/
    └── captions/
        └── {clip_id}.json            # Prosody captions (the "caption" field)
```

**Effective training set:** 7150 clips (intersection of talkvid and flowface).

**Deformation map generation pipeline:**
1. `compute_flame()` in `marionette/flame/flame.py` — takes `fit.npz` params, returns `offsets_3d: (V, 3)`
2. `PropRenderer.render()` in `marionette/conditioning/mesh2img.py` — rasterizes per-vertex offsets onto 2D grid
3. `THConditioning.forward()` in `marionette/conditioning/th_conditioning.py` — orchestrates the above, channels `[42:45]` of `pos_enc` are the 3ch deformation
