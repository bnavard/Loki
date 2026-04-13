# INSTRUCTIONS: Marigold-Style Training for Video → Expression Map Video

## Motivation

Our previous attempts to fine-tune pretrained video diffusion models to generate FLAME expression dense field maps have failed. The models retain strong bias toward natural video generation — even with full parameter fine-tuning, the output is either natural-looking video or noise, never the structured expression maps we need.

The **Marigold** paper (Ke et al., CVPR 2024 Oral, Best Paper Candidate; https://arxiv.org/abs/2312.02145) solved an analogous problem: repurposing Stable Diffusion 2 (a text-to-image model trained on natural images) to generate depth maps, surface normals, and other non-natural structured outputs from natural image inputs. We adapt their approach to our video domain.

**Reference implementation:** https://github.com/prs-eth/Marigold
**Diffusers pipeline:** `https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py`
**Community pipeline:** `https://huggingface.co/logologolab/cute_playful_logo_lora/blob/main/diffusers/examples/community/marigold_depth_estimation.py`

---

## Marigold's Core Technique (from the paper)

Marigold starts from **Stable Diffusion 2**, a pretrained **text-to-image** diffusion model, and adapts it:

1. **Both the input and target are images processed through the same frozen VAE:**
   - Input RGB image `x` → VAE encoder → input latent `z^(x)` (4 channels)
   - Target depth/normals `d` → VAE encoder → target latent `z^(d)` (4 channels)

2. **Concatenate along the channel dimension at every denoising step:**
   - `z_input = cat(z^(d)_t, z^(x))` — noisy target latent + clean input latent → 8 channels

3. **Modify only the input layer of the U-Net:**
   - The U-Net's **first conv layer** (input layer only) is expanded from 4 to 8 input channels
   - The pretrained weight tensor of this input layer is **duplicated** and **divided by 2**
   - This preserves activation magnitudes at initialization
   - **No other layers are modified architecturally**

4. **Fine-tune only the U-Net.** VAE is frozen. Text encoder is frozen. Marigold passes **empty text embeddings** to the U-Net — the input image latent is the sole conditioning signal.

5. **Noise is added only to the target latent.** The input image latent is always clean.

6. **Inference:** Start from pure noise for the target latent. At each DDIM step, re-concatenate the clean input latent with the current noisy target latent, predict noise, step the scheduler.

---

## Our Adaptation: Video Marigold with Flow Matching

### The Analogy

| Marigold | Our adaptation |
|---|---|
| Input: natural **image** (RGB photo) | Input: natural **video** (talking-head clip) |
| Target: **depth map** (non-natural image) | Target: **deformation map video** (non-natural video) |
| Same image, same scene | Same clip, same frames, temporally aligned |
| Both encoded through the same image VAE | Both encoded through the same video VAE |
| Model: SD2 (text-to-image, DDPM) | Model: Wan2.1-T2V-1.3B (text-to-video, flow matching) |

The key parallel: in Marigold, the natural RGB image and the depth map are of the **same scene** — the depth map is a structured re-representation of the same spatial content. In our case, the natural talking-head video and the deformation map video are of the **same clip** — the deformation map is a structured re-representation of the same facial dynamics. Frame N of the natural video corresponds exactly to frame N of the deformation map.

### Model

**Wan2.1-T2V-1.3B** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) — a 1.3B parameter text-to-video diffusion model using flow matching (velocity prediction).

### Flow Matching vs DDPM

Marigold's core *principle* transfers directly — concatenate clean conditioning with noisy target at the input layer. But the *mechanics* must follow Wan's flow matching formulation:

| Marigold (DDPM/DDIM) | Our adaptation (Flow Matching) |
|---|---|
| Noise schedule: `noisy = √ᾱ_t * data + √(1-ᾱ_t) * noise` | Linear interpolation: `noisy = (1-t) * data + t * noise` |
| Model predicts **noise** (ε) | Model predicts **velocity** (v = noise - data) |
| Loss: `MSE(pred_noise, actual_noise)` | Loss: `MSE(pred_velocity, actual_velocity)` |
| Inference: DDIM scheduler steps | Inference: Euler ODE integration (t=1 → t=0) |
| Text: **empty** embeddings (unused) | Text: **prosody captions** (actively used) |

### Task

- **Input conditioning:** Natural talking-head video of a person → VAE encode → clean video latent
- **Text conditioning:** Description of speech delivery/prosody
- **Target output:** Expression deformation map video of the **same clip** → VAE encode → target video latent (noised during training, denoised during inference)

---

## Architecture Modification

Following Marigold, we modify **only the input layer** of the DiT transformer.

The Wan2.1 DiT has a `patch_embedding` layer (Conv3d) that maps from `in_channels` (the VAE's `z_dim`, which is 16) to the hidden dimension. We double its input channels to accept the concatenated conditioning:

The code below is adapted directly from Marigold's `_replace_unet_conv_in()` method, applied to Wan's DiT `patch_embedding` instead of SD's U-Net `conv_in`:

```python
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# --- Marigold's input layer replacement (from their actual code) ---
# Original Marigold code for reference:
#   _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
#   _bias = self.model.unet.conv_in.bias.clone()       # [320]
#   _weight = _weight.repeat((1, 2, 1, 1))             # [320, 8, 3, 3]
#   _weight *= 0.5                                      # half activation magnitude
#   _new_conv_in = Conv2d(8, 320, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#   _new_conv_in.weight = Parameter(_weight)
#   _new_conv_in.bias = Parameter(_bias)
#   self.model.unet.conv_in = _new_conv_in

# Our adaptation for Wan2.1 DiT:
original_conv = transformer.patch_embedding  # Conv3d(16, hidden_dim, ...)
_weight = original_conv.weight.clone()       # [out_ch, 16, kT, kH, kW]
_bias = original_conv.bias.clone() if original_conv.bias is not None else None

# Repeat along input channel dim (dim=1) to double: 16 → 32
_weight = _weight.repeat((1, 2, 1, 1, 1))   # [out_ch, 32, kT, kH, kW]
_weight *= 0.5                                # half activation magnitude

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

**This is the ONLY architectural change.** All other layers remain identical to the pretrained model. The rest of the transformer is fine-tuned end-to-end (all parameters updated by optimizer), exactly as Marigold fine-tunes the entire U-Net.

**Verify the actual attribute name** — the input conv may be named differently in Wan2.1. Inspect:
```python
for name, mod in transformer.named_modules():
    if isinstance(mod, nn.Conv3d):
        print(name, mod)
```

---

## Training Data

Each training sample is a triplet from the **same clip**:
- **`natural_video`**: The original talking-head video from `data/talkvid/talkvid/{clip_id}.mp4` — frames `[3, T, H, W]` in [-1, 1]
- **`target_video`**: Expression deformation map video `[3, T, H, W]` in [-1, 1] — generated from the existing FLAME pipeline (`THConditioning.forward()`, channels `[42:45]` of `pos_enc`), temporally aligned frame-by-frame with the natural video
- **`text`**: Prosody caption from `data/derived/captions/{clip_id}.json` (the `"caption"` field)

**Critical:** The natural video and deformation map must be from the **same clip** and **temporally aligned**. Frame N of the natural video must correspond to frame N of the deformation map. This is naturally satisfied since both are derived from the same source video with FLAME tracking.

---

## Training Loop

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
scheduler = pipe.scheduler

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
    natural_video = batch["natural_video"]   # [B, 3, T, H, W] talking-head video, [-1, 1]
    target_video = batch["target_video"]     # [B, 3, T, H, W] deformation map, [-1, 1]
    text = batch["text"]                     # list of caption strings

    with torch.no_grad():
        # Encode text
        text_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        text_embeds = text_encoder(**text_inputs.to(device)).last_hidden_state

        # Encode natural video through VAE (this is the CONDITIONING — always clean)
        natural_5d = natural_video.to(device, dtype=torch.float32)  # VAE needs float32
        natural_latent = vae.encode(natural_5d).latent_dist.mode()
        # Shape: [B, C_latent, T_latent, h, w] e.g. [B, 16, T_lat, 64, 64]

        # Encode target deformation map video through VAE (this is the TARGET — will be noised)
        target_5d = target_video.to(device, dtype=torch.float32)
        target_latent = vae.encode(target_5d).latent_dist.mode()
        # Shape: [B, C_latent, T_latent, h, w] — same shape as natural_latent

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
    # [noisy_target | clean_natural_video] along channel dimension
    # The natural video latent is ALWAYS clean — never noised
    model_input = torch.cat([noisy_target, natural_latent], dim=1)
    # Shape: [B, 2*C_latent, T_latent, h, w] e.g. [B, 32, T_lat, 64, 64]

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

---

## Inference

At inference time, the user provides a natural talking-head video and a text description. The model generates the corresponding deformation map video.

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

**Note:** The natural video latent is re-concatenated at **every** denoising step — it is never modified, exactly as Marigold does with the input image latent.

**Classifier-Free Guidance at inference (optional):**
If trained with CFG dropout, apply guidance at inference:
```python
# Conditioned prediction (with text)
vel_cond = transformer(model_input, t, text_embeds).sample

# Unconditioned prediction (null text)
vel_uncond = transformer(model_input, t, null_embeds).sample

# Guided velocity
velocity = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
```

---

## What to Fine-Tune

**Full fine-tuning** (if LoRA underfits)
- Unfreeze all transformer parameters
- Lower learning rate: 5e-6
- Enable gradient checkpointing

---

## VAE Considerations

- Wan2.1 VAE: `z_dim=16`, `scale_factor_spatial=8`, `scale_factor_temporal=4`
- Frame count must satisfy `4k+1` rule (e.g., 1, 5, 9, 13, ..., 81)
- **Both the natural video and deformation map video must have the same frame count** — they're from the same clip
- **VAE must run in float32** for encoding/decoding quality
- Normalize latents with `vae.config.latents_mean` and `vae.config.latents_std`
- Both natural video latent and deformation map latent use the **same normalization stats** since they go through the same VAE

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | |
| Batch size per GPU | 1-2 | |
| Gradient accumulation | 4 | |
| Learning rate | 1e-5 (LoRA) or 5e-6 (full) | |
| Training steps | 10,000-25,000 | |
| CFG dropout | 0.1 | Drop text 10% |
| Mixed precision | bf16 | |
| Gradient clipping | 1.0 | |
| Target frames | 81 (4k+1 rule, ~3.2s at 25fps) | |
| Resolution | 512×512 | |

---

## Key Differences from Previous Attempts

| Previous approach | Marigold approach |
|---|---|
| Text-only conditioning | Natural video + text conditioning via input concatenation |
| Standard input layer | Input layer doubled with weight duplication / 2 |
| Model generates from scratch (noise → deformation) | Model generates conditioned on paired natural video at every step |
| No spatial/temporal anchoring | Full spatiotemporal anchoring from the natural video |
| Deformation map has no visual similarity to training data | Natural video conditioning is in-distribution for the pretrained model |

---

## Handling the Full 45-Channel Expression Field

The target is the 3-channel deformation field only. At inference time, to reconstruct the full 45-channel field:
1. Generate the 3-channel deformation video from natural video + text
2. Extract FLAME shape/pose parameters from the natural video via FLAME tracking
3. Recompute the 42-channel positional encoding per frame from shape/pose
4. Concatenate: `[pos_enc_42ch, generated_deform_3ch]` → full 45-channel field
5. Feed into the rendering UNet (`marionette/`)

---

## Future Work

### TI2V-5B with Native Image Conditioning
Use **Wan2.2-TI2V-5B** which natively supports image + text → video. Instead of Marigold's channel-doubling trick, use the model's built-in image conditioning. Requires understanding TI2V-5B's internal input structure.

### Text-Only Generation (No Driving Video)
The current Marigold approach requires a natural video at inference — it generates the deformation map paired with that video. To achieve text-only generation (no driving video needed), a future stage would train a separate model to generate the natural video from text, or generate deformation maps conditioned on a single reference image + text instead of a full video.

---

## File Structure

```
text_to_expr_field/
├── scripts/
│   ├── train_marigold.py              # Marigold-style training
│   └── inference_marigold.py          # Marigold-style inference
├── src/
│   ├── dataset_marigold.py            # Returns (natural_video, target_video, text) triplets
│   ├── model_marigold.py              # Input layer modification + loading utilities
│   └── ...
└── configs/
    └── train_marigold_config.yaml     # Hyperparameters
```

---

## Reference

Study the original Marigold implementation before coding:
- **Training:** `https://github.com/prs-eth/Marigold` — study `train.py` and the trainer class
- **Diffusers pipeline:** `diffusers/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py` — inference loop with concatenation
- **Community pipeline:** `diffusers/examples/community/marigold_depth_estimation.py` — simpler reference showing the core loop
- **Paper Section 3.2:** The adapted denoising U-Net, weight duplication formula for the input layer
- **v1.1 (arxiv 2505.09358):** Zero-SNR and trailing timestep improvements

The core pattern: at every denoising step, concatenate `[noisy_target_latent, clean_conditioning_latent]` along the channel dimension and pass to the denoiser. The conditioning latent provides spatiotemporal anchoring. The model predicts the update for the target latent only.
