# INSTRUCTIONS: Text-to-Expression Dense Field Video Generation Pipeline

## Overview

### Motivation

The existing talking-head generation system (implemented in `marionette/`) produces high-quality talking-head videos, but it requires two external conditioning signals at inference time: (1) a FLAME expression dense field extracted from a **driving video** via 3DMM tracking, and (2) audio from that same driving video. This means the user must supply an actual video of someone speaking with the desired facial expressions — a significant barrier to usability.

**The goal of this pipeline is to remove the dependency on a driving video entirely.** Instead of requiring a real person's video to extract expression dynamics from, we train a generative model that can synthesize the expression dense field directly from a text description. Combined with off-the-shelf text-to-speech (TTS) for audio generation, this enables a **text-only interface**: the user provides a single reference portrait image and a text prompt (e.g., "Say 'Hello, welcome to the presentation' in a warm, professional tone"), and the system generates the complete talking-head video with no driving video needed.

The approach is modular — we fine-tune a pretrained video diffusion model (Wan2.2-T2V-A14B) to generate expression dense field videos from text, while the downstream rendering UNet remains unchanged. The generated expression field is fed into the existing rendering pipeline in exactly the same format it currently expects from real FLAME tracking output.

### What This Document Covers

### What This Document Covers

This document describes how to build a pipeline that fine-tunes **Wan2.2-T2V-A14B** (via HuggingFace `diffusers`, model ID: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`) to generate **45-channel FLAME expression dense spatial field videos** from text descriptions.

The 45-channel expression dense field consists of:
- **42 channels:** sinusoidal Fourier positional encoding of 3D vertex positions, rasterized onto a 2D UV grid (3 coordinates × 14 values [7 frequency bands × sin/cos])
- **3 channels:** expression deformation (per-vertex Δx, Δy, Δz displacement from neutral face)

These are rasterized onto a 2D UV grid via PyTorch3D barycentric interpolation.

**Goal:** Given a text prompt describing what a person says and how they say it, generate a T=16 frame expression dense field video at 480×480 resolution. This generated field will later be consumed by the talking-head rendering UNet implemented in `marionette/` as a spatial conditioning signal (the `pos_enc` tensor in `THConditioning.forward()`).

**This document covers Phase 1: text-only conditioning.** Audio cross-attention conditioning will be added in a future phase.

**Project folder:** All code for this pipeline lives in a dedicated folder named `text_to_expr_field/` at the project root.

---

## Architecture Summary

```
Text prompt
    │
    ▼
┌──────────────────┐
│  UMT5 Encoder    │  (frozen, Wan2.2's native text encoder)
└─────┬────────────┘
      │ text embeddings
      ▼
┌────────────────────────────────────┐
│  Wan2.2 DiT MoE (A14B)            │  ← LoRA fine-tuned on expression field videos
│  - transformer   (high-noise exp.) │
│  - transformer_2 (low-noise exp.)  │
│  (Flow Matching)                   │
└─────┬──────────────────────────────┘
      │ denoised latent [61, 16, 60, 60]
      ▼
┌──────────────────┐
│  Wan2.2 VAE      │  (frozen, 3ch in / 3ch out, z_dim=16)
│  Decoder         │  decode 15 sub-videos independently
└─────┬────────────┘
      │ 15 × [T=16, 3, 480, 480]
      ▼
  Reassemble → [T=16, 45, 480, 480]
  Expression dense field video
```

**What is frozen:** VAE (encoder + decoder), UMT5 text encoder.
**What is trained:** LoRA adapters on **both** DiT experts (transformer and transformer_2).

---

## Key Design: Channel Reshaping for VAE Compatibility

The Wan2.2 VAE accepts 3-channel RGB video input and outputs 3-channel video. Our expression dense field has 45 channels. We handle this by reshaping:

### Encoding (training data preparation)

1. Start with expression dense field: `[T=16, 45, 480, 480]`
2. Separate into 15 groups of 3 channels each:
   - Groups 0–13: positional encoding (42 channels → 14 × [T, 3, 480, 480])
   - Group 14: deformation field ([T, 3, 480, 480])
3. Stack temporally: `[T×15, 3, 480, 480]` = `[240, 3, 480, 480]`
4. **Pad 1 frame** to satisfy Wan2.2 VAE's `4k+1` frame requirement: `[241, 3, 480, 480]`
   - The padding frame can be a repeat of the last frame.
5. Pass through VAE encoder as a single video: → latent `[61, 16, 60, 60]`
   - Temporal: 241 frames → (241-1)/4 + 1 = 61 latent temporal steps
   - Spatial: 480/8 = 60 latent spatial steps
   - Channels: 16 (Wan2.2 VAE z_dim)

### Decoding (inference)

1. DiT generates latent: `[61, 16, 60, 60]`
2. VAE decodes: → `[241, 3, 480, 480]`
3. Drop the padding frame: → `[240, 3, 480, 480]`
4. Reshape back: `[240, 3, 480, 480]` → `[16, 15, 3, 480, 480]` → `[16, 45, 480, 480]`
   - Channel order must be preserved: groups 0–13 are positional encoding, group 14 is deformation.

### VAE Reconstruction Quality

This reshaping has been validated — the pretrained Wan2.2 VAE can encode and decode each of the 15 three-channel groups with acceptable reconstruction quality. A few high-frequency positional encoding bands show minor loss, which is tolerable for downstream rendering.

---

## Wan2.2-T2V-A14B: Key Model Details

- **Architecture:** Mixture-of-Experts (MoE) Diffusion Transformer
  - `transformer` (high-noise expert, ~14B params): handles early denoising steps (high noise / low SNR)
  - `transformer_2` (low-noise expert, ~14B params): handles late denoising steps (low noise / high SNR)
  - Total: ~27B params, but only ~14B active per step
  - Transition between experts is determined by SNR threshold
- **Text encoder:** UMT5-XXL (frozen)
- **VAE:** AutoencoderKLWan — `z_dim=16`, `in_channels=3`, `out_channels=3`, `scale_factor_temporal=4`, `scale_factor_spatial=8`
- **Noise schedule:** Flow Matching (velocity prediction)
- **Frame count rule:** Must satisfy `4k + 1` (e.g., 1, 5, 9, 13, 17, 21, ..., 241, ...)
- **Diffusers model ID:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

---

## Dataset

### Source Layout

```
data/
├── talkvid/
│   ├── talkvid/
│   │   └── {clip_id}.mp4              # 8313 source videos, 25fps, ~5s each
│   └── audio/
│       └── {clip_id}.wav              # 16kHz mono audio, 8313 clips
│
└── flowface/
    └── {clip_id}/                     # 7150 clips with FLAME tracking
        ├── fit.npz                    # FLAME params (shape, expr, rot, tra, eye_rot)
        ├── reference_images.json
        ├── images/cam0/*.jpg          # Extracted video frames
        └── bg/cam0/*.png              # Foreground masks
```

**Effective training set: 7150 clips** (intersection of talkvid and flowface). The pipeline must filter to this intersection at startup.

### Derived Data (produced by preprocessing)

```
data/
└── derived/
    ├── expr_field_videos/
    │   └── {clip_id}.pt               # Full expression dense field tensor [T_total, 45, 480, 480]
    ├── captions/
    │   └── {clip_id}.json             # {"transcription": "...", "prosody": "...", "caption": "..."}
    └── manifest.json                  # List of valid clip_ids with paths to all derived data
```

---

## Pipeline Steps

Three stages: (1) Preprocessing, (2) Training, (3) Inference.

---

## Stage 1: Preprocessing

### Step 1.1: Generate Expression Dense Field Videos

**Input:** `data/flowface/{clip_id}/fit.npz` for each clip.
**Output:** `data/derived/expr_field_videos/{clip_id}.pt` — a float tensor `[T_total, 45, 480, 480]` where `T_total` is the total number of frames in the clip (~125 frames for 5s at 25fps).

**How to generate the 45-channel expression dense field:**

The existing codebase has the full pipeline. The relevant modules are:

1. `marionette/flame/flame.py` → `compute_flame()` — takes `fit.npz` parameters, returns `offsets_3d: (V, 3)` (per-vertex displacement from neutral FLAME mesh).
2. `marionette/conditioning/mesh2img.py` → `PropRenderer.render()` — rasterizes per-vertex offsets onto a 2D grid via PyTorch3D barycentric interpolation.
3. `marionette/conditioning/th_conditioning.py` → `THConditioning.forward()` — orchestrates the above, producing the full 45-channel `pos_enc` tensor. Channels `[0:42]` are the sinusoidal positional encoding, channels `[42:45]` are the deformation field.

**Implementation:**

Write `text_to_expr_field/scripts/preprocess_expr_fields.py`:

1. Iterate over all `data/flowface/{clip_id}/fit.npz` files.
2. For each clip, load `fit.npz` and extract FLAME parameters for all frames.
3. For each frame, call the existing pipeline to produce the full `pos_enc` tensor at 480×480: shape `[45, 480, 480]`.
4. Stack all frames: `[T_total, 45, 480, 480]`.
5. Save as a `.pt` file (float32 or float16 to preserve precision).

**Normalization note:** The existing codebase's `THConditioning.forward()` pipeline handles value normalization internally. Verify that the output tensor values are in a range compatible with the Wan2.2 VAE's expected input distribution (VAE expects pixel values normalized to [-1, 1]). If the existing pipeline already outputs values in an appropriate range, no additional normalization is needed. If not, apply a linear rescaling and document the min/max values used. The current codebase has not had issues with this, so it is likely already handled — but confirm before proceeding.

**Parallelization:** Embarrassingly parallel. Use multiprocessing array jobs across the 8 GPUs (PyTorch3D rasterization is GPU-accelerated).

**Storage estimate:** 7150 clips × ~125 frames × 45 channels × 480 × 480 × 2 bytes (float16) ≈ **1.7 TB**. If storage is a constraint, consider computing fields on-the-fly during training, or storing only the 16-frame windows needed per epoch.

### Step 1.2: Generate Text Captions

**Input:** `data/talkvid/audio/{clip_id}.wav` for each clip.
**Output:** `data/derived/captions/{clip_id}.json`:
```json
{
  "clip_id": "abc123",
  "transcription": "I can't believe you did that",
  "prosody": "The speaker sounds surprised and amused, with rising intonation on 'believe' and a slight laugh at the end. Pace is moderate.",
  "caption": "A person says: 'I can't believe you did that.' The speaker sounds surprised and amused, with rising intonation on 'believe' and a slight laugh at the end. Pace is moderate."
}
```

#### Step 1.2a: ASR Transcription

Use `openai/whisper-large-v3` via HuggingFace `transformers`:

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_id = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
```

For each `{clip_id}.wav`: load 16kHz mono audio, run Whisper, store transcription string.

**Throughput:** ~5-10 clips/second for 5-second audio on an H200.

#### Step 1.2b: Prosody Description via Audio Language Model

Use `Qwen/Qwen2-Audio-7B-Instruct` via HuggingFace `transformers`.

Prompt for each clip:

```
Listen to this audio clip carefully. Describe in 2-3 concise sentences:
1. The emotional tone of the speaker (e.g., happy, neutral, angry, sad, excited, sarcastic)
2. Speaking pace (slow, moderate, fast) and rhythm (steady, varied, halting)
3. Notable prosodic features: emphasis on specific words, pitch changes, pauses, laughter, sighs, or other non-verbal sounds

Be specific and concise. Do not transcribe the words — only describe HOW they are spoken.
```

**Note:** Qwen2-Audio accepts audio inputs directly. Refer to the HuggingFace model card for the exact API. Audio should be 16kHz.

**Important context:** The dataset has diverse speakers. Prosody descriptions are critical because different speakers express the same emotion differently, and the expression dense fields are identity-agnostic (FLAME geometry, not appearance). The text labels must capture delivery style variation.

#### Step 1.2c: Combine into Structured Caption

```python
caption = f"A person says: '{transcription}' {prosody_description}"
```

Save as `data/derived/captions/{clip_id}.json`.

**Quality check:** Manually inspect 20-30 random samples to verify transcription accuracy and prosody description variety.

### Step 1.3: Build Training Manifest

Write `text_to_expr_field/scripts/build_manifest.py`:

1. Scan `data/flowface/` for all clip_ids with `fit.npz`.
2. Verify each clip_id exists in both `data/talkvid/talkvid/{clip_id}.mp4` and `data/talkvid/audio/{clip_id}.wav`.
3. Verify derived data exists: `data/derived/expr_field_videos/{clip_id}.pt` and `data/derived/captions/{clip_id}.json`.
4. Output `data/derived/manifest.json`:

```json
[
  {
    "clip_id": "abc123",
    "expr_field_path": "data/derived/expr_field_videos/abc123.pt",
    "caption_file": "data/derived/captions/abc123.json",
    "audio_file": "data/talkvid/audio/abc123.wav",
    "num_frames": 125
  }
]
```

5. Print dataset statistics: total clips, average duration, clips with missing data.

### Step 1.4 (Optional): Precompute VAE Latents

This step is **optional but recommended** if disk space permits. It avoids running the VAE encoder during training.

If implemented: for each clip, load the expression field tensor, sample a 16-frame window, reshape to [241, 3, 480, 480] (with padding), encode through frozen VAE, and cache the latent `[61, 16, 60, 60]` to disk.

If not precomputed, the dataset class must perform VAE encoding on-the-fly during training. This adds latency but saves disk space.

---

## Stage 2: Training

### Step 2.1: Dataset Class

Create `text_to_expr_field/src/dataset.py`:

```python
class ExprFieldDataset(Dataset):
    def __init__(self, manifest_path, captions_dir, num_frames=16):
        ...
```

**`__getitem__` logic:**

1. Load the full expression field tensor from `{clip_id}.pt`: `[T_total, 45, 480, 480]`.
2. Randomly sample a contiguous 16-frame window: `[16, 45, 480, 480]`.
3. **Reshape for VAE:**
   - Separate into 15 groups: `[16, 15, 3, 480, 480]`
   - Flatten temporal and group dims: `[240, 3, 480, 480]`
   - Pad 1 frame (repeat last): `[241, 3, 480, 480]`
4. **Encode through frozen VAE** (if not precomputed):
   - Rearrange to VAE input format: `[1, 3, 241, 480, 480]` (batch=1, channels=3, frames=241, H=480, W=480)
   - VAE encode → latent `[1, 16, 61, 60, 60]`
   - Remove batch dim: `[16, 61, 60, 60]`
5. Load caption from `{clip_id}.json`, use the `"caption"` field.
6. Return `{"latent": latent, "caption": caption_text}`.

**If precomputed latents are used:** skip steps 3-4 above, load latent directly from cache.

### Step 2.2: Training Script

Create `text_to_expr_field/scripts/train.py`.

**Model loading:**

```python
from diffusers import WanPipeline
import torch

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
)

# Wan2.2-A14B has TWO transformer experts
transformer_high_noise = pipe.transformer      # high-noise expert
transformer_low_noise = pipe.transformer_2     # low-noise expert
vae = pipe.vae                                 # Frozen
text_encoder = pipe.text_encoder               # Frozen (UMT5)
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler                     # Flow matching
```

**LoRA configuration — BOTH experts need LoRA:**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=[
        # VERIFY THESE by inspecting the model:
        # for name, mod in transformer_high_noise.named_modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         print(name, mod.in_features, mod.out_features)
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
    ],
    lora_dropout=0.05,
)

transformer_high_noise = get_peft_model(transformer_high_noise, lora_config)
transformer_low_noise = get_peft_model(transformer_low_noise, lora_config)

transformer_high_noise.print_trainable_parameters()
transformer_low_noise.print_trainable_parameters()
```

**CRITICAL:** Inspect the actual module names before applying LoRA. The placeholder names above (`to_q`, `to_k`, etc.) must be verified against the Wan2.2 transformer architecture.

**Freezing:**
```python
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
# Both transformers handled by LoRA — only LoRA params are trainable
```

**Training loop (simplified pseudocode):**

```python
optimizer = torch.optim.AdamW(
    list(transformer_high_noise.parameters()) + list(transformer_low_noise.parameters()),
    lr=1e-5, weight_decay=0.01
)

for batch in dataloader:
    latents = batch["latent"]           # [B, 16, 61, 60, 60]
    captions = batch["caption"]

    # Encode text
    text_inputs = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    text_embeds = text_encoder(**text_inputs).last_hidden_state

    # Sample noise and timesteps (flow matching)
    noise = torch.randn_like(latents)
    timesteps = torch.rand(B, device=device)  # Uniform [0, 1]

    # Select expert based on SNR / timestep
    # High noise (early denoising) → transformer_high_noise
    # Low noise (late denoising) → transformer_low_noise
    # The boundary_ratio from the pipeline config determines the split
    # IMPORTANT: Refer to diffusers WanPipeline source for exact expert selection logic

    # Interpolate (flow matching)
    noisy_latents = (1 - timesteps) * latents + timesteps * noise

    # Predict velocity with the appropriate expert
    velocity_pred = selected_transformer(noisy_latents, timesteps, encoder_hidden_states=text_embeds)

    # Loss
    target_velocity = noise - latents
    loss = F.mse_loss(velocity_pred, target_velocity)

    # CFG: randomly drop text conditioning 10% of the time
    # Replace text_embeds with null embeddings during dropout

    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**CRITICAL NOTES:**
- The pseudocode above is simplified. The actual implementation **must** use the Wan2.2-specific flow matching formulation and expert selection logic. Refer to the `diffusers` source code for `WanPipeline` to understand: (a) how timesteps map to SNR, (b) where the expert boundary is, (c) exact velocity parameterization.
- Both experts are trained simultaneously — each training step randomly samples a timestep, selects the appropriate expert, and backpropagates through that expert's LoRA parameters only.
- Wan2.2 uses **flow matching** (not DDPM). The noise schedule and velocity prediction must match the pretrained formulation exactly.

**Distributed training:**

```bash
# 8×H200 with accelerate
accelerate launch --num_processes 8 --mixed_precision bf16 text_to_expr_field/scripts/train.py
```

**Hyperparameters:**

| Parameter | Value | Notes |
|---|---|---|
| Batch size per GPU | 1 | MoE 14B + 61-frame latents are memory-heavy |
| Gradient accumulation | 4 | Effective batch size = 32 |
| Learning rate | 1e-5 | Standard for LoRA on large DiT |
| LR schedule | Cosine with warmup | 500 warmup steps |
| LoRA rank | 128 | Reduce to 64 if overfitting |
| LoRA alpha | 128 | Scaling factor 1.0 |
| Training steps | 15,000–25,000 | Monitor validation loss |
| CFG dropout prob | 0.1 | Drop text 10% of the time |
| Mixed precision | bf16 | Native on H200 |
| Gradient clipping | 1.0 | |
| Optimizer | AdamW | weight_decay=0.01 |

**Logging and checkpointing:**
- Log to wandb or tensorboard: loss, LR, gradient norms.
- Save LoRA checkpoints (both experts) every 2,000 steps.
- Every 2,000 steps, run inference on 5 fixed validation prompts, decode generated latents through VAE, reassemble to 45-channel expression fields, and save as videos for visual inspection.

### Step 2.3: Validation

**Validation set:** Hold out ~150 clips (~2%).

**Quantitative metrics:**
- Validation loss (MSE on velocity prediction).
- Decode generated latents → reassemble to [16, 45, 480, 480] → compare against GT:
  - Per-frame MSE on deformation channels ([42:45]).
  - Per-frame MSE on positional encoding channels ([0:42]).
  - Temporal smoothness: L1 between consecutive frames.

**Qualitative:**
- Visualize generated expression fields side-by-side with GT.
- Check lip-region deformation dynamics correspond to speech in the caption.
- Check temporal coherence (no flickering).

---

## Stage 3: Inference

### Step 3.1: Inference Script

Create `text_to_expr_field/scripts/inference.py`:

```python
from diffusers import WanPipeline
from peft import PeftModel
import torch

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
)

# Load LoRA for BOTH experts
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    "checkpoints/text_to_expr_field/best_lora_high_noise",
)
pipe.transformer_2 = PeftModel.from_pretrained(
    pipe.transformer_2,
    "checkpoints/text_to_expr_field/best_lora_low_noise",
)

prompt = "A person says: 'Hello, welcome to the presentation.' The speaker sounds warm and professional, with moderate pace and clear enunciation."

# Generate — the pipeline handles expert switching internally
# num_frames must satisfy 4k+1 rule: we need 241 frames
output = pipe(
    prompt=prompt,
    num_frames=241,
    height=480,
    width=480,
    guidance_scale=7.5,
    num_inference_steps=50,
)

# output.frames: [241, 3, 480, 480] (or similar — check diffusers output format)
generated_video = output.frames  # shape: [241, 3, 480, 480]
```

### Step 3.2: Reassemble to 45-Channel Expression Field

```python
# Drop padding frame
generated_video = generated_video[:240]  # [240, 3, 480, 480]

# Reshape back to [16, 45, 480, 480]
generated_video = generated_video.reshape(16, 15, 3, 480, 480)  # [T, groups, 3, H, W]
expr_field = generated_video.reshape(16, 45, 480, 480)           # [T, 45, H, W]

# Channels 0:42 = positional encoding, 42:45 = deformation
```

### Step 3.3: Feeding into the Rendering UNet

The reassembled `[16, 45, 480, 480]` expression dense field is in the exact format expected by the rendering UNet in `marionette/` — specifically, it is the `pos_enc` tensor consumed by `THConditioning.forward()` in `marionette/conditioning/th_conditioning.py`. It can be directly used as the spatial conditioning signal, concatenated with the reference mask channel (channel 45, added separately) to form the full 46-channel input.

---

## Switching from LoRA to Full Fine-Tuning

If LoRA results are insufficient:

1. Remove LoRA. Load both transformers without `get_peft_model`.
2. Unfreeze all parameters: `transformer.requires_grad_(True)`, `transformer_2.requires_grad_(True)`.
3. Reduce LR to `5e-6`.
4. Enable gradient checkpointing:
```python
transformer.enable_gradient_checkpointing()
transformer_2.enable_gradient_checkpointing()
```
5. Use DeepSpeed ZeRO Stage 2 or FSDP for memory efficiency across 8×H200.
6. All other settings remain the same.

---

## Future Phase 2: Adding Audio Cross-Attention

Placeholder for the next phase:

1. Add frozen `wav2vec2-base-960h` encoder.
2. Learned linear projection: wav2vec2 768-dim → DiT hidden dim.
3. Insert cross-attention layers in each DiT block (both experts): video tokens query wav2vec2 features. Zero-initialize output projections.
4. Train only new audio cross-attention layers (keep existing LoRA frozen or continue joint training).
5. During training: GT audio from `data/talkvid/audio/{clip_id}.wav`.
6. During inference: TTS-generated audio from text prompt.
7. CFG: independently drop text (10%), audio (10%), both (5%) for dual-guidance.

---

## File Structure

```
project_root/
├── text_to_expr_field/
│   ├── scripts/
│   │   ├── preprocess_expr_fields.py     # Step 1.1
│   │   ├── generate_captions.py          # Step 1.2
│   │   ├── build_manifest.py             # Step 1.3
│   │   ├── train.py                      # Step 2.2
│   │   └── inference.py                  # Step 3.1
│   ├── src/
│   │   ├── dataset.py                    # Step 2.1
│   │   └── utils.py                      # Reshaping, reassembly utilities
│   └── configs/
│       └── train_config.yaml             # All hyperparameters
├── data/
│   ├── talkvid/                          # Source (existing)
│   ├── flowface/                         # FLAME tracking (existing)
│   └── derived/                          # Preprocessing outputs
│       ├── expr_field_videos/
│       ├── captions/
│       └── manifest.json
├── checkpoints/
│   └── text_to_expr_field/
│       ├── step_{N}_high_noise/          # LoRA checkpoint (high-noise expert)
│       └── step_{N}_low_noise/           # LoRA checkpoint (low-noise expert)
└── marionette/    # Existing codebase: talking-head rendering UNet
    ├── flame/flame.py                    #   compute_flame() — FLAME param → vertex offsets
    ├── conditioning/mesh2img.py          #   PropRenderer.render() — rasterize offsets to 2D
    └── conditioning/th_conditioning.py   #   THConditioning.forward() — produces pos_enc [45ch]
```

---

## Dependencies

```
torch>=2.4.0
diffusers>=0.34.0       # Must support WanPipeline with Wan2.2-A14B MoE (transformer + transformer_2)
transformers>=4.40       # Whisper, Qwen2-Audio, UMT5
peft>=0.10               # LoRA
accelerate>=0.30         # Distributed training
flash_attn               # Required by Wan2.2
pytorch3d                # Deformation map rasterization (existing dependency)
wandb                    # Logging (or tensorboard)
soundfile                # Audio loading
```

**Verify diffusers compatibility before starting:**
```python
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", torch_dtype=torch.bfloat16)
print(type(pipe.transformer))    # Should be WanTransformer3DModel
print(type(pipe.transformer_2))  # Should be WanTransformer3DModel (low-noise expert)
print(pipe.vae.config.z_dim)     # Should be 16
print(pipe.vae.config.scale_factor_temporal)  # Should be 4
print(pipe.vae.config.scale_factor_spatial)   # Should be 8
```

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| VAE lossy on some PE frequency bands | Already validated — acceptable. Monitor reconstruction of high-frequency bands. |
| 241-frame latent (61 temporal steps) is large | Memory-intensive. Use gradient checkpointing + FSDP. Batch size 1 per GPU. |
| MoE expert selection during training | Must correctly route timesteps to the right expert. Study diffusers WanPipeline source carefully. |
| 7150 clips for 27B-param model | LoRA reduces effective trainable params. If underfitting, increase LoRA rank to 256. |
| Text captions too generic across diverse speakers | Inspect Qwen2-Audio prosody descriptions. If too uniform, try stronger prompting or a different audio LM. |
| Generated fields temporally jittery | Add temporal smoothness penalty during training, or post-process with temporal Gaussian blur. |
| diffusers version doesn't support Wan2.2-A14B | Wan2.2 diffusers integration was released Jul 28, 2025. Ensure diffusers is up to date. |
