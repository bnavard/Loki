# Marionette — Paper Context

This document provides the prior context needed to collaboratively draft the
Marionette paper. It describes what the system does, why it exists, and how it
works — enough for Claude to reason about abstract/intro/methodology/experiment
sections without needing to re-read the codebase each time.

---

## One-sentence summary

MY COMMENT: We also wanna say that we replace the FLAME conditionig by using Marigold model to predict the expression map generatively rather than the original FLAME parameterization technique. Our technique to generate the deformation map reduces the time needed by the FLAME fitting model (that takes nearly 40 minutes for a single sample) to a few minutes ( I will provide you with the exact number but consider something less that 5min for a 5 second clip with 120 something frame count at 25 fps.). We need to frame this in a way that does not seem we are just plug and play the Marigold approach for the expression map generation. We need to sell this in a way that will look compelling


Marionette demonstrates that video diffusion models for talking-head generation benefit from explicit spatial priors over facial dynamics, achieved by conditioning on FLAME expression deformation maps rather than unstructured motion signals, enabling identity-preserving synthesis from audio or driving video with minimal per-identity tuning.

---

## Motivation

### The problem

Generating realistic talking-head video from a single portrait photo and an
audio/expression driving signal remains challenging. Existing methods either:

- Operate in pixel space with GAN-based warping (e.g. FOMM, face-vid2vid),
  producing artifacts at large poses and lacking temporal coherence over long
  sequences; or
- Use end-to-end video diffusion models that struggle to disentangle identity
  from motion, requiring expensive per-identity fine-tuning (e.g. DreamTalk,
  AniPortrait).

A key gap is the **conditioning interface**: how to tell the diffusion model
*what the face should be doing* at each frame without leaking identity
information or requiring paired driving video at inference time.

### Our insight

The FLAME 3D morphable model already provides a compact, identity-agnostic
decomposition of facial dynamics: per-vertex expression deformation offsets
that encode *how much* each face region moves relative to a neutral shape.
When rasterized onto a 2D grid via differentiable rendering (PyTorch3D), this
produces a dense **expression deformation map** — a spatial heatmap that
highlights active facial regions (mouth opening, brow raising, eye blinking)
without encoding any appearance.

We condition a video diffusion model on this map. The model learns that bright
regions in the deformation map correspond to facial dynamics that need to be
synthesized with high fidelity, while dark regions (forehead, background) can
be hallucinated from the reference frame. This is a much more informative
conditioning signal than raw driving video (which mixes appearance with motion)
or sparse keypoints (which lack spatial granularity).

### The Marigold connection

Obtaining ground-truth deformation maps requires FLAME tracking, which needs
a driving video. To enable generation from audio alone (no driving video), we
train a separate **Marigold-style per-frame predictor** that generates the
expression deformation map from a natural face image. This adapts the Marigold
depth-estimation approach (Ke et al., CVPR 2024) to face deformation: given
a natural face frame, produce its corresponding deformation map. The predictor
is trained on (face frame, deformation map) pairs using channel concatenation
and rectified flow.

This creates a two-module pipeline:
1. **Expression map generator** (Marigold-style) — predicts deformation maps
   from face frames (or eventually from audio features).
2. **Video diffusion renderer** (Marionette) — generates the talking-head video
   conditioned on the expression map + audio + reference identity.

The ablation studies isolate the contribution of each module and each
conditioning signal.

---

## System architecture

### Module A: Expression map generator (`marigold_training/`)

- **Base model:** SD3.5 Medium (24-layer MMDiT, ~1.5B parameters).
- **Task:** Image-to-image translation (face frame → deformation map).
- **Training:** Marigold-style channel doubling — the input conv is expanded
  from 16 to 32 channels. At each denoising step, the clean face-frame latent
  and the noisy deformation-map latent are concatenated along the channel
  dimension. Null-text conditioning (unconditional generation).
- **Flow matching:** Rectified flow with velocity prediction.
- **Multi-resolution noise:** Pyramid of noise at progressively lower
  resolutions, helping the model learn smooth spatial coherence of deformation
  gradients.
- **Data:** ~900k (face frame, deformation map) pairs from 7150 clips × ~127
  frames per clip. The deformation maps are the 3-channel expression
  deformation field from FLAME rasterization (channels 42:45 of the full 45ch
  expression field, pre ref_mask).
- **Resolution:** 512 × 512. Inference is per-frame; temporal coherence
  emerges from the coherence of the input video.

### Module B: Video diffusion renderer (`marionette/`)

- **Base model:** Stable Diffusion 2.1 UNet, extended with 3D spatiotemporal
  attention (2D self-attention replaced with 3D; convolutions remain per-frame).
- **Latent space:** SD 2.1 VAE, 4 channels, 8× spatial downsampling.
  Resolution 512 → 64 × 64 latents.
- **Video window:** T=16 frames. Frame 0 is the reference (identity source);
  frames 1–15 are generated. All 16 frames participate in 3D attention.
- **Reference passthrough:** For reference slots (ref_mask=1), the UNet
  bypasses its learned prediction and outputs the known noise residual
  `x - z_input`, so the reference passes through unchanged while still
  participating in attention for identity conditioning.

#### Conditioning signals

| Signal | Mechanism | Dimension | Description |
|---|---|---|---|
| **Expression map** | Spatial addition to first UNet feature map via learned linear | 46ch (full) or 4ch (deform-only) or 1ch (ref-mask-only) | FLAME-rasterized vertex positions + deformation offsets + reference mask |
| **Audio** | Cross-attention in every transformer block | 1024-dim context | wav2vec2-base backbone (frozen) → learned 768→1024 linear projection. Per-frame tokens from a ±2 frame audio window. |
| **Reference frame** | VAE-encoded latent injected at frame 0 via z_input + ref_mask | 4ch latent | Ground-truth appearance; zeroed for generated frames |

The 46-channel expression map decomposes as:
- Channels 0–41: Sinusoidal positional encoding of rasterized 3D vertex
  positions (14 frequency bands × 3 axes = 42ch). Encodes WHERE the face
  surface is.
- Channels 42–44: Per-vertex expression deformation offsets (Δx, Δy, Δz),
  normalized by the training-set standard deviation (0.0104). Encodes HOW
  MUCH the face moves.
- Channel 45: Reference mask (1 for reference frame, 0 for generation frames).

#### Expression-weighted diffusion loss

The standard MSE denoising loss is weighted per-pixel by expression
deformation magnitude:

```
weight = 1.0 + α × normalize(||deformation||)
loss = weighted_mean(MSE(ε_pred, ε) × weight)
```

- `α = 0`: uniform loss (all pixels equal). This is the default.
- `α = 5.0`: expression-weighted (active face regions amplified, static
  regions at baseline 1.0 — never suppressed).
- The full expression map is always computed internally for loss weighting,
  even when the UNet conditioning is ablated to fewer channels.

---

## Ablation studies

The experiments are designed to answer four independent questions:

### 1. Expression source ablation (`experiments/ablate_expr_source/`)

**Question:** What spatial conditioning signal gives the denoiser the most
useful information about facial dynamics?

| Variant | Channels | What the UNet sees |
|---|---|---|
| `gt_full` | 46 | Full FLAME: geometry (positional encoding) + motion (deformation) |
| `gt_baseline` | 4 | FLAME deformation only (3ch heatmap + 1ch ref_mask) |
| `marigold` | 4 | Marigold-generated deformation (learned, not rasterized) |
| `driving_video` | 4 | Raw driving video downsampled to 64×64 (3ch RGB + 1ch ref_mask) |

All 4ch variants use the same UNet architecture (`condition_channels=4`) so
the comparison is fair on channel budget. `gt_full` is the upper-bound
reference.

**Key insight tested:** The FLAME decomposition separates *what moves* from
*what it looks like*. The deformation map highlights active regions (mouth,
brows) and suppresses static ones (forehead, background). Raw video at 64×64
carries the same implicit information but doesn't decompose it — the UNet must
learn to extract motion salience from blurry appearance. We hypothesise the
structured signal is strictly better.

### 2. Conditioning channel ablation (`experiments/ablate_conditioning/`)

**Question:** Do the 42 positional-encoding channels (WHERE the face is)
actually help, or is the 3ch deformation (HOW it moves) sufficient?

Compares full 46ch vs deform-only 4ch vs ref-mask-only 1ch. Tests whether
vertex-position information provides geometric anchoring that improves spatial
accuracy of the generated faces.

### 3. Audio ablation (`experiments/ablate_audio/`)

**Question:** Does wav2vec2 cross-attention improve lip sync and expressiveness
beyond what the spatial expression map already provides?

Flips the audio encoder on/off. The expression map already encodes mouth
dynamics; audio may be redundant for motion but critical for fine-grained lip
shape and timing.

### 4. Loss weighting ablation (`experiments/ablate_loss_weighting/`)

**Question:** Does amplifying the denoising loss on high-deformation regions
actually help, or does uniform loss converge to the same quality?

Compares `α=0` (uniform) vs `α=5.0` (weighted). The hypothesis is that
weighted loss forces the denoiser to allocate capacity to the regions that
matter most for perceptual quality (mouth, eyes) at the expense of
less-important static regions.

---

## Dataset

- **Source:** TalkVid — ~8300 talking-head clips from YouTube, each ~5 seconds
  at 25 FPS, 512 × 512 resolution.
- **FLAME tracking:** Pixel3DMM → FlowFace format (`fit.npz`) for ~7150 clips.
  Each fit contains per-frame FLAME parameters: shape (150), expression (65),
  rotation (3), translation (3), eye rotation (3), neck rotation (3), plus
  camera intrinsics/extrinsics.
- **Audio:** 16 kHz mono WAV, extracted from source clips.
- **Captions:** Whisper large-v3 ASR + Qwen2-Audio prosody descriptions
  (used for the Marigold text conditioning path, currently null-text).
- **Train/val split:** Identity-based 90/10 split to prevent identity leakage.
  All clips from the same YouTube video go to the same split.
- **Effective training set:** ~7150 clips (~900k frames) for Module A;
  ~7150 clips (16-frame non-overlapping windows → ~55k training samples) for
  Module B.

---

## Key design decisions worth discussing in the paper

1. **Deformation maps as conditioning, not generation target.** We don't
   generate the deformation map and then render it (two-stage pixel pipeline).
   Instead, the deformation map is a *conditioning signal* that guides the
   diffusion model — the model generates natural video directly, with the
   expression map steering its attention.

2. **Marigold for expression prediction, not for video generation.** The
   Marigold approach is used only for the per-frame expression predictor
   (Module A), not for the video renderer (Module B). Module B uses a standard
   latent video diffusion architecture with 3D attention.

3. **Reference passthrough, not reference encoding.** The identity reference
   isn't encoded into a CLIP embedding or a face recognition vector — it's
   passed through as a literal VAE latent at frame 0. The 3D attention allows
   generated frames to attend to it directly, preserving fine-grained identity
   features without an information bottleneck.

4. **Expression-weighted loss as an alternative to architecture changes.** Rather
   than designing a face-specific attention mask or region-of-interest head, we
   modulate the standard diffusion loss to achieve per-region prioritisation.
   This is simpler and composes cleanly with any conditioning variant.

5. **Structured conditioning beats unstructured.** The driving-video ablation
   explicitly tests this: the same information (facial motion) delivered as a
   structured deformation heatmap vs raw low-resolution pixels. This is the
   central claim of the paper.

---

## Suggested paper structure

### Abstract (~150 words)
Problem → Insight (FLAME deformation as conditioning) → Method (two-module
pipeline) → Key result (structured beats unstructured, Marigold-generated
comparable to GT).

### 1. Introduction
- Talking-head generation landscape: GANs vs diffusion, identity vs motion.
- The conditioning bottleneck: how to tell a diffusion model what the face
  should do.
- Our contribution: FLAME expression maps as structured spatial conditioning
  for video diffusion, with a Marigold-trained predictor to generate them.

### 2. Related work
- Talking-head generation (FOMM, face-vid2vid, DreamTalk, AniPortrait, etc.)
- Latent video diffusion models (LDM, Make-A-Video, Wan, etc.)
- Monocular depth/normal estimation via diffusion (Marigold, GeoWizard, etc.)
- 3DMM-based face representation (FLAME, DECA, etc.)

### 3. Method
- 3.1 Expression map representation (FLAME → rasterised deformation field)
- 3.2 Video diffusion with structured conditioning (UNet modifications,
  spatial injection, reference passthrough, audio cross-attention)
- 3.3 Expression-weighted loss
- 3.4 Marigold-style expression predictor (channel doubling, per-frame
  training, inference on coherent video)

### 4. Experiments
- 4.1 Setup (dataset, metrics, baselines)
- 4.2 Expression source ablation (gt_full vs gt_baseline vs marigold vs
  driving_video)
- 4.3 Conditioning channel ablation (46ch vs 4ch vs 1ch)
- 4.4 Audio ablation
- 4.5 Loss weighting ablation
- 4.6 Qualitative results

### 5. Conclusion

---

## Terminology glossary

| Term | Meaning |
|---|---|
| Expression map | The full 45ch spatial field: 42ch positional encoding + 3ch deformation (before appending ref_mask) |
| Deformation map | The 3ch subset (channels 42–44): per-pixel expression deformation magnitude and direction |
| Expression field | Same as expression map (used interchangeably in the codebase) |
| Driving video | The source video providing facial dynamics — either as FLAME parameters or raw frames |
| Reference frame | Frame 0 of the generation window; provides identity via VAE latent passthrough |
| Deform-only | 4ch conditioning variant: 3ch deformation + 1ch ref_mask (drops 42ch positional encoding) |
| gt / GT | Ground-truth expression map, rasterized from FLAME tracking on the fly |
| Marigold source | Expression map generated by the Marigold-trained Module A predictor |
| Channel budget | Number of spatial conditioning channels the UNet receives (1, 4, or 46) |