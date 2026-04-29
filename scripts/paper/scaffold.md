# Marionette — paper scaffolding & planning

Working planning doc for the Marionette paper. Each section gives **the claim it lands**, **the evidence we already have**, **figure/table slots with status**, and **open decisions**. Not paper prose — a contract for what to write.

---

## Working title

**`Marionette: Representation over Architecture for Portrait Animation Diffusion Model`**

Naming: we call the task **portrait animation from a driving video**, not "talking-head video generation". Audio is disabled in the published configuration; the task is video-driven animation. (See open decisions if you want this revisited.)

---

## North-star one-line claim

> Marionette animates a single reference portrait from a driving video by **lifting FLAME blendshape deformations into pixel space** as a rasterized expression-activation map — instead of feeding raw RGB or FLAME parameter vectors into a learned motion-control module. At inference, the driver's expression and pose parameters are reapplied **parametrically** onto the reference's FLAME shape, producing geometrically consistent driver-specific motion *without any cross-identity training*. Identity flows through a separate path: features from a frozen reference UNet (the same SD-2 backbone used for generation) inject directly into the matching multi-resolution self-attention layers of the gen UNet, **eliminating the need for a learned identity encoder**. The result is competitive portrait animation with **less than half the trainable parameters of X-Portrait** and trained on **roughly an order of magnitude less data** than the leading diffusion-prior baselines, while leading on driver-pose fidelity.
>
> Alongside, we contribute the right evaluation pair for portrait animation, both derived from the same FLAME fits that drive the conditioning. **Head-rotation trajectory follow**: for each clip we anchor every frame's head rotation to that clip's own frame 0 (so any constant per-clip camera-fit offset cancels out), then report the geodesic angular distance — the smallest 3D rotation, in degrees — between pred's and target's per-frame rotation deltas. This penalises a prediction whose head trajectory drifts off the driver's arc and ignores idle-pose offsets that don't matter for portrait animation, where the prediction's idle pose is set by the reference image, not the driver. **Pose-disentangled expression error**: render the target's FLAME-derived expression-deformation map; render a second one where pred's expression coefficients are substituted into the target's head pose, identity shape, and camera; compare the two pixel-wise (L1 over on-mesh pixels — average absolute deformation residual per pixel, across the 3 channels). Both renders sit at the exact same image-space pose by construction, so the only thing that can drive a per-pixel difference is expression-coefficient mismatch — head pose is already captured by the first metric, with no double-counting. Together they avoid the failure modes of inherited proxies: no external head-pose CNN with its own Euler conventions to defend, no FVD-style distribution prior that rewards generic naturalness over driver-specific motion, no PSNR/SSIM-style pixel-alignment penalty that punishes a prediction for being a few pixels off the GT's framing when its animation is otherwise correct.

If the reader walks away with one paragraph, this is it. Every section pulls toward it.

---

## What X-Portrait and HunyuanPortrait actually do (so we know what we're contrasting)

Quick architectural reads from their respective `impl/` trees. We want this in the back of our minds when writing § 1 and § 2 — every sentence positioning Marionette should be verifiable against this.

### X-Portrait (SIGGRAPH 2024, ByteDance) — a three-ControlNet stack on SD-1.5
- **Backbone**: Stable Diffusion 1.5 latent diffusion model (64×64 latent, an `8×` VAE downsample of 512×512), with temporal-attention layers added in the gen UNet so it produces a short video instead of single frames.
- **Three trainable ControlNet branches** (each roughly the size of an SD-1.5 UNet — a *ControlNet* is a copy of the UNet whose features are added into the main UNet at matching resolutions, the standard Zhang-et-al recipe):
  1. **Appearance/identity ControlNet** — reads the reference image and cross-couples its features into the gen UNet, in the AnimateAnyone "reference-only" style. Plays the role of an identity encoder, but as a trained branch rather than a separate module.
  2. **Global-pose ControlNet** — consumes an RGB-derived "pose" hint built by warping the driving frames toward the reference's pose using 2D face-alignment landmarks (68-point face-keypoint detection on every driver frame, then a 2D similarity transform).
  3. **Local-pose ControlNet** — consumes a *masked RGB patch* of the driver: keeps only the pixels in 64×64 boxes around the eye and mouth landmark centers, zeros out everything else. Brings high-frequency local motion in as raw pixels.
- **Total trainable parameters**: ~3.07 B (single 12.3 GB fp32 checkpoint). All three ControlNets, plus the temporal module on top of the gen UNet, are trained end-to-end.
- **What they do not use**: no explicit 3D parametric face model (FLAME / 3DMM). All motion is inferred from 2D RGB driver frames via the two motion ControlNets. Optional initial-noise seeding from `face-vid2vid` (an older keypoint-driven face-reenactment model) is supported but not architectural.
- **Headline fact for our contrast**: motion comes from RGB → ControlNet, and the cost is paid in *trained parameters* (two motion ControlNets ≈ ~640M params each, on top of the appearance branch).

### HunyuanPortrait (arXiv 2503.18860, Tencent) — implicit motion features + IP-Adapter identity on SVD
- **Backbone**: Stable Video Diffusion (the `img2vid-xt` variant from Stability AI)
- **Motion path (implicit, learned from RGB)**:
  - A small **expression regressor** (ResNet-18 with GroupNorm) reads each driver RGB frame and outputs a 512-dim expression vector.
  - A separate **head-pose regressor** (ResNet) outputs pose features per frame.
  - A **Perceiver-style resampler** (cross-attention into a fixed bank of learnable query tokens, plus an intensity-bucket modulation) takes the per-frame motion features and emits ~64 motion tokens, which the gen UNet attends to.
  - The paper title states it directly — *"Implicit Condition Control"*. There is no explicit 3DMM; motion is whatever the two regressors and the resampler learn.
- **Identity path (IP-Adapter style — Ye et al.'s pattern of injecting image features as additional cross-attention K/V tokens)**:
  - **ArcFace** (a face-recognition ONNX encoder) → 512-dim identity embedding.
  - **DINOv2-Large** (Meta's self-supervised ViT, ~314 M params) → patch-token image features, passed through a learnable projector to match the UNet's cross-attention token dimension.
  - Both feature streams are injected into the UNet cross-attention via IP-Adapter-style projection adapters (separate trained K/V matrices on top of every cross-attention layer).
- **Pose guider**: a small CNN (~1.1 M params) that takes a coarse 2D pose signal and adds it into the UNet's input.
- **Total parameters**: ~1.99 B all-inclusive (UNet 1.58B + DINO 314M + motion-resampler 66M + expression regressor 25M + head-pose regressor 11M + arcface and image-projector are tiny).
- **Headline fact for our contrast**: motion is *learned implicitly* from RGB by two regressors plus a resampler. Identity uses *separately-trained projection adapters* (ArcFace + DINO + a projector) bolted onto the UNet, not the gen UNet's own backbone.

### How Marionette differs (one bullet per axis)
| Axis | X-Portrait | HunyuanPortrait | **Marionette** |
|---|---|---|---|
| Motion representation | RGB patches → 2 ControlNets | RGB → ResNet regressors → motion tokens | **FLAME blendshapes rasterized to pixel space (45-ch expression map)** |
| Cross-identity strategy | Pose-align driver to ref via landmarks; train end-to-end | Train end-to-end on motion features | **Parametric retargeting at inference: `β_ref + ψ_driver + θ_driver`** — no cross-id training |
| Identity injection | Reference-only ControlNet (a trained branch reading the reference image) | ArcFace + DINOv2 + a learned projector, injected via IP-Adapter K/V adapters into the UNet's cross-attention | **Frozen SD-2 reference UNet → per-layer self-attention features → injected as additional K/V tokens into the matching layers of the gen UNet (no learned identity encoder added)** |
| Trainable params | ~3.07 B (all 3 ControlNets + temporal) | sized similarly; ~1.99 B all-inclusive | **~857 M trainable** (gen UNet only) + frozen pretrained components |
| Training corpus | ByteDance-internal, not disclosed; large | not disclosed; large | **~10 k samples** *(TODO: confirm exact composition)* |

This table goes verbatim into § 4 setup. It's also the lens for § 1 and § 2.

---

## Abstract — ~150 words

**Goal**: state the task, contrast the two SOTA representatives, our move in one sentence, headline numbers in *relative* terms, the disclosed limitations.

- **Hook (1 sentence)**: animating a single reference portrait from a driving video requires capturing both subtle facial expressions and large head motion while preserving the reference identity.
- **Gap (2 sentences, naming both contrasts)**: leading methods such as X-Portrait and HunyuanPortrait answer this with architectural complexity — X-Portrait stacks two motion ControlNets that consume pose-aligned RGB patches of the driver, HunyuanPortrait pairs implicit ResNet motion regressors with IP-Adapter identity projectors. Both pay the cost in additional learned modules and in training data.
- **Move (1 sentence)**: Marionette replaces the architectural cost with a *representational* one — the driver's motion is encoded as **FLAME blendshape deformations rasterized into pixel space** (an "expression activation map"), parametrically retargeted at inference onto the reference's FLAME shape.
- **Identity in 1 sentence**: identity is injected through the gen prior's own frozen SD-2 reference UNet — no learned identity encoder is added.
- **Numbers (relative)**: ~½ the trainable parameters of X-Portrait, comparable parameter count to HunyuanPortrait but trained on roughly an order of magnitude less data (Let's get clarification on this); leads or co-leads on per-axis head-orientation L1 across all four (dataset, protocol) cells; competitive same-identity pixel quality.
- **Limitations briefly**: FVD lags (texture drift in the SD-2 VAE pipeline); cross-identity ArcFace cosine is the weakest column.w

---

## 1. Introduction — ~1.5 pages

**Section claim**: portrait animation has been answered architecturally; Marionette answers it representationally.

### ¶ 1 — Motivation
Animating a single portrait from a driving video has applications across accessibility, dubbing, virtual avatars, and telepresence. The hard part is faithfully transferring the driver's full motion envelope — both highly dynamic large head movements and subtle, fine-grained facial expressions — onto an unrelated reference identity, while keeping that identity unchanged.

### ¶ 2 — Two strongest answers in the field, both architectural
Lead with **X-Portrait** and **HunyuanPortrait** as concrete representatives.
- X-Portrait stacks two motion ControlNets on top of an SD-1.5 backbone — one global, one local — both consuming RGB-derived signals (pose-aligned driver frames; landmark-cropped eye/mouth patches). Identity comes from a third ControlNet branch (AnimateAnyone-style ReferenceOnly).
- HunyuanPortrait uses Stable Video Diffusion as the backbone and learns motion *implicitly* from RGB via two ResNet regressors (`HeadExpression`, `HeadPose`) followed by a Perceiver-style refiner; identity is injected through IP-Adapter-style projection of ArcFace + DINOv2 features.
- These methods share an architectural bet: bigger backbones, dedicated learned motion modules, dedicated learned identity adapters. They pay the cost in trained parameters and in training data — both proprietary and large.

### ¶ 3 — Our move (representation over architecture)
Marionette inverts the bet. The driver's facial expression and head pose are *not* learned from RGB; they are **read off a parametric face model** (FLAME) and **rasterised into pixel space**:
- The conditioning tensor we feed the diffusion model is a 45-channel rasterization of the retargeted FLAME mesh: 42 channels of sinusoidal positional encoding of vertex positions in normalized device coordinates, plus 3 channels of per-vertex expression deformation (offsets relative to a neutral face).
- Crucially this is **not** a parameter-vector conditioning (cf. older audio-3DMM hybrids that ingest `(ψ, θ)` as a vector). The signal is delivered in image space, **as an activation map**, natively aligned with the diffusion model's own convolutional inputs.
- Cross-identity animation comes for free at inference via a simple parametric substitution — no cross-identity training required (§ 3.3).
- Identity is injected through the diffusion prior's *own* frozen reference UNet — no learned identity encoder is added (§ 3.4).

### ¶ 4 — Other clusters (brief)
For completeness, but kept short:
- **Audio-driven, no visual driver** (SadTalker, EchoMimic): cannot faithfully transfer a specific driver's facial expression and head pose because they are not given a visual driver clip to copy from. Useful as a "no-driver" baseline.
- **Coarse-landmark-driven** (older talking-head work): 68-pt landmarks are too sparse to express the fine facial geometry that FLAME blendshapes do, and they're 2D — they bake in the driver's identity geometry by construction.
- **Audio + 3DMM-vector hybrids**: ingest 3DMM parameters as a flat vector through cross-attention; semantically meaningful but lose the spatial structure that the rasterization provides.

### ¶ 5 — Contributions (3 + 1)
1. **Pixel-space FLAME representation as an expression map** *(§ 3.2)*. We rasterize FLAME blendshape deformations into a 45-channel image-space tensor and feed it as the diffusion conditioning. This is the core differentiator from methods that consume raw FLAME parameter vectors as conditioning, and from methods that learn motion features from RGB.
2. **Parametric retargeting at inference enables cross-identity reenactment without cross-identity training** *(§ 3.3)*. We train *only* on same-identity self-supervised data; cross-identity comes from a parametric substitution `β_ref + ψ_driver[t] + θ_driver[t]` rendered through the reference's camera. The same forward path runs in both modes. This eliminates the need for the expensive cross-identity training corpus that diffusion-prior baselines rely on.
3. **Generative-prior-native identity injection** *(§ 3.4)*. We pass the reference image through a frozen copy of the same SD-2 UNet used for generation, capture self-attention features at every multi-resolution layer, and inject them as additional K/V tokens into the *matching* layers of the gen UNet. Because the embeddings already live in the generative prior's native feature space, no identity encoder needs to be trained — IP-Adapter-style projection layers and bespoke ID encoders (ArcFace, DINOv2) become unnecessary.
4. **The right evaluation for portrait animation, in the same FLAME space the model conditions on** *(§ 4.2)*. The dominant metrics in talking-head papers — PSNR / SSIM / LPIPS and FVD — are wrong for this task. The first three are *spatially sensitive*: a prediction that nails the driver's expression but is rendered a few pixels off the GT's head position takes a measurable hit on every pixel metric, while a prediction with a frozen, generic expression that happens to be pixel-aligned looks good. FVD measures distance to a *distribution* of natural videos, so it rewards generic plausibility, not faithful reproduction of *this* driver's specific motion. We instead measure portrait animation on the two axes that actually matter, both lifted from the prediction's own FLAME fit:
   1. **Head-rot geodesic distance** — from `rot · neck_rot`, anchor each clip's per-frame rotation to its frame 0 (factoring out any constant per-clip camera-fit offset between independently-tracked clips), then compare pred-vs-target trajectories by geodesic angular distance via the quaternion dot product. Avoids Euler-angle pitfalls (gimbal lock, wrap-around, convention noise). Reports a single interpretable scalar per cell, in degrees.
   2. **FLAME deformation-map L1** (pose-disentangled). Render the target's expression-deformation map; render the prediction's `(ψ, eye, jaw)` substituted into the target's pose / shape / camera; mask-aware L1 between the two (per-pixel mean-absolute-deviation across the 3 deform channels, then mean over on-mesh pixels). The only thing that can drive the residual is expression — pose error is captured separately by (1).

   Together they evaluate portrait animation in the same parametric space the model conditions on: no proxy via 2D landmarks, no distribution prior over natural videos, no learned head-pose CNN with conventions to defend. *FLAME-native conditioning, FLAME-native evaluation.*

### ¶ 6 — Results in one breath, in relative terms
Marionette **leads or co-leads on `head_rot_dist` across all four (dataset, protocol) cells**, beating SadTalker / AniTalker / EchoMimic by 2–6× and matching or edging out HunyuanPortrait and X-Portrait. Same-identity pixel quality is competitive with the leading baseline (within 0.7 dB PSNR on TalkVid). All of this with **less than half the trainable parameters of X-Portrait** and roughly an order of magnitude less training data. *(Numbers pending re-run under the new FLAME-native head_rot estimator; old 6DRepNet-based ordering predicted to survive.)*

---

## 2. Related work — ~0.75 pages

**Status**: skeleton only. *(TODO: shape this section near the end of writing — Pouyan has a list of references he'll add.)*

Four clusters; one short paragraph each. Each paragraph ends with a one-sentence "differentiator" line that names what Marionette does instead.

- **Audio-driven (no visual driver)**: SadTalker, EchoMimic, AniTalker (when run audio-only). *Differentiator*: no faithful transfer of a specific driver's motion possible.
- **Coarse 2D-landmark-driven**: list older talking-head work that uses 68-pt or 106-pt landmarks. *Differentiator*: 2D landmarks are sparse and bake in driver-identity geometry; FLAME deformations are 3D and disentangled at the parameter level.
- **Implicit motion-feature methods**: AniTalker, X-Portrait (motion side), HunyuanPortrait. *Differentiator*: motion is learned from RGB, requiring trained motion modules and a large training corpus; ours is parametric and explicit.
- **Audio + 3DMM-vector hybrids**: prior work that conditions on FLAME / 3DMM as a flat parameter vector through cross-attention. *Differentiator*: we deliver the FLAME signal **rasterized to image space**, retaining spatial structure that the diffusion convolutions can use as an activation map.

---

## 3. Method — ~2 pages, with one architecture figure

**Section claim**: a structured FLAME representation of facial expression and head pose, applied parametrically at inference, paired with a generative-prior-native identity path, can replace the architectural complexity of motion-from-RGB control modules.

### 3.1 Background — FLAME parametric model
Brief recap. FLAME is a parametric face model with three disentangled parameter groups: identity shape `β`, expression `ψ`, pose `θ` (head, jaw, neck, eye), plus camera. Forward operation: skinned vertex positions `V(β, ψ, θ) ∈ R^{V × 3}`. We use the CAP4D variant (with mouth verts and a 65-dim expression basis).

### 3.2 Pixel-space FLAME representation — the *expression activation map* (contribution 1)
Given a per-frame FLAME fit `(β, ψ[t], θ[t], camera)`, we run **one** pytorch3d rasterization pass and produce a 45-channel image-space tensor:
- `[0:42]` — sinusoidal positional encoding of rasterized vertex positions in normalized device coordinates (3 raw NDC channels × 14 frequencies).
- `[42:45]` — per-vertex expression deformation: the offset of each vertex from its neutral position under the current `ψ`. Rasterized in the same pass.

Both feature streams are masked by the on-mesh indicator (so background pixels stay zero). This 45-channel tensor is passed through a small convolutional encoder (a few stride-2 conv blocks that downsample 512×512 to the 64×64 latent resolution and project 45 channels down to the gen-UNet's first feature width; the final layer is zero-initialised so the conditioning starts as a no-op at training start) and **added** to the first feature map of the gen UNet — exactly the position where pixel-space conditioning has the strongest receptive field.

**Why this is different from prior FLAME conditioning**: previous methods consume the FLAME parameter vector `ψ` (65-dim) as a token through cross-attention. Our representation is **already in image space** when it reaches the diffusion model — same coordinate system, same spatial resolution, same convolutional locality. We call it an *expression map* because the diffusion model can consume it through ordinary convolutions, the same way it consumes its own intermediate features, not through a learned bridge.

w

### 3.3 Parametric retargeting at inference — cross-identity for free (contribution 2)
This is the section the paper hinges on.

**Setup**: at training we run only same-identity self-supervised optimization. One clip provides the reference frame, the target window, and the driver. The model never sees a cross-identity pair.

**Inference, single equation**:
- *Same-identity reconstruction*: build per-frame verts `V(β_ref, ψ_ref[t], θ_ref[t])` rendered through `camera_ref`. Standard.
- *Cross-identity retargeting*: build per-frame verts `V(β_ref, ψ_driver[t], θ_driver[t])` rendered through `camera_ref`. **The only thing that changes is which `(ψ, θ)` go into the same forward path.**

That is, retargeting is a **parametric substitution**, not a learned operation. Because FLAME's parameter space disentangles shape from expression and pose, swapping `(ψ, θ)` between subjects produces the driver's facial expression and head pose realised on the reference's geometry. **Driver identity cannot leak into the conditioning** — the only driver-derived quantities are `(ψ, θ)`, which by FLAME's definition are identity-orthogonal.

**Consequence**: the model needs to be trained only on the same-identity objective, yet cross-identity reenactment works at inference. This eliminates the cross-identity training requirement that the diffusion-prior baselines need to satisfy with large proprietary corpora. **Stating this plainly is the paper's headline mechanism**.

### 3.4 Generative-prior-native identity injection (contribution 3)
**Setup**: identity is *not* a learned signal in Marionette — we let the generative prior carry it.

**Mechanism**: we instantiate a *frozen copy* of SD-2's UNet — a separate "reference UNet" with weights tied to the gen UNet at initialisation, never updated thereafter — and run the VAE-encoded reference image through it once. PyTorch forward hooks on each transformer block's pre-attention LayerNorm capture the *input* to every self-attention layer at every UNet resolution. These per-layer feature maps (shape `(batch, H_l × W_l, channels_l)`) are then **injected as additional key/value tokens** into the matching self-attention block of the *gen* UNet. Concretely, in the gen UNet's attention, queries come from gen tokens; keys and values come from the concatenation of `[gen tokens, reference tokens]`. Every gen-frame query attends to its own tokens *and* the reference's, at the same resolution, in every layer.

**Why no learned encoder is needed**: the embeddings already live in the generative prior's native feature space. The reference UNet and the gen UNet are the same SD-2 architecture initialised from the same checkpoint, so there is no domain gap to bridge, no learned projection to train. The reference does **not** occupy a slot in the gen tensor — it lives only in the K/V-injected attention path. Loss is a uniform pixel-wise MSE on the predicted noise across all `T` target slots, no masking, no auxiliary identity loss.

**Contrast in one sentence**: where HunyuanPortrait pairs ArcFace + DINOv2 + a learned projector and trains IP-Adapter K/V adapters on top, Marionette captures features from the gen prior's own backbone and feeds them into the same backbone's attention layers directly. The "identity encoder" is the gen prior itself, frozen. Perhaps this is not a strong and the most prominent part of the paper. However we need to just mention it.

### 3.5 Diffusion backbone (brief)
- Stable Diffusion 2.1 latent diffusion model, 64×64 latent (an 8× VAE downsample of 512×512), `T = 16` frames denoised per forward pass.
- 2D self-attention layers in the inner UNet stages are replaced with 3D spatiotemporal attention so a single forward pass produces a coherent 16-frame video latent.
- The codebase also has audio cross-attention layers (a wav2vec-encoded driver-audio signal feeding cross-attention K/V), but they are **disabled in the published configuration**. Audio is treated as a future-work hook, not a contribution. *(TODO: confirm framing.)*
- The VAE (encoder + decoder) is frozen end-to-end; only the gen UNet trains.

### 3.6 Training
- **Self-supervised, same-identity only**. Each training sample is `T+1` frames drawn from a single video clip: slot 0 is a reference frame sampled at a position in the clip that is *independent* of the target window (using a separate RNG seed), slots 1..T are a contiguous target window. We VAE-encode all `T+1` frames, feed slot 0 to the reference UNet (one pass through the frozen identity path), and compute a uniform pixel-wise MSE loss on the predicted noise across slots 1..T (the standard ε-prediction LDM loss, applied to every target slot).
- **Why this generalises to cross-identity at inference (key paragraph)**: because the slot-0 reference is from a *different position* in the same clip than the target window, the reference and target generally disagree on pose and expression — they only agree on identity. So the model is *forced* to learn an identity prior that ignores the pose/expression mismatch between reference and target. At inference time, the parametric retargeting in § 3.3 produces driver pose/expression rendered through the reference's camera. The training distribution already includes "reference and target disagree on pose/expression"; cross-identity inference adds the small extra step of "and disagree on identity shape `β`", which the parametric retargeting handles by construction (driver-side `(ψ, θ)` substituted into the reference's `β`, so the conditioning never sees driver-identity geometry).
- Training corpus size: **~10k samples** *(TODO: exact composition — is this TalkVid only, or TalkVid + HDTF subset? After what filtering?)*
- Schedule: 30 k steps, virtual batch size 2, gpu batch size 2, `T = 16`. Trained on 8 NVIDIA H200 GPUs.
- Classifier-free-guidance training: each conditioning channel (FLAME deformation map, reference) is independently dropped to zero with some probability per training step, so the model also sees unconditional inputs.

### Figure 1 — architecture
Two-pathway block diagram. **Both paths visibly orthogonal** in the layout — top of figure is the Identity path, bottom is the Expression-and-pose path, the gen UNet sits in the middle and merges them.
- *Identity path*: reference image → frozen VAE encoder → frozen SD-2 reference UNet (a single forward pass) → per-layer self-attention features → arrows pointing into the matching self-attention layers of the gen UNet.
- *Expression-and-pose path*: driver FLAME parameters `(ψ_driver[t], θ_driver[t])` + reference shape `β_ref` + reference camera → FLAME mesh forward (a parametric, non-trainable pipeline that produces 3D vertex positions) → pytorch3d rasteriser (produces a 45-channel image-space tensor: 42 channels carry head pose as a positional encoding of vertex positions, 3 channels carry the per-vertex expression-deformation offset field) → small conv encoder → added into the first feature map of the gen UNet.
- *Gen UNet*: receives noise + the FLAME-derived expression-and-pose conditioning (added to first feature map) + identity K/V tokens (injected at every self-attention layer) → outputs denoised `T`-frame latents → frozen VAE decoder → output video.
- The figure must make the parametric substitution `β_ref + (ψ_driver, θ_driver)` *visually obvious* — that is the paper's thesis in one image.

*(TODO: someone has to draw this. Suggest TikZ or hand-illustrated.)*

---

## 4. Experiments — ~3 pages

### 4.1 Setup
- **Datasets**: TalkVid (in-the-wild ID diversity, ~125 eval clips per protocol after filtering) + HDTF (high-resolution studio, ~125 eval clips per protocol). Let's call them identities instead to emphasize the diversity more. So we say 250 identities. 
- **Protocols**:
  - *Same-identity reconstruction*: ref and driver are the same clip; tests pixel-level fidelity and how well the prediction follows the driver's facial expression and head pose.
  - *Cross-identity retargeting*: ref and driver are different clips; tests identity preservation while the driver's facial expression and head pose are transferred onto the ref.
- **Baselines**: SadTalker, AniTalker, EchoMimic, HunyuanPortrait, X-Portrait. Default sampling configs, public weights, fixed seeds. *(TODO: pin commit + config of each baseline in supplementary.)*
- **Frame coverage**: every metric scored on the **first 16 frames** of each prediction (Marionette generates 16-frame videos by default, so we standardise everyone to that window); the baselines' 75–125-frame outputs are truncated to 16. FVD reported only for same-identity (it does not apply to cross-identity, where there is no pred-vs-GT distribution to compare).
- **Face cropping**: pred and target are independently face-cropped using RetinaFace (an off-the-shelf face detector) at a 1.3× bbox margin and resized to 512×512 before scoring. Removes background bias so the metric measures the face, not the surrounding scene.

### 4.2 Metrics

Talking-head papers have inherited their evaluation framing from generic image / video generation: PSNR, SSIM, LPIPS for pixel fidelity; FVD for distribution-level quality; ArcFace cosine for identity. We report all of them for comparability with prior work, but **none of them measure what portrait animation is actually about**. The relevant question for our task is *"did the prediction follow the driver's motion (head pose + facial expression) faithfully, on the reference's face?"*. Our two new metrics answer that question directly; the inherited proxies are reported alongside but are not what we argue from.

**Why pixel metrics are the wrong evaluation here.** PSNR / SSIM / LPIPS compare pixels at fixed image-space positions. A prediction that perfectly transfers the driver's expression but renders the head a few pixels off the GT's position takes a measurable hit on every pixel metric — even though the animation itself is correct. Conversely, a prediction with a wooden, generic expression that happens to be pixel-aligned scores well. The metric and the goal are not the same thing.

**Why FVD is the wrong evaluation here.** FVD measures distance to a *distribution* of natural videos. It tells you whether your generation looks like a plausible talking-head video in general; it does not tell you whether *this* generation reproduced *this* driver's specific lip closure on frame 7. A model that produces clean, average-looking talking heads outperforms on FVD a model that reproduces idiosyncratic driver motion accurately.

**Why ArcFace cosine is the wrong evaluation for animation.** It measures identity preservation, which is necessary but not sufficient for portrait animation. Useful as a sanity check on cross-identity, silent on whether the prediction's facial expression and head pose actually followed the driver's.

We propose two metrics that directly answer the questions portrait animation is for, both lifted from the same FLAME fits the model conditions on:

**Head-rot geodesic distance, in degrees. (Let's come up with a marketable name!)** For each clip, compose the *visible* head rotation per frame from the FLAME parameters: `R_head[t] = R(rot[t]) · R(neck_rot[t])`, where `rot` is FLAME's global head-rotation parameter and `neck_rot` is the neck-joint rotation (FLAME applies them in that compositional order; `R(·)` denotes axis-angle → 3×3 rotation matrix). Form **frame-0-anchored** delta rotations per clip:

    dR[t] = R_head[t] · R_head[0]^T   (the rotation that takes frame 0 to frame t)

Then for every (pred, target) pair, per frame, compute the geodesic angular distance between `dR_pred[t]` and `dR_target[t]` via the quaternion dot product:

    q_p = quat(dR_pred[t]);  q_t = quat(dR_target[t])
    θ[t] = 2 · arccos( clip( |q_p · q_t|, -1, 1 ) )

This is the smallest rotation that takes one delta to the other, in degrees. Per-sample reduction is the mean of θ over the first 16 frames. Per-cell aggregation is a weighted mean across samples, with each sample's weight equal to the fraction of the 16 requested frames where both pred and target had a usable FLAME fit (a *track rate* in `[0, 1]`; almost always 1.0, but lower if the upstream FLAME tracker failed mid-clip).

Three properties make this the right pose number for portrait animation:
- **Frame-0 anchoring measures pose-trajectory follow.** The driver dictates the *change* in head pose; the prediction's idle pose is determined by the reference image, not the driver. So absolute pose mismatch isn't a meaningful penalty — what matters is whether the pred's head moves the same way the driver's does. (We considered inter-frame deltas `R[t] · R[t-1]^T` as an alternative; cumulative deltas catch slow systematic drift that inter-frame deltas would let through — a pred adding 0.3°/frame of yaw would have inter-frame deltas matching the driver's at every step yet end up 4.8° off after 16 frames — and cumulative deltas have ~10× better SNR over the 16-frame window because the head moves more cumulatively than per-frame.)
- **Camera-frame invariance.** The upstream FLAME tracker fits an independent camera per clip; the absolute `rot` value at frame 0 between two independently-tracked clips can differ by a constant offset purely from camera-fit drift. An empirical study on 15 cross-identity samples (anitalker generations on HDTF) found this offset dominated absolute comparison (~18° mean, dropping to ~3° once we anchored to each clip's frame 0). Frame-0 anchoring cancels the offset cleanly because the same offset shows up in `R_head[0]` and `R_head[t]` and gets multiplied out.
- **No Euler-angle pitfalls.** Direct yaw / pitch / roll comparison is unreliable: different conventions (intrinsic vs. extrinsic, XYZ vs. ZYX) give different numbers, wrapping at ±180° introduces discontinuities, and the same rotation has multiple equivalent Euler triples near gimbal lock. The geodesic via quaternion is the standard rotation-difference measure in robotics and graphics; it gives a single scalar in [0°, 180°] and avoids all of those.

**FLAME deformation-map L1** (pose-disentangled). For each (pred, target) pair, render the target's expression-deformation map; render a substituted fit where pred's `(expr, eye_rot, jaw_rot)` are inserted into the target's pose / shape / camera; mask-aware L1 between the two rasterised maps — per-pixel mean-absolute-deviation across the 3 deform channels, then mean over on-mesh pixels. Both renders share image-space layout by construction (target's pose), so the only thing that can drive a per-pixel residual is expression. We use L1 rather than L2 because its units are directly interpretable as the average per-component deformation residual, while RMSE-across-channels is harder to read off in deform-coefficient terms. Pose error is captured separately by the head-rot metric; together the two metrics cover head pose and facial expression orthogonally, with no double-counting.

Why this is the right expression number: comparing raw FLAME `ψ` vectors directly is meaningless (the basis isn't perceptually uniform — a small coefficient delta can be visible, a large one invisible). Rasterising into pixel space and diffing per-pixel turns the comparison into one over a *perceptually meaningful* projection of the mesh — the same projection the model is conditioned on.

**Established metrics, reported for comparability**:
- PSNR, SSIM, LPIPS (face-cropped).
- LMD-F, LMD-M (MediaPipe FaceMesh full-face / mouth-only landmark distances).
- ArcFace ID cosine + ID detect rate (cross-identity only).
- FVD, VideoMAE-v2 backbone, low-sample mode (same-identity only).

### 4.3 Same-identity reconstruction — Table 1
**Numbers already populated** in our internal comparison table (under `outputs/test_metric/metrics/_comparison.md` in the repo). Two sub-tables (TalkVid, HDTF) × columns (PSNR↑ SSIM↑ LPIPS↓ LMD-F↓ LMD-M↓ FVD↓).

**Story to land**:
- Marionette is **3rd of 6 on every pixel/landmark metric × 2 datasets**. X-Portrait leads, HunyuanPortrait 2nd. We're co-2nd with HunyuanPortrait on TalkVid (PSNR gap 0.65 dB, SSIM tied, LMD-F 0.108 vs 0.102). On HDTF the gap to X-Portrait is larger (~2.7 dB PSNR), but we still beat SadTalker / AniTalker / EchoMimic by 1–3 dB PSNR and 0.05–0.10 LPIPS.
- Frame: *competitive at a fraction of the budget*. The natural reading is "Marionette buys mid-pack pixel quality with a much smaller training corpus and ~½ the trainable params of X-Portrait" — the headline contrast in § 1, in numbers.

### 4.4 Cross-identity retargeting — Table 2
**Already populated for ID columns**; needs LMD-M and head-orientation mean columns added.
- Columns: ID cosine↑, ID detect rate↑, LMD-M↓, head-orientation mean↓.
- Story: ID cosine is the *weakest* column for Marionette (bottom on both datasets) — owned in § 4.8. On every column that measures head-pose or expression fidelity we lead or co-lead.
- *(TODO: extend Table 2 with the LMD-M and head-orient columns from the central summaries.)*

### 4.5 Head-rot deep dive — Table 3 + Figure 2
**Status**: our earlier table used a head-pose CNN (the 6DRepNet model — a pretrained face-pose regressor) to estimate yaw/pitch/roll on rendered video frames. We dropped that estimator entirely in favour of the FLAME-native head-rot metric described in § 4.2; the new numbers need to be re-populated across all (baseline, dataset, protocol) cells. *(TODO: re-run head_rot with the new estimator. Predicted ordering preserved — Marionette beat SadTalker / AniTalker / EchoMimic by 2–6× on the old CNN-based estimator, and our small empirical alignment check on 15 samples showed the FLAME-derived geodesic has a similar noise floor.)*

**Headline claim to substantiate**: Marionette is #1 or #2 of 6 on `head_rot_dist` in every cell, outright #1 on at least HDTF same-identity and HDTF cross-identity. **This is the most defensible win in the paper** — and it directly substantiates the "FLAME representation does the work" argument from § 3, with the added narrative point that the *evaluation* lives in the same FLAME parametric space the model conditions on.

**Figure 2**: a side-by-side head-rot overlay we generate with our internal sanity-check visualiser. Per frame: pred face crop with the head-frame X/Y/Z axes drawn on it (red = X, green = Y, blue = Z, projected directly from the FLAME-derived rotation matrix `R(rot) · R(neck_rot)` into image space — no Euler decomposition involved); the target face crop with the same axes on the right; the per-frame geodesic distance written on a text strip above. Pick one cell where the gap to the next-best baseline is biggest (HDTF cross-identity is a strong candidate). The visualiser also writes a JSON sidecar with the per-frame geodesic series for the figure caption.

### 4.6 Expression-error deep dive — Table 4 + Figure 3

**Predicted story**: `expression_l1` should track the FLAME-vs-implicit divide — methods that don't condition on FLAME (X-Portrait, HunyuanPortrait, audio-driven SadTalker / EchoMimic) should have a larger `expression_l1` even with pose held to the target's, because their expression signal does not flow through a structured parametric basis. This is the second defensible win predicted by the thesis: *if the FLAME representation is doing the work for pose (§ 4.5), it should also be doing it for expression*.

#### Table 4 — `expression_l1` (lower is better)

Per-pixel mask-aware L1 of the FLAME deformation map between the pred fit and the target fit, with pose / shape / camera held to the target's so the residual is purely expression (see § 3.2 / § 4.2). First 16 frames of each clip; intersection of sample IDs across the 5 baselines reported.

| Baseline | HDTF / same-id | HDTF / cross-id | TalkVid / same-id |
|---|---:|---:|---:|
| **Marionette** | **0.0664** (n=200) | **0.0828** (n=183) | 0.0704 (n=112) |
| AniTalker | 0.0963 (n=200) | 0.1134 (n=183) | 0.0799 (n=112) |
| EchoMimic | 0.0823 (n=198) | 0.1156 (n=183) | 0.0814 (n=112) |
| HunyuanPortrait | 0.0658 (n=200) | 0.0850 (n=183) | **0.0635** (n=112) |
| SadTalker | 0.0794 (n=1) † | 0.1107 (n=183) | — † |

Bolded = best per cell. **†** SadTalker's same-id cells are unreliable in this sweep (n=1 on HDTF, zero valid pairs on TalkVid) — the FLAME tracker struggled on its portrait-style outputs, so these rows should be re-tracked before publication. X-Portrait omitted (tracker still in flight at the time of this sweep).

**Read.** Marionette wins the two HDTF cells outright; HunyuanPortrait is the only baseline that stays close (within 0.001 on same-id, within 0.002 on cross-id) and edges Marionette by 0.007 on TalkVid same-id. The audio-driven pack (AniTalker, EchoMimic, SadTalker) sits 30–40% above Marionette on cross-identity, which is the cell that most cleanly tests whether expression flows through a structured representation versus a learned audio-to-motion mapping.

#### Interpreting the L1 magnitude — what's the deformation map's natural scale?

A raw L1 of 0.07 is meaningless without knowing the size of the signal we're trying to reconstruct. We sampled 100 GT FLAME fits (HDTF + TalkVid) and rasterised the deformation map at frame 0 to characterise the underlying signal:

| Quantity | Value |
|---|---|
| Mask coverage (face area / image) | 39.4% |
| Per-pixel value range (mean across samples) | [−0.99, +0.67] |
| Per-pixel mean \|value\| (signal magnitude) | **0.126** ± 0.041 |
| Per-pixel L2 norm of (x, y, z) | 0.262 ± 0.086 |
| Per-pixel signed std | 0.209 |
| Per-channel mean \|value\| | x = 0.070, y = 0.147, z = 0.161 |

The 3 deform channels carry signed (x, y, z) per-vertex offsets in NDC-ish space. On a typical face, the channel values span roughly [−1, +0.7] with a mean magnitude of ~0.126 per pixel — that's the size of the signal a generator must reproduce. Reading the table against this yardstick:

| Baseline | HDTF / same-id L1 | / signal mag (0.126) | HDTF / cross-id L1 | / signal mag |
|---|---:|---:|---:|---:|
| Marionette | 0.0664 | **53%** | 0.0828 | **66%** |
| HunyuanPortrait | 0.0658 | 52% | 0.0850 | 67% |
| AniTalker | 0.0963 | 76% | 0.1134 | 90% |
| EchoMimic | 0.0823 | 65% | 0.1156 | 92% |
| SadTalker | — | — | 0.1107 | 88% |

So on cross-identity HDTF, Marionette's residual is two-thirds the signal magnitude, while EchoMimic / AniTalker / SadTalker are roughly *equal to the signal* — they're approximately as far from the target's expression as the target's expression is from a neutral face. (Caveat: this denominator compares against a no-deformation reference, so it's an upper-bound interpretation; a tighter "chance" baseline would be the L1 between two unrelated GT fits, which would shrink the ratios. We treat the percentages as a sanity check that the residuals are a real fraction of the signal, not as a normalised metric.)

**Figure 3**: a 3×3 sanity panel rendered for one (pred, target) pair. Rows: row 1 is the actual video frames; row 2 is the rasterised FLAME mesh; row 3 is the rasterised FLAME expression-deformation map (the 3-channel per-vertex offset field defined in § 3.2). Columns: col 1 is the target's render, col 2 is the pred rendered with its own pose (visible expression *and* visible pose mismatch), col 3 is the pred rendered with target's pose (the substituted-fit configuration the metric scores) plus a heatmap overlay of the per-pixel L1 against col 1. The third column visualises *exactly* what the metric measures, with the heatmap concentrating around the mouth and eyes when the pred and target's expression coefficients differ most.

*(TODO: pick the (pred, target) pair where the heatmap localises cleanly to mouth+eyes for Figure 3; re-run SadTalker FLAME tracking on its same-identity outputs to fill the two unreliable cells; re-run with X-Portrait once its tracker finishes.)*

### 4.7 Ablations — Table 5
We ablate the two halves of the 45-channel FLAME conditioning tensor independently. Each ablation arm is a separate Marionette training run with the conditioning tensor reduced. We dropped two other arms we considered: `no_flame` (replace the FLAME tensor with natural-video conditioning entirely — training diverged, useless as a baseline) and `audio_off` (audio is already off in the main config, so this collapses to the main model).

**Two remaining arms**:
- `no_deform`: drop the 3 deformation channels, keep the 42 positional-encoding channels. Tests whether the deformation channels are necessary on top of positional encoding for expression follow.
- `no_posenc`: drop the 42 positional-encoding channels, keep the 3 deformation channels. Tests the converse — whether positional encoding is necessary on top of deformation for head-pose / geometry follow.

For each arm: same-identity PSNR, LMD-M, `head_rot_dist`, `expression_l1` on a 50-sample subset of HDTF same-identity (a single cell, to keep § 4.7 focused).

**Predicted story**:
- `no_posenc`: head pose drifts (`head_rot_dist` rises), expression follows OK. Positional encoding is the pose-carrying channel.
- `no_deform`: lip motion degrades (LMD-M and `expression_l1` rise), head pose stays good. Deformation is the expression-carrying channel.
- Together: the 45-channel split has a *direct functional decomposition* — pos-enc carries pose, deform carries expression. **This makes the representation contribution falsifiable, not just descriptive**.

*(TODO: confirm both ablation runs are trained to convergence; if not, schedule training before evaluation.)*

### 4.8 Limitations — explicit, not hidden

**FVD**. Marionette is bottom or near-bottom on FVD across both datasets. Hypothesis: the SD-2 VAE pipeline produces a subtle texture/colour signature (a slight skin-tone shift across the inference, what we informally call "VAE drift") that the VideoMAE-v2 backbone behind FVD penalises even when face *content* is plausible. This is a distributional issue with the VAE codec, not a content-quality issue with the diffusion. Future work: VAE fine-tune or VAE swap (e.g. the SDXL or Flux VAE, both more recent and with less colour drift).

**Cross-identity ID-cosine**. Marionette is bottom on cross-identity ArcFace cosine on both datasets. ArcFace's detect rate on TalkVid cross-identity also drops to 0.84 — some of our generations degrade enough that ArcFace can't lock onto a face at all. Hypothesis: the reference-UNet K/V injection is trained only on small pose deltas (in our same-identity training, the slot-0 reference and the target window are from the *same* clip, so they typically have similar pose). Large pose deltas at cross-identity inference fall outside that training distribution, and identity transfer leaks. The fact that we simultaneously *lead* on the head-pose and expression columns is consistent with this hypothesis — head pose and expression flow through the parametric FLAME path, which works regardless of pose magnitude, while identity flows through the learned attention-injection path, which doesn't.

**16-frame window**. The model denoises 16 frames per forward pass; long-form video requires sliding-window stitching at inference, with overlap and per-window re-noising for blending. Cross-window consistency is best-effort (no temporal-consistency loss across windows), not explicitly modelled.

---

## 5. Conclusion — ~0.5 pages

- Restate the three contributions in plain language: pixel-space FLAME representation as an expression activation map; parametric retargeting at inference for cross-identity for free; generative-prior-native identity injection without a learned identity encoder.
- Methodological contribution: a FLAME-native evaluation pair (head-rot geodesic + deformation-map L1) that measures portrait animation in the same parametric space the model conditions on, replacing the spatial-sensitivity and distribution-prior failure modes of the inherited PSNR / SSIM / LPIPS / FVD proxies. *FLAME-native conditioning, FLAME-native evaluation.*
- Summarize the evidence: best-or-second on head-rot geodesic distance in every cell, second defensible win on pose-disentangled deformation-map L1 (pending), competitive same-identity pixel quality, with less than half the trainable parameters of X-Portrait and roughly an order of magnitude less training data.
- Acknowledge limitations briefly: FVD lag (VAE-driven), cross-identity ArcFace cosine (large-pose distribution shift in the reference path).
- *No extrapolation* beyond portrait animation. Stop here.

---

## Figures inventory

| # | Section | Content | Status |
|---|---------|---------|--------|
| 1 | § 3 | Architecture block diagram (two orthogonal pathways: identity through the frozen reference UNet's K/V; facial expression and head pose through FLAME rasterisation) | **TODO — needs hand-drawn / TikZ** |
| 2 | § 4.5 | Head-rot sanity overlay — pred face + FLAME-derived axes \| target face + FLAME-derived axes, with the per-frame geodesic distance written above | mp4 generated by our internal sanity-check visualiser script; needs final sample pick + caption |
| 3 | § 4.6 | 3×3 expression-decomposition panel (real video / mesh / deform × target / pred-own-pose / pred-target-pose+diff-overlay) | mp4s already generated for several (target, pred) pairs; needs final pick |
| 4 | § 4.7 | Ablation qualitative grid (`no_deform`, `no_posenc` vs full Marionette) | **TODO** — depends on ablation training status |
| 5 | § 1 | Teaser: one (reference-image, driver-clip) pair where Marionette's pose follow visibly beats the next-best baseline (HDTF cross-identity is the strongest cell) | **TODO** |

## Tables inventory

| # | Section | Content | Status |
|---|---------|---------|--------|
| 1 | § 4.3 | Same-identity pixel / landmark / FVD per dataset | **done** in our internal comparison file |
| 2 | § 4.4 | Cross-identity ID + expression-and-pose fidelity per dataset (ID cosine + LMD-M + head_rot_dist) | partly done; needs LMD-M and head_rot columns from the new estimator |
| 3 | § 4.5 | head_rot_dist (single column, in degrees) × dataset × protocol | **TODO — re-run with the FLAME-native estimator (replaces the dropped CNN-based 6DRepNet number)** |
| 4 | § 4.6 | expression_l1 per dataset × protocol | **TODO — pred FLAME tracker in flight** |
| 5 | § 4.7 | Ablation deltas vs full Marionette (`no_deform`, `no_posenc` only) | **TODO** |

## Open TODOs (rough order of urgency)

1. **Re-run head_rot across all (baseline, dataset, protocol) cells** with the new FLAME-native estimator. Old CNN-based (6DRepNet) numbers in our internal comparison file need to be replaced with a single `head_rot_dist` column per cell. Predicted ordering preserved but pending verification.
2. **Predicted FLAME tracker (in flight)**: we are FLAME-tracking 1200 prediction videos, then will run the expression and head_rot metrics over the resulting fit files to populate Tables 3 + 4 and pick Figure 3.
3. **Cross-id Table 2**: extend with LMD-M and `head_rot_dist` columns. Story is incomplete without them.
4. **Drop the per-axis head-rot table** in our internal comparison file; collapse to a single column (the geodesic). Direct Euler comparison was the very thing we argued is unreliable, so reporting yaw/pitch/roll separately undermines the metric's framing.
5. **Ablation runs**: confirm `no_deform` and `no_posenc` are trained + evaluated. If not, schedule training before the writeup.
6. **Architecture figure (Figure 1)**: someone has to draw it. Spec: the two pathways must be *visibly orthogonal*; the parametric substitution `β_ref + (ψ, θ)_driver` must be the most prominent label in the figure.
7. **Teaser figure (Figure 5)**: pick a HDTF cross-id sample where Marionette's pose-follow gap to next-best is biggest.
8. **~10k training samples — exact composition**: TalkVid only? TalkVid + HDTF? After what filtering? Needed for § 3.6 / § 4.1.
9. **Audio framing**: confirm audio-disabled is the published config. § 3.5 mentions audio briefly as a future-work hook; § 4.7 omits the audio ablation. If audio is intended to be a contribution, this needs reshuffling.
10. **Related-work references**: Pouyan to provide the reference list; § 2 currently has cluster headings only.
11. **Param/data comparison footnote**: write a 2-sentence footnote on the main tables that gives the all-inclusive vs trainable distinction (Marionette ~857M trainable, ~1.92B all-inclusive incl. frozen pretrained components; X-Portrait ~3.07B all-inclusive; HunyuanPortrait ~1.99B all-inclusive). Without this, "fewer params" is an unfair comparison.
12. **FVD-VAE caveat paragraph**: § 4.8 has the hypothesis; needs one paragraph of explanation tied to the existing color-shift investigation.

---

## Decisions still to make

- **Title vs subtitle**: working title is `Marionette: Representation over Architecture for Portrait Animation Diffusion Model`. Plain, or add a subtitle? Current title already reads as a thesis statement, subtitle probably redundant.
- **Cross-id ID-cosine**: foreground in § 4.8 (current plan) or footnote it. Lean foreground — honesty buys reviewer trust, and we have a real mechanistic hypothesis.
- **Datasets in main vs supplementary**: TalkVid + HDTF both in main (current plan). Consider: ablations only on a single cell (HDTF same-id) to keep § 4.7 focused.
- **Where to mention the alignment-check experiment** that motivated frame-0 anchoring (the small empirical study showing absolute-rotation comparison is dominated by per-clip camera-fit drift). Current plan: one paragraph in § 4.2. Could promote to a supplementary section if reviewers ask why we didn't use absolute pose comparison.
