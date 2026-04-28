# Marionette — paper scaffolding & planning

Working planning doc for the Marionette paper. Each section gives **the claim it lands**, **the evidence we already have**, **figure/table slots with status**, and **open decisions**. Not paper prose — a contract for what to write.

---

## Working title

**`Marionette: Representation over Architecture for Portrait Animation Diffusion Model`**

Subtitle option (if a subtitle is wanted): *Representation over Architecture in Identity-Preserving Portrait Animation*.

Naming: we call the task **portrait animation from a driving video**, not "talking-head video generation". Audio is disabled in the published configuration; the task is video-driven animation. (See open decisions if you want this revisited.)

---

## North-star one-line claim

> Marionette animates a single reference portrait from a driving video by **lifting FLAME blendshape deformations into pixel space** as a rasterized expression-activation map — instead of feeding raw RGB or FLAME parameter vectors into a learned motion-control module. At inference, the driver's expression and pose parameters are reapplied **parametrically** onto the reference's FLAME shape, producing geometrically consistent driver-specific motion *without any cross-identity training*. Identity flows through a separate path: features from a frozen reference UNet (the same SD-2 backbone used for generation) inject directly into the matching multi-resolution self-attention layers of the gen UNet, **eliminating the need for a learned identity encoder**. The result is competitive portrait animation with **less than half the trainable parameters of X-Portrait** and trained on **roughly an order of magnitude less data** than the leading diffusion-prior baselines, while leading on driver-pose fidelity.

If the reader walks away with one paragraph, this is it. Every section pulls toward it.

---

## What X-Portrait and HunyuanPortrait actually do (so we know what we're contrasting)

Quick architectural reads from their respective `impl/` trees. We want this in the back of our minds when writing § 1 and § 2 — every sentence positioning Marionette should be verifiable against this.

### X-Portrait (SIGGRAPH 2024, ByteDance) — a three-ControlNet stack on SD-1.5
- **Backbone**: SD-1.5 LDM, 64×64 latent, with temporal attention bolted on (`ControlledUnetModelAttn_Temporal_Pose_Local`).
- **Three trainable ControlNet branches** (each ~the size of an SD-1.5 UNet):
  1. `ControlNetReferenceOnly` — appearance/identity. AnimateAnyone-style "reference-only" control: the reference image flows through this branch and its features are read into the gen UNet through cross-coupling, not via a separate identity encoder.
  2. `ControlNet` (global pose) — consumes an RGB-derived "pose" hint built from the driving video, after pose-aligning the driving frames to the reference using face-alignment 68-pt landmarks (`adjust_driving_video_to_src_image`).
  3. `ControlNet` (local pose) — consumes a *masked RGB patch* of the driving video around eye and mouth landmark centers (`extract_local_feature_from_single_img`), zeroing everything else. Brings high-frequency local motion in via raw pixels.
- **Total trainable params**: ~3.07 B (all in one `model_state-415001.th`, 12.3 GB fp32). All three ControlNets, plus the temporal module on top of the gen UNet, are trained.
- **What they do not use**: no explicit 3D parametric face model (FLAME / 3DMM). All motion is inferred from 2D RGB driver frames via two ControlNets that consume rendered/cropped pixel patches. Optional initial-noise from a separate `facevid2vid` model is supported but not architectural.
- **Headline fact for our contrast**: motion comes from RGB → ControlNet, and the cost is paid in *trained parameters* (two motion ControlNets ≈ ~640M each).

### HunyuanPortrait (arXiv 2503.18860, Tencent) — implicit motion features + IP-Adapter identity on SVD
- **Backbone**: Stable Video Diffusion (`SVD-img2vid-xt`) — a 25-frame video diffusion model, much heavier inference path than SD-1.5 / SD-2.
- **Motion path (implicit, from RGB)**:
  - `HeadExpression` (ResNet-18-GN regressor over RGB) → 512-dim expression vector per frame.
  - `HeadPose` (ResNet regressor) → pose features.
  - `IntensityAwareMotionRefiner` — Perceiver-style resampler that takes per-frame motion features + intensity buckets and produces ~64 motion tokens.
  - The paper title says it explicitly: *"Implicit Condition Control"* — there is no explicit 3DMM. Motion is whatever those regressors learn.
- **Identity path (IP-Adapter style)**:
  - `ArcFace` ONNX encoder → identity embedding.
  - `DINOv2-Large` (vit_large, 314M params) → image features, projected by `ImageProjector` to UNet token dim.
  - Both injected via IPAdapterAttnProcessor (separate K/V projection adapters) into UNet cross-attention.
- **Pose guider**: small CNN that processes a coarse pose signal (~1.1M params).
- **Total params**: ~1.99 B all-inclusive (UNet 1.58B + DINO 314M + motion_proj 66M + expression 25M + headpose 11M + arcface tiny + image_proj tiny).
- **Headline fact for our contrast**: motion is *learned implicitly* from RGB through two regressors and a Perceiver resampler. Identity uses *separately-trained projection adapters* (ArcFace + DINO + image_proj), not the gen UNet's own backbone.

### How Marionette differs (one bullet per axis)
| Axis | X-Portrait | HunyuanPortrait | **Marionette** |
|---|---|---|---|
| Motion representation | RGB patches → 2 ControlNets | RGB → ResNet regressors → motion tokens | **FLAME blendshapes rasterized to pixel space (45-ch activation map)** |
| Cross-identity strategy | Pose-align driver to ref via landmarks; train end-to-end | Train end-to-end on motion features | **Parametric retargeting at inference: `β_ref + ψ_driver + θ_driver`** — no cross-id training |
| Identity injection | ReferenceOnly ControlNet (trained branch) | ArcFace + DINO + image_proj → IP-Adapter K/V | **Frozen SD-2 ref UNet → multi-res self-attn K/V → into matching gen UNet layers (no learned identity encoder)** |
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
- **Numbers (relative)**: ~½ the trainable parameters of X-Portrait, comparable parameter count to HunyuanPortrait but trained on roughly an order of magnitude less data; leads or co-leads on per-axis head-orientation L1 across all four (dataset, protocol) cells; competitive same-identity pixel quality.
- **Limitations briefly**: FVD lags (texture drift in the SD-2 VAE pipeline); cross-identity ArcFace cosine is the weakest column.

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
Marionette inverts the bet. Motion is not learned from RGB; it is **read off a parametric face model** (FLAME) and **rasterized into pixel space**:
- The conditioning tensor we feed the diffusion model is a 45-channel rasterization of the retargeted FLAME mesh: 42 channels of sinusoidal positional encoding of vertex positions in normalized device coordinates, plus 3 channels of per-vertex expression deformation (offsets relative to a neutral face).
- Crucially this is **not** a parameter-vector conditioning (cf. older audio-3DMM hybrids that ingest `(ψ, θ)` as a vector). The signal is delivered in image space, **as an activation map**, natively aligned with the diffusion model's own convolutional inputs.
- Cross-identity animation comes for free at inference via a simple parametric substitution — no cross-identity training required (§ 3.3).
- Identity is injected through the diffusion prior's *own* frozen reference UNet — no learned identity encoder is added (§ 3.4).

### ¶ 4 — Other clusters (brief)
For completeness, but kept short:
- **Audio-driven, no visual driver** (SadTalker, EchoMimic): cannot faithfully transfer a specific driver's motion. Useful as a "no-driver" baseline.
- **Coarse-landmark-driven** (older talking-head work): 68-pt landmarks are too sparse to express the fine facial geometry that FLAME blendshapes do, and they're 2D — they bake in the driver's identity geometry by construction.
- **Audio + 3DMM-vector hybrids**: ingest 3DMM parameters as a flat vector through cross-attention; semantically meaningful but lose the spatial structure that the rasterization provides.

### ¶ 5 — Contributions (3 + 1)
1. **Pixel-space FLAME representation as an expression activation map** *(§ 3.2)*. We rasterize FLAME blendshape deformations into a 45-channel image-space tensor and feed it as the diffusion conditioning. This is the core differentiator from methods that consume raw FLAME parameter vectors as conditioning, and from methods that learn motion features from RGB.
2. **Parametric retargeting at inference enables cross-identity reenactment without cross-identity training** *(§ 3.3)*. We train *only* on same-identity self-supervised data; cross-identity comes from a parametric substitution `β_ref + ψ_driver[t] + θ_driver[t]` rendered through the reference's camera. The same forward path runs in both modes. This eliminates the need for the expensive cross-identity training corpus that diffusion-prior baselines rely on.
3. **Generative-prior-native identity injection** *(§ 3.4)*. We pass the reference image through a frozen copy of the same SD-2 UNet used for generation, capture self-attention features at every multi-resolution layer, and inject them as additional K/V tokens into the *matching* layers of the gen UNet. Because the embeddings already live in the generative prior's native feature space, no identity encoder needs to be trained — IP-Adapter-style projection layers and bespoke ID encoders (ArcFace, DINOv2) become unnecessary.
4. **Two new evaluation metrics for portrait animation** *(secondary, § 4.2)*. (i) Per-axis head-orientation L1 in degrees from 6DRepNet — a missing baseline number in current papers. (ii) **Pose-disentangled deformation-map L1** — render the rasterized FLAME deformation map twice, once with the predicted pose and once with the GT pose substituted in, to isolate expression error from pose error. Both adopted by the community would standardize what current papers conflate.

### ¶ 6 — Results in one breath, in relative terms
Marionette **leads or co-leads on per-axis head-orientation L1 across all four (dataset, protocol) cells** (best mean on HDTF same-id at 0.94°, best mean on HDTF cross-id at 2.17°, second on TalkVid same-id and cross-id, beating the bottom three baselines by 2–6× consistently). Same-identity pixel quality is competitive with the leading baseline (within 0.7 dB PSNR on TalkVid). All of this with **less than half the trainable parameters of X-Portrait** and roughly an order of magnitude less training data.

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

**Section claim**: a structured FLAME representation, applied parametrically at inference, paired with a generative-prior-native identity path, can replace the architectural complexity of motion-from-RGB control modules.

### 3.1 Background — FLAME parametric model
Brief recap. FLAME is a parametric face model with three disentangled parameter groups: identity shape `β`, expression `ψ`, pose `θ` (head, jaw, neck, eye), plus camera. Forward operation: skinned vertex positions `V(β, ψ, θ) ∈ R^{V × 3}`. We use the CAP4D variant (with mouth verts and a 65-dim expression basis).

### 3.2 Pixel-space FLAME representation — the *expression activation map* (contribution 1)
Given a per-frame FLAME fit `(β, ψ[t], θ[t], camera)`, we run **one** pytorch3d rasterization pass and produce a 45-channel image-space tensor:
- `[0:42]` — sinusoidal positional encoding of rasterized vertex positions in normalized device coordinates (3 raw NDC channels × 14 frequencies).
- `[42:45]` — per-vertex expression deformation: the offset of each vertex from its neutral position under the current `ψ`. Rasterized in the same pass.

Both feature streams are masked by the on-mesh indicator. This 45-channel tensor is passed through a small conv encoder (`ConditioningEncoder`, 512 → 64 channels, zero-init final layer) and **added** to the first feature map of the gen UNet — exactly the position where pixel-space conditioning has the strongest receptive field.

**Why this is different from prior FLAME conditioning**: previous methods consume the FLAME parameter vector `ψ` (65-dim) as a token through cross-attention. Our representation is **already in image space** when it reaches the diffusion model — same coordinate system, same spatial resolution, same convolutional locality. We call it an *activation map* because the diffusion model can consume it through ordinary convolutions, the same way it consumes its own intermediate features, not through a learned bridge.

**Why this is different from RGB-derived motion (X-Portrait, HunyuanPortrait)**: the deformation field comes from *parametric* FLAME forward, not from a regressor over driver RGB. There is no driver-identity leakage in the conditioning by construction — the deformation is purely a function of `(β, ψ, θ)`, which we control. § 3.3 makes this load-bearing.

### 3.3 Parametric retargeting at inference — cross-identity for free (contribution 2)
This is the section the paper hinges on.

**Setup**: at training we run only same-identity self-supervised optimization. One clip provides the reference frame, the target window, and the driver. The model never sees a cross-identity pair.

**Inference, single equation**:
- *Same-identity reconstruction*: build per-frame verts `V(β_ref, ψ_ref[t], θ_ref[t])` rendered through `camera_ref`. Standard.
- *Cross-identity retargeting*: build per-frame verts `V(β_ref, ψ_driver[t], θ_driver[t])` rendered through `camera_ref`. **The only thing that changes is which `(ψ, θ)` go into the same forward path.**

That is, retargeting is a **parametric substitution**, not a learned operation. Because FLAME's parameter space disentangles shape from expression and pose, swapping `(ψ, θ)` between subjects produces a geometrically consistent driver-specific motion adapted to the reference's morphology. **Driver identity cannot leak into the conditioning** — the only driver-derived quantities are `(ψ, θ)`, which by FLAME's definition are identity-orthogonal.

**Consequence**: the model needs to be trained only on the same-identity objective, yet cross-identity reenactment works at inference. This eliminates the cross-identity training requirement that the diffusion-prior baselines need to satisfy with large proprietary corpora. **Stating this plainly is the paper's headline mechanism**.

### 3.4 Generative-prior-native identity injection (contribution 3)
**Setup**: identity is *not* a learned signal in Marionette — we let the generative prior carry it.

**Mechanism**: we instantiate a frozen copy of SD-2's UNet (`RefFeatureExtractor`) and run the VAE-encoded reference image through it once. Forward hooks on each `BasicTransformerBlock`'s pre-attention LayerNorm capture the input to every self-attention block at every UNet resolution. These per-layer features `(B, HW_k, D_k)` are then **injected as additional K/V tokens** into the matching self-attention block of the *gen* UNet. Every gen-frame query attends to its own tokens *and* the reference's, at the same resolution, every layer.

**Why no learned encoder is needed**: the embeddings already live in the generative prior's native feature space. Both the ref UNet and the gen UNet are the same SD-2 architecture with the same checkpoint at initialization. There is no domain gap to bridge, no learned projection to train. The reference does **not** occupy a slot in the gen tensor — it lives only in the K/V-injected attention path. Loss is uniform ε-MSE across all T target slots; no masking, no auxiliary identity loss.

**Contrast in one sentence**: where HunyuanPortrait pairs ArcFace + DINOv2 + an `ImageProjector` and trains IP-Adapter K/V projection adapters, Marionette captures features from the gen prior's own backbone and injects them directly. The "identity encoder" is the gen prior itself, frozen.

### 3.5 Diffusion backbone (brief)
- SD-2.1 LDM, latent res 64×64 (8× VAE downsample of 512×512), `T = 16` frames per forward pass.
- 2D self-attention replaced with 3D spatiotemporal attention for inner stages.
- Audio cross-attention exists in the codebase but is **disabled in the published configuration**; treated as a future-work hook, not a contribution. *(TODO: confirm with Pouyan that this is how we want to frame audio.)*
- VAE frozen end-to-end.

### 3.6 Training
- **Self-supervised, same-identity only**. Each `(T+1)`-frame sample comes from a single clip: slot 0 is a reference frame drawn from an *independently sampled position* in the clip (different seed than the target window), slots 1..T are the target window. VAE-encode all T+1, feed slot 0 to the ref UNet, ε-MSE loss uniformly over slots 1..T.
- **Why this generalizes to cross-identity at inference (key paragraph)**: the slot-0 ref is from a different position in the same clip than the target window, so the model is *forced* to learn an identity prior that generalizes across pose / expression mismatch. At inference, parametric retargeting (§ 3.3) gives the same surface form — driver pose/expression rendered in the reference's camera — except `β` no longer matches the target. The model has already learned to bridge "pose mismatch" between ref slot and target window; cross-identity inference is the same thing with a `β` substitution that the FLAME representation handles parametrically.
- Training corpus size: **~10k samples** *(TODO: exact composition — TalkVid only? TalkVid + HDTF subset? Include filtering criteria.)*
- Schedule: 30 k steps, virtual batch 2, gpu batch 2, T = 16. Trained on 4–8 H200s.
- CFG: bernoulli-dropout the conditioning during training.

### Figure 1 — architecture
Two-pathway block diagram. **Both paths visibly orthogonal** in the layout — top-of-figure path is Identity, bottom path is Motion, the gen UNet is in the middle.
- *Identity path*: ref image → VAE encode → frozen SD-2 ref UNet (one pass) → per-layer self-attn features → arrows into matching layers of gen UNet.
- *Motion path*: driver fit `(ψ_driver, θ_driver)` + ref shape `β_ref` + ref camera → FLAME forward → 45-ch rasterizer → ConditioningEncoder → added to first feature map of gen UNet.
- *Gen UNet*: receives noise + spatial_cond (added) + identity K/V (injected) → denoised T-frame latents → frozen VAE decode → output video.
- The figure must make the parametric substitution `β_ref + (ψ, θ)_driver` *visually obvious* — that is the paper's thesis in one image.

*(TODO: someone has to draw this. Suggest tikz or hand-illustrated.)*

---

## 4. Experiments — ~3 pages

### 4.1 Setup
- **Datasets**: TalkVid (in-the-wild ID diversity, ~125 eval clips per protocol after filtering) + HDTF (high-resolution studio, ~125 eval clips per protocol).
- **Protocols**:
  - *Same-identity reconstruction*: ref and driver are the same clip; tests pixel-level fidelity and motion follow.
  - *Cross-identity retargeting*: ref and driver are different clips; tests identity preservation under motion transfer.
- **Baselines**: SadTalker, AniTalker, EchoMimic, HunyuanPortrait, X-Portrait. Default sampling configs, public weights, fixed seeds. *(TODO: pin commit + config of each baseline in supplementary.)*
- **Frame coverage**: every metric scored on the **first 16 frames** of each prediction (matches Marionette's `cfg.inference.n_frames=16`); SOTA's 75–125-frame outputs truncated to 16. FVD reported only for same-identity (it does not apply to cross-id).
- **Face cropping**: pred and target independently RetinaFace-cropped (1.3× bbox) and resized to 512×512 before scoring. Removes background bias; the metric measures the face, not the lighting.

### 4.2 Metrics
Split deliberately into **established** and **new (this work)** to make the methodological contribution visible.

**Established**:
- PSNR, SSIM, LPIPS (face-cropped).
- LMD-F, LMD-M (MediaPipe FaceMesh full-face / mouth-only landmark distances).
- ArcFace ID cosine + ID detect rate (cross-identity only).
- FVD, VideoMAE-v2 backbone, low-sample mode (same-identity only).

**New (this work)**:
- **Head-orientation L1, per-axis, in degrees**. 6DRepNet on face crops; report yaw/pitch/roll L1 separately and `axis_mean = (yaw + pitch + roll) / 3` as the headline. Weighted by detect rate. Filling a gap — current papers do not report head-pose fidelity in absolute degrees.
- **Pose-disentangled deformation-map L1** *(in flight — Phase B)*. For each (pred, GT) pair we render the rasterized FLAME deformation map twice: once *combined* (pred uses its own `(ψ_pred, θ_pred)`) and once *pure* (pred's `(ψ_pred, eye_rot, jaw_rot)` substituted into GT's `(θ_gt, β_gt, camera_gt)` — only expression coefficients can drive a residual). L1 against GT's deformation map; the *pure* variant is the headline expression-error metric. Implementation already validated by the side-by-side panels in `outputs/test_metric/expr_sanity/`. *(TODO: numbers from § 4.6.)*

### 4.3 Same-identity reconstruction — Table 1
**Already populated** in `outputs/test_metric/metrics/_comparison.md`. Two sub-tables (TalkVid, HDTF) × columns (PSNR↑ SSIM↑ LPIPS↓ LMD-F↓ LMD-M↓ FVD↓).

**Story to land**:
- Marionette is **3rd of 6 on every pixel/landmark metric × 2 datasets**. X-Portrait leads, HunyuanPortrait 2nd. We're co-2nd with HunyuanPortrait on TalkVid (PSNR gap 0.65 dB, SSIM tied, LMD-F 0.108 vs 0.102). On HDTF the gap to X-Portrait is larger (~2.7 dB PSNR), but we still beat SadTalker / AniTalker / EchoMimic by 1–3 dB PSNR and 0.05–0.10 LPIPS.
- Frame: *competitive at a fraction of the budget*. The natural reading is "Marionette buys mid-pack pixel quality with a much smaller training corpus and ~½ the trainable params of X-Portrait" — the headline contrast in § 1, in numbers.

### 4.4 Cross-identity retargeting — Table 2
**Already populated for ID columns**; needs LMD-M and head-orientation mean columns added.
- Columns: ID cosine↑, ID detect rate↑, LMD-M↓, head-orientation mean↓.
- Story: ID cosine is the *weakest* column for Marionette (bottom on both datasets) — owned in § 4.8. On every motion-fidelity column we lead or co-lead.
- *(TODO: extend Table 2 with the LMD-M and head-orient columns from the central summaries.)*

### 4.5 Head-orientation deep dive — Table 3 + Figure 2
**Already populated** in `_comparison.md` → "Head-orientation error (degrees, L1 vs driver/GT)".

**Headline claim** (paste verbatim): *Marionette is **#1 or #2 of 6 in every (axis × dataset × protocol) cell***, outright #1 on the headline mean for HDTF same-id (0.94°), HDTF cross-id (2.17°), and pitch on TalkVid same-id (1.54°). Beats SadTalker/AniTalker/EchoMimic by 2–6× consistently. **This is the most defensible win in the paper** — and it directly substantiates the "FLAME representation does the work" argument from § 3.

**Figure 2**: 6-row sanity panel (one row per baseline) for one selected sample, showing yaw/pitch/roll axes drawn on each baseline's predicted face vs the driver, with per-axis L1 in the title. Data already generated under `outputs/test_metric/head_pose_sanity/`. Pick one cell where the gap to the next-best is biggest (HDTF cross-id is a strong candidate).

### 4.6 Expression-error deep dive — Table 4 + Figure 3 *(Phase B — in flight)*
**Status**: tracker running on 1200 pred clips (50 samples per cell × 6 baselines × 4 cells), staged with first-16-frame truncation. After tracking, we wire `expression` as a metric group in `evaluator.py`, populate Table 4.

**Predicted story**: pure-expression L1 (with pose held to GT) should track the FLAME-vs-implicit divide — methods that don't condition on FLAME should have a larger pure-expression L1 even after pose is held constant. This is the second defensible win predicted by the thesis: *if the FLAME representation is doing the work for pose (§ 4.5), it should also be doing it for expression*.

**Figure 3**: the 3×3 sanity panel (real video / mesh / deform × GT / combined / pure-with-diff-overlay) — already generated under `outputs/test_metric/expr_sanity/` (10+ pairs, see `chuck_schumer_vs_aoc.mp4` etc. for examples).

*(TODO: populate Table 4 once tracker finishes; pick the figure-3 pair that shows the heatmap concentrated around mouth + eyes most cleanly.)*

### 4.7 Ablations — Table 5
**Two arms** (dropped `no_flame`, training diverged; dropped `audio_off` since audio is off in the main config):
- `no_deform` — drop the 3-ch deformation channels, keep the 42-ch pos-enc. Tests whether the deformation channels are necessary for expression follow on top of pos-enc.
- `no_posenc` — drop the 42-ch pos-enc, keep the 3-ch deformation. Tests the converse — whether pos-enc is necessary for pose/geometry follow on top of deformation.

For each: same-id PSNR, LMD-M, head-orientation mean, pure-expression L1 on a 50-sample subset of HDTF same-id (single cell to keep the table focused).

**Predicted story**:
- `no_posenc`: head pose drifts (head-orientation mean rises), expression OK. Pos-enc is the pose-carrying channel.
- `no_deform`: lip motion degrades (LMD-M and pure-expression rise), head pose stays good. Deform is the expression-carrying channel.
- Together: the 45-channel split has a *direct functional decomposition* — pos-enc carries pose, deform carries expression. **This makes the representation contribution falsifiable, not just descriptive**.

*(TODO: confirm both ablation runs are trained to convergence; if not, schedule training before evaluation.)*

### 4.8 Limitations — explicit, not hidden

**FVD**. Marionette is bottom or near-bottom on FVD across both datasets. Hypothesis: the SD-2 VAE pipeline produces a subtle texture/color signature (the "pale-skin / VAE-drift" symptom flagged in `_comparison.md`) that VideoMAE-v2's natural-video prior penalizes even when the face content is plausible. Distributional issue, not a content-quality one. Future work: VAE fine-tune or VAE swap (SDXL VAE, Flux VAE).

**Cross-identity ID-cosine**. Marionette is bottom on cross-id ArcFace cosine on both datasets. ArcFace detect rate on TalkVid cross-id also drops to 0.84 — some generations degrade enough that ArcFace cannot lock on. Hypothesis: the reference-UNet K/V injection is trained only on small pose deltas (slot-0 ref vs target window from the same clip); large pose deltas at cross-identity inference fall outside that distribution, and identity transfer leaks. The fact that we lead on motion-fidelity columns simultaneously is consistent with this — motion is *parametric* and works regardless of pose magnitude, but identity is *learned* and does not.

**16-frame window**. Long-form video requires sliding-window stitching at inference; cross-window consistency is best-effort, not modeled.

---

## 5. Conclusion — ~0.5 pages

- Restate the three contributions in plain language: pixel-space FLAME representation as an expression activation map; parametric retargeting at inference for cross-identity for free; generative-prior-native identity injection without a learned identity encoder.
- Summarize the evidence: best-or-second on per-axis head-orientation in 4/4 cells, competitive same-identity pixel quality, with less than half the trainable parameters of X-Portrait and roughly an order of magnitude less training data.
- Acknowledge limitations briefly: FVD lag (VAE-driven), cross-identity ArcFace cosine (large-pose distribution shift in the reference path).
- *No extrapolation* beyond portrait animation. Stop here.

---

## Figures inventory

| # | Section | Content | Status |
|---|---------|---------|--------|
| 1 | § 3 | Architecture block diagram (two orthogonal pathways: identity through frozen ref UNet K/V, motion through FLAME rasterization) | **TODO — needs hand-drawn / tikz** |
| 2 | § 4.5 | 6-row head-orientation sanity (axes drawn on faces) | data exists in `outputs/test_metric/head_pose_sanity/`; needs final pick + caption |
| 3 | § 4.6 | 3×3 expression-decomposition panel (real video / mesh / deform × GT / combined / pure+diff-overlay) | data exists in `outputs/test_metric/expr_sanity/`; needs final pick |
| 4 | § 4.7 | Ablation qualitative grid (`no_deform`, `no_posenc` vs full Marionette) | **TODO** — depends on ablation training status |
| 5 | § 1 | Teaser: one (ref, driver) pair where Marionette's pose follow visibly beats next-best (HDTF cross-id is the strongest cell) | **TODO** |

## Tables inventory

| # | Section | Content | Status |
|---|---------|---------|--------|
| 1 | § 4.3 | Same-id pixel/landmark/FVD per dataset | **done** in `_comparison.md` |
| 2 | § 4.4 | Cross-id ID + motion-fidelity per dataset (ID + LMD-M + head-orient mean) | partly done; needs LMD-M + head-orient columns |
| 3 | § 4.5 | Head-orientation per-axis × dataset × protocol | **done** in `_comparison.md` |
| 4 | § 4.6 | Deformation-map-diff (pure + combined) per dataset × protocol | **TODO — Phase B in flight** |
| 5 | § 4.7 | Ablation deltas vs full Marionette (`no_deform`, `no_posenc` only) | **TODO** |

## Open TODOs (rough order of urgency)

1. **Phase B (in flight)**: FLAME-track 1200 pred clips → wire `expression` group into `evaluator.py` → populate Table 4 + pick Figure 3.
2. **Cross-id Table 2**: extend with LMD-M and head-orientation mean columns. Story is incomplete without them.
3. **Ablation runs**: confirm `no_deform` and `no_posenc` are trained + evaluated. If not, schedule training before the writeup.
4. **Architecture figure (Figure 1)**: someone has to draw it. Spec: the two pathways must be *visibly orthogonal*; the parametric substitution `β_ref + (ψ, θ)_driver` must be the most prominent label in the figure.
5. **Teaser figure (Figure 5)**: pick a HDTF cross-id sample where Marionette's pose-follow gap to next-best is biggest. Numbers for the picker live in `_comparison.md` § Head-orientation HDTF cross-id.
6. **~10k training samples — exact composition**: TalkVid only? TalkVid + HDTF? After what filtering? Needed for § 3.6 / § 4.1.
7. **Audio framing**: confirm audio-disabled is the published config. § 3.5 mentions audio briefly as a future-work hook; § 4.7 omits the audio ablation. If audio is intended to be a contribution, this needs reshuffling.
8. **Related-work references**: Pouyan to provide the reference list; § 2 currently has cluster headings only.
9. **Param/data comparison footnote**: write a 2-sentence footnote on the main tables that gives the all-inclusive vs trainable distinction (Marionette ~857M trainable, ~1.92B all-inclusive incl. frozen pretrained components; X-Portrait ~3.07B all-inclusive; HunyuanPortrait ~1.99B all-inclusive). Without this, "fewer params" is an unfair comparison.
10. **FVD-VAE caveat paragraph**: § 4.8 has the hypothesis; needs one paragraph of explanation tied to the existing color-shift investigation.

---

## Decisions still to make

- **Title vs subtitle**: keep `Marionette: Pixel-Space FLAME Conditioning for Portrait Animation` plain, or add a punchier subtitle like *Representation over Architecture in Identity-Preserving Portrait Animation*?
- **Cross-id ID-cosine**: foreground in § 4.8 (current plan) or footnote it. Lean foreground — honesty buys reviewer trust, and we have a real mechanistic hypothesis.
- **Datasets in main vs supplementary**: TalkVid + HDTF both in main (current plan). Consider: ablations only on a single cell (HDTF same-id) to keep § 4.7 focused.
- **Naming the metric**: "pose-disentangled deformation-map L1" is clear but long. Alternatives: "pure-expression L1", "expression activation L1". Lean *pure-expression L1* in body, full name once at first introduction.
- **Whether to pitch the two new metrics as a methodological contribution (they appear in the contributions list) or as an evaluation choice (one paragraph in § 4.2)**. Currently pitched as a secondary contribution. Lean keep it that way — current papers under-report on head pose specifically, and the field would benefit.
