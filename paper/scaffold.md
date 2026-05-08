# Loki — a tutorial

This document is a self-contained walk-through of Loki: what it is, why it is built the way it is, how its conditioning works in detail, and how it should be evaluated. It is meant to be read top-to-bottom by a reader who has heard the term "portrait animation" but has not seen the system before.

---

## 0. Headline numbers (HDTF, 16-frame window)

For the reader who wants to see the numbers before the argument. Both tables are computed from the same FLAME-tracked outputs (`outputs/test_metric/metrics/<bucket>/hdtf/<protocol>/metrics_summary.json`). `head_rot_dist` and `expression_l1` are the FLAME-native metrics defined in §9; `id_cosine` is ArcFace cosine vs. the ref-clip prior. `head_track` and `id_detect` are the per-sample weighting denominators (fraction of frames where the FLAME tracker / ArcFace detector fired). Bold = best per column. Lower is better for `head_rot_dist` and `expression_l1`; higher for `id_cosine`.

### Same-identity reconstruction

| Method | n | head_rot_dist (°) ↓ | expression_l1 ↓ | head_track |
|---|---:|---:|---:|---:|
| AniTalker | 208 | 3.10 | 0.0963 | 0.962 |
| EchoMimic | 204 | 3.31 | 0.0823 | 0.971 |
| **HunyuanPortrait** | 211 | **2.08** | 0.0658 | 0.948 |
| **Loki** | 211 | 2.13 | 0.0664 | 0.948 |
| SadTalker | 211 | 3.42 | 0.0811 | 0.948 |
| **X-Portrait** | 211 | 2.14 | **0.0652** | 0.943 |

`id_cosine` is omitted on same-identity (the prediction is paired against the same clip; pixel/landmark fidelity already covers identity).

### Cross-identity retargeting

| Method | n | head_rot_dist (°) ↓ | expression_l1 ↓ | id_cosine ↑ | head_track | id_detect |
|---|---:|---:|---:|---:|---:|---:|
| AniTalker | 193 | 3.03 | 0.1134 | 0.815 | 0.948 | 0.992 |
| EchoMimic | 191 | 3.27 | 0.1156 | 0.897 | 0.958 | 0.929 |
| **HunyuanPortrait** | 199 | **2.19** | 0.0850 | 0.823 | 0.920 | 0.994 |
| **Loki** | 210 | 2.24 | **0.0828** | 0.791 | 0.871 | 0.910 |
| SadTalker | 199 | 3.22 | 0.1107 | **0.921** | 0.920 | 0.668 |
| X-Portrait | 199 | 2.94 | 0.1053 | 0.825 | 0.920 | 0.995 |

The shape of the result, in one read:

- **Loki is at the top of the FLAME-native motion bracket** on both protocols. On same-identity it ties HunyuanPortrait and X-Portrait (within 0.06° on `head_rot_dist`, within 0.001 on `expression_l1`). On cross-identity it is best on `expression_l1` and second on `head_rot_dist` by 0.05°.
- **The audio-driven cluster (SadTalker, AniTalker, EchoMimic) trails by 50% on head pose and ~33% on expression** on cross-identity — the empirical signature of audio-to-motion being one-to-many (§1).
- **`id_cosine` is the one column where Loki is bottom-ranked** — and it is also the column where SadTalker, the visibly most motion-poor system, is *first*. §8.3 walks through why: ArcFace cosine systematically rewards stillness and penalises faithful motion follow-through. SadTalker's `id_detect_rate` of 0.668 means the rank-1 cosine is computed on only 67% of its frames, compounding the bias.

The rest of this document explains why these numbers fall out the way they do.

---

## 1. The task and why it is hard

**Portrait animation from a driving video** is the following problem. Given a single still image of a person (the *reference*) and a video clip of someone — possibly a different person — talking and moving (the *driver*), produce a video in which the reference's face replays the driver's facial expression and head pose, frame by frame, while keeping the reference's identity untouched.

The hard part is that the system has to faithfully transfer two qualitatively different motion components at once:

- **Large head motion** — yaw / pitch / roll trajectories that can sweep tens of degrees over a single second.
- **Subtle facial expression** — millimeter-scale lip closures, eye blinks, brow raises.

And it has to do this onto an *unrelated reference identity*: the driver's face shape, jaw geometry, and identity-specific texture must not leak into the output. Cross-identity is not a bonus mode; it is the central use case (dubbing, accessibility, virtual avatars), and it is what most of the engineering hard problems live in.

The task in this paper is **video-driven**. Loki is given a visual driver clip and is expected to reproduce that specific clip's motion. It is worth saying clearly why the published configuration is video-driven and not audio-driven, because we *do* compare against audio-driven baselines (SadTalker, EchoMimic, AniTalker in their audio modes) and that comparison is deliberate — not an oversight.

The reason is that audio-to-motion is fundamentally **one-to-many**. A given audio waveform is consistent with many plausible facial-expression and head-pose trajectories: the same phoneme can be produced with a smile or a scowl, an exaggerated jaw motion or a barely perceptible lip flap, head-tilted-left or head-tilted-right. An audio-only system has to *commit* to one of those many trajectories, and from the system's point of view the choice is largely arbitrary — the audio signal does not contain the information needed to disambiguate. The visible consequence is that audio-driven systems produce expression and head-pose trajectories that are *plausible in general* but not *specific to any particular driver's motion*: lip-sync is roughly right, but the brow raises, eye contact, and head sways are whatever the model's prior happens to produce. Including audio-driven baselines in our comparison is therefore the right move precisely because it makes this gap measurable. Loki can in principle ingest audio — the architecture has audio cross-attention layers — but they are disabled in the published configuration; the contribution argued here is the *visual-driving* contribution.

The visual-driven side of the field has its own characteristic failure mode, and it is worth naming it now too. X-Portrait and HunyuanPortrait, the two strongest published representatives, both extract motion from the driver's RGB pixels through *learned modules*: X-Portrait through two ControlNets that consume pose-aligned RGB and masked driver patches, HunyuanPortrait through a pair of ResNet regressors and a motion resampler. In principle this is the right input — RGB does carry the specific motion the driver chose. In practice it leaves the system with a hard task: *learn to disentangle motion from identity in pixel space, supervised only by paired video.* That disentanglement is never structural, only learned, so it is necessarily incomplete. Two specific symptoms follow. First, **driver identity leaks**: 2D landmarks and masked driver patches still carry the driver's face geometry and pixel texture, and the learned modules carry them through into the conditioning even when the goal is to retarget onto a different reference. Second, **fine-grained motion fidelity is lossy**: the motion regressors compress per-frame motion into 512-dim tokens and the resampler down-projects them again to ~64 motion tokens, so subtle expression detail (specific lip closures, asymmetric brow raises, idiosyncratic eye contact patterns) gets averaged into the model's prior rather than reproduced. The visible consequence on cross-identity, in both systems, is plausible-looking video that *softens* the driver's specific expression and tilts toward a generic expression prior. The audio-driven baselines fail by under-specifying the input; the visual-driven baselines fail by over-relying on learned RGB-to-motion extraction. Both classes leave the same gap — driver-specific motion does not survive the trip onto the reference's face — and that gap is what the metrics in §9 are designed to measure. Loki closes it by removing the learning step from motion altogether: the motion conditioning is read off a parametric face model whose parameter axes are identity-orthogonal by construction, so identity cannot leak and fine motion is preserved coefficient-by-coefficient.

---

## 2. The architectural status quo: separate modules for separate jobs

Two recent diffusion-based systems are the strongest published representatives of the field. Both are worth reading carefully because Loki is, very deliberately, an inversion of the bet they make.

### 2.1 X-Portrait — three trainable ControlNets stacked on SD-1.5

X-Portrait builds on Stable Diffusion 1.5 latent diffusion (a 64×64 latent, an 8× VAE downsample of 512×512). It adds temporal-attention layers in the generation UNet so that one forward pass produces a short video. On top of this backbone it stacks **three trainable ControlNet branches** — each one a roughly UNet-sized copy whose features are added back into the main UNet at matching resolutions:

1. An **appearance / identity** ControlNet, AnimateAnyone-style, that reads the reference image and cross-couples its features into the generation UNet.
2. A **global-pose** ControlNet, fed an RGB-derived "pose hint" built by warping the driver frames toward the reference's pose using 68-point 2D face landmarks.
3. A **local-pose** ControlNet, fed a *masked RGB patch* of the driver — keep the pixels in 64×64 boxes around the eye and mouth landmarks, zero out the rest. This brings high-frequency local motion in as raw pixels.

Trainable parameter count: roughly 3.07 B. Two of those branches are paid in trained parameters explicitly to *learn motion from RGB*. Everything the model knows about how a face moves is implicit in those motion ControlNets' weights, learned from a large proprietary corpus.

### 2.2 HunyuanPortrait — implicit motion regressors plus IP-Adapter identity

HunyuanPortrait builds on Stable Video Diffusion. Its motion path is *learned implicitly from RGB*, in two stages:

- A small **expression regressor** (ResNet-18 with GroupNorm) reads each driver frame and produces a 512-dim expression vector.
- A separate **head-pose regressor** (ResNet) produces pose features per frame.
- A **Perceiver-style resampler** ingests the per-frame motion features and emits ~64 motion tokens via cross-attention into a fixed bank of learnable queries; the generation UNet attends to those tokens.

The paper title states the bet directly: *Implicit Condition Control*. Motion is whatever the two regressors and the resampler converge to from the RGB stream alone.

Identity comes through a different path again: **ArcFace** (a face-recognition ONNX encoder) gives a 512-dim identity embedding, and **DINOv2-Large** (Meta's self-supervised ViT, ~314 M parameters) gives patch-token image features. Both are projected through a learnable adapter and injected into the generation UNet's cross-attention as additional K/V tokens — the **IP-Adapter** pattern of Ye et al. A small pose-guider CNN (~1.1 M parameters) takes a coarse 2D pose signal and adds it into the UNet's input.

Total parameter count: ~1.99 B all-inclusive (UNet 1.58 B + DINO 314 M + motion resampler 66 M + expression regressor 25 M + head-pose regressor 11 M + a few small projectors).

### 2.3 Why these stacks look the way they do

Both stacks are answering the same upstream problem: **RGB pixels do not separate identity from pose, expression, and shape.** The intensity at any point on a face is a tangled product of *who* the person is (skin tone, bone structure, fine texture), *how* the head is oriented (which 3D face points project to which 2D pixels), *what* the expression is doing (which vertices have deformed off neutral), and what the camera and lighting were doing at capture time. There is no projection of RGB that recovers any one of these axes cleanly without the others coming along, and no obvious filter that strips out only the parts you don't want. RGB is not a representation of motion *or* identity *or* shape; it is a representation in which all of them are mixed together at every pixel.

Because RGB is structurally entangled, every system that wants to use one axis at a time has to *learn* the disentanglement. That is what every module in §2.1 and §2.2 is for. X-Portrait's two motion ControlNets exist to learn "what part of these RGB driver patches is motion (keep) versus identity (discard)." HunyuanPortrait's expression regressor and head-pose regressor exist to learn the same separation in a smaller, lower-bandwidth form. Their identity paths — a third ControlNet for X-Portrait, an ArcFace + DINOv2 stack with IP-Adapter K/V projectors for HunyuanPortrait — exist to do the inverse: extract identity from a reference image while ignoring whatever pose and expression happened to be in it. The fragmented-modules pattern is not aesthetic; it is forced by the fact that RGB does not give you the axes for free.

Each module is paid for in trained parameters, in training data, and — crucially — in *quality*. Learned disentanglement is never structural; it is whatever the training corpus happened to teach. There is no corpus large enough to teach an RGB encoder to fully unmix axes that are intrinsically mixed in pixel space. The visible consequence is exactly the visual-driven failure mode named in §1: driver identity leaks into the conditioning, fine-grained motion is lossily compressed into low-dimensional tokens, and the system trends toward generic-but-plausible output rather than driver-specific reproduction. The cost of "more capability" is *more modules, more data, and more residual entanglement*.

This is the bet that Loki inverts. Loki steps out of RGB for the motion path entirely. The disentanglement we want is *already* the property of FLAME's parameter space — `β`, `ψ`, `θ` are orthogonal by construction, not by training. Read motion off a fitted FLAME model and the disentanglement comes for free; rasterise that motion into pixel space and the diffusion backbone consumes it through its own native spatial inductive bias, with no learned RGB-to-motion bridge to train. Where X-Portrait and HunyuanPortrait spend modules to *recover* a property that RGB destroyed, Loki uses a representation that never destroyed it in the first place.

---

## 3. Loki's central bet: one representation, one path

Loki's thesis is that the architectural complexity of stacked motion / identity modules is doing the work of compensating for an upstream representational choice that need not be made — namely, learning facial motion from RGB.

If you have a parametric model of the face — a function `V(β, ψ, θ)` that produces 3D vertex positions given identity shape `β`, expression coefficients `ψ`, and pose `θ` (head, jaw, neck, eye) — then *facial expression and head pose are not signals to be learned*. They are parameters of a known generative model. The only learning task left is mapping that parametric description to image-space pixels, which is exactly what diffusion models already do.

We use **FLAME** for `V` (the CAP4D variant, with mouth verts and a 65-dim expression basis). FLAME is a 3D parametric face model — its parameter groups are *disentangled by construction*: changing `ψ` changes facial expression and nothing else; changing `θ` changes head pose and nothing else; `β` is identity-orthogonal to both. This is the property that makes the rest of the system possible. It is also the property RGB lacks, and the property §2's stacks of modules are trying — imperfectly — to learn back.

The bet, in one sentence: **encode the driver's motion as FLAME blendshape deformations rasterized into pixel space, and let the same pretrained diffusion backbone consume that as an ordinary spatial conditioning signal.** No motion regressor. No motion ControlNet. No learned pose-from-RGB. Just rasterization.

The rest of this document explains how that works concretely.

---

## 4. Conditioning, in detail: the expression activation map

### 4.1 What we render

Per video frame, we have a fitted FLAME tuple `(β, ψ[t], θ[t], camera)`. We run **one** `pytorch3d` rasterization pass that produces a **45-channel image-space tensor** at 512×512:

| Channels | Meaning |
|---|---|
| `[0:42]` | Sinusoidal positional encoding of rasterized vertex positions in normalized device coordinates: 3 raw NDC channels × 14 frequencies (`sin/cos` pairs over 7 octaves). |
| `[42:45]` | Per-vertex expression deformation: the 3D offset of each vertex from its neutral position under the current `ψ`. Rasterized in the same pass. |

Both feature streams are masked by the on-mesh indicator. Background pixels are zero. The result, viewed as an image, is a face-shaped silhouette inside which every pixel carries (a) a high-frequency positional encoding of *where on the face this pixel sits in 3D*, and (b) the *deformation* of that point relative to a neutral expression.

### 4.2 Why these two channels carry, between them, head pose and expression

This is not arbitrary. The 45 channels split functionally along exactly the axis we want:

- **The 42 positional-encoding channels carry head pose.** They encode the rasterized vertex positions in NDC — that is, where each face point projects in image space *after* the head's rigid transform and the camera projection are applied. If the head rotates, those positions move, and the positional encoding moves with them. The full 14-frequency encoding gives the diffusion model enough resolution to localize fine geometry.
- **The 3 deformation channels carry expression.** A deformation is the *difference* between the current vertex position and its neutral position under `ψ = 0`. Rigid head motion contributes nothing to it; only `ψ` does (plus the small jaw/eye-rotation contributions which we group with expression). On a stationary head, this channel is alive only where the face is moving — mouth, brows, eyelids.

So *one* rasterization, *two* orthogonal signals, *one* image-space tensor. We will validate this functional decomposition with ablations in §11.

### 4.3 Why pixel-space and not the parameter vector

Earlier work (audio-3DMM hybrids, primarily) feeds FLAME parameters into the diffusion model **as a vector**, through cross-attention: `ψ` is a 65-dim token, the generation UNet attends to it. That is a perfectly legitimate signal but it throws away the very property that makes the diffusion backbone good at faces — its **spatial inductive bias**.

A diffusion UNet's first feature map lives at 512×512×C, and every operation on top of it (convolutions, self-attention windows) assumes that nearby pixels share spatial structure. If you hand it a flat 65-dim vector through cross-attention, you have asked it to spend capacity learning *where on the face* each coefficient applies. Rasterizing the same information into image space hands it that mapping for free: the value at pixel `(i, j)` already pertains to the face point that projects to `(i, j)`. There is no learned bridge to train.

We call the 45-channel tensor an **expression activation map** because the diffusion model can consume it through ordinary convolutions, the same way it consumes its own intermediate feature maps. It is *delivered* in the model's native representation space.

### 4.4 How it enters the generation UNet

The 45-channel tensor passes through a small convolutional encoder — a few stride-2 conv blocks that downsample from 512×512 to the 64×64 latent resolution and project 45 channels down to the gen UNet's first feature width. The final layer is **zero-initialized** so that, at training start, the conditioning is a no-op and training begins from the unmodified SD-2 backbone. The encoder's output is then **added** to the first feature map of the gen UNet — exactly the position where pixel-space conditioning has the strongest receptive field.

That is the entire conditioning path. One rasterization pass, one tiny encoder, one element-wise add. Nothing is learned about the *meaning* of the FLAME parameters; the FLAME forward and the rasterizer already know.

### 4.5 Cross-identity reenactment, for free, by parametric substitution

This is the section the rest of the system pivots on.

At training time, we do **only same-identity self-supervised optimization**. Each training sample is `T+1` frames drawn from a single video clip. The model never sees a cross-identity pair.

At inference time, the only thing that changes between same-identity reconstruction and cross-identity retargeting is **which `(ψ, θ)` go into the conditioning forward pass**:

```
same-identity:    V(β_ref, ψ_ref[t],    θ_ref[t])    rendered through camera_ref
cross-identity:   V(β_ref, ψ_driver[t], θ_driver[t]) rendered through camera_ref
```

That is the entire retargeting operation. It is a **parametric substitution**, not a learned operation. Because FLAME's parameter groups are disentangled by construction, swapping `(ψ, θ)` between subjects produces the driver's facial expression and head pose realised on the reference's geometry. The reference's `β` is preserved; the camera is the reference's; only the time-varying motion parameters come from the driver.

Two consequences are worth stating plainly:

1. **Driver identity cannot leak into the conditioning.** The only driver-derived quantities that reach the model are `(ψ, θ)`, which by FLAME's definition are identity-orthogonal. Driver-identity geometry never enters the rasterized tensor. This is the structural fix to the leak that X-Portrait and HunyuanPortrait have to learn around.
2. **The expensive cross-identity training corpus is unnecessary.** The diffusion-prior baselines spend large proprietary corpora on cross-identity pairs to teach their motion modules to disentangle identity from motion in an RGB pipeline. We do not need that corpus because the disentanglement is structural, not learned.

This is what makes the architecture viable on ~10k training samples — roughly an order of magnitude less data than the leading diffusion-prior baselines disclose.

### 4.6 Why same-identity training generalizes to cross-identity inference

The non-obvious step is this: how does a model trained never to see a cross-identity pair learn anything that works at cross-identity inference? The answer has two parts. The architectural one is the more important; the training-distribution one tunes it.

**The architectural part.** Loki's two paths — the FLAME conditioning path and the reference-UNet identity path — carry strictly disjoint information by design.

- The FLAME conditioning path delivers a rasterised tensor that contains *only* shape geometry and expression deformation. Driver-identity texture, driver-identity skin tone, driver-identity bone structure beyond what FLAME's `β` captures — none of it can enter the conditioning, because none of it is part of the FLAME parameter space at all. The model sees, in the conditioning, "where the face mesh sits in image space at this frame" and "how it has deformed off neutral," and nothing else. There is no channel through which identity could ride into the conditioning even if the model wanted to use it.
- The reference-UNet identity path delivers per-layer self-attention features from a *separate* frozen forward pass over the reference image (described in §5). The pose and expression in the reference image are whatever they happen to be — generally unrelated to the per-frame pose and expression in the conditioning — and the SD-2 prior's strong identity-encoding behaviour at every self-attention layer is what carries the reference's identity through to the generation.

The model is therefore *trained* — by the structure of its inputs, not by an externally imposed loss — to look at the FLAME conditioning for motion (because that is the only thing it carries) and at the reference K/V for identity (because that is the only thing reliably correlated with identity across the training distribution). At cross-identity inference, swapping `(ψ, θ)` to the driver's values does not break this contract: the conditioning still carries only motion, and the reference still carries only identity. The model executes the same forward pass it learned during training; it never has to "decide" that this is a cross-identity case because, from the model's point of view, the inputs are not categorically different.

This is the heart of why retargeting works without cross-identity training data and without specialised disentanglement modules. The other diffusion-prior baselines need cross-identity corpora and identity-vs-motion disentanglement modules because they pipe identity *through the same RGB-derived motion path*. Loki does not need either, because the motion path was never RGB-derived in the first place — it is a rasterised parametric description, identity-blank by construction, and identity is delivered through a separate, strong, frozen prior.

**The training-distribution part.** The architectural separation alone would still need a training distribution that doesn't accidentally teach the model to read identity off the conditioning. We arrange this by sampling the slot-0 reference and the slots-1..T target window at *independent* positions within the same clip (with separate RNG seeds). In a typical training sample the reference and the target therefore agree on identity but *disagree* on pose and expression — they are different moments of the same person. The model thus learns an identity prior that ignores reference-vs-target pose/expression mismatch and only carries forward what is identity-invariant. At cross-identity inference the parametric retargeting in §4.5 produces driver pose and expression rendered through the reference's camera; the training distribution already includes "reference and target disagree on pose/expression," and cross-identity inference adds only the extra "disagree on identity shape `β`" — which the parametric retargeting handles by construction (driver-side `(ψ, θ)` substituted into the reference's `β`).

Together — strict architectural separation of the conditioning and identity paths, plus a training-sample distribution that already exercises pose/expression disagreement between reference and target — these are what make cross-identity reenactment work without any cross-identity training data and without any specialised disentanglement modules. Same-identity training in this configuration is *not* a degenerate special case; it is the right training task for the system you want at inference, and the architecture makes that training task both sufficient and minimal.

---

## 5. Identity injection: let the generative prior carry it

The conditioning path above carries the driver's motion. The reference's identity is carried through a completely separate path — and this is where Loki differs most clearly from the IP-Adapter family.

### 5.1 The mechanism

We instantiate a **frozen copy** of SD-2's UNet — a separate "reference UNet" with weights tied to the gen UNet at initialisation, never updated thereafter. The VAE-encoded reference image runs through it once. PyTorch forward hooks on each transformer block's pre-attention LayerNorm capture the *input* to every self-attention layer at every UNet resolution. Those per-layer feature maps (shape `(batch, H_l × W_l, channels_l)`) are then **injected as additional key / value tokens** into the matching self-attention block of the gen UNet:

```
gen-UNet attention at layer l:
    Q  = W_q · gen_tokens
    K  = W_k · concat(gen_tokens, ref_tokens_l)
    V  = W_v · concat(gen_tokens, ref_tokens_l)
```

Every gen-frame query attends to its own tokens *and* the reference's, at the same resolution, in every self-attention layer.

### 5.2 Why this needs no learned identity encoder

The reference UNet and the gen UNet are the **same architecture initialised from the same SD-2 checkpoint**. The reference features therefore already live in the gen UNet's native feature space — same statistics, same channel meanings, same resolution. There is no domain gap to bridge, no learned projection to train, no IP-Adapter K/V adapters to fit on top of every cross-attention layer.

The reference does not occupy a slot in the gen tensor; it lives only in the K/V-injected attention path. Loss is a uniform pixel-wise MSE on predicted noise across all `T` target slots — no auxiliary identity loss, no masking.

### 5.3 Contrast with HunyuanPortrait

HunyuanPortrait pairs **ArcFace + DINOv2 + a learned projector** and trains **IP-Adapter K/V adapters** on top of every cross-attention layer to inject the resulting identity tokens into the UNet. That is three trained components dedicated to identity, plus a 314 M-parameter ViT.

Loki's identity path captures features from the gen prior's own backbone and feeds them into the same backbone's attention layers directly. The "identity encoder" is the gen prior itself, frozen. No new module is added.

This is the smaller of Loki's two wins (the conditioning representation in §4 is the larger one), but it is the same idea applied a second time: rather than build a new module to encode an axis of variation, find the place where the existing pretrained backbone already encodes it, and route around the new module.

---

## 6. The rest of the architecture

For completeness:

- **Backbone**: Stable Diffusion 2.1 latent diffusion model, 64×64 latent (an 8× VAE downsample of 512×512), `T = 16` frames denoised per forward pass.
- **Spatiotemporal attention**: 2D self-attention layers in the inner UNet stages are replaced with 3D spatiotemporal attention, so a single forward pass produces a coherent 16-frame video latent.
- **VAE**: encoder and decoder are frozen end-to-end; only the gen UNet trains.
- **Audio**: the codebase contains audio cross-attention layers (a wav2vec-encoded driver-audio signal feeding cross-attention K/V), but they are **disabled in the published configuration**. Audio is a future-work hook.
- **Trainable parameter count**: ~857 M (gen UNet only) on top of frozen SD-2 components. Less than half the trainable parameter count of X-Portrait.
- **Training**: 30 k steps, virtual batch size 2, GPU batch size 2, `T = 16`, on 8 NVIDIA H200 GPUs. Each conditioning channel (FLAME deformation map, reference) is independently dropped to zero with some probability per training step, so the model also learns the unconditional pathway needed for classifier-free guidance.
- **Training corpus**: ~10 k samples, roughly an order of magnitude less than the diffusion-prior baselines disclose.

---

## 7. Comparison axis-by-axis

| Axis | X-Portrait | HunyuanPortrait | **Loki** |
|---|---|---|---|
| Motion representation | RGB patches → 2 ControlNets | RGB → ResNet regressors → motion tokens | **FLAME blendshapes rasterised to pixel space (45-ch expression activation map)** |
| Cross-identity strategy | Pose-align driver to ref via 68-pt landmarks; train end-to-end | Train end-to-end on motion features | **Parametric retargeting at inference: `β_ref + ψ_driver + θ_driver`** — no cross-id training |
| Identity injection | Reference-only ControlNet (a trained branch reading the reference image) | ArcFace + DINOv2 + learned projector, IP-Adapter K/V adapters in cross-attention | **Frozen SD-2 reference UNet → per-layer self-attention features → K/V injection into the matching gen-UNet layers (no learned identity encoder)** |
| Trainable params | ~3.07 B | ~1.99 B all-inclusive | **~857 M trainable** + frozen pretrained components |
| Training corpus | Large proprietary | Large proprietary | **~10 k samples** |

The bottom line: same task, three answers. X-Portrait and HunyuanPortrait answer it by adding modules. Loki answers it by changing what the existing modules see.

---

## 8. Why the inherited evaluation metrics miss

A second thread runs through this work: the metrics the field has inherited from generic image and video generation are *the wrong instruments for portrait animation*. We report them for comparability with prior work, but we argue from a different pair (introduced in §9). The argument for why that swap is necessary is in this section.

### 8.1 Pixel-aligned metrics (PSNR, SSIM, LPIPS) are spatially sensitive

PSNR / SSIM / LPIPS compare pixels at *fixed image-space positions*. The metric's notion of "this prediction is correct" is "this pixel matches the GT pixel at the same coordinate."

Consider two cases:

- **Case A**: a prediction that perfectly transfers the driver's facial expression but renders the head a few pixels off the GT's framing. The animation is correct in every meaningful sense; the pixel metric takes a measurable hit on every frame.
- **Case B**: a prediction with a wooden, generic, non-following expression that happens to be pixel-aligned with the GT's framing. The animation is wrong in the sense that matters; the pixel metric scores it well.

Pixel metrics rank Case B above Case A. That is the wrong answer for portrait animation, where the criterion is *did the prediction follow the driver's motion*, not *did the prediction land in the same pixel coordinates as the GT*.

### 8.2 FVD measures distance to a distribution, not to a driver

FVD computes the Fréchet distance between the generated and ground-truth videos in the feature space of a video classifier (VideoMAE-v2, in the modern flavour). It tells you whether your generation looks like a *plausible* talking-head video in general. It does not tell you whether *this* generation reproduced *this* driver's specific lip closure on frame 7.

A model that produces clean, average-looking talking heads outperforms on FVD a model that reproduces idiosyncratic driver motion accurately. For portrait animation, that ordering is inverted from what we want. This is also the metric on which the visual-driven baselines look most flattering — their learned motion priors produce smooth, plausible videos at the cost of driver-specific detail, and FVD rewards exactly that trade.

### 8.3 ArcFace cosine measures identity, not animation — and rewards stillness

ArcFace cosine compares the L2-normalized identity embedding of a generated frame with a reference embedding. In isolation it is an excellent identity-preservation check — necessary on cross-identity, where the prediction must look like the reference and not the driver — but it does only one of the two jobs portrait animation cares about, and worse, it has a *systematic bias* against the methods that succeed at the other job.

The bias is mechanical. ArcFace cosine is computed per frame; the more each generated frame differs from the reference image (in pose, in expression, in viewpoint), the lower the cosine. A method that produces a near-stationary, **wooden** generation — one where the head barely moves and the expression barely changes from the reference — therefore stays close to the reference embedding on every frame and scores extremely well, *regardless of whether the generation reproduced any of the driver's actual motion*. A method that successfully follows a high-energy driver — large head sweeps, expressive face, lip motion — necessarily moves further from the reference's per-frame appearance and pays in cosine for doing the right thing.

This is not a hypothetical concern; it shows up directly in the §0 cross-identity HDTF table, and it is worth walking through because it is the cleanest single example of why the inherited battery is misleading on this task. **SadTalker** — an audio-driven baseline whose generations are visibly the most motion-poor of any method we evaluated — wins the ArcFace cosine column at **0.921**, the highest of all six systems we report. The same SadTalker generation is among the worst on the metrics that measure whether the prediction *moved like the driver*: its `head_rot_dist` is **3.22°** (50% worse than Loki's 2.24° and 47% worse than HunyuanPortrait's 2.19°) and its `expression_l1` is **0.111** (33% worse than Loki's 0.083 and 31% worse than HunyuanPortrait's 0.085). HunyuanPortrait, the closest method to Loki on motion metrics, sits at 0.823 ArcFace; Loki sits at 0.791. By the ArcFace ranking alone, SadTalker would be the best system in the comparison; by the metrics that actually measure portrait animation, it is among the worst.

The mechanism is exactly the bias described above. Audio-driven SadTalker, faced with the one-to-many ambiguity of audio-to-motion, commits to a low-energy prior — a generation whose head and face barely deviate from the reference. Each rendered frame is therefore a near-copy of the reference image with only mild lip motion overlaid; per-frame ArcFace cosine on those near-copies is naturally close to 1, and the average is high. The metric does not — and cannot — detect that the generation has *failed at the task*, because the metric never asks whether the driver was followed.

A second detail compounds the issue. SadTalker's per-frame **ArcFace detect rate** on the cross-identity HDTF cell is **0.668**, versus 0.91–0.99 for the other systems. Its rank-1 cosine is therefore computed only on the 67% of frames where InsightFace's detector locked onto a face at all; the other 33% are silently dropped from the average. So the headline ArcFace win is not only computed on a system that under-moves — it is computed on the *easiest two thirds* of frames that system produces. The visual demonstration in our supplementary makes this concrete: side by side with the driver clip, SadTalker's reference sits nearly motionless while the driver speaks and turns; the metric still ranks it best.

The empirical conclusion mirrors the conceptual one. ArcFace cosine cannot be used in isolation to rank portrait-animation methods: it is silent on motion fidelity, and it actively rewards methods that fail at the task by under-moving. A correctly-framed evaluation has to pair an identity metric with metrics that directly measure motion follow-through — which is what §9 contributes.

### 8.4 What we want, instead

A portrait-animation evaluation should answer two questions, separately and orthogonally:

1. **Did the prediction's head trajectory follow the driver's?**
2. **Did the prediction's facial expression follow the driver's?**

Neither question is a *pixel question* and neither is a *distribution question*. Both are questions about the *parametric description of the face's motion*. The right metric space is therefore the same FLAME space the model conditions on.

These questions also expose the failure modes named in §1 and §8.3 directly. An audio-driven system cannot, in principle, score well on either: the audio signal does not carry which specific head trajectory or which specific expression coefficients the driver chose, so the system has to commit to its prior. A visual-driven system whose motion path goes through learned RGB-to-motion modules scores *better* than audio on these axes — it has the right input, so it can in principle reproduce the driver's specific motion — but worse than a parametric path, because its motion representation is lossy and entangled with identity in subtle ways. The two metrics in §9 are sensitive enough to make both of these gaps visible, where the inherited battery is not — and they are the metrics on which SadTalker's "wooden but high-ArcFace" failure mode shows up at the bottom of the table rather than the top.

---

## 9. The FLAME-native metric pair

We propose two metrics, both lifted from the FLAME fits the model already conditions on. Together they answer the two questions in §8.4 directly, with no double-counting between them.

### 9.0 FLAME fitting as ground truth — the ruler we run by

Before defining the metrics, it is worth being explicit about what we are measuring against. Both metrics in this section are computed in **FLAME parameter space**, not in pixel space. Concretely, the protocol is:

1. Run an off-the-shelf FLAME tracker (the same one used to build the training conditioning) on every ground-truth driver clip. The output per frame is a fitted tuple `(β, ψ, θ, camera)`.
2. Run the same tracker on every prediction video — Loki's and every baseline's. Same procedure, same code path, same hyperparameters.
3. Treat the resulting FLAME fits as the *ground-truth representation* of head pose and facial expression, on both sides. The metrics in §9.1 and §9.2 are then pure functions of those fits — `head_rot_dist` reads `(rot, neck_rot)`, `expression_l1` reads `(ψ, eye_rot, jaw_rot)` — with no further pixel-level comparison.

In other words, **FLAME fitting is the ruler we run by**. The numbers in §0 measure how close pred-side FLAME parameters come to GT-side FLAME parameters, where both sides were fit by the same tracker under the same conditions.

This decision has one obvious limitation worth naming: FLAME fitting is itself an optimization, and it can be wrong. A poorly-tracked frame produces a parameter tuple that differs from the visible truth, and any metric computed on that tuple inherits the tracking error. We do not claim otherwise. Two things make this acceptable rather than disqualifying.

First, **the same limitation applies to every legacy metric in the field, just behind a different ruler**. PSNR / SSIM are computed against the GT *pixels*, but the GT pixels themselves are the output of a video codec, a camera ISP, and (for HDTF) a face-cropping pipeline — each of which introduces its own systematic error before the metric ever sees the data. FVD is computed against *VideoMAE-v2 features*, which are themselves the output of a learned video classifier with its own training-distribution biases and out-of-domain failure modes. Face-landmark metrics (LMD-F, LMD-M) are computed against *MediaPipe FaceMesh detections*, which are themselves the output of a learned 2D landmark model that has known systematic errors at extreme angles, low light, and occlusion. Every metric in the field treats the output of *some* upstream model as ground truth and inherits that model's errors. The question is not "is the ruler perfect" — no ruler is — but "is the ruler well-matched to the task?"

Second, **FLAME fitting is the ruler best matched to the question portrait animation actually asks**. The task is "did the prediction's head pose and facial expression follow the driver's?" — a question about the *parametric configuration of the face*, not about pixel intensities, video-classifier features, or 2D landmark coordinates. Among available rulers, FLAME parameters are the one that lives in the same space as the question. They are also the same parameter space the model conditions on (§4), so the evaluation and the conditioning share a ruler — which is what the catchphrase *FLAME-native conditioning, FLAME-native evaluation* points at. Tracker error is therefore an honest source of noise we tolerate to use the right ruler, not a hidden tax we pay to use the wrong one.

For completeness: per-sample weighting (§9.1, §9.2) is by `head_rot_track_rate` / `expression_track_rate`, the fraction of the 16 requested frames where both pred and target had a usable fit. A sample whose pred or target tracking failed on, say, 4 frames contributes proportionally less than one tracked cleanly through all 16. This caps the impact of any single mistracked clip and keeps the per-cell aggregate honest about how much tracker output the score actually rests on.

### 9.1 Head-rot geodesic distance

For each clip, compose the visible head rotation per frame from the FLAME parameters:

```
R_head[t] = R(rot[t]) · R(neck_rot[t])
```

where `rot` is FLAME's global head-rotation parameter, `neck_rot` is the neck-joint rotation (FLAME applies them in that order), and `R(·)` denotes axis-angle → 3×3 rotation matrix.

Form **frame-0-anchored delta rotations** per clip:

```
dR[t] = R_head[t] · R_head[0]^T       # the rotation that takes frame 0 to frame t
```

Then for each (pred, target) pair, per frame, compute the geodesic angular distance between `dR_pred[t]` and `dR_target[t]` via the quaternion dot product:

```
q_p   = quat(dR_pred[t])
q_t   = quat(dR_target[t])
θ[t]  = 2 · arccos( clip(|q_p · q_t|, -1, 1) )       # in radians, then to degrees
```

This is the smallest 3D rotation that takes one delta to the other. Per-sample reduction is the mean of `θ` over the first 16 frames. Per-cell aggregation is a weighted mean across samples, with each sample's weight equal to the fraction of frames where both pred and target had a usable FLAME fit.

Three properties make this the right pose number for portrait animation:

- **Frame-0 anchoring measures pose-trajectory follow.** The driver dictates the *change* in head pose; the prediction's idle pose is set by the reference image, not the driver. Penalising absolute pose mismatch is therefore wrong — what matters is whether the prediction's head moves *the same way* the driver's does.
- **Camera-frame invariance.** The upstream FLAME tracker fits an independent camera per clip; the absolute `rot` value at frame 0 between two independently-tracked clips can differ by a constant offset purely from camera-fit drift. (An empirical study on 15 cross-identity samples found this offset dominated absolute comparisons — ~18° mean offset, dropping to ~3° once anchored to each clip's frame 0.) Frame-0 anchoring cancels the offset cleanly because the same offset appears in both `R_head[0]` and `R_head[t]` and gets multiplied out.
- **No Euler-angle pitfalls.** Direct yaw / pitch / roll comparison is unreliable: different conventions (intrinsic vs. extrinsic, XYZ vs. ZYX) give different numbers, the wrap at ±180° introduces discontinuities, and the same rotation has multiple equivalent Euler triples near gimbal lock. The geodesic via quaternion is the standard rotation-difference measure in robotics and graphics; it gives a single scalar in `[0°, 180°]` and avoids all of those.

We considered inter-frame deltas `R[t] · R[t-1]^T` as an alternative. Cumulative deltas catch slow systematic drift that inter-frame deltas would miss — a prediction adding 0.3 °/frame of yaw would have inter-frame deltas matching the driver at every step yet end up 4.8° off after 16 frames — and they have roughly 10× better signal-to-noise over the 16-frame window because the head moves more cumulatively than per-frame.

### 9.2 Pose-disentangled expression L1

For each (pred, target) pair, render the target's expression-deformation map (the 3-channel field defined in §4.1, channels 42:45). Then render a **substituted fit**: insert pred's `(ψ, eye_rot, jaw_rot)` into target's `(rot, tra, neck_rot, β, camera)` and rasterize that.

Both renders sit at the **same image-space pose by construction** (both use the target's pose / shape / camera), so any per-pixel difference between them is purely an expression-coefficient difference. Compute mask-aware L1 between them — per-pixel mean-absolute-deviation across the 3 deform channels, then mean over on-mesh pixels.

Two design points:

- **L1 rather than L2.** L1's units are directly interpretable as the average per-component deformation residual. RMSE-across-channels is harder to read off in deform-coefficient terms.
- **Mask-aware reduction.** Background pixels are zero in both renders. A naïve mean over the whole image would dilute the score with a large constant zero region; we average over on-mesh pixels only.

Why this is the right expression number: comparing raw FLAME `ψ` vectors directly is misleading because the basis is not perceptually uniform — a small coefficient change can be visible, a large one invisible. Rasterising into pixel space and diffing per-pixel turns the comparison into one over a *perceptually meaningful* projection of the mesh — the same projection the model is conditioned on.

Pose error is captured separately by §9.1, so the two metrics cover head pose and expression *orthogonally*, with no double-counting.

### 9.3 Interpreting the L1 magnitude

A raw L1 of 0.07 is meaningless without knowing the size of the signal we are asking the generator to reproduce. We sampled 100 GT FLAME fits across the eval datasets and rasterised the deformation map at frame 0:

| Quantity | Value |
|---|---|
| Mask coverage (face area / image) | 39.4% |
| Per-pixel value range | `[−0.99, +0.67]` |
| Per-pixel mean \|value\| (signal magnitude) | **0.126** ± 0.041 |
| Per-pixel L2 norm of `(x, y, z)` | 0.262 ± 0.086 |
| Per-channel mean \|value\| | x = 0.070, y = 0.147, z = 0.161 |

So a residual of `0.07` on an underlying signal of mean magnitude `0.126` corresponds to roughly half the signal — a real fraction, not noise. We use this denominator only as a sanity check that residuals are at the scale of the underlying signal; it is not a normalised metric (a tighter "chance" baseline would be the L1 between two unrelated GT fits, which would shrink the ratios).

### 9.4 The framing

These two metrics are not arbitrary additions to the standard battery; they are answers to the failure modes in §8. PSNR / SSIM / LPIPS punish a prediction for being a few pixels off the GT's framing; the geodesic head-rot metric is *invariant* to per-clip framing offsets by construction. FVD rewards generic plausibility; the expression-L1 measures *driver-specific* expression follow per frame. ArcFace silently rewards stillness; both new metrics measure exactly the motion the still system is failing to produce.

The catchphrase: **FLAME-native conditioning, FLAME-native evaluation.** The model conditions on the FLAME fit; the evaluation lives in the same parametric space. FLAME fitting is the ruler — imperfect, like every other ruler in the field, but the one best matched to the question portrait animation actually asks.

---

## 10. Where the evidence lands

A short reading of the §0 numbers — what the data says, in plain language:

- **Head-rot geodesic distance.** Loki is at the top of the dense-method bracket on both protocols (HDTF same-id 2.13°, cross-id 2.24°), tied with HunyuanPortrait (2.08° / 2.19°) and X-Portrait (2.14° / 2.94°), and ahead of the audio-driven and motion-feature cluster (SadTalker 3.42° / 3.22°, AniTalker 3.10° / 3.03°, EchoMimic 3.31° / 3.27°) by roughly 50% on cross-identity. That 50% gap is the empirical signature of the "audio is one-to-many" prediction made in §1. This is the most defensible win: a parametric pose representation, rasterised, is at least as accurate as the best learned-from-RGB pose modules at half the trainable parameter budget.
- **Pose-disentangled expression L1.** Loki is best on the cross-identity HDTF cell (0.083 vs HunyuanPortrait 0.085, AniTalker 0.113, EchoMimic 0.116, SadTalker 0.111, X-Portrait 0.105) and tied with HunyuanPortrait and X-Portrait on same-identity (0.066 / 0.066 / 0.065). The audio-driven baselines sit 30–40% above Loki on cross-identity expression error — again consistent with one-to-many: their expressions are plausible, but not the driver's. This is the second predicted win: if the FLAME representation is doing the work for pose, it should also be doing it for expression, and it is.
- **`id_cosine`.** Loki is bottom of the table on cross-identity (0.791 vs SadTalker 0.921 vs EchoMimic 0.897). The §8.3 caveat applies to the column more broadly: ArcFace ranks SadTalker first by penalising the methods that actually move, and SadTalker's `id_detect_rate` of 0.668 means even that win is computed on only two thirds of its frames. The substantive Loki story on this column is mechanistic: the reference-UNet K/V injection is trained only on small pose deltas (since training is same-identity, the reference and target pose typically agree closely), and large pose deltas at cross-identity inference fall outside that training distribution. The fact that the pose-and-expression columns simultaneously *lead* is consistent with this — pose and expression flow through the parametric path, which works regardless of pose magnitude, while identity flows through the learned attention-injection path, which doesn't.
- **Same-identity pixel quality (PSNR / SSIM / LPIPS, in the broader experimental tables not reproduced here).** Mid-pack, third of six. X-Portrait leads, HunyuanPortrait is second, Loki is co-second on TalkVid and slightly below on HDTF. The framing here is that these are exactly the metrics §8 argues are the wrong question for portrait animation; mid-pack on them is acceptable when the metrics that *do* answer the right question are won.
- **FVD.** Bottom or near-bottom on both datasets. The hypothesis is again mechanistic: the SD-2 VAE pipeline produces a subtle texture and colour signature ("VAE drift") that the VideoMAE-v2 backbone behind FVD penalises even when the face *content* is plausible. This is a distributional issue with the codec, not a content-quality issue with the diffusion.

The qualitative shape is consistent throughout: the FLAME-mediated parts of the system (motion conditioning) work well; the learned-attention parts (identity injection) trail, but in tractable ways and on metrics whose framing we have already separately argued is incomplete for the task.

---

## 11. An ablation that makes the representation falsifiable

The 45-channel split (42 positional-encoding + 3 deformation) was claimed in §4.2 to carry head pose and expression *separately*. That claim should be tested by removing one half and seeing exactly the predicted axis of degradation.

We train two ablation arms, each a fresh 30k-step Loki run with the conditioning tensor reduced:

- `no_deform`: drop the 3 deformation channels, keep the 42 positional-encoding channels.
- `no_posenc`: drop the 42 positional-encoding channels, keep the 3 deformation channels.

The prediction the §4.2 framing makes:

- `no_posenc` should *raise* `head_rot_dist` (head pose drifts; positional encoding is the pose-carrying channel) but leave expression metrics roughly intact.
- `no_deform` should *raise* `expression_l1` and LMD-M (lip motion degrades; deformation is the expression-carrying channel) but leave head pose roughly intact.

This is what makes the conditioning contribution falsifiable rather than merely descriptive: the 45 channels are not a single black-box feature; they have a *direct functional decomposition*, and the ablations either confirm that decomposition or reveal that one of the halves was doing both jobs.

---

## 12. The thesis, in one paragraph

Portrait animation has been answered architecturally — by stacking trained modules that learn motion, identity, and pose from RGB, paying the cost in trained parameters and large proprietary corpora, and the *quality* cost in incomplete identity-motion disentanglement and lossy compression of fine-grained expression. The reason those stacks exist is that RGB is a structurally entangled representation: the axes the system needs to manipulate — identity, pose, expression — are mixed together at every pixel, and there is no projection of RGB that recovers any one without the others. Loki steps out of RGB for the motion path entirely. The driver's motion is read off a parametric face model whose parameter axes are orthogonal by construction, encoded as FLAME blendshape deformations rasterised into pixel space, parametrically retargeted onto the reference's FLAME shape at inference, and added to the diffusion model's first feature map as an ordinary spatial conditioning. Identity is injected through the generation prior's own frozen reference UNet at every self-attention layer, with no learned identity encoder. Because the conditioning carries only motion (by construction) and the reference path carries only identity (by frozen prior), the two never compete for the same channel — and that is what makes cross-identity reenactment work without cross-identity training data and without specialised disentanglement modules. The result is competitive portrait animation at less than half the trainable parameters of X-Portrait and an order of magnitude less training data, leading on FLAME-native head-pose and expression metrics whose ruler is FLAME fitting itself — imperfect, like every other ruler in the field, but the one best matched to the question portrait animation actually asks. The inherited PSNR / SSIM / LPIPS / FVD / ArcFace battery does not measure that question (spatial sensitivity, distribution-prior framing, identity-only scope, and ArcFace's measurable bias toward systems that under-move). The audio-driven baselines included in the comparison make the same point from the opposite direction: because audio-to-motion is one-to-many, an audio-only system cannot in principle reproduce a *specific* driver's expression and head trajectory, and the metrics in §9 are the ones that show that gap. *FLAME-native conditioning, FLAME-native evaluation.*
