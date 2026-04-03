# SUMMARY.md
# CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models

## Paper Reference
- **Authors**: Felix Taubner, Ruihang Zhang, Mathieu Tuli, David B. Lindell
- **Venue**: CVPR 2025 (Oral)
- **Repo**: https://github.com/felixtaubner/cap4d

---

## High-Level Overview

CAP4D is a two-stage pipeline that creates animatable, photorealistic 4D (dynamic 3D) portrait avatars from an arbitrary number of reference images (1 to 100+). 

- **Stage 1 — Morphable Multi-View Diffusion Model (MMDM)**: A fine-tuned Stable Diffusion 2.1 model that generates hundreds of novel-view images with diverse expressions from the input reference images.
- **Stage 2 — 4D Avatar Reconstruction**: The generated + reference images are used to reconstruct an animatable avatar based on 3D Gaussian Splatting (built on top of the GaussianAvatars framework).

---

## Stage 1: Morphable Multi-View Diffusion Model (MMDM)

### Architecture
- Initialized from **Stable Diffusion 2.1**.
- Images are encoded into latent space using the **pre-trained VAE** encoder.
- The latent diffusion model processes `R` reference latent images `Z_ref` and `G` generated latent images `Z_gen` **in parallel**.
- Key architectural modifications from vanilla SD 2.1:
  - **2D attention layers are replaced with 3D attention** after each 2D residual block (i.e., attention over two spatial dims + one dimension across all input images). This enables information sharing across multiple reference and generated views.
  - **Cross-attention layers are removed** (no text conditioning is used).
  - The model is **fully fine-tuned** (all parameters).

### Conditioning Signals
For each reference and generated image, the model receives additional conditioning images concatenated to the noisy latent input `Z_ref/gen`. These conditioning maps are:

1. **3D Pose Maps (`P_ref/gen`)**: Rasterized canonical 3D coordinates of the head geometry. Obtained by:
   - Fitting a **FLAME** 3DMM model to each reference image using an off-the-shelf head tracker.
   - Recovering a template mesh `T` and per-image 3D models `M_ref` with shape, head pose, expression blendshapes, camera intrinsics/extrinsics.
   - Assigning each vertex a texture equal to its 3D position on the template mesh `T`.
   - Rasterizing this textured mesh from each camera viewpoint, then applying a learned linear projection `γ` to produce the pose map.

2. **Expression Deformation Maps (`E_ref/gen`)**: Rasterized 3D deformations of the geometry relative to the neutral expression mesh. Captures per-vertex displacement caused by expressions. Processed through a learned linear projection `η`.

3. **View Direction Maps (`V_ref/gen`)**: The direction of each camera ray in the coordinate frame of the first camera. Encodes viewpoint information.

4. **Binary Masks (`B_ref/gen`)**: Indicate whether each image slot is a reference image or a generated image.

All four conditioning signals are concatenated and appended to the latent reference images `Z_ref` to form the full network input.

### 3DMM Sampling for Generated Views
For the generated images, novel 3DMMs `M_gen` are sampled by:
- Choosing desired head poses, expressions, and camera positions.
- These define the conditioning maps `C_gen = {P_gen, E_gen, V_gen, B_gen}` for the generated views.

### Stochastic Input-Output (I/O) Conditioning
A key technical contribution enabling the model to handle an arbitrary number of reference images and generate hundreds of consistent views:

- During each denoising step `t`, the model randomly samples a subset of reference images and previously generated images to use as conditioning context.
- At each step, the set of "reference" slots and "generated" slots is reshuffled stochastically.
- This allows iterative generation: previously generated images can serve as references in subsequent generation rounds.
- Enables scaling from 1 reference image to 100+ reference images at inference, and generating hundreds of novel views iteratively.

### Training
- The MMDM learns the joint probability: `P(I_gen | I_ref, C_ref, C_gen)`.
- Trained on multi-view video datasets with tracked FLAME meshes.
- Standard diffusion denoising objective (predict noise).

---

## Stage 2: 4D Avatar Reconstruction (3D Gaussian Splatting)

### Representation
- Built on the **GaussianAvatars** framework.
- The avatar is represented as a set of **3D Gaussians rigged to a FLAME mesh**.
- Each Gaussian is attached to a triangle of the FLAME mesh and defined in local triangle coordinates, so when the mesh deforms (via expression blendshapes), the Gaussians move accordingly.

### Expression-Dependent Deformation U-Net
- A lightweight **U-Net** predicts per-Gaussian expression-dependent deformations.
- Input: the current FLAME expression parameters.
- Output: additional offsets to Gaussian positions, enabling fine-grained effects like wrinkles that depend on expression.

### Training the 4D Avatar
- The generated images from Stage 1 (hundreds of views with diverse expressions) plus the original reference images are used as training data.
- Each training image has an associated tracked FLAME 3DMM (either from the original tracker for reference images, or the sampled 3DMM used during generation).
- The Gaussians are optimized using:
  - **L1 photometric loss** between rendered and target images.
  - **LPIPS perceptual loss** for perceptual quality.
- Standard 3DGS densification and pruning strategies are applied.

### Real-Time Rendering
- Once trained, the avatar can be animated in real time by:
  - Providing new FLAME expression parameters (e.g., from a driving video or speech-driven animation).
  - The FLAME mesh deforms, Gaussians follow, and the U-Net applies expression-dependent corrections.
  - Standard Gaussian splatting rasterization produces the final image.

---

## Key Data Flow Summary

```
Input Reference Images
    │
    ▼
3DMM Tracker (FLAME fitting)
    │
    ├──► M_ref (tracked 3DMMs per reference image)
    │       │
    │       ▼
    │    Rasterize conditioning maps: P_ref, E_ref, V_ref, B_ref
    │
    ├──► 3DMM Sampler → M_gen (novel poses/expressions/views)
    │       │
    │       ▼
    │    Rasterize conditioning maps: P_gen, E_gen, V_gen, B_gen
    │
    ▼
VAE Encoder → Z_ref (latent reference images)
    │
    ▼
MMDM (Stable Diffusion 2.1 + 3D attention, no cross-attn)
    │   - Input: Z_ref + conditioning maps (concatenated)
    │   - Stochastic I/O conditioning across denoising steps
    │   - Iterative generation of hundreds of views
    │
    ▼
VAE Decoder → I_gen (generated multi-view images with diverse expressions)
    │
    ▼
Stage 2: 4D Avatar Reconstruction
    │   - 3D Gaussians rigged to FLAME mesh
    │   - Expression-dependent deformation U-Net
    │   - Trained on I_ref + I_gen with tracked/sampled 3DMMs
    │   - Losses: L1 + LPIPS
    │
    ▼
Real-Time Animatable 4D Avatar (driven by FLAME expression params)
```

---

## Expected Codebase Structure (based on methodology)

- **3DMM tracking/fitting**: FLAME model fitting, camera parameter estimation, blendshape extraction (likely uses FlowFace or Pixel3DMM trackers).
- **Conditioning map generation**: Rasterization of FLAME meshes to produce pose maps, expression deformation maps, view direction maps, binary masks. Includes learned linear projections `γ` and `η`.
- **MMDM model**: Modified Stable Diffusion 2.1 UNet with 3D attention replacing 2D attention. VAE encoder/decoder. Stochastic I/O sampling logic for training and inference.
- **3DMM sampler**: Logic for sampling novel head poses, expressions, and camera positions for generation targets.
- **4D avatar reconstruction**: GaussianAvatars-based 3DGS representation. FLAME mesh rigging. Expression-dependent deformation U-Net. Training loop with L1 + LPIPS losses.
- **Inference/animation**: Loading trained avatar, driving with new FLAME parameters, real-time Gaussian splatting rendering.

---

## Key Dependencies / External Models
- **Stable Diffusion 2.1** (base diffusion model)
- **FLAME** 3D Morphable Model (head geometry, expression blendshapes)
- **FlowFace / Pixel3DMM** (3DMM head tracker)
- **GaussianAvatars** (3DGS avatar representation)
- **PyTorch3D** (differentiable rasterization)
- **NeuS2** (not directly used but referenced for mesh reconstruction in related work)
