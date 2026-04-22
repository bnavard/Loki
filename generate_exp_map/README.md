# FLAME Expression Map Generation

Generates `fit.npz` (FLAME 3DMM tracking parameters) from talking-head videos. The pipeline has two phases using [pixel3dmm](https://github.com/SimonGiebenhain/pixel3dmm) for 3D face tracking.

## Pipeline Overview

```
Phase 1: pixel3dmm Tracking
  video.mp4 → preprocessing → normals prediction → UV prediction → FLAME tracking
  Output: per-frame .frame checkpoints + .ply meshes

Phase 2: FlowFace Conversion
  checkpoints + meshes → FLAME re-fitting → gaze tracking → background matting
  Output: fit.npz + images/ + bg/ (FlowFace format)
```

The final `fit.npz` at `data/flowface/{video_name}/fit.npz` contains per-frame FLAME expression, pose, shape, and eye rotation parameters used by all downstream pipelines.

## Setup

One script installs everything:

```bash
cd <repo_root>
bash generate_exp_map/scripts/setup.sh
```

The script is **idempotent** — completed steps are detected and skipped, so
it's safe to re-run after a failure.

### Before you run it

| Requirement | Why | Notes |
|---|---|---|
| `conda` on `PATH` | creates the `expmapgen` env (Python 3.9) | Miniconda or Anaconda both fine. |
| **CUDA 11.8 toolkit** with `nvcc` + headers | nvdiffrast compiles from source against CUDA 11.8 | The script searches `$CUDA_HOME_118`, `/usr/local/cuda-11.8`, `/opt/cuda-11.8`, and `$CONDA_PREFIX`. If none is valid it will prompt you interactively. A toolkit without `include/cuda_runtime.h` will be rejected — you need the full dev install, not just the driver. |
| FLAME website account | FLAME 2020 + 2023 model downloads are credentialed | Register at <https://flame.is.tue.mpg.de/>. The script prompts for username (email) and password in step 9. |
| `data/assets/flame/flame2023.pkl` and `flame2023_no_jaw.pkl` (if you want the chumpy fix) | step 9.5 rewrites these under NumPy 1.23 so downstream loaders don't hit `numpy._core` paths or chumpy breakage | Step skips any file it can't find and keeps going. A `.bak` is written the first time. |
| `data/weights/l2cs/L2CSNet_gaze360.pkl` and `data/weights/rvm/rvm_mobilenetv3.pth` | Phase 2 gaze + background matting | Step 10 only warns if missing; setup still completes. Phase 2 will fail later without them. |
| Outbound network | clones GitHub repos, pulls PyTorch / PyTorch3D / nvdiffrast / FLAME / gdown-hosted weights | Corporate proxies blocking `dl.fbaipublicfiles.com`, `drive.google.com`, or `download.is.tue.mpg.de` will break the run. |

### What the script does

1. `conda create -n expmapgen python=3.9`.
2. PyTorch 2.0.1 + CUDA 11.8, PyTorch3D 0.7.4 (prebuilt wheel).
3. **nvdiffrast** — installs GCC 11 via conda-forge, builds against CUDA 11.8
   with `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"` (the `+PTX` gives
   forward compatibility on Hopper/H100+).
4. Clones + patches pixel3dmm (`patches/pixel3dmm_fixes.patch`) and installs
   it editable.
5. Clones + patches facer, MICA, PIPNet; builds PIPNet's NMS extension;
   downloads PIPNet + pixel3dmm checkpoints via `gdown`.
6. Copies our optimized `pixel3dmm_{preprocessing,inference,segmentation}.py`
   into `pixel3dmm/scripts/` and patches FaceBoxesV2 to use relative imports.
7. Installs remaining deps (insightface, onnxruntime-gpu 1.16.3, face-alignment,
   trimesh, decord, omegaconf, tyro, pytorch-lightning 2.0.0, …) and pins
   `numpy==1.23` (chumpy + several MICA modules break on NumPy 2.x).
8. Prompts for FLAME credentials, downloads FLAME 2023 + 2020 into MICA's
   `data/` dir.
9. Rewrites `data/assets/flame/flame2023*.pkl` with chumpy stripped and
   numpy 1.x paths (see step 9.5 in the script).
10. Creates `data/flame_tracking/{preprocessing,tracking,logs}` and
    `data/flowface/`.

### After it finishes

```bash
conda activate expmapgen
cd <repo_root>
bash generate_exp_map/scripts/run_multi_gpu.sh
```

No environment-variable exports are needed for the normal run scripts.
If you invoke pixel3dmm's own tools directly, the legacy env vars are:

```bash
export PIXEL3DMM_CODE_BASE=<repo_root>/generate_exp_map/pixel3dmm
export PIXEL3DMM_PREPROCESSED_DATA=data/flame_tracking/preprocessing
export PIXEL3DMM_TRACKING_OUTPUT=data/flame_tracking/tracking
```

### Common failure modes

- **nvdiffrast build fails with "unsupported GNU version"** — the host GCC is
  >11. The script installs conda-forge GCC 11 and points `CC`/`CXX` at it;
  confirm `${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc` exists after
  step 4.
- **nvdiffrast build can't find `cuda_runtime.h`** — the CUDA 11.8 path is
  driver-only or a stub. Point `CUDA_HOME_118` at a full toolkit and re-run.
- **`gdown` quota / "file too large to scan"** — Google Drive rate-limits
  unauthenticated downloads. Re-run the script; it skips completed files.
- **FLAME download returns a tiny HTML file** — the credentials were wrong;
  the login page was saved instead of the zip. Delete the bad `.pkl`/`.zip`
  under MICA `data/` and re-run.
- **Step 9.5 errors loading a pkl** — usually means the pickle is already
  the restored backup. Delete the `.bak` and re-run (or skip — the step is
  optional if downstream loaders succeed as-is).

### Key versions

| Package | Version | Why pinned |
|---|---|---|
| Python | 3.9 | pixel3dmm + facer + PIPNet compatibility |
| PyTorch | 2.0.1+cu118 | matches the PyTorch3D wheel below |
| PyTorch3D | 0.7.4 | prebuilt wheel for py39/cu118/pyt201 |
| nvdiffrast | HEAD | built from source against CUDA 11.8 |
| onnxruntime-gpu | 1.16.3 | last release supporting CUDA 11.8 + cuDNN 8 |
| pytorch-lightning | 2.0.0 | MICA's Lightning-based tracking expects ≤2.0 |
| numpy | 1.23 | chumpy and several MICA modules break on NumPy 2.x |
| insightface | 0.7.3 | MICA preprocessing pin |

## Usage

### Single video (single GPU)

```bash
cd <repo_root>
bash generate_exp_map/scripts/run_single_gpu.sh data/talkvid/talkvid/CLIP_ID.mp4

# With custom output directory and GPU:
bash generate_exp_map/scripts/run_single_gpu.sh data/talkvid/talkvid/CLIP_ID.mp4 data/flowface 2
```

### All videos (multi-GPU)

```bash
cd <repo_root>
bash generate_exp_map/scripts/run_multi_gpu.sh

# With custom settings:
bash generate_exp_map/scripts/run_multi_gpu.sh data/talkvid/talkvid data/flowface 8 2
#                                               data_dir           output_dir   gpus workers/gpu
```

Both scripts run Phase 1 (pixel3dmm tracking) followed by Phase 2 (FlowFace conversion) automatically. Already-completed videos are skipped on re-run.

## Structure

```
generate_exp_map/
├── scripts/                                       # User-facing entry points
│   ├── run_single_gpu.sh                          # Process one video (Phase 1 + Phase 2)
│   └── run_multi_gpu.sh                           # Process all videos in parallel
├── src/                                           # Internal modules (imported by scripts)
│   ├── flame_tracking.py                          # Phase 1: single video tracking
│   ├── flame_tracking_parallel.py                 # Phase 1: multi-GPU scheduler
│   ├── convert_to_flowface.py                     # Phase 2: tracking → fit.npz
│   ├── convert_to_flowface_parallel.py            # Phase 2: multi-GPU scheduler
│   ├── pixel3dmm_preprocessing.py                 # pixel3dmm internal (copy to pixel3dmm/scripts/)
│   ├── pixel3dmm_inference.py                     # pixel3dmm internal (copy to pixel3dmm/scripts/)
│   ├── pixel3dmm_segmentation.py                  # pixel3dmm internal (copy to pixel3dmm/scripts/)
│   ├── l2cs_eye_tracker.py                        # Gaze direction estimation
│   ├── l2cs/                                      # L2CS model package
│   └── robust_video_matting/                      # Background matting model
├── patches/
│   └── pixel3dmm_fixes.patch                      # Bug fixes for upstream pixel3dmm
└── README.md
```

## Pretrained Weights

Located in `data/weights/` (shared repo-wide):

| Weight | Source | Used by |
|---|---|---|
| `l2cs/L2CSNet_gaze360.pkl` | [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | Gaze tracking in Phase 2 |
| `rvm/rvm_mobilenetv3.pth` | [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | Background matting in Phase 2 |
| pixel3dmm checkpoints | [pixel3dmm](https://github.com/SimonGiebenhain/pixel3dmm) | Phase 1 (normals + UV prediction) |

## Output

Phase 1 output (intermediate):
```
data/flame_tracking/
├── preprocessing/{video_name}/     # cropped frames, segmentation, MICA, p3dmm predictions
└── tracking/{video_name}_nV1_.../  # per-frame .frame checkpoints + .ply meshes
```

Phase 2 output (final, used by downstream):
```
data/flowface/{video_name}/
├── fit.npz                  # FLAME parameters (shape, expr, rot, tra, eye_rot, neck_rot, camera)
├── images/cam0/             # extracted frames (or symlink to video)
├── bg/cam0/                 # background alpha masks
├── reference_images.json    # selected reference frame indices
├── cam_static.npz           # static camera trajectory
└── cam_orbit.npz            # orbit camera trajectory
```
