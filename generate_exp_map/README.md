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

### 1. Create the conda environment

```bash
conda create -n expmapgen python=3.9 -y
conda activate expmapgen
```

### 2. Install core packages

```bash
# PyTorch 2.0.1 + CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch3D 0.7.4 (prebuilt wheel for Python 3.9, CUDA 11.8, PyTorch 2.0.1)
pip install --no-index --no-cache-dir pytorch3d==0.7.4 \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

# nvdiffrast (differentiable rasterizer, needed by pixel3dmm tracking)
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

### 3. Install pixel3dmm and dependencies

```bash
# Clone pixel3dmm
git clone https://github.com/SimonGiebenhain/pixel3dmm.git /path/to/pixel3dmm

# Apply our patches (bug fixes + compatibility)
cd /path/to/pixel3dmm
git apply <repo_root>/generate_exp_map/patches/pixel3dmm_fixes.patch

# Install pixel3dmm in editable mode
pip install -e .

# Copy our optimized scripts into pixel3dmm
cp <repo_root>/generate_exp_map/src/pixel3dmm_preprocessing.py scripts/
cp <repo_root>/generate_exp_map/src/pixel3dmm_inference.py scripts/
cp <repo_root>/generate_exp_map/src/pixel3dmm_segmentation.py scripts/
```

### 4. Install remaining dependencies

```bash
# Face detection and segmentation
pip install insightface==0.7.3     # face detection (MICA preprocessing)
pip install face-alignment          # face landmark detection (L2CS gaze tracking)
pip install facer                   # face semantic segmentation (pixel3dmm preprocessing)

# 3D and video
pip install trimesh                 # mesh I/O (PLY loading)
pip install decord                  # fast video decoding
pip install omegaconf               # config management
pip install tyro                    # CLI argument parsing

# General
pip install numpy==1.23.0 scipy opencv-python tqdm
```

### 5. Download pretrained weights

```bash
cd /path/to/pixel3dmm

# FLAME 2023 model (requires registration at https://flame.is.tue.mpg.de/)
bash download_flame2023.sh

# pixel3dmm pretrained checkpoints (normals + UV prediction)
# Follow pixel3dmm README for download instructions
# Expected at: /path/to/pixel3dmm/pretrained_weights/{normals.ckpt, uv.ckpt}
```

Additional weights for Phase 2 (already in `data/weights/` if repo is set up):
- `data/weights/l2cs/L2CSNet_gaze360.pkl` — gaze direction estimation ([L2CS-Net](https://github.com/Ahmednull/L2CS-Net))
- `data/weights/rvm/rvm_mobilenetv3.pth` — background matting ([RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting))

### 6. Set environment variables

```bash
export PIXEL3DMM_CODE_BASE=/path/to/pixel3dmm
export PIXEL3DMM_PREPROCESSED_DATA=data/flame_tracking/preprocessing
export PIXEL3DMM_TRACKING_OUTPUT=data/flame_tracking/tracking
```

Add these to your `~/.bashrc` or activate script to persist across sessions.

### Version summary

| Package | Version | Purpose |
|---|---|---|
| Python | 3.9 | pixel3dmm compatibility |
| PyTorch | 2.0.1+cu118 | GPU compute |
| PyTorch3D | 0.7.4 | Differentiable mesh rendering |
| nvdiffrast | latest | Differentiable rasterization (tracking) |
| insightface | 0.7.3 | Face detection (MICA) |
| face-alignment | latest | Face landmarks (gaze tracking) |
| facer | latest | Face semantic segmentation |
| trimesh | latest | Mesh I/O |
| decord | 0.6.0 | Video decoding |
| pixel3dmm | patched | 3D face tracking pipeline |

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
