#!/bin/bash
# =============================================================================
# Full setup for the FLAME expression map generation pipeline.
#
# Creates the conda environment, clones and patches pixel3dmm, installs
# all dependencies, and downloads pretrained weights.
#
# Usage:
#   cd <repo_root>
#   bash generate_exp_map/scripts/setup.sh
#
# After setup:
#   conda activate expmapgen
#   bash generate_exp_map/scripts/run_multi_gpu.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
P3DMM_DIR="${REPO_ROOT}/generate_exp_map/pixel3dmm"

echo "============================================================"
echo "Setting up FLAME expression map generation pipeline"
echo "Repo root: ${REPO_ROOT}"
echo "pixel3dmm: ${P3DMM_DIR}"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Create conda environment
# =============================================================================
echo "===== Step 1: Creating conda environment 'expmapgen' ====="

if conda info --envs | grep -q "expmapgen"; then
    echo "Environment 'expmapgen' already exists. Skipping creation."
else
    conda create -n expmapgen python=3.9 -y
fi

# Activate in subshell context
eval "$(conda shell.bash hook)"
conda activate expmapgen

echo "Python: $(python --version)"
echo ""

# =============================================================================
# Step 2: Install PyTorch + CUDA
# =============================================================================
echo "===== Step 2: Installing PyTorch 2.0.1 + CUDA 11.8 ====="

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo ""

# =============================================================================
# Step 3: Install PyTorch3D
# =============================================================================
echo "===== Step 3: Installing PyTorch3D 0.7.4 ====="

# fvcore + iopath are PyTorch3D dependencies that must be installed first
pip install fvcore iopath

pip install --no-index --no-cache-dir pytorch3d==0.7.4 \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

echo ""

# =============================================================================
# Step 4: Install nvdiffrast
# =============================================================================
echo "===== Step 4: Installing nvdiffrast ====="

# nvdiffrast requires:
#   - CUDA 11.8 toolkit (nvcc must match PyTorch's CUDA version)
#   - GCC ≤ 11 (CUDA 11.8 rejects GCC > 11) — installed via conda-forge
#   - ninja build system
#   - TORCH_CUDA_ARCH_LIST with PTX for forward compat (Hopper/H200)
#
# Set CUDA_HOME_118 if CUDA 11.8 is not at the default path.

pip install setuptools wheel ninja cython

# Install GCC 11 via conda-forge (no system permissions needed)
echo "Installing GCC 11 via conda-forge..."
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y

CONDA_GCC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

if [ ! -f "${CONDA_GCC}" ]; then
    # Fallback: some conda-forge builds use a different prefix
    CONDA_GCC="${CONDA_PREFIX}/bin/x86_64-conda_cos6-linux-gnu-gcc"
    CONDA_GXX="${CONDA_PREFIX}/bin/x86_64-conda_cos6-linux-gnu-g++"
fi

echo "GCC: $(${CONDA_GCC} --version | head -1)"

CUDA_118="${CUDA_HOME_118:-/home/pouyan/cuda/cuda118}"

if [ ! -f "${CUDA_118}/bin/nvcc" ]; then
    echo "ERROR: CUDA 11.8 not found at ${CUDA_118}/bin/nvcc"
    echo "Set CUDA_HOME_118 to your CUDA 11.8 toolkit path."
    exit 1
fi

echo "CUDA 11.8: ${CUDA_118}"

# 8.9+PTX includes PTX for forward compat with SM 9.0+ (Hopper/H100/H200)
CUDA_HOME="${CUDA_118}" \
PATH="${CUDA_118}/bin:${PATH}" \
CC="${CONDA_GCC}" \
CXX="${CONDA_GXX}" \
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX" \
MAX_JOBS=1 \
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

echo ""

# =============================================================================
# Step 5: Clone and patch pixel3dmm
# =============================================================================
echo "===== Step 5: Cloning and patching pixel3dmm ====="

if [ -d "${P3DMM_DIR}" ]; then
    echo "pixel3dmm already cloned at ${P3DMM_DIR}. Skipping clone."
else
    git clone https://github.com/SimonGiebenhain/pixel3dmm.git "${P3DMM_DIR}"
fi

cd "${P3DMM_DIR}"

# Apply patches (skip if already applied)
if git apply --check "${REPO_ROOT}/generate_exp_map/patches/pixel3dmm_fixes.patch" 2>/dev/null; then
    git apply "${REPO_ROOT}/generate_exp_map/patches/pixel3dmm_fixes.patch"
    echo "Patches applied."
else
    echo "Patches already applied or not applicable. Skipping."
fi

cd "${REPO_ROOT}"
echo ""

# =============================================================================
# Step 6: Install pixel3dmm + preprocessing pipeline
# =============================================================================
echo "===== Step 6: Installing pixel3dmm + preprocessing pipeline ====="

# Install pixel3dmm in editable mode
pip install -e "${P3DMM_DIR}"

PREPROC_DIR="${P3DMM_DIR}/src/pixel3dmm/preprocessing"

# --- facer (face parsing) ---
echo "[6a] Setting up facer..."
cd "${PREPROC_DIR}"
if [ ! -d "facer" ]; then
    git clone https://github.com/FacePerceiver/facer.git
fi
cd facer
cp "${PREPROC_DIR}/replacement_code/farl.py" facer/face_parsing/farl.py
cp "${PREPROC_DIR}/replacement_code/facer_transform.py" facer/transform.py
pip install -e .
cd "${REPO_ROOT}"

# --- MICA (3D face reconstruction) ---
echo "[6b] Setting up MICA..."
cd "${PREPROC_DIR}"
if [ ! -d "MICA" ]; then
    git clone https://github.com/Zielon/MICA.git
fi
cd MICA
cp "${PREPROC_DIR}/replacement_code/install_mica_download_flame.sh" install.sh
cp "${PREPROC_DIR}/replacement_code/mica_demo.py" demo.py
cp "${PREPROC_DIR}/replacement_code/mica.py" micalib/models/mica.py
chmod +x install.sh
bash install.sh
cd "${REPO_ROOT}"

# --- PIPNet (face landmark detection) ---
echo "[6c] Setting up PIPNet..."
cd "${PREPROC_DIR}"
if [ ! -d "PIPNet" ]; then
    git clone https://github.com/jhb86253817/PIPNet.git
fi
cd PIPNet/FaceBoxesV2/utils
if ! ls ../../FaceBoxesV2/utils/nms/*.so 2>/dev/null | grep -q .; then
    bash make.sh
fi
cd ../..
mkdir -p snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/
if [ ! -f "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth" ]; then
    pip install gdown
    gdown --id 1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C \
        -O snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth
fi
cd "${REPO_ROOT}"

# --- pixel3dmm pretrained weights ---
echo "[6d] Downloading pixel3dmm pretrained weights..."
mkdir -p "${P3DMM_DIR}/pretrained_weights"
cd "${P3DMM_DIR}/pretrained_weights"
if [ ! -f "uv.ckpt" ]; then
    pip install gdown
    gdown --id 1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY -O ./uv.ckpt
fi
if [ ! -f "normals.ckpt" ]; then
    gdown --id 1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD -O ./normals.ckpt
fi
cd "${REPO_ROOT}"

echo ""

# =============================================================================
# Step 7: Copy optimized scripts into pixel3dmm
# =============================================================================
echo "===== Step 7: Copying optimized scripts ====="

cp "${REPO_ROOT}/generate_exp_map/src/pixel3dmm_preprocessing.py" "${P3DMM_DIR}/scripts/"
cp "${REPO_ROOT}/generate_exp_map/src/pixel3dmm_inference.py" "${P3DMM_DIR}/scripts/"
cp "${REPO_ROOT}/generate_exp_map/src/pixel3dmm_segmentation.py" "${P3DMM_DIR}/scripts/"

echo "Copied 3 optimized scripts to ${P3DMM_DIR}/scripts/"

# Create __init__.py files so PIPNet/MICA are importable as Python packages
# (pixel3dmm imports them as pixel3dmm.preprocessing.PIPNet.FaceBoxesV2.*)
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/__init__.py" 2>/dev/null
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/__init__.py" 2>/dev/null
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/utils/__init__.py" 2>/dev/null
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/MICA/__init__.py" 2>/dev/null
# Patch FaceBoxesV2 bare imports to relative imports (needed when loaded as a package)
FB_DET="${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/faceboxes_detector.py"
if [ -f "${FB_DET}" ]; then
    sed -i 's/^from detector import/from .detector import/' "${FB_DET}"
    sed -i 's/^from utils\./from .utils./' "${FB_DET}"
fi
echo "Created __init__.py files and patched imports for PIPNet/MICA"
echo ""

# =============================================================================
# Step 8: Install remaining Python dependencies
# =============================================================================
echo "===== Step 8: Installing remaining dependencies ====="

# Face detection and segmentation
pip install insightface==0.7.3 onnxruntime-gpu
pip install face-alignment

# 3D and video
pip install trimesh
pip install decord
pip install omegaconf
pip install tyro

# General
pip install scipy opencv-python tqdm pyyaml environs mediapy loguru distinctipy einops chumpy
pip install pytorch-lightning==2.0.0 wandb tensorboard pyvista dreifus

# Pin numpy LAST — must be >=1.26 (FLAME pkl files use numpy._core from 2.x)
# but <2.0 (PyTorch3D 0.7.4 and other compiled extensions need numpy 1.x)
pip install "numpy>=1.26,<2"

echo ""

# =============================================================================
# Step 9: Download pretrained weights
# =============================================================================
echo "===== Step 9: Checking pretrained weights ====="

# pixel3dmm checkpoints
if [ -f "${P3DMM_DIR}/pretrained_weights/normals.ckpt" ] && \
   [ -f "${P3DMM_DIR}/pretrained_weights/uv.ckpt" ]; then
    echo "pixel3dmm checkpoints found."
else
    echo ""
    echo "WARNING: pixel3dmm pretrained checkpoints not found!"
    echo "Expected at: ${P3DMM_DIR}/pretrained_weights/{normals.ckpt, uv.ckpt}"
    echo "Follow pixel3dmm README to download them."
    echo ""
fi

# FLAME models (requires registration at https://flame.is.tue.mpg.de/)
MICA_DATA="${P3DMM_DIR}/src/pixel3dmm/preprocessing/MICA/data"

# FLAME 2023 (used by pixel3dmm tracking)
if [ ! -f "${MICA_DATA}/FLAME2023/flame2023.pkl" ]; then
    echo ""
    echo "FLAME 2023 model not found. Downloading..."
    echo "This requires your FLAME website credentials."
    echo "(Register at https://flame.is.tue.mpg.de/ if you haven't)"
    echo ""
    read -p "FLAME username (email): " FLAME_USER
    read -sp "FLAME password: " FLAME_PASS
    echo ""

    FLAME_USER_ENC=$(python -c "import urllib.parse; print(urllib.parse.quote('${FLAME_USER}'))")
    FLAME_PASS_ENC=$(python -c "import urllib.parse; print(urllib.parse.quote('${FLAME_PASS}'))")

    # Download FLAME 2023
    wget --post-data "username=${FLAME_USER_ENC}&password=${FLAME_PASS_ENC}" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1' \
        -O '/tmp/FLAME2023.zip' --no-check-certificate --continue
    unzip -o /tmp/FLAME2023.zip -d "${MICA_DATA}/"
    rm -f /tmp/FLAME2023.zip

    # Download FLAME 2020 (needed by MICA for generic_model.pkl)
    wget --post-data "username=${FLAME_USER_ENC}&password=${FLAME_PASS_ENC}" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
        -O '/tmp/FLAME2020.zip' --no-check-certificate --continue
    unzip -o /tmp/FLAME2020.zip -d "${MICA_DATA}/FLAME2020/"
    rm -f /tmp/FLAME2020.zip

    echo "FLAME models downloaded."
else
    echo "FLAME 2023 model found."
fi

# Verify FLAME 2020 generic_model.pkl (critical for MICA)
if [ ! -f "${MICA_DATA}/FLAME2020/generic_model.pkl" ]; then
    echo "WARNING: FLAME2020/generic_model.pkl not found!"
    echo "MICA preprocessing will fail without it."
    echo "Download FLAME 2020 from https://flame.is.tue.mpg.de/ and place"
    echo "generic_model.pkl in ${MICA_DATA}/FLAME2020/"
fi

# L2CS gaze weights
L2CS_WEIGHTS="${REPO_ROOT}/data/weights/l2cs/L2CSNet_gaze360.pkl"
if [ -f "${L2CS_WEIGHTS}" ]; then
    echo "L2CS gaze weights found."
else
    echo ""
    echo "WARNING: L2CS gaze weights not found at ${L2CS_WEIGHTS}"
    echo "Phase 2 (FlowFace conversion) requires this for gaze tracking."
    echo ""
fi

# RVM background matting weights
RVM_WEIGHTS="${REPO_ROOT}/data/weights/rvm/rvm_mobilenetv3.pth"
if [ -f "${RVM_WEIGHTS}" ]; then
    echo "RVM matting weights found."
else
    echo ""
    echo "WARNING: RVM matting weights not found at ${RVM_WEIGHTS}"
    echo "Phase 2 (FlowFace conversion) requires this for background matting."
    echo ""
fi

echo ""

# =============================================================================
# Step 10: Create output directories
# =============================================================================
echo "===== Step 10: Creating output directories ====="

mkdir -p "${REPO_ROOT}/data/flame_tracking/preprocessing"
mkdir -p "${REPO_ROOT}/data/flame_tracking/tracking"
mkdir -p "${REPO_ROOT}/data/flame_tracking/logs"
mkdir -p "${REPO_ROOT}/data/flowface"

echo "Created data/flame_tracking/{preprocessing,tracking,logs} and data/flowface/"
echo ""

# =============================================================================
# Step 11: Verify installation
# =============================================================================
echo "===== Step 11: Verifying installation ====="

python -c "
import torch; print(f'  torch=={torch.__version__} (CUDA {torch.version.cuda})')
import torchvision; print(f'  torchvision=={torchvision.__version__}')
import pytorch3d; print(f'  pytorch3d=={pytorch3d.__version__}')
import numpy; print(f'  numpy=={numpy.__version__}')
import cv2; print(f'  opencv=={cv2.__version__}')
import trimesh; print(f'  trimesh=={trimesh.__version__}')
import omegaconf; print(f'  omegaconf=={omegaconf.__version__}')
import pixel3dmm; print(f'  pixel3dmm: OK')
print('  All imports passed!')
" || {
    echo "ERROR: Some imports failed. Check the output above."
    exit 1
}

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Before running, set the environment variable:"
echo "  export PIXEL3DMM_CODE_BASE=${P3DMM_DIR}"
echo ""
echo "Then run:"
echo "  conda activate expmapgen"
echo "  cd ${REPO_ROOT}"
echo "  bash generate_exp_map/scripts/run_multi_gpu.sh"
echo "============================================================"
