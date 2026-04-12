#!/bin/bash
# =============================================================================
# Full setup for the FLAME expression map generation pipeline.
#
# Creates the conda environment, clones and patches pixel3dmm, installs
# all dependencies, and downloads pretrained weights.
#
# Idempotent: safe to re-run after a failure — already-completed steps
# are detected and skipped automatically.
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
ENV_NAME="expmapgen"

echo "============================================================"
echo "Setting up FLAME expression map generation pipeline"
echo "Repo root: ${REPO_ROOT}"
echo "pixel3dmm: ${P3DMM_DIR}"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Create conda environment
# =============================================================================
echo "===== Step 1: Conda environment ====="

if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "SKIP: Environment '${ENV_NAME}' already exists."
else
    conda create -n "${ENV_NAME}" python=3.9 -y
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "Python: $(python --version)"
echo ""

# =============================================================================
# Step 2: Install PyTorch + CUDA
# =============================================================================
echo "===== Step 2: PyTorch ====="

if python -c "import torch; assert torch.__version__.startswith('2.0.1')" 2>/dev/null; then
    echo "SKIP: PyTorch 2.0.1 already installed."
else
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
fi
echo ""

# =============================================================================
# Step 3: Install PyTorch3D
# =============================================================================
echo "===== Step 3: PyTorch3D ====="

if python -c "import pytorch3d" 2>/dev/null; then
    echo "SKIP: PyTorch3D already installed."
else
    pip install fvcore iopath
    pip install --no-index --no-cache-dir pytorch3d==0.7.4 \
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
fi
echo ""

# =============================================================================
# Step 4: Install nvdiffrast
# =============================================================================
echo "===== Step 4: nvdiffrast ====="

if python -c "import nvdiffrast" 2>/dev/null; then
    echo "SKIP: nvdiffrast already installed."
else
    # nvdiffrast requires CUDA 11.8 nvcc + GCC ≤ 11 + ninja
    pip install setuptools wheel ninja cython

    # Install GCC 11 via conda-forge (no system permissions needed)
    if ! ls "${CONDA_PREFIX}"/bin/*conda*gnu-gcc 2>/dev/null | grep -q .; then
        echo "Installing GCC 11 via conda-forge..."
        conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
    else
        echo "GCC 11 already installed via conda."
    fi

    CONDA_GCC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    CONDA_GXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
    if [ ! -f "${CONDA_GCC}" ]; then
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
fi
echo ""

# =============================================================================
# Step 5: Clone and patch pixel3dmm
# =============================================================================
echo "===== Step 5: pixel3dmm clone + patch ====="

if [ -d "${P3DMM_DIR}" ]; then
    echo "SKIP: pixel3dmm already cloned."
else
    git clone https://github.com/SimonGiebenhain/pixel3dmm.git "${P3DMM_DIR}"
fi

cd "${P3DMM_DIR}"
if git apply --check "${REPO_ROOT}/generate_exp_map/patches/pixel3dmm_fixes.patch" 2>/dev/null; then
    git apply "${REPO_ROOT}/generate_exp_map/patches/pixel3dmm_fixes.patch"
    echo "Patches applied."
else
    echo "SKIP: Patches already applied."
fi
cd "${REPO_ROOT}"
echo ""

# =============================================================================
# Step 6: Install pixel3dmm + preprocessing pipeline
# =============================================================================
echo "===== Step 6: pixel3dmm + preprocessing ====="

PREPROC_DIR="${P3DMM_DIR}/src/pixel3dmm/preprocessing"

# pixel3dmm itself
if python -c "import pixel3dmm" 2>/dev/null; then
    echo "SKIP: pixel3dmm already installed."
else
    pip install -e "${P3DMM_DIR}"
fi

# --- facer ---
echo "[6a] facer..."
if [ -d "${PREPROC_DIR}/facer" ] && python -c "import facer" 2>/dev/null; then
    echo "SKIP: facer already installed."
else
    cd "${PREPROC_DIR}"
    [ ! -d "facer" ] && git clone https://github.com/FacePerceiver/facer.git
    cd facer
    cp "${PREPROC_DIR}/replacement_code/farl.py" facer/face_parsing/farl.py 2>/dev/null || true
    cp "${PREPROC_DIR}/replacement_code/facer_transform.py" facer/transform.py 2>/dev/null || true
    pip install -e .
    cd "${REPO_ROOT}"
fi

# --- MICA ---
echo "[6b] MICA..."
if [ -d "${PREPROC_DIR}/MICA/micalib" ]; then
    echo "SKIP: MICA already cloned and patched."
else
    cd "${PREPROC_DIR}"
    [ ! -d "MICA" ] && git clone https://github.com/Zielon/MICA.git
    cd MICA
    cp "${PREPROC_DIR}/replacement_code/install_mica_download_flame.sh" install.sh 2>/dev/null || true
    cp "${PREPROC_DIR}/replacement_code/mica_demo.py" demo.py 2>/dev/null || true
    cp "${PREPROC_DIR}/replacement_code/mica.py" micalib/models/mica.py 2>/dev/null || true
    chmod +x install.sh
    bash install.sh
    cd "${REPO_ROOT}"
fi

# --- PIPNet ---
echo "[6c] PIPNet..."
cd "${PREPROC_DIR}"
[ ! -d "PIPNet" ] && git clone https://github.com/jhb86253817/PIPNet.git

# Build NMS extension
cd PIPNet/FaceBoxesV2/utils
if ! ls nms/*.so 2>/dev/null | grep -q .; then
    sed -i 's/np\.int_t/np.intp_t/g' nms/cpu_nms.pyx
    bash make.sh
else
    echo "SKIP: NMS already compiled."
fi
cd ../..

# Download PIPNet snapshot
mkdir -p snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/
if [ ! -f "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth" ]; then
    pip install gdown
    gdown --id 1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C \
        -O snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth
else
    echo "SKIP: PIPNet snapshot already downloaded."
fi
cd "${REPO_ROOT}"

# --- pixel3dmm pretrained weights ---
echo "[6d] pixel3dmm weights..."
mkdir -p "${P3DMM_DIR}/pretrained_weights"
cd "${P3DMM_DIR}/pretrained_weights"
if [ ! -f "uv.ckpt" ] || [ ! -f "normals.ckpt" ]; then
    pip install gdown
    [ ! -f "uv.ckpt" ] && gdown --id 1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY -O ./uv.ckpt
    [ ! -f "normals.ckpt" ] && gdown --id 1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD -O ./normals.ckpt
else
    echo "SKIP: pixel3dmm checkpoints already downloaded."
fi
cd "${REPO_ROOT}"
echo ""

# =============================================================================
# Step 7: Copy optimized scripts + patch imports
# =============================================================================
echo "===== Step 7: Optimized scripts + import patches ====="

cp "${REPO_ROOT}/generate_exp_map/src/pixel3dmm_preprocessing.py" "${P3DMM_DIR}/scripts/"
cp "${REPO_ROOT}/generate_exp_map/src/pixel3dmm_inference.py" "${P3DMM_DIR}/scripts/"
cp "${REPO_ROOT}/generate_exp_map/src/pixel3dmm_segmentation.py" "${P3DMM_DIR}/scripts/"

# __init__.py files so PIPNet/MICA are importable as packages
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/__init__.py" 2>/dev/null || true
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/__init__.py" 2>/dev/null || true
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/utils/__init__.py" 2>/dev/null || true
touch "${P3DMM_DIR}/src/pixel3dmm/preprocessing/MICA/__init__.py" 2>/dev/null || true

# Patch FaceBoxesV2 bare imports to relative
FB_DET="${P3DMM_DIR}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/faceboxes_detector.py"
if [ -f "${FB_DET}" ] && grep -q "^from detector import" "${FB_DET}"; then
    sed -i 's/^from detector import/from .detector import/' "${FB_DET}"
    sed -i 's/^from utils\./from .utils./' "${FB_DET}"
    echo "Patched FaceBoxesV2 imports."
else
    echo "SKIP: FaceBoxesV2 imports already patched."
fi
echo ""

# =============================================================================
# Step 8: Install remaining Python dependencies
# =============================================================================
echo "===== Step 8: Remaining dependencies ====="

pip install insightface==0.7.3 onnxruntime-gpu
pip install face-alignment
pip install trimesh decord omegaconf tyro
pip install scipy opencv-python tqdm pyyaml environs mediapy loguru distinctipy einops chumpy
pip install pytorch-lightning==2.0.0 wandb tensorboard pyvista dreifus

# Pin numpy LAST — must be >=1.26 (FLAME pkl files use numpy._core)
# but <2.0 (compiled extensions need numpy 1.x ABI)
pip install "numpy>=1.26,<2"

echo ""

# =============================================================================
# Step 9: FLAME models (requires registration)
# =============================================================================
echo "===== Step 9: FLAME models ====="

MICA_DATA="${P3DMM_DIR}/src/pixel3dmm/preprocessing/MICA/data"

if [ -f "${MICA_DATA}/FLAME2023/flame2023.pkl" ] && [ -f "${MICA_DATA}/FLAME2020/generic_model.pkl" ]; then
    echo "SKIP: FLAME 2023 + 2020 models already present."
else
    echo ""
    echo "FLAME models required. This needs your FLAME website credentials."
    echo "(Register at https://flame.is.tue.mpg.de/ if you haven't)"
    echo ""
    read -p "FLAME username (email): " FLAME_USER
    read -sp "FLAME password: " FLAME_PASS
    echo ""

    FLAME_USER_ENC=$(python -c "import urllib.parse; print(urllib.parse.quote('${FLAME_USER}'))")
    FLAME_PASS_ENC=$(python -c "import urllib.parse; print(urllib.parse.quote('${FLAME_PASS}'))")

    if [ ! -f "${MICA_DATA}/FLAME2023/flame2023.pkl" ]; then
        echo "Downloading FLAME 2023..."
        wget --post-data "username=${FLAME_USER_ENC}&password=${FLAME_PASS_ENC}" \
            'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1' \
            -O '/tmp/FLAME2023.zip' --no-check-certificate --continue
        unzip -o /tmp/FLAME2023.zip -d "${MICA_DATA}/"
        rm -f /tmp/FLAME2023.zip
    fi

    if [ ! -f "${MICA_DATA}/FLAME2020/generic_model.pkl" ]; then
        echo "Downloading FLAME 2020..."
        wget --post-data "username=${FLAME_USER_ENC}&password=${FLAME_PASS_ENC}" \
            'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
            -O '/tmp/FLAME2020.zip' --no-check-certificate --continue
        unzip -o /tmp/FLAME2020.zip -d "${MICA_DATA}/FLAME2020/"
        rm -f /tmp/FLAME2020.zip
    fi

    echo "FLAME models downloaded."
fi
echo ""

# =============================================================================
# Step 10: Phase 2 weights check
# =============================================================================
echo "===== Step 10: Phase 2 weights ====="

L2CS_WEIGHTS="${REPO_ROOT}/data/weights/l2cs/L2CSNet_gaze360.pkl"
RVM_WEIGHTS="${REPO_ROOT}/data/weights/rvm/rvm_mobilenetv3.pth"

[ -f "${L2CS_WEIGHTS}" ] && echo "L2CS gaze weights: OK" || echo "WARNING: L2CS weights missing at ${L2CS_WEIGHTS}"
[ -f "${RVM_WEIGHTS}" ] && echo "RVM matting weights: OK" || echo "WARNING: RVM weights missing at ${RVM_WEIGHTS}"
echo ""

# =============================================================================
# Step 11: Create output directories
# =============================================================================
echo "===== Step 11: Output directories ====="

mkdir -p "${REPO_ROOT}/data/flame_tracking/preprocessing"
mkdir -p "${REPO_ROOT}/data/flame_tracking/tracking"
mkdir -p "${REPO_ROOT}/data/flame_tracking/logs"
mkdir -p "${REPO_ROOT}/data/flowface"

echo "OK"
echo ""

# =============================================================================
# Step 12: Verify installation
# =============================================================================
echo "===== Step 12: Verification ====="

python -c "
import torch; print(f'  torch {torch.__version__} (CUDA {torch.version.cuda})')
import torchvision; print(f'  torchvision {torchvision.__version__}')
import pytorch3d; print(f'  pytorch3d {pytorch3d.__version__}')
import nvdiffrast; print(f'  nvdiffrast {nvdiffrast.__version__}')
import numpy; print(f'  numpy {numpy.__version__}')
import cv2; print(f'  opencv {cv2.__version__}')
import pixel3dmm; print(f'  pixel3dmm OK')
print('  All imports passed!')
" || {
    echo "ERROR: Some imports failed."
    exit 1
}

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Run:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_ROOT}"
echo "  bash generate_exp_map/scripts/run_multi_gpu.sh"
echo "============================================================"
