#!/bin/bash
# =============================================================================
# One-shot HunyuanPortrait environment setup.
#
# Steps:
#   1. Create the `hunyuan_portrait` conda env from env.yml
#      (python 3.10 + torch 2.1.0+cu121).
#   2. Install our pinned requirements.txt (diffusers 0.29.0, moviepy 1.0.1,
#      onnxruntime-gpu 1.19.2, huggingface_hub[cli], …).
#   3. Clone HunyuanPortrait's upstream repo into `impl/` and pin the commit.
#   4. Download all pretrained weights (~5–6 GB total):
#       - Stable Video Diffusion configs (JSONs)
#       - SVD VAE fp16 safetensors
#       - yoloface_v5m.pt (face detector)
#       - arcface.onnx (identity embedding)
#       - HunyuanPortrait's own 7 .pth files under `hyportrait/`
#
# Idempotent: already-completed steps are detected and skipped (per-file
# sentinels for the big artefacts), so it's safe to re-run after a failed step
# or a flaky download.
#
# Usage (from anywhere — the script self-locates relative to its own path):
#
#   bash experiments/sota_comparison/hunyuan_portrait/setup_env.sh
#
# After it finishes:
#
#   conda activate marionette
#   PYTHONPATH=. python experiments/sota_comparison/hunyuan_portrait/run_inference.py \
#       --dataset hdtf --protocol cross_identity --n_samples 200 \
#       --clip_duration_s 3.0 --seed 42
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV_NAME="hunyuan_portrait"
UPSTREAM_URL="https://github.com/Tencent-Hunyuan/HunyuanPortrait.git"
IMPL_DIR="${HERE}/impl"
WEIGHTS_DIR="${IMPL_DIR}/pretrained_weights"

echo "============================================================"
echo "[setup_env.sh] HunyuanPortrait one-shot setup"
echo "  script dir : ${HERE}"
echo "  env name   : ${ENV_NAME}"
echo "  upstream   : ${UPSTREAM_URL}"
echo "============================================================"
echo ""

# Locate conda. We first try `conda` on PATH (the clean case), then fall
# back to common install prefixes so the script still works in shells where
# conda was installed system-wide but never `conda init`-ed.
find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base
        return 0
    fi
    local candidates=(
        /opt/miniforge3  /opt/miniconda3  /opt/anaconda3
        "${HOME}/miniforge3"  "${HOME}/miniconda3"  "${HOME}/anaconda3"
    )
    for c in "${candidates[@]}"; do
        if [[ -f "${c}/etc/profile.d/conda.sh" ]]; then
            echo "${c}"
            return 0
        fi
    done
    return 1
}

CONDA_BASE="$(find_conda_base || true)"
if [[ -z "${CONDA_BASE}" ]]; then
    echo "ERROR: conda not found. Checked PATH and:"
    echo "         /opt/miniforge3  /opt/miniconda3  /opt/anaconda3"
    echo "         \$HOME/miniforge3  \$HOME/miniconda3  \$HOME/anaconda3"
    echo "Install Miniforge/Miniconda, or source its profile in your shell:"
    echo "  source /path/to/conda/etc/profile.d/conda.sh"
    exit 1
fi

# =============================================================================
# Step 1: conda env
# =============================================================================
echo "===== Step 1: conda env ====="
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk 'NR>2 {print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "SKIP: env '${ENV_NAME}' already exists."
else
    conda env create -f "${HERE}/env.yml"
fi

conda activate "${ENV_NAME}"
echo "active python: $(python --version)"
echo ""

# =============================================================================
# Step 2: pip requirements
# =============================================================================
echo "===== Step 2: pip requirements ====="
pip install -r "${HERE}/requirements.txt"
echo ""

# =============================================================================
# Step 3: clone upstream + pin commit
# =============================================================================
echo "===== Step 3: upstream repo ====="

if [[ -d "${IMPL_DIR}/.git" ]]; then
    echo "SKIP: ${IMPL_DIR} already cloned."
else
    git clone "${UPSTREAM_URL}" "${IMPL_DIR}"
fi

( cd "${IMPL_DIR}" && git rev-parse HEAD > "${HERE}/COMMIT_PIN.txt" )
echo "pinned commit: $(cat "${HERE}/COMMIT_PIN.txt")"
echo ""

# =============================================================================
# Step 4: model weights (per-file idempotent)
# =============================================================================
echo "===== Step 4: model weights ====="
mkdir -p "${WEIGHTS_DIR}"
cd "${WEIGHTS_DIR}"

# --- 4a: Stable Video Diffusion config JSONs (vae/, unet/, scheduler/) ------
if [[ -f scheduler/scheduler_config.json ]] \
   && [[ -f unet/config.json ]] \
   && [[ -f vae/config.json ]]; then
    echo "[4a] SKIP: SVD config JSONs already present."
else
    echo "[4a] SVD config JSONs..."
    huggingface-cli download --resume-download \
        stabilityai/stable-video-diffusion-img2vid-xt \
        --local-dir . --include "*.json"
fi

# --- 4b: yoloface (face detector) ------------------------------------------
if [[ -f yoloface_v5m.pt ]]; then
    echo "[4b] SKIP: yoloface_v5m.pt already present."
else
    echo "[4b] yoloface_v5m.pt..."
    wget -c https://huggingface.co/LeonJoe13/Sonic/resolve/main/yoloface_v5m.pt
fi

# --- 4c: SVD VAE weights ---------------------------------------------------
if [[ -f vae/diffusion_pytorch_model.fp16.safetensors ]]; then
    echo "[4c] SKIP: SVD VAE fp16 weights already present."
else
    echo "[4c] SVD VAE fp16 weights..."
    wget -c \
        https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors \
        -P vae
fi

# --- 4d: ArcFace (identity embedding) --------------------------------------
if [[ -f arcface.onnx ]]; then
    echo "[4d] SKIP: arcface.onnx already present."
else
    echo "[4d] arcface.onnx..."
    wget -c https://huggingface.co/FoivosPar/Arc2Face/resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx
fi

# --- 4e: HunyuanPortrait's own weights (7 .pth files) ----------------------
# Any one of the 7 shards being missing triggers a re-download of the whole
# set; huggingface-cli itself skips files already on disk so this is cheap.
HYPORTRAIT_FILES=(
    dino.pth expression.pth headpose.pth image_proj.pth
    motion_proj.pth pose_guider.pth unet.pth
)
need_hyportrait=0
for f in "${HYPORTRAIT_FILES[@]}"; do
    if [[ ! -f "hyportrait/${f}" ]]; then
        need_hyportrait=1
        break
    fi
done
if [[ "${need_hyportrait}" -eq 0 ]]; then
    echo "[4e] SKIP: hyportrait/ fully populated."
else
    echo "[4e] HunyuanPortrait weights..."
    huggingface-cli download --resume-download tencent/HunyuanPortrait \
        --local-dir hyportrait
fi

cd "${HERE}"
echo ""

# =============================================================================
# Step 5: Verify the layout matches upstream's expectation
# =============================================================================
echo "===== Step 5: layout verification ====="
EXPECTED=(
    "pretrained_weights/arcface.onnx"
    "pretrained_weights/yoloface_v5m.pt"
    "pretrained_weights/vae/diffusion_pytorch_model.fp16.safetensors"
    "pretrained_weights/vae/config.json"
    "pretrained_weights/unet/config.json"
    "pretrained_weights/scheduler/scheduler_config.json"
    "pretrained_weights/hyportrait/dino.pth"
    "pretrained_weights/hyportrait/expression.pth"
    "pretrained_weights/hyportrait/headpose.pth"
    "pretrained_weights/hyportrait/image_proj.pth"
    "pretrained_weights/hyportrait/motion_proj.pth"
    "pretrained_weights/hyportrait/pose_guider.pth"
    "pretrained_weights/hyportrait/unet.pth"
)
missing=0
for rel in "${EXPECTED[@]}"; do
    if [[ ! -f "${IMPL_DIR}/${rel}" ]]; then
        echo "  MISSING: ${rel}"
        missing=1
    fi
done
if [[ "${missing}" -eq 1 ]]; then
    echo "ERROR: some weights are missing. Re-run this script (downloads resume)."
    exit 1
fi
echo "all expected weights present."
echo ""

# =============================================================================
# Done
# =============================================================================
echo "============================================================"
echo "[setup_env.sh] HunyuanPortrait setup complete."
echo ""
echo "Next steps (from repo root):"
echo "  conda activate marionette"
echo "  PYTHONPATH=. python experiments/sota_comparison/hunyuan_portrait/run_inference.py \\"
echo "      --dataset hdtf \\"
echo "      --protocol cross_identity \\"
echo "      --n_samples 200 \\"
echo "      --clip_duration_s 3.0 \\"
echo "      --seed 42"
echo "============================================================"
