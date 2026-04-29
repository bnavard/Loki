#!/bin/bash
# =============================================================================
# One-shot EchoMimic environment setup.
#
# Steps:
#   1. Create the `echomimic` conda env (python 3.10 + torch 2.1.0+cu121).
#   2. Install our requirements.txt (upstream pins + huggingface_hub[cli]
#      for the weight download).
#   3. Clone EchoMimic's upstream repo into `impl/` and pin the commit.
#   4. Download the audio-only checkpoint subset of `BadToBest/EchoMimic`
#      (~10 GB) into `impl/pretrained_weights/`. The full HF repo is 34 GB
#      across multiple variants (`*_pose`, `*_acc`); we explicitly include
#      only what `infer_audio2vid.py` needs.
#
# Idempotent: per-file sentinels skip artefacts already on disk, so it's
# safe to re-run after a flaky HuggingFace connection.
#
# Usage (from anywhere — the script self-locates relative to its own path):
#
#   bash experiments/sota_comparison/echomimic/setup_env.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV_NAME="echomimic"
UPSTREAM_URL="https://github.com/antgroup/echomimic.git"
IMPL_DIR="${HERE}/impl"
WEIGHTS_DIR="${IMPL_DIR}/pretrained_weights"

echo "============================================================"
echo "[setup_env.sh] EchoMimic one-shot setup"
echo "  script dir : ${HERE}"
echo "  env name   : ${ENV_NAME}"
echo "  upstream   : ${UPSTREAM_URL}"
echo "============================================================"
echo ""

# Locate conda. PATH first, then common install prefixes.
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
    echo "Install Miniforge/Miniconda, or source its profile in your shell."
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

if [[ -f "${IMPL_DIR}/infer_audio2vid.py" ]]; then
    # Accept either a git clone or a manual zip-extract.
    echo "SKIP: ${IMPL_DIR} already populated (infer_audio2vid.py present)."
else
    git clone "${UPSTREAM_URL}" "${IMPL_DIR}"
fi

if [[ -d "${IMPL_DIR}/.git" ]]; then
    ( cd "${IMPL_DIR}" && git rev-parse HEAD > "${HERE}/COMMIT_PIN.txt" )
    echo "pinned commit: $(cat "${HERE}/COMMIT_PIN.txt")"
else
    echo "NOTE: ${IMPL_DIR} has no .git — likely a zip-extracted mirror. "
    echo "      Record the upstream commit manually in COMMIT_PIN.txt."
fi
echo ""

# =============================================================================
# Step 4: checkpoints (audio-only subset of BadToBest/EchoMimic)
# =============================================================================
echo "===== Step 4: checkpoints ====="
mkdir -p "${WEIGHTS_DIR}"

# The full HF repo is 34 GB across multiple model variants:
#   - denoising_unet{,_acc,_pose,_pose_acc}.pth   (3.4 GB each)
#   - motion_module{,_acc,_pose,_pose_acc}.pth    (1.82 GB each)
#   - reference_unet{,_pose}.pth                  (3.26 GB each)
#   - face_locator{,_pose}.pth                    (4.35 MB each)
#   - audio_processor/, sd-vae-ft-mse/, sd-image-variations-diffusers/
# We explicitly --include only the **audio-only / non-accelerated** subset
# that `infer_audio2vid.py` consumes, totalling ~10 GB. That's the variant
# referenced in `configs/prompts/animation.yaml`.
need_download=0
for sentinel in \
    "${WEIGHTS_DIR}/denoising_unet.pth" \
    "${WEIGHTS_DIR}/reference_unet.pth" \
    "${WEIGHTS_DIR}/motion_module.pth" \
    "${WEIGHTS_DIR}/face_locator.pth" \
    "${WEIGHTS_DIR}/audio_processor/whisper_tiny.pt" \
    "${WEIGHTS_DIR}/sd-vae-ft-mse/config.json" \
    "${WEIGHTS_DIR}/sd-image-variations-diffusers/model_index.json"; do
    if [[ ! -f "${sentinel}" ]]; then
        need_download=1
        break
    fi
done

if [[ "${need_download}" -eq 0 ]]; then
    echo "SKIP: ckpts already present."
else
    echo "Downloading audio-only subset of BadToBest/EchoMimic → ${WEIGHTS_DIR}"
    # `huggingface-cli` (not the newer `hf`): we're pinned to
    # huggingface_hub<0.31 because diffusers 0.24.0 imports the
    # since-removed `hf_cache_home` symbol — see the comment in
    # requirements.txt. Resume is automatic; partial files re-hash
    # and continue. The `_pose` and `_acc` variants are deliberately
    # excluded — pose-driven and accelerated paths are out of scope here.
    huggingface-cli download BadToBest/EchoMimic \
        --repo-type model \
        --local-dir "${WEIGHTS_DIR}" \
        --include \
            "denoising_unet.pth" \
            "reference_unet.pth" \
            "motion_module.pth" \
            "face_locator.pth" \
            "audio_processor/*" \
            "sd-vae-ft-mse/*" \
            "sd-image-variations-diffusers/*"
fi
echo ""

# =============================================================================
# Step 5: layout verification
# =============================================================================
echo "===== Step 5: layout verification ====="
EXPECTED=(
    "infer_audio2vid.py"
    "configs/prompts/animation.yaml"
    "configs/inference/inference_v2.yaml"
    "pretrained_weights/denoising_unet.pth"
    "pretrained_weights/reference_unet.pth"
    "pretrained_weights/motion_module.pth"
    "pretrained_weights/face_locator.pth"
    "pretrained_weights/audio_processor/whisper_tiny.pt"
    "pretrained_weights/sd-vae-ft-mse/config.json"
    "pretrained_weights/sd-image-variations-diffusers/model_index.json"
)
missing=0
for rel in "${EXPECTED[@]}"; do
    if [[ ! -f "${IMPL_DIR}/${rel}" ]]; then
        echo "  MISSING: ${rel}"
        missing=1
    fi
done
if [[ "${missing}" -eq 1 ]]; then
    echo "ERROR: some files are missing. Re-run this script — downloads resume."
    exit 1
fi
echo "all expected files present."
echo ""

# =============================================================================
# Done
# =============================================================================
echo "============================================================"
echo "[setup_env.sh] EchoMimic setup complete."
echo ""
echo "Next steps (from repo root):"
echo "  conda activate marionette"
echo "  PYTHONPATH=. python experiments/sota_comparison/echomimic/run_inference.py \\"
echo "      --dataset hdtf \\"
echo "      --protocol cross_identity \\"
echo "      --n_samples 200 \\"
echo "      --clip_duration_s 3.0 \\"
echo "      --seed 42"
echo "============================================================"
