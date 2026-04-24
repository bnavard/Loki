#!/bin/bash
# =============================================================================
# One-shot X-Portrait environment setup.
#
# Steps:
#   1. Create the `xportrait` conda env from env.yml
#      (python 3.9 + torch 2.0.1+cu118 — upstream hard-requires this stack).
#   2. Install our requirements.txt (upstream's pins + gdown for the GDrive
#      checkpoint + a few small tweaks documented inline).
#   3. Clone X-Portrait's upstream repo into `impl/` and pin the commit.
#   4. Download the pre-trained checkpoint `model_state-415001.th` (~3 GB)
#      from Google Drive into `impl/checkpoint/`.
#
# Idempotent: already-completed steps are detected and skipped, so it's safe
# to re-run after a failed step or a flaky GDrive connection.
#
# Usage (from anywhere — the script self-locates relative to its own path):
#
#   bash experiments/sota_comparison/xportrait/setup_env.sh
#
# After it finishes:
#
#   conda activate marionette
#   PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \
#       --dataset talkvid --protocol cross_identity --n_samples 125 \
#       --clip_duration_s 5.0 --seed 42
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV_NAME="xportrait"
UPSTREAM_URL="https://github.com/bytedance/X-Portrait.git"
IMPL_DIR="${HERE}/impl"
CKPT_DIR="${IMPL_DIR}/checkpoint"
CKPT_FILE="${CKPT_DIR}/model_state-415001.th"

# Upstream's demo script hardcodes this exact filename; do not rename.
CKPT_GDRIVE_ID="1VOpVg25EQTUlbHOvuLEFi8rBhVd2KlxQ"

echo "============================================================"
echo "[setup_env.sh] X-Portrait one-shot setup"
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
# Step 4: checkpoint download
# =============================================================================
echo "===== Step 4: checkpoint ====="
mkdir -p "${CKPT_DIR}"

if [[ -f "${CKPT_FILE}" ]]; then
    size_mb="$(du -m "${CKPT_FILE}" | cut -f1)"
    echo "SKIP: ${CKPT_FILE} already present (${size_mb} MB)."
else
    echo "Downloading ${CKPT_FILE}"
    echo "  GDrive id: ${CKPT_GDRIVE_ID}"
    # gdown handles Google Drive's large-file confirmation handshake + resume
    # transparently; --fuzzy lets it accept either the file id or a full URL.
    # Output filename fixed — upstream's inference.py looks for this name.
    gdown --id "${CKPT_GDRIVE_ID}" -O "${CKPT_FILE}"
fi
echo ""

# =============================================================================
# Step 5: layout verification
# =============================================================================
echo "===== Step 5: layout verification ====="
EXPECTED=(
    "core/test_xportrait.py"
    "config/cldm_v15_appearance_pose_local_mm.yaml"
    "checkpoint/model_state-415001.th"
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
echo "[setup_env.sh] X-Portrait setup complete."
echo ""
echo "Next steps (from repo root):"
echo "  conda activate marionette"
echo "  PYTHONPATH=. python experiments/sota_comparison/xportrait/run_inference.py \\"
echo "      --dataset talkvid \\"
echo "      --protocol cross_identity \\"
echo "      --n_samples 125 \\"
echo "      --clip_duration_s 5.0 \\"
echo "      --seed 42"
echo "============================================================"
