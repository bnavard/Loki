#!/bin/bash
# =============================================================================
# One-shot SadTalker environment setup.
#
# Steps:
#   1. Create the `sadtalker` conda env from env.yml (python 3.8 + torch 2.1).
#   2. Install our pinned requirements.txt (NOT upstream's — ours is compatible
#      with the torch 2.1 stack; upstream still targets torch 1.12).
#   3. Clone SadTalker's upstream repo into `impl/` and pin the commit.
#   4. Download SadTalker's model checkpoints (~2 GB).
#
# Idempotent: already-completed steps are detected and skipped, so it's safe
# to re-run after a failed step.
#
# Usage (from anywhere — the script self-locates relative to its own path):
#
#   bash experiments/sota_comparison/sadtalker/setup_env.sh
#
# After it finishes:
#
#   conda activate sadtalker     # only if you want to poke at the env directly
#   # otherwise launch the runner from the marionette env:
#   conda activate marionette
#   PYTHONPATH=. python experiments/sota_comparison/sadtalker/run_inference.py \
#       --dataset hdtf --protocol same_identity_reconstruction --n_samples 346
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV_NAME="sadtalker"
UPSTREAM_URL="https://github.com/OpenTalker/SadTalker.git"
IMPL_DIR="${HERE}/impl"

echo "============================================================"
echo "[setup_env.sh] SadTalker one-shot setup"
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
# Step 2: pip requirements (our pinned set, NOT upstream's)
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
# Step 4: model weights
# =============================================================================
echo "===== Step 4: model weights ====="

# SadTalker's download_models.sh pulls its checkpoints into impl/checkpoints/
# and (optionally) impl/gfpgan/. ~2 GB total. The script itself is idempotent
# enough — it uses wget with `-c` resume — but we guard the outer bash run so
# a completed download isn't re-fetched.
CKPT_DIR="${IMPL_DIR}/checkpoints"
if [[ -d "${CKPT_DIR}" ]] && [[ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]]; then
    echo "SKIP: checkpoints/ already populated."
else
    ( cd "${IMPL_DIR}" && bash scripts/download_models.sh )
fi
echo ""

# =============================================================================
# Done
# =============================================================================
echo "============================================================"
echo "[setup_env.sh] SadTalker setup complete."
echo ""
echo "Next steps (from repo root):"
echo "  conda activate marionette"
echo "  PYTHONPATH=. python experiments/sota_comparison/sadtalker/run_inference.py \\"
echo "      --dataset hdtf \\"
echo "      --protocol same_identity_reconstruction \\"
echo "      --n_samples 346 \\"
echo "      --clip_duration_s 3.0 \\"
echo "      --seed 42"
echo "============================================================"
