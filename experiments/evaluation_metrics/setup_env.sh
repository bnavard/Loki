#!/bin/bash
# =============================================================================
# One-shot evaluation_metrics environment setup.
#
# Creates the `evaluation_metrics` conda env (Python 3.11 + Torch 2.1 +
# CUDA 11.8) and installs every metric dependency: torchmetrics, lpips,
# cdfvd, mediapipe, insightface (buffalo_l = RetinaFace + ArcFace),
# decord, imageio[ffmpeg], scikit-image (used by tests only).
#
# Idempotent: already-completed steps are detected and skipped, so it's
# safe to re-run after a failure.
#
# Usage (from anywhere — the script self-locates relative to its own path):
#
#   bash experiments/evaluation_metrics/setup_env.sh
#
# After it finishes:
#
#   conda activate evaluation_metrics
#   PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \
#       --run-dir outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>/
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV_NAME="evaluation_metrics"

echo "============================================================"
echo "[setup_env.sh] evaluation_metrics one-shot setup"
echo "  script dir : ${HERE}"
echo "  env name   : ${ENV_NAME}"
echo "============================================================"
echo ""

# Locate conda. Try `conda` on PATH first, then fall back to common install
# prefixes so the script still works in shells where conda was installed
# system-wide but never `conda init`-ed.
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

# MKL's conda activate hook expands `MKL_INTERFACE_LAYER` /
# `MKL_THREADING_LAYER` before they're defined, which trips `set -u`.
# Drop `-u` around the activate, restore it after.
set +u
conda activate "${ENV_NAME}"
set -u
echo "active python: $(python --version)"
echo ""

# =============================================================================
# Step 2: pip requirements
# =============================================================================
# env.yml already pip-installs requirements.txt during env creation. This
# step is a guard for re-runs against an existing env where requirements
# may have drifted — pip is a no-op when everything is already pinned.
echo "===== Step 2: pip requirements (re-pin) ====="
pip install -r "${HERE}/requirements.txt"
echo ""

# =============================================================================
# Step 3a: download MediaPipe FaceLandmarker model bundle (~3 MB)
# =============================================================================
# Mediapipe 0.10.x dropped the legacy `mp.solutions.face_mesh` API; the
# Tasks API replacement (`mediapipe.tasks.vision.FaceLandmarker`) requires
# a `.task` model bundle on disk.
echo "===== Step 3a: download FaceLandmarker bundle ====="
MP_MODEL_DIR="$(cd "${HERE}/../../" && pwd)/data/weights/mediapipe"
MP_MODEL_PATH="${MP_MODEL_DIR}/face_landmarker_v2_with_blendshapes.task"
mkdir -p "${MP_MODEL_DIR}"
if [[ -f "${MP_MODEL_PATH}" ]]; then
    echo "SKIP: ${MP_MODEL_PATH} already present."
else
    wget -q -O "${MP_MODEL_PATH}" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_blendshapes/float16/1/face_landmarker_v2_with_blendshapes.task"
    echo "downloaded ${MP_MODEL_PATH}"
fi
echo ""

# =============================================================================
# Step 3b: pre-warm InsightFace's buffalo_l pack (~330 MB)
# =============================================================================
# Triggers the auto-download into ~/.insightface/ so the first metrics run
# doesn't surprise the user with a long blocking download.
echo "===== Step 3: warm InsightFace buffalo_l ====="
python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("buffalo_l ready.")
PY
echo ""

# =============================================================================
# Done
# =============================================================================
echo "============================================================"
echo "[setup_env.sh] evaluation_metrics setup complete."
echo ""
echo "Next steps (from repo root):"
echo "  conda activate ${ENV_NAME}"
echo "  PYTHONPATH=. python experiments/evaluation_metrics/compute_metrics.py \\"
echo "      --run-dir outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>/"
echo ""
echo "Run the test suite first to confirm the env is healthy:"
echo "  PYTHONPATH=. pytest experiments/evaluation_metrics/tests/ -v"
echo "============================================================"
