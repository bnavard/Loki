#!/bin/bash
# =============================================================================
# One-shot AniTalker environment setup.
#
# Steps:
#   1. Create the `anitalker` conda env from env.yml
#      (python 3.9 + torch 2.0.1+cu118; see env.yml for why we deviate
#       from upstream's torch 1.8 pin).
#   2. Install our requirements.txt (upstream pins + gfpgan for face-SR).
#   3. Clone AniTalker's upstream repo into `impl/` and pin the commit.
#   4. Download the HuggingFace ckpt bundle (`taocode/anitalker_ckpts`,
#      ~3 GB total) into `impl/ckpts/`. Includes:
#        - stage1.ckpt                        (~188 MB)
#        - stage2_audio_only_hubert.ckpt      (~342 MB)  ← used by our runner
#        - stage2_full_control_hubert.ckpt    (~359 MB)
#        - stage2_full_control_mfcc.ckpt      (~249 MB)
#        - stage2_pose_only_hubert.ckpt       (~348 MB)
#        - stage2_pose_only_mfcc.ckpt         (~237 MB)
#        - chinese-hubert-large/              (~1.2 GB, HuBERT feature model)
#
# Idempotent: already-completed steps are detected and skipped, so it's safe
# to re-run after a failed step or a flaky HuggingFace connection.
#
# Usage (from anywhere — the script self-locates relative to its own path):
#
#   bash experiments/sota_comparison/anitalker/setup_env.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV_NAME="anitalker"
UPSTREAM_URL="https://github.com/X-LANCE/AniTalker.git"
IMPL_DIR="${HERE}/impl"
CKPT_DIR="${IMPL_DIR}/ckpts"

echo "============================================================"
echo "[setup_env.sh] AniTalker one-shot setup"
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
# Step 2b: re-pin torch stack to +cu118 build
#
# Some packages in requirements.txt (e.g. espnet, gfpgan/basicsr,
# pytorch-lightning) declare `torch==2.0.1` without a local version, so pip
# happily satisfies that constraint by replacing the +cu118 wheel from
# env.yml with the default-PyPI cu117 wheel. The result: torch on CUDA 11.7
# but torchvision still on CUDA 11.8, which crashes at runtime with
# "PyTorch and torchvision were compiled with different CUDA versions".
# Force-reinstalling at the end (with --no-deps so we don't perturb anything
# else) guarantees the env ends up on a consistent cu118 stack regardless
# of which package pulled torch backwards mid-resolution.
# =============================================================================
echo "===== Step 2b: re-pin torch stack to +cu118 ====="
pip install --force-reinstall --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  torchaudio==2.0.2+cu118
python - <<'PY'
import torch, torchvision
assert torch.version.cuda == "11.8", f"torch CUDA mismatch: {torch.version.cuda}"
assert "+cu118" in torchvision.__version__, f"torchvision mismatch: {torchvision.__version__}"
print(f"  torch       {torch.__version__} (cuda {torch.version.cuda})")
print(f"  torchvision {torchvision.__version__}")
PY
echo ""

# =============================================================================
# Step 3: clone upstream + pin commit
# =============================================================================
echo "===== Step 3: upstream repo ====="

if [[ -f "${IMPL_DIR}/code/demo.py" ]]; then
    # Accept either a git clone or a manual zip-extract into impl/ (the
    # github clone has been slow/flaky in some networks, so users sometimes
    # place a zipped mirror here directly).
    echo "SKIP: ${IMPL_DIR} already populated (code/demo.py present)."
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
# Step 4: checkpoints (pull the whole taocode/anitalker_ckpts repo)
# =============================================================================
echo "===== Step 4: checkpoints ====="
mkdir -p "${CKPT_DIR}"

# Per-file sentinels — any missing artefact triggers a re-download. The
# underlying `hf download` call is idempotent: files already on disk are
# hash-checked and skipped, partial downloads resume automatically.
need_download=0
for sentinel in \
    "${CKPT_DIR}/stage1.ckpt" \
    "${CKPT_DIR}/stage2_audio_only_hubert.ckpt" \
    "${CKPT_DIR}/chinese-hubert-large/config.json"; do
    if [[ ! -f "${sentinel}" ]]; then
        need_download=1
        break
    fi
done

if [[ "${need_download}" -eq 0 ]]; then
    echo "SKIP: ckpts already present."
else
    echo "Downloading taocode/anitalker_ckpts → ${CKPT_DIR}"
    # Newer `hf` CLI (>= 0.31) exposes `hf download`; resume is automatic
    # so we drop the old `--resume-download` flag that older huggingface_hub
    # releases accepted.
    hf download taocode/anitalker_ckpts \
        --repo-type model \
        --local-dir "${CKPT_DIR}"
fi
echo ""

# =============================================================================
# Step 5: layout verification
# =============================================================================
echo "===== Step 5: layout verification ====="
EXPECTED=(
    "code/demo.py"
    "ckpts/stage1.ckpt"
    "ckpts/stage2_audio_only_hubert.ckpt"
    "ckpts/chinese-hubert-large/config.json"
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
echo "[setup_env.sh] AniTalker setup complete."
echo ""
echo "Next steps (from repo root):"
echo "  conda activate marionette"
echo "  PYTHONPATH=. python experiments/sota_comparison/anitalker/run_inference.py \\"
echo "      --dataset talkvid \\"
echo "      --protocol cross_identity \\"
echo "      --n_samples 125 \\"
echo "      --clip_duration_s 5.0 \\"
echo "      --seed 42"
echo "============================================================"
