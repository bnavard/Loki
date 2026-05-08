# Source-only helper: activate the `loki` conda env and expose
# `PYTHON` pointing at the env's interpreter. Sourced by:
#   - run_eval_metrics.sh
#   - sanity_check/visualize_*.sh
#
# All evaluation metrics run inside the same env that trains and
# generates the model — head-rot and expression need pytorch3d (already
# in loki), and id_cosine needs `onnxruntime-gpu` against torch's
# bundled cuDNN 9. We prepend that cuDNN dir to `LD_LIBRARY_PATH` so
# onnxruntime-gpu's CUDAExecutionProvider can dlopen
# `libcudnn_*.so.9` at session-create time.

# `find_conda_base`: print the conda installation root or return non-zero.
find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base
        return 0
    fi
    local candidates=(
        /opt/miniforge3 /opt/miniconda3 /opt/anaconda3
        "${HOME}/miniforge3" "${HOME}/miniconda3" "${HOME}/anaconda3"
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
    echo "ERROR: conda not found. Looked in PATH and:" >&2
    echo "  /opt/miniforge3  /opt/miniconda3  /opt/anaconda3" >&2
    echo "  \$HOME/miniforge3  \$HOME/miniconda3  \$HOME/anaconda3" >&2
    exit 1
fi

# MKL's conda activate hook expands `MKL_INTERFACE_LAYER` /
# `MKL_THREADING_LAYER` before they're defined, which trips `set -u`.
# Bracket the activate.
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate loki
set -u

PYTHON="$(command -v python)"
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: \`python\` not on PATH after activating loki." >&2
    exit 1
fi

# onnxruntime-gpu loads libcudnn_*.so.9 from torch's nvidia cudnn wheel.
CUDNN_LIB="$("${PYTHON}" -c 'import os, nvidia.cudnn as c; print(os.path.join(os.path.dirname(c.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "${CUDNN_LIB}" && -d "${CUDNN_LIB}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi
