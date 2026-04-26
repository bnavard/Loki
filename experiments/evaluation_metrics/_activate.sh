# Source-only helper: activate the `evaluation_metrics` conda env and
# expose `PYTHON` pointing at the env's interpreter.
#
# Sourced by:
#   - run_eval_metrics_on_sota.sh
#   - sanity_check/visualize_batch.sh
#
# Both scripts run on different boxes where the env can live at any of
# `/venv/`, `/opt/miniforge3/envs/`, `~/miniconda3/envs/`, etc., so we
# can't bind-mount the python path. Instead we locate conda's profile
# script (probing PATH first, then a few well-known prefixes), source
# it, and `conda activate evaluation_metrics`. `PYTHON` is then just
# `$(command -v python)`.

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
conda activate evaluation_metrics
set -u

# Now PATH points at the env. Resolve once so `PYTHON` is stable across
# subshells (CUDA_VISIBLE_DEVICES propagation etc.).
PYTHON="$(command -v python)"
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: \`python\` not on PATH after activating evaluation_metrics." >&2
    echo "Did \`bash experiments/evaluation_metrics/setup_env.sh\` run?" >&2
    exit 1
fi
