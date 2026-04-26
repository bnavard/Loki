#!/bin/bash
# =============================================================================
# Multi-GPU batch driver for compute_metrics.py.
#
# Walks every run dir under
# `outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_*/`,
# round-robin assigns them to the available GPUs (default 8), and runs
# `compute_metrics.py --run-dir <run_dir> --skip-fvd` in parallel —
# one worker per GPU, each worker processes its bucket of run dirs
# sequentially. Each `metrics_summary.json` is mirrored into a
# centralized tree for at-a-glance comparison:
#
#   outputs/test_metric/metrics/<baseline>/<dataset>/<protocol>/
#       └── metrics_summary.json
#
# The per-sample `metrics.jsonl` and `metrics_summary.json` also stay at
# the original `<run_dir>/` so per-run drilldown is still trivial.
#
# Per-worker logs land at `outputs/test_metric/metrics/_worker_<gpu>.log`
# so you can tail any worker's progress independently of the others.
#
# GPU memory: each worker loads LPIPS (AlexNet ~250 MB) + InsightFace
# buffalo_l (~330 MB) + MediaPipe (CPU). ~600 MB per worker — comfortable
# on H200 / 8000.
#
# Runtime: 5 baselines × 2 datasets × 2 protocols = 20 run dirs; with 8
# GPUs = 3 dirs per GPU on average. Each same-id dir takes ~5–8 min,
# each cross-id dir ~3–5 min. Wall-clock for the full sweep with FVD
# skipped: ~20–30 min.
#
# Idempotent: any run dir that already has a `metrics_summary.json`
# locally AND a centralized copy is skipped on re-run.
#
# Usage (from repo root):
#
#   bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh
#
# Override the GPU count (e.g. when sharing the box):
#
#   NUM_GPUS=4 bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh
#
# Pin to a specific subset of GPUs:
#
#   GPUS="0 2 4 6" bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh
#
# Background + log tail:
#
#   bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh \
#       > outputs/test_metric/metrics/_batch.log 2>&1 &
#   tail -f outputs/test_metric/metrics/_worker_0.log
# =============================================================================

set -u
set -o pipefail

# Locate conda + activate the `evaluation_metrics` env. Sets `$PYTHON`
# to the env's interpreter regardless of where the env lives on disk.
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_activate.sh
source "${HERE}/_activate.sh"

OUT_ROOT="outputs/test_metric/metrics"
CLI="experiments/evaluation_metrics/compute_metrics.py"
SOTA_ROOT="outputs/sota_comparison"

mkdir -p "${OUT_ROOT}"

# Resolve GPU list. `GPUS="..."` (space-separated indices) wins over
# `NUM_GPUS=N` (which expands to `0 1 ... N-1`). Default: 8 GPUs.
if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_LIST <<< "${GPUS}"
else
    NUM_GPUS="${NUM_GPUS:-8}"
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
fi
N_GPUS="${#GPU_LIST[@]}"

# Collect every run dir (sorted for stable assignment across re-runs).
mapfile -t ALL_DIRS < <(
    for d in "${SOTA_ROOT}"/*/*/*/run_*/; do
        [[ -d "$d" ]] && echo "$d"
    done | sort
)
N_DIRS="${#ALL_DIRS[@]}"

echo "============================================================"
echo "[batch] ${N_DIRS} run dirs across ${N_GPUS} GPU(s): ${GPU_LIST[*]}"
echo "============================================================"

# Round-robin assign run dirs to GPUs.
declare -A bucket
for ((i=0; i<N_DIRS; i++)); do
    gpu="${GPU_LIST[$(( i % N_GPUS ))]}"
    bucket["$gpu"]+="${ALL_DIRS[$i]}|"
done

# -----------------------------------------------------------------------------
# Worker: processes one bucket sequentially on its assigned GPU.
# -----------------------------------------------------------------------------
worker() {
    local gpu="$1"
    local dirs_str="$2"
    local n_done=0 n_skipped=0 n_failed=0

    # `|`-separated → array.
    IFS='|' read -r -a dirs <<< "${dirs_str%|}"
    local n="${#dirs[@]}"

    for run_dir in "${dirs[@]}"; do
        local rel="${run_dir#${SOTA_ROOT}/}"
        rel="${rel%/}"
        IFS='/' read -r baseline dataset protocol _run <<< "${rel}"

        local summary_local="${run_dir}metrics_summary.json"
        local summary_central="${OUT_ROOT}/${baseline}/${dataset}/${protocol}/metrics_summary.json"
        mkdir -p "$(dirname "${summary_central}")"

        if [[ -f "${summary_local}" ]] && [[ -f "${summary_central}" ]]; then
            echo "[gpu ${gpu}] [skip] ${baseline}/${dataset}/${protocol}"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        echo "[gpu ${gpu}] >>> ${baseline}/${dataset}/${protocol}"
        if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. \
                "${PYTHON}" "${CLI}" --run-dir "${run_dir}" --skip-fvd 2>&1 | tail -3; then
            if [[ -f "${summary_local}" ]]; then
                cp "${summary_local}" "${summary_central}"
                n_done=$((n_done + 1))
                echo "[gpu ${gpu}]     summary → ${summary_central}"
            else
                echo "[gpu ${gpu}] [FAIL] ${baseline}/${dataset}/${protocol} — no summary"
                n_failed=$((n_failed + 1))
            fi
        else
            n_failed=$((n_failed + 1))
            echo "[gpu ${gpu}] [FAIL] ${baseline}/${dataset}/${protocol}"
        fi
    done

    echo ""
    echo "[gpu ${gpu}] worker done  ok=${n_done}  skip=${n_skipped}  fail=${n_failed}  /  total=${n}"
}

# -----------------------------------------------------------------------------
# Launch one worker per GPU; each writes to its own log.
# -----------------------------------------------------------------------------
pids=()
for gpu in "${GPU_LIST[@]}"; do
    dirs_str="${bucket[$gpu]:-}"
    if [[ -z "${dirs_str}" ]]; then
        continue
    fi
    log="${OUT_ROOT}/_worker_${gpu}.log"
    worker "${gpu}" "${dirs_str}" > "${log}" 2>&1 &
    pids+=($!)
    echo "[batch] launched worker pid=$! on GPU ${gpu} → ${log}"
done

# Wait for all workers; tally exit codes.
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=$((fail + 1))
    fi
done

echo ""
echo "============================================================"
if (( fail == 0 )); then
    echo "[batch] all workers finished cleanly."
else
    echo "[batch] ${fail} worker(s) reported a failure — see _worker_<gpu>.log."
fi
echo "[batch] central summaries under ${OUT_ROOT}/"
echo "============================================================"
