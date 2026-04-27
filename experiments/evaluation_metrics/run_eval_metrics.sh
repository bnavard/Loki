#!/bin/bash
# =============================================================================
# Unified multi-GPU metrics runner — Marionette + every SOTA baseline in
# one parallel sweep.
#
#   outputs/test_metric/metrics/<bucket>/<dataset>/<protocol>/
#       ├── metrics.jsonl            (one row per sample)
#       └── metrics_summary.json     (aggregates + fvd)
#
# `<bucket>` = `marionette` for `outputs/marionette_eval/` runs, the
# baseline name for `outputs/sota_comparison/<baseline>/` runs. A single
# `outputs/test_metric/metrics/*/<dataset>/<protocol>/metrics_summary.json`
# glob picks up every model uniformly for the comparison table.
#
# Mode handling — by default (`METRICS=auto`), each invocation only
# computes metric groups whose headline value isn't already in the
# central summary, leaving everything else untouched. So this script
# can be re-run safely after adding a new metric (e.g. head_orientation):
# already-evaluated dirs only get the new group on top of cached
# per-sample numbers.
#
#   METRICS=auto       (default) — top up missing groups
#   METRICS=all                  — full overwrite
#   METRICS=head_orientation,fvd — recompute only those groups
#   FRESH=1                      — wipe summary + metrics.jsonl first,
#                                  then `auto` will compute everything
#
# Frame coverage: every metric is computed on the **first 16 frames**
# of each prediction, matching Marionette's `cfg.inference.n_frames=16`.
# SOTA generations longer than 16 frames are silently truncated; cdfvd's
# random-clip-sampling asymmetry is collapsed to "first 16" on both
# pred and GT.
#
# Per-GPU memory: ~600 MB (LPIPS AlexNet + InsightFace buffalo_l +
# MediaPipe + 6DRepNet). Comfortable on H200 / A6000 / 4090.
#
# Usage (from repo root):
#
#   bash experiments/evaluation_metrics/run_eval_metrics.sh
#   METRICS=head_orientation bash experiments/evaluation_metrics/run_eval_metrics.sh
#   METRICS=all              bash experiments/evaluation_metrics/run_eval_metrics.sh
#   FRESH=1                  bash experiments/evaluation_metrics/run_eval_metrics.sh
#   NUM_GPUS=4               bash experiments/evaluation_metrics/run_eval_metrics.sh
#   GPUS="0 2 4 6"           bash experiments/evaluation_metrics/run_eval_metrics.sh
#
# Background + log:
#   bash experiments/evaluation_metrics/run_eval_metrics.sh \
#       > outputs/test_metric/metrics/_batch.log 2>&1 &
#   tail -f outputs/test_metric/metrics/_batch.log
# =============================================================================

set -u
set -o pipefail

# shellcheck source=_activate.sh
source "$(dirname "${BASH_SOURCE[0]}")/_activate.sh"

OUT_ROOT="outputs/test_metric/metrics"
CLI="experiments/evaluation_metrics/compute_metrics.py"
MARIONETTE_ROOT="outputs/marionette_eval"
SOTA_ROOT="outputs/sota_comparison"

METRICS="${METRICS:-auto}"
FRESH="${FRESH:-0}"

mkdir -p "${OUT_ROOT}"

# Resolve GPU list. `GPUS=...` (space-separated indices) wins over
# `NUM_GPUS=N` (which expands to `0 1 ... N-1`). Default: 8 GPUs.
if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_LIST <<< "${GPUS}"
else
    NUM_GPUS="${NUM_GPUS:-8}"
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
fi
N_GPUS="${#GPU_LIST[@]}"

# -----------------------------------------------------------------------------
# Build the unified work queue. Each entry is "<bucket>|<run_dir>".
# -----------------------------------------------------------------------------
WORK=()
for d in "${MARIONETTE_ROOT}"/*/*/run_*/; do
    [[ -d "$d" ]] && WORK+=("marionette|${d}")
done
for d in "${SOTA_ROOT}"/*/*/*/run_*/; do
    [[ -d "$d" ]] || continue
    rel="${d#${SOTA_ROOT}/}"
    rel="${rel%/}"
    baseline="${rel%%/*}"
    WORK+=("${baseline}|${d}")
done
N_TOTAL="${#WORK[@]}"

echo "============================================================"
echo "[batch] ${N_TOTAL} run dirs across ${N_GPUS} GPU(s): ${GPU_LIST[*]}"
echo "[batch] METRICS=${METRICS}  FRESH=${FRESH}"
echo "============================================================"

if (( N_TOTAL == 0 )); then
    echo "[batch] no run dirs found under ${MARIONETTE_ROOT} or ${SOTA_ROOT}. exiting."
    exit 0
fi

# Round-robin assign to GPUs.
declare -A bucket
for ((i=0; i<N_TOTAL; i++)); do
    gpu="${GPU_LIST[$(( i % N_GPUS ))]}"
    bucket["$gpu"]+="${WORK[$i]}^"
done

# -----------------------------------------------------------------------------
# Worker — sequential within a single GPU. Idempotency lives entirely in
# `compute_metrics.py --metrics auto`: that side reads the existing
# summary and computes only missing groups, so the worker just hands it
# the run dir and the desired mode.
# -----------------------------------------------------------------------------
worker() {
    local gpu="$1"
    local items_str="$2"
    local n_done=0 n_failed=0

    IFS='^' read -r -a items <<< "${items_str%^}"
    local n="${#items[@]}"

    for item in "${items[@]}"; do
        local b="${item%%|*}"
        local run_dir="${item#*|}"

        # Both trees end in `.../<dataset>/<protocol>/run_<ts>/`.
        local rel="${run_dir%/}"
        local protocol; protocol="$(basename "$(dirname "${rel}")")"
        local dataset;  dataset="$(basename "$(dirname "$(dirname "${rel}")")")"

        local out_dir="${OUT_ROOT}/${b}/${dataset}/${protocol}"
        mkdir -p "${out_dir}"

        if [[ "${FRESH}" == "1" ]]; then
            rm -f  "${out_dir}/metrics_summary.json" "${out_dir}/metrics.jsonl"
            rm -rf "${out_dir}/_fvd"
        fi

        echo "[gpu ${gpu}] >>> ${b}/${dataset}/${protocol}  metrics=${METRICS}"
        if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. \
                "${PYTHON}" "${CLI}" \
                    --run-dir    "${run_dir}" \
                    --output-dir "${out_dir}" \
                    --metrics    "${METRICS}" \
                    2>&1 | tail -2; then
            n_done=$((n_done + 1))
        else
            n_failed=$((n_failed + 1))
            echo "[gpu ${gpu}] [FAIL] ${b}/${dataset}/${protocol}"
        fi
    done

    echo ""
    echo "[gpu ${gpu}] worker done  ok=${n_done}  fail=${n_failed}  /  total=${n}"
}

pids=()
for gpu in "${GPU_LIST[@]}"; do
    items_str="${bucket[$gpu]:-}"
    [[ -z "${items_str}" ]] && continue
    worker "${gpu}" "${items_str}" &
    pids+=($!)
    echo "[batch] launched worker pid=$! on GPU ${gpu}"
done

fail=0
for pid in "${pids[@]}"; do
    wait "${pid}" || fail=$((fail + 1))
done

echo ""
echo "============================================================"
if (( fail == 0 )); then
    echo "[batch] all workers finished cleanly."
else
    echo "[batch] ${fail} worker(s) reported a failure — re-scan stdout for [FAIL] lines."
fi
echo "[batch] central summaries under ${OUT_ROOT}/"
echo "============================================================"
