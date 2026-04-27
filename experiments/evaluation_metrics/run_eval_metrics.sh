#!/bin/bash
# =============================================================================
# Unified multi-GPU metrics runner — Marionette + every SOTA baseline in one
# parallel sweep, results centralized at:
#
#   outputs/test_metric/metrics/<bucket>/<dataset>/<protocol>/
#       ├── metrics.jsonl
#       └── metrics_summary.json
#
# `<bucket>` = `marionette` for `outputs/marionette_eval/...` runs,
#              `<baseline>` (sadtalker / anitalker / …) for SOTA runs.
# A single `outputs/test_metric/metrics/*/<dataset>/<protocol>/metrics_summary.json`
# glob picks up every model uniformly for the comparison table.
#
# Frame coverage: every metric — PSNR / SSIM / LPIPS / LMD-F / LMD-M /
# id_cosine / FVD — is computed on the **first 16 frames** of each
# prediction, matching Marionette's panel length (`cfg.inference.n_frames=16`).
# SOTA generations longer than 16 frames are silently truncated; cdfvd's
# random-clip-sampling asymmetry is collapsed to "first 16" on both pred
# and GT.
#
# Per-GPU memory: ~600 MB (LPIPS AlexNet + InsightFace buffalo_l +
# MediaPipe). Comfortable on H200 / A6000 / 4090.
#
# Skip semantics:
#   default (WITH_FVD=0):
#     - skip a dir if its central `metrics_summary.json` already exists
#   WITH_FVD=1:
#     - skip if the summary already has an `fvd` field
#     - otherwise pass `--fvd-only` to compute_metrics.py so only FVD is
#       added on top of the cached per-sample numbers
#   FRESH=1:
#     - ignore all caches; delete the central summary + metrics.jsonl +
#       _fvd staging tree before running
#
# Usage (from repo root):
#
#   bash experiments/evaluation_metrics/run_eval_metrics.sh
#   WITH_FVD=1 bash experiments/evaluation_metrics/run_eval_metrics.sh
#   FRESH=1    bash experiments/evaluation_metrics/run_eval_metrics.sh
#   NUM_GPUS=4 bash experiments/evaluation_metrics/run_eval_metrics.sh
#   GPUS="0 2 4 6" bash experiments/evaluation_metrics/run_eval_metrics.sh
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

WITH_FVD="${WITH_FVD:-0}"
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
# Marionette is one bucket; each SOTA baseline is its own bucket.
# Both trees share the same `<dataset>/<protocol>/run_<ts>/` tail so
# the worker parses both uniformly.
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
echo "[batch] WITH_FVD=${WITH_FVD}  FRESH=${FRESH}"
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
# Worker — sequential within a single GPU.
# -----------------------------------------------------------------------------
worker() {
    local gpu="$1"
    local items_str="$2"
    local n_done=0 n_skipped=0 n_failed=0

    IFS='^' read -r -a items <<< "${items_str%^}"
    local n="${#items[@]}"

    for item in "${items[@]}"; do
        local b="${item%%|*}"
        local run_dir="${item#*|}"

        # Both trees end in `.../<dataset>/<protocol>/run_<ts>/`, so the
        # same parsing works for Marionette and SOTA buckets.
        local rel="${run_dir%/}"
        local protocol; protocol="$(basename "$(dirname "${rel}")")"
        local dataset;  dataset="$(basename "$(dirname "$(dirname "${rel}")")")"

        local out_dir="${OUT_ROOT}/${b}/${dataset}/${protocol}"
        local summary="${out_dir}/metrics_summary.json"
        mkdir -p "${out_dir}"

        if [[ "${FRESH}" == "1" ]]; then
            rm -f  "${summary}" "${out_dir}/metrics.jsonl"
            rm -rf "${out_dir}/_fvd"
        fi

        local has_fvd=0
        if [[ -f "${summary}" ]] && grep -q '"fvd":' "${summary}"; then
            has_fvd=1
        fi

        # Decide skip / mode. FVD is N/A for cross-identity (the
        # evaluator unconditionally skips it there), so treat any
        # cross-id summary as complete regardless of WITH_FVD —
        # otherwise the runner would loop trying to add an fvd field
        # that will never appear.
        if [[ -f "${summary}" ]]; then
            if [[ "${WITH_FVD}" != "1" ]] \
               || [[ "${protocol}" == "cross_identity" ]] \
               || (( has_fvd )); then
                echo "[gpu ${gpu}] [skip] ${b}/${dataset}/${protocol}"
                n_skipped=$((n_skipped + 1))
                continue
            fi
        fi

        local extra_flags=()
        if [[ "${WITH_FVD}" != "1" ]]; then
            extra_flags+=("--skip-fvd")
        elif [[ -f "${summary}" ]]; then
            extra_flags+=("--fvd-only")
        fi

        echo "[gpu ${gpu}] >>> ${b}/${dataset}/${protocol}  ${extra_flags[*]}"
        if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. \
                "${PYTHON}" "${CLI}" \
                    --run-dir    "${run_dir}" \
                    --output-dir "${out_dir}" \
                    "${extra_flags[@]}" \
                    2>&1 | tail -3; then
            if [[ -f "${summary}" ]]; then
                n_done=$((n_done + 1))
                echo "[gpu ${gpu}]     summary → ${summary}"
            else
                echo "[gpu ${gpu}] [FAIL] ${b}/${dataset}/${protocol} — no summary"
                n_failed=$((n_failed + 1))
            fi
        else
            n_failed=$((n_failed + 1))
            echo "[gpu ${gpu}] [FAIL] ${b}/${dataset}/${protocol}"
        fi
    done

    echo ""
    echo "[gpu ${gpu}] worker done  ok=${n_done}  skip=${n_skipped}  fail=${n_failed}  /  total=${n}"
}

# -----------------------------------------------------------------------------
# Launch one worker per GPU.
# -----------------------------------------------------------------------------
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
