#!/bin/bash
# =============================================================================
# Unified multi-GPU metrics runner — Loki + every SOTA baseline in
# one parallel sweep.
#
#   outputs/test_metric/metrics/<bucket>/<dataset>/<protocol>/
#       ├── metrics.jsonl            (one row per sample)
#       └── metrics_summary.json     (aggregates)
#
# `<bucket>` = `loki` for `outputs/loki_eval/` runs, the
# baseline name for `outputs/sota_comparison/<baseline>/` runs, and
# `loki_<arm>_abl` for `outputs/condition_ablation_eval/<arm>/`
# runs (so ablations group with Loki and can't be confused with a
# SOTA baseline). A single
# `outputs/test_metric/metrics/*/<dataset>/<protocol>/metrics_summary.json`
# glob picks up every model uniformly for the comparison table.
#
# Mode handling — by default (`METRICS=auto`), each invocation only
# computes metric groups whose headline value isn't already in the
# central summary, leaving everything else untouched. So this script
# can be re-run safely after adding a new metric group: already-evaluated
# dirs only get the new group on top of cached per-sample numbers.
#
#   METRICS=auto                       (default) — top up missing groups
#   METRICS=all                                  — full overwrite
#   METRICS=head_rot                             — recompute only that group
#   FRESH=1                                      — wipe summary + metrics.jsonl
#                                                  first, then `auto` recomputes
#
# Frame coverage: every metric is computed on the **first 16 frames**
# of each prediction, matching Loki's `cfg.inference.n_frames=16`.
# SOTA generations longer than 16 frames are silently truncated.
#
# Per-GPU memory: ~600 MB (InsightFace buffalo_l + FLAME skinner +
# pytorch3d rasterizer for the FLAME-derived metrics). Comfortable on
# H200 / A6000 / 4090.
#
# Parallelism: every run dir launches as its own background process,
# pinned to a GPU via `CUDA_VISIBLE_DEVICES = (item_index mod N_GPUS)`.
# All processes run concurrently — no inner sequential loop. With
# 12 HDTF run dirs and `NUM_GPUS=6` (default), each GPU hosts exactly
# 2 concurrent processes (peak ≈ 1.2 GB GPU memory).
#
# Usage (from repo root):
#
#   bash experiments/evaluation_metrics/run_eval_metrics.sh
#   METRICS=head_rot         bash experiments/evaluation_metrics/run_eval_metrics.sh
#   METRICS=all              bash experiments/evaluation_metrics/run_eval_metrics.sh
#   FRESH=1                  bash experiments/evaluation_metrics/run_eval_metrics.sh
#   NUM_GPUS=12              bash experiments/evaluation_metrics/run_eval_metrics.sh
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
LOKI_ROOT="outputs/loki_eval"
SOTA_ROOT="outputs/sota_comparison"
ABLATION_ROOT="outputs/condition_ablation_eval"

METRICS="${METRICS:-auto}"
FRESH="${FRESH:-0}"

mkdir -p "${OUT_ROOT}"

# Resolve GPU list. `GPUS=...` (space-separated indices) wins over
# `NUM_GPUS=N` (which expands to `0 1 ... N-1`). Default: 6 GPUs —
# the only round-robin split that lands every GPU with exactly 2 of
# the 12 HDTF run dirs.
if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_LIST <<< "${GPUS}"
else
    NUM_GPUS="${NUM_GPUS:-6}"
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
fi
N_GPUS="${#GPU_LIST[@]}"

# -----------------------------------------------------------------------------
# Build the unified work queue. Each entry is "<bucket>|<run_dir>".
# -----------------------------------------------------------------------------
WORK=()
for d in "${LOKI_ROOT}"/hdtf/*/run_*/; do
    [[ -d "$d" ]] && WORK+=("loki|${d}")
done
for d in "${SOTA_ROOT}"/*/hdtf/*/run_*/; do
    [[ -d "$d" ]] || continue
    rel="${d#${SOTA_ROOT}/}"
    rel="${rel%/}"
    baseline="${rel%%/*}"
    WORK+=("${baseline}|${d}")
done
# Condition-ablation runs share the SOTA-style `<arm>/<dataset>/<protocol>/run_<ts>/`
# layout. The bucket is `loki_<arm>_abl` so ablation results group
# next to `loki` in the central summary tree and can't be confused
# with a SOTA baseline.
for d in "${ABLATION_ROOT}"/*/hdtf/*/run_*/; do
    [[ -d "$d" ]] || continue
    rel="${d#${ABLATION_ROOT}/}"
    rel="${rel%/}"
    arm="${rel%%/*}"
    WORK+=("loki_${arm}_abl|${d}")
done
N_TOTAL="${#WORK[@]}"

echo "============================================================"
echo "[batch] ${N_TOTAL} run dirs across ${N_GPUS} GPU(s): ${GPU_LIST[*]}"
echo "[batch] METRICS=${METRICS}  FRESH=${FRESH}"
echo "============================================================"

if (( N_TOTAL == 0 )); then
    echo "[batch] no run dirs found under ${LOKI_ROOT}, ${SOTA_ROOT}, or ${ABLATION_ROOT}. exiting."
    exit 0
fi

# Idempotency lives entirely in `compute_metrics.py --metrics auto`:
# that side reads the existing summary and computes only missing groups,
# so each task just hands it the run dir and the desired mode.
run_one() {
    local gpu="$1"
    local item="$2"
    local b="${item%%|*}"
    local run_dir="${item#*|}"

    # Both trees end in `.../<dataset>/<protocol>/run_<ts>/`.
    local rel="${run_dir%/}"
    local protocol; protocol="$(basename "$(dirname "${rel}")")"
    local dataset;  dataset="$(basename "$(dirname "$(dirname "${rel}")")")"

    local out_dir="${OUT_ROOT}/${b}/${dataset}/${protocol}"
    mkdir -p "${out_dir}"
    if [[ "${FRESH}" == "1" ]]; then
        rm -f "${out_dir}/metrics_summary.json" "${out_dir}/metrics.jsonl"
    fi

    echo "[gpu ${gpu}] >>> ${b}/${dataset}/${protocol}  metrics=${METRICS}"
    if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. \
            "${PYTHON}" "${CLI}" \
                --run-dir    "${run_dir}" \
                --output-dir "${out_dir}" \
                --metrics    "${METRICS}" \
                2>&1 | tail -2; then
        echo "[gpu ${gpu}] [OK]   ${b}/${dataset}/${protocol}"
    else
        echo "[gpu ${gpu}] [FAIL] ${b}/${dataset}/${protocol}"
        return 1
    fi
}

# Spawn one background process per run dir; round-robin pin to GPUs.
pids=()
for ((i=0; i<N_TOTAL; i++)); do
    gpu="${GPU_LIST[$(( i % N_GPUS ))]}"
    run_one "${gpu}" "${WORK[$i]}" &
    pids+=($!)
    echo "[batch] launched pid=$! on GPU ${gpu} for ${WORK[$i]%%|*}"
done

fail=0
for pid in "${pids[@]}"; do
    wait "${pid}" || fail=$((fail + 1))
done

echo ""
echo "============================================================"
if (( fail == 0 )); then
    echo "[batch] all ${N_TOTAL} run dirs finished cleanly."
else
    echo "[batch] ${fail} run(s) failed — re-scan stdout for [FAIL] lines."
fi
echo "[batch] central summaries under ${OUT_ROOT}/"
echo "============================================================"
