#!/bin/bash
# =============================================================================
# Run pixel3dmm FLAME tracking on every generated panel.mp4 produced by
# `run_eval_inference.sh`, writing the per-sample `fit.npz` into the
# bucket-aware predictions tree the metrics evaluator expects:
#
#     data/flame_tracking/preds/<arm>/hdtf/<protocol>/<sample_id>/fit.npz
#
# For each (arm, protocol), we:
#   1. Locate the latest `outputs/condition_ablation_eval/<arm>/hdtf/<protocol>/run_*/`
#   2. Stage symlinks `<sample_id>.mp4 -> <run_dir>/samples/<sample_id>/panel.mp4`
#      under a per-(arm, protocol) staging dir so `run_multi_gpu.sh`'s flat-dir
#      input contract is satisfied. Resuming is automatic — pixel3dmm skips
#      videos whose tracking output already exists.
#   3. Shell out to `generate_exp_map/scripts/run_multi_gpu.sh` against the
#      staging dir, with the bucket-aware output dir.
#
# Conda env: `expmapgen` (pixel3dmm + nvdiffrast). Activate before running.
#
# Usage (from repo root):
#
#     conda activate expmapgen
#     bash experiments/condition_ablation/track_ablation_panels.sh
# 
#
# Override-able via env vars:
#     ARMS="no_posenc no_deform"           # subset of arms
#     PROTOCOLS="cross_identity"           # subset of protocols
#     NUM_GPUS=1                           # single H100 default
#     WORKERS_PER_GPU=2
#     RUN_TS=20260502_120000               # pin to a specific run timestamp
#                                          # (default: latest run_* dir)
# =============================================================================

set -u
set -o pipefail

ARMS="${ARMS:-no_posenc no_deform flame_vector}"
PROTOCOLS="${PROTOCOLS:-same_identity_reconstruction cross_identity}"
NUM_GPUS="${NUM_GPUS:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
RUN_TS="${RUN_TS:-}"

ABLATION_ROOT="outputs/condition_ablation_eval"
PREDS_ROOT="data/flame_tracking/preds"
STAGE_ROOT="data/flame_tracking/_ablation_staging"
TRACKER="generate_exp_map/scripts/run_multi_gpu.sh"

mkdir -p "${STAGE_ROOT}"

echo "============================================================"
echo "[track-panels] arms:       ${ARMS}"
echo "[track-panels] protocols:  ${PROTOCOLS}"
echo "[track-panels] gpus:       ${NUM_GPUS}  (${WORKERS_PER_GPU} workers/gpu)"
echo "[track-panels] run_ts:     ${RUN_TS:-<latest>}"
echo "============================================================"

# Resolve the run dir to track for a given (arm, protocol). When RUN_TS
# is unset, prefer the lexicographically-latest `run_*` dir — matches
# the timestamp ordering produced by run_inference.py.
resolve_run_dir() {
    local arm="$1"
    local proto="$2"
    local base="${ABLATION_ROOT}/${arm}/hdtf/${proto}"
    if [[ ! -d "${base}" ]]; then
        return 1
    fi
    if [[ -n "${RUN_TS}" ]]; then
        local pinned="${base}/run_${RUN_TS}"
        [[ -d "${pinned}" ]] || return 1
        echo "${pinned}"
        return 0
    fi
    # shellcheck disable=SC2012
    local latest
    latest="$(ls -1d "${base}"/run_* 2>/dev/null | sort | tail -n 1)"
    [[ -n "${latest}" && -d "${latest}" ]] || return 1
    echo "${latest}"
}

fail=0
for arm in ${ARMS}; do
    for proto in ${PROTOCOLS}; do
        echo ""
        echo "[track-panels] >>> arm=${arm}  protocol=${proto}"

        run_dir="$(resolve_run_dir "${arm}" "${proto}" || true)"
        if [[ -z "${run_dir}" ]]; then
            echo "[track-panels] [SKIP] no run dir under ${ABLATION_ROOT}/${arm}/hdtf/${proto}/"
            continue
        fi

        samples_dir="${run_dir}/samples"
        if [[ ! -d "${samples_dir}" ]]; then
            echo "[track-panels] [SKIP] no samples/ under ${run_dir}"
            continue
        fi

        stage_dir="${STAGE_ROOT}/${arm}/hdtf/${proto}"
        rm -rf "${stage_dir}"
        mkdir -p "${stage_dir}"

        n_staged=0
        for sample_path in "${samples_dir}"/*/; do
            [[ -d "${sample_path}" ]] || continue
            sid="$(basename "${sample_path}")"
            panel="${sample_path}panel.mp4"
            if [[ ! -f "${panel}" ]]; then
                continue
            fi
            ln -s "$(readlink -f "${panel}")" "${stage_dir}/${sid}.mp4"
            n_staged=$((n_staged + 1))
        done
        echo "[track-panels] staged ${n_staged} panels at ${stage_dir}"

        if (( n_staged == 0 )); then
            echo "[track-panels] [SKIP] no panels to track for ${arm}/${proto}"
            continue
        fi

        out_dir="${PREDS_ROOT}/${arm}/hdtf/${proto}"
        mkdir -p "${out_dir}"

        echo "[track-panels] tracking → ${out_dir}"
        if bash "${TRACKER}" "${stage_dir}" "${out_dir}" "${NUM_GPUS}" "${WORKERS_PER_GPU}"; then
            echo "[track-panels] [OK]   arm=${arm}  protocol=${proto}"
        else
            echo "[track-panels] [FAIL] arm=${arm}  protocol=${proto}"
            fail=$((fail + 1))
        fi
    done
done

echo ""
echo "============================================================"
if (( fail == 0 )); then
    echo "[track-panels] all (arm, protocol) pairs finished cleanly."
else
    echo "[track-panels] ${fail} pair(s) failed — re-scan stdout for [FAIL]."
fi
echo "[track-panels] preds under ${PREDS_ROOT}/<arm>/hdtf/<protocol>/"
echo "============================================================"
exit "${fail}"
