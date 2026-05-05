#!/bin/bash
# =============================================================================
# One-off top-up runner — single GPU, parallel processes.
#
# Walks every bucket already present under outputs/test_metric/metrics/,
# resolves it back to its inference run dir, and invokes
# compute_metrics.py --metrics auto so only the groups missing from the
# bucket's existing metrics_summary.json get computed.
#
# In practice that means the just-restored psnr / ssim / lpips groups
# are added to every same-identity bucket, while head_rot / expression /
# id are left alone. FVD is not part of this script — it lives in
# compute_fvd.py and runs separately.
#
# Concurrency: every process is pinned to the same GPU (default 0). The
# per-sample evaluator peaks at ~1 GB GPU memory with the new LPIPS
# AlexNet on top, so JOBS=4 is comfortable on a 16 GB card; bump to 6–8
# on 24 GB+. Tune with the JOBS env var.
#
# Excludes marionette_flame_vector_abl by default (override via
# EXCLUDE_BUCKETS).
#
# Usage (from repo root):
#
#     bash experiments/evaluation_metrics/_topup_missing.sh
#     JOBS=6  bash experiments/evaluation_metrics/_topup_missing.sh
#     GPU=1   bash experiments/evaluation_metrics/_topup_missing.sh
#     EXCLUDE_BUCKETS="marionette_flame_vector_abl marionette_no_deform_abl" \
#         bash experiments/evaluation_metrics/_topup_missing.sh
#
# Background + log:
#     bash experiments/evaluation_metrics/_topup_missing.sh \
#         > outputs/test_metric/metrics/_topup.log 2>&1 &
#     tail -f outputs/test_metric/metrics/_topup.log
# =============================================================================
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_activate.sh
source "${HERE}/_activate.sh"

JOBS="${JOBS:-4}"
GPU="${GPU:-0}"
METRICS="${METRICS:-auto}"
EXCLUDE_BUCKETS="${EXCLUDE_BUCKETS:-marionette_flame_vector_abl}"

OUT_ROOT="outputs/test_metric/metrics"
CLI="experiments/evaluation_metrics/compute_metrics.py"

# -----------------------------------------------------------------------------
# Map a metrics-tree bucket name back to the inference-output root that
# holds its run_<ts> dirs. Mirrors run_eval_metrics.sh's bucket naming.
# -----------------------------------------------------------------------------
bucket_to_runroot() {
    local b="$1"
    case "$b" in
        marionette)
            echo "outputs/marionette_eval"
            ;;
        marionette_*_abl)
            local arm="${b#marionette_}"
            arm="${arm%_abl}"
            echo "outputs/condition_ablation_eval/${arm}"
            ;;
        *)
            echo "outputs/sota_comparison/${b}"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Build the work queue: one entry per (bucket, protocol) pair.
# Picks the latest run_<ts> dir under the inference root if there are
# multiple (timestamps sort lexicographically).
# -----------------------------------------------------------------------------
WORK=()
SKIPPED=()

for bucket_dir in "${OUT_ROOT}"/*/; do
    [[ -d "$bucket_dir" ]] || continue
    bucket="$(basename "${bucket_dir%/}")"

    skip=false
    for ex in $EXCLUDE_BUCKETS; do
        [[ "$bucket" == "$ex" ]] && { skip=true; break; }
    done
    if $skip; then
        SKIPPED+=("$bucket  (excluded by EXCLUDE_BUCKETS)")
        continue
    fi

    run_root="$(bucket_to_runroot "$bucket")"
    if [[ ! -d "$run_root" ]]; then
        SKIPPED+=("$bucket  (inference root ${run_root} missing)")
        continue
    fi

    for proto_dir in "${bucket_dir}"hdtf/*/; do
        [[ -d "$proto_dir" ]] || continue
        protocol="$(basename "${proto_dir%/}")"

        # Latest run dir for this (bucket, protocol).
        run_dir="$(ls -1d "${run_root}/hdtf/${protocol}/run_"*/ 2>/dev/null \
                   | sort | tail -n1)"
        if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
            SKIPPED+=("$bucket/$protocol  (no run_<ts> under ${run_root}/hdtf/${protocol})")
            continue
        fi

        out_dir="${proto_dir%/}"
        WORK+=("${bucket}|hdtf|${protocol}|${run_dir%/}|${out_dir}")
    done
done

echo "============================================================"
echo "[topup] queued: ${#WORK[@]} bucket/protocol pairs"
echo "[topup] JOBS=${JOBS}  GPU=${GPU}  METRICS=${METRICS}"
echo "[topup] excluded: ${EXCLUDE_BUCKETS}"
if (( ${#SKIPPED[@]} > 0 )); then
    echo "[topup] skipped:"
    for s in "${SKIPPED[@]}"; do echo "          ${s}"; done
fi
echo "============================================================"

if (( ${#WORK[@]} == 0 )); then
    echo "[topup] nothing to do."
    exit 0
fi

# -----------------------------------------------------------------------------
# Single-GPU parallel dispatch. `wait -n` blocks until *any* child
# finishes, freeing a slot. No `set -e`, so a single failure doesn't
# tear the whole sweep down.
# -----------------------------------------------------------------------------
in_flight=0
declare -A pid_label
for item in "${WORK[@]}"; do
    IFS='|' read -r bucket dataset protocol run_dir out_dir <<< "$item"
    (
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" "$CLI" \
            --run-dir    "$run_dir" \
            --output-dir "$out_dir" \
            --metrics    "$METRICS" 2>&1 \
          | sed "s|^|[${bucket}/${protocol}] |"
    ) &
    pid=$!
    pid_label[$pid]="${bucket}/${protocol}"
    echo "[topup] launched pid=${pid} for ${bucket}/${protocol}"
    in_flight=$(( in_flight + 1 ))

    if (( in_flight >= JOBS )); then
        if wait -n; then
            :
        else
            echo "[topup] one job exited non-zero (continuing)"
        fi
        in_flight=$(( in_flight - 1 ))
    fi
done

# Drain remaining children.
while (( in_flight > 0 )); do
    if wait -n; then
        :
    else
        echo "[topup] one job exited non-zero (continuing)"
    fi
    in_flight=$(( in_flight - 1 ))
done

echo "============================================================"
echo "[topup] complete."
echo "============================================================"