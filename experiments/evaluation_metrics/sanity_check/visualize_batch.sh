#!/bin/bash
# =============================================================================
# Batch driver for visualize_sample.py.
#
# Walks every run dir under `outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_*/`
# and writes per-sample overlay mp4 + curves PNG (same-id only) for N
# evenly-spread sample IDs into a centralized tree:
#
#   outputs/test_metric/visualizations/<baseline>/<dataset>/<protocol>/<sample_id>/
#       ├── metrics_overlay.mp4
#       └── metrics_curves.png   # same-identity only
#
# Picks first / middle / last sample IDs (alphabetically) for visual
# diversity. Existing outputs are skipped — re-runs only fill the gaps.
#
# Runtime: ~30 min for the full 5 baselines × 2 datasets × 2 protocols ×
# 3 samples = 60 invocations; each visualize_sample.py call reloads
# InsightFace + LPIPS + MediaPipe (~15 s startup) plus per-sample work.
#
# Usage (from repo root, env doesn't need to be active — script uses
# the venv's python directly):
#
#   bash experiments/evaluation_metrics/sanity_check/visualize_batch.sh
#
# To run in background and tail progress:
#
#   bash experiments/evaluation_metrics/sanity_check/visualize_batch.sh \
#       > outputs/test_metric/visualizations/_batch.log 2>&1 &
#   tail -f outputs/test_metric/visualizations/_batch.log
# =============================================================================

set -u
set -o pipefail

OUT_ROOT="outputs/test_metric/visualizations"
N_PER_RUN=3
PYTHON="/venv/evaluation_metrics/bin/python"
VIZ="experiments/evaluation_metrics/sanity_check/visualize_sample.py"

mkdir -p "${OUT_ROOT}"

n_total=0
n_done=0
n_skipped=0
n_failed=0

for run_dir in outputs/sota_comparison/*/*/*/run_*/; do
    [[ -d "${run_dir}" ]] || continue

    # outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_<ts>
    rel="${run_dir#outputs/sota_comparison/}"
    rel="${rel%/}"
    IFS='/' read -r baseline dataset protocol _run <<< "${rel}"

    if [[ ! -d "${run_dir}samples" ]]; then
        continue
    fi
    mapfile -t samples < <(ls "${run_dir}samples/" | sort)
    n="${#samples[@]}"
    if [[ "${n}" -eq 0 ]]; then
        continue
    fi

    # Spread N picks evenly over the sample list.
    if [[ "${n}" -le "${N_PER_RUN}" ]]; then
        indices=(); for ((i=0; i<n; i++)); do indices+=("$i"); done
    else
        indices=(0 $((n / 2)) $((n - 1)))
    fi

    for i in "${indices[@]}"; do
        sid="${samples[$i]}"
        dst="${OUT_ROOT}/${baseline}/${dataset}/${protocol}/${sid}"
        mkdir -p "${dst}"
        n_total=$((n_total + 1))

        # Skip if the canonical artifact already exists. PNG is only
        # produced for same-id; cross-id only writes the mp4.
        if [[ -f "${dst}/metrics_overlay.mp4" ]]; then
            if [[ "${protocol}" == "cross_identity" ]] || [[ -f "${dst}/metrics_curves.png" ]]; then
                n_skipped=$((n_skipped + 1))
                echo "[skip] ${baseline}/${dataset}/${protocol}/${sid}"
                continue
            fi
        fi

        echo ">>> ${baseline}/${dataset}/${protocol}/${sid}"
        if PYTHONPATH=. "${PYTHON}" "${VIZ}" \
                --run-dir   "${run_dir}" \
                --sample-id "${sid}" \
                --out-mp4   "${dst}/metrics_overlay.mp4" \
                --out-png   "${dst}/metrics_curves.png" \
                2>&1 | tail -2; then
            n_done=$((n_done + 1))
        else
            n_failed=$((n_failed + 1))
            echo "[FAIL] ${baseline}/${dataset}/${protocol}/${sid}"
        fi
    done
done

echo ""
echo "============================================================"
echo "[batch] done  ${n_done}  skipped  ${n_skipped}  failed  ${n_failed}  /  total  ${n_total}"
echo "[batch] outputs under ${OUT_ROOT}/"
echo "============================================================"
