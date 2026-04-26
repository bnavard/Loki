#!/bin/bash
# =============================================================================
# Batch driver for compute_metrics.py.
#
# Walks every run dir under
# `outputs/sota_comparison/<baseline>/<dataset>/<protocol>/run_*/`,
# runs `compute_metrics.py --run-dir <run_dir>` (with `--skip-fvd` for the
# first pass), and mirrors each `metrics_summary.json` into a centralized
# tree for at-a-glance comparison:
#
#   outputs/test_metric/metrics/<baseline>/<dataset>/<protocol>/
#       └── metrics_summary.json
#
# The per-sample `metrics.jsonl` and `metrics_summary.json` also stay at
# the original `<run_dir>/` so per-run drilldown is still trivial.
#
# Runtime: ~10–15 min per same-id run dir (LPIPS + MediaPipe per frame
# over ~125–212 samples), ~3–5 min per cross-id run dir (ArcFace per
# frame). With FVD off, the full 20-run sweep is about 2–3 hours.
#
# Usage (from repo root):
#
#   bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh
#
# Background + log tail:
#
#   bash experiments/evaluation_metrics/run_eval_metrics_on_sota.sh \
#       > outputs/test_metric/metrics/_batch.log 2>&1 &
#   tail -f outputs/test_metric/metrics/_batch.log
# =============================================================================

set -u
set -o pipefail

OUT_ROOT="outputs/test_metric/metrics"
PYTHON="/venv/evaluation_metrics/bin/python"
CLI="experiments/evaluation_metrics/compute_metrics.py"

mkdir -p "${OUT_ROOT}"

n_total=0
n_done=0
n_skipped=0
n_failed=0

for run_dir in outputs/sota_comparison/*/*/*/run_*/; do
    [[ -d "${run_dir}" ]] || continue

    rel="${run_dir#outputs/sota_comparison/}"
    rel="${rel%/}"
    IFS='/' read -r baseline dataset protocol _run <<< "${rel}"

    n_total=$((n_total + 1))
    summary_local="${run_dir}metrics_summary.json"
    summary_central="${OUT_ROOT}/${baseline}/${dataset}/${protocol}/metrics_summary.json"
    mkdir -p "$(dirname "${summary_central}")"

    if [[ -f "${summary_local}" ]] && [[ -f "${summary_central}" ]]; then
        echo "[skip] ${baseline}/${dataset}/${protocol}"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    echo ">>> ${baseline}/${dataset}/${protocol}"
    if PYTHONPATH=. "${PYTHON}" "${CLI}" --run-dir "${run_dir}" --skip-fvd \
            2>&1 | tail -3; then
        if [[ -f "${summary_local}" ]]; then
            cp "${summary_local}" "${summary_central}"
            n_done=$((n_done + 1))
            echo "    summary → ${summary_central}"
        else
            echo "[FAIL] ${baseline}/${dataset}/${protocol} — no summary written"
            n_failed=$((n_failed + 1))
        fi
    else
        n_failed=$((n_failed + 1))
        echo "[FAIL] ${baseline}/${dataset}/${protocol}"
    fi
done

echo ""
echo "============================================================"
echo "[batch] done  ${n_done}  skipped  ${n_skipped}  failed  ${n_failed}  /  total  ${n_total}"
echo "[batch] central summaries under ${OUT_ROOT}/"
echo "============================================================"
