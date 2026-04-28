#!/bin/bash
# =============================================================================
# Sanity-check batch driver for the head-rotation visualizer.
#
# Picks N (default 5) sample IDs per `(dataset, protocol)` cell from the
# 6-way intersection of "samples Marionette + every SOTA baseline produced",
# so each chosen sample renders a comparable side-by-side overlay across
# every model. Default seed = 42.
#
# Per (baseline, sample) pair, calls
# `sanity_check/visualize_head_rot.py` and lands the output at:
#
#   outputs/test_metric/head_rot_sanity/<bucket>/<dataset>/<protocol>/<sample_id>/
#       ├── overlay.mp4    # pred ‖ target with FLAME axes drawn per frame
#       └── overlay.json   # per-frame geodesic distance + summary
#
# Round-robin across GPUs (default 8). Idempotent: any (sample, baseline)
# pair with an existing overlay.mp4 is skipped.
#
# Total work: 5 × 4 × 6 = 120 invocations. With model-load overhead per
# subprocess (~5–10 s each) and 8 workers in parallel, expect ~3–5 min
# total wall-clock.
#
# Usage (from repo root):
#
#   bash experiments/evaluation_metrics/sanity_check/visualize_head_rot_batch.sh
#
#   N_SAMPLES=10 bash experiments/evaluation_metrics/sanity_check/visualize_head_rot_batch.sh
#   SEED=7 NUM_GPUS=4 bash experiments/evaluation_metrics/sanity_check/visualize_head_rot_batch.sh
# =============================================================================

set -u
set -o pipefail

# shellcheck source=../_activate.sh
source "$(dirname "${BASH_SOURCE[0]}")/../_activate.sh"

N_SAMPLES="${N_SAMPLES:-5}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
VIZ="experiments/evaluation_metrics/sanity_check/visualize_head_rot.py"
OUT_ROOT="outputs/test_metric/head_rot_sanity"

# -----------------------------------------------------------------------------
# Build the work queue. Pure-python for set intersection + seeded sampling;
# bash for the GPU round-robin and subprocess fan-out.
# -----------------------------------------------------------------------------
mapfile -t WORK < <(
    PYTHONPATH=. "${PYTHON}" - <<PY
import random
from pathlib import Path

BASELINES = ["marionette", "anitalker", "echomimic", "hunyuan_portrait",
             "sadtalker", "xportrait"]
DATASETS  = ["talkvid", "hdtf"]
PROTOCOLS = ["same_identity_reconstruction", "cross_identity"]

def latest(p):
    runs = sorted([d for d in p.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None

def discover(ds, pr, b):
    root = (Path("outputs/marionette_eval") / ds / pr if b == "marionette"
            else Path("outputs/sota_comparison") / b / ds / pr)
    run = latest(root)
    if run is None:
        return set()
    sd = run / "samples"
    if not sd.is_dir():
        return set()
    return {d.name for d in sd.iterdir() if (d / "panel.mp4").is_file()}

rng = random.Random(${SEED})
for ds in DATASETS:
    for pr in PROTOCOLS:
        sets = [discover(ds, pr, b) for b in BASELINES]
        inter = sorted(set.intersection(*sets)) if all(sets) else []
        if not inter:
            continue
        picks = rng.sample(inter, min(${N_SAMPLES}, len(inter)))
        for sid in sorted(picks):
            for b in BASELINES:
                print(f"{b}|{ds}|{pr}|{sid}")
PY
)

N_WORK="${#WORK[@]}"
echo "============================================================"
echo "[batch] ${N_WORK} work units across ${NUM_GPUS} GPU(s)  (N_SAMPLES=${N_SAMPLES}  SEED=${SEED})"
echo "============================================================"
if (( N_WORK == 0 )); then
    echo "[batch] empty work queue — no 6-way-shared samples discovered. exiting."
    exit 0
fi

# Round-robin assign to GPUs.
declare -A BUCKET
for ((i=0; i<N_WORK; i++)); do
    gpu=$(( i % NUM_GPUS ))
    BUCKET["$gpu"]+="${WORK[$i]}^"
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
        IFS='|' read -r b ds pr sid <<< "$item"
        local out_dir="${OUT_ROOT}/${b}/${ds}/${pr}/${sid}"
        local out_mp4="${out_dir}/overlay.mp4"
        mkdir -p "${out_dir}"

        if [[ -f "${out_mp4}" ]]; then
            echo "[gpu ${gpu}] [skip] ${b}/${ds}/${pr}/${sid}"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        echo "[gpu ${gpu}] >>> ${b}/${ds}/${pr}/${sid}"
        if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. \
                "${PYTHON}" "${VIZ}" \
                    --baseline   "${b}" \
                    --dataset    "${ds}" \
                    --protocol   "${pr}" \
                    --sample-id  "${sid}" \
                    --out-mp4    "${out_mp4}" \
                    2>&1 | tail -1; then
            if [[ -f "${out_mp4}" ]]; then
                n_done=$((n_done + 1))
            else
                echo "[gpu ${gpu}] [FAIL] ${b}/${ds}/${pr}/${sid} — no mp4 written"
                n_failed=$((n_failed + 1))
            fi
        else
            n_failed=$((n_failed + 1))
            echo "[gpu ${gpu}] [FAIL] ${b}/${ds}/${pr}/${sid}"
        fi
    done

    echo ""
    echo "[gpu ${gpu}] worker done  ok=${n_done}  skip=${n_skipped}  fail=${n_failed}  /  total=${n}"
}

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    items_str="${BUCKET[$gpu]:-}"
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
    echo "[batch] ${fail} worker(s) reported a failure — see [FAIL] lines above."
fi
echo "[batch] outputs under ${OUT_ROOT}/"
echo "============================================================"
