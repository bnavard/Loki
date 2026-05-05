#!/bin/bash
# =============================================================================
# Sequential inference sweep for the three condition_ablation arms.
#
# 3 arms (no_posenc, no_deform, flame_vector) × 2 protocols
# (same_identity_reconstruction, cross_identity) = 6 inference runs against
# the curated HDTF manifest.
#
# Sequential by design — single H100 in this environment. Each arm reuses
# the same `experiments/marionette_eval/run_inference.py` runner against
# its own per-arm `eval.yaml`. Output goes to
# `outputs/condition_ablation_eval/<arm>/hdtf/<protocol>/run_<ts>/`,
# matching the layout the metrics evaluator's `_derive_bucket` recognises.
#
# Conda env: `marionette` (the inference path is local, in-process).
#
# Usage (from repo root):
#
#     conda activate marionette
#     bash experiments/condition_ablation/run_eval_inference.sh
#
# Background + log:
#     bash experiments/condition_ablation/run_eval_inference.sh \
#         > outputs/condition_ablation_eval/_sweep.log 2>&1 &
#     tail -f outputs/condition_ablation_eval/_sweep.log
#
# Override-able via env vars:
#     ARMS="no_posenc no_deform"           # subset of arms
#     PROTOCOLS="cross_identity"           # subset of protocols
#     N_SAMPLES=212                        # passes through to run_inference
#     CLIP_DURATION_S=3.0
#     SEED=42
#     GPU=0                                # CUDA_VISIBLE_DEVICES
#     EXTRA_ARGS="--n_take 4"              # debug / smoke test
#     RUN_DIR=outputs/condition_ablation_eval/flame_vector/hdtf/cross_identity/run_20260503_040114
#         # resume an existing run dir — overrides the auto-timestamped path
#         # so already-written `samples/<sid>/panel.mp4` files are kept and the
#         # runner only generates the remainder. Only valid when ARMS and
#         # PROTOCOLS resolve to a single (arm, protocol) pair.
#     RESUME=1                             # default 1 — for each (arm, protocol)
#         # without an explicit RUN_DIR, auto-resume into the latest
#         # `run_*` dir under that bucket if one exists. RESUME=0 forces
#         # every (arm, protocol) to start a fresh timestamped run.
# =============================================================================

set -u
set -o pipefail

ARMS="${ARMS:-no_posenc no_deform flame_vector}"
PROTOCOLS="${PROTOCOLS:-same_identity_reconstruction cross_identity}"
N_SAMPLES="${N_SAMPLES:-212}"
CLIP_DURATION_S="${CLIP_DURATION_S:-3.0}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_DIR="${RUN_DIR:-}"
RESUME="${RESUME:-1}"

# Guard rail: RUN_DIR is a single directory — using it across multiple
# (arm, protocol) pairs would funnel every run into the same dir and is
# almost certainly a mistake. Fail loudly rather than silently corrupting.
if [[ -n "${RUN_DIR}" ]]; then
    n_arms=$(printf '%s\n' ${ARMS} | wc -l)
    n_protos=$(printf '%s\n' ${PROTOCOLS} | wc -l)
    if (( n_arms != 1 || n_protos != 1 )); then
        echo "[ablation-eval] [ERROR] RUN_DIR set but ARMS=${ARMS} × PROTOCOLS=${PROTOCOLS} resolves to ${n_arms}×${n_protos} pairs." >&2
        echo "[ablation-eval]         RUN_DIR is for resuming a single (arm, protocol). Restrict ARMS/PROTOCOLS or unset RUN_DIR." >&2
        exit 2
    fi
fi

ROOT="experiments/condition_ablation"
RUNNER="experiments/marionette_eval/run_inference.py"

mkdir -p outputs/condition_ablation_eval

echo "============================================================"
echo "[ablation-eval] arms:       ${ARMS}"
echo "[ablation-eval] protocols:  ${PROTOCOLS}"
echo "[ablation-eval] n_samples:  ${N_SAMPLES}"
echo "[ablation-eval] gpu:        ${GPU}"
echo "[ablation-eval] extra:      ${EXTRA_ARGS:-<none>}"
echo "============================================================"

fail=0
for arm in ${ARMS}; do
    cfg="${ROOT}/${arm}/eval.yaml"
    if [[ ! -f "${cfg}" ]]; then
        echo "[ablation-eval] [SKIP] no eval.yaml for arm=${arm} (${cfg})"
        continue
    fi
    for proto in ${PROTOCOLS}; do
        echo ""
        echo "[ablation-eval] >>> arm=${arm}  protocol=${proto}"
        # Resolve which run dir to write into:
        #   1. global RUN_DIR (single-pair override) wins
        #   2. else, if RESUME=1, the latest run_* under this (arm, protocol)
        #   3. else, leave unset → runner creates a fresh timestamped dir
        # Panel-level skip is enforced by run_inference.py via panel.mp4
        # presence; rng draws stay aligned because the runner advances
        # per-sample even on skip.
        pair_run_dir=""
        if [[ -n "${RUN_DIR}" ]]; then
            pair_run_dir="${RUN_DIR}"
        elif [[ "${RESUME}" == "1" ]]; then
            bucket_base="outputs/condition_ablation_eval/${arm}/hdtf/${proto}"
            # shellcheck disable=SC2012
            pair_run_dir="$(ls -1d ${bucket_base}/run_* 2>/dev/null | sort | tail -1)"
        fi

        out_args=()
        if [[ -n "${pair_run_dir}" ]]; then
            out_args+=(--output_dir "${pair_run_dir}")
            echo "[ablation-eval]      resuming into ${pair_run_dir}"
        else
            echo "[ablation-eval]      starting fresh run_<ts>/"
        fi

        if CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python "${RUNNER}" \
                --protocol "${proto}" \
                --config "${cfg}" \
                --n_samples "${N_SAMPLES}" \
                --clip_duration_s "${CLIP_DURATION_S}" \
                --seed "${SEED}" \
                "${out_args[@]}" \
                ${EXTRA_ARGS}; then
            echo "[ablation-eval] [OK]   arm=${arm}  protocol=${proto}"
        else
            echo "[ablation-eval] [FAIL] arm=${arm}  protocol=${proto}"
            fail=$((fail + 1))
        fi
    done
done

echo ""
echo "============================================================"
if (( fail == 0 )); then
    echo "[ablation-eval] all runs finished cleanly."
else
    echo "[ablation-eval] ${fail} run(s) failed — re-scan stdout for [FAIL]."
fi
echo "[ablation-eval] outputs under outputs/condition_ablation_eval/"
echo "============================================================"
exit "${fail}"
