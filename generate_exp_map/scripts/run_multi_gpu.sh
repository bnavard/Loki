#!/bin/bash
# =============================================================================
# Generate fit.npz for all videos (multi-GPU parallel)
#
# Runs both phases across multiple GPUs:
#   Phase 1: pixel3dmm FLAME tracking (parallel across GPUs)
#   Phase 2: FlowFace conversion (parallel across GPUs)
#
# Each phase resumes from where it left off — already completed videos
# are skipped automatically.
#
# Prerequisites:
#   - pixel3dmm installed and patched (see README.md)
#   - p3dmm conda environment activated
#   - PIXEL3DMM_CODE_BASE set
#
# Usage:
#   cd <repo_root>
#   bash generate_exp_map/scripts/run_multi_gpu.sh [data_dir] [output_dir] [num_gpus] [workers_per_gpu]
#
# Examples:
#   bash generate_exp_map/scripts/run_multi_gpu.sh
#   bash generate_exp_map/scripts/run_multi_gpu.sh data/talkvid/talkvid data/flowface 8 2
# =============================================================================

set -e

DATA_DIR="${1:-data/talkvid/talkvid}"
OUTPUT_DIR="${2:-data/flowface}"
NUM_GPUS="${3:-8}"
WORKERS_PER_GPU="${4:-2}"

# Auto-detect pixel3dmm if not set
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PIXEL3DMM_CODE_BASE="${PIXEL3DMM_CODE_BASE:-${SCRIPT_DIR}/../pixel3dmm}"
export PIXEL3DMM_PREPROCESSED_DATA="${PIXEL3DMM_PREPROCESSED_DATA:-data/flame_tracking/preprocessing}"
export PIXEL3DMM_TRACKING_OUTPUT="${PIXEL3DMM_TRACKING_OUTPUT:-data/flame_tracking/tracking}"

PREPROCESSING_DIR="${PIXEL3DMM_PREPROCESSED_DATA}"
TRACKING_DIR="${PIXEL3DMM_TRACKING_OUTPUT}"

echo "============================================================"
echo "Generating fit.npz for all videos"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "Workers/GPU: ${WORKERS_PER_GPU}"
echo "============================================================"
echo ""

# --- Phase 1: pixel3dmm FLAME tracking (parallel) ---
echo "========== Phase 1: pixel3dmm FLAME tracking =========="
PYTHONPATH=. python generate_exp_map/src/flame_tracking_parallel.py \
    --data_dirs "${DATA_DIR}" \
    --num_gpus "${NUM_GPUS}" \
    --workers_per_gpu "${WORKERS_PER_GPU}"

echo ""
echo "Phase 1 complete."
echo ""

# --- Phase 2: FlowFace conversion (parallel) ---
# Build GPU list: 0 1 2 ... (NUM_GPUS-1)
GPU_LIST=$(seq 0 $((NUM_GPUS - 1)))

echo "========== Phase 2: FlowFace conversion =========="
PYTHONPATH=. python generate_exp_map/src/convert_to_flowface_parallel.py \
    --preprocessing_dir "${PREPROCESSING_DIR}" \
    --tracking_dir "${TRACKING_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpus ${GPU_LIST}

echo ""
echo "============================================================"
echo "Done. fit.npz files saved to ${OUTPUT_DIR}/"
echo "============================================================"
