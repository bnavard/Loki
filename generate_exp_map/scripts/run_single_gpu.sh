#!/bin/bash
# =============================================================================
# Generate fit.npz for a single video (single GPU)
#
# Runs both phases:
#   Phase 1: pixel3dmm FLAME tracking (preprocessing → normals → UV → tracking)
#   Phase 2: FlowFace conversion (FLAME re-fitting + gaze + background matting)
#
# Prerequisites:
#   - pixel3dmm installed and patched (see README.md)
#   - p3dmm conda environment activated
#   - PIXEL3DMM_CODE_BASE set
#
# Usage:
#   cd <repo_root>
#   bash generate_exp_map/scripts/run_single_gpu.sh <video_path> [output_dir] [gpu_id]
#
# Examples:
#   bash generate_exp_map/scripts/run_single_gpu.sh data/talkvid/talkvid/my_video.mp4
#   bash generate_exp_map/scripts/run_single_gpu.sh data/talkvid/talkvid/my_video.mp4 data/flowface 0
# =============================================================================

set -e

VIDEO_PATH="${1:?Usage: $0 <video_path> [output_dir] [gpu_id]}"
OUTPUT_DIR="${2:-data/flowface}"
GPU_ID="${3:-0}"

# Resolve video name (strip extension)
VIDEO_NAME=$(basename "${VIDEO_PATH}" .mp4)

# Default intermediate directories
PREPROCESSING_DIR="${PIXEL3DMM_PREPROCESSED_DATA:-outputs/flame_tracking/preprocessing}"
TRACKING_DIR="${PIXEL3DMM_TRACKING_OUTPUT:-outputs/flame_tracking/tracking}"
TRACKING_SUFFIX="_nV1_noPho_uv2000.0_n1000.0"

echo "============================================================"
echo "Generating fit.npz for: ${VIDEO_NAME}"
echo "GPU: ${GPU_ID}"
echo "Video: ${VIDEO_PATH}"
echo "Output: ${OUTPUT_DIR}/${VIDEO_NAME}/"
echo "============================================================"
echo ""

# --- Phase 1: pixel3dmm FLAME tracking ---
echo "[Phase 1] pixel3dmm FLAME tracking..."
export CUDA_VISIBLE_DEVICES=${GPU_ID}
PYTHONPATH=. python generate_exp_map/src/flame_tracking.py "${VIDEO_PATH}" --no-log

echo ""
echo "[Phase 1] Complete."
echo ""

# --- Phase 2: FlowFace conversion ---
TRACKING_PATH="${TRACKING_DIR}/${VIDEO_NAME}${TRACKING_SUFFIX}"
PREPROCESS_PATH="${PREPROCESSING_DIR}/${VIDEO_NAME}"
RGB_PATH="${PREPROCESS_PATH}/rgb"
FLOWFACE_OUTPUT="${OUTPUT_DIR}/${VIDEO_NAME}"

if [ ! -d "${TRACKING_PATH}/checkpoint" ]; then
    echo "ERROR: Tracking output not found at ${TRACKING_PATH}/checkpoint"
    echo "Phase 1 may have failed. Check logs."
    exit 1
fi

echo "[Phase 2] Converting to FlowFace format (fit.npz)..."
PYTHONPATH=. python generate_exp_map/src/convert_to_flowface.py \
    --video_path "${RGB_PATH}" \
    --tracking_path "${TRACKING_PATH}" \
    --preprocess_path "${PREPROCESS_PATH}" \
    --output_path "${FLOWFACE_OUTPUT}" \
    --device cuda

echo ""
echo "============================================================"
echo "Done. Output: ${FLOWFACE_OUTPUT}/fit.npz"
echo "============================================================"
