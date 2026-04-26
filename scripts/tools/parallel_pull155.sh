#!/bin/bash
# Parallel scp transfer from 10.2.1.15 using chunked folder batches.
#
# Usage:
#   bash scripts/tools/parallel_pull155.sh <remote_src> <local_dest> [num_workers]
#
# Examples:
#   bash scripts/tools/parallel_pull155.sh /data0/pouyan/flame_tracking/preprocessing data/flame_tracking/ 8
#   bash scripts/tools/parallel_pull155.sh /data0/pouyan/flame_tracking/tracking data/flame_tracking/ 8

set -euo pipefail

REMOTE_SRC="${1:?Usage: $0 <remote_src> <local_dest> [num_workers]}"
LOCAL_DEST="${2:?Usage: $0 <remote_src> <local_dest> [num_workers]}"
NUM_WORKERS="${3:-8}"
REMOTE_HOST="pouyan@10.2.1.15"

# Ensure remote source exists
if ! ssh "$REMOTE_HOST" "[ -d '$REMOTE_SRC' ]"; then
    echo "Error: remote source directory '$REMOTE_SRC' does not exist on $REMOTE_HOST"
    exit 1
fi

# Get the base name of remote source dir (e.g., "preprocessing" or "tracking")
SRC_BASENAME=$(basename "$REMOTE_SRC")

# Ensure local destination directory exists
echo "Ensuring local directory exists: ${LOCAL_DEST}/${SRC_BASENAME}/"
mkdir -p "${LOCAL_DEST}/${SRC_BASENAME}"

# Collect all subdirectories/files from the remote
mapfile -t ITEMS < <(ssh "$REMOTE_HOST" "ls -1 '$REMOTE_SRC'")
TOTAL=${#ITEMS[@]}

echo "Found $TOTAL items in ${REMOTE_HOST}:${REMOTE_SRC}"
echo "Transferring to ${LOCAL_DEST}/${SRC_BASENAME}/"
echo "Using $NUM_WORKERS parallel workers"
echo ""

# Compute chunk size
CHUNK_SIZE=$(( (TOTAL + NUM_WORKERS - 1) / NUM_WORKERS ))

# Launch workers
PIDS=()
for (( i=0; i<NUM_WORKERS; i++ )); do
    START=$(( i * CHUNK_SIZE ))
    if (( START >= TOTAL )); then
        break
    fi

    # Slice the items for this worker
    WORKER_ITEMS=("${ITEMS[@]:$START:$CHUNK_SIZE}")
    WORKER_COUNT=${#WORKER_ITEMS[@]}

    (
        echo "[Worker $i] Transferring $WORKER_COUNT items (indices $START-$((START + WORKER_COUNT - 1)))"
        for item in "${WORKER_ITEMS[@]}"; do
            scp -r -q "${REMOTE_HOST}:${REMOTE_SRC}/${item}" "${LOCAL_DEST}/${SRC_BASENAME}/"
        done
        echo "[Worker $i] Done."
    ) &

    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} workers. Waiting for completion..."
echo ""

# Wait for all workers and track failures
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        ((FAILED++))
    fi
done

if (( FAILED > 0 )); then
    echo ""
    echo "WARNING: $FAILED worker(s) had errors."
    exit 1
else
    echo ""
    echo "All transfers complete."
fi
