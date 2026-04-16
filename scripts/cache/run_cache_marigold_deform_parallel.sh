#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate marionette

cd /data/pouyan/baseline/repository/cap4d

export HF_TOKEN=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
export PYTHONPATH=.

for i in 0 1 2 3; do
    python scripts/cache/cache_marigold_deform.py \
        --gpu $i --num_gpus 4 &
done
wait

echo "All GPUs finished."
