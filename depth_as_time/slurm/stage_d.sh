#!/usr/bin/env bash
# Job D (1 node, 4 GPUs): extra seeds for the key reuse comparison (2 shared blocks with block 6
# separate vs. untied 4 and 2 blocks), LM1B, 150k steps. Resumable.
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
STEPS=${STEPS:-150000}
run() {  # gpu seed tag variant [extra args...]
  local gpu=$1 seed=$2 tag=$3 v=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python compress.py --teacher FMLM-B-LM1B --variant $v --steps $STEPS --seed $seed \
    --batch_size 32 --eval_every 25000 --n_eval 256 --log_every 1000 --out results/compress --tag "${tag}_s${seed}" "$@" \
    >> logs/compress_lm1b_${v}${tag}_s${seed}.log 2>&1
}
run 0 1 _g2b6 tied_struct --groups '0-5,7-11|6' &
run 1 2 _g2b6 tied_struct --groups '0-5,7-11|6' &
(run 2 1 '' untied --K 4; run 2 1 '' untied --K 2) &
(run 3 2 '' untied --K 4; run 3 2 '' untied --K 2) &
wait
echo "stage D done"
