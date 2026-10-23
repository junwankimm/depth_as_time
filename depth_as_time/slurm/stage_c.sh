#!/usr/bin/env bash
# Job C (1 node, 4 GPUs): fewer-than-6-block structures (with clock) and the matching untied
# controls (N distinct blocks, applied once), LM1B, 150k steps each. Resumable.
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
STEPS=${STEPS:-150000}
run() {  # gpu tag variant [extra args...]
  local gpu=$1 tag=$2 v=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu python compress.py --teacher FMLM-B-LM1B --variant $v --steps $STEPS \
    --batch_size 32 --eval_every 25000 --n_eval 256 --log_every 1000 --out results/compress --tag "$tag" "$@" \
    >> logs/compress_lm1b_${v}${tag}.log 2>&1
}
# Round 1 (highest ROI): the 4- and 3-block structures with their equal-parameter controls.
# Round 2 (diagnostic / curve completion): 2 blocks with a separate block 6, untied N = 2, 1.
(run 0 _g4 tied_struct --groups '0|1-5|6|7-11'; run 0 _g2b6 tied_struct --groups '0-5,7-11|6') &
(run 1 '' untied --K 4; run 1 '' untied --K 2) &
(run 2 _g3 tied_struct --groups '0-5|6|7-11'; run 2 '' untied --K 1) &
run 3 '' untied --K 3 &
wait
echo "stage C done"
