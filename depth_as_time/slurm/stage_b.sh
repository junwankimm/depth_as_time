#!/usr/bin/env bash
# Job B (1 node, 4 GPUs): Sec. 5 compression, LM1B variants plus the untied K=6 control and
# OWT tied_struct; runs on the same GPU go in sequence. Resumable: rerunning continues each
# run from its last.pt (finished runs exit immediately).
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
STEPS=${STEPS:-150000}
run() {  # gpu variant [extra args...]
  local gpu=$1 v=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python compress.py --teacher FMLM-B-LM1B --variant $v --steps $STEPS \
    --batch_size 32 --eval_every 25000 --n_eval 256 --log_every 1000 --out results/compress "$@" \
    >> logs/compress_lm1b_$v.log 2>&1
}
run_owt() {  # gpu
  CUDA_VISIBLE_DEVICES=$1 python compress.py --teacher FMLM-B-OWT --variant tied_struct \
    --groups '0|1-3|4|5-10|11' --tokenizer openai-community/gpt2 --steps $STEPS --batch_size 4 \
    --eval_every 25000 --n_eval 128 --log_every 1000 --out results/compress \
    >> logs/compress_owt_tied_struct.log 2>&1
}
(run 0 tied1; run 0 untied --K 6) &
(run 1 tied1_noclock; run_owt 1) &
run 2 tied2_alt &
(run 3 tied_struct; run 3 tied2_split) &
wait
echo "stage B done"
