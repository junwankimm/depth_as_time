#!/usr/bin/env bash
# Direction C on LM1B, round 2: schedules that reach tau = 1 early (block 8 or 10), lam = 1,
# so that early exit at that block can replace the full depth; plus a longer uniform run.
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
CUDA_VISIBLE_DEVICES=0 python trajsup.py --lam 1 --finish_block 8 --time_budget 95 > logs/trajsup_lm1b_lam1_fin8.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python trajsup.py --lam 1 --finish_block 10 --time_budget 95 > logs/trajsup_lm1b_lam1_fin10.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python trajsup.py --lam 0 --finish_block 8 --time_budget 95 --tag _ctrl > logs/trajsup_lm1b_lam0_fin8.log 2>&1 &
wait
echo "trajsup LM1B round 2 done"
