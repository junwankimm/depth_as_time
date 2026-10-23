#!/usr/bin/env bash
# Direction C on LM1B: lam = 0 (endpoint-only control), 0.1, 1; one per GPU, 80-minute training budget.
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python trajsup.py --lam 1 --steps 30 --log_every 10 --n_eval 64 --eval_seeds 1 --out results/trajsup_smoke \
  > logs/trajsup_smoke.log 2>&1 || { tail -30 logs/trajsup_smoke.log; exit 1; }
grep -E "^step|endpoint" logs/trajsup_smoke.log
i=0
for lam in 0 0.1 1; do
  CUDA_VISIBLE_DEVICES=$i python trajsup.py --lam $lam --time_budget 80 > logs/trajsup_lm1b_lam$lam.log 2>&1 &
  i=$((i + 1))
done
wait
echo "trajsup LM1B done"
