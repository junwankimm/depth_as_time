#!/usr/bin/env bash
# Direction C on the toy: trajectory supervision (uniform schedule) vs endpoint-only fine-tuning.
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
mkdir -p logs/toy_trajsup
# smoke first: abort the job if the script is broken
python toy/toy_trajsup.py --config dep_c8 --lam 1 --steps 20 --n_probe 256 --out results/toy_trajsup_smoke \
  > logs/toy_trajsup/smoke.log 2>&1 || { tail -20 logs/toy_trajsup/smoke.log; exit 1; }
jobs=()
for s in 0 1 2; do for cfg in dep_c1 dep_c2 dep_c4 dep_c8 modes_m2; do for lam in 0 0.1 1; do
  jobs+=("--config $cfg --seed $s --lam $lam"); done; done; done
for cfg in dep_c1 dep_c8; do for lam in 0 1; do jobs+=("--config $cfg --n_blocks 16 --lam $lam"); done; done
printf '%s\n' "${jobs[@]}" | awk '{print NR-1 " " $0}' | xargs -P 16 -L 1 bash -c \
  'i=$0; shift 0; CUDA_VISIBLE_DEVICES=$((i % 2)) python toy/toy_trajsup.py "$@" > "logs/toy_trajsup/$(echo "$@" | tr -d "-" | tr " " "_").log" 2>&1'
echo "toy C done"
