#!/usr/bin/env bash
# Direction B: toy sweep over dependency length and number of modes; 4 configs per GPU, 2 GPUs.
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
mkdir -p logs/toy_sweep
i=0
for cfg in dep_c1 dep_c2 dep_c4 dep_c8 modes_m2 modes_m16 modes_m64 modes_m256; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) python toy/toy_sweep.py --config $cfg > logs/toy_sweep/$cfg.log 2>&1 &
  i=$((i + 1))
done
wait
echo "toy sweep B done"
