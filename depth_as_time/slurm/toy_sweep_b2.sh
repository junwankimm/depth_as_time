#!/usr/bin/env bash
# Direction B robustness: 2 more seeds (new random datasets) for all configs, depth 16 for four
# configs, and a wider model for the configs that failed to fit at width 128 (m = 64, 256).
set -uo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
mkdir -p logs/toy_sweep
jobs=()
for s in 1 2; do for cfg in dep_c1 dep_c2 dep_c4 dep_c8 modes_m2 modes_m16 modes_m64 modes_m256; do
  jobs+=("--config $cfg --seed $s"); done; done
for cfg in dep_c1 dep_c8 modes_m16 modes_m64; do jobs+=("--config $cfg --n_blocks 16"); done
for cfg in modes_m64 modes_m256; do jobs+=("--config $cfg --hidden 256"); done
i=0
for j in "${jobs[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) python toy/toy_sweep.py $j > "logs/toy_sweep/$(echo $j | tr -d '-' | tr ' ' '_').log" 2>&1 &
  i=$((i + 1))
done
wait
echo "toy sweep B2 done"
