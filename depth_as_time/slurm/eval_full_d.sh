#!/usr/bin/env bash
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python eval_compress.py --teacher FMLM-B-LM1B --runs 'results/compress/lm1b_*_s[12]' --out results/compress_eval/lm1b_seeds.json
echo "eval done"
