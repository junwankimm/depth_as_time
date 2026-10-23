#!/usr/bin/env bash
# Paper-protocol evaluation of all LM1B compression runs (after job C).
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python eval_compress.py --teacher FMLM-B-LM1B --runs 'results/compress/lm1b_*' --out results/compress_eval/lm1b_c.json
echo "eval done"
