#!/usr/bin/env bash
# Paper-protocol evaluation of all compression runs: 1,024 one-step samples x 3 seeds.
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python eval_compress.py --teacher FMLM-B-LM1B --runs 'results/compress/lm1b_*' --out results/compress_eval/lm1b.json
python eval_compress.py --teacher FMLM-B-OWT --runs 'results/compress/owt_*' --tokenizer openai-community/gpt2 \
  --batch_size 16 --out results/compress_eval/owt.json
echo "eval done"
