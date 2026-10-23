#!/usr/bin/env bash
# Smoke test for the untied K=6 control (LM1B) and OWT tied_struct: init search, loss trend, step time.
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python compress.py --teacher FMLM-B-LM1B --variant untied --K 6 --steps 300 --warmup 100 --log_every 50 \
  --eval_every 300 --n_eval 64 --ckpt_every 300 --out results/compress_smoke3
python compress.py --teacher FMLM-B-OWT --variant tied_struct --groups '0|1-3|4|5-10|11' --batch_size 4 \
  --tokenizer openai-community/gpt2 --steps 300 --warmup 100 --log_every 50 --eval_every 300 --n_eval 32 \
  --ckpt_every 300 --out results/compress_smoke3
echo "smoke C done"
