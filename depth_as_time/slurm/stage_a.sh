#!/usr/bin/env bash
# Job A (1 GPU): block diagnostics, toy Sec. 5 compression, LM1B compression smoke test.
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python probe_blocks.py --teacher FMLM-B-LM1B
python probe_blocks.py --teacher toy:cities:exact
for v in tied1 tied1_noclock tied2_alt tied2_split; do
  python compress.py --teacher toy:cities:exact --variant $v --steps 6000 --batch_size 1024 \
    --lr 3e-4 --warmup 500 --eval_every 2000 --n_eval 1024 --ckpt_every 2000 --split 4 --out results/compress_toy
done
python compress.py --teacher toy:cities:exact --variant tied2_alt --K 4 --steps 6000 --batch_size 1024 \
  --lr 3e-4 --warmup 500 --eval_every 2000 --n_eval 1024 --ckpt_every 2000 --out results/compress_toy
python compress.py --teacher FMLM-B-LM1B --variant tied1 --steps 300 --eval_every 300 --n_eval 64 \
  --ckpt_every 300 --out results/compress_smoke
echo "stage A done"
