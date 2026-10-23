#!/usr/bin/env bash
# LM1B compression smoke test after the warm-start fix: init search, loss trend, step time.
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
for v in tied1 tied_struct tied2_alt; do
  python compress.py --teacher FMLM-B-LM1B --variant $v --steps 1500 --warmup 500 --log_every 100 \
    --eval_every 1500 --n_eval 64 --ckpt_every 1500 --out results/compress_smoke2
done
echo "smoke B done"
