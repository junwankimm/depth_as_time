#!/usr/bin/env bash
# Token dependency vs. commit, token classes, multi-step commit order (LM1B, then OWT).
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python probe_dependency.py --student FMLM-B-LM1B --out results/dependency/lm1b
python probe_dependency.py --student FMLM-B-OWT --tokenizer openai-community/gpt2 --n_dep 8 --chunk 8 \
  --n_cls 64 --n_roll 32 --batch_size 4 --out results/dependency/owt
echo "dependency done"
