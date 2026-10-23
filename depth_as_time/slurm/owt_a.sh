#!/usr/bin/env bash
# OWT replication, part A: one-step lens (+ teacher ruler, probes, Gen PPL by layer).
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python probe_onestep.py --student FMLM-B-OWT --teacher FLM-B-OWT --tokenizer openai-community/gpt2 \
  --n_samples 128 --batch_size 4 --n_text 4 --out results/owt/onestep
echo "owt A done"
