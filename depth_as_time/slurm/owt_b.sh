#!/usr/bin/env bash
# OWT replication, part B: block diagnostics, state matching, controls, endpoint sweep.
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python probe_blocks.py --teacher FMLM-B-OWT --n_samples 16 --batch_size 4
python probe_statematch.py --student FMLM-B-OWT --teacher FLM-B-OWT --tokenizer openai-community/gpt2 \
  --n_samples 64 --batch_size 4 --out results/owt/statematch
python probe_controls.py --student FMLM-B-OWT --teacher FLM-B-OWT --fit_batches 8 --eval_batches 8 \
  --batch_size 4 --out results/owt/controls
python probe_multistep.py --student FMLM-B-OWT --n_samples 64 --batch_size 4 --rollout_steps 2,4 \
  --out results/owt/multistep
echo "owt B done"
