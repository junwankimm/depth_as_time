#!/usr/bin/env bash
# Direction A: per-token commit analysis on LM1B and OWT.
set -euo pipefail
cd /ddn/scratch/nyu/awg4562/flm/depth_as_time
python probe_commit.py --student FMLM-B-LM1B --teacher FLM-B-LM1B --n_samples 256 --batch_size 16 --out results/commit/lm1b
python probe_commit.py --student FMLM-B-OWT --teacher FLM-B-OWT --n_samples 64 --batch_size 4 --out results/commit/owt
echo "commit A done"
