#!/usr/bin/env bash
#SBATCH --job-name=flm-setup
#SBATCH --partition=beta_test
#SBATCH --account=ny_sainingxie_advancing
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/ddn/scratch/nyu/awg4562/flm/depth_as_time/logs/%x-%j.out
set -euo pipefail
source /ddn/scratch/nyu/awg4562/flm/depth_as_time/slurm/env.sh
# shellcheck disable=SC2046
srun $(container_args) --container-env=FLM_VENV,PIP_CACHE_DIR,HF_HOME,XDG_CACHE_HOME \
    bash depth_as_time/slurm/setup_env.sh
