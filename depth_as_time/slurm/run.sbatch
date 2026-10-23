#!/usr/bin/env bash
# Run one command inside the NGC container with the FLM venv on 1 test GPU.
#   sbatch depth_as_time/slurm/run.sbatch python depth_as_time/probe_onestep.py ...
#SBATCH --job-name=dat
#SBATCH --partition=beta_test
#SBATCH --account=ny_sainingxie_advancing
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/ddn/scratch/nyu/awg4562/flm/depth_as_time/logs/%x-%j.out
set -euo pipefail
source /ddn/scratch/nyu/awg4562/flm/depth_as_time/slurm/env.sh
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "cmd: $*"
# shellcheck disable=SC2046
srun $(container_args) \
    --container-env=HF_HOME,TORCH_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,MPLCONFIGDIR,FLM_CKPT_DIR \
    bash -c "source ${FLM_VENV}/bin/activate && cd ${FLM_ROOT}/depth_as_time && $*"
echo "end=$(date -Is)"
