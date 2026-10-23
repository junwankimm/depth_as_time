#!/usr/bin/env bash
#SBATCH --job-name=dat-compress
#SBATCH --partition=beta
#SBATCH --account=ny_sainingxie_advancing
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --output=/ddn/scratch/nyu/awg4562/flm/depth_as_time/logs/%x-%j.out
set -euo pipefail
source /ddn/scratch/nyu/awg4562/flm/depth_as_time/slurm/env.sh
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
# shellcheck disable=SC2046
srun $(container_args) \
    --container-env=HF_HOME,TORCH_HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,MPLCONFIGDIR,FLM_CKPT_DIR \
    bash -c "source ${FLM_VENV}/bin/activate && bash ${FLM_ROOT}/depth_as_time/slurm/stage_b.sh"
echo "end=$(date -Is)"
