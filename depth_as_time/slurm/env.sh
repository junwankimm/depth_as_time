# Shared Slurm/container settings for the depth-as-time probes on Empire AI
# Beta. Source this from sbatch scripts.

export SLURM_ACCOUNT_NAME=ny_sainingxie_advancing
export SCRATCH_ROOT=/ddn/scratch/nyu/awg4562
export FLM_ROOT=${SCRATCH_ROOT}/flm
export FLM_VENV=${SCRATCH_ROOT}/envs/flm
export FLM_CKPT_DIR=${SCRATCH_ROOT}/models/flm

# NGC PyTorch image (arm64 build, ships torch + flash-attn for Blackwell).
export CONTAINER_SQSH="${SCRATCH_ROOT}/containers/pytorch_26.06-py3.sqsh"
export CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT}"

# Keep every cache on Scratch; Home is not mounted in the container.
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/.cache/pip"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/.cache/triton"
export MPLCONFIGDIR="${SCRATCH_ROOT}/.cache/matplotlib"

container_args() {
    echo "--container-image=${CONTAINER_SQSH}"
    echo "--container-mounts=${CONTAINER_MOUNTS}"
    echo "--container-workdir=${FLM_ROOT}"
    echo "--no-container-mount-home"
}
