#!/usr/bin/env bash
# Build the FLM venv on top of the NGC PyTorch container. Runs INSIDE the
# container. --system-site-packages reuses the container's aarch64/Blackwell
# torch, flash-attn and triton instead of the x86 pins in requirements.txt.
set -euo pipefail

VENV=${FLM_VENV:-/ddn/scratch/nyu/awg4562/envs/flm}
[[ -x "$VENV/bin/python" ]] || python -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

CONSTRAINTS="$VENV/container-constraints.txt"
python -m pip list --format=freeze 2>/dev/null \
    | grep -i -E '^(torch|numpy|flash[-_]attn|triton|pyarrow)==' > "$CONSTRAINTS"
echo "== pinned from container:"; cat "$CONSTRAINTS"

# requirements.txt minus torch / triton / numpy (container) and wandb/nvitop.
python -m pip install -q -c "$CONSTRAINTS" \
    datasets==3.6.0 einops==0.8.1 entmax==1.3 hydra-core==1.3.2 \
    lightning==2.6.0 omegaconf==2.3.0 rich scipy timm==1.0.22 \
    tokenizers==0.22.1 torchmetrics==1.8.2 transformers==4.57.3 \
    huggingface_hub safetensors matplotlib scikit-learn

echo "== import check"
python - <<'PY'
import torch, flash_attn, transformers, lightning, scipy, timm, einops
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("flash_attn", flash_attn.__version__, "transformers", transformers.__version__)
import sys; sys.path.insert(0, "/ddn/scratch/nyu/awg4562/flm")
import models.dit, utils
print("flm imports OK")
PY
