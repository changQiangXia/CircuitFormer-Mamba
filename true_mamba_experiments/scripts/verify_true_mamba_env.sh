#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-circuitformer-true-mamba}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# /etc/network_turbo is an AutoDL-specific acceleration entry.
# On other platforms, network access needs to be handled outside this script.
source /etc/network_turbo >/dev/null 2>&1 || true

# Some hosts export OMP_NUM_THREADS=0, which libgomp rejects.
export OMP_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import json
import torch
import causal_conv1d
import mamba_ssm
from mamba_ssm import Mamba

model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 32, 64, device="cuda")
with torch.no_grad():
    y = model(x)

print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "causal_conv1d": causal_conv1d.__file__,
    "mamba_ssm": mamba_ssm.__file__,
    "input_shape": list(x.shape),
    "output_shape": list(y.shape),
    "param_count": sum(p.numel() for p in model.parameters()),
}, ensure_ascii=False, indent=2))
PY

conda run --no-capture-output -n "${ENV_NAME}" \
  python "${REPO_ROOT}/scripts/smoke_test_cpu.py"
