#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-circuitformer-true-mamba}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQ_FILE="${REPO_ROOT}/true_mamba_experiments/requirements-true-mamba.txt"

source /etc/network_turbo || true

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

conda run -n "${ENV_NAME}" python -m pip install \
  --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

conda run -n "${ENV_NAME}" python -m pip install \
  --no-cache-dir \
  -r "${REQ_FILE}"

conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir spconv-cu121==2.3.8
conda run -n "${ENV_NAME}" python -m pip install \
  --no-cache-dir \
  torch-scatter \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

conda run -n "${ENV_NAME}" env OMP_NUM_THREADS=1 python -m pip install --no-build-isolation \
  --no-cache-dir \
  causal-conv1d==1.6.1 \
  mamba-ssm==2.3.1

echo "Environment ready: ${ENV_NAME}"
echo "Recommended verification:"
echo "  bash true_mamba_experiments/scripts/verify_true_mamba_env.sh ${ENV_NAME}"
