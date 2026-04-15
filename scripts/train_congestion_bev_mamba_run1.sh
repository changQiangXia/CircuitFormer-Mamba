#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-circuitformer}"
RUN_TS="${1:-$(date -u +'%Y-%m-%d_%H-%M-%S_UTC')}"
SAVE_DIR="${ROOT}/exp/congestion_bev_mamba_run1_${RUN_TS}"

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"

echo "Prepared BEV Mamba output directory:"
echo "  ${SAVE_DIR}"

WANDB_MODE=disabled conda run -n "${ENV_NAME}" python train.py \
  data.num_workers=8 \
  data.batch_size=8 \
  trainer.devices=1 \
  trainer.precision=16-mixed \
  trainer.sync_batchnorm=False \
  trainer.log_every_n_steps=10 \
  model.lr=0.001 \
  experiment.seed=3407 \
  model.bev_mamba.enabled=True \
  model.bev_mamba.num_blocks=1 \
  model.bev_mamba.inner_dim=64 \
  model.bev_mamba.scan_downsample=4 \
  model.bev_mamba.dw_kernel_size=3 \
  experiment.save_dir="${SAVE_DIR}"
