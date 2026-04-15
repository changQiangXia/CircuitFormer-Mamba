#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-circuitformer}"
RUN_TS="${1:-$(date -u +'%Y-%m-%d_%H-%M-%S_UTC')}"
SAVE_DIR="${ROOT}/exp/congestion_rerun_lr1e-3_seed3407_${RUN_TS}"

mkdir -p "${SAVE_DIR}"

echo "Prepared retrain output directory:"
echo "  ${SAVE_DIR}"

cd "${ROOT}"

WANDB_MODE=disabled conda run -n "${ENV_NAME}" python train.py \
  data.num_workers=8 \
  data.batch_size=8 \
  trainer.devices=1 \
  trainer.precision=16-mixed \
  trainer.sync_batchnorm=False \
  trainer.log_every_n_steps=10 \
  model.lr=0.001 \
  experiment.seed=3407 \
  experiment.save_dir="${SAVE_DIR}"
