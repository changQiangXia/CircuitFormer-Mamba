#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-circuitformer-true-mamba}"
RUN_TS="${1:-$(date -u +'%Y-%m-%d_%H-%M-%S_UTC')}"
SAVE_DIR="${ROOT}/exp/preflight_true_mamba_scheme_b_${RUN_TS}"
LOG_FILE="${SAVE_DIR}/preflight.log"

# /etc/network_turbo is an AutoDL-specific acceleration entry.
# On other platforms, network access needs to be handled outside this script.
source /etc/network_turbo >/dev/null 2>&1 || true

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"

{
  echo "preflight_save_dir=${SAVE_DIR}"
  echo "preflight_env=${ENV_NAME}"
  echo "preflight_started_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
} | tee "${LOG_FILE}"

OMP_NUM_THREADS=1 WANDB_MODE=disabled conda run -n "${ENV_NAME}" python train.py \
  data.num_workers=0 \
  data.batch_size=8 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  trainer.sync_batchnorm=False \
  trainer.log_every_n_steps=1 \
  trainer.max_epochs=1 \
  trainer.check_val_every_n_epoch=1 \
  +trainer.gradient_clip_val=1.0 \
  +trainer.gradient_clip_algorithm=norm \
  +trainer.limit_train_batches=1 \
  +trainer.limit_val_batches=1 \
  +trainer.num_sanity_val_steps=0 \
  model.lr=0.001 \
  experiment.seed=3407 \
  model.bev_mamba.enabled=False \
  model.true_mamba.enabled=True \
  model.true_mamba.num_blocks=1 \
  model.true_mamba.d_state=16 \
  model.true_mamba.d_conv=4 \
  model.true_mamba.expand=2 \
  model.true_mamba.downsample=4 \
  model.true_mamba.bidirectional=True \
  model.true_mamba.use_input_norm=False \
  model.true_mamba.use_mask=True \
  model.true_mamba.mask_pool_mode=max \
  model.true_mamba.out_proj_init_zero=False \
  model.true_mamba.out_proj_init_std=0.001 \
  model.true_mamba.use_residual_scale=True \
  model.true_mamba.residual_scale_init=0.001 \
  model.true_mamba.remask_after_upsample=True \
  experiment.save_dir="${SAVE_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "preflight_finished_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_FILE}"
