#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-circuitformer-true-mamba}"
SAVE_DIR="${1:?save_dir is required}"
RESUME_CKPT="${2:-${SAVE_DIR}/last.ckpt}"
MONITOR_INTERVAL="${MONITOR_INTERVAL_SECONDS:-600}"

source /etc/network_turbo >/dev/null 2>&1 || true

cd "${ROOT}"

cat > "${SAVE_DIR}/resume_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
exec env OMP_NUM_THREADS=1 WANDB_MODE=disabled conda run --no-capture-output -n "${ENV_NAME}" python train.py \\
  data.num_workers=8 \\
  data.batch_size=8 \\
  trainer.devices=1 \\
  trainer.precision=16-mixed \\
  trainer.sync_batchnorm=False \\
  trainer.log_every_n_steps=10 \\
  model.lr=0.001 \\
  experiment.seed=3407 \\
  model.bev_mamba.enabled=False \\
  model.true_mamba.enabled=True \\
  model.true_mamba.num_blocks=1 \\
  model.true_mamba.d_state=16 \\
  model.true_mamba.d_conv=4 \\
  model.true_mamba.expand=2 \\
  model.true_mamba.downsample=1 \\
  model.true_mamba.bidirectional=True \\
  model.true_mamba.out_proj_init_zero=True \\
  experiment.save_dir="${SAVE_DIR}" \\
  experiment.resume_ckpt_path="${RESUME_CKPT}"
EOF
chmod +x "${SAVE_DIR}/resume_command.sh"

nohup bash "${SAVE_DIR}/resume_command.sh" > "${SAVE_DIR}/resume.log" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${SAVE_DIR}/train.pid"

nohup bash "${ROOT}/true_mamba_experiments/scripts/monitor_true_mamba_run.sh" "${SAVE_DIR}" "${MONITOR_INTERVAL}" > "${SAVE_DIR}/monitor_launcher.log" 2>&1 &
MONITOR_PID=$!
echo "${MONITOR_PID}" > "${SAVE_DIR}/monitor.pid"

echo "save_dir=${SAVE_DIR}"
echo "resume_ckpt=${RESUME_CKPT}"
echo "train_pid=${TRAIN_PID}"
echo "monitor_pid=${MONITOR_PID}"
echo "monitor_interval_seconds=${MONITOR_INTERVAL}"
