#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: acceptance_monitor.sh <run_dir> [interval_seconds] [target_time] }"
INTERVAL="${2:-300}"
TARGET_TIME="${3:-2026-04-15 02:40:00 UTC}"

CKPT_PATH="${RUN_DIR}/last.ckpt"
LOG_PATH="${RUN_DIR}/acceptance_monitor_$(date -u +'%Y-%m-%d_%H-%M-%S_UTC').log"

while true; do
  echo "=== $(date -u +'%Y-%m-%d %H:%M:%S UTC') ==="
  echo "target_time: ${TARGET_TIME}"

  if [[ -f "${CKPT_PATH}" ]]; then
    python - "${CKPT_PATH}" <<'PY'
import os
import sys
import time
import torch

path = sys.argv[1]
st = os.stat(path)
ckpt = torch.load(path, map_location="cpu")
print(f"ckpt_path: {path}")
print(f"ckpt_mtime_utc: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(st.st_mtime))}")
print(f"epoch: {ckpt.get('epoch')}")
print(f"global_step: {ckpt.get('global_step')}")
PY
  else
    echo "ckpt_path: ${CKPT_PATH}"
    echo "ckpt_status: missing"
  fi

  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader || true
  ps -ef | grep -E 'python train.py|conda run -n .* python train.py' | grep -v grep || true
  echo
  sleep "${INTERVAL}"
done >> "${LOG_PATH}" 2>&1
