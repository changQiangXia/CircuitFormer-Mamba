#!/usr/bin/env bash
set -euo pipefail

SAVE_DIR="${1:?save_dir is required}"
INTERVAL_SECONDS="${2:-600}"
LOG_FILE="${SAVE_DIR}/monitor.log"
PID_FILE="${SAVE_DIR}/train.pid"
LAUNCHER_LOG="${SAVE_DIR}/launcher.log"

touch "${LOG_FILE}"

while true; do
  TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  TRAIN_PID=""
  if [[ -f "${PID_FILE}" ]]; then
    TRAIN_PID="$(cat "${PID_FILE}")"
  fi

  STATUS="missing_pid"
  if [[ -n "${TRAIN_PID}" ]] && ps -p "${TRAIN_PID}" >/dev/null 2>&1; then
    STATUS="running"
  elif [[ -n "${TRAIN_PID}" ]]; then
    STATUS="stopped"
  fi

  {
    echo "=== ${TS} ==="
    echo "status=${STATUS}"
    echo "train_pid=${TRAIN_PID:-NA}"
    echo "interval_seconds=${INTERVAL_SECONDS}"
    echo "[children]"
    if [[ -n "${TRAIN_PID}" ]]; then
      ps -o pid,ppid,cmd --ppid "${TRAIN_PID}" || true
    fi
    echo "[gpu]"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    echo "[compute_apps]"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    echo "[latest_ckpts]"
    find "${SAVE_DIR}" -maxdepth 1 -type f \( -name 'epoch=*.ckpt' -o -name 'last.ckpt' \) | sort || true
    echo "[launcher_tail]"
    if [[ -f "${LAUNCHER_LOG}" ]]; then
      tail -n 20 "${LAUNCHER_LOG}" || true
    fi
    echo
  } >> "${LOG_FILE}"

  if [[ "${STATUS}" != "running" ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
