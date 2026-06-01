#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/VLABench}
OPENPI="${REPO}/third_party/openpi"
SERVER_PY=${SERVER_PY:-"${OPENPI}/.venv/bin/python"}
EVAL_PY=${EVAL_PY:-"${OPENPI}/examples/vlabench/.venv/bin/python"}

CONFIG=${CONFIG:-pi05_ft_vlabench_primitive}
CKPT=${CKPT:-checkpoints/pi05-primitive-10task}
TRACK=${TRACK:-track_1_in_distribution}
NEP=${NEP:-50}
PORT=${PORT:-8010}
REPLAN_STEPS=${REPLAN_STEPS:-8}
DETERMINISM=${DETERMINISM:-1}
CUDA_DEVICE=${CUDA_DEVICE:-0}
MUJOCO_DEVICE=${MUJOCO_DEVICE:-${CUDA_DEVICE}}
VISUALIZATION=${VISUALIZATION:-1}
SKIP_DONE=${SKIP_DONE:-1}
KILL_EXISTING_SERVER=${KILL_EXISTING_SERVER:-1}
METRICS=${METRICS:-"success_rate intention_score progress_score"}

TASKS=${TASKS:-"add_condiment insert_flower select_book select_chemistry_tube select_drink select_fruit select_mahjong select_painting select_poker select_toy"}
SAVE_DIR=${SAVE_DIR:-"${REPO}/research/r3m_action_predictor/eval_runs/pi05_baseline_${TRACK}_r${REPLAN_STEPS}_${NEP}ep"}
LOG_ROOT=${LOG_ROOT:-"${REPO}/research/r3m_action_predictor/pipeline_logs/pi05_baseline_${TRACK}_r${REPLAN_STEPS}_${NEP}ep_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "${SAVE_DIR}" "${LOG_ROOT}"

export VLABENCH_DETERMINISM="${DETERMINISM}"
export VLABENCH_ROOT="${REPO}/VLABench"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_DEVICE}"

log() {
  echo "=== [$(date '+%F %T')] $*" | tee -a "${LOG_ROOT}/baseline.log"
}

cleanup() {
  local exit_code=$?
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ "${KILL_EXISTING_SERVER}" == "1" ]]; then
    pkill -f serve_policy.py 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

task_done() {
  local task="$1"
  local metrics_path="${SAVE_DIR}/${TRACK}/metrics.json"
  [[ "${SKIP_DONE}" == "1" && -f "${metrics_path}" ]] || return 1
  "${EVAL_PY}" - "${metrics_path}" "${task}" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
task = sys.argv[2]
try:
    metrics = json.loads(metrics_path.read_text())
except Exception:
    sys.exit(1)
sys.exit(0 if task in metrics else 1)
PY
}

summarize_task() {
  local task="$1"
  local metrics_path="${SAVE_DIR}/${TRACK}/metrics.json"
  [[ -f "${metrics_path}" ]] || return 0
  "${EVAL_PY}" - "${metrics_path}" "${task}" <<'PY' | tee -a "${LOG_ROOT}/summary.jsonl"
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
task = sys.argv[2]
row = metrics.get(task)
if row is None:
    sys.exit(0)
print(json.dumps({
    "task": task,
    "success_rate": row.get("success_rate"),
    "intention_score": row.get("intention_score"),
    "progress_score": row.get("progress_score"),
}, ensure_ascii=False))
PY
}

write_config() {
  "${EVAL_PY}" - "${LOG_ROOT}/baseline_config.json" <<PY
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "config": "${CONFIG}",
    "checkpoint": "${CKPT}",
    "track": "${TRACK}",
    "n_episode": ${NEP},
    "port": ${PORT},
    "replan_steps": ${REPLAN_STEPS},
    "determinism": "${DETERMINISM}",
    "cuda_device": "${CUDA_DEVICE}",
    "mujoco_device": "${MUJOCO_DEVICE}",
    "visualization": "${VISUALIZATION}",
    "metrics": "${METRICS}",
    "tasks": "${TASKS}".split(),
    "save_dir": "${SAVE_DIR}",
}, indent=2) + "\\n")
PY
}

wait_for_server() {
  local ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "SERVER DIED EARLY"
      tail -80 "${LOG_ROOT}/server_${PORT}.log" || true
      exit 1
    fi
    if python3 -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',${PORT}))==0 else 1)" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    log "server never opened port ${PORT}"
    tail -80 "${LOG_ROOT}/server_${PORT}.log" || true
    exit 2
  fi
}

run_eval_task() {
  local task="$1"
  if task_done "${task}"; then
    log "Skipping ${task}; found existing metrics in ${SAVE_DIR}/${TRACK}/metrics.json"
    summarize_task "${task}"
    return
  fi

  log "START eval_${task}"
  local eval_args=(
    --args.host 127.0.0.1
    --args.port "${PORT}"
    --args.replan-steps "${REPLAN_STEPS}"
    --args.tasks "${task}"
    --args.eval-track "${TRACK}"
    --args.n-episode "${NEP}"
    --args.save-dir "${SAVE_DIR}"
    --args.metrics "${METRICS}"
  )
  if [[ "${VISUALIZATION}" == "1" ]]; then
    eval_args+=(--args.visulization)
  fi

  cd "${OPENPI}"
  PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  MUJOCO_EGL_DEVICE_ID="${MUJOCO_DEVICE}" \
    "${EVAL_PY}" examples/vlabench/eval.py "${eval_args[@]}" \
    > "${LOG_ROOT}/eval_${task}.log" 2>&1

  log "DONE eval_${task}"
  summarize_task "${task}"
}

main() {
  write_config
  log "Logs: ${LOG_ROOT}"
  log "Save dir: ${SAVE_DIR}"
  log "Running pi05 baseline on ${TRACK}, episodes=${NEP}, replan_steps=${REPLAN_STEPS}"

  cd "${OPENPI}"
  if [[ "${KILL_EXISTING_SERVER}" == "1" ]]; then
    pkill -f serve_policy.py 2>/dev/null || true
    sleep 3
  fi

  log "Starting pi05 server on :${PORT}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    "${SERVER_PY}" scripts/serve_policy.py \
      --port "${PORT}" --env VLABENCH policy:checkpoint \
      --policy.config="${CONFIG}" --policy.dir="${CKPT}" \
      > "${LOG_ROOT}/server_${PORT}.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server
  log "Server ready"
  sleep 5

  read -r -a task_list <<< "${TASKS}"
  for task in "${task_list[@]}"; do
    run_eval_task "${task}"
  done

  log "Baseline evaluation completed"
  log "Metrics: ${SAVE_DIR}/${TRACK}/metrics.json"
  log "Summary: ${LOG_ROOT}/summary.jsonl"
}

main "$@"
