#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ubuntu/VLABench
OPENPI="${REPO}/third_party/openpi"
SERVER_PY="${OPENPI}/.venv/bin/python"
EVAL_PY="${OPENPI}/examples/vlabench/.venv/bin/python"
CKPT=checkpoints/pi05-primitive-10task
CONFIG=pi05_ft_vlabench_primitive

TASK=${TASK:-add_condiment}
TRACK=${TRACK:-track_1_in_distribution}
NEP=${NEP:-50}
PORT=${PORT:-8000}
SAVE_DIR=${SAVE_DIR:-${REPO}/research/r3m_action_predictor/eval_runs/add_condiment_hybrid_conf0979}
REPLAN_STEPS=${REPLAN_STEPS:-5}
CONF_THRESHOLD=${CONF_THRESHOLD:-0.975}
MIN_STEP_CONF=${MIN_STEP_CONF:-0.94}
MAX_SUB_FRAC=${MAX_SUB_FRAC:-0.25}
DECISION_METRIC=${DECISION_METRIC:-confidence}
RISK_HEAD_PATH=${RISK_HEAD_PATH:-}
RISK_THRESHOLD=${RISK_THRESHOLD:-}
DETERMINISM=${DETERMINISM:-1}
LOG_DIR=${LOG_DIR:-${SAVE_DIR}/_logs}

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

if ! "${EVAL_PY}" -c 'import hydra, omegaconf' >/dev/null 2>&1; then
  uv pip install --python "${EVAL_PY}" "hydra-core>=1.3.2"
fi

export VLABENCH_DETERMINISM="${DETERMINISM}"
export VLABENCH_ROOT="${REPO}/VLABench"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

cd "${OPENPI}"
pkill -f serve_policy.py 2>/dev/null || true
sleep 3

echo "=== [$(date +%T)] starting pi05 server on :${PORT} ==="
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "${SERVER_PY}" scripts/serve_policy.py \
  --port "${PORT}" --env VLABENCH policy:checkpoint \
  --policy.config="${CONFIG}" --policy.dir="${CKPT}" \
  > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
SERVER_PID=$!

PORT_OK=0
for i in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "=== SERVER DIED EARLY ==="
    tail -80 "${LOG_DIR}/server_${PORT}.log"
    exit 1
  fi
  if python3 -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',${PORT}))==0 else 1)" 2>/dev/null; then
    echo "=== [$(date +%T)] server ready ==="
    PORT_OK=1
    break
  fi
  sleep 5
done

if [[ "${PORT_OK}" != "1" ]]; then
  echo "=== server never opened port ==="
  tail -80 "${LOG_DIR}/server_${PORT}.log"
  kill "${SERVER_PID}" 2>/dev/null || true
  exit 2
fi
sleep 5

echo "=== [$(date +%T)] running hybrid eval task=${TASK} track=${TRACK} episodes=${NEP} decision=${DECISION_METRIC} ==="
cd "${REPO}"
EVAL_ARGS=(
  --args.host 127.0.0.1
  --args.port "${PORT}"
  --args.replan-steps "${REPLAN_STEPS}"
  --args.tasks "${TASK}"
  --args.eval-track "${TRACK}"
  --args.n-episode "${NEP}"
  --args.save-dir "${SAVE_DIR}"
  --args.confidence-threshold "${CONF_THRESHOLD}"
  --args.min-step-confidence "${MIN_STEP_CONF}"
  --args.max-substitution-fraction "${MAX_SUB_FRAC}"
  --args.decision-metric "${DECISION_METRIC}"
  --args.predictor-device cpu
  --args.visulization
)
if [[ -n "${RISK_HEAD_PATH}" ]]; then
  EVAL_ARGS+=(--args.risk-head-path "${RISK_HEAD_PATH}")
fi
if [[ -n "${RISK_THRESHOLD}" ]]; then
  EVAL_ARGS+=(--args.risk-threshold "${RISK_THRESHOLD}")
fi

PYTHONPATH="${REPO}/research/r3m_action_predictor/src:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="" \
"${EVAL_PY}" -m r3m_action_predictor.live_eval \
  "${EVAL_ARGS[@]}" \
  > "${LOG_DIR}/hybrid_eval_${PORT}.log" 2>&1
EVAL_EXIT=$?
echo "=== [$(date +%T)] hybrid eval exit ${EVAL_EXIT} ==="
tail -40 "${LOG_DIR}/hybrid_eval_${PORT}.log"

kill "${SERVER_PID}" 2>/dev/null || true
pkill -f serve_policy.py 2>/dev/null || true
sleep 2
exit "${EVAL_EXIT}"
