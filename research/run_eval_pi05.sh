#!/bin/bash
# Reusable single-GPU runner for evaluating Pi05-ft-primitive on VLABench.
# Starts a fresh pi05 policy server, runs the VLABench eval client for one
# (track, task) pair, then tears the server down.
#
# Usage:
#   TASK=select_fruit TRACK=track_1_in_distribution NEP=2 \
#   SAVE_DIR=/abs/out  bash run_eval_pi05.sh
#
# Env knobs:
#   TASK   (default select_fruit)      VLABench task name
#   TRACK  (default track_1_in_distribution)
#   NEP    (default 2)                 episodes per task
#   PORT   (default 8000)
#   SAVE_DIR (default research/smoke_results)
#   DETERMINISM (default 1)            forwarded to the server as VLABENCH_DETERMINISM
set -uo pipefail

REPO=/home/ubuntu/VLABench
OPENPI=$REPO/third_party/openpi
CKPT=checkpoints/pi05-primitive-10task
CONFIG=pi05_ft_vlabench_primitive

TASK=${TASK:-select_fruit}
TRACK=${TRACK:-track_1_in_distribution}
NEP=${NEP:-2}
PORT=${PORT:-8000}
SAVE_DIR=${SAVE_DIR:-$REPO/research/smoke_results}
DETERMINISM=${DETERMINISM:-1}
LOG=${LOG_DIR:-$REPO/research/eval_logs}
mkdir -p "$SAVE_DIR" "$LOG"

# Determinism is enforced server-side by scripts/serve_policy.py, gated on
# VLABENCH_DETERMINISM (default on). Map this script's DETERMINISM knob onto it
# so the deterministic XLA flags have a single source of truth.
export VLABENCH_DETERMINISM="$DETERMINISM"

cd "$OPENPI" || exit 99
pkill -f serve_policy.py 2>/dev/null || true
sleep 3

echo "=== [$(date +%T)] starting server (det=$DETERMINISM) ==="
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/serve_policy.py \
  --port "$PORT" --env VLABENCH policy:checkpoint \
  --policy.config="$CONFIG" --policy.dir="$CKPT" \
  > "$LOG/server_${PORT}.log" 2>&1 &
SERVER_PID=$!

# Wait for the server to start listening on the port.
PORT_OK=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== SERVER DIED EARLY (iter $i) ==="; tail -40 "$LOG/server_${PORT}.log"; exit 1
  fi
  if python3 -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; then
    echo "=== [$(date +%T)] server ready (iter $i) ==="; PORT_OK=1; break
  fi
  sleep 5
done
[ "$PORT_OK" = 1 ] || { echo "=== server never opened port ==="; tail -40 "$LOG/server_${PORT}.log"; kill "$SERVER_PID" 2>/dev/null; exit 2; }
sleep 5

echo "=== [$(date +%T)] running eval: track=$TRACK task=$TASK nep=$NEP save=$SAVE_DIR ==="
VLABENCH_ROOT=$REPO/VLABench MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  examples/vlabench/.venv/bin/python examples/vlabench/eval.py \
  --args.host 127.0.0.1 --args.port "$PORT" \
  --args.eval_track "$TRACK" --args.tasks "$TASK" \
  --args.n-episode "$NEP" --args.save_dir "$SAVE_DIR" \
  > "$LOG/eval_${PORT}.log" 2>&1
EVAL_EXIT=$?
echo "=== [$(date +%T)] eval exit $EVAL_EXIT ==="
tail -3 "$LOG/eval_${PORT}.log"

kill "$SERVER_PID" 2>/dev/null || true
pkill -f serve_policy.py 2>/dev/null || true
sleep 2
echo "=== DONE task=$TASK eval_exit=$EVAL_EXIT ==="
exit "$EVAL_EXIT"
