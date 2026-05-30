#!/bin/bash
# Determinism A/B: run the SAME (track,task,nep) eval twice with fresh servers
# and diff the per-episode detail_info.json. DET=$1 (0 or 1) toggles XLA flags.
set -uo pipefail
REPO=/home/ubuntu/VLABench
DET=${1:-0}
TASK=${TASK:-select_fruit}
TRACK=${TRACK:-track_1_in_distribution}
NEP=${NEP:-2}
OUT=$REPO/research/det_runs/det${DET}
rm -rf "$OUT"; mkdir -p "$OUT"

for R in 1 2; do
  echo "############ DET=$DET RUN $R ############"
  SAVE_DIR="$OUT/run$R" LOG_DIR="$OUT/log$R" DETERMINISM=$DET \
    TASK="$TASK" TRACK="$TRACK" NEP="$NEP" \
    bash "$REPO/research/run_eval_pi05.sh"
  echo "exit=$?"
done

A="$OUT/run1/$TRACK/$TASK/detail_info.json"
B="$OUT/run2/$TRACK/$TASK/detail_info.json"
echo "############ COMPARISON (DET=$DET) ############"
echo "--- run1 detail ---"; cat "$A" 2>&1
echo "--- run2 detail ---"; cat "$B" 2>&1
if [ -f "$A" ] && [ -f "$B" ]; then
  if diff -q "$A" "$B" >/dev/null 2>&1; then
    echo "RESULT_DET${DET}: IDENTICAL (deterministic)"
  else
    echo "RESULT_DET${DET}: DIFFERENT (non-deterministic)"
    echo "--- diff ---"; diff "$A" "$B"
  fi
else
  echo "RESULT_DET${DET}: MISSING_OUTPUT"
fi
echo "############ DET=$DET COMPLETE ############"
exit 0
