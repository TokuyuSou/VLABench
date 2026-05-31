#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${ROOT}/third_party/openpi/.venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Missing OpenPI Python at ${PY}" >&2
  exit 1
fi

if ! "${PY}" -c 'import hydra' >/dev/null 2>&1; then
  uv pip install --python "${PY}" "hydra-core>=1.3.2"
fi

cd "${ROOT}"
PYTHONPATH="${ROOT}/research/r3m_action_predictor/src:${PYTHONPATH:-}" \
"${PY}" -m r3m_action_predictor.cli \
  --task-name add_condiment \
  --task-regex '^Add .* to the dish$' \
  --max-episodes 100 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --prev-horizon 8 \
  --pred-horizon 8 \
  --r3m-model resnet18 \
  --embedding-batch-size 96 \
  --epochs 35 \
  --batch-size 1024 \
  --model-kind mlp_gru \
  --target-mode absolute \
  --width 256 \
  --view-dim 128 \
  --hidden-dim 512 \
  --seed 7
