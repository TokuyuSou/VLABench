# Environment setup — no-vision / envelope pipeline (`run_full_sweep.sh`)

How to make the **end-to-end no-vision envelope pipeline** run correctly for any of the
10 VLABench primitive tasks. This is the path that produced
`eval_runs/<task>/<task>_novis_h10_envelope_sweep` (e.g. `select_toy`, `select_poker`).

One command runs **data-collection → training → evaluation**:

```bash
cd /home/ubuntu/VLABench
TASK=select_drink bash research/r3m_action_predictor/local/run_full_sweep.sh
```

| stage | step in `run_full_sweep.sh` | venv | GPU |
|-------|-----------------------------|------|-----|
| **data collection + train** | STEP1 (`cli --no-vision` → `ensure_proprio_features` extracts state+action, then trains) | client | yes (train); collection is CPU/network only |
| **evaluation** | STEP2 envelope → STEP3 offline cal → STEP5 cal run → STEP6 skip evals → STEP8 summary | client + pi05 **server** | yes (sim + server) |

This document covers only what is **specific to this pipeline**. The shared eval/server
foundation (the two venvs, the `pi05` openpi branch, the checkpoint, VLABench assets,
track configs, determinism) is documented and verified in:

- [`REPRODUCE_pi05_ft_primitive.md`](../../REPRODUCE_pi05_ft_primitive.md) — exact pi05 eval reproduction
- [`research/ENVIRONMENT_NOTES.md`](../ENVIRONMENT_NOTES.md) — verified environment inventory
- [`research/DETERMINISM.md`](../DETERMINISM.md) — the deterministic-XLA fix

---

## 1. Two virtualenvs (do **not** merge them)

Same split as the eval (server=JAX/3.11, client=MuJoCo/3.10), but the **client venv
additionally needs the predictor stack** (`torch`, `pyarrow`, `matplotlib`) — these are
*not* listed in ENVIRONMENT_NOTES because that doc was eval-only.

| venv | path | needed for | key contents |
|------|------|-----------|--------------|
| **server** | `third_party/openpi/.venv` | pi05 policy server (eval stage) | Python 3.11, `jax 0.5.3`+CUDA, `openpi` |
| **client** | `third_party/openpi/examples/vlabench/.venv` | cli / train / envelope / calibrate / live_eval | Python 3.10, `VLABench`, `openpi_client`, **`torch`**, **`pyarrow`** (proprio parquet read), **`matplotlib`** (report PDF), `huggingface_hub` |

## 2. PYTHONPATH

The package is run from source — the client venv must see `src/`. `run_full_sweep.sh`
exports this for you (`PYTHONPATH=research/r3m_action_predictor/src`); set it manually
only if you invoke a module directly.

## 3. Prerequisites (must exist on disk)

- openpi submodule on the **`pi05`** branch (config `pi05_ft_vlabench_primitive` + the
  `pi05` `eval.py`). See REPRODUCE §1.
- Checkpoint `third_party/openpi/checkpoints/pi05-primitive-10task/` (`params/` + `assets/`).
- `VLABench/assets/{base,obj,robots,scenes}` and `VLABench/configs/evaluation/tracks/track_1_*.json`.
- HF access to `VLABench/vlabench_primitive_ft_lerobot` for the collection stage — or a
  pre-populated `data/vlabench_primitive_ft_lerobot/features_proprio/` (all 5000 episodes
  are already cached on this box, so STEP1 collection is an instant cache hit).

## 4. Runtime environment variables — handled by the script

`run_full_sweep.sh` sets everything needed at run time, so you normally set **none**:

`PYTHONPATH`, `HF_HUB_DISABLE_XET=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`, and (around the
server) `VLABENCH_DETERMINISM=1`, `VLABENCH_ROOT`, `MUJOCO_GL=egl`,
`MUJOCO_EGL_DEVICE_ID=0`, `CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. (Set `VLABENCH_DETERMINISM=0` to trade
reproducibility for XLA-autotune throughput.)

## 5. Verify the environment (≈5 s, no GPU allocation)

```bash
cd /home/ubuntu/VLABench
PYTHONPATH=research/r3m_action_predictor/src \
  third_party/openpi/examples/vlabench/.venv/bin/python - <<'PY'
import torch, numpy, pyarrow, matplotlib, huggingface_hub, openpi_client, VLABench  # client deps
import r3m_action_predictor.cli, r3m_action_predictor.build_envelope_gate            # package on PYTHONPATH
print("client venv OK | torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
third_party/openpi/.venv/bin/python -c "import jax; print('server venv OK | jax', jax.__version__, jax.devices())"
ls third_party/openpi/checkpoints/pi05-primitive-10task/params >/dev/null && echo "checkpoint OK"
```

## 6. Run

```bash
# full pipeline for one task (collect → train → envelope → calibrate → skip sweep → summary)
TASK=select_drink bash research/r3m_action_predictor/local/run_full_sweep.sh
# quick smoke (2 episodes per stage; from the script header)
TASK=select_drink NEP=2 CAL_NEP=2 SWEEP_TAG=smoke bash research/r3m_action_predictor/local/run_full_sweep.sh
```

Supported `TASK` (REGEX dict in the script): `add_condiment insert_flower select_book
select_chemistry_tube select_drink select_fruit select_mahjong select_painting
select_poker select_toy`. Useful knobs: `NEP` (eval episodes, default 50), `CAL_NEP`
(cal-run episodes, 20), `SKIPS` (`"0.10 0.20 0.30"`), `RUN_INF`, `SKIP_BASELINE`,
`REUSE_CALRUN`, `SWEEP_TAG`, `EPOCHS`, `PRED_H`/`PREV_H`.

## 7. Operational notes

- **One GPU job at a time.** Training and eval use the single GPU and the pi05 server
  binds port `8011`; do not start a second pipeline (or a rollout collection) while one
  is running.
- **Long runs detached.** Launch so they survive SSH disconnect:
  ```bash
  setsid nohup env TASK=select_drink bash research/r3m_action_predictor/local/run_full_sweep.sh \
    > research/r3m_action_predictor/local/sweep_select_drink.log 2>&1 < /dev/null &
  ```
  Monitor the log read-only; do not `pkill` the running server.

---

## Verified inventory (this box, 2026-06-10)

| item | value |
|------|-------|
| client venv | Python 3.10.20 · torch 2.7.1+cu126 (cuda ✓) · numpy 1.25.0 · pyarrow 24.0.0 · matplotlib 3.5.3 · huggingface_hub 0.36.2 · openpi_client 0.1.0 · VLABench (importable) |
| server venv | Python 3.11 · jax 0.5.3 (CUDA) |
| openpi | detached `HEAD @ 55197c2` (functional; REPRODUCE references `pi05 @ 788ef1f` — newer commit, same `pi05_ft_vlabench_primitive` config) |
| checkpoint / assets / tracks | `pi05-primitive-10task` (params+assets), `VLABench/assets`, `configs/evaluation/tracks` — all present |
| proprio feature cache | all 5000 episodes cached under `data/.../features_proprio/` |
