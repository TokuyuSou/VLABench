# Reproducing the Pi05-ft-primitive evaluation on VLABench

This document records the **exact, verified steps** to evaluate the
`Pi05-ft-primitive` checkpoint
([VLABench/pi05-primitive-10task](https://huggingface.co/VLABench/pi05-primitive-10task),
README reference Track-1 SR = 40.6%) on VLABench, on a **single-GPU** machine,
plus the knowledge gained and the determinism fix that was applied.

It was validated on this box:

| Item | Value |
|------|-------|
| GPU | 1× NVIDIA RTX A6000 (49 GB), driver 570.195, CUDA 12.8 |
| OS | Ubuntu (Linux 6.14), `MUJOCO_GL=egl` |
| openpi venv | `third_party/openpi/.venv` — Python 3.11, `jax 0.5.3` (CUDA) |
| eval venv | `third_party/openpi/examples/vlabench/.venv` — Python 3.10, VLABench + `openpi_client` |
| openpi branch | **`pi05`** (commit `788ef1f`) |

---

## 0. TL;DR

```bash
# one task, 2 episodes, fresh server + client, deterministic by default
TASK=select_fruit TRACK=track_1_in_distribution NEP=2 \
  bash research/run_eval_pi05.sh
# -> writes research/smoke_results/<track>/<task>/{metrics.json,detail_info.json,videos/*.mp4}
```

A full Track-1 reproduction (all 10 tasks × 50 episodes) is the loop in
[§5](#5-full-track-evaluation).

---

## 1. The single most important fact: use the `pi05` branch of openpi

The checkpoint is a **π0.5 (flow-matching, `pi05=True`)** model. The eval goes
through openpi's *policy-server + client* architecture:

```
serve_policy.py  (openpi .venv, JAX/GPU)  ──websocket :8000──►  examples/vlabench/eval.py  (eval .venv, MuJoCo)
        └ loads checkpoint + norm_stats                                  └ steps the VLABench env, sends obs, applies actions
```

The openpi submodule defaults to the **`main`** branch, which **does not contain**:

- the training/serve config **`pi05_ft_vlabench_primitive`** (only `pi0_*`/`pi0_fast_*` exist on `main`), and
- the `eval.py` that sends the observation keys this checkpoint expects
  (`observation/second_image`, `last_action`, `robot_frame`).

The checkpoint's own `README.md` says to *"checkout to the branch `pi05`"*. So
the first and decisive step is:

```bash
cd third_party/openpi
git checkout pi05          # config pi05_ft_vlabench_primitive + vlabench_scripts/ + correct eval.py
```

> The checkpoint README references scripts under `vla_bench_scipts/`; on the
> `pi05` branch they actually live in **`vlabench_scripts/`** (typo in the
> README). Those scripts are written for an 8-GPU SLURM-style cluster
> (`/inspire/...` paths, `conda activate arvla`, `nvidia-smi`-based GPU fan-out).
> On a single-GPU box use `research/run_eval_pi05.sh` (below) instead.

---

## 2. Prerequisites that were already in place

These were present and only **verified**, not installed:

- `third_party/openpi/.venv` with `jax 0.5.3` seeing the GPU
  (`jax.devices() -> [CudaDevice(id=0)]`) and `openpi` importable.
- `third_party/openpi/examples/vlabench/.venv` with `openpi_client` and
  `VLABench` importable.
- `uv 0.11.17` on PATH.
- VLABench assets unpacked under `VLABench/assets/` (`base/ obj/ robots/ scenes/`).
- The checkpoint at
  `third_party/openpi/checkpoints/pi05-primitive-10task/`
  (`params/` orbax dir + `assets/vlabench/vlabench_ft_primitive/norm_stats.json`).
- The standard episode configs at
  `VLABench/configs/evaluation/tracks/track_{1,2,3,4,6}_*.json`.

No extra Python packages had to be installed. See
[ENVIRONMENT_NOTES.md](research/ENVIRONMENT_NOTES.md) for the full inventory and
the commands used to verify each item.

---

## 3. How the pieces fit (so the args make sense)

**Server** — `scripts/serve_policy.py`:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/serve_policy.py \
  --port 8000 --env VLABENCH policy:checkpoint \
  --policy.config=pi05_ft_vlabench_primitive \
  --policy.dir=checkpoints/pi05-primitive-10task
```

- The config (`Pi0Config(pi05=True, action_horizon=10, paligemma_variant="gemma_2b")`)
  must match the checkpoint — that is exactly what `pi05_ft_vlabench_primitive`
  encodes.
- Norm stats are loaded **from the checkpoint's own `assets/`**, not the config
  assets dir — the log line to look for is
  `Loaded norm stats from .../checkpoints/pi05-primitive-10task/assets/vlabench/vlabench_ft_primitive`.
- Checkpoint restore takes ~7 s (6.2 GiB) and the model occupies ~44 GB GPU RAM.
  Wait until `server listening on 0.0.0.0:8000` before starting the client.

**Client** — `examples/vlabench/eval.py` (the `pi05` version):

```bash
VLABENCH_ROOT=<repo>/VLABench MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  examples/vlabench/.venv/bin/python examples/vlabench/eval.py \
  --args.host 127.0.0.1 --args.port 8000 \
  --args.eval_track track_1_in_distribution --args.tasks select_fruit \
  --args.n-episode 2 --args.save_dir <out>
```

Two things are **mandatory** and easy to miss:

1. **`VLABENCH_ROOT` must be set.** `eval.py` reads the fixed episode set from
   `$VLABENCH_ROOT/configs/evaluation/tracks/<eval_track>.json`. Without it you
   get a `NoneType`/path error.
2. **`MUJOCO_GL=egl`** (headless rendering). The env also renders four camera
   views per step; the policy uses three of them
   (`obs["rgb"] -> second_image, _, image, image_wrist`).

Output layout written by the `Evaluator`:

```
<save_dir>/<track>/
  metrics.json                       # {task: {success_rate, intention_score, progress_score}}
  <task>/detail_info.json            # per-episode success / consumed_step / scores
  <task>/videos/<i>_success_<bool>_progress_<x.xx>.mp4
```

---

## 4. Smoke test (what was actually run and passed)

```bash
TASK=select_fruit TRACK=track_1_in_distribution NEP=2 bash research/run_eval_pi05.sh
```

Result — server restored the checkpoint, served on `:8000`, the client ran
`select_fruit 2/2` in ~1m48s and wrote valid `metrics.json` /
`detail_info.json` / two `.mp4`s. Eval exit code `0`. ✅

Benign log noise you **can ignore**:

- `Unable to initialize backend 'rocm'/'tpu'` — JAX probing absent backends.
- `WARNING:absl:Failed to converge after 99 steps: err_norm=...` — the MuJoCo
  IK solver for the arm; expected, not an error.
- `opening handshake failed / did not receive a valid HTTP request` in the
  **server** log — that is the readiness TCP probe in `run_eval_pi05.sh`
  opening and closing the port; harmless.

---

## 5. Full track evaluation

The reference numbers in the checkpoint card are **n_episode = 50** per task
across the 10 primitive tasks. On one GPU, run them sequentially (each task is
~30–60 min):

```bash
TASKS="add_condiment insert_flower select_book select_drink select_chemistry_tube \
       select_mahjong select_toy select_fruit select_painting select_poker"
for t in $TASKS; do
  TASK="$t" TRACK=track_1_in_distribution NEP=50 \
    SAVE_DIR="$PWD/research/track1_results" \
    bash research/run_eval_pi05.sh
done
```

Then average the `success_rate` over the 10 `metrics.json` task entries to get
the Track-1 SR (card value 0.406). Other tracks: swap `TRACK=` for
`track_2_cross_category`, `track_3_common_sense`, `track_4_semantic_instruction`,
`track_6_unseen_texture` (these are the tracks shipped in
`VLABench/configs/evaluation/tracks/`).

> The cluster scripts (`vlabench_scripts/multi_run_vlabench.sh`) parallelize one
> eval process per GPU and then call `examples/vlabench/summarize.py` to
> aggregate. On a single GPU, the sequential loop above is the equivalent.

---

## 6. Determinism — the problem and the fix

See [DETERMINISM.md](research/DETERMINISM.md) for the full investigation and
evidence. Summary:

**Problem.** Episodes are *fixed* (the track JSON; `load_env` forces
`random_init=False` when an episode config is given) and the policy uses a fixed
seed (`jax.random.key(0)`), yet **two identical runs diverged** — e.g.
`select_fruit` episode 0 *failed* (progress 0.50) on one run and *succeeded*
(198 steps, progress 1.0) on the next, with different video hashes. The cause is
**GPU/XLA numerical non-determinism** in the π0.5 forward pass, which compounds
over the 200-step closed-loop rollout and flips boundary episodes.

**Fix (minimal, applied).** In
[`third_party/openpi/scripts/serve_policy.py`](third_party/openpi/scripts/serve_policy.py),
**before** JAX/XLA initialize, set deterministic XLA flags by default:

```python
if os.environ.get("VLABENCH_DETERMINISM", "1") != "0":
    os.environ["XLA_FLAGS"] = (... + " --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0").strip()
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
```

This is server-side only (the MuJoCo client is already deterministic) and is the
smallest change that makes the metric reproducible. Escape hatch:
`export VLABENCH_DETERMINISM=0` to restore XLA autotuning throughput.

**Verification.** With the fix, two fresh `select_fruit` runs are
**byte-identical** (same `detail_info.json`, same video md5s); the pre-fix
baseline was not. Reproduce with `research/det_test.sh` (the per-run output
artifacts are regenerable and not committed).

---

## 7. File map of what was added (all under version control friendly paths)

| Path | Purpose |
|------|---------|
| `REPRODUCE_pi05_ft_primitive.md` | this guide |
| `research/run_eval_pi05.sh` | single-GPU runner: start server → wait for port → run eval → teardown |
| `research/det_test.sh` | determinism A/B harness (runs the same eval twice, diffs outputs) |
| `research/ENVIRONMENT_NOTES.md` | verified environment inventory + checks |
| `research/DETERMINISM.md` | determinism investigation, evidence, and fix rationale |
| `third_party/openpi` @ `pi05` | submodule checked out to the required branch |
| `third_party/openpi/scripts/serve_policy.py` | + deterministic-XLA preamble (the fix) |
