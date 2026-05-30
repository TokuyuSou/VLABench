# Environment notes (Pi05-ft-primitive / VLABench eval)

Inventory of the environment as it was found and **verified** on this machine.
Nothing here had to be freshly installed; this is the checklist to reproduce the
setup on another box.

## Hardware / drivers

```
GPU      : 1× NVIDIA RTX A6000, 49140 MiB
Driver   : 570.195.03      CUDA (driver) : 12.8
nvcc     : release 12.8
OS       : Linux 6.14 (Ubuntu), bash
```

## Two virtualenvs (do not merge them)

The server and the client run in **separate** venvs on purpose — JAX/openpi need
3.11, while VLABench (mujoco/dm_control stack) runs in 3.10.

| venv | path | key contents | verify |
|------|------|--------------|--------|
| openpi server | `third_party/openpi/.venv` | Python 3.11, `jax 0.5.3`+CUDA, `openpi` | `\.venv/bin/python -c "import jax; print(jax.devices())"` → `[CudaDevice(id=0)]` |
| VLABench client | `third_party/openpi/examples/vlabench/.venv` | Python 3.10, `VLABench`, `openpi_client` | `examples/vlabench/.venv/bin/python -c "import VLABench, openpi_client"` |

`uv 0.11.17` is on PATH (the cluster scripts call `uv run`, but the explicit
`.venv/bin/python` invocations in `research/run_eval_pi05.sh` avoid needing it).

## openpi submodule

```bash
cd third_party/openpi
git checkout pi05        # REQUIRED — see REPRODUCE_pi05_ft_primitive.md §1
git rev-parse HEAD       # 788ef1f...
```

Sanity check the config loads in the server venv:

```bash
.venv/bin/python -c "from openpi.training import config as C; \
  c=C.get_config('pi05_ft_vlabench_primitive'); \
  print(c.name, type(c.model).__name__, getattr(c.model,'pi05',None))"
# -> pi05_ft_vlabench_primitive Pi0Config True
```

## Checkpoint

```
third_party/openpi/checkpoints/pi05-primitive-10task/
├── params/                       # orbax (ocdbt) – 6.2 GiB, restores in ~7s
├── assets/vlabench/vlabench_ft_primitive/norm_stats.json   # {"norm_stats": {"state","actions"}}
├── _CHECKPOINT_METADATA
└── README.md                     # official eval/train instructions + reference SR table
```

The server log must show it loading norm stats from **this** assets dir, not from
`assets/pi05_ft_vlabench_primitive/...` (it logs a harmless "Norm stats not found"
for the config dir first, then loads from the checkpoint dir).

## Episode configs / assets

```
VLABench/configs/evaluation/tracks/track_{1,2,3,4,6}_*.json   # fixed standard episodes
VLABench/assets/{base,obj,robots,scenes}/                     # downloaded sim assets
```

Each track JSON is `{task_name: [ {episode...} × 50 ]}` for 10 tasks — i.e. the
episodes are fully specified, which is what makes the env side deterministic.

## Required environment variables at run time

| var | value | why |
|-----|-------|-----|
| `VLABENCH_ROOT` | `<repo>/VLABench` | client locates `configs/evaluation/tracks/<track>.json` |
| `MUJOCO_GL` | `egl` | headless GPU rendering |
| `MUJOCO_EGL_DEVICE_ID` | `0` | pick the GPU for EGL |
| `CUDA_VISIBLE_DEVICES` | `0` | single GPU |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.9` | let JAX preallocate; model is ~44 GB |
| `VLABENCH_DETERMINISM` | `1` (default) | deterministic XLA; set `0` to disable (see DETERMINISM.md) |

## Gotchas encountered

- The checkpoint README path `vla_bench_scipts/` is a typo for
  **`vlabench_scripts/`**.
- The stock cluster scripts assume 8 GPUs, conda envs `arvla`/`base`, and
  `/inspire/...` data paths — not usable as-is here; use `research/run_eval_pi05.sh`.
- Start the client only **after** `server listening on 0.0.0.0:8000` appears, or
  the websocket connect fails.
- `examples/vlabench/eval.py` exposes a `--args.seed` but it is **unused**;
  determinism does **not** come from it (episodes are fixed; see DETERMINISM.md).
