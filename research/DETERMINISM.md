# Determinism of the Pi05-ft-primitive VLABench eval — investigation & fix

## Question

When the eval is run repeatedly with the *same* checkpoint and the *same*
episodes, does it produce the *same* metrics? It must, for the success-rate to be
a meaningful, reproducible benchmark number.

## What is already deterministic (by design)

- **Episodes are fixed.** `eval.py` loads
  `$VLABENCH_ROOT/configs/evaluation/tracks/<track>.json`, a fully-specified set
  of 50 episodes per task. In `VLABench/envs/__init__.py`, `load_env` sets
  `random_init = False` whenever an `episode_config` is supplied
  ("forbid random initialization if given episode config"). So scene layout,
  object poses, textures and instruction are not random.
- **The policy seed is fixed.** openpi's `Policy` uses `jax.random.key(0)` when no
  rng is passed (`policy.py`), and `create_trained_policy` never passes one. The
  flow-matching sampler (`pi0.py: sample_actions`) draws its noise from that key.
  The key is advanced statefully per `infer`, but identically across runs because
  a fresh server always starts from `key(0)`.
- **MuJoCo physics** is deterministic for a fixed initial state and action stream.
- `eval.py --args.seed` exists but is **never used** — it is a red herring.

So in principle the run should be deterministic. **In practice it was not.**

## Evidence of non-determinism (baseline, before fix)

Harness: `research/det_test.sh` runs the same `(track, task, n_episode)` twice,
each with a *fresh* server, and diffs `detail_info.json` + video md5s.
Task `select_fruit` (track_1, 2 episodes) was used because it sits right on the
success/fail boundary and is therefore a sensitive discriminator.

Two identical baseline runs **diverged**:

| | run 1 ep0 | run 2 ep0 |
|---|---|---|
| success | `false` | `true` |
| consumed_step | 200 | 198 |
| progress_score | 0.50 | 1.00 |
| video md5 (ep0) | `d3c97f36…` | `fc303040…` |

The per-step MuJoCo IK `err_norm` values also differed between runs. Conclusion:
the divergence originates **upstream of physics**, in the **GPU/XLA forward pass**
of the π0.5 model. XLA, by default, may select non-deterministic
convolution/reduction kernels and autotune kernel choices per process; tiny
floating-point differences in the predicted action chunk compound over the
200-step closed-loop rollout and flip boundary episodes.

## Fix (minimal, server-side)

The non-determinism is entirely on the **server** (model inference); the client
(MuJoCo) is already deterministic. The smallest correct change is to force
deterministic XLA **before JAX initializes**, at the top of
`third_party/openpi/scripts/serve_policy.py`:

```python
import os as _os
if _os.environ.get("VLABENCH_DETERMINISM", "1") != "0":
    _det_flags = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") + " " + _det_flags).strip()
    _os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
```

Rationale for each piece:

- `--xla_gpu_deterministic_ops=true` — forces deterministic GPU kernels
  (deterministic reductions/scatters).
- `--xla_gpu_autotune_level=0` — disables convolution autotuning, whose chosen
  algorithm can vary per process and is itself a determinism source.
- `TF_CUDNN_DETERMINISTIC=1` — deterministic cuDNN paths.
- Gated on `VLABENCH_DETERMINISM` (default on) so throughput-sensitive users can
  opt out with `export VLABENCH_DETERMINISM=0`.

Why this location: `XLA_FLAGS` is consumed when the XLA backend initializes; the
flags are set as the very first statements in the entry-point module, before
`import openpi...` triggers JAX. It applies no matter how the server is launched
(directly, via `vlabench_scripts/serve_policy.sh`, or via `research/run_eval_pi05.sh`).

## Verification (after fix)

Re-ran `research/det_test.sh 0` — the runner exports **no** XLA flags, so the
determinism comes *solely* from the source change. Result:

```
RESULT_DET0: IDENTICAL (deterministic)
```

- `detail_info.json` identical across the two fresh runs.
- Both episode videos **byte-identical** (matching md5s).
- Even the MuJoCo IK `err_norm` sequence is identical run-to-run
  (`0.0936719, 0.274723` in both), confirming the action stream is now bitwise
  reproducible.

The earlier `DET=1` A/B (flags exported by the shell) had already shown
byte-identical videos; this final run proves the in-source default achieves the
same thing without any external flag.

## Cost

Disabling autotuning can slow inference modestly, which is an acceptable
trade-off for a benchmark whose entire purpose is a reproducible number. Opt out
per-run with `VLABENCH_DETERMINISM=0` if raw throughput is needed and exact
reproducibility is not.

## Reproduce this investigation

```bash
# baseline (to see non-determinism) – temporarily set VLABENCH_DETERMINISM=0:
VLABENCH_DETERMINISM=0 bash research/det_test.sh 0   # -> DIFFERENT (pre-fix behaviour)
# with the fix active (default):
bash research/det_test.sh 0                           # -> IDENTICAL
```

Evidence artifacts: `research/det_runs/det0/` (post-fix, identical),
`research/det_test_verify.log`, and the pre-fix divergence captured in
`research/det_test_det0.log`.
