# R3M Action-Chunk Predictor

This experiment tests whether a lightweight model can predict future VLABench
action chunks from:

- previous action history,
- current multi-view R3M ResNet embeddings,
- current robot self state.

The official raw primitive HDF5 release is split into about 761 GiB of archives,
which does not fit on this machine. For the first runnable task experiment, this
code uses the official VLABench LeRobot-format primitive fine-tuning dataset and
downloads only the selected task episodes.

Official sources checked:

- Project page: https://vlabench.github.io/
- Code: https://github.com/OpenMOSS/VLABench
- Raw primitive FT dataset: https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_dataset
- LeRobot primitive FT dataset: https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot
- R3M: https://github.com/facebookresearch/r3m

## Default Experiment

The default task category is `add_condiment`, selected from official task
strings matching `Add .* to the dish`.

```bash
cd /home/ubuntu/VLABench
bash research/r3m_action_predictor/run_add_condiment.sh
```

The script uses the existing OpenPI virtual environment at
`third_party/openpi/.venv`, downloads the selected official LeRobot episode
parquet files, computes frozen R3M-ResNet18 embeddings for the three image
views, trains the chunk predictor, and writes metrics under
`research/r3m_action_predictor/outputs/`.

Large/generated files live under ignored directories:

- `data/`
- `cache/`
- `outputs/`
- `_hf_meta/`

## Code Layout

- `src/r3m_action_predictor/config.py`: experiment defaults and constants.
- `src/r3m_action_predictor/hf_data.py`: Hugging Face metadata, episode selection, downloads, and splits.
- `src/r3m_action_predictor/r3m_features.py`: official R3M checkout/loading and feature extraction.
- `src/r3m_action_predictor/data.py`: feature stores, normalization, and chunk datasets.
- `src/r3m_action_predictor/model.py`: the lightweight action-chunk predictor.
- `src/r3m_action_predictor/risk.py`: demo-derived safe/unsafe labels and the runtime risk head.
- `src/r3m_action_predictor/metrics.py`: losses, baselines, and evaluation metrics.
- `src/r3m_action_predictor/train.py`: training loop and checkpoint writing.
- `src/r3m_action_predictor/train_risk_head.py`: trains a small risk head on the frozen predictor's demo errors.
- `src/r3m_action_predictor/inference.py`: offline/online predictor loading, optional risk scoring.
- `src/r3m_action_predictor/live_eval.py`: VLABench hybrid Pi0/R3M evaluation policy.
- `src/r3m_action_predictor/cli.py`: orchestration-only command-line entry point.

See `RESULTS.md` for the first `add_condiment` run.

## Improved Residual Transformer

The best current run uses a probabilistic residual Transformer, 200 official
episodes, and task-stratified splitting:

```bash
cd /home/ubuntu/VLABench
bash research/r3m_action_predictor/run_add_condiment_transformer.sh
```

For online use after R3M feature extraction:

```python
from r3m_action_predictor.inference import load_predictor

predictor = load_predictor(
    "research/r3m_action_predictor/outputs/add_condiment_r3m18_residual_transformer_200/best_model.pt"
)
out = predictor.predict(embeddings, state, prev_actions)

actions = out["actions"]                    # [8, 7]
action_std = out["action_std"]              # [8, 7]
confidence = out["confidence"]              # scalar
confidence_per_step = out["confidence_per_step"]
```

## Demo-Derived Risk Head

The runtime confidence from the probabilistic action predictor can miss critical
moments. The simpler replacement is a small binary risk head trained only from
demo data: run the frozen action predictor on held-out demo chunks, label a
chunk as safe when its first 5 predicted actions stay close to the expert
actions, then train the risk head to predict that safe/unsafe label.

```bash
cd /home/ubuntu/VLABench
PYTHONPATH=research/r3m_action_predictor/src \
  third_party/openpi/examples/vlabench/.venv/bin/python \
  -m r3m_action_predictor.train_risk_head --device cpu
```

The trained checkpoint is written to:

```text
research/r3m_action_predictor/outputs/add_condiment_risk_head_demo/risk_head.pt
```

For online hybrid evaluation, keep the old confidence mode by default, or switch
to the risk head with environment variables:

```bash
DECISION_METRIC=risk \
RISK_HEAD_PATH=research/r3m_action_predictor/outputs/add_condiment_risk_head_demo/risk_head.pt \
SAVE_DIR=/home/ubuntu/VLABench/research/r3m_action_predictor/eval_runs/add_condiment_hybrid_risk \
bash research/r3m_action_predictor/run_hybrid_add_condiment_eval.sh
```
