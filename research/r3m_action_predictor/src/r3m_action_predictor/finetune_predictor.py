"""Fine-tune a demo-trained action predictor on the base VLA's successful rollouts (covariate-shift
correction). Reuses the exact demo training loop (``train.train_model``) -- only the data source,
the warm-start, and the regularisation knobs differ.

Anti-overfitting measures (the rollout-success set is small):
  - warm-start from the demo checkpoint (not from scratch);
  - mix in the demo train episodes as an anchor (``--demo-fraction``, default 1.0) to prevent
    catastrophic forgetting / drift;
  - reuse the demo-fit normalizers (NOT refit) so the warm-started weights stay valid;
  - low LR (default 3e-5) and few epochs (default 25) with cosine decay;
  - val-based best-checkpoint selection (a held-out split of the combined pool);
  - inherited weight decay + dropout from the demo config.

Demo and rollout episodes share the proprio npz schema, so ``FeatureStore``/``ChunkDataset`` consume
both unchanged. Writes a new checkpoint (never overwrites the source).

Usage:
    python -m r3m_action_predictor.finetune_predictor \
        --action-run-dir outputs/<task>/<task>_novis_h10_envelope \
        --rollout-features <rollout_dir> --output-dir <ft_dir>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ChunkDataset, FeatureStore, Normalizer
from .hf_data import EpisodeInfo
from .train import train_model


def _episodes_from_manifest_train(manifest: dict, demo_fraction: float, seed: int) -> list[EpisodeInfo]:
    by_index = {int(e["episode_index"]): e for e in manifest["episodes"]}
    train_idx = [int(i) for i in manifest["splits"]["train"]]
    if demo_fraction < 1.0:
        rng = random.Random(seed)
        rng.shuffle(train_idx)
        train_idx = train_idx[: max(0, round(len(train_idx) * demo_fraction))]
    fields = EpisodeInfo.__dataclass_fields__.keys()
    return [EpisodeInfo(**{k: by_index[i][k] for k in fields}) for i in train_idx]


def _rollout_success_episodes(rollout_dir: Path, success_only: bool) -> list[EpisodeInfo]:
    index = json.loads((rollout_dir / "index.json").read_text())
    eps = []
    for e in index["episodes"]:
        if success_only and not e.get("is_success", False):
            continue
        eps.append(EpisodeInfo(
            episode_index=int(e["episode_index"]), task=e["task"], length=int(e["T"]),
            parquet_path="", feature_path=str(rollout_dir / e["file"]),
        ))
    return eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action-run-dir", type=Path, required=True, help="Demo-trained predictor dir (best_model.pt + manifest.json)")
    ap.add_argument("--rollout-features", type=Path, required=True, help="Dir from collect_rollouts (ep_*.npz + index.json)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--demo-fraction", type=float, default=1.0, help="Fraction of demo train episodes to mix in as anchor (0=rollout only)")
    ap.add_argument("--success-only", action="store_true", default=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ckpt = torch.load(args.action_run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = ExperimentConfig().to_jsonable()
    config.update(ckpt["config"])
    normalizers = {k: Normalizer(np.asarray(v["mean"], np.float32), np.asarray(v["std"], np.float32))
                   for k, v in ckpt["normalizers"].items()}
    config["action_dim"] = int(normalizers["action"].mean.shape[-1])
    # fine-tune overrides (regularisation); everything else (model_kind, horizons, use_vision,
    # target_mode, weights, dropout, weight_decay) is inherited from the demo run.
    config["epochs"] = args.epochs
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size

    manifest = json.loads((args.action_run_dir / "manifest.json").read_text())
    demo_eps = _episodes_from_manifest_train(manifest, args.demo_fraction, args.seed)
    rollout_eps = _rollout_success_episodes(args.rollout_features, args.success_only)
    if len(rollout_eps) == 0:
        raise RuntimeError(f"No success rollout episodes in {args.rollout_features}")
    print(f"Fine-tune pool: {len(demo_eps)} demo (anchor, frac={args.demo_fraction}) + "
          f"{len(rollout_eps)} rollout-success episodes", flush=True)

    pool = demo_eps + rollout_eps
    random.Random(args.seed).shuffle(pool)
    n_val = max(1, round(len(pool) * args.val_ratio))
    val_eps, train_eps = pool[:n_val], pool[n_val:]
    prev_h, pred_h, obs_h = config["prev_horizon"], config["pred_horizon"], config.get("obs_horizon", 1)

    def loader(eps, shuffle):
        ds = ChunkDataset(FeatureStore(eps), normalizers, prev_h, pred_h, obs_h)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True), len(ds)

    train_loader, n_train = loader(train_eps, True)
    val_loader, n_val_s = loader(val_eps, False)
    print(f"chunks: train={n_train} val={n_val_s} | warm-start from {args.action_run_dir/'best_model.pt'}", flush=True)

    result = train_model(
        train_loader, val_loader, val_loader, normalizers, config, args.output_dir,
        init_state_dict=ckpt["model_state"],
    )
    # manifest for downstream tools (build_envelope_gate etc.): the fine-tune train/val split.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [{"episode_index": e.episode_index, "task": e.task, "length": e.length,
                 "parquet_path": e.parquet_path, "feature_path": e.feature_path} for e in pool]
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "repo_id": manifest.get("repo_id"), "config": config, "episodes": episodes,
        "splits": {"train": [e.episode_index for e in train_eps],
                   "val": [e.episode_index for e in val_eps],
                   "test": [e.episode_index for e in val_eps]},
        "finetune": {"source": str(args.action_run_dir), "rollout_features": str(args.rollout_features),
                     "n_demo": len(demo_eps), "n_rollout_success": len(rollout_eps),
                     "demo_fraction": args.demo_fraction, "lr": args.lr, "epochs": args.epochs},
    }, indent=2) + "\n")
    print(f"DONE fine-tune: best_val_rmse={result['best_val_rmse']:.5f} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
