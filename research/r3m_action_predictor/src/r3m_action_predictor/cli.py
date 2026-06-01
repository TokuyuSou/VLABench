from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import make_loaders
from .hf_data import (
    cleanup_episode_parquets,
    download_episode_parquets,
    download_metadata,
    select_episodes,
    split_episodes,
    write_manifest,
)
from .r3m_features import extract_all_features
from .train import train_model


def parse_args() -> ExperimentConfig:
    defaults = ExperimentConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--task-name", default=defaults.task_name)
    p.add_argument("--task-regex", default=defaults.task_regex)
    p.add_argument("--max-episodes", type=int, default=defaults.max_episodes)
    p.add_argument("--train-ratio", type=float, default=defaults.train_ratio)
    p.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    p.add_argument("--no-stratified-split", dest="stratified_split", action="store_false")
    p.add_argument("--prev-horizon", type=int, default=defaults.prev_horizon)
    p.add_argument("--pred-horizon", type=int, default=defaults.pred_horizon)
    p.add_argument("--obs-horizon", type=int, default=defaults.obs_horizon)
    p.add_argument("--r3m-model", default=defaults.r3m_model, choices=("resnet18", "resnet34", "resnet50"))
    p.add_argument("--embedding-batch-size", type=int, default=defaults.embedding_batch_size)
    p.add_argument("--force-features", action="store_true")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument(
        "--model-kind",
        default=defaults.model_kind,
        choices=("mlp_gru", "prob_transformer", "shared_prefix_risk_transformer"),
    )
    p.add_argument("--target-mode", default=defaults.target_mode, choices=("absolute", "residual"))
    p.add_argument("--view-dim", type=int, default=defaults.view_dim)
    p.add_argument("--width", type=int, default=defaults.width)
    p.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    p.add_argument("--transformer-layers", type=int, default=defaults.transformer_layers)
    p.add_argument("--transformer-heads", type=int, default=defaults.transformer_heads)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--nll-weight", type=float, default=defaults.nll_weight)
    p.add_argument("--mse-weight", type=float, default=defaults.mse_weight)
    p.add_argument("--smoothness-weight", type=float, default=defaults.smoothness_weight)
    p.add_argument("--prefix-risk-weight", type=float, default=defaults.prefix_risk_weight)
    p.add_argument("--prefix-risk-warmup-epochs", type=int, default=defaults.prefix_risk_warmup_epochs)
    p.add_argument("--prefix-pos-threshold", type=float, default=defaults.prefix_pos_threshold)
    p.add_argument("--prefix-rot-threshold", type=float, default=defaults.prefix_rot_threshold)
    p.add_argument("--prefix-gripper-threshold", type=float, default=defaults.prefix_gripper_threshold)
    p.add_argument("--critical-action-weight", type=float, default=defaults.critical_action_weight)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    p.add_argument("--meta-dir", type=Path, default=defaults.meta_dir)
    p.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    p.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    args = p.parse_args()
    return replace(defaults, **vars(args))


def main() -> None:
    cfg = parse_args()
    config = cfg.to_jsonable()
    _set_seed(cfg.seed)

    _, _, episodes_jsonl = download_metadata(cfg.meta_dir)
    selected = select_episodes(episodes_jsonl, cfg.task_regex, cfg.max_episodes, cfg.seed)
    if not selected:
        raise RuntimeError(f"No episodes matched {cfg.task_regex!r}")

    print(
        f"Selected {len(selected)} official episodes for {cfg.task_name}; "
        f"{sum(e['length'] for e in selected)} frames",
        flush=True,
    )
    episodes, extracted = _prepare_feature_files(selected, cfg, config)
    print(f"Feature extraction completed; newly extracted episodes: {extracted}", flush=True)

    train_eps, val_eps, test_eps = split_episodes(
        episodes,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.seed,
        stratified=cfg.stratified_split,
    )
    write_manifest(
        cfg.output_dir / "manifest.json",
        episodes,
        {"train": train_eps, "val": val_eps, "test": test_eps},
        config,
    )
    print(f"Episode split: train={len(train_eps)} val={len(val_eps)} test={len(test_eps)}", flush=True)

    train_loader, val_loader, test_loader, normalizers, sample_counts = make_loaders(
        train_eps,
        val_eps,
        test_eps,
        cfg.prev_horizon,
        cfg.pred_horizon,
        cfg.batch_size,
        cfg.obs_horizon,
    )
    print(f"Frame-chunk samples: {sample_counts}", flush=True)

    result = train_model(train_loader, val_loader, test_loader, normalizers, config, cfg.output_dir)
    result["sample_counts"] = sample_counts
    result["episode_counts"] = {"train": len(train_eps), "val": len(val_eps), "test": len(test_eps)}
    result["config"] = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    print("Final test metrics:", flush=True)
    for key, value in result["test_metrics"].items():
        print(f"  {key}: {value:.6f}", flush=True)
    print(f"Wrote {cfg.output_dir / 'metrics.json'}", flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def _prepare_feature_files(selected: list[dict], cfg: ExperimentConfig, config: dict) -> tuple[list, int]:
    episodes = []
    extracted_total = 0
    batch_size = 50
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        batch_eps = download_episode_parquets(batch, cfg.data_dir)
        episodes.extend(batch_eps)
        write_manifest(cfg.output_dir / "manifest.json", episodes, {}, config)
        extracted = extract_all_features(
            batch_eps,
            r3m_model=cfg.r3m_model,
            batch_size=cfg.embedding_batch_size,
            force=cfg.force_features,
            cache_dir=cfg.cache_dir,
        )
        extracted_total += extracted
        removed_count, removed_bytes = cleanup_episode_parquets(batch_eps)
        if removed_count:
            print(
                f"Removed {removed_count} parquet files after feature extraction; "
                f"freed {removed_bytes / (1024**3):.2f} GiB",
                flush=True,
            )
    return episodes, extracted_total


if __name__ == "__main__":
    main()
