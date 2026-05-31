from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ChunkDataset, FeatureStore, Normalizer
from .hf_data import EpisodeInfo
from .metrics import batch_to_device, evaluate
from .model import build_model


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = _with_defaults(ckpt["config"])
    normalizers = {
        k: Normalizer(np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
        for k, v in ckpt["normalizers"].items()
    }

    manifest = json.loads((run_dir / "manifest.json").read_text())
    episodes = {e["episode_index"]: EpisodeInfo(**e) for e in manifest["episodes"]}
    loaders = {}
    for split, ids in manifest["splits"].items():
        store = FeatureStore([episodes[i] for i in ids])
        ds = ChunkDataset(store, normalizers, config["prev_horizon"], config["pred_horizon"])
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    sample = next(iter(loaders["train"]))
    model = build_model(config, embed_dim=sample["embeddings"].shape[-1], num_views=sample["embeddings"].shape[1])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    report = {"run_dir": str(run_dir), "config": config, "splits": {}}
    for split, loader in loaders.items():
        split_report = evaluate(model, loader, normalizers["action"], device, config.get("target_mode", "absolute"))
        split_report.update(_detailed_metrics(model, loader, normalizers["action"], device, config.get("target_mode", "absolute")))
        report["splits"][split] = split_report

    out_path = run_dir / args.output_name
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(_short_report(report), indent=2))
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--output-name", default="diagnostics.json")
    return p.parse_args()


def _with_defaults(config: dict) -> dict:
    defaults = ExperimentConfig().to_jsonable()
    defaults.update(config)
    return defaults


@torch.no_grad()
def _detailed_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    action_norm: Normalizer,
    device: torch.device,
    target_mode: str,
) -> dict:
    model.eval()
    action_mean = torch.from_numpy(action_norm.mean).to(device=device, dtype=torch.float32)
    action_std = torch.from_numpy(action_norm.std).to(device=device, dtype=torch.float32)
    horizon_sq = []
    task_sq: dict[str, list[float]] = defaultdict(list)
    task_count: dict[str, int] = defaultdict(int)
    episode_sq: dict[str, list[float]] = defaultdict(list)
    confidence_rows = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        repeat = batch["raw_prev_actions"][:, -1:, :].repeat(1, batch["raw_target"].shape[1], 1)
        model_out = model(batch["embeddings"], batch["state"], batch["prev_actions"])
        if isinstance(model_out, tuple):
            pred_norm, log_std = model_out
            pred_std = torch.exp(log_std) * action_std
        else:
            pred_norm, pred_std = model_out, None
        if target_mode == "residual":
            pred = repeat + pred_norm * action_std
        else:
            pred = pred_norm * action_std + action_mean
        err = (pred - batch["raw_target"]).detach().cpu().numpy()
        sq = np.square(err).mean(axis=2)
        horizon_sq.append(sq)

        tasks = batch["task"]
        episode_ids = batch["episode_index"].detach().cpu().numpy().tolist()
        for i, task in enumerate(tasks):
            task_sq[task].append(float(sq[i].mean()))
            task_count[task] += 1
            episode_sq[str(episode_ids[i])].append(float(sq[i].mean()))

        if pred_std is not None:
            std_scalar = pred_std.detach().cpu().numpy().reshape(err.shape[0], -1).mean(axis=1)
            err_scalar = (np.abs(err) / action_norm.std.reshape(1, 1, -1)).reshape(err.shape[0], -1).mean(axis=1)
            for s, e in zip(std_scalar.tolist(), err_scalar.tolist()):
                confidence_rows.append({"predicted_std": s, "confidence": 1.0 / (1.0 + s), "norm_abs_error": e})

    horizon_sq_arr = np.concatenate(horizon_sq, axis=0)
    out = {
        "horizon_rmse": np.sqrt(horizon_sq_arr.mean(axis=0)).tolist(),
        "task_rmse": {
            task: float(np.sqrt(np.mean(vals)))
            for task, vals in sorted(task_sq.items())
        },
        "task_sample_count": dict(sorted(task_count.items())),
        "worst_episodes": sorted(
            [
                {"episode_index": int(ep), "rmse": float(np.sqrt(np.mean(vals)))}
                for ep, vals in episode_sq.items()
            ],
            key=lambda x: x["rmse"],
            reverse=True,
        )[:10],
    }
    if confidence_rows:
        out["confidence_examples"] = confidence_rows[:10]
    return out


def _short_report(report: dict) -> dict:
    out = {}
    for split, metrics in report["splits"].items():
        out[split] = {
            "model_rmse": metrics["model_rmse"],
            "repeat_last_rmse": metrics["repeat_last_rmse"],
            "horizon_rmse": metrics["horizon_rmse"],
            "task_rmse": metrics["task_rmse"],
        }
        for key in ("predicted_std_mean", "confidence_mean", "confidence_error_corr"):
            if key in metrics:
                out[split][key] = metrics[key]
    return out


if __name__ == "__main__":
    main()

