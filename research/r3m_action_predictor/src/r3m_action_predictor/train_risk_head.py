from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ChunkDataset, FeatureStore, Normalizer
from .hf_data import EpisodeInfo
from .model import build_model
from .risk import (
    RiskHead,
    RiskLabelConfig,
    action_output_to_raw_and_norm,
    make_risk_features,
    safe_labels,
)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    run_dir = args.action_run_dir
    action_ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    action_config = ExperimentConfig().to_jsonable()
    action_config.update(action_ckpt["config"])
    normalizers = {
        k: Normalizer(np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
        for k, v in action_ckpt["normalizers"].items()
    }
    manifest = json.loads((run_dir / "manifest.json").read_text())
    episodes = {e["episode_index"]: EpisodeInfo(**e) for e in manifest["episodes"]}

    label_splits = args.label_splits.split(",")
    label_eps = []
    for split in label_splits:
        label_eps.extend(episodes[i] for i in manifest["splits"][split])
    rng = random.Random(args.seed)
    rng.shuffle(label_eps)
    n_train = max(1, int(round(len(label_eps) * args.train_episode_fraction)))
    train_eps = label_eps[:n_train]
    calib_eps = label_eps[n_train:] or label_eps[-1:]

    train_loader, calib_loader, action_model, risk_head, input_dim = build_components(
        train_eps,
        calib_eps,
        action_ckpt,
        action_config,
        normalizers,
        args,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "train_episodes": len(train_eps),
                    "calib_episodes": len(calib_eps),
                    "input_dim": input_dim,
                    "train_samples": len(train_loader.dataset),
                    "calib_samples": len(calib_loader.dataset),
                },
                indent=2,
            )
        )
        return

    device = torch.device(args.device)
    action_model.to(device).eval()
    risk_head.to(device)
    action_mean = torch.from_numpy(normalizers["action"].mean).to(device)
    action_std = torch.from_numpy(normalizers["action"].std).to(device)
    label_cfg = RiskLabelConfig(
        horizon=args.label_horizon,
        pos_threshold=args.pos_threshold,
        rot_threshold=args.rot_threshold,
        gripper_threshold=args.gripper_threshold,
    )

    opt = torch.optim.AdamW(risk_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            risk_head,
            action_model,
            train_loader,
            opt,
            action_mean,
            action_std,
            action_config["target_mode"],
            label_cfg,
            device,
        )
        calib = evaluate_risk(
            risk_head,
            action_model,
            calib_loader,
            action_mean,
            action_std,
            action_config["target_mode"],
            label_cfg,
            args.target_precision,
            args.target_coverage,
            device,
        )
        history.append({"epoch": epoch, "train_loss": loss, **calib})
        score = threshold_score(calib, args.target_precision, args.target_coverage)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in risk_head.state_dict().items()}
        print(
            f"epoch {epoch:03d} loss={loss:.4f} auc={calib['auc']:.4f} "
            f"safe_rate={calib['safe_rate']:.3f} thr={calib['selected_threshold']:.4f} "
            f"precision={calib['selected_precision']:.3f} coverage={calib['selected_coverage']:.3f}",
            flush=True,
        )

    if best_state is not None:
        risk_head.load_state_dict(best_state)
    final = evaluate_risk(
        risk_head,
        action_model,
        calib_loader,
        action_mean,
        action_std,
        action_config["target_mode"],
        label_cfg,
        args.target_precision,
        args.target_coverage,
        device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "risk_state": risk_head.state_dict(),
            "input_dim": input_dim,
            "config": vars(args),
            "action_config": action_config,
            "label_config": vars(label_cfg),
            "selected_threshold": final["selected_threshold"],
            "calib_metrics": final,
            "history": history,
            "train_episode_indices": [ep.episode_index for ep in train_eps],
            "calib_episode_indices": [ep.episode_index for ep in calib_eps],
        },
        args.output_dir / "risk_head.pt",
    )
    (args.output_dir / "risk_metrics.json").write_text(
        json.dumps({"final": final, "history": history}, indent=2) + "\n"
    )
    print(f"Wrote {args.output_dir / 'risk_head.pt'}")


def build_components(train_eps, calib_eps, action_ckpt, action_config, normalizers, args):
    train_store = FeatureStore(train_eps)
    sample_ep = train_store.episodes[0]
    action_model = build_model(
        action_config,
        embed_dim=sample_ep["embeddings"].shape[-1],
        num_views=sample_ep["embeddings"].shape[1],
    )
    action_model.load_state_dict(action_ckpt["model_state"])
    dummy = ChunkDataset(train_store, normalizers, action_config["prev_horizon"], action_config["pred_horizon"])[0]
    with torch.no_grad():
        model_out = action_model(
            dummy["embeddings"][None],
            dummy["state"][None],
            dummy["prev_actions"][None],
        )
        action_mean = torch.from_numpy(normalizers["action"].mean)
        action_std = torch.from_numpy(normalizers["action"].std)
        _, pred_norm_abs, pred_std_norm = action_output_to_raw_and_norm(
            model_out,
            {k: v[None] if torch.is_tensor(v) else v for k, v in dummy.items()},
            action_mean,
            action_std,
            action_config["target_mode"],
        )
        features = make_risk_features(
            dummy["embeddings"][None],
            dummy["state"][None],
            dummy["prev_actions"][None],
            pred_norm_abs,
            pred_std_norm,
        )
    risk_head = RiskHead(features.shape[-1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    train_ds = ChunkDataset(train_store, normalizers, action_config["prev_horizon"], action_config["pred_horizon"])
    calib_ds = ChunkDataset(FeatureStore(calib_eps), normalizers, action_config["prev_horizon"], action_config["pred_horizon"])
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True),
        action_model,
        risk_head,
        features.shape[-1],
    )


def train_one_epoch(risk_head, action_model, loader, opt, action_mean, action_std, target_mode, label_cfg, device):
    risk_head.train()
    losses = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.no_grad():
            pred_raw, pred_norm_abs, pred_std_norm = action_output_to_raw_and_norm(
                action_model(batch["embeddings"], batch["state"], batch["prev_actions"]),
                batch,
                action_mean,
                action_std,
                target_mode,
            )
            labels, _ = safe_labels(pred_raw, batch["raw_target"], label_cfg)
            features = make_risk_features(batch["embeddings"], batch["state"], batch["prev_actions"], pred_norm_abs, pred_std_norm)
        logits = risk_head(features)
        pos_weight = _pos_weight(labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(risk_head.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_risk(
    risk_head,
    action_model,
    loader,
    action_mean,
    action_std,
    target_mode,
    label_cfg,
    target_precision,
    target_coverage,
    device,
):
    risk_head.eval()
    probs, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        pred_raw, pred_norm_abs, pred_std_norm = action_output_to_raw_and_norm(
            action_model(batch["embeddings"], batch["state"], batch["prev_actions"]),
            batch,
            action_mean,
            action_std,
            target_mode,
        )
        y, _ = safe_labels(pred_raw, batch["raw_target"], label_cfg)
        x = make_risk_features(batch["embeddings"], batch["state"], batch["prev_actions"], pred_norm_abs, pred_std_norm)
        probs.append(torch.sigmoid(risk_head(x)).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    selected = select_threshold(p, y, target_precision=target_precision, target_coverage=target_coverage)
    return {
        "safe_rate": float(y.mean()),
        "auc": auc_score(p, y),
        **selected,
    }


def select_threshold(probs: np.ndarray, labels: np.ndarray, target_precision: float, target_coverage: float) -> dict:
    thresholds = np.unique(probs)[::-1]
    best = None
    for thr in thresholds:
        mask = probs >= thr
        coverage = float(mask.mean())
        if not mask.any():
            continue
        precision = float(labels[mask].mean())
        row = {"selected_threshold": float(thr), "selected_precision": precision, "selected_coverage": coverage}
        if precision >= target_precision and coverage <= target_coverage:
            if best is None or row["selected_coverage"] > best["selected_coverage"]:
                best = row
    if best is not None:
        return best
    # Fallback: most conservative threshold with maximum precision.
    rows = []
    for thr in thresholds:
        mask = probs >= thr
        if mask.any():
            rows.append((float(labels[mask].mean()), float(mask.mean()), float(thr)))
    precision, coverage, thr = max(rows, key=lambda x: (x[0], -x[1]))
    return {"selected_threshold": thr, "selected_precision": precision, "selected_coverage": coverage}


def threshold_score(metrics: dict, target_precision: float, target_coverage: float) -> float:
    precision = metrics["selected_precision"]
    coverage = metrics["selected_coverage"]
    return precision - abs(coverage - target_coverage) * 0.1


def auc_score(probs: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = probs[labels]
    neg = probs[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _pos_weight(labels: torch.Tensor) -> torch.Tensor:
    pos = labels.sum().clamp_min(1.0)
    neg = (labels.numel() - labels.sum()).clamp_min(1.0)
    return (neg / pos).detach()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--action-run-dir", type=Path, default=Path("research/r3m_action_predictor/outputs/add_condiment_r3m18_residual_transformer_200"))
    p.add_argument("--output-dir", type=Path, default=Path("research/r3m_action_predictor/outputs/add_condiment_risk_head_demo"))
    p.add_argument("--label-splits", default="val,test")
    p.add_argument("--train-episode-fraction", type=float, default=0.7)
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--pos-threshold", type=float, default=0.03)
    p.add_argument("--rot-threshold", type=float, default=0.20)
    p.add_argument("--gripper-threshold", type=float, default=0.5)
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--target-coverage", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
