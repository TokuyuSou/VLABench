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
from .hf_data import EpisodeInfo, cleanup_episode_parquets, download_episode_parquets, download_metadata, select_episodes
from .metrics import log_scalars, make_tb_writer
from .model import build_model
from .r3m_features import extract_all_features
from .risk import (
    RiskHead,
    RiskLabelConfig,
    action_output_to_raw_and_norm,
    chunk_cost_labels,
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
    action_config["action_dim"] = int(normalizers["action"].mean.shape[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    episodes = {e["episode_index"]: EpisodeInfo(**e) for e in manifest["episodes"]}

    if args.external_risk_episodes > 0:
        label_eps = build_external_risk_episodes(action_config, manifest, args)
    else:
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

    repr_std = action_std  # action normalizer std lives in the model (sin/cos repr) space

    opt = torch.optim.AdamW(risk_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = make_tb_writer(args.output_dir)
    history = []
    best_state = None
    best_score = -1e30
    for epoch in range(1, args.epochs + 1):
        if args.objective == "regression":
            loss = train_one_epoch_reg(
                risk_head, action_model, train_loader, opt, action_mean, action_std,
                action_config["target_mode"], repr_std, args.grip_weight, args.label_horizon, device,
            )
            calib = evaluate_reg(
                risk_head, action_model, calib_loader, action_mean, action_std,
                action_config["target_mode"], repr_std, args.grip_weight, args.label_horizon,
                args.target_coverage, device,
            )
            score = -calib["selected_true_cost"]  # lower true cost among substituted = better
            print(
                f"epoch {epoch:03d} loss={loss:.4f} spearman={calib['spearman']:.3f} mae={calib['mae']:.4f} "
                f"@cov={calib['selected_coverage']:.3f}: true_cost={calib['selected_true_cost']:.4f} "
                f"grip_mismatch={calib['selected_grip_mismatch']:.3f} thr={calib['selected_threshold']:.4f}",
                flush=True,
            )
        else:
            loss = train_one_epoch(
                risk_head, action_model, train_loader, opt, action_mean, action_std,
                action_config["target_mode"], label_cfg, device,
            )
            calib = evaluate_risk(
                risk_head, action_model, calib_loader, action_mean, action_std,
                action_config["target_mode"], label_cfg, args.target_precision, args.target_coverage, device,
            )
            score = threshold_score(calib, args.target_precision, args.target_coverage)
            print(
                f"epoch {epoch:03d} loss={loss:.4f} auc={calib['auc']:.4f} "
                f"safe_rate={calib['safe_rate']:.3f} thr={calib['selected_threshold']:.4f} "
                f"precision={calib['selected_precision']:.3f} coverage={calib['selected_coverage']:.3f}",
                flush=True,
            )
        history.append({"epoch": epoch, "train_loss": loss, **calib})
        log_scalars(writer, "train", {"loss": loss, "score": score}, epoch)
        log_scalars(writer, "calib", calib, epoch)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in risk_head.state_dict().items()}

    if best_state is not None:
        risk_head.load_state_dict(best_state)
    if args.objective == "regression":
        final = evaluate_reg(
            risk_head, action_model, calib_loader, action_mean, action_std,
            action_config["target_mode"], repr_std, args.grip_weight, args.label_horizon,
            args.target_coverage, device,
        )
    else:
        final = evaluate_risk(
            risk_head, action_model, calib_loader, action_mean, action_std,
            action_config["target_mode"], label_cfg, args.target_precision, args.target_coverage, device,
        )
    log_scalars(writer, "final", final, args.epochs)
    if writer is not None:
        writer.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "risk_state": risk_head.state_dict(),
            "input_dim": input_dim,
            "objective": args.objective,
            "grip_weight": args.grip_weight,
            "config": vars(args),
            "action_config": action_config,
            "label_config": vars(label_cfg),
            "selected_threshold": final["selected_threshold"],
            "calib_metrics": final,
            "history": history,
            "train_episode_indices": [ep.episode_index for ep in train_eps],
            "calib_episode_indices": [ep.episode_index for ep in calib_eps],
            "label_episode_indices": [ep.episode_index for ep in label_eps],
        },
        args.output_dir / "risk_head.pt",
    )
    (args.output_dir / "risk_metrics.json").write_text(
        json.dumps({"final": final, "history": history}, indent=2) + "\n"
    )
    print(f"Wrote {args.output_dir / 'risk_head.pt'}")


def build_external_risk_episodes(action_config, action_manifest, args) -> list[EpisodeInfo]:
    task_regex = args.task_regex or action_config["task_regex"]
    meta_dir = args.meta_dir or Path(action_config["meta_dir"])
    data_dir = args.data_dir or Path(action_config["data_dir"])
    cache_dir = args.cache_dir or Path(action_config["cache_dir"])

    _, _, episodes_jsonl = download_metadata(meta_dir)
    candidates = select_episodes(
        episodes_jsonl,
        task_regex=task_regex,
        max_episodes=0,
        seed=args.seed,
    )
    action_episode_indices = {int(ep["episode_index"]) for ep in action_manifest["episodes"]}
    candidates = [ep for ep in candidates if int(ep["episode_index"]) not in action_episode_indices]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.external_risk_episodes]
    if len(candidates) < args.external_risk_episodes:
        raise RuntimeError(
            f"Only found {len(candidates)} external risk episodes for {task_regex!r}; "
            f"requested {args.external_risk_episodes}"
        )
    candidates.sort(key=lambda ep: int(ep["episode_index"]))
    episodes: list[EpisodeInfo] = []
    extracted = 0
    removed_count = 0
    removed_bytes = 0
    for start in range(0, len(candidates), 50):
        batch = candidates[start : start + 50]
        batch_eps = download_episode_parquets(batch, data_dir)
        episodes.extend(batch_eps)
        extracted += extract_all_features(
            batch_eps,
            r3m_model=action_config["r3m_model"],
            batch_size=action_config["embedding_batch_size"],
            force=False,
            cache_dir=cache_dir,
        )
        batch_removed_count, batch_removed_bytes = cleanup_episode_parquets(batch_eps)
        removed_count += batch_removed_count
        removed_bytes += batch_removed_bytes
    print(
        f"External risk episodes: {len(episodes)} selected outside action predictor manifest; "
        f"newly extracted features: {extracted}; "
        f"removed parquets: {removed_count} ({removed_bytes / (1024**3):.2f} GiB)",
        flush=True,
    )
    return episodes


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


def train_one_epoch_reg(
    risk_head, action_model, loader, opt, action_mean, action_std, target_mode, repr_std, grip_weight, horizon, device
):
    """Regress the continuous gripper-weighted normalized chunk cost (Huber loss)."""
    risk_head.train()
    losses = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.no_grad():
            pred_raw, pred_norm_abs, pred_std_norm = action_output_to_raw_and_norm(
                action_model(batch["embeddings"], batch["state"], batch["prev_actions"]),
                batch, action_mean, action_std, target_mode,
            )
            cost = chunk_cost_labels(pred_raw, batch["raw_target"], repr_std, grip_weight, horizon)
            features = make_risk_features(
                batch["embeddings"], batch["state"], batch["prev_actions"], pred_norm_abs, pred_std_norm
            )
        pred_cost = F.softplus(risk_head(features))  # nonnegative
        loss = F.smooth_l1_loss(pred_cost, cost)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(risk_head.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_reg(
    risk_head, action_model, loader, action_mean, action_std, target_mode, repr_std, grip_weight, horizon,
    target_coverage, device,
):
    risk_head.eval()
    pred_costs, true_costs, grip_mm = [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        pred_raw, pred_norm_abs, pred_std_norm = action_output_to_raw_and_norm(
            action_model(batch["embeddings"], batch["state"], batch["prev_actions"]),
            batch, action_mean, action_std, target_mode,
        )
        cost = chunk_cost_labels(pred_raw, batch["raw_target"], repr_std, grip_weight, horizon)
        feat = make_risk_features(
            batch["embeddings"], batch["state"], batch["prev_actions"], pred_norm_abs, pred_std_norm
        )
        pred_cost = F.softplus(risk_head(feat))
        h = min(horizon, pred_raw.shape[1], batch["raw_target"].shape[1])
        gm = ((pred_raw[:, :h, 6] > 0.5) != (batch["raw_target"][:, :h, 6] > 0.5)).any(dim=1).float()
        pred_costs.append(pred_cost.cpu().numpy())
        true_costs.append(cost.cpu().numpy())
        grip_mm.append(gm.cpu().numpy())
    pc = np.concatenate(pred_costs)
    tc = np.concatenate(true_costs)
    gm = np.concatenate(grip_mm)
    n = len(pc)
    k = max(1, int(round(target_coverage * n)))
    sel = np.argsort(pc)[:k]  # substitute the lowest-predicted-cost chunks
    tau = float(pc[sel].max())
    return {
        "spearman": _spearman(pc, tc),
        "pearson": _pearson(pc, tc),
        "mae": float(np.mean(np.abs(pc - tc))),
        "true_cost_mean": float(tc.mean()),
        "pred_cost_mean": float(pc.mean()),
        "selected_coverage": float(k / n),
        "selected_true_cost": float(tc[sel].mean()),
        "selected_grip_mismatch": float(gm[sel].mean()),
        "overall_grip_mismatch": float(gm.mean()),
        "selected_cost_threshold": tau,
        "selected_threshold": float(1.0 / (1.0 + tau)),  # risk_safe_probability-space for live_eval
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--action-run-dir", type=Path, default=Path("research/r3m_action_predictor/outputs/add_condiment_r3m18_residual_transformer_200"))
    p.add_argument("--output-dir", type=Path, default=Path("research/r3m_action_predictor/outputs/add_condiment_risk_head_demo"))
    p.add_argument("--label-splits", default="val,test")
    p.add_argument("--external-risk-episodes", type=int, default=0)
    p.add_argument("--task-regex", default=None)
    p.add_argument("--meta-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--train-episode-fraction", type=float, default=0.7)
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--pos-threshold", type=float, default=0.03)
    p.add_argument("--rot-threshold", type=float, default=0.20)
    p.add_argument("--gripper-threshold", type=float, default=0.5)
    p.add_argument("--objective", choices=("classification", "regression"), default="classification")
    p.add_argument("--grip-weight", type=float, default=3.0)
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
