from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .action_repr import repr_std_to_euler_std
from .data import Normalizer
from .risk import RiskLabelConfig, action_output_to_raw_and_norm, safe_prefix_labels
from .model import run_model, split_model_output


def make_tb_writer(out_dir):
    """Return a TensorBoard SummaryWriter writing to ``<out_dir>/tb``, or None if the
    tensorboard package is unavailable (logging is best-effort and never blocks training)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # tensorboard not installed
        print(f"TensorBoard unavailable ({exc}); skipping TB logging.", flush=True)
        return None
    return SummaryWriter(log_dir=str(Path(out_dir) / "tb"))


def log_scalars(writer, prefix: str, metrics: dict, step: int) -> None:
    """Log every numeric value in ``metrics`` under ``prefix/`` at the given step."""
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _wrap_rot_err(err: np.ndarray) -> np.ndarray:
    """Wrap the Euler-rotation error columns (3:6) into (-pi, pi] so that physically
    identical orientations (e.g. predicting +pi for a -pi target) score ~zero."""
    err = np.asarray(err, dtype=np.float64).copy()
    err[..., 3:6] = (err[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
    return err


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    normalizers: dict[str, Normalizer],
    device: torch.device,
    target_mode: str = "absolute",
) -> dict[str, float]:
    model.eval()
    # "action" is the sin/cos repr normalizer (model decode); "action_euler" is the
    # raw 7-D Euler normalizer (constant-mean baseline and NRMSE scaling).
    action_norm = normalizers["action"]
    euler_std = normalizers["action_euler"].std
    euler_mean = torch.from_numpy(normalizers["action_euler"].mean).to(device=device, dtype=torch.float32)
    action_mean = torch.from_numpy(action_norm.mean).to(device=device, dtype=torch.float32)
    action_std = torch.from_numpy(action_norm.std).to(device=device, dtype=torch.float32)

    preds, targets, repeat_preds, mean_preds = [], [], [], []
    pred_stds = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        target = batch["raw_target"]
        repeat = batch["raw_prev_actions"][:, -1:, :].repeat(1, target.shape[1], 1)
        model_out = run_model(model, batch)
        _, _, prefix_logits = split_model_output(model_out)
        # Decode model output from sin/cos repr back to 7-D Euler.
        pred, _, pred_std_norm = action_output_to_raw_and_norm(
            model_out, batch, action_mean, action_std, target_mode
        )
        mean = euler_mean.view(1, 1, -1).repeat(target.shape[0], target.shape[1], 1)
        preds.append(pred.cpu())
        targets.append(target.cpu())
        repeat_preds.append(repeat.cpu())
        mean_preds.append(mean.cpu())
        if pred_std_norm is not None:
            pred_stds.append(repr_std_to_euler_std(pred_std_norm * action_std).cpu())
        if prefix_logits is not None:
            labels, _ = safe_prefix_labels(
                pred,
                target,
                RiskLabelConfig(horizon=prefix_logits.shape[1]),
            )
            probs = torch.sigmoid(prefix_logits[:, : labels.shape[1]])
            out_prefix = getattr(evaluate, "_prefix_cache", None)
            if out_prefix is None:
                out_prefix = {"probs": [], "labels": []}
                setattr(evaluate, "_prefix_cache", out_prefix)
            out_prefix["probs"].append(probs.cpu())
            out_prefix["labels"].append(labels.cpu())

    target = torch.cat(targets).numpy()
    pred = torch.cat(preds).numpy()
    repeat = torch.cat(repeat_preds).numpy()
    mean = torch.cat(mean_preds).numpy()

    out = {}
    out.update(_metrics_for("model", pred, target, euler_std))
    out.update(_metrics_for("repeat_last", repeat, target, euler_std))
    out.update(_metrics_for("train_mean", mean, target, euler_std))
    out["rmse_improvement_over_repeat"] = (
        (out["repeat_last_rmse"] - out["model_rmse"]) / max(out["repeat_last_rmse"], 1e-12)
    )
    out["first_rmse_improvement_over_repeat"] = (
        (out["repeat_last_first_rmse"] - out["model_first_rmse"])
        / max(out["repeat_last_first_rmse"], 1e-12)
    )
    if pred_stds:
        std = torch.cat(pred_stds).numpy()
        norm_err = np.abs(_wrap_rot_err(pred - target)) / euler_std.reshape(1, 1, -1)
        std_scalar = std.reshape(std.shape[0], -1).mean(axis=1)
        err_scalar = norm_err.reshape(norm_err.shape[0], -1).mean(axis=1)
        confidence = 1.0 / (1.0 + std_scalar)
        out["predicted_std_mean"] = float(np.mean(std))
        out["predicted_std_p90"] = float(np.percentile(std, 90))
        out["confidence_mean"] = float(np.mean(confidence))
        out["confidence_p10"] = float(np.percentile(confidence, 10))
        out["uncertainty_error_corr"] = _safe_corr(std_scalar, err_scalar)
        out["confidence_error_corr"] = _safe_corr(confidence, err_scalar)
    prefix_cache = getattr(evaluate, "_prefix_cache", None)
    if prefix_cache is not None:
        probs = torch.cat(prefix_cache["probs"]).numpy()
        labels = torch.cat(prefix_cache["labels"]).numpy()
        out.update(_prefix_metrics(probs, labels))
        delattr(evaluate, "_prefix_cache")
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _metrics_for(name: str, pred: np.ndarray, target: np.ndarray, action_std: np.ndarray) -> dict[str, float]:
    err = _wrap_rot_err(pred - target)
    mse = float(np.mean(np.square(err)))
    var = float(np.mean(np.square(target - target.mean(axis=(0, 1), keepdims=True))))
    gripper_pred = pred[..., 6] > 0.5
    gripper_target = target[..., 6] > 0.5
    return {
        f"{name}_mae": float(np.mean(np.abs(err))),
        f"{name}_rmse": math.sqrt(mse),
        f"{name}_nrmse": float(np.sqrt(np.mean(np.square(err / action_std.reshape(1, 1, -1))))),
        f"{name}_first_mae": float(np.mean(np.abs(err[:, 0]))),
        f"{name}_first_rmse": float(np.sqrt(np.mean(np.square(err[:, 0])))),
        f"{name}_continuous_rmse": float(np.sqrt(np.mean(np.square(err[..., :6])))),
        f"{name}_gripper_accuracy": float(np.mean(gripper_pred == gripper_target)),
        f"{name}_r2": 1.0 - mse / max(var, 1e-12),
    }


def _prefix_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    labels_bool = np.asarray(labels > 0.5, dtype=bool)
    flat_p = probs.reshape(-1)
    flat_y = labels_bool.reshape(-1)
    out = {
        "prefix_safe_rate": float(np.mean(flat_y)),
        "prefix_prob_mean": float(np.mean(flat_p)),
        "prefix_auc": _auc(flat_p, flat_y),
    }
    for threshold in (0.5, 0.7, 0.9):
        mask = flat_p >= threshold
        out[f"prefix_precision_at_{threshold:g}"] = float(np.mean(flat_y[mask])) if mask.any() else 0.0
        out[f"prefix_coverage_at_{threshold:g}"] = float(np.mean(mask))
    return out


def _auc(probs: np.ndarray, labels: np.ndarray) -> float:
    pos = probs[labels]
    neg = probs[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
