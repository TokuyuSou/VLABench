from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Normalizer


def normalized_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    first = F.mse_loss(pred[:, 0], target[:, 0])
    smooth = F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    return mse + 0.3 * first + 0.05 * smooth


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    action_norm: Normalizer,
    device: torch.device,
    target_mode: str = "absolute",
) -> dict[str, float]:
    model.eval()
    preds, targets, repeat_preds, mean_preds = [], [], [], []
    pred_stds = []
    norm_abs_errors = []
    action_mean = torch.from_numpy(action_norm.mean).to(device=device, dtype=torch.float32)
    action_std = torch.from_numpy(action_norm.std).to(device=device, dtype=torch.float32)
    for batch in loader:
        batch = batch_to_device(batch, device)
        target = batch["raw_target"]
        repeat = batch["raw_prev_actions"][:, -1:, :].repeat(1, target.shape[1], 1)
        model_out = model(batch["embeddings"], batch["state"], batch["prev_actions"])
        pred_norm, log_std = _split_model_output(model_out)
        if target_mode == "residual":
            pred = repeat + pred_norm * action_std
            pred_std = torch.exp(log_std) * action_std if log_std is not None else None
        else:
            pred = pred_norm * action_std + action_mean
            pred_std = torch.exp(log_std) * action_std if log_std is not None else None
        mean = action_mean.view(1, 1, -1).repeat(target.shape[0], target.shape[1], 1)
        preds.append(pred.cpu())
        targets.append(target.cpu())
        repeat_preds.append(repeat.cpu())
        mean_preds.append(mean.cpu())
        if pred_std is not None:
            pred_stds.append(pred_std.cpu())
            norm_abs_errors.append((torch.abs(pred - target) / action_std).cpu())

    target = torch.cat(targets).numpy()
    pred = torch.cat(preds).numpy()
    repeat = torch.cat(repeat_preds).numpy()
    mean = torch.cat(mean_preds).numpy()

    out = {}
    out.update(_metrics_for("model", pred, target, action_norm.std))
    out.update(_metrics_for("repeat_last", repeat, target, action_norm.std))
    out.update(_metrics_for("train_mean", mean, target, action_norm.std))
    out["rmse_improvement_over_repeat"] = (
        (out["repeat_last_rmse"] - out["model_rmse"]) / max(out["repeat_last_rmse"], 1e-12)
    )
    out["first_rmse_improvement_over_repeat"] = (
        (out["repeat_last_first_rmse"] - out["model_first_rmse"])
        / max(out["repeat_last_first_rmse"], 1e-12)
    )
    if pred_stds:
        std = torch.cat(pred_stds).numpy()
        norm_err = torch.cat(norm_abs_errors).numpy()
        std_scalar = std.reshape(std.shape[0], -1).mean(axis=1)
        err_scalar = norm_err.reshape(norm_err.shape[0], -1).mean(axis=1)
        confidence = 1.0 / (1.0 + std_scalar)
        out["predicted_std_mean"] = float(np.mean(std))
        out["predicted_std_p90"] = float(np.percentile(std, 90))
        out["confidence_mean"] = float(np.mean(confidence))
        out["confidence_p10"] = float(np.percentile(confidence, 10))
        out["uncertainty_error_corr"] = _safe_corr(std_scalar, err_scalar)
        out["confidence_error_corr"] = _safe_corr(confidence, err_scalar)
    return out


def _split_model_output(model_out):
    if isinstance(model_out, tuple):
        return model_out
    return model_out, None


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
    err = pred - target
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
