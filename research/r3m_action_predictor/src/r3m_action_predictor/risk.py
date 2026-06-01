from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .action_repr import RAW_ACTION_DIM, REPR_ACTION_DIM, euler_to_repr, repr_to_euler
from .config import ACTION_DIM
from .model import split_model_output


@dataclass(frozen=True)
class RiskLabelConfig:
    horizon: int = 5
    pos_threshold: float = 0.03
    rot_threshold: float = 0.20
    gripper_threshold: float = 0.5


class RiskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def make_risk_features(
    embeddings: torch.Tensor,
    state: torch.Tensor,
    prev_actions: torch.Tensor,
    pred_norm_abs: torch.Tensor,
    pred_std_norm: torch.Tensor | None,
) -> torch.Tensor:
    repeat_norm = prev_actions[:, -1:, :].repeat(1, pred_norm_abs.shape[1], 1)
    pred_delta_norm = pred_norm_abs - repeat_norm
    if pred_std_norm is None:
        pred_std_norm = torch.zeros_like(pred_norm_abs)
    return torch.cat(
        [
            embeddings.flatten(1),
            state.flatten(1),
            prev_actions.flatten(1),
            pred_norm_abs.flatten(1),
            pred_delta_norm.flatten(1),
            pred_std_norm.flatten(1),
        ],
        dim=-1,
    )


def safe_labels(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    cfg: RiskLabelConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    h = min(cfg.horizon, pred_actions.shape[1], target_actions.shape[1])
    pred = pred_actions[:, :h]
    target = target_actions[:, :h]
    pos_err = torch.linalg.norm(pred[..., :3] - target[..., :3], dim=-1).amax(dim=1)
    rot_err = torch.linalg.norm(wrap_angle(pred[..., 3:6] - target[..., 3:6]), dim=-1).amax(dim=1)
    pred_grip = pred[..., 6] > cfg.gripper_threshold
    target_grip = target[..., 6] > cfg.gripper_threshold
    grip_ok = (pred_grip == target_grip).all(dim=1)
    safe = (pos_err <= cfg.pos_threshold) & (rot_err <= cfg.rot_threshold) & grip_ok
    return safe.float(), {"pos_err": pos_err, "rot_err": rot_err, "grip_ok": grip_ok.float()}


def safe_prefix_labels(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    cfg: RiskLabelConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    h = min(cfg.horizon, pred_actions.shape[1], target_actions.shape[1])
    pred = pred_actions[:, :h]
    target = target_actions[:, :h]
    pos_step = torch.linalg.norm(pred[..., :3] - target[..., :3], dim=-1)
    rot_step = torch.linalg.norm(wrap_angle(pred[..., 3:6] - target[..., 3:6]), dim=-1)
    pred_grip = pred[..., 6] > cfg.gripper_threshold
    target_grip = target[..., 6] > cfg.gripper_threshold
    grip_ok_step = pred_grip == target_grip

    pos_ok = torch.cummax(pos_step, dim=1).values <= cfg.pos_threshold
    rot_ok = torch.cummax(rot_step, dim=1).values <= cfg.rot_threshold
    grip_ok = torch.cummin(grip_ok_step.float(), dim=1).values > 0.5
    safe = (pos_ok & rot_ok & grip_ok).float()
    return safe, {
        "prefix_pos_err": pos_step,
        "prefix_rot_err": rot_step,
        "prefix_grip_ok": grip_ok_step.float(),
    }


def chunk_cost_labels(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    repr_std: torch.Tensor,
    grip_weight: float = 3.0,
    horizon: int = 8,
) -> torch.Tensor:
    """Continuous, gripper-weighted normalized error of a predicted chunk vs expert.

    Replaces the arbitrary binary safe/unsafe label. The continuous-action error is the
    RMS of the per-dim error in the model's normalized sin/cos representation, so rotation
    is wraparound-safe and on the same scale the predictor is trained in (= "normalized
    RMSE"). The binary gripper open/close state is handled specially: every step whose
    predicted state disagrees with the expert adds ``grip_weight**2`` to the squared
    error, because a wrong grasp is catastrophic. Returns one nonnegative scalar per chunk
    (lower = safer to substitute).
    """
    h = min(horizon, pred_actions.shape[1], target_actions.shape[1])
    pred = pred_actions[:, :h]
    target = target_actions[:, :h]
    # 9 continuous dims: position (0:3) + sin/cos of the 3 Euler angles (3:9). Gripper (9) excluded.
    z = (euler_to_repr(pred)[..., :9] - euler_to_repr(target)[..., :9]) / repr_std[:9]
    cont_sq = z.square().mean(dim=-1)                                          # (B, h)
    grip_mismatch = ((pred[..., 6] > 0.5) != (target[..., 6] > 0.5)).float()   # (B, h)
    per_step_sq = cont_sq + (grip_weight ** 2) * grip_mismatch                 # (B, h)
    return per_step_sq.mean(dim=1).clamp_min(0.0).sqrt()                       # (B,)


def action_output_to_raw_and_norm(
    model_out,
    batch: dict[str, torch.Tensor],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # New checkpoints use the 10-D sin/cos action representation. Older K=8
    # checkpoints in this study used raw 7-D Euler actions. Decode according to
    # the normalizer dimension so risk/eval can consume either checkpoint family.
    pred_norm, log_std, _ = split_model_output(model_out)
    pred_std_norm = torch.exp(log_std) if log_std is not None else None
    action_dim = int(action_mean.shape[-1])

    if target_mode == "residual":
        if action_dim == RAW_ACTION_DIM:
            repeat_action = batch["raw_prev_actions"][:, -1:, :].repeat(1, pred_norm.shape[1], 1)
        elif action_dim == REPR_ACTION_DIM:
            repeat_action = euler_to_repr(batch["raw_prev_actions"][:, -1:, :]).repeat(1, pred_norm.shape[1], 1)
        else:
            raise ValueError(f"Unsupported action dim: {action_dim}")
        repeat_norm = batch["prev_actions"][:, -1:, :].repeat(1, pred_norm.shape[1], 1)
        pred_action = repeat_action + pred_norm * action_std
        pred_norm_abs = repeat_norm + pred_norm
    else:
        pred_action = pred_norm * action_std + action_mean
        pred_norm_abs = pred_norm
    if action_dim == RAW_ACTION_DIM:
        pred_raw = pred_action
    elif action_dim == REPR_ACTION_DIM:
        pred_raw = repr_to_euler(pred_action)
    else:
        raise ValueError(f"Unsupported action dim: {action_dim}")
    return pred_raw, pred_norm_abs, pred_std_norm


def load_risk_head(path: str | Path, device: torch.device | str = "cpu") -> tuple[RiskHead, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = RiskHead(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["risk_state"])
    model.eval().to(device)
    return model, ckpt
