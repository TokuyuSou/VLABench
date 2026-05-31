from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import ACTION_DIM


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


def action_output_to_raw_and_norm(
    model_out,
    batch: dict[str, torch.Tensor],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(model_out, tuple):
        pred_norm, log_std = model_out
        pred_std_norm = torch.exp(log_std)
    else:
        pred_norm = model_out
        pred_std_norm = None

    if target_mode == "residual":
        repeat_raw = batch["raw_prev_actions"][:, -1:, :].repeat(1, pred_norm.shape[1], 1)
        repeat_norm = batch["prev_actions"][:, -1:, :].repeat(1, pred_norm.shape[1], 1)
        pred_raw = repeat_raw + pred_norm * action_std
        pred_norm_abs = repeat_norm + pred_norm
    else:
        pred_raw = pred_norm * action_std + action_mean
        pred_norm_abs = pred_norm
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

