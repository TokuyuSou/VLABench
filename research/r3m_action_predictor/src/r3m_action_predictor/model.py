from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Models consume/produce actions in the continuous sin/cos representation, so the
# action I/O dimension here is the 10-D repr dim (not the raw 7-D Euler dim).
from .action_repr import REPR_ACTION_DIM as ACTION_DIM
from .config import STATE_DIM


@dataclass
class PrefixRiskOutput:
    mean: torch.Tensor
    log_std: torch.Tensor
    prefix_logits: torch.Tensor


class ActionChunkPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_views: int,
        pred_horizon: int,
        action_dim: int,
        view_dim: int,
        width: int,
        hidden_dim: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.view_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, view_dim),
            nn.GELU(),
            nn.Linear(view_dim, view_dim),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.prev_encoder = nn.GRU(
            input_size=action_dim,
            hidden_size=width,
            num_layers=1,
            batch_first=True,
        )
        fused_dim = num_views * view_dim + width + width
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_horizon * action_dim),
        )

    def forward(self, embeddings: torch.Tensor, state: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        bsz, num_views, embed_dim = embeddings.shape
        view = self.view_proj(embeddings.reshape(bsz * num_views, embed_dim)).reshape(bsz, -1)
        state_h = self.state_proj(state)
        _, prev_h = self.prev_encoder(prev_actions)
        fused = torch.cat([view, state_h, prev_h[-1]], dim=-1)
        return self.fusion(fused).reshape(bsz, self.pred_horizon, self.action_dim)


class ProbabilisticTransformerActionPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_views: int,
        prev_horizon: int,
        pred_horizon: int,
        action_dim: int,
        view_dim: int,
        width: int,
        hidden_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.context_len = num_views + 1 + prev_horizon

        self.view_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, view_dim),
            nn.GELU(),
            nn.Linear(view_dim, width),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.query_tokens = nn.Parameter(torch.randn(pred_horizon, width) * 0.02)
        self.pos = nn.Parameter(torch.randn(self.context_len + pred_horizon, width) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 2 * action_dim),
        )

    def forward(self, embeddings: torch.Tensor, state: torch.Tensor, prev_actions: torch.Tensor):
        bsz, num_views, embed_dim = embeddings.shape
        view = self.view_proj(embeddings.reshape(bsz * num_views, embed_dim)).reshape(bsz, num_views, -1)
        state_tok = self.state_proj(state).unsqueeze(1)
        action = self.action_proj(prev_actions)
        query = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)
        tokens = torch.cat([view, state_tok, action, query], dim=1)
        tokens = tokens + self.pos.unsqueeze(0)
        hidden = self.encoder(tokens)
        out = self.head(hidden[:, -self.pred_horizon :])
        mean, log_std = out.chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)


class SharedEncoderPrefixRiskTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_views: int,
        obs_horizon: int,
        prev_horizon: int,
        pred_horizon: int,
        action_dim: int,
        view_dim: int,
        width: int,
        hidden_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.context_len = obs_horizon * num_views + 1 + prev_horizon

        self.view_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, view_dim),
            nn.GELU(),
            nn.Linear(view_dim, width),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.query_tokens = nn.Parameter(torch.randn(pred_horizon, width) * 0.02)
        self.pos = nn.Parameter(torch.randn(self.context_len + pred_horizon, width) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.action_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 2 * action_dim),
        )
        self.prefix_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, state: torch.Tensor, prev_actions: torch.Tensor):
        if embeddings.ndim == 3:
            embeddings = embeddings.unsqueeze(1)
        bsz, obs_horizon, num_views, embed_dim = embeddings.shape
        view = self.view_proj(embeddings.reshape(bsz * obs_horizon * num_views, embed_dim))
        view = view.reshape(bsz, obs_horizon * num_views, -1)
        state_tok = self.state_proj(state).unsqueeze(1)
        action = self.action_proj(prev_actions)
        query = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)
        tokens = torch.cat([view, state_tok, action, query], dim=1)
        tokens = tokens + self.pos[: tokens.shape[1]].unsqueeze(0)
        hidden = self.encoder(tokens)
        query_hidden = hidden[:, -self.pred_horizon :]
        out = self.action_head(query_hidden)
        mean, log_std = out.chunk(2, dim=-1)
        prefix_logits = self.prefix_head(query_hidden).squeeze(-1)
        return PrefixRiskOutput(mean=mean, log_std=log_std.clamp(-5.0, 2.0), prefix_logits=prefix_logits)


def split_model_output(model_out):
    if isinstance(model_out, PrefixRiskOutput):
        return model_out.mean, model_out.log_std, model_out.prefix_logits
    if isinstance(model_out, tuple):
        if len(model_out) == 3:
            return model_out
        mean, log_std = model_out
        return mean, log_std, None
    return model_out, None, None


def build_model(config: dict, embed_dim: int, num_views: int) -> nn.Module:
    kind = config.get("model_kind", "mlp_gru")
    action_dim = int(config.get("action_dim", ACTION_DIM))
    if kind == "mlp_gru":
        return ActionChunkPredictor(
            embed_dim=embed_dim,
            num_views=num_views,
            pred_horizon=config["pred_horizon"],
            action_dim=action_dim,
            view_dim=config["view_dim"],
            width=config["width"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )
    if kind == "prob_transformer":
        return ProbabilisticTransformerActionPredictor(
            embed_dim=embed_dim,
            num_views=num_views,
            prev_horizon=config["prev_horizon"],
            pred_horizon=config["pred_horizon"],
            action_dim=action_dim,
            view_dim=config["view_dim"],
            width=config["width"],
            hidden_dim=config["hidden_dim"],
            transformer_layers=config.get("transformer_layers", 4),
            transformer_heads=config.get("transformer_heads", 4),
            dropout=config["dropout"],
        )
    if kind == "shared_prefix_risk_transformer":
        return SharedEncoderPrefixRiskTransformer(
            embed_dim=embed_dim,
            num_views=num_views,
            obs_horizon=config.get("obs_horizon", 1),
            prev_horizon=config["prev_horizon"],
            pred_horizon=config["pred_horizon"],
            action_dim=action_dim,
            view_dim=config["view_dim"],
            width=config["width"],
            hidden_dim=config["hidden_dim"],
            transformer_layers=config.get("transformer_layers", 4),
            transformer_heads=config.get("transformer_heads", 4),
            dropout=config["dropout"],
        )
    raise ValueError(f"Unknown model_kind: {kind}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
