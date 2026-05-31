from __future__ import annotations

import torch
import torch.nn as nn

from .config import ACTION_DIM, STATE_DIM


class ActionChunkPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_views: int,
        pred_horizon: int,
        view_dim: int,
        width: int,
        hidden_dim: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = ACTION_DIM
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
            input_size=ACTION_DIM,
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
            nn.Linear(hidden_dim, pred_horizon * ACTION_DIM),
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
        view_dim: int,
        width: int,
        hidden_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = ACTION_DIM
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
            nn.LayerNorm(ACTION_DIM),
            nn.Linear(ACTION_DIM, width),
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
            nn.Linear(width, 2 * ACTION_DIM),
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


def build_model(config: dict, embed_dim: int, num_views: int) -> nn.Module:
    kind = config.get("model_kind", "mlp_gru")
    if kind == "mlp_gru":
        return ActionChunkPredictor(
            embed_dim=embed_dim,
            num_views=num_views,
            pred_horizon=config["pred_horizon"],
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
