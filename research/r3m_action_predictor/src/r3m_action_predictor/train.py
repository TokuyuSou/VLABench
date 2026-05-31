from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import Normalizer
from .metrics import batch_to_device, evaluate, normalized_loss
from .model import build_model, parameter_count


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    normalizers: dict[str, Normalizer],
    config: dict,
    out_dir: Path,
) -> dict:
    sample = next(iter(train_loader))
    model = build_model(config, embed_dim=sample["embeddings"].shape[-1], num_views=sample["embeddings"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config["epochs"], 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, config["epochs"] + 1):
        train_loss = _train_one_epoch(model, train_loader, opt, scaler, device, config)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, normalizers["action"], device, config.get("target_mode", "absolute"))
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_rmse={val_metrics['model_rmse']:.5f} "
            f"repeat={val_metrics['repeat_last_rmse']:.5f}",
            flush=True,
        )
        if val_metrics["model_rmse"] < best_val:
            best_val = val_metrics["model_rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, normalizers["action"], device, config.get("target_mode", "absolute"))
    result = {
        "predictor_parameter_count": parameter_count(model),
        "best_val_rmse": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }
    _write_checkpoint(model, normalizers, config, out_dir)
    return result


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict,
) -> float:
    model.train()
    losses = []
    for batch in train_loader:
        batch = batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            model_out = model(batch["embeddings"], batch["state"], batch["prev_actions"])
            target = _training_target(batch, config)
            loss = _loss_for_output(model_out, target, config)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _training_target(batch: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    if config.get("target_mode", "absolute") == "residual":
        repeat_norm = batch["prev_actions"][:, -1:, :].repeat(1, batch["target"].shape[1], 1)
        return batch["target"] - repeat_norm
    return batch["target"]


def _loss_for_output(model_out, target: torch.Tensor, config: dict) -> torch.Tensor:
    if isinstance(model_out, tuple):
        mean, log_std = model_out
        var = torch.exp(2.0 * log_std)
        nll = 0.5 * ((target - mean).square() / var + 2.0 * log_std)
        mse = normalized_loss(mean, target)
        return config.get("nll_weight", 1.0) * nll.mean() + config.get("mse_weight", 0.25) * mse
    return normalized_loss(model_out, target)


def _write_checkpoint(
    model: nn.Module,
    normalizers: dict[str, Normalizer],
    config: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "normalizers": {k: v.to_json() for k, v in normalizers.items()},
        },
        out_dir / "best_model.pt",
    )
    (out_dir / "normalizers.json").write_text(
        json.dumps({k: v.to_json() for k, v in normalizers.items()}, indent=2) + "\n"
    )
