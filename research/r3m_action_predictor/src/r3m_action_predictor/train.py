from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .action_repr import euler_to_repr, repr_to_euler
from .data import Normalizer
from .metrics import batch_to_device, evaluate, log_scalars, make_tb_writer
from .model import build_model, parameter_count, run_model, split_model_output
from .risk import RiskLabelConfig, safe_prefix_labels


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    normalizers: dict[str, Normalizer],
    config: dict,
    out_dir: Path,
    init_state_dict: dict | None = None,
) -> dict:
    sample = next(iter(train_loader))
    if config.get("model_kind") == "finetune_encoder":
        # Image batch [B, V, 3, H, W]; the encoder owns embed_dim, normalizers bake in standardization.
        model = build_model(config, embed_dim=0, num_views=sample["images"].shape[-4], normalizers=normalizers)
    else:
        model = build_model(config, embed_dim=sample["embeddings"].shape[-1], num_views=sample["embeddings"].shape[-2])
    # Warm-start (e.g. rollout fine-tuning continues from a demo-trained checkpoint). Default None
    # preserves the from-scratch behaviour of every existing caller.
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if hasattr(model, "param_groups"):  # discriminative LR (e.g. low LR on a fine-tuned encoder)
        encoder_lr = config["lr"] * float(config.get("encoder_lr_mult", 0.1))
        opt = torch.optim.AdamW(model.param_groups(config["lr"], encoder_lr), weight_decay=config["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config["epochs"], 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    writer = make_tb_writer(out_dir)
    best_state = None
    best_val = float("inf")
    history = []
    tmode = config.get("target_mode", "absolute")
    # Decode train metrics (RMSE etc.) on a val-sized shuffled subset each epoch, so the log shows
    # the same interpretable numbers for train as for val (the raw loss alone is hard to read, and
    # train-vs-val RMSE makes over/under-fitting visible). Subset keeps the extra eval cost ~= val.
    train_eval_batches = max(1, len(val_loader))
    for epoch in range(1, config["epochs"] + 1):
        train_loss = _train_one_epoch(model, train_loader, opt, scaler, device, config, epoch, normalizers["action"])
        scheduler.step()
        val_metrics = evaluate(model, val_loader, normalizers, device, tmode)
        train_metrics = evaluate(model, train_loader, normalizers, device, tmode, max_batches=train_eval_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_metrics": train_metrics, **val_metrics})
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} "
            f"train_rmse={train_metrics['model_rmse']:.5f} "
            f"val_rmse={val_metrics['model_rmse']:.5f} "
            f"repeat={val_metrics['repeat_last_rmse']:.5f}",
            flush=True,
        )
        log_scalars(writer, "train", {"loss": train_loss, "lr": opt.param_groups[0]["lr"]}, epoch)
        log_scalars(writer, "train_eval", train_metrics, epoch)
        log_scalars(writer, "val", val_metrics, epoch)
        if val_metrics["model_rmse"] < best_val:
            best_val = val_metrics["model_rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            # Persist the best checkpoint as soon as it improves, so a killed/crashed run keeps it
            # (the current weights ARE the new best at this point).
            _write_checkpoint(model, normalizers, config, out_dir)
            print(f"  saved best checkpoint (val_rmse={best_val:.5f})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, normalizers, device, config.get("target_mode", "absolute"))
    log_scalars(writer, "test", test_metrics, config["epochs"])
    log_scalars(writer, "summary", {"best_val_rmse": best_val}, config["epochs"])
    if writer is not None:
        writer.close()
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
    epoch: int,
    action_norm: Normalizer,
) -> float:
    model.train()
    losses = []
    action_mean = torch.from_numpy(action_norm.mean).to(device=device, dtype=torch.float32)
    action_std = torch.from_numpy(action_norm.std).to(device=device, dtype=torch.float32)
    # grad_accum_steps>1 (image fine-tuning) sums gradients over micro-batches so a small image
    # batch matches the frozen baseline's effective batch. accum==1 is exactly the original loop.
    accum = max(1, int(config.get("grad_accum_steps", 1)))
    n_batches = len(train_loader)
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(train_loader):
        batch = batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            model_out = run_model(model, batch)
            target = _training_target(batch, config)
            loss = _loss_for_output(model_out, target, batch, action_mean, action_std, config, epoch)
        scaler.scale(loss / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == n_batches:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _training_target(batch: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    if config.get("target_mode", "absolute") == "residual":
        repeat_norm = batch["prev_actions"][:, -1:, :].repeat(1, batch["target"].shape[1], 1)
        return batch["target"] - repeat_norm
    return batch["target"]


def _loss_for_output(
    model_out,
    target: torch.Tensor,
    batch: dict[str, torch.Tensor],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    config: dict,
    epoch: int,
) -> torch.Tensor:
    mean, log_std, prefix_logits = split_model_output(model_out)
    step_weights = _step_weights(batch, config)
    smoothness_weight = float(config.get("smoothness_weight", 0.0))
    if log_std is not None:
        var = torch.exp(2.0 * log_std)
        nll = 0.5 * ((target - mean).square() / var + 2.0 * log_std)
        nll_loss = (nll * step_weights).mean()
        mse = _weighted_normalized_loss(mean, target, step_weights, smoothness_weight)
        action_loss = config.get("nll_weight", 1.0) * nll_loss + config.get("mse_weight", 0.25) * mse
    else:
        action_loss = _weighted_normalized_loss(mean, target, step_weights, smoothness_weight)

    risk_weight = config.get("prefix_risk_weight", 0.0)
    warmup = config.get("prefix_risk_warmup_epochs", 0)
    if prefix_logits is None or risk_weight <= 0.0 or epoch <= warmup:
        return action_loss

    pred_raw = _model_mean_to_raw(mean.detach(), batch, action_mean, action_std, config)
    label_cfg = RiskLabelConfig(
        horizon=prefix_logits.shape[1],
        pos_threshold=config.get("prefix_pos_threshold", 0.03),
        rot_threshold=config.get("prefix_rot_threshold", 0.20),
        gripper_threshold=config.get("prefix_gripper_threshold", 0.5),
    )
    prefix_labels, _ = safe_prefix_labels(pred_raw, batch["raw_target"], label_cfg)
    prefix_logits = prefix_logits[:, : prefix_labels.shape[1]]
    prefix_loss = nn.functional.binary_cross_entropy_with_logits(
        prefix_logits,
        prefix_labels,
        pos_weight=_prefix_pos_weight(prefix_labels),
    )
    return action_loss + risk_weight * prefix_loss


def _weighted_normalized_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    smoothness_weight: float = 0.0,
) -> torch.Tensor:
    # The whole predicted K-step chunk is executed, so every step matters equally;
    # there is no special weighting on the first step. The optional smoothness term
    # matches predicted vs. expert step-to-step deltas and is off unless requested.
    mse = ((pred - target).square() * weights).mean()
    if smoothness_weight <= 0.0:
        return mse
    smooth = torch.mean((pred[:, 1:] - pred[:, :-1] - (target[:, 1:] - target[:, :-1])).square())
    return mse + smoothness_weight * smooth


def _step_weights(batch: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    weight_scale = float(config.get("critical_action_weight", 0.0))
    if weight_scale <= 0.0:
        return torch.ones_like(batch["target"])
    raw = batch["raw_target"]
    prev = torch.cat([batch["raw_prev_actions"][:, -1:, :], raw[:, :-1]], dim=1)
    pos_delta = torch.linalg.norm(raw[..., :3] - prev[..., :3], dim=-1, keepdim=True)
    rot_delta = torch.linalg.norm(raw[..., 3:6] - prev[..., 3:6], dim=-1, keepdim=True)
    grip_delta = (raw[..., 6:7] - prev[..., 6:7]).abs() > 0.25
    critical = (pos_delta > 0.025).float() + (rot_delta > 0.15).float() + grip_delta.float()
    weights = 1.0 + weight_scale * critical.clamp(max=1.0)
    return weights.expand_as(batch["target"])


def _model_mean_to_raw(
    mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    # ``mean`` is in normalized sin/cos repr space; reconstruct the repr prediction
    # then decode to raw 7-D Euler for the demo-error risk labels.
    if config.get("target_mode", "absolute") == "residual":
        repeat_repr = euler_to_repr(batch["raw_prev_actions"][:, -1:, :]).repeat(1, mean.shape[1], 1)
        pred_repr = repeat_repr + mean * action_std.view(1, 1, -1)
    else:
        pred_repr = mean * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
    return repr_to_euler(pred_repr)


def _prefix_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    pos = labels.sum().clamp_min(1.0)
    neg = (labels.numel() - labels.sum()).clamp_min(1.0)
    return (neg / pos).detach()


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
