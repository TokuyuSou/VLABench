from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import Normalizer
from .model import build_model
from .r3m_features import ensure_r3m_repo, load_r3m_from_repo, pil_bytes_to_chw_uint8
from .risk import action_output_to_raw_and_norm, load_risk_head, make_risk_features


class LoadedPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
        risk_head_path: str | Path | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ExperimentConfig().to_jsonable()
        self.config.update(ckpt["config"])
        self.normalizers = {
            k: Normalizer(np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
            for k, v in ckpt["normalizers"].items()
        }

        # R3M feature shape is fixed after feature extraction: [num_views, embed_dim].
        num_views = int(np.asarray(self.normalizers["embedding"].mean).shape[0])
        embed_dim = int(np.asarray(self.normalizers["embedding"].mean).shape[-1])
        if num_views == 1:
            # The training normalizer stores one shared per-view mean with shape [1, D].
            num_views = 3
        self.model = build_model(self.config, embed_dim=embed_dim, num_views=num_views)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(self.device)
        self.risk_head = None
        self.risk_checkpoint = None
        if risk_head_path:
            self.risk_head, self.risk_checkpoint = load_risk_head(risk_head_path, device=self.device)

    @torch.no_grad()
    def predict(self, embeddings: np.ndarray, state: np.ndarray, prev_actions: np.ndarray) -> dict:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        prev_actions = np.asarray(prev_actions, dtype=np.float32)

        emb_norm = self.normalizers["embedding"].encode(embeddings)
        state_norm = self.normalizers["state"].encode(state)
        prev_norm = self.normalizers["action"].encode(prev_actions)

        batch = {
            "embeddings": torch.from_numpy(emb_norm[None]).to(self.device),
            "state": torch.from_numpy(state_norm[None]).to(self.device),
            "prev_actions": torch.from_numpy(prev_norm[None]).to(self.device),
            "raw_prev_actions": torch.from_numpy(prev_actions[None]).to(self.device),
        }
        model_out = self.model(batch["embeddings"], batch["state"], batch["prev_actions"])
        action_mean = torch.from_numpy(self.normalizers["action"].mean).to(self.device)
        action_std = torch.from_numpy(self.normalizers["action"].std).to(self.device)
        pred, pred_norm_abs, std_norm = action_output_to_raw_and_norm(
            model_out,
            batch,
            action_mean,
            action_std,
            self.config.get("target_mode", "absolute"),
        )

        out = {"actions": pred[0].detach().cpu().numpy()}
        if std_norm is not None:
            raw_std = std_norm * action_std
            confidence_per_step = 1.0 / (1.0 + std_norm.mean(dim=-1))
            out.update(
                {
                    "action_std": raw_std[0].detach().cpu().numpy(),
                    "confidence_per_step": confidence_per_step[0].detach().cpu().numpy(),
                    "confidence": float(confidence_per_step.mean().detach().cpu()),
                }
            )
        if self.risk_head is not None:
            risk_features = make_risk_features(
                batch["embeddings"],
                batch["state"],
                batch["prev_actions"],
                pred_norm_abs,
                std_norm,
            )
            risk_logit = self.risk_head(risk_features)
            risk_p_safe = torch.sigmoid(risk_logit)
            threshold = self.risk_checkpoint.get("selected_threshold") if self.risk_checkpoint else None
            out.update(
                {
                    "risk_safe_probability": float(risk_p_safe.item()),
                    "risk_logit": float(risk_logit.item()),
                    "risk_threshold": float(threshold) if threshold is not None else None,
                }
            )
        return out


def load_predictor(
    checkpoint_path: str | Path,
    device: str | None = None,
    risk_head_path: str | Path | None = None,
) -> LoadedPredictor:
    return LoadedPredictor(Path(checkpoint_path), device=device, risk_head_path=risk_head_path)


class OnlineR3MActionPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        r3m_cache_dir: str | Path,
        r3m_model: str = "resnet18",
        device: str = "cpu",
        risk_head_path: str | Path | None = None,
    ):
        self.device = torch.device(device)
        self.predictor = load_predictor(checkpoint_path, device=device, risk_head_path=risk_head_path)
        r3m_dir = ensure_r3m_repo(Path(r3m_cache_dir))
        self.r3m = load_r3m_from_repo(r3m_dir, r3m_model).to(self.device)
        self.r3m.eval()

    @torch.no_grad()
    def encode_views(self, image: np.ndarray, second_image: np.ndarray, wrist_image: np.ndarray) -> np.ndarray:
        views = [image, second_image, wrist_image]
        tensors = []
        for view in views:
            arr = np.asarray(view, dtype=np.uint8)
            # Reuse the training feature path's CHW uint8 convention.
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image, got shape={arr.shape}")
            tensors.append(torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous())
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            emb = self.r3m(batch, obs_shape=[3, batch.shape[-2], batch.shape[-1]])
        return emb.detach().float().cpu().numpy()

    @torch.no_grad()
    def predict_from_observation(
        self,
        image: np.ndarray,
        second_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prev_actions: np.ndarray,
    ) -> dict:
        embeddings = self.encode_views(image=image, second_image=second_image, wrist_image=wrist_image)
        return self.predictor.predict(embeddings, state, prev_actions)
