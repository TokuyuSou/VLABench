from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .action_repr import RAW_ACTION_DIM, REPR_ACTION_DIM, euler_to_repr, repr_std_to_euler_std
from .config import ExperimentConfig
from .data import Normalizer
from .model import build_model, run_model, split_model_output
from .r3m_features import ensure_r3m_repo, load_r3m_from_repo, pil_bytes_to_chw_uint8
from .risk import action_output_to_raw_and_norm, load_risk_head, make_risk_features


class LoadedPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
        risk_head_path: str | Path | None = None,
        retrieval_cache_path: str | Path | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ExperimentConfig().to_jsonable()
        self.config.update(ckpt["config"])
        self.normalizers = {
            k: Normalizer(np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
            for k, v in ckpt["normalizers"].items()
        }
        self.model_action_dim = int(np.asarray(self.normalizers["action"].mean).shape[-1])
        self.config["action_dim"] = self.model_action_dim

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

        # Retrieval cache: only relevant for a retrieval_aug model. Use the explicit path, else
        # auto-discover one saved next to the checkpoint. Absent cache -> model runs hint-free.
        self.retrieval_cache = None
        if getattr(self.model, "uses_retrieval", False):
            from .retrieval import RetrievalCache

            cache_path = Path(retrieval_cache_path) if retrieval_cache_path else (
                Path(checkpoint_path).parent / "retrieval_cache.npz"
            )
            if cache_path.exists():
                self.retrieval_cache = RetrievalCache.load(cache_path)

    @torch.no_grad()
    def predict(self, embeddings: np.ndarray, state: np.ndarray, prev_actions: np.ndarray, task: str | None = None) -> dict:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        prev_actions = np.asarray(prev_actions, dtype=np.float32)

        emb_norm = self.normalizers["embedding"].encode(embeddings)
        state_norm = self.normalizers["state"].encode(state)
        # prev_actions arrive as raw 7-D Euler. Some existing checkpoints use raw
        # Euler model-space actions; newer ones use the 10-D sin/cos representation.
        prev_norm = self.normalizers["action"].encode(self._actions_to_model_space(prev_actions))

        batch = {
            "embeddings": torch.from_numpy(emb_norm[None]).to(self.device),
            "state": torch.from_numpy(state_norm[None]).to(self.device),
            "prev_actions": torch.from_numpy(prev_norm[None]).to(self.device),
            "raw_prev_actions": torch.from_numpy(prev_actions[None]).to(self.device),
        }
        self._add_retrieval(batch, embeddings, state, task)
        model_out = run_model(self.model, batch)
        _, _, prefix_logits = split_model_output(model_out)
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
        # Envelope/aleatoric gates score in the normalized sin/cos repr space the predictor
        # natively emits its residual in: pred_delta_norm is the per-step residual from the
        # last executed action (= pred_norm_abs - repeat_norm), pred_std_norm the predicted std.
        repeat_norm = batch["prev_actions"][:, -1:, :].repeat(1, pred_norm_abs.shape[1], 1)
        out["pred_delta_norm"] = (pred_norm_abs - repeat_norm)[0].detach().cpu().numpy()
        if std_norm is not None:
            out["pred_std_norm"] = std_norm[0].detach().cpu().numpy()
        if std_norm is not None:
            raw_model_std = std_norm * action_std
            if self.model_action_dim == RAW_ACTION_DIM:
                raw_std = raw_model_std
            elif self.model_action_dim == REPR_ACTION_DIM:
                # Map the per-dim sin/cos std back to a 7-D Euler-space std.
                raw_std = repr_std_to_euler_std(raw_model_std)
            else:
                raise ValueError(f"Unsupported action dim: {self.model_action_dim}")
            confidence_per_step = 1.0 / (1.0 + std_norm.mean(dim=-1))
            out.update(
                {
                    "action_std": raw_std[0].detach().cpu().numpy(),
                    "confidence_per_step": confidence_per_step[0].detach().cpu().numpy(),
                    "confidence": float(confidence_per_step.mean().detach().cpu()),
                }
            )
        if prefix_logits is not None:
            prefix_p = torch.sigmoid(prefix_logits[0])
            out.update(
                {
                    "prefix_safe_probability": prefix_p.detach().cpu().numpy(),
                    "risk_safe_probability": float(prefix_p.mean().detach().cpu()),
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
            objective = self.risk_checkpoint.get("objective", "classification") if self.risk_checkpoint else "classification"
            threshold = self.risk_checkpoint.get("selected_threshold") if self.risk_checkpoint else None
            if objective == "regression":
                # Head predicts a nonnegative cost (lower = safer). Map to a safe
                # probability so the existing risk-gated policy can threshold it
                # (selected_threshold is already stored in this 1/(1+cost) space).
                cost = torch.nn.functional.softplus(risk_logit)
                risk_p_safe = 1.0 / (1.0 + cost)
                out["risk_predicted_cost"] = float(cost.item())
            else:
                risk_p_safe = torch.sigmoid(risk_logit)
            out.update(
                {
                    "risk_safe_probability": float(risk_p_safe.item()),
                    "risk_logit": float(risk_logit.item()),
                    "risk_threshold": float(threshold) if threshold is not None else None,
                }
            )
        return out

    def _add_retrieval(self, batch: dict, embeddings: np.ndarray, state: np.ndarray, task: str | None) -> None:
        """Attach instruction-filtered retrieval hints to the batch (no-op without a cache/task)."""
        if self.retrieval_cache is None or task is None:
            return
        from .retrieval import make_keys, retrieve

        obs_key, state_key = make_keys(self.normalizers, embeddings, state)
        a, s, m = retrieve(
            self.retrieval_cache, task, obs_key, state_key,
            int(self.config.get("retrieval_k", 4)),
            float(self.config.get("retrieval_w_obs", 1.0)),
            float(self.config.get("retrieval_w_state", 1.0)),
        )
        batch["retr_actions"] = torch.from_numpy(a[None]).to(self.device)
        batch["retr_sim"] = torch.from_numpy(s[None]).to(self.device)
        batch["retr_mask"] = torch.from_numpy(m[None]).to(self.device)

    def _actions_to_model_space(self, actions: np.ndarray) -> np.ndarray:
        if self.model_action_dim == RAW_ACTION_DIM:
            return actions.astype(np.float32, copy=False)
        if self.model_action_dim == REPR_ACTION_DIM:
            return euler_to_repr(actions)
        raise ValueError(f"Unsupported action dim: {self.model_action_dim}")


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
    def encode_view_history(self, frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        encoded = [
            self.encode_views(image=image, second_image=second_image, wrist_image=wrist_image)
            for image, second_image, wrist_image in frames
        ]
        return np.stack(encoded, axis=0)

    @torch.no_grad()
    def predict_from_observation(
        self,
        image: np.ndarray,
        second_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prev_actions: np.ndarray,
        image_history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        task: str | None = None,
    ) -> dict:
        obs_horizon = int(self.predictor.config.get("obs_horizon", 1))
        if obs_horizon > 1:
            frames = list(image_history or [])
            if not frames:
                frames = [(image, second_image, wrist_image)]
            frames = frames[-obs_horizon:]
            while len(frames) < obs_horizon:
                frames.insert(0, frames[0])
            embeddings = self.encode_view_history(frames)
        else:
            embeddings = self.encode_views(image=image, second_image=second_image, wrist_image=wrist_image)
        return self.predictor.predict(embeddings, state, prev_actions, task=task)
