from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .action_repr import RAW_ACTION_DIM, REPR_ACTION_DIM, SINCOS_REPR_SLICE, euler_to_repr
from .config import ACTION_DIM, STATE_DIM
from .hf_data import EpisodeInfo


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class OnlineMoments:
    def __init__(self, shape: tuple[int, ...]):
        self.count = 0
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x64 = x.astype(np.float64, copy=False)
        self.count += x64.shape[0]
        self.sum += x64.sum(axis=0)
        self.sumsq += np.square(x64).sum(axis=0)

    def finish(self, eps: float = 1e-6) -> Normalizer:
        mean = self.sum / max(self.count, 1)
        var = self.sumsq / max(self.count, 1) - np.square(mean)
        std = np.sqrt(np.maximum(var, eps * eps))
        return Normalizer(mean.astype(np.float32), std.astype(np.float32))


class FeatureStore:
    def __init__(self, episodes: Iterable[EpisodeInfo]):
        self.episodes = []
        for ep in episodes:
            item = dict(np.load(ep.feature_path, allow_pickle=True))
            item["episode_index"] = ep.episode_index
            item["task"] = ep.task
            item["embeddings"] = item["embeddings"].astype(np.float32)
            item["states"] = item["states"].astype(np.float32)
            item["actions"] = item["actions"].astype(np.float32)
            self.episodes.append(item)

    def iter_arrays(self):
        for ep in self.episodes:
            yield ep["embeddings"], ep["states"], ep["actions"]


def fit_normalizers(store: FeatureStore) -> dict[str, Normalizer]:
    emb_dim = store.episodes[0]["embeddings"].shape[-1]
    emb_m = OnlineMoments((1, emb_dim))
    state_m = OnlineMoments((STATE_DIM,))
    # "action" is the continuous sin/cos model space; "action_euler" keeps the raw
    # 7-D Euler statistics for metric scaling and the constant-mean baseline.
    action_m = OnlineMoments((REPR_ACTION_DIM,))
    action_euler_m = OnlineMoments((ACTION_DIM,))
    for embeddings, states, actions in store.iter_arrays():
        emb_m.update(embeddings.reshape(-1, emb_dim))
        state_m.update(states)
        action_m.update(euler_to_repr(actions))
        action_euler_m.update(actions)
    action_norm = action_m.finish()
    # Leave the sin/cos angle columns unstandardized (mean 0, std 1). Standardizing
    # them divides each column by its own std, and a near-constant angle axis (e.g. a
    # fixed roll/pitch whose cos has std ~ 1e-4) gets divided by a tiny number. That
    # explodes both the action loss and the risk chunk-cost on physically negligible
    # dimensions. Identity scaling preserves the unit-circle geometry so a sin/cos
    # error stays proportional to the true angular error. Position (0:3) and gripper
    # (9) keep their fitted statistics. Both atan2 decoding and the residual path are
    # unaffected because they read these same mean/std values back.
    action_norm.mean[SINCOS_REPR_SLICE] = 0.0
    action_norm.std[SINCOS_REPR_SLICE] = 1.0
    return {
        "embedding": emb_m.finish(),
        "state": state_m.finish(),
        "action": action_norm,
        "action_euler": action_euler_m.finish(),
    }


class ChunkDataset(Dataset):
    def __init__(
        self,
        store: FeatureStore,
        normalizers: dict[str, Normalizer],
        prev_horizon: int,
        pred_horizon: int,
        obs_horizon: int = 1,
    ):
        self.store = store
        self.normalizers = normalizers
        self.prev_horizon = prev_horizon
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.index: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(store.episodes):
            length = ep["actions"].shape[0]
            start = max(prev_horizon, obs_horizon - 1)
            for t in range(start, length - pred_horizon + 1):
                self.index.append((ep_i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, t = self.index[idx]
        ep = self.store.episodes[ep_i]
        if self.obs_horizon > 1:
            embeddings = self.normalizers["embedding"].encode(
                ep["embeddings"][t - self.obs_horizon + 1 : t + 1]
            )
        else:
            embeddings = self.normalizers["embedding"].encode(ep["embeddings"][t])
        state = self.normalizers["state"].encode(ep["states"][t])
        prev_euler = ep["actions"][t - self.prev_horizon : t]
        target_euler = ep["actions"][t : t + self.pred_horizon]
        prev_actions = self.normalizers["action"].encode(_actions_to_model_space(prev_euler, self.normalizers))
        target = self.normalizers["action"].encode(_actions_to_model_space(target_euler, self.normalizers))

        return {
            "embeddings": torch.from_numpy(embeddings.astype(np.float32)),
            "state": torch.from_numpy(state.astype(np.float32)),
            "prev_actions": torch.from_numpy(prev_actions.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "raw_target": torch.from_numpy(ep["actions"][t : t + self.pred_horizon].astype(np.float32)),
            "raw_prev_actions": torch.from_numpy(
                ep["actions"][t - self.prev_horizon : t].astype(np.float32)
            ),
            "episode_index": torch.tensor(ep["episode_index"], dtype=torch.int64),
            "frame_index": torch.tensor(t, dtype=torch.int64),
            "task": ep["task"],
        }


def _actions_to_model_space(actions: np.ndarray, normalizers: dict[str, Normalizer]) -> np.ndarray:
    action_dim = int(np.asarray(normalizers["action"].mean).shape[-1])
    if action_dim == RAW_ACTION_DIM:
        return actions.astype(np.float32, copy=False)
    if action_dim == REPR_ACTION_DIM:
        return euler_to_repr(actions)
    raise ValueError(f"Unsupported action normalizer dim: {action_dim}")


def make_loaders(
    train: list[EpisodeInfo],
    val: list[EpisodeInfo],
    test: list[EpisodeInfo],
    prev_horizon: int,
    pred_horizon: int,
    batch_size: int,
    obs_horizon: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Normalizer], dict[str, int]]:
    train_store = FeatureStore(train)
    normalizers = fit_normalizers(train_store)
    datasets = {
        "train": ChunkDataset(train_store, normalizers, prev_horizon, pred_horizon, obs_horizon),
        "val": ChunkDataset(FeatureStore(val), normalizers, prev_horizon, pred_horizon, obs_horizon),
        "test": ChunkDataset(FeatureStore(test), normalizers, prev_horizon, pred_horizon, obs_horizon),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }
    return (
        loaders["train"],
        loaders["val"],
        loaders["test"],
        normalizers,
        {split: len(ds) for split, ds in datasets.items()},
    )
