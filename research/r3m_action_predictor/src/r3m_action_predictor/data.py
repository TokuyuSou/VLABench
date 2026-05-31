from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
    action_m = OnlineMoments((ACTION_DIM,))
    for embeddings, states, actions in store.iter_arrays():
        emb_m.update(embeddings.reshape(-1, emb_dim))
        state_m.update(states)
        action_m.update(actions)
    return {"embedding": emb_m.finish(), "state": state_m.finish(), "action": action_m.finish()}


class ChunkDataset(Dataset):
    def __init__(
        self,
        store: FeatureStore,
        normalizers: dict[str, Normalizer],
        prev_horizon: int,
        pred_horizon: int,
    ):
        self.store = store
        self.normalizers = normalizers
        self.prev_horizon = prev_horizon
        self.pred_horizon = pred_horizon
        self.index: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(store.episodes):
            length = ep["actions"].shape[0]
            for t in range(prev_horizon, length - pred_horizon + 1):
                self.index.append((ep_i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, t = self.index[idx]
        ep = self.store.episodes[ep_i]
        embeddings = self.normalizers["embedding"].encode(ep["embeddings"][t])
        state = self.normalizers["state"].encode(ep["states"][t])
        prev_actions = self.normalizers["action"].encode(ep["actions"][t - self.prev_horizon : t])
        target = self.normalizers["action"].encode(ep["actions"][t : t + self.pred_horizon])

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


def make_loaders(
    train: list[EpisodeInfo],
    val: list[EpisodeInfo],
    test: list[EpisodeInfo],
    prev_horizon: int,
    pred_horizon: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Normalizer], dict[str, int]]:
    train_store = FeatureStore(train)
    normalizers = fit_normalizers(train_store)
    datasets = {
        "train": ChunkDataset(train_store, normalizers, prev_horizon, pred_horizon),
        "val": ChunkDataset(FeatureStore(val), normalizers, prev_horizon, pred_horizon),
        "test": ChunkDataset(FeatureStore(test), normalizers, prev_horizon, pred_horizon),
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
