"""Retrieval cache for the action predictor.

Builds a non-parametric memory from a demo set that is DISJOINT from the predictor's
training data (this disjointness is what makes retrieval-augmented training leakage-safe).
Each cache entry is one frame: key = (instruction, L2-normalized obs embedding, L2-normalized
proprio state), value = that demo's next ``pred_horizon`` actions in the model's normalized
sin/cos space. Retrieval hard-filters by instruction (so "select" tasks cannot pull neighbours
that head to a different target) and then does an exact cosine kNN over a weighted obs+state key.

Everything here is plain NumPy (caches are small, ~10-30k frames); no faiss dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .action_repr import euler_to_repr
from .data import FeatureStore, Normalizer


def _l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def make_keys(normalizers: dict[str, Normalizer], emb_t: np.ndarray, state_t: np.ndarray):
    """(obs_key, state_key): per-modality normalized + L2-normalized 1-D query vectors."""
    obs = _l2norm(normalizers["embedding"].encode(emb_t).reshape(-1).astype(np.float32))
    state = _l2norm(normalizers["state"].encode(state_t).reshape(-1).astype(np.float32))
    return obs, state


@dataclass
class RetrievalCache:
    instr_ids: np.ndarray  # [N] int32
    obs_keys: np.ndarray   # [N, Do] float32, L2-normalized
    state_keys: np.ndarray  # [N, Ds] float32, L2-normalized
    values: np.ndarray     # [N, K, A] float32 (normalized sin/cos chunk)
    vocab: dict[str, int]  # instruction string -> id
    pred_horizon: int
    source_ep: np.ndarray  # [N] int64, the episode each entry came from (for leave-one-out)

    @property
    def action_dim(self) -> int:
        return self.values.shape[-1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            instr_ids=self.instr_ids,
            obs_keys=self.obs_keys,
            state_keys=self.state_keys,
            values=self.values,
            source_ep=self.source_ep,
            pred_horizon=np.int64(self.pred_horizon),
        )
        path.with_suffix(".vocab.json").write_text(json.dumps(self.vocab))

    @staticmethod
    def load(path: str | Path) -> "RetrievalCache":
        path = Path(path)
        d = np.load(path)
        vocab = json.loads(path.with_suffix(".vocab.json").read_text())
        # source_ep is optional for backward compat with caches built before leave-one-out;
        # -1 means "unknown source" so exclude_ep can never match (i.e. no exclusion).
        source_ep = d["source_ep"] if "source_ep" in d.files else np.full(d["instr_ids"].shape[0], -1, np.int64)
        return RetrievalCache(
            d["instr_ids"], d["obs_keys"], d["state_keys"], d["values"], vocab, int(d["pred_horizon"]), source_ep
        )


def build_cache(store: FeatureStore, normalizers: dict[str, Normalizer], pred_horizon: int) -> RetrievalCache:
    """One entry per frame t (1 <= t, t+pred_horizon <= len) of every episode in ``store``."""
    vocab: dict[str, int] = {}
    instr_ids, obs_keys, state_keys, values, source_ep = [], [], [], [], []
    action_norm = normalizers["action"]
    for ep in store.episodes:
        iid = vocab.setdefault(ep["task"], len(vocab))
        emb, states, actions = ep["embeddings"], ep["states"], ep["actions"]
        for t in range(1, actions.shape[0] - pred_horizon + 1):
            obs, state = make_keys(normalizers, emb[t], states[t])
            value = action_norm.encode(euler_to_repr(actions[t : t + pred_horizon])).astype(np.float32)
            instr_ids.append(iid)
            obs_keys.append(obs)
            state_keys.append(state)
            values.append(value)
            source_ep.append(int(ep["episode_index"]))
    return RetrievalCache(
        np.asarray(instr_ids, np.int32),
        np.asarray(obs_keys, np.float32),
        np.asarray(state_keys, np.float32),
        np.asarray(values, np.float32),
        vocab,
        pred_horizon,
        np.asarray(source_ep, np.int64),
    )


def retrieve(cache: RetrievalCache, task: str, obs_key: np.ndarray, state_key: np.ndarray,
             k: int, w_obs: float = 1.0, w_state: float = 1.0, exclude_ep: int | None = None):
    """Return (actions [k,K,A], sim [k], mask [k]) for the instruction-filtered kNN.

    Unfilled slots (unknown instruction, or fewer than k neighbours) are zeros with mask 0,
    so the model always receives a fixed [k, ...] tensor and learns to ignore masked hints.
    ``exclude_ep`` drops every entry that came from that episode (leave-one-out): used during
    risk-head/predictor training so a frame can never retrieve its own (or its episode's) future.
    """
    K, A = cache.pred_horizon, cache.action_dim
    out_a = np.zeros((k, K, A), np.float32)
    out_s = np.zeros((k,), np.float32)
    out_m = np.zeros((k,), np.float32)
    iid = cache.vocab.get(task, -1)
    if iid < 0:
        return out_a, out_s, out_m
    sel = np.nonzero(cache.instr_ids == iid)[0]
    if exclude_ep is not None:
        sel = sel[cache.source_ep[sel] != exclude_ep]
    if sel.size == 0:
        return out_a, out_s, out_m
    score = w_obs * (cache.obs_keys[sel] @ obs_key) + w_state * (cache.state_keys[sel] @ state_key)
    n = min(k, sel.size)
    top_local = np.argpartition(-score, n - 1)[:n]
    top_local = top_local[np.argsort(-score[top_local])]
    top = sel[top_local]
    out_a[:n] = cache.values[top]
    out_s[:n] = score[top_local]
    out_m[:n] = 1.0
    return out_a, out_s, out_m


def attach_retrieval(store: FeatureStore, cache: RetrievalCache, normalizers: dict[str, Normalizer],
                     k: int, w_obs: float = 1.0, w_state: float = 1.0, exclude_self: bool = True) -> None:
    """Precompute per-frame retrieval for every episode and store it on the episode dicts.

    ``ChunkDataset`` then indexes ``ep["retr_*"][t]`` by frame, so the (potentially expensive)
    retrieval runs once up front rather than inside the data loader. ``exclude_self`` drops the
    episode's own cache entries (leave-one-out): harmless when ``store`` is disjoint from the
    cache, and essential when they overlap (e.g. the risk head trains on the cache episodes).
    """
    K, A = cache.pred_horizon, cache.action_dim
    for ep in store.episodes:
        T = ep["actions"].shape[0]
        ra = np.zeros((T, k, K, A), np.float32)
        rs = np.zeros((T, k), np.float32)
        rm = np.zeros((T, k), np.float32)
        exclude_ep = int(ep["episode_index"]) if exclude_self else None
        for t in range(1, T - K + 1):
            obs, state = make_keys(normalizers, ep["embeddings"][t], ep["states"][t])
            a, s, m = retrieve(cache, ep["task"], obs, state, k, w_obs, w_state, exclude_ep=exclude_ep)
            ra[t], rs[t], rm[t] = a, s, m
        ep["retr_actions"], ep["retr_sim"], ep["retr_mask"] = ra, rs, rm
