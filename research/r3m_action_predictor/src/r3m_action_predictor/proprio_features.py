"""Lightweight proprio-only feature cache for no-vision predictors.

A no-vision (proprio-only) predictor ignores the visual tokens entirely, so it never needs the
R3M embeddings -- only the per-frame ``state`` and ``actions`` sequences. Those two columns live
in the same LeRobot parquet as the (large) image columns, but parquet column projection lets us
read just them over HTTP range requests via the HF filesystem: no 60+ MiB parquet download, no
image decode, no R3M forward (~0.6 s/episode vs ~5 s for the full R3M path, and no GPU/disk).

The cache lives in a separate ``features_proprio/`` dir (distinct from the R3M ``features/`` dir)
and stores only states/actions; ``FeatureStore`` synthesises a zero embedding placeholder for the
shared schema. Use it via ``cli.py --no-vision`` or the standalone ``extract_all_proprio.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import REPO_ID
from .hf_data import EpisodeInfo, parquet_name

PROPRIO_SUBDIR = "features_proprio"


def proprio_feature_path(data_dir: str | Path, episode_index: int) -> Path:
    return Path(data_dir) / PROPRIO_SUBDIR / f"episode_{episode_index:06d}.npz"


def _read_state_actions(fs, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read only the state/actions columns of one episode's parquet over the HF filesystem."""
    path = f"datasets/{REPO_ID}/{parquet_name(episode_index)}"
    pf = pq.ParquetFile(fs.open(path, "rb"))
    table = pf.read(columns=["state", "actions"])
    states = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
    return states, actions


def ensure_proprio_features(
    selected: list[dict],
    data_dir: str | Path,
    *,
    force: bool = False,
    log_every: int = 50,
) -> tuple[list[EpisodeInfo], int]:
    """Ensure a proprio (state+actions) cache exists for each selected episode.

    ``selected`` is the same list of ``{"episode_index", "task", "length"}`` dicts produced by
    ``hf_data.select_episodes``. Returns ``(episodes, newly_extracted)`` where ``episodes`` is a
    list of ``EpisodeInfo`` whose ``feature_path`` points at the proprio npz (drop-in for
    ``FeatureStore``). The HF filesystem handle is created lazily, so a fully-cached run does no
    network I/O at all."""
    data_dir = Path(data_dir)
    fs = None
    out: list[EpisodeInfo] = []
    extracted = 0
    for i, ep in enumerate(selected, 1):
        idx = int(ep["episode_index"])
        fpath = proprio_feature_path(data_dir, idx)
        if force or not fpath.exists():
            if fs is None:
                from huggingface_hub import HfFileSystem

                fs = HfFileSystem()
            states, actions = _read_state_actions(fs, idx)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fpath,
                states=states,
                actions=actions,
                episode_index=np.asarray(idx, dtype=np.int64),
                task=np.asarray(ep["task"]),
            )
            extracted += 1
        out.append(
            EpisodeInfo(
                episode_index=idx,
                task=ep["task"],
                length=int(ep["length"]),
                parquet_path="",  # never downloaded for proprio
                feature_path=str(fpath),
            )
        )
        if log_every and (i % log_every == 0 or i == len(selected)):
            print(f"Proprio features {i}/{len(selected)} (newly extracted {extracted})", flush=True)
    return out, extracted
