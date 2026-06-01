from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import REPO_ID


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task: str
    length: int
    parquet_path: str
    feature_path: str


def download_metadata(meta_dir: Path) -> tuple[Path, Path, Path]:
    meta_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        paths.append(
            Path(
                hf_hub_download(
                    REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=meta_dir,
                )
            )
        )
    return tuple(paths)  # type: ignore[return-value]


def select_episodes(episodes_jsonl: Path, task_regex: str, max_episodes: int, seed: int) -> list[dict]:
    pattern = re.compile(task_regex)
    selected = []
    with episodes_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            task = row["tasks"][0]
            if pattern.match(task):
                selected.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "task": task,
                        "length": int(row["length"]),
                    }
                )
    random.Random(seed).shuffle(selected)
    if max_episodes > 0:
        selected = selected[:max_episodes]
    selected.sort(key=lambda x: x["episode_index"])
    return selected


def parquet_name(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def download_episode_parquets(episodes: list[dict], data_dir: Path) -> list[EpisodeInfo]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out: list[EpisodeInfo] = []
    for i, ep in enumerate(episodes, 1):
        feature_path = data_dir / "features" / f"episode_{ep['episode_index']:06d}_r3m.npz"
        expected_path = data_dir / parquet_name(ep["episode_index"])
        if feature_path.exists():
            path = expected_path
        else:
            path = Path(
                hf_hub_download(
                    REPO_ID,
                    repo_type="dataset",
                    filename=parquet_name(ep["episode_index"]),
                    local_dir=data_dir,
                )
            )
        out.append(
            EpisodeInfo(
                episode_index=ep["episode_index"],
                task=ep["task"],
                length=ep["length"],
                parquet_path=str(path),
                feature_path=str(feature_path),
            )
        )
        if i % 10 == 0 or i == len(episodes):
            print(f"Downloaded/verified {i}/{len(episodes)} episode parquet files", flush=True)
    return out


def cleanup_episode_parquets(episodes: list[EpisodeInfo]) -> tuple[int, int]:
    removed = 0
    bytes_removed = 0
    for ep in episodes:
        parquet_path = Path(ep.parquet_path)
        feature_path = Path(ep.feature_path)
        if not feature_path.exists() or not parquet_path.exists():
            continue
        try:
            size = parquet_path.stat().st_size
            parquet_path.unlink()
        except FileNotFoundError:
            continue
        removed += 1
        bytes_removed += size
    return removed, bytes_removed


def split_episodes(
    episodes: list[EpisodeInfo], train_ratio: float, val_ratio: float, seed: int, stratified: bool = True
) -> tuple[list[EpisodeInfo], list[EpisodeInfo], list[EpisodeInfo]]:
    if stratified:
        return _stratified_split(episodes, train_ratio, val_ratio, seed)
    shuffled = episodes[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def _stratified_split(
    episodes: list[EpisodeInfo], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[EpisodeInfo], list[EpisodeInfo], list[EpisodeInfo]]:
    rng = random.Random(seed)
    groups: dict[str, list[EpisodeInfo]] = {}
    for ep in episodes:
        groups.setdefault(ep.task, []).append(ep)

    train: list[EpisodeInfo] = []
    val: list[EpisodeInfo] = []
    test: list[EpisodeInfo] = []
    for task, eps in sorted(groups.items()):
        eps = eps[:]
        rng.shuffle(eps)
        n = len(eps)
        if n < 3:
            train.extend(eps)
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train.extend(eps[:n_train])
        val.extend(eps[n_train : n_train + n_val])
        test.extend(eps[n_train + n_val :])

    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def write_manifest(
    path: Path,
    episodes: list[EpisodeInfo],
    splits: dict[str, list[EpisodeInfo]],
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": REPO_ID,
        "config": config,
        "episodes": [asdict(ep) for ep in episodes],
        "splits": {name: [ep.episode_index for ep in eps] for name, eps in splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
