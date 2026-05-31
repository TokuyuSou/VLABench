from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image

from .config import VIEWS
from .hf_data import EpisodeInfo


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_r3m_repo(cache_dir: Path) -> Path:
    r3m_dir = cache_dir / "r3m"
    if not r3m_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/r3m.git", str(r3m_dir)])
    return r3m_dir


def load_r3m_from_repo(r3m_dir: Path, model_name: str) -> nn.Module:
    sys.path.insert(0, str(r3m_dir))
    from r3m import load_r3m

    model = load_r3m(model_name)
    model.eval()
    return model


def pil_bytes_to_chw_uint8(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def encode_images(
    r3m: nn.Module,
    image_records: list[dict],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(image_records), batch_size):
        batch_records = image_records[start : start + batch_size]
        images = torch.stack([pil_bytes_to_chw_uint8(row["bytes"]) for row in batch_records])
        images = images.to(device=device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                emb = r3m(images, obs_shape=[3, images.shape[-2], images.shape[-1]])
        chunks.append(emb.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def extract_episode_features(
    ep: EpisodeInfo,
    r3m: nn.Module,
    batch_size: int,
    device: torch.device,
    force: bool,
) -> None:
    feature_path = Path(ep.feature_path)
    if feature_path.exists() and not force:
        return

    table = pq.read_table(ep.parquet_path, columns=[*VIEWS, "state", "actions"])
    data = table.to_pydict()
    states = np.asarray(data["state"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    view_embeddings = [
        encode_images(r3m, data[view], batch_size=batch_size, device=device).astype(np.float16)
        for view in VIEWS
    ]
    embeddings = np.stack(view_embeddings, axis=1)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        embeddings=embeddings,
        states=states,
        actions=actions,
        episode_index=np.asarray(ep.episode_index, dtype=np.int64),
        task=np.asarray(ep.task),
    )


def extract_all_features(
    episodes: list[EpisodeInfo],
    r3m_model: str,
    batch_size: int,
    force: bool,
    cache_dir: Path,
) -> int:
    missing = [ep for ep in episodes if force or not Path(ep.feature_path).exists()]
    if not missing:
        print("All R3M feature files already exist; reusing them.", flush=True)
        return 0

    r3m_dir = ensure_r3m_repo(cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading R3M {r3m_model} on {device}", flush=True)
    r3m = load_r3m_from_repo(r3m_dir, r3m_model).to(device)

    t0 = time.time()
    for i, ep in enumerate(missing, 1):
        extract_episode_features(ep, r3m, batch_size, device, force)
        if i % 5 == 0 or i == len(missing):
            elapsed = time.time() - t0
            print(f"Extracted R3M features {i}/{len(missing)} in {elapsed/60:.1f} min", flush=True)
    return len(missing)
