from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ID = "VLABench/vlabench_primitive_ft_lerobot"
VIEWS = ("image", "second_image", "wrist_image")
ACTION_DIM = 7
STATE_DIM = 7


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    task_name: str = "add_condiment"
    task_regex: str = r"^Add .* to the dish$"
    max_episodes: int = 100
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    stratified_split: bool = True
    prev_horizon: int = 8
    pred_horizon: int = 8
    r3m_model: str = "resnet18"
    embedding_batch_size: int = 96
    force_features: bool = False
    epochs: int = 35
    batch_size: int = 1024
    model_kind: str = "mlp_gru"
    target_mode: str = "absolute"
    view_dim: int = 128
    width: int = 256
    hidden_dim: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 4
    dropout: float = 0.05
    nll_weight: float = 1.0
    mse_weight: float = 0.25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 7
    data_dir: Path = experiment_root() / "data" / "vlabench_primitive_ft_lerobot"
    meta_dir: Path = experiment_root() / "_hf_meta"
    cache_dir: Path = experiment_root() / "cache"
    output_dir: Path = experiment_root() / "outputs" / "add_condiment_r3m18_chunk8"

    def to_jsonable(self) -> dict:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        return out
