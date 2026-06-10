"""Build the envelope-gate sigma file from demonstration features (no training).

The envelope gate decides, online, whether the local predictor's chunk may substitute
the VLA without ever training a risk head: it accepts a chunk when, for every step and
every *continuous* action dim, the predicted residual from the last executed action
stays within ``threshold * sigma`` -- plus a gripper veto. ``sigma`` is the per-dim
spread of demonstration actions.

Unlike the GR00T/PandaOmron port (identity action space), the VLABench action is 7-D
Euler and the model works in the 10-D sin/cos representation. To avoid angle wraparound
the envelope is built and scored entirely in the *normalized sin/cos repr space* -- the
same space the predictor natively emits its residual in (``pred_norm_abs - repeat_norm``).
The gripper is excluded from the continuous envelope and vetoed separately in raw space.

Continuous repr dims (the envelope target): position ``0:3`` + the six sin/cos angle
columns ``3:9``. The gripper is repr/raw col 6 (raw) / 9 (repr) and is handled by the
veto, not the envelope.

Usage:
    python -m r3m_action_predictor.build_envelope_gate \
        --action-run-dir outputs/<task>/<predictor> \
        --out outputs/<task>/<predictor>/envelope_gate.npz \
        --sigma-kind value --all-success
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .action_repr import REPR_ACTION_DIM, euler_to_repr
from .config import ExperimentConfig
from .data import FeatureStore, Normalizer
from .hf_data import EpisodeInfo

# Continuous repr dims covered by the envelope (pos 0:3 + sin/cos 3:9); gripper (9) excluded.
ENVELOPE_CONT_DIMS = list(range(0, REPR_ACTION_DIM - 1))
GRIP_RAW_IDX = 6  # gripper column in the raw 7-D Euler action (vetoed, not enveloped)


def _episodes_from_manifest(manifest: dict, splits: list[str]) -> list[EpisodeInfo]:
    by_index = {int(e["episode_index"]): e for e in manifest["episodes"]}
    want: list[int] = []
    for split in splits:
        want.extend(int(i) for i in manifest["splits"][split])
    fields = EpisodeInfo.__dataclass_fields__.keys()
    return [EpisodeInfo(**{k: by_index[i][k] for k in fields}) for i in want]


def _normalized_repr_actions(actions: np.ndarray, action_norm: Normalizer) -> np.ndarray:
    """Raw 7-D Euler episode actions -> normalized 10-D sin/cos repr (same space as the model)."""
    return action_norm.encode(euler_to_repr(actions.astype(np.float32)))


def build_sigma(
    store: FeatureStore,
    action_norm: Normalizer,
    sigma_kind: str,
    offset_horizons: list[int],
) -> tuple[np.ndarray, int]:
    """Per-dim sigma in normalized repr space. ``value`` = std of action values;
    ``offset`` = std of multi-horizon offsets ``a[j+m]-a[j]`` (the residual scale)."""
    chunks: list[np.ndarray] = []
    for _, _, actions in store.iter_arrays():
        rep = _normalized_repr_actions(actions, action_norm)  # [T, 10]
        if sigma_kind == "value":
            chunks.append(rep)
        elif sigma_kind == "offset":
            for m in offset_horizons:
                if rep.shape[0] > m:
                    chunks.append(rep[m:] - rep[:-m])
        else:
            raise ValueError(f"Unknown sigma-kind: {sigma_kind}")
    if not chunks:
        raise RuntimeError("No demonstration frames found to estimate sigma.")
    stacked = np.concatenate(chunks, axis=0)
    sigma = stacked.std(axis=0).astype(np.float32)
    return sigma, len(store.episodes)   # raw per-dim std; the meaningful floor is applied by the caller


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action-run-dir", type=Path, required=True,
                    help="Predictor run dir containing best_model.pt and manifest.json")
    ap.add_argument("--out", type=Path, required=True, help="Output envelope_gate.npz path")
    ap.add_argument("--splits", default="train",
                    help="Comma-separated manifest splits to fit sigma on (default: train)")
    ap.add_argument("--sigma-kind", choices=("value", "offset"), default="value")
    ap.add_argument("--offset-horizons", default="1,2,4,8",
                    help="Offsets m for sigma-kind=offset (a[j+m]-a[j])")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="Default sigma multiplier stored in the file (calibrated later)")
    ap.add_argument("--gripper-threshold", type=float, default=0.5)
    ap.add_argument("--sigma-floor", type=float, default=0.05,
                    help="Floor on each per-dim sigma (normalized repr units). A near-constant angle "
                         "dim (e.g. cos of a fixed wrist roll/pitch) has sigma->0, which makes "
                         "|residual|/sigma explode and dominate the max envelope score; flooring keeps "
                         "the score driven by dims that actually vary, while a large anomalous "
                         "deviation on a near-constant dim still raises the score. 0 disables.")
    ap.add_argument("--all-success", action="store_true",
                    help="Treat every episode as success (VLABench demos are all successful). "
                         "Kept for parity with the rollout-feature path; demo features have no "
                         "is_success flag so this is the expected mode here.")
    args = ap.parse_args()

    import torch

    ckpt = torch.load(args.action_run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = ExperimentConfig().to_jsonable()
    config.update(ckpt["config"])
    action_norm = Normalizer(
        np.asarray(ckpt["normalizers"]["action"]["mean"], dtype=np.float32),
        np.asarray(ckpt["normalizers"]["action"]["std"], dtype=np.float32),
    )
    action_dim = int(action_norm.mean.shape[-1])
    if action_dim != REPR_ACTION_DIM:
        raise ValueError(
            f"Envelope gate expects the {REPR_ACTION_DIM}-D sin/cos repr action space, "
            f"got action_dim={action_dim}. (Raw-Euler checkpoints are not supported here.)"
        )

    manifest = json.loads((args.action_run_dir / "manifest.json").read_text())
    episodes = _episodes_from_manifest(manifest, args.splits.split(","))
    store = FeatureStore(episodes)

    offset_horizons = [int(x) for x in args.offset_horizons.split(",") if x]
    raw_sigma, n_success = build_sigma(store, action_norm, args.sigma_kind, offset_horizons)
    # Floor near-constant dims so they cannot blow up / dominate the max envelope score.
    sigma = np.maximum(raw_sigma, np.float32(args.sigma_floor)) if args.sigma_floor > 0 else raw_sigma
    floored = [int(i) for i in np.where(raw_sigma < args.sigma_floor)[0]]

    grip_mid = float(action_norm.mean[REPR_ACTION_DIM - 1])  # gripper repr-mean (diagnostic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        sigma=sigma,
        raw_sigma=raw_sigma,
        sigma_floor=np.float32(args.sigma_floor),
        cont_dims=np.asarray(ENVELOPE_CONT_DIMS, dtype=np.int64),
        grip_raw_idx=np.int64(GRIP_RAW_IDX),
        grip_mid=np.float32(grip_mid),
        gripper_threshold=np.float32(args.gripper_threshold),
        threshold=np.float32(args.threshold),
        sigma_kind=args.sigma_kind,
        action_dim=np.int64(action_dim),
        n_success=np.int64(n_success),
        all_success=np.bool_(args.all_success),
    )
    print(f"Wrote {args.out}")
    print(f"  episodes={n_success}  sigma_kind={args.sigma_kind}  cont_dims={ENVELOPE_CONT_DIMS}")
    print(f"  sigma_floor={args.sigma_floor}  floored_dims(raw sigma < floor)={floored}")
    np.set_printoptions(precision=4, suppress=True)
    print(f"  sigma(cont)={sigma[ENVELOPE_CONT_DIMS]}")
    print(f"  sigma(gripper)={sigma[REPR_ACTION_DIM - 1]:.4f}  grip_mid={grip_mid:.4f}")


if __name__ == "__main__":
    main()
