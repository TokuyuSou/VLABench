"""Calibrate the envelope/aleatoric gate threshold for a target VLA-skip fraction.

These gates score LOWER = safer (substitute when ``score <= threshold``), the opposite of
the risk/confidence gates handled by ``calibrate_threshold.py``. The threshold is chosen as a
function of the target skip rate (never hand-picked). Because the deployed (hybrid) score
distribution drifts from the demonstrations (covariate shift), calibration is two-pass:

  1. OFFLINE (default): run the frozen predictor over demonstration chunks, build the env-score
     distribution and the gripper-change mask, and pick the threshold that passes the
     cooldown-adjusted conditional rate C. Each substitute forces ``cooldown`` VLA calls, so the
     realized substitution rate R and the conditional pass rate C relate by
         R ~= C / (1 + cooldown * C)   ==>   C = R / (1 - cooldown * R).

  2. ONLINE (``--from-run-dir <run1>``): replay run #1's logged per-chunk ``decision_score``
     through the exact per-episode gating to re-pick the threshold on the *deployed* score CDF.

Usage:
    # pass 1 (offline, from demo features)
    python -m r3m_action_predictor.calibrate_gate_threshold \
        --action-run-dir outputs/<task>/<predictor> \
        --envelope outputs/<task>/<predictor>/envelope_gate.npz \
        --metric envelope --target-skip 0.10 [--splits train]

    # pass 2 (online, from the run-1 chunk log)
    python -m r3m_action_predictor.calibrate_gate_threshold \
        --from-run-dir eval_runs/<run1> --metric envelope --target-skip 0.10 \
        [--min-vla 1 --cooldown 1 --max-consecutive 0]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------
# Shared gating simulation (per-episode, lower-is-better)
# --------------------------------------------------------------------------------------
def simulate(eps, thr, min_vla, cooldown, max_consec):
    """Replay the per-episode gate at threshold ``thr``; return (subs, replans).

    ``eps`` maps episode -> list of (score, gripper_changed) in replan order. A chunk
    substitutes when score is not None, ``score <= thr``, not gripper-changed, and the
    structural gates (per-episode min-VLA / cooldown / consecutive) allow it."""
    subs = repl = 0
    for chunks in eps.values():
        vla_calls = cd = consec = 0
        for score, grip_changed in chunks:
            repl += 1
            can = (
                score is not None
                and not grip_changed
                and score <= thr
                and vla_calls >= min_vla
                and cd == 0
                and (max_consec <= 0 or consec < max_consec)
            )
            if can:
                subs += 1
                consec += 1
                cd = cooldown
            else:
                vla_calls += 1
                consec = 0
                if cd > 0:
                    cd -= 1
    return subs, repl


def pick_threshold(eps, target, grid, min_vla, cooldown, max_consec):
    curve = [(t, simulate(eps, t, min_vla, cooldown, max_consec)) for t in grid]
    curve = [(t, s / r if r else 0.0) for t, (s, r) in curve]
    best = min(curve, key=lambda ts: abs(ts[1] - target))
    return best, curve


# --------------------------------------------------------------------------------------
# Pass 2: online replay from a run's chunk_decisions.jsonl
# --------------------------------------------------------------------------------------
def load_run_episodes(path: Path):
    rows = collections.defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        if d.get("event") != "chunk_decision":
            continue
        order = d.get("replan_index", d.get("global_step", 0))
        score = d.get("candidate_env_score", d.get("decision_score"))
        grip = bool(d.get("candidate_gripper_changed", False))
        rows[d["episode_index"]].append((order, score, grip))
    eps = {}
    for e, lst in rows.items():
        lst.sort(key=lambda x: x[0])
        eps[e] = [(s, g) for _, s, g in lst]
    return eps


# --------------------------------------------------------------------------------------
# Pass 1: offline from demonstration features
# --------------------------------------------------------------------------------------
def offline_scores(action_run_dir: Path, envelope_path: Path, metric: str, splits: list[str], device: str):
    """Run the frozen predictor over demo chunks -> per-chunk (env_score, gripper_changed)."""
    import torch

    from .build_envelope_gate import _episodes_from_manifest
    from .config import ExperimentConfig
    from .data import ChunkDataset, FeatureStore, Normalizer
    from .model import build_model, run_model
    from .risk import action_output_to_raw_and_norm

    ckpt = torch.load(action_run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = ExperimentConfig().to_jsonable()
    config.update(ckpt["config"])
    normalizers = {
        k: Normalizer(np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
        for k, v in ckpt["normalizers"].items()
    }
    config["action_dim"] = int(normalizers["action"].mean.shape[-1])

    env = dict(np.load(envelope_path, allow_pickle=True))
    sigma = np.asarray(env["sigma"], dtype=np.float32)
    cont = np.asarray(env["cont_dims"], dtype=np.int64)
    grip_idx = int(env["grip_raw_idx"])
    grip_thr = float(env["gripper_threshold"])

    manifest = json.loads((action_run_dir / "manifest.json").read_text())
    episodes = _episodes_from_manifest(manifest, splits)
    store = FeatureStore(episodes)
    ds = ChunkDataset(store, normalizers, config["prev_horizon"], config["pred_horizon"])

    dev = torch.device(device)
    model = build_model(config, embed_dim=store.episodes[0]["embeddings"].shape[-1],
                        num_views=store.episodes[0]["embeddings"].shape[1])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(dev)
    action_mean = torch.from_numpy(normalizers["action"].mean).to(dev)
    action_std = torch.from_numpy(normalizers["action"].std).to(dev)

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=2)
    scores, grip_changed = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            model_out = run_model(model, batch)
            pred_raw, pred_norm_abs, _ = action_output_to_raw_and_norm(
                model_out, batch, action_mean, action_std, config["target_mode"]
            )
            repeat_norm = batch["prev_actions"][:, -1:, :].repeat(1, pred_norm_abs.shape[1], 1)
            delta = (pred_norm_abs - repeat_norm).abs()
            if metric == "envelope":
                s = (delta[:, :, cont] / torch.from_numpy(sigma[cont]).to(dev)).amax(dim=(1, 2))
            else:  # aleatoric: predicted std over continuous dims
                _, log_std, _ = _split(model_out)
                std = torch.exp(log_std)
                s = std[:, :, cont].mean(dim=(1, 2))
            last = batch["raw_prev_actions"][:, -1, grip_idx]
            gc = ((pred_raw[:, :, grip_idx] > grip_thr) != (last[:, None] > grip_thr)).any(dim=1)
            scores.append(s.cpu().numpy())
            grip_changed.append(gc.cpu().numpy())
    return np.concatenate(scores), np.concatenate(grip_changed)


def _split(model_out):
    from .model import split_model_output

    return split_model_output(model_out)


def offline_pick(scores: np.ndarray, grip_changed: np.ndarray, target_skip: float, cooldown: int):
    """Pick the threshold so the cooldown-adjusted conditional pass rate C is met:
    mean((score <= thr) & ~gripper_changed) == C, with C = R / (1 - cooldown*R)."""
    denom = 1.0 - cooldown * target_skip
    C = target_skip / denom if denom > 1e-9 else 1.0
    C = float(min(max(C, 0.0), 1.0))
    ok = ~grip_changed
    grid = np.unique(scores)
    best_thr, best_gap = float(grid[-1]), 1e9
    for thr in grid:
        frac = float(((scores <= thr) & ok).mean())
        gap = abs(frac - C)
        if gap < best_gap:
            best_gap, best_thr = gap, float(thr)
    realized = float(((scores <= best_thr) & ok).mean())
    return best_thr, C, realized


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", choices=("envelope", "aleatoric"), default="envelope")
    ap.add_argument("--target-skip", type=float, default=0.10)
    ap.add_argument("--cooldown", type=int, default=1)
    ap.add_argument("--min-vla", type=int, default=1)
    ap.add_argument("--max-consecutive", type=int, default=0)
    ap.add_argument("--emit", default="", help="if set, write 'target thr skip' line(s) here")
    # pass 2 (online)
    ap.add_argument("--from-run-dir", type=Path, default=None,
                    help="Run dir (or chunk_decisions.jsonl) to recalibrate on deployed scores")
    # pass 1 (offline)
    ap.add_argument("--action-run-dir", type=Path, default=None)
    ap.add_argument("--envelope", type=Path, default=None)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.from_run_dir is not None:
        path = a.from_run_dir
        if path.is_dir():
            hits = list(path.rglob("chunk_decisions.jsonl"))
            if not hits:
                raise FileNotFoundError(f"No chunk_decisions.jsonl under {path}")
            path = hits[0]
        eps = load_run_episodes(path)
        grid = [round(x, 4) for x in np.linspace(0.0, 5.0, 1001)]
        (thr, skip), curve = pick_threshold(eps, a.target_skip, grid, a.min_vla, a.cooldown, a.max_consecutive)
        print(f"[online] episodes={len(eps)} metric={a.metric}")
        print(f"[online] target_skip={a.target_skip:.2f} -> threshold={thr:.4f} predicted_skip={skip:.4f}")
        result = (a.target_skip, thr, skip)
    else:
        if a.action_run_dir is None or a.envelope is None:
            ap.error("offline mode needs --action-run-dir and --envelope (or use --from-run-dir)")
        scores, grip = offline_scores(a.action_run_dir, a.envelope, a.metric, a.splits.split(","), a.device)
        thr, C, realized = offline_pick(scores, grip, a.target_skip, a.cooldown)
        print(f"[offline] chunks={len(scores)} gripper_change_rate={float(grip.mean()):.4f}")
        print(f"[offline] target_skip={a.target_skip:.2f} cooldown={a.cooldown} -> "
              f"conditional_pass_C={C:.4f} threshold={thr:.4f} (offline pass-rate={realized:.4f})")
        print("[offline] NOTE: run eval #1 then recalibrate with --from-run-dir for the on-target threshold.")
        result = (a.target_skip, thr, realized)

    if a.emit:
        Path(a.emit).write_text(f"{result[0]:.2f} {result[1]:.4f} {result[2]:.4f}\n")


if __name__ == "__main__":
    main()
