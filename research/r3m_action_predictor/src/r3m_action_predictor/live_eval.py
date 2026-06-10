from __future__ import annotations

import collections
import dataclasses
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import Policy
from VLABench.robots import *  # noqa: F403,F401 - registers robot classes for load_env
from VLABench.tasks import *  # noqa: F403,F401 - registers task classes for load_env
from VLABench.utils.utils import quaternion_to_euler

from .inference import OnlineR3MActionPredictor
from .metrics import skip_by_outcome


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 5
    tasks: str = "add_condiment"
    eval_track: str = "track_1_in_distribution"
    n_episode: int = 50
    metrics: str = "success_rate intention_score progress_score"
    save_dir: str = "research/r3m_action_predictor/eval_runs/add_condiment_hybrid"
    visulization: bool = True
    confidence_threshold: float = 0.979
    min_step_confidence: float = 0.94
    max_substitution_fraction: float | None = None
    min_vla_calls_before_substitute: int = 3
    vla_cooldown_after_substitute: int = 1
    max_consecutive_substitutions: int = 0
    checkpoint_path: str = (
        "research/r3m_action_predictor/outputs/"
        "add_condiment_r3m18_residual_transformer_200/best_model.pt"
    )
    risk_head_path: str | None = None
    decision_metric: str = "confidence"
    risk_threshold: float | None = None
    # Envelope / aleatoric gates (learning-free; decision_metric="envelope"|"aleatoric").
    # For these gates a LOWER score means safer-to-substitute, so a candidate is accepted
    # when score <= threshold (the opposite direction from risk/confidence).
    envelope_gate_path: str | None = None
    envelope_threshold: float | None = None
    aleatoric_threshold: float | None = None
    r3m_cache_dir: str = "research/r3m_action_predictor/cache"
    predictor_device: str = "cpu"
    log_full_chunks: bool = True
    # Familiarity veto (observation gate; OFF unless familiarity_pool_path is set).
    # Vetoes a substitution when the current R3M view embeddings sit farther from the
    # predictor's train demos (cosine kNN, see FamiliarityGate) than the threshold.
    # Pool + val-calibrated default threshold come from build_familiarity_pool.py.
    familiarity_pool_path: str | None = None
    familiarity_threshold: float | None = None


class FamiliarityGate:
    """Observation-familiarity veto for the substitution gate.

    dk = mean over the 3 views of (1 - mean top-k cosine similarity) between the current
    raw R3M view embeddings and the predictor's train-demo pool (built offline by
    build_familiarity_pool.py). Substitution is vetoed when dk > threshold; a false veto
    only costs one extra VLA call. Offline analysis (select_toy/select_poker sweeps):
    dk is uncorrelated with the envelope score, and executed substitutions with high dk
    concentrate 98-100% in ultimately-failed episodes."""

    def __init__(self, pool_path: str | Path, device: str, threshold: float | None = None, topk: int | None = None):
        import torch

        self._torch = torch
        data = np.load(pool_path, allow_pickle=True)
        emb = torch.from_numpy(np.asarray(data["embeddings"], dtype=np.float32))
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.device = torch.device(device)
        self.pool = emb.to(self.device)  # (N, V, 512), L2-normalized per view
        self.topk = int(topk if topk is not None else data.get("topk", 5))
        self.threshold = float(threshold if threshold is not None else data["suggested_threshold"])
        self.pool_path = str(pool_path)

    @property
    def n_frames(self) -> int:
        return int(self.pool.shape[0])

    def distance(self, view_embeddings: np.ndarray) -> float:
        """view_embeddings: (V, 512) raw R3M output of encode_views (view order = VIEWS)."""
        t = self._torch
        with t.no_grad():
            q = t.from_numpy(np.asarray(view_embeddings, dtype=np.float32)).to(self.device)
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            sims = t.einsum("vd,nvd->nv", q, self.pool)  # (N, V)
            top = sims.topk(min(self.topk, sims.shape[0]), dim=0).values  # (k, V)
            return float((1.0 - top.mean(dim=0)).mean().item())


class HybridPi0Policy(Policy):
    def __init__(
        self,
        client,
        predictor: OnlineR3MActionPredictor,
        log_dir: Path,
        *,
        replan_steps: int,
        confidence_threshold: float,
        min_step_confidence: float,
        decision_metric: str,
        risk_threshold: float | None,
        max_substitution_fraction: float | None,
        min_vla_calls_before_substitute: int,
        vla_cooldown_after_substitute: int,
        max_consecutive_substitutions: int,
        log_full_chunks: bool,
        envelope_gate: dict | None = None,
        envelope_threshold: float | None = None,
        aleatoric_threshold: float | None = None,
        familiarity_gate: FamiliarityGate | None = None,
    ):
        self.model = client
        self.predictor = predictor
        self.familiarity_gate = familiarity_gate
        self.familiarity_vetoes = 0
        self.replan_steps = replan_steps
        self.confidence_threshold = confidence_threshold
        self.min_step_confidence = min_step_confidence
        self.decision_metric = decision_metric
        self.risk_threshold = risk_threshold
        # Envelope/aleatoric gate config. cont_dims/sigma/grip come from the npz built by
        # build_envelope_gate; aleatoric falls back to the continuous repr dims if no file.
        self.envelope_gate = envelope_gate
        self.envelope_threshold = envelope_threshold
        self.aleatoric_threshold = aleatoric_threshold
        if envelope_gate is not None:
            self._env_sigma = np.asarray(envelope_gate["sigma"], dtype=np.float32)
            self._env_cont_dims = np.asarray(envelope_gate["cont_dims"], dtype=np.int64)
            self._grip_raw_idx = int(envelope_gate["grip_raw_idx"])
            self._grip_threshold = float(envelope_gate["gripper_threshold"])
            if envelope_threshold is None and "threshold" in envelope_gate:
                self.envelope_threshold = float(envelope_gate["threshold"])
        else:
            self._env_sigma = None
            self._env_cont_dims = np.arange(0, 9, dtype=np.int64)  # pos(3)+sincos(6)
            self._grip_raw_idx = 6
            self._grip_threshold = 0.5
        self.max_substitution_fraction = max_substitution_fraction
        self.min_vla_calls_before_substitute = min_vla_calls_before_substitute
        self.vla_cooldown_after_substitute = vla_cooldown_after_substitute
        self.max_consecutive_substitutions = max_consecutive_substitutions
        self.log_full_chunks = log_full_chunks

        self.action_plan = collections.deque()
        self.action_meta = collections.deque()
        predictor_config = getattr(self.predictor.predictor, "config", {})
        self.prev_horizon = int(predictor_config.get("prev_horizon", replan_steps))
        self.pred_horizon = int(predictor_config.get("pred_horizon", replan_steps))
        if self.replan_steps > self.pred_horizon:
            raise ValueError(
                f"replan_steps={self.replan_steps} exceeds predictor pred_horizon={self.pred_horizon}"
            )
        self.prev_actions = collections.deque(maxlen=max(self.prev_horizon, 1))
        obs_horizon = int(predictor_config.get("obs_horizon", 1))
        self.image_history = collections.deque(maxlen=max(obs_horizon, 1))

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.step_log_path = log_dir / "step_events.jsonl"
        self.chunk_log_path = log_dir / "chunk_decisions.jsonl"
        self.summary_path = log_dir / "policy_summary.json"
        self._step_f = self.step_log_path.open("w")
        self._chunk_f = self.chunk_log_path.open("w")

        self.episode_index = -1
        self.episode_step = 0
        self.global_step = 0
        self.replan_index = 0
        self.vla_calls = 0
        self.substitutions = 0
        self.rejected_candidates = 0
        self.cooldown_remaining = 0
        self.consecutive_substitutions = 0
        self.episode_stats: list[dict[str, Any]] = []
        self._current_episode_stats: dict[str, Any] | None = None

    def reset(self):
        self._finalize_episode_stats()
        self.episode_index += 1
        self.episode_step = 0
        self.replan_index = 0
        self.cooldown_remaining = 0
        self.consecutive_substitutions = 0
        self.action_plan.clear()
        self.action_meta.clear()
        self.prev_actions.clear()
        self.image_history.clear()
        self._current_episode_stats = {
            "episode_index": self.episode_index,
            "vla_calls": 0,
            "substitutions": 0,
            "rejected_candidates": 0,
            "executed_steps": 0,
            "candidate_confidences": [],
            "used_confidences": [],
        }

    def predict(self, obs, **kwargs):
        self._remember_observation(obs)
        if len(self.action_plan) == 0:
            self._plan_next_chunk(obs)

        action = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        meta = self.action_meta.popleft()
        self.prev_actions.append(action.copy())

        target_pos, target_euler, gripper = action[:3], action[3:6], float(action[-1])
        gripper_state = np.ones(2) * 0.04 if gripper >= 0.1 else np.zeros(2)
        abs_target_pos = target_pos.copy() + obs["robot_frame"]

        self._log_step(obs, action, abs_target_pos, target_euler, gripper_state, meta)
        self.episode_step += 1
        self.global_step += 1
        if self._current_episode_stats is not None:
            self._current_episode_stats["executed_steps"] += 1
        return abs_target_pos, target_euler, gripper_state

    def _plan_next_chunk(self, obs) -> None:
        state = _policy_state_from_obs(obs)
        candidate = None
        decision = {
            "event": "chunk_decision",
            "episode_index": self.episode_index,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "replan_index": self.replan_index,
            "history_ready": len(self.prev_actions) >= self.prev_horizon,
            "prev_horizon": self.prev_horizon,
            "pred_horizon": self.pred_horizon,
            "threshold": self.confidence_threshold,
            "min_step_confidence": self.min_step_confidence,
            "decision_metric": self.decision_metric,
            "risk_threshold": self.risk_threshold,
            "max_substitution_fraction": self.max_substitution_fraction,
            "cooldown_remaining": self.cooldown_remaining,
            "consecutive_substitutions": self.consecutive_substitutions,
            "max_consecutive_substitutions": self.max_consecutive_substitutions,
        }

        if len(self.prev_actions) >= self.prev_horizon:
            second_image, _, image, image_wrist = obs["rgb"]
            candidate = self.predictor.predict_from_observation(
                image=image,
                second_image=second_image,
                wrist_image=image_wrist,
                state=state,
                prev_actions=np.asarray(self.prev_actions, dtype=np.float32),
                image_history=list(self.image_history),
                task=obs.get("instruction"),
            )
            conf = float(candidate.get("confidence", 0.0))
            conf_per_step = np.asarray(candidate.get("confidence_per_step", []), dtype=np.float32)
            min_exec_conf = float(np.min(conf_per_step[: self.replan_steps])) if len(conf_per_step) else 0.0
            risk_p_safe = candidate.get("risk_safe_probability")
            prefix_probs = np.asarray(candidate.get("prefix_safe_probability", []), dtype=np.float32)
            risk_threshold = self.risk_threshold
            if risk_threshold is None:
                risk_threshold = candidate.get("risk_threshold")
            accepted_prefix_len = self._accepted_prefix_len(prefix_probs, risk_threshold)
            decision.update(
                {
                    "candidate_confidence": conf,
                    "candidate_min_exec_step_confidence": min_exec_conf,
                    "candidate_risk_safe_probability": risk_p_safe,
                    "candidate_prefix_safe_probability": prefix_probs.tolist() if len(prefix_probs) else None,
                    "candidate_accepted_prefix_len": accepted_prefix_len,
                    "candidate_risk_threshold": risk_threshold,
                    "candidate_action_std_mean": float(np.mean(candidate["action_std"])),
                    "candidate_action_std_max": float(np.max(candidate["action_std"])),
                    "candidate_actions_first": candidate["actions"][: self.replan_steps].tolist()
                    if self.log_full_chunks
                    else None,
                    "candidate_confidence_per_step": conf_per_step.tolist() if self.log_full_chunks else None,
                }
            )
            if self._current_episode_stats is not None:
                self._current_episode_stats["candidate_confidences"].append(conf)
            if self.decision_metric in ("envelope", "aleatoric"):
                decision["candidate_env_score"] = self._gate_score(candidate)
                decision["candidate_gripper_changed"] = self._gripper_changes(candidate)
            if self.familiarity_gate is not None:
                emb = self.predictor.encode_views(
                    image=image, second_image=second_image, wrist_image=image_wrist
                )
                decision["familiarity_dk"] = self.familiarity_gate.distance(emb)
                decision["familiarity_threshold"] = self.familiarity_gate.threshold

        use_substitute, reject_reasons = self._should_substitute(decision, candidate)
        decision["use_substitute"] = use_substitute
        decision["reject_reasons"] = reject_reasons
        score, threshold = self._decision_score_and_threshold(decision)
        decision["decision_score"] = score
        decision["decision_threshold"] = threshold

        if use_substitute:
            exec_len = int(decision.get("candidate_accepted_prefix_len") or self.replan_steps)
            actions = np.asarray(candidate["actions"], dtype=np.float32)[:exec_len]
            self.substitutions += 1
            self.consecutive_substitutions += 1
            self.cooldown_remaining = self.vla_cooldown_after_substitute
            if self._current_episode_stats is not None:
                self._current_episode_stats["substitutions"] += 1
                self._current_episode_stats["used_confidences"].append(float(score or 0.0))
            source = "substitute"
            confidence = float(score or 0.0)
            vla_called = False
        else:
            if candidate is not None:
                self.rejected_candidates += 1
                if self._current_episode_stats is not None:
                    self._current_episode_stats["rejected_candidates"] += 1
            actions = self._call_vla(obs, state)
            source = "vla"
            confidence = None
            vla_called = True
            self.vla_calls += 1
            self.consecutive_substitutions = 0
            if self._current_episode_stats is not None:
                self._current_episode_stats["vla_calls"] += 1
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

        decision.update(
            {
                "source": source,
                "vla_called": vla_called,
                "vla_calls_total": self.vla_calls,
                "substitutions_total": self.substitutions,
                "replacement_fraction_so_far": self._replacement_fraction(),
            }
        )
        self._write_jsonl(self._chunk_f, decision)
        for i, action in enumerate(actions[: self.replan_steps]):
            self.action_plan.append(np.asarray(action, dtype=np.float32))
            self.action_meta.append(
                {
                    "source": source,
                    "replan_index": self.replan_index,
                    "within_chunk_index": i,
                    "chunk_confidence": confidence,
                    "chunk_decision_metric": self.decision_metric,
                    "chunk_risk_safe_probability": candidate.get("risk_safe_probability") if candidate else None,
                    "chunk_prefix_safe_probability": candidate.get("prefix_safe_probability").tolist()
                    if candidate and candidate.get("prefix_safe_probability") is not None
                    else None,
                    "chunk_executed_len": len(actions),
                    "vla_called_for_chunk": vla_called,
                }
            )
        self.replan_index += 1

    def _should_substitute(self, decision: dict, candidate: dict | None) -> tuple[bool, list[str]]:
        reasons = []
        if candidate is None:
            reasons.append("no_candidate_or_history")
            return False, reasons
        if self.decision_metric in ("envelope", "aleatoric"):
            # Learning-free gates: LOWER score = safer. Accept when score <= threshold and the
            # predicted gripper open/close state never flips within the executed prefix.
            score = decision.get("candidate_env_score")
            threshold = self.envelope_threshold if self.decision_metric == "envelope" else self.aleatoric_threshold
            if score is None or threshold is None:
                reasons.append("gate_score_unavailable")
            elif float(score) > float(threshold):
                reasons.append("envelope_above_threshold")
            if decision.get("candidate_gripper_changed"):
                reasons.append("gripper_change")
        elif self.decision_metric == "risk":
            prefix_len = decision.get("candidate_accepted_prefix_len")
            if prefix_len is not None:
                if prefix_len < 1:
                    reasons.append("prefix_risk_below_threshold")
            else:
                risk_p_safe = decision.get("candidate_risk_safe_probability")
                risk_threshold = self.risk_threshold
                if risk_threshold is None:
                    risk_threshold = decision.get("candidate_risk_threshold")
                if risk_p_safe is None or risk_threshold is None:
                    reasons.append("risk_score_unavailable")
                elif float(risk_p_safe) < float(risk_threshold):
                    reasons.append("risk_below_threshold")
        else:
            conf = decision.get("candidate_confidence", 0.0)
            min_step_conf = decision.get("candidate_min_exec_step_confidence", 0.0)
            if conf < self.confidence_threshold:
                reasons.append("confidence_below_threshold")
            if min_step_conf < self.min_step_confidence:
                reasons.append("step_confidence_below_threshold")
        if self.familiarity_gate is not None:
            dk = decision.get("familiarity_dk")
            if dk is None:
                # gate enabled but no distance computed (encode failure): fail safe -> VLA
                reasons.append("familiarity_unavailable")
            elif float(dk) > self.familiarity_gate.threshold:
                reasons.append("familiarity_veto")
                self.familiarity_vetoes += 1
        # Per-episode VLA-call count (not the global cumulative one): the threshold calibration
        # simulates vla_calls=0 at each episode start, so the online gate must match it or later
        # episodes would substitute the moment history is ready. Falls back to the global counter
        # only if episode stats are somehow unavailable.
        episode_vla = (
            self._current_episode_stats["vla_calls"]
            if self._current_episode_stats is not None
            else self.vla_calls
        )
        if episode_vla < self.min_vla_calls_before_substitute:
            reasons.append("min_vla_calls_not_met")
        if self.cooldown_remaining > 0:
            reasons.append("cooldown")
        if (
            self.max_consecutive_substitutions > 0
            and self.consecutive_substitutions >= self.max_consecutive_substitutions
        ):
            reasons.append("consecutive_substitution_limit")
        if self.max_substitution_fraction is not None:
            next_fraction = (self.substitutions + 1) / max(self.vla_calls + self.substitutions + 1, 1)
            if next_fraction > self.max_substitution_fraction:
                reasons.append("replacement_budget")
        safety_len = int(decision.get("candidate_accepted_prefix_len") or self.replan_steps)
        safety_len = min(max(safety_len, 1), self.replan_steps)
        if not _actions_are_safe(np.asarray(candidate["actions"][:safety_len]), np.asarray(self.prev_actions[-1])):
            reasons.append("action_safety_gate")
        return len(reasons) == 0, reasons

    def _decision_score_and_threshold(self, decision: dict) -> tuple[float | None, float | None]:
        if self.decision_metric in ("envelope", "aleatoric"):
            threshold = self.envelope_threshold if self.decision_metric == "envelope" else self.aleatoric_threshold
            return decision.get("candidate_env_score"), threshold
        if self.decision_metric == "risk":
            threshold = self.risk_threshold
            if threshold is None:
                threshold = decision.get("candidate_risk_threshold")
            return decision.get("candidate_risk_safe_probability"), threshold
        return decision.get("candidate_confidence"), self.confidence_threshold

    def _gate_score(self, candidate: dict) -> float | None:
        """Learning-free gate score over the executed prefix (lower = safer to substitute).

        envelope:  max over (steps x continuous repr dims) of |residual_norm| / sigma.
        aleatoric: mean over (steps x continuous repr dims) of the predicted normalized std.
        Both live in the normalized sin/cos repr space; the gripper dim is vetoed separately."""
        cont = self._env_cont_dims
        if self.decision_metric == "envelope":
            delta = candidate.get("pred_delta_norm")
            if delta is None or self._env_sigma is None:
                return None
            delta = np.abs(np.asarray(delta, dtype=np.float32)[: self.replan_steps][:, cont])
            return float(np.max(delta / self._env_sigma[cont]))
        std = candidate.get("pred_std_norm")
        if std is None:
            return None
        std = np.asarray(std, dtype=np.float32)[: self.replan_steps][:, cont]
        return float(np.mean(std))

    def _gripper_changes(self, candidate: dict) -> bool:
        """True if the predicted gripper open/close state flips from the last executed action
        anywhere in the executed prefix (raw 7-D space, threshold from the envelope file)."""
        pred = np.asarray(candidate["actions"], dtype=np.float32)[: self.replan_steps]
        last = np.asarray(self.prev_actions[-1], dtype=np.float32)
        idx, thr = self._grip_raw_idx, self._grip_threshold
        return bool(np.any((pred[:, idx] > thr) != (last[idx] > thr)))

    def _call_vla(self, obs, state: np.ndarray) -> np.ndarray:
        second_image, _, image, image_wrist = obs["rgb"]
        policy_input = {
            "observation/image": image,
            "observation/second_image": second_image,
            "observation/wrist_image": image_wrist,
            "observation/state": state,
            "prompt": obs["instruction"],
        }
        action_chunk = np.asarray(self.model.infer(policy_input)["actions"], dtype=np.float32)
        if len(action_chunk) < self.replan_steps:
            raise RuntimeError(
                f"Policy returned {len(action_chunk)} actions, need at least {self.replan_steps}"
            )
        return action_chunk[: self.replan_steps]

    def _log_step(self, obs, action, abs_target_pos, target_euler, gripper_state, meta) -> None:
        row = {
            "event": "step_action",
            "episode_index": self.episode_index,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "instruction": obs.get("instruction"),
            "source": meta["source"],
            "replan_index": meta["replan_index"],
            "within_chunk_index": meta["within_chunk_index"],
            "chunk_confidence": meta["chunk_confidence"],
            "chunk_decision_metric": meta["chunk_decision_metric"],
            "chunk_risk_safe_probability": meta["chunk_risk_safe_probability"],
            "chunk_prefix_safe_probability": meta["chunk_prefix_safe_probability"],
            "chunk_executed_len": meta["chunk_executed_len"],
            "vla_called_for_chunk": meta["vla_called_for_chunk"],
            "action_robot_frame": action.tolist(),
            "target_pos_world": np.asarray(abs_target_pos).tolist(),
            "target_euler": np.asarray(target_euler).tolist(),
            "gripper_state": np.asarray(gripper_state).tolist(),
            "history_len_after": len(self.prev_actions),
        }
        self._write_jsonl(self._step_f, row)

    def _replacement_fraction(self) -> float:
        return self.substitutions / max(self.vla_calls + self.substitutions, 1)

    def _finalize_episode_stats(self) -> None:
        if self._current_episode_stats is None:
            return
        stats = dict(self._current_episode_stats)
        stats["replacement_fraction"] = stats["substitutions"] / max(
            stats["vla_calls"] + stats["substitutions"], 1
        )
        stats["candidate_confidence_mean"] = _mean_or_none(stats["candidate_confidences"])
        stats["used_confidence_mean"] = _mean_or_none(stats["used_confidences"])
        self.episode_stats.append(stats)
        self._current_episode_stats = None

    def _success_by_episode(self) -> dict[int, bool]:
        """episode_index -> success, parsed from the saved video filenames
        (<idx>_success_<bool>_progress_<p>.mp4). Empty if visualization was off."""
        out: dict[int, bool] = {}
        for v in glob.glob(str(self.log_dir.parent / "*" / "videos" / "*.mp4")):
            m = re.match(r"(\d+)_success_(True|False)", os.path.basename(v))
            if m:
                out[int(m.group(1))] = m.group(2) == "True"
        return out

    def write_summary(self, metrics: dict[str, Any]) -> None:
        self._finalize_episode_stats()
        summary = {
            "policy": self.name,
            "decision_metric": self.decision_metric,
            "confidence_threshold": self.confidence_threshold,
            "min_step_confidence": self.min_step_confidence,
            "risk_threshold": self.risk_threshold,
            "max_substitution_fraction": self.max_substitution_fraction,
            "min_vla_calls_before_substitute": self.min_vla_calls_before_substitute,
            "vla_cooldown_after_substitute": self.vla_cooldown_after_substitute,
            "max_consecutive_substitutions": self.max_consecutive_substitutions,
            "total_vla_calls": self.vla_calls,
            "total_substitutions": self.substitutions,
            "total_replans": self.vla_calls + self.substitutions,
            "replacement_fraction": self._replacement_fraction(),
            "rejected_candidates": self.rejected_candidates,
            "familiarity_threshold": (
                self.familiarity_gate.threshold if self.familiarity_gate is not None else None
            ),
            "familiarity_vetoes": self.familiarity_vetoes,
            "episode_stats": self.episode_stats,
            "skip_by_outcome": skip_by_outcome(self.episode_stats, self._success_by_episode()),
            "metrics": metrics,
            "step_log": str(self.step_log_path),
            "chunk_log": str(self.chunk_log_path),
        }
        self.summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
        self._step_f.flush()
        self._chunk_f.flush()

    def close(self) -> None:
        self._step_f.close()
        self._chunk_f.close()

    @staticmethod
    def _write_jsonl(handle, row: dict) -> None:
        handle.write(json.dumps(_jsonable(row), separators=(",", ":")) + "\n")
        handle.flush()

    @property
    def name(self):
        return f"pi05_r3m_{self.decision_metric}_substitute"

    def _remember_observation(self, obs) -> None:
        second_image, _, image, image_wrist = obs["rgb"]
        self.image_history.append((image, second_image, image_wrist))

    def _accepted_prefix_len(self, prefix_probs: np.ndarray, threshold: float | None) -> int | None:
        if len(prefix_probs) == 0:
            return None
        if threshold is None:
            threshold = 0.5
        limit = min(len(prefix_probs), self.replan_steps)
        accepted = 0
        for i, prob in enumerate(prefix_probs[:limit], start=1):
            if float(prob) >= float(threshold):
                accepted = i
            else:
                break
        return accepted


def _policy_state_from_obs(obs) -> np.ndarray:
    state = obs["ee_state"]
    pos, quat, gripper_state = state[:3], state[3:7], state[-1]
    ee_euler = quaternion_to_euler(quat)
    pos = np.asarray(pos, dtype=np.float32).copy()
    pos -= np.array([0, -0.4, 0.78], dtype=np.float32)
    return np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)]).astype(np.float32)


def _actions_are_safe(actions: np.ndarray, last_action: np.ndarray) -> bool:
    if not np.all(np.isfinite(actions)):
        return False
    abs_bounds = np.array([0.75, 0.90, 0.65, 3.25, 1.75, 3.25, 1.05], dtype=np.float32)
    if np.any(np.abs(actions) > abs_bounds):
        return False
    if np.any(actions[:, 6] < -0.05) or np.any(actions[:, 6] > 1.05):
        return False
    pos_jump = np.max(np.abs(actions[:, :3] - last_action[None, :3]))
    if pos_jump > 0.18:
        return False
    return True


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def main(args: Args) -> None:
    save_dir = Path(args.save_dir)
    episode_configs = None
    if args.eval_track is not None:
        with open(Path(os.environ["VLABENCH_ROOT"]) / "configs/evaluation/tracks" / f"{args.eval_track}.json") as f:
            episode_configs = json.load(f)
        save_dir = save_dir / args.eval_track
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks.split(" ") if args.tasks else list(episode_configs.keys())
    metrics = args.metrics.split(" ")
    log_dir = save_dir / "hybrid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "hybrid_config.json").write_text(json.dumps(dataclasses.asdict(args), indent=2) + "\n")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    predictor = OnlineR3MActionPredictor(
        checkpoint_path=args.checkpoint_path,
        r3m_cache_dir=args.r3m_cache_dir,
        device=args.predictor_device,
        risk_head_path=args.risk_head_path,
    )
    envelope_gate = None
    if args.envelope_gate_path:
        envelope_gate = dict(np.load(args.envelope_gate_path, allow_pickle=True))
    familiarity_gate = None
    if args.familiarity_pool_path:
        familiarity_gate = FamiliarityGate(
            args.familiarity_pool_path,
            device=args.predictor_device,
            threshold=args.familiarity_threshold,
        )
        print(
            f"[live_eval] familiarity gate: pool={familiarity_gate.n_frames} frames "
            f"threshold={familiarity_gate.threshold:.4f} topk={familiarity_gate.topk}",
            flush=True,
        )
    policy = HybridPi0Policy(
        client,
        predictor,
        log_dir,
        replan_steps=args.replan_steps,
        confidence_threshold=args.confidence_threshold,
        min_step_confidence=args.min_step_confidence,
        decision_metric=args.decision_metric,
        risk_threshold=args.risk_threshold,
        max_substitution_fraction=args.max_substitution_fraction,
        min_vla_calls_before_substitute=args.min_vla_calls_before_substitute,
        vla_cooldown_after_substitute=args.vla_cooldown_after_substitute,
        max_consecutive_substitutions=args.max_consecutive_substitutions,
        log_full_chunks=args.log_full_chunks,
        envelope_gate=envelope_gate,
        envelope_threshold=args.envelope_threshold,
        aleatoric_threshold=args.aleatoric_threshold,
        familiarity_gate=familiarity_gate,
    )

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_configs,
        max_substeps=1,
        save_dir=str(save_dir),
        visulization=args.visulization,
        metrics=metrics,
    )
    try:
        results = evaluator.evaluate(policy)
        policy.write_summary(results)
        # Default visual report (best-effort; never fail the eval over a plotting error).
        try:
            from .visualize_run import make_report

            report = make_report(save_dir, meta={
                "task": args.tasks, "replan_steps": args.replan_steps,
                "prev_horizon": policy.prev_horizon, "pred_horizon": policy.pred_horizon,
            })
            print(f"[live_eval] report -> {report}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[live_eval] report generation skipped: {exc}", flush=True)
    finally:
        policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
