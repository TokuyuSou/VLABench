from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
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
    max_substitution_fraction: float = 0.22
    min_vla_calls_before_substitute: int = 3
    vla_cooldown_after_substitute: int = 1
    checkpoint_path: str = (
        "research/r3m_action_predictor/outputs/"
        "add_condiment_r3m18_residual_transformer_200/best_model.pt"
    )
    risk_head_path: str | None = None
    decision_metric: str = "confidence"
    risk_threshold: float | None = None
    r3m_cache_dir: str = "research/r3m_action_predictor/cache"
    predictor_device: str = "cpu"
    log_full_chunks: bool = True


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
        max_substitution_fraction: float,
        min_vla_calls_before_substitute: int,
        vla_cooldown_after_substitute: int,
        log_full_chunks: bool,
    ):
        self.model = client
        self.predictor = predictor
        self.replan_steps = replan_steps
        self.confidence_threshold = confidence_threshold
        self.min_step_confidence = min_step_confidence
        self.decision_metric = decision_metric
        self.risk_threshold = risk_threshold
        self.max_substitution_fraction = max_substitution_fraction
        self.min_vla_calls_before_substitute = min_vla_calls_before_substitute
        self.vla_cooldown_after_substitute = vla_cooldown_after_substitute
        self.log_full_chunks = log_full_chunks

        self.action_plan = collections.deque()
        self.action_meta = collections.deque()
        self.prev_actions = collections.deque(maxlen=8)

        log_dir.mkdir(parents=True, exist_ok=True)
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
        self.episode_stats: list[dict[str, Any]] = []
        self._current_episode_stats: dict[str, Any] | None = None

    def reset(self):
        self._finalize_episode_stats()
        self.episode_index += 1
        self.episode_step = 0
        self.replan_index = 0
        self.cooldown_remaining = 0
        self.action_plan.clear()
        self.action_meta.clear()
        self.prev_actions.clear()
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
            "history_ready": len(self.prev_actions) >= 8,
            "threshold": self.confidence_threshold,
            "min_step_confidence": self.min_step_confidence,
            "decision_metric": self.decision_metric,
            "risk_threshold": self.risk_threshold,
            "max_substitution_fraction": self.max_substitution_fraction,
            "cooldown_remaining": self.cooldown_remaining,
        }

        if len(self.prev_actions) >= 8:
            second_image, _, image, image_wrist = obs["rgb"]
            candidate = self.predictor.predict_from_observation(
                image=image,
                second_image=second_image,
                wrist_image=image_wrist,
                state=state,
                prev_actions=np.asarray(self.prev_actions, dtype=np.float32),
            )
            conf = float(candidate.get("confidence", 0.0))
            conf_per_step = np.asarray(candidate.get("confidence_per_step", []), dtype=np.float32)
            min_exec_conf = float(np.min(conf_per_step[: self.replan_steps])) if len(conf_per_step) else 0.0
            risk_p_safe = candidate.get("risk_safe_probability")
            risk_threshold = self.risk_threshold
            if risk_threshold is None:
                risk_threshold = candidate.get("risk_threshold")
            decision.update(
                {
                    "candidate_confidence": conf,
                    "candidate_min_exec_step_confidence": min_exec_conf,
                    "candidate_risk_safe_probability": risk_p_safe,
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

        use_substitute, reject_reasons = self._should_substitute(decision, candidate)
        decision["use_substitute"] = use_substitute
        decision["reject_reasons"] = reject_reasons
        score, threshold = self._decision_score_and_threshold(decision)
        decision["decision_score"] = score
        decision["decision_threshold"] = threshold

        if use_substitute:
            actions = np.asarray(candidate["actions"], dtype=np.float32)[: self.replan_steps]
            self.substitutions += 1
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
                    "vla_called_for_chunk": vla_called,
                }
            )
        self.replan_index += 1

    def _should_substitute(self, decision: dict, candidate: dict | None) -> tuple[bool, list[str]]:
        reasons = []
        if candidate is None:
            reasons.append("no_candidate_or_history")
            return False, reasons
        if self.decision_metric == "risk":
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
        if self.vla_calls < self.min_vla_calls_before_substitute:
            reasons.append("min_vla_calls_not_met")
        if self.cooldown_remaining > 0:
            reasons.append("cooldown")
        next_fraction = (self.substitutions + 1) / max(self.vla_calls + self.substitutions + 1, 1)
        if next_fraction > self.max_substitution_fraction:
            reasons.append("replacement_budget")
        if not _actions_are_safe(np.asarray(candidate["actions"][: self.replan_steps]), np.asarray(self.prev_actions[-1])):
            reasons.append("action_safety_gate")
        return len(reasons) == 0, reasons

    def _decision_score_and_threshold(self, decision: dict) -> tuple[float | None, float | None]:
        if self.decision_metric == "risk":
            threshold = self.risk_threshold
            if threshold is None:
                threshold = decision.get("candidate_risk_threshold")
            return decision.get("candidate_risk_safe_probability"), threshold
        return decision.get("candidate_confidence"), self.confidence_threshold

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

    def write_summary(self, metrics: dict[str, Any]) -> None:
        self._finalize_episode_stats()
        summary = {
            "policy": self.name,
            "decision_metric": self.decision_metric,
            "confidence_threshold": self.confidence_threshold,
            "min_step_confidence": self.min_step_confidence,
            "risk_threshold": self.risk_threshold,
            "max_substitution_fraction": self.max_substitution_fraction,
            "total_vla_calls": self.vla_calls,
            "total_substitutions": self.substitutions,
            "total_replans": self.vla_calls + self.substitutions,
            "replacement_fraction": self._replacement_fraction(),
            "rejected_candidates": self.rejected_candidates,
            "episode_stats": self.episode_stats,
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
        if self.decision_metric == "risk":
            return "pi05_r3m_risk_substitute"
        return "pi05_r3m_conf_substitute"


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
        log_full_chunks=args.log_full_chunks,
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
    finally:
        policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
