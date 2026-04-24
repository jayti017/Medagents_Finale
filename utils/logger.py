"""
utils/logger.py — MedAgents-X Structured Logger
=================================================
Logs every episode's metrics in a structured format for graph generation.

Key improvement (Guideline Point 15):
  Each reward component is logged as its OWN flat column, not buried in a dict.
  This lets you track individual reward functions separately during training,
  which is how you detect reward hacking and monitor what the model is optimising.

Logged data supports:
  1. Reward vs Episodes          (line graph)
  2. Confidence vs Steps         (line graph, within a case)
  3. Before vs After             (compare pre/post training runs)
  4. Accuracy over time          (line graph)
  5. Per-component reward curves (correct_diagnosis, evidence, confidence_calib...)
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# ─── Individual reward columns (Guideline Point 15) ───────────────────────────
REWARD_COMPONENTS = [
    "correct_diagnosis",
    "evidence_based_reasoning",
    "confidence_calibration",
    "efficiency",
    "critical_disease_detected",
]


class EpisodeLogger:
    """
    Records per-episode and per-step metrics.
    Individual reward components are logged as flat fields (not nested dicts)
    so training dashboards can track each signal independently.
    """

    def __init__(self, run_name: str = "default", log_dir: str = LOG_DIR):
        self.run_name = run_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.episode_log_path = os.path.join(log_dir, f"{run_name}_{ts}_episodes.jsonl")
        self.step_log_path    = os.path.join(log_dir, f"{run_name}_{ts}_steps.jsonl")
        self.summary_path     = os.path.join(log_dir, f"{run_name}_{ts}_summary.json")

        self._episodes: List[Dict] = []
        self._steps: List[Dict] = []
        self._episode_num = 0

        print(f"[Logger] Run '{run_name}' logging to: {log_dir}")

    # ── Episode-level logging ─────────────────────────────────────────────────

    def log_episode(
        self,
        case_id: int,
        total_reward: float,
        is_correct: bool,
        predicted_diagnosis: str,
        correct_diagnosis: str,
        confidence: float,
        steps_taken: int,
        severity: str,
        reward_breakdown: Dict[str, float],
        phase: str = "pre_training",
    ) -> None:
        """
        Log one complete episode. Each reward component is a flat top-level field
        so training dashboards can plot them as separate columns (Guideline Point 15).
        """
        self._episode_num += 1

        record = {
            "episode":    self._episode_num,
            "case_id":    case_id,
            "phase":      phase,
            "timestamp":  datetime.utcnow().isoformat(),
            "total_reward":   round(float(total_reward), 4),
            "is_correct":     is_correct,
            "predicted":      predicted_diagnosis,
            "correct":        correct_diagnosis,
            "confidence":     round(float(confidence), 4),
            "steps_taken":    steps_taken,
            "severity":       severity,
        }

        # ── Flat reward component columns (Guideline Point 15) ────────────────
        for comp in REWARD_COMPONENTS:
            record[f"reward_{comp}"] = round(float(reward_breakdown.get(comp, 0.0)), 4)

        self._episodes.append(record)

        with open(self.episode_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Step-level logging ────────────────────────────────────────────────────

    def log_step(
        self,
        episode: int,
        step: int,
        stage: str,
        agent: str,
        action_type: str,
        step_reward: float,
        confidence: Optional[float],
        cumulative_reward: float,
    ) -> None:
        """Log one environment step for intra-episode confidence/reward traces."""
        record = {
            "episode":          episode,
            "step":             step,
            "stage":            stage,
            "agent":            agent,
            "action_type":      action_type,
            "step_reward":      round(float(step_reward), 4),
            "confidence":       round(float(confidence), 4) if confidence is not None else None,
            "cumulative_reward": round(float(cumulative_reward), 4),
            "timestamp":        datetime.utcnow().isoformat(),
        }
        self._steps.append(record)

        with open(self.step_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────

    def save_summary(self) -> Dict[str, Any]:
        """
        Compute and save run summary.
        Includes per-component reward averages as separate keys.
        """
        if not self._episodes:
            return {}

        total   = len(self._episodes)
        correct = sum(1 for e in self._episodes if e["is_correct"])
        rewards = [e["total_reward"] for e in self._episodes]
        confs   = [e["confidence"]   for e in self._episodes]

        def moving_avg(lst, w=5):
            return [
                round(sum(lst[max(0, i-w):i+1]) / len(lst[max(0, i-w):i+1]), 4)
                for i in range(len(lst))
            ]

        # ── Per-component averages (Guideline Point 15) ───────────────────────
        reward_components = {}
        for comp in REWARD_COMPONENTS:
            key = f"reward_{comp}"
            vals = [e.get(key, 0.0) for e in self._episodes]
            reward_components[comp] = round(sum(vals) / total, 4)

        summary = {
            "run_name":              self.run_name,
            "total_episodes":        total,
            "accuracy":              round(correct / total, 4),
            "mean_reward":           round(sum(rewards) / total, 4),
            "mean_confidence":       round(sum(confs) / total, 4),
            "reward_curve":          rewards,
            "reward_curve_smoothed": moving_avg(rewards),
            "confidence_curve":      confs,
            "accuracy_curve": [
                round(sum(1 for e in self._episodes[:i+1] if e["is_correct"]) / (i+1), 4)
                for i in range(total)
            ],
            "reward_components":     reward_components,
            "by_phase":              self._phase_breakdown(),
            "generated_at":          datetime.utcnow().isoformat(),
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Logger] Summary saved → {self.summary_path}")
        return summary

    def _phase_breakdown(self) -> Dict[str, Any]:
        phases = {}
        for ep in self._episodes:
            p = ep["phase"]
            if p not in phases:
                phases[p] = {"count": 0, "correct": 0, "rewards": []}
            phases[p]["count"]   += 1
            phases[p]["correct"] += int(ep["is_correct"])
            phases[p]["rewards"].append(ep["total_reward"])

        result = {}
        for p, data in phases.items():
            n = data["count"]
            result[p] = {
                "episodes":    n,
                "accuracy":    round(data["correct"] / n, 4) if n else 0,
                "mean_reward": round(sum(data["rewards"]) / n, 4) if n else 0,
            }
        return result

    @property
    def episode_count(self) -> int:
        return self._episode_num

    def get_rewards(self) -> List[float]:
        return [e["total_reward"] for e in self._episodes]

    def get_confidences(self) -> List[float]:
        return [e["confidence"] for e in self._episodes]

    def get_accuracy_over_time(self) -> List[float]:
        correct_so_far = 0
        acc = []
        for ep in self._episodes:
            correct_so_far += int(ep["is_correct"])
            acc.append(round(correct_so_far / (len(acc) + 1), 4))
        return acc
