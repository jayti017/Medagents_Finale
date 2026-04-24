"""
environment.py — MedAgents-X OpenEnv Environment
==================================================
OpenEnv-compliant environment for multi-agent clinical decision-making.

Follows the standard loop:
    state = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

Key design features:
  - Partial observability: agents see only what's appropriate per stage
  - Delayed test results: tests only revealed at Stage 2
  - Dynamic state: state changes as agents interact
  - Multi-agent: different agent roles active at different stages
  - Reward signal: dense (per-step) + sparse (end-of-episode)
"""

import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional

from task import TaskPipeline, Stage, Action
from reward import compute_reward, compute_step_reward
from memory import MemorySystem


# ─── Load Dataset ─────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")

def _load_dataset() -> List[Dict]:
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


# ─── State Definition ─────────────────────────────────────────────────────────

class MedState:
    """
    Represents the full environment state at a given timestep.

    Attributes:
        case_id:        ID of the active patient case
        stage:          Current pipeline stage name
        observation:    Partial observation available to agents at this step
        step_count:     Number of steps taken so far in this episode
        episode_done:   Whether the episode has ended
        cumulative_reward: Sum of rewards collected so far
    """

    def __init__(self, case_id: int, stage: str, observation: Dict, step_count: int = 0):
        self.case_id = case_id
        self.stage = stage
        self.observation = observation
        self.step_count = step_count
        self.episode_done = False
        self.cumulative_reward = 0.0

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "observation": self.observation,
            "step_count": self.step_count,
            "episode_done": self.episode_done,
            "cumulative_reward": self.cumulative_reward,
        }


# ─── Environment ──────────────────────────────────────────────────────────────

class MedAgentsEnv:
    """
    MedAgents-X: Multi-Agent Clinical Decision RL Environment.

    OpenEnv interface:
        reset()         → MedState
        step(action)    → (next_state: MedState, reward: float, done: bool, info: dict)

    Also exposes:
        render()        → print current state
        get_valid_actions() → list of valid actions for current stage
    """

    def __init__(
        self,
        shuffle: bool = True,
        max_steps_per_episode: int = 20,
        memory_system: Optional[MemorySystem] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = _load_dataset()
        self.shuffle = shuffle
        self.max_steps = max_steps_per_episode
        self.memory = memory_system or MemorySystem()

        if seed is not None:
            random.seed(seed)

        # Episode state
        self._case: Optional[Dict] = None
        self._pipeline: Optional[TaskPipeline] = None
        self._state: Optional[MedState] = None
        self._episode_idx: int = 0
        self._case_queue: List[Dict] = []
        self._agent_outputs: Dict[str, Any] = {}
        self._step_rewards: List[float] = []

        self._setup_queue()

    def _setup_queue(self) -> None:
        """Prepare shuffled or ordered queue of cases."""
        self._case_queue = self.dataset.copy()
        if self.shuffle:
            random.shuffle(self._case_queue)
        self._episode_idx = 0

    # ── OpenEnv: reset ────────────────────────────────────────────────────────

    def reset(self) -> MedState:
        """
        Start a new episode with the next patient case.
        Returns the initial partial state (symptoms only visible).
        """
        if self._episode_idx >= len(self._case_queue):
            self._setup_queue()  # cycle through dataset

        self._case = self._case_queue[self._episode_idx]
        self._episode_idx += 1
        self._pipeline = TaskPipeline(case_id=self._case["id"])
        self._agent_outputs = {role: {} for role in ["gp","radiologist","pathologist","specialist","supervisor","oversight"]}
        self._step_rewards = []

        obs = self._pipeline.get_current_observation(self._case)
        self._state = MedState(
            case_id=self._case["id"],
            stage=obs["stage"],
            observation=obs,
            step_count=0,
        )
        return self._state

    # ── OpenEnv: step ─────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[MedState, float, bool, Dict[str, Any]]:
        """
        Execute one agent action and return the environment response.

        Args:
            action: An Action object with agent_role, action_type, content

        Returns:
            next_state:  Updated MedState
            reward:      Float reward signal (dense per-step)
            done:        Whether the episode has ended
            info:        Additional metadata (for logging/debugging)
        """
        assert self._pipeline is not None, "Call reset() before step()"
        assert self._state is not None, "Call reset() before step()"

        prev_stage = self._pipeline.current_stage

        # Execute action in pipeline
        result = self._pipeline.execute_action(action)

        # Store agent output for reward/memory later
        if action.content:
            self._agent_outputs[action.agent_role] = action.content

        # Determine progress
        progress_made = result["transition"]
        action_valid = result["success"]

        # Compute dense step reward
        step_reward = compute_step_reward(
            stage=prev_stage.name,
            action_valid=action_valid,
            progress_made=progress_made,
        )
        self._step_rewards.append(step_reward)

        # Increment step count
        self._state.step_count += 1
        self._state.cumulative_reward += step_reward

        # Build next observation
        obs = self._pipeline.get_current_observation(self._case)
        self._state.stage = obs["stage"]
        self._state.observation = obs

        # Check episode termination
        done = self._pipeline.is_complete or self._state.step_count >= self.max_steps
        self._state.episode_done = done

        info = {
            "action_result": result,
            "step_reward": step_reward,
            "current_stage": self._pipeline.current_stage.name,
            "valid_actions": obs.get("valid_actions", []),
            "active_agents": obs.get("active_agents", []),
            "step_count": self._state.step_count,
        }

        # On episode end: compute terminal reward and record memory
        if done:
            terminal_reward, memory_info = self._end_episode()
            self._state.cumulative_reward += terminal_reward
            info["terminal_reward"] = terminal_reward
            info["final_reward"] = self._state.cumulative_reward
            info["memory_recorded"] = memory_info
            step_reward += terminal_reward  # include terminal in returned reward

        return self._state, step_reward, done, info

    # ── End-of-Episode Logic ──────────────────────────────────────────────────

    def _end_episode(self) -> Tuple[float, Dict]:
        """Compute terminal reward and record feedback to memory."""
        summary = self._pipeline.get_summary()
        case = self._case

        predicted = summary.get("final_diagnosis") or ""
        correct = case["correct_diagnosis"]
        confidence = summary.get("final_confidence") or 0.5
        tests_ordered = summary.get("tests_ordered", [])
        reasoning_steps = summary.get("reasoning_steps", [])
        steps_taken = summary.get("total_steps", 0)

        reward_result = compute_reward(
            predicted_diagnosis=predicted,
            correct_diagnosis=correct,
            confidence=confidence,
            tests_ordered=tests_ordered,
            valid_tests=case["tests"],
            reasoning_steps=reasoning_steps,
            steps_taken=steps_taken,
            severity=case.get("severity", "medium"),
            is_critical=case.get("critical", False),
        )

        terminal_reward = reward_result["total_reward"]
        is_correct = reward_result["is_correct"]

        # Build supervisor feedback for memory
        supervisor_feedback = {
            "general": {
                "correct": is_correct,
                "predicted": predicted,
                "expected": correct,
                "reward_breakdown": reward_result["breakdown"],
                "penalties": reward_result["penalties"],
            },
            "mistake_type": "wrong_diagnosis" if not is_correct else "none",
            "corrective_rule": (
                f"For symptoms {case['symptoms']}, correct diagnosis is '{correct}'. "
                f"Key test: {list(case['tests'].items())[0]}."
                if not is_correct else ""
            ),
        }

        # Record to memory system
        self.memory.record_episode(
            case_id=case["id"],
            agent_outputs=self._agent_outputs,
            supervisor_feedback=supervisor_feedback,
            total_reward=terminal_reward,
            correct=is_correct,
        )

        return terminal_reward, {
            "is_correct": is_correct,
            "reward_breakdown": reward_result["breakdown"],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_valid_actions(self) -> List[str]:
        """Return valid action types for the current stage."""
        if self._pipeline is None:
            return []
        from task import STAGE_VALID_ACTIONS
        return STAGE_VALID_ACTIONS.get(self._pipeline.current_stage, [])

    def get_active_agents(self) -> List[str]:
        """Return agent roles active at the current stage."""
        if self._pipeline is None:
            return []
        from task import STAGE_AGENTS
        return STAGE_AGENTS.get(self._pipeline.current_stage, [])

    def render(self) -> None:
        """Print a human-readable summary of the current state."""
        if self._state is None:
            print("[MedAgentsEnv] No active episode. Call reset() first.")
            return
        print("\n" + "═" * 60)
        print(f"  MedAgents-X | Case #{self._state.case_id} | Step {self._state.step_count}")
        print("═" * 60)
        print(f"  Stage         : {self._state.stage}")
        print(f"  Active Agents : {self.get_active_agents()}")
        print(f"  Valid Actions : {self.get_valid_actions()}")

        vis = self._state.observation.get("visible_info", {})
        if "symptoms" in vis:
            print(f"  Symptoms      : {vis['symptoms']}")
        if "possible_diseases" in vis:
            print(f"  Differentials : {vis['possible_diseases']}")
        if "test_results" in vis:
            print(f"  Test Results  : {vis['test_results']}")

        print(f"  Cumulative ΣR : {self._state.cumulative_reward:.4f}")
        print(f"  Done          : {self._state.episode_done}")
        print("═" * 60 + "\n")

    @property
    def n_cases(self) -> int:
        return len(self.dataset)
