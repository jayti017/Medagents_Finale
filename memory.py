"""
memory.py — MedAgents-X Feedback + Memory System
==================================================
Simulates agent self-improvement through structured feedback storage.

Architecture:
  - Each agent has its own memory namespace
  - Supervisor writes structured feedback after each case
  - Agents read relevant memory before acting (simulating learning)
  - Oversight AI aggregates cross-case patterns

Memory is stored in a JSON file for persistence across runs.
This system is designed to bridge pre-training behavior and post-training improvement.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory_store.json")

# Agent roles (must match agent implementations)
AGENT_ROLES = ["gp", "radiologist", "pathologist", "specialist", "supervisor", "oversight"]


class AgentMemory:
    """
    Persistent memory store for a single agent.
    Stores feedback entries, mistake patterns, and performance history.
    """

    def __init__(self, role: str, store: Dict):
        self.role = role
        self.store = store  # reference to shared store dict

        # Initialize namespace if missing
        if role not in self.store:
            self.store[role] = {
                "feedback_log": [],
                "mistake_patterns": defaultdict(int),
                "performance_history": [],
                "learned_rules": [],
            }

    def add_feedback(self, case_id: int, feedback: Dict[str, Any]) -> None:
        """Store supervisor feedback for a specific case."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "case_id": case_id,
            "feedback": feedback,
        }
        self.store[self.role]["feedback_log"].append(entry)

    def record_mistake(self, mistake_type: str) -> None:
        """Track the type and frequency of mistakes."""
        mistakes = self.store[self.role]["mistake_patterns"]
        if mistake_type not in mistakes:
            mistakes[mistake_type] = 0
        mistakes[mistake_type] += 1

    def record_performance(self, case_id: int, reward: float, confidence: float, correct: bool) -> None:
        """Log per-episode performance for graph generation."""
        self.store[self.role]["performance_history"].append({
            "case_id": case_id,
            "reward": reward,
            "confidence": confidence,
            "correct": correct,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def add_learned_rule(self, rule: str) -> None:
        """
        Store a generalizable rule derived from past mistakes.
        These are injected into agent prompts to simulate pre-training improvement.
        """
        rules = self.store[self.role]["learned_rules"]
        if rule not in rules:
            rules.append(rule)

    def get_relevant_memory(self, symptoms: List[str], top_k: int = 3) -> List[Dict]:
        """
        Retrieve top_k most relevant past feedback entries for given symptoms.
        Relevance = number of overlapping symptom keywords in feedback text.
        """
        symptom_set = set(s.lower() for s in symptoms)
        scored = []

        for entry in self.store[self.role]["feedback_log"]:
            text = json.dumps(entry["feedback"]).lower()
            overlap = sum(1 for s in symptom_set if s in text)
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def get_learned_rules(self) -> List[str]:
        """Return all stored learned rules for this agent."""
        return self.store[self.role].get("learned_rules", [])

    def get_mistake_summary(self) -> Dict[str, int]:
        """Return a summary of mistake patterns."""
        return dict(self.store[self.role].get("mistake_patterns", {}))

    def get_performance_history(self) -> List[Dict]:
        """Return full performance history for this agent."""
        return self.store[self.role].get("performance_history", [])


class MemorySystem:
    """
    Centralized memory manager for the entire multi-agent system.
    Handles persistence (load/save) and provides per-agent access.
    """

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.store = self._load()
        # Ensure all default lists use plain dicts (not defaultdict) after loading
        for role in AGENT_ROLES:
            if role not in self.store:
                self.store[role] = {
                    "feedback_log": [],
                    "mistake_patterns": {},
                    "performance_history": [],
                    "learned_rules": [],
                }

    def _load(self) -> Dict:
        """Load memory from disk, or initialize empty if not found."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        """Persist memory to disk."""
        with open(self.filepath, "w") as f:
            json.dump(self.store, f, indent=2)

    def get_agent_memory(self, role: str) -> AgentMemory:
        """Return the AgentMemory object for a given role."""
        if role not in AGENT_ROLES:
            raise ValueError(f"Unknown agent role: {role}. Valid: {AGENT_ROLES}")
        return AgentMemory(role=role, store=self.store)

    def record_episode(
        self,
        case_id: int,
        agent_outputs: Dict[str, Any],
        supervisor_feedback: Dict[str, Any],
        total_reward: float,
        correct: bool,
    ) -> None:
        """
        After each episode, record feedback for all agents and extract patterns.

        Args:
            case_id:             Dataset case ID
            agent_outputs:       Dict mapping role → output for this case
            supervisor_feedback: Structured feedback from supervisor agent
            total_reward:        Final episode reward
            correct:             Whether final diagnosis was correct
        """
        for role in AGENT_ROLES:
            mem = self.get_agent_memory(role)

            # Record performance
            confidence = agent_outputs.get(role, {}).get("confidence", 0.5)
            mem.record_performance(
                case_id=case_id,
                reward=total_reward,
                confidence=confidence,
                correct=correct,
            )

            # Store supervisor feedback
            role_feedback = supervisor_feedback.get(role, supervisor_feedback.get("general", {}))
            mem.add_feedback(case_id=case_id, feedback=role_feedback)

            # Extract and store learned rules from feedback
            if not correct:
                mistake_type = supervisor_feedback.get("mistake_type", "unknown_error")
                mem.record_mistake(mistake_type)
                rule = supervisor_feedback.get("corrective_rule", "")
                if rule:
                    mem.add_learned_rule(rule)

        self.save()

    def get_cross_case_patterns(self) -> Dict[str, Any]:
        """
        Oversight AI view: aggregate patterns across all agents and cases.
        Returns stats useful for detecting systemic biases or blindspots.
        """
        patterns = {}
        for role in AGENT_ROLES:
            mem = self.get_agent_memory(role)
            history = mem.get_performance_history()
            if not history:
                patterns[role] = {"cases": 0, "accuracy": 0.0, "avg_reward": 0.0}
                continue

            total = len(history)
            correct_count = sum(1 for h in history if h["correct"])
            avg_reward = sum(h["reward"] for h in history) / total
            patterns[role] = {
                "cases": total,
                "accuracy": round(correct_count / total, 4),
                "avg_reward": round(avg_reward, 4),
                "mistake_patterns": mem.get_mistake_summary(),
            }
        return patterns

    def get_all_rewards(self) -> List[float]:
        """
        Aggregate reward history across all agents (for global reward curve).
        Uses supervisor's performance log as the canonical episode record.
        """
        sup_mem = self.get_agent_memory("supervisor")
        return [h["reward"] for h in sup_mem.get_performance_history()]

    def clear(self) -> None:
        """Reset all memory (for fresh training runs)."""
        self.store = {}
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
