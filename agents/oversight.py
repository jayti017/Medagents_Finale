"""
agents/oversight.py — Oversight AI Agent
==========================================
Cross-case pattern detection and system-level feedback.
Active at: Stage.OVERSIGHT_FEEDBACK
"""

from typing import Dict, List, Any
from task import Action


class OversightAgent:
    """
    Oversight AI: analyzes patterns across multiple cases to detect systemic
    blindspots and push corrective rules to all agent memories.

    This agent operates at the meta-level — it does not diagnose, it monitors
    the diagnostic system itself.
    """

    def __init__(self, memory=None):
        self.memory = memory
        self.case_history: List[Dict] = []

    def act(self, observation: Dict[str, Any]) -> List[Action]:
        stage_name = observation.get("stage", "")
        if stage_name != "OVERSIGHT_FEEDBACK":
            return []

        visible = observation.get("visible_info", {})
        case_id = visible.get("case_id")
        final_dx = visible.get("final_diagnosis", "")
        approved = visible.get("approved", False)
        action_log = visible.get("action_log", [])

        self.case_history.append({
            "case_id": case_id,
            "final_diagnosis": final_dx,
            "approved": approved,
            "n_actions": len(action_log),
        })

        actions = []

        # ── Log the case pattern ──────────────────────────────────────────────
        pattern_note = (
            f"Case #{case_id}: final_dx='{final_dx}', "
            f"approved={approved}, steps={len(action_log)}."
        )
        actions.append(
            Action(
                agent_role="oversight",
                action_type="log_pattern",
                content={
                    "pattern": pattern_note,
                    "reasoning": "Recording case outcome for cross-case trend analysis.",
                    "confidence": 0.95,
                },
            )
        )

        # ── Derive and push rules if enough history ──────────────────────────
        if len(self.case_history) >= 3:
            rules = self._derive_rules()
            for rule in rules:
                actions.append(
                    Action(
                        agent_role="oversight",
                        action_type="update_agent_rules",
                        content={
                            "rule": rule,
                            "reasoning": "Cross-case pattern identified. Pushing corrective rule to agent memory.",
                            "confidence": 0.80,
                        },
                    )
                )
                # Push rule to memory for all agents
                if self.memory:
                    for role in ["gp", "specialist", "pathologist", "radiologist"]:
                        mem = self.memory.get_agent_memory(role)
                        mem.add_learned_rule(rule)

        # ── Periodic summary report ───────────────────────────────────────────
        if len(self.case_history) % 5 == 0:
            report = self._generate_report()
            actions.append(
                Action(
                    agent_role="oversight",
                    action_type="generate_report",
                    content={
                        "report": report,
                        "reasoning": f"Oversight report after {len(self.case_history)} cases.",
                        "confidence": 0.88,
                    },
                )
            )

        return actions

    def _derive_rules(self) -> List[str]:
        """
        Derive generalizable rules from recent case outcomes.
        Returns up to 2 rules based on approval rate and patterns.
        """
        rules = []
        recent = self.case_history[-5:]
        approval_rate = sum(1 for c in recent if c["approved"]) / len(recent)

        if approval_rate < 0.5:
            rules.append(
                "Low approval rate detected. Agents should increase evidence-gathering "
                "before proposing diagnosis. Prioritize confirmatory tests."
            )

        avg_steps = sum(c["n_actions"] for c in recent) / len(recent)
        if avg_steps > 12:
            rules.append(
                "High average step count. Agents should streamline test ordering "
                "and avoid redundant actions to improve efficiency."
            )

        return rules

    def _generate_report(self) -> Dict[str, Any]:
        """Generate an oversight summary report."""
        total = len(self.case_history)
        approved = sum(1 for c in self.case_history if c["approved"])
        avg_steps = sum(c["n_actions"] for c in self.case_history) / max(total, 1)

        patterns = {}
        if self.memory:
            patterns = self.memory.get_cross_case_patterns()

        return {
            "total_cases": total,
            "approval_rate": round(approved / max(total, 1), 3),
            "avg_steps_per_case": round(avg_steps, 2),
            "agent_patterns": patterns,
        }
