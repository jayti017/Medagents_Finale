"""
agents/supervisor.py — Supervisor Agent
========================================
Reviews the proposed diagnosis, approves or rejects it, and provides
structured feedback to all agents for memory-driven improvement.
Active at: Stage.SUPERVISOR_REVIEW
"""

import random
from typing import Dict, List, Any, Optional

from task import Action


class SupervisorAgent:
    """
    Supervisor: case-level quality control and feedback generator.

    Responsibilities:
      1. Evaluate the specialist's proposed diagnosis against full case evidence
      2. Approve or reject (with reasons)
      3. Generate structured per-agent feedback stored in memory
      4. Escalate genuinely ambiguous cases

    Pre-training: may approve incorrect diagnoses at high noise levels.
    Post-training: stricter, evidence-driven approval logic.
    """

    def __init__(self, memory=None, noise_level: float = 0.25):
        self.memory = memory
        self.noise_level = noise_level
        self.feedback_log: List[Dict] = []

    def act(self, observation: Dict[str, Any]) -> List[Action]:
        stage_name = observation.get("stage", "")
        if stage_name != "SUPERVISOR_REVIEW":
            return []

        visible = observation.get("visible_info", {})
        full_case = visible.get("full_case", {})
        proposed_dx = visible.get("proposed_diagnosis", "")
        confidence = visible.get("proposed_confidence", 0.5) or 0.5
        all_outputs = visible.get("all_stage_outputs", {})

        correct_dx = full_case.get("correct_diagnosis", "")
        is_critical = full_case.get("critical", False)
        severity = full_case.get("severity", "medium")

        actions = []

        # ── Evaluate the diagnosis ────────────────────────────────────────────
        is_correct = (proposed_dx or "").strip().lower() == correct_dx.strip().lower()

        # Pre-training noise: supervisor may miss an incorrect diagnosis
        supervisor_error = random.random() < self.noise_level
        perceived_correct = is_correct or (supervisor_error and not is_critical)

        # ── Decision ──────────────────────────────────────────────────────────
        if perceived_correct:
            action_type = "approve_diagnosis"
            message = (
                f"Diagnosis '{proposed_dx}' approved. "
                f"Evidence aligns with expected findings for case #{full_case.get('id')}."
            )
        else:
            action_type = "reject_diagnosis"
            message = (
                f"Diagnosis '{proposed_dx}' rejected. "
                f"Expected: '{correct_dx}'. "
                f"Returning to test analysis for re-evaluation."
            )

        # For critical cases, always double-check
        if is_critical and not is_correct:
            action_type = "escalate_case"
            message = (
                f"CRITICAL CASE ESCALATED. Proposed '{proposed_dx}' conflicts with "
                f"expected '{correct_dx}'. Immediate specialist re-review required."
            )

        actions.append(
            Action(
                agent_role="supervisor",
                action_type=action_type,
                content={
                    "reasoning": message,
                    "confidence": round(confidence * (0.9 if not is_correct else 1.0), 3),
                    "diagnosis": proposed_dx,
                },
            )
        )

        # ── Feedback generation ───────────────────────────────────────────────
        feedback = self._generate_feedback(
            proposed_dx=proposed_dx,
            correct_dx=correct_dx,
            is_correct=is_correct,
            confidence=confidence,
            severity=severity,
            all_outputs=all_outputs,
            full_case=full_case,
        )
        self.feedback_log.append(feedback)

        actions.append(
            Action(
                agent_role="supervisor",
                action_type="provide_feedback",
                content={
                    "feedback": feedback,
                    "reasoning": f"Structured post-case feedback generated for case #{full_case.get('id')}.",
                    "confidence": 0.90,
                },
            )
        )

        return actions

    def _generate_feedback(
        self,
        proposed_dx: str,
        correct_dx: str,
        is_correct: bool,
        confidence: float,
        severity: str,
        all_outputs: Dict,
        full_case: Dict,
    ) -> Dict[str, Any]:
        """
        Build structured feedback dict for storage in agent memory.
        """
        symptoms = full_case.get("symptoms", [])
        tests = full_case.get("tests", {})
        is_critical = full_case.get("critical", False)

        # Identify what went wrong
        if not is_correct:
            mistake_type = "wrong_diagnosis"
            if confidence > 0.80:
                mistake_type = "overconfident_wrong_diagnosis"
        elif confidence > 0.90 and is_correct:
            mistake_type = "none"
        else:
            mistake_type = "none"

        corrective_rule = ""
        if not is_correct:
            key_test = list(tests.items())[0] if tests else ("test", "result")
            corrective_rule = (
                f"For symptoms {symptoms}, key confirmatory test is "
                f"'{key_test[0]}' showing '{key_test[1]}'. "
                f"Correct diagnosis: '{correct_dx}'."
            )

        return {
            "case_id": full_case.get("id"),
            "proposed": proposed_dx,
            "expected": correct_dx,
            "is_correct": is_correct,
            "confidence": confidence,
            "severity": severity,
            "is_critical": is_critical,
            "mistake_type": mistake_type,
            "corrective_rule": corrective_rule,
            "gp_assessment": all_outputs.get("INITIAL_ASSESSMENT", []),
            "specialist_decision": all_outputs.get("DIAGNOSIS_DECISION", []),
            "general": {
                "predicted": proposed_dx,
                "expected": correct_dx,
                "corrective_rule": corrective_rule,
            },
        }

    def get_latest_feedback(self) -> Optional[Dict]:
        """Return the most recent feedback entry."""
        return self.feedback_log[-1] if self.feedback_log else None
