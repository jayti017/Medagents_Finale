"""
agents/radiologist.py — Radiologist Agent
==========================================
Interprets imaging test results (X-ray, MRI, CT, etc.).
Active at: Stage.TEST_ANALYSIS
"""

import random
from typing import Dict, List, Any

from task import Action


# ─── Imaging test → finding patterns ─────────────────────────────────────────
IMAGING_INTERPRETATIONS = {
    "mri": {
        "tumor detected":  ("malignant lesion identified on MRI", "brain tumor", 0.88),
        "infection":       ("spinal/cerebral infection pattern on MRI", "spine infection", 0.82),
        "normal":          ("no abnormality detected on MRI", None, 0.75),
    },
    "xray": {
        "bronchial thickening": ("airway wall thickening on CXR consistent with bronchitis", "chronic bronchitis", 0.78),
        "joint damage":         ("periarticular erosions and joint space narrowing on XR", "arthritis", 0.80),
        "normal":               ("no acute cardiopulmonary process on CXR", None, 0.72),
    },
    "ecg": {
        "abnormal": ("ST-segment elevation / ischemic changes on ECG", "heart attack", 0.92),
        "normal":   ("normal sinus rhythm, no acute changes", None, 0.70),
    },
}

# Tests this agent handles
IMAGING_TESTS = {"mri", "xray", "ecg", "ct", "ultrasound", "pet_scan"}


class RadiologistAgent:
    """
    Radiologist: interprets imaging-based diagnostic tests.

    Pre-training: moderate accuracy with partial result interpretation.
    Post-training: improved sensitivity for subtle findings.
    """

    def __init__(self, memory=None, noise_level: float = 0.30):
        self.memory = memory
        self.noise_level = noise_level

    def act(self, observation: Dict[str, Any]) -> List[Action]:
        """Interpret imaging results present in the observation."""
        stage_name = observation.get("stage", "")
        if stage_name != "TEST_ANALYSIS":
            return []

        visible = observation.get("visible_info", {})
        test_results = visible.get("test_results", {})
        symptoms = visible.get("symptoms", [])

        actions = []

        for test_name, result_text in test_results.items():
            if test_name.lower() in IMAGING_TESTS:
                action = self._interpret_imaging(test_name, result_text, symptoms)
                if action:
                    actions.append(action)

        # If no imaging tests found, still produce an action to avoid stalling
        if not actions:
            actions.append(
                Action(
                    agent_role="radiologist",
                    action_type="interpret_imaging",
                    content={
                        "test": "general_imaging_review",
                        "finding": "No imaging studies ordered or available for this case.",
                        "suggested_diagnosis": None,
                        "confidence": 0.5,
                        "reasoning": "Deferring to pathology for non-imaging workup.",
                    },
                )
            )

        return actions

    def _interpret_imaging(self, test_name: str, result_text: str, symptoms: List[str]) -> Action:
        """
        Interpret a single imaging test result and produce an action.
        """
        test_lower = test_name.lower()
        result_lower = result_text.lower()

        # Look up interpretation table
        interpretation_map = IMAGING_INTERPRETATIONS.get(test_lower, {})
        matched_finding = None
        matched_dx = None
        base_conf = 0.65

        for key, (finding, dx, conf) in interpretation_map.items():
            if key.lower() in result_lower:
                matched_finding = finding
                matched_dx = dx
                base_conf = conf
                break

        # If no match, produce a generic finding
        if not matched_finding:
            matched_finding = f"{test_name} result: {result_text} — requires clinical correlation"
            matched_dx = None
            base_conf = 0.55

        # Apply pre-training noise
        if random.random() < self.noise_level:
            base_conf = max(0.3, base_conf - random.uniform(0.1, 0.25))
            matched_dx = None  # fail to identify the correct condition

        # Check memory for past similar cases
        memory_note = ""
        if self.memory:
            past = self.memory.get_relevant_memory(symptoms, top_k=1)
            if past:
                memory_note = f" (Memory context: {past[0]['feedback'].get('predicted', '')})"

        reasoning = (
            f"Imaging interpretation for {test_name}: {matched_finding}. "
            f"Result text: '{result_text}'.{memory_note}"
        )

        # Flag abnormal findings
        action_type = "interpret_imaging"
        if matched_dx is not None:
            action_type = "flag_abnormal"

        return Action(
            agent_role="radiologist",
            action_type=action_type,
            content={
                "test": test_name,
                "finding": matched_finding,
                "suggested_diagnosis": matched_dx,
                "confidence": round(base_conf + random.uniform(-0.05, 0.05), 3),
                "reasoning": reasoning,
            },
        )
