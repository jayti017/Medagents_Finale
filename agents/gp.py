"""
agents/gp.py — General Physician Agent
========================================
Performs initial patient assessment from symptoms and forms a differential diagnosis.
Active at: Stage.INITIAL_ASSESSMENT and Stage.TEST_RECOMMENDATION
"""

import random
from typing import Dict, List, Any, Optional

from task import Action, Stage


# ─── Symptom → Disease heuristics (pre-training baseline) ─────────────────────
# These represent naive associations the GP uses before memory-guided improvement.
SYMPTOM_DISEASE_MAP = {
    "fever":              ["flu", "malaria", "dengue", "viral fever", "sepsis", "covid-19"],
    "cough":              ["flu", "tuberculosis", "chronic bronchitis", "covid-19", "asthma", "lung cancer"],
    "chest pain":         ["heart attack", "angina", "anxiety"],
    "headache":           ["migraine", "meningitis", "brain tumor", "viral fever"],
    "weight loss":        ["tuberculosis", "cancer", "lymphoma", "diabetes type 1"],
    "fatigue":            ["anemia", "kidney disease", "hepatitis", "lymphoma"],
    "joint pain":         ["dengue", "arthritis", "chikungunya"],
    "rash":               ["dengue", "chickenpox", "allergy", "eczema"],
    "shortness of breath":["heart attack", "asthma", "lung cancer"],
    "confusion":          ["sepsis", "meningitis"],
    "yellow skin":        ["hepatitis", "liver disease"],
    "frequent urination": ["diabetes type 2", "urinary tract infection"],
    "loss of taste":      ["covid-19"],
    "neck stiffness":     ["meningitis", "migraine"],
    "blood in sputum":    ["lung cancer", "tuberculosis"],
    "night sweats":       ["tuberculosis", "lymphoma"],
    "pale skin":          ["anemia"],
    "blisters":           ["chickenpox"],
    "swelling":           ["kidney disease", "heart failure"],
}

# Critical symptom flags that should always be escalated
CRITICAL_SYMPTOMS = {
    "blood in sputum", "chest pain", "confusion", "neck stiffness",
    "severe headache", "shortness of breath", "sweating", "low platelets",
    "breathlessness"
}


class GPAgent:
    """
    General Physician: first-contact agent that sees raw symptoms and
    generates an initial differential diagnosis with confidence.

    Pre-training: Uses heuristic symptom matching (low accuracy).
    Post-training: Will use RL-improved policy (higher accuracy).
    """

    def __init__(self, memory=None, noise_level: float = 0.35):
        """
        Args:
            memory:      AgentMemory instance (injected from MemorySystem)
            noise_level: Probability of making a suboptimal choice (simulates pre-training noise)
        """
        self.memory = memory
        self.noise_level = noise_level  # 0.35 = poor pre-training performance

    # ── Core Act Method ───────────────────────────────────────────────────────

    def act(self, observation: Dict[str, Any]) -> List[Action]:
        """
        Given the current observation, return a list of actions for this stage.
        """
        stage_name = observation.get("stage", "")
        visible = observation.get("visible_info", {})
        symptoms = visible.get("symptoms", [])
        case_id = observation.get("case_id", 0)

        actions = []

        if stage_name == "INITIAL_ASSESSMENT":
            actions.extend(self._initial_assessment(symptoms, case_id))

        elif stage_name == "TEST_RECOMMENDATION":
            possible_diseases = visible.get("possible_diseases", [])
            actions.extend(self._recommend_tests(symptoms, possible_diseases, case_id))

        return actions

    # ── Stage 0: Initial Assessment ───────────────────────────────────────────

    def _initial_assessment(self, symptoms: List[str], case_id: int) -> List[Action]:
        """Form differential diagnosis from symptoms."""
        differential = self._build_differential(symptoms)
        critical_flags = [s for s in symptoms if s.lower() in CRITICAL_SYMPTOMS]

        # Check memory for relevant past feedback
        memory_hint = ""
        if self.memory:
            past = self.memory.get_relevant_memory(symptoms, top_k=2)
            if past:
                memory_hint = f"Past similar cases: {[p['feedback'] for p in past]}"

        reasoning = (
            f"Patient presents with: {', '.join(symptoms)}. "
            f"Differential diagnosis based on symptom matching: {differential}. "
            f"{('CRITICAL FLAGS: ' + str(critical_flags) + '. Escalating.') if critical_flags else ''}"
            f"{(' Memory hint: ' + memory_hint) if memory_hint else ''}"
        )

        actions = [
            Action(
                agent_role="gp",
                action_type="form_differential",
                content={
                    "differential": differential,
                    "reasoning": reasoning,
                    "confidence": self._estimate_confidence(differential),
                },
            )
        ]

        if critical_flags:
            actions.append(
                Action(
                    agent_role="gp",
                    action_type="note_critical_flags",
                    content={
                        "flags": critical_flags,
                        "reasoning": f"Critical symptoms detected: {critical_flags}. Urgent workup required.",
                    },
                )
            )

        return actions

    # ── Stage 1: Test Recommendation ──────────────────────────────────────────

    def _recommend_tests(
        self, symptoms: List[str], possible_diseases: List[str], case_id: int
    ) -> List[Action]:
        """Recommend diagnostic tests based on differential."""
        tests_to_order = self._select_tests(possible_diseases)

        actions = []
        for test in tests_to_order:
            actions.append(
                Action(
                    agent_role="gp",
                    action_type="order_test",
                    content={
                        "test_name": test,
                        "reasoning": f"Ordering {test} to narrow differential for {possible_diseases}.",
                        "confidence": random.uniform(0.5, 0.9),
                    },
                )
            )
        return actions

    # ── Heuristics ────────────────────────────────────────────────────────────

    def _build_differential(self, symptoms: List[str]) -> List[str]:
        """
        Build differential diagnosis list from symptom-disease heuristics.
        Applies noise to simulate pre-training imperfection.
        """
        disease_scores: Dict[str, int] = {}
        for symptom in symptoms:
            related = SYMPTOM_DISEASE_MAP.get(symptom.lower(), [])
            for d in related:
                disease_scores[d] = disease_scores.get(d, 0) + 1

        # Sort by score descending
        sorted_diseases = sorted(disease_scores, key=lambda d: disease_scores[d], reverse=True)
        differential = sorted_diseases[:3] if sorted_diseases else ["unknown"]

        # Pre-training noise: occasionally shuffle or drop the top result
        if random.random() < self.noise_level:
            random.shuffle(differential)

        return differential

    def _estimate_confidence(self, differential: List[str]) -> float:
        """Estimate confidence based on differential size (fewer = more confident)."""
        base = 0.9 - (len(differential) - 1) * 0.15
        noise = random.uniform(-0.1, 0.1)
        return round(max(0.2, min(0.95, base + noise)), 3)

    def _select_tests(self, possible_diseases: List[str]) -> List[str]:
        """Select appropriate tests for the differential."""
        disease_tests = {
            "flu":                   ["rapid_test"],
            "covid-19":              ["rt_pcr", "rapid_test"],
            "tuberculosis":          ["sputum", "xray"],
            "lung cancer":           ["biopsy", "xray"],
            "heart attack":          ["troponin", "ecg"],
            "diabetes type 2":       ["blood_sugar"],
            "diabetes type 1":       ["blood_sugar", "insulin"],
            "dengue":                ["platelet_count", "antigen"],
            "malaria":               ["blood"],
            "meningitis":            ["csf"],
            "brain tumor":           ["mri"],
            "hepatitis":             ["liver_function"],
            "anemia":                ["hemoglobin"],
            "kidney disease":        ["creatinine"],
            "sepsis":                ["blood"],
            "asthma":                ["spirometry"],
            "skin cancer":           ["biopsy"],
            "lymphoma":              ["biopsy"],
            "stomach cancer":        ["endoscopy"],
            "arthritis":             ["xray"],
            "psoriasis":             ["biopsy"],
            "spine infection":       ["mri"],
        }

        tests = set()
        for disease in possible_diseases:
            for t in disease_tests.get(disease.lower(), []):
                tests.add(t)

        # Noise: occasionally add an irrelevant test (unnecessary test penalty)
        if random.random() < self.noise_level * 0.5:
            tests.add(random.choice(["cbc", "urine", "allergy_test"]))

        return list(tests)[:3]  # cap at 3 tests per GP recommendation
