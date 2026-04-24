"""
agents/specialist.py — Specialist Agent
========================================
Senior physician who synthesizes all available evidence and proposes the final diagnosis.
Active at: Stage.TEST_RECOMMENDATION (co-orders) and Stage.DIAGNOSIS_DECISION
"""

import random
from typing import Dict, List, Any, Optional

from task import Action


# ─── Disease → required evidence map ─────────────────────────────────────────
# For each disease, what test findings are strongly confirmatory?
CONFIRMATORY_EVIDENCE = {
    "flu":                    ["rapid_test", "negative covid"],
    "covid-19":               ["rt_pcr", "positive"],
    "tuberculosis":           ["sputum", "tb bacteria"],
    "lung cancer":            ["biopsy", "malignant"],
    "heart attack":           ["troponin", "ecg", "elevated", "abnormal"],
    "diabetes type 2":        ["blood_sugar", "high"],
    "diabetes type 1":        ["insulin", "low"],
    "dengue":                 ["platelet_count", "antigen", "dengue"],
    "malaria":                ["blood", "parasite"],
    "meningitis":             ["csf", "infection"],
    "brain tumor":            ["mri", "tumor"],
    "hepatitis":              ["liver_function", "abnormal"],
    "anemia":                 ["hemoglobin", "low"],
    "kidney disease":         ["creatinine", "high"],
    "sepsis":                 ["blood", "infection markers"],
    "asthma":                 ["spirometry", "airflow"],
    "skin cancer":            ["biopsy", "melanoma"],
    "lymphoma":               ["biopsy", "cancerous"],
    "stomach cancer":         ["endoscopy", "tumor"],
    "arthritis":              ["xray", "joint"],
    "psoriasis":              ["biopsy", "psoriatic"],
    "spine infection":        ["mri", "infection"],
    "food poisoning":         ["stool", "toxins"],
    "viral fever":            ["blood", "normal viral"],
    "eczema":                 ["skin_exam", "inflammation"],
    "chickenpox":             ["clinical", "lesions"],
    "allergy":                ["allergy_test", "positive"],
    "vitiligo":               ["wood_lamp", "depigmentation"],
    "fungal infection":       ["skin_scrape", "fungus"],
    "cancer":                 ["biopsy", "malignant"],
    "urinary tract infection":["urine", "bacteria"],
    "chronic bronchitis":     ["xray", "bronchial"],
}


class SpecialistAgent:
    """
    Specialist: synthesizes GP assessment + test results into a final diagnosis.

    Decision logic:
      1. Collect all agent suggestions from the observation
      2. Score each candidate disease against test evidence
      3. Propose highest-scoring diagnosis with calibrated confidence
      4. Apply noise to simulate pre-training imperfection
    """

    def __init__(self, memory=None, noise_level: float = 0.30):
        self.memory = memory
        self.noise_level = noise_level

    def act(self, observation: Dict[str, Any]) -> List[Action]:
        stage_name = observation.get("stage", "")
        visible = observation.get("visible_info", {})
        actions = []

        if stage_name == "TEST_RECOMMENDATION":
            actions.extend(self._co_order_tests(visible))

        elif stage_name == "DIAGNOSIS_DECISION":
            actions.extend(self._make_diagnosis(visible, observation.get("case_id", 0)))

        return actions

    # ── Stage 1 co-ordering ───────────────────────────────────────────────────

    def _co_order_tests(self, visible: Dict) -> List[Action]:
        """Specialist may add specialised tests the GP missed."""
        possible_diseases = visible.get("possible_diseases", [])
        symptoms = visible.get("symptoms", [])
        critical_tests = []

        # If any critical disease possible, ensure key test is ordered
        critical_diseases = {"heart attack", "lung cancer", "meningitis", "sepsis", "brain tumor"}
        for d in possible_diseases:
            if d.lower() in critical_diseases:
                evidence_keys = CONFIRMATORY_EVIDENCE.get(d.lower(), [])
                if evidence_keys:
                    critical_tests.append(evidence_keys[0])

        actions = []
        for test in critical_tests[:1]:  # One specialist test order per stage
            actions.append(
                Action(
                    agent_role="specialist",
                    action_type="order_test",
                    content={
                        "test_name": test,
                        "reasoning": f"Specialist-ordered: critical workup for suspected {possible_diseases}.",
                        "confidence": 0.80,
                    },
                )
            )
        return actions

    # ── Stage 3: Diagnosis Decision ───────────────────────────────────────────

    def _make_diagnosis(self, visible: Dict, case_id: int) -> List[Action]:
        """Synthesize all evidence to produce final diagnosis."""
        possible_diseases = visible.get("possible_diseases", [])
        test_results = visible.get("test_results", {})
        symptoms = visible.get("symptoms", [])

        # Fallback: if possible_diseases is empty, pull from GP assessment
        if not possible_diseases:
            gp_assessment = visible.get("gp_assessment", [])
            for entry in gp_assessment:
                content = entry.get("content", {})
                possible_diseases = content.get("differential", [])
                if possible_diseases:
                    break

        # Score each candidate disease
        scores = {}
        for disease in possible_diseases:
            scores[disease] = self._score_disease(disease, test_results, symptoms)

        # Optionally refine differential
        if len(possible_diseases) > 1:
            sorted_candidates = sorted(scores, key=lambda d: scores[d], reverse=True)
            diagnosis = sorted_candidates[0]
            confidence = round(min(0.95, scores[diagnosis] / 10.0 + 0.5), 3)
        elif possible_diseases:
            diagnosis = possible_diseases[0]
            confidence = 0.70
        else:
            # Last-resort: derive from test results
            diagnosis = list(test_results.keys())[0].replace("_", " ") if test_results else "unknown"
            confidence = 0.40

        # Pre-training noise: sometimes pick wrong diagnosis
        if random.random() < self.noise_level and len(possible_diseases) > 1:
            # Pick the second-best candidate
            diagnosis = sorted(scores, key=lambda d: scores[d], reverse=True)[-1]
            confidence = min(confidence, 0.65)

        # Memory-guided correction
        if self.memory:
            rules = self.memory.get_learned_rules()
            for rule in rules:
                # If rule mentions a disease relevant to current symptoms
                for d in possible_diseases:
                    if d.lower() in rule.lower():
                        # Trust the rule: override noise
                        rule_disease = d
                        diagnosis = rule_disease
                        confidence = min(confidence + 0.05, 0.95)
                        break

        # Flag uncertainty if confidence is low
        actions = []
        if confidence < 0.6:
            actions.append(
                Action(
                    agent_role="specialist",
                    action_type="flag_uncertainty",
                    content={
                        "reasoning": f"Low confidence ({confidence:.2f}) across candidates: {scores}",
                        "confidence": confidence,
                    },
                )
            )

        # Always produce a diagnosis
        actions.append(
            Action(
                agent_role="specialist",
                action_type="propose_diagnosis",
                content={
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "reasoning": (
                        f"Synthesized evidence: symptoms={symptoms}, "
                        f"tests={test_results}, scores={scores}. "
                        f"Final diagnosis: {diagnosis} (confidence={confidence})."
                    ),
                    "evidence_scores": scores,
                },
            )
        )
        return actions

    def _score_disease(self, disease: str, test_results: Dict, symptoms: List[str]) -> float:
        """Score a disease against available test evidence and symptoms."""
        score = 0.0
        evidence_keys = CONFIRMATORY_EVIDENCE.get(disease.lower(), [])
        test_text = " ".join(f"{k} {v}" for k, v in test_results.items()).lower()

        for keyword in evidence_keys:
            if keyword.lower() in test_text:
                score += 2.0

        # Symptom bonus
        for symptom in symptoms:
            if symptom.lower() in disease.lower() or disease.lower() in symptom.lower():
                score += 0.5

        return round(score, 2)
