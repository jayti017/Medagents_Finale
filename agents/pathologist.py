"""
agents/pathologist.py — Pathologist Agent
==========================================
Interprets laboratory and biopsy test results.
Active at: Stage.TEST_ANALYSIS
"""

import random
from typing import Dict, List, Any

from task import Action


# ─── Lab test interpretation patterns ─────────────────────────────────────────
LAB_INTERPRETATIONS = {
    "biopsy": {
        "malignant cells":       ("Biopsy shows malignant cellular morphology", "lung cancer", 0.91),
        "cancerous lymph cells": ("Lymph node biopsy positive for lymphoma", "lymphoma", 0.90),
        "malignant melanoma":    ("Skin biopsy confirms malignant melanoma", "skin cancer", 0.93),
        "malignant":             ("Tissue biopsy confirms malignancy", "cancer", 0.90),
        "psoriatic cells":       ("Skin biopsy shows psoriatic histopathology", "psoriasis", 0.85),
    },
    "blood": {
        "parasite present":          ("Blood smear positive for plasmodium", "malaria", 0.92),
        "infection markers high":    ("Elevated WBC, CRP, procalcitonin — sepsis pattern", "sepsis", 0.88),
        "normal viral":              ("Viral pattern on CBC; no bacterial growth", "viral fever", 0.75),
    },
    "sputum": {
        "tb bacteria detected": ("AFB smear positive — Mycobacterium tuberculosis", "tuberculosis", 0.94),
    },
    "urine": {
        "bacteria present": ("Significant bacteriuria on urine culture", "urinary tract infection", 0.88),
    },
    "platelet_count": {
        "low": ("Thrombocytopenia — platelet count critically low", "dengue", 0.82),
    },
    "hemoglobin": {
        "low": ("Hemoglobin below normal range — iron deficiency/anemia pattern", "anemia", 0.87),
    },
    "liver_function": {
        "abnormal": ("Elevated AST/ALT/bilirubin — hepatocellular injury pattern", "hepatitis", 0.86),
    },
    "creatinine": {
        "high": ("Elevated serum creatinine — reduced GFR, renal insufficiency", "kidney disease", 0.89),
    },
    "troponin": {
        "elevated": ("Troponin I significantly elevated — myocardial injury confirmed", "heart attack", 0.95),
    },
    "csf": {
        "infection present": ("CSF shows pleocytosis and elevated protein — bacterial meningitis", "meningitis", 0.93),
    },
    "antigen": {
        "dengue positive": ("NS1 antigen positive — active dengue infection", "dengue", 0.91),
    },
    "stool": {
        "toxins present": ("Stool toxin assay positive — bacterial food poisoning", "food poisoning", 0.85),
    },
    "insulin": {
        "low": ("Serum insulin undetectable — Type 1 diabetes pattern", "diabetes type 1", 0.88),
    },
    "blood_sugar": {
        "high": ("Fasting blood glucose elevated — hyperglycemia", "diabetes type 2", 0.83),
    },
    "spirometry": {
        "reduced airflow": ("FEV1/FVC ratio <0.70 — obstructive pattern consistent with asthma", "asthma", 0.84),
    },
    "endoscopy": {
        "tumor found": ("Endoscopy reveals gastric mass with irregular borders", "stomach cancer", 0.91),
    },
    "rt_pcr": {
        "positive": ("RT-PCR positive for SARS-CoV-2", "covid-19", 0.97),
    },
    "rapid_test": {
        "negative covid": ("Rapid antigen test negative for COVID-19", "flu", 0.72),
    },
    "skin_exam": {
        "inflammation": ("Skin examination shows spongiosis — eczematous pattern", "eczema", 0.80),
    },
    "skin_scrape": {
        "fungus present": ("KOH preparation positive for fungal hyphae", "fungal infection", 0.90),
    },
    "wood_lamp": {
        "depigmentation": ("Wood's lamp shows melanin loss — vitiligo", "vitiligo", 0.88),
    },
    "allergy_test": {
        "positive": ("Skin prick test positive for multiple allergens", "allergy", 0.83),
    },
    "clinical": {
        "typical lesions": ("Classic vesicular lesions in various stages — chickenpox", "chickenpox", 0.91),
    },
}

# Tests that pathologist handles (non-imaging)
LAB_TESTS = set(LAB_INTERPRETATIONS.keys())


class PathologistAgent:
    """
    Pathologist: interprets lab results, biopsies, and non-imaging test outputs.

    Pre-training: decent lab pattern matching, noisy under uncertainty.
    Post-training: higher recall on ambiguous results.
    """

    def __init__(self, memory=None, noise_level: float = 0.28):
        self.memory = memory
        self.noise_level = noise_level

    def act(self, observation: Dict[str, Any]) -> List[Action]:
        """Interpret lab/biopsy results from current observation."""
        stage_name = observation.get("stage", "")
        if stage_name != "TEST_ANALYSIS":
            return []

        visible = observation.get("visible_info", {})
        test_results = visible.get("test_results", {})
        symptoms = visible.get("symptoms", [])

        actions = []
        for test_name, result_text in test_results.items():
            if test_name.lower() in LAB_TESTS:
                action = self._interpret_lab(test_name, result_text, symptoms)
                if action:
                    actions.append(action)

        if not actions:
            actions.append(
                Action(
                    agent_role="pathologist",
                    action_type="interpret_lab",
                    content={
                        "test": "general_lab_review",
                        "finding": "No lab tests ordered or available for this case.",
                        "suggested_diagnosis": None,
                        "confidence": 0.5,
                        "reasoning": "Deferring to radiology for imaging-based workup.",
                    },
                )
            )

        return actions

    def _interpret_lab(self, test_name: str, result_text: str, symptoms: List[str]) -> Action:
        """Interpret a single lab test result."""
        test_lower = test_name.lower()
        result_lower = result_text.lower()

        interp_map = LAB_INTERPRETATIONS.get(test_lower, {})
        matched_finding = None
        matched_dx = None
        base_conf = 0.65

        for key, (finding, dx, conf) in interp_map.items():
            if key.lower() in result_lower:
                matched_finding = finding
                matched_dx = dx
                base_conf = conf
                break

        if not matched_finding:
            matched_finding = f"Lab result for {test_name}: {result_text} — requires specialist review"
            matched_dx = None
            base_conf = 0.55

        # Pre-training noise
        if random.random() < self.noise_level:
            base_conf = max(0.3, base_conf - random.uniform(0.1, 0.25))
            if random.random() < 0.5:
                matched_dx = None

        # Memory enrichment
        memory_note = ""
        if self.memory:
            past = self.memory.get_relevant_memory(symptoms, top_k=1)
            if past:
                feedback = past[0].get("feedback", {})
                if isinstance(feedback, dict):
                    memory_note = f" (Learned rule: {feedback.get('corrective_rule', '')})"

        reasoning = (
            f"Lab interpretation for {test_name}: {matched_finding}. "
            f"Raw result: '{result_text}'.{memory_note}"
        )

        action_type = "flag_abnormal" if matched_dx else "interpret_lab"

        return Action(
            agent_role="pathologist",
            action_type=action_type,
            content={
                "test": test_name,
                "finding": matched_finding,
                "suggested_diagnosis": matched_dx,
                "confidence": round(base_conf + random.uniform(-0.04, 0.04), 3),
                "reasoning": reasoning,
            },
        )
