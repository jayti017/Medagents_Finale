"""
reward.py — MedAgents-X Reward System
======================================
Defines the reward function used for RL training.
All reward signals are designed to be logged per-step for line graph visualization.

Reward breakdown (total up to +1.0 possible per episode):
  +0.40  Correct final diagnosis
  +0.20  Evidence-based reasoning (tests ordered → support diagnosis)
  +0.20  Appropriate confidence (not overconfident, not underconfident)
  +0.10  Efficiency (fewer unnecessary tests)
  +0.10  No critical disease missed

Penalties:
  -0.50  Wrong diagnosis
  -0.20  Critical disease missed (when correct_diagnosis is critical)
  -0.15  Overconfidence on wrong answer (confidence > 0.8 and wrong)
  -0.10  Unnecessary test ordered (test not in dataset's defined tests)
  -0.05  Step delay penalty per extra step beyond minimum required
"""

from typing import Dict, Any, List


# ─── Severity multipliers ──────────────────────────────────────────────────────
SEVERITY_MULTIPLIER = {
    "low": 1.0,
    "medium": 1.2,
    "high": 1.5,
    "critical": 2.0,
}

# ─── Minimum steps required per stage ─────────────────────────────────────────
MIN_STEPS_FOR_CASE = 4  # assessment → test_rec → test_analysis → diagnosis


def compute_reward(
    predicted_diagnosis: str,
    correct_diagnosis: str,
    confidence: float,
    tests_ordered: List[str],
    valid_tests: Dict[str, str],
    reasoning_steps: List[str],
    steps_taken: int,
    severity: str = "medium",
    is_critical: bool = False,
) -> Dict[str, Any]:
    """
    Compute the full reward for a completed episode.

    Args:
        predicted_diagnosis:  Agent's final diagnosis string
        correct_diagnosis:    Ground truth from dataset
        confidence:           Float in [0, 1] — agent's stated confidence
        tests_ordered:        List of test names the agent requested
        valid_tests:          Dict of valid tests from the dataset case
        reasoning_steps:      List of reasoning strings produced by agents
        steps_taken:          Total number of steps used in this episode
        severity:             Case severity label
        is_critical:          Whether correct_diagnosis is a critical disease

    Returns:
        Dict with keys: total_reward, breakdown, shaping_bonus, penalty_detail
    """

    breakdown = {}
    penalties = {}
    multiplier = SEVERITY_MULTIPLIER.get(severity, 1.0)

    # ── 1. Correct diagnosis reward ────────────────────────────────────────────
    correct = predicted_diagnosis.strip().lower() == correct_diagnosis.strip().lower()
    breakdown["correct_diagnosis"] = 0.40 if correct else 0.0

    # ── 2. Evidence-based reasoning ────────────────────────────────────────────
    # At least one ordered test must be present in the valid test set
    valid_test_keys = set(valid_tests.keys())
    ordered_set = set(t.lower() for t in tests_ordered)
    evidence_overlap = len(ordered_set & valid_test_keys)
    evidence_score = min(evidence_overlap / max(len(valid_test_keys), 1), 1.0) * 0.20
    breakdown["evidence_based_reasoning"] = round(evidence_score, 4)

    # ── 3. Confidence calibration ──────────────────────────────────────────────
    # Ideal: correct + high confidence, or wrong + low confidence
    if correct:
        # Reward confidence proportionally when correct
        conf_score = confidence * 0.20
    else:
        # Penalize if confident but wrong
        conf_score = (1.0 - confidence) * 0.10  # partial reward for low conf on wrong
    breakdown["confidence_calibration"] = round(conf_score, 4)

    # ── 4. Efficiency (unnecessary test penalty) ───────────────────────────────
    unnecessary = [t for t in tests_ordered if t.lower() not in valid_test_keys]
    unnecessary_penalty = min(len(unnecessary) * 0.10, 0.30)
    penalties["unnecessary_tests"] = -round(unnecessary_penalty, 4)
    efficiency_score = max(0.10 - unnecessary_penalty, 0.0)
    breakdown["efficiency"] = round(efficiency_score, 4)

    # ── 5. Critical disease not missed ────────────────────────────────────────
    if is_critical:
        if correct:
            breakdown["critical_disease_detected"] = 0.10
        else:
            breakdown["critical_disease_detected"] = 0.0
            penalties["critical_miss"] = -0.20
    else:
        breakdown["critical_disease_detected"] = 0.10  # not critical, no risk

    # ── 6. Step delay penalty ──────────────────────────────────────────────────
    extra_steps = max(steps_taken - MIN_STEPS_FOR_CASE, 0)
    delay_penalty = min(extra_steps * 0.05, 0.20)
    penalties["step_delay"] = -round(delay_penalty, 4)

    # ── 7. Wrong diagnosis penalty ────────────────────────────────────────────
    if not correct:
        penalties["wrong_diagnosis"] = -0.50

    # ── 8. Overconfidence penalty ─────────────────────────────────────────────
    if not correct and confidence > 0.8:
        penalties["overconfidence"] = -0.15

    # ── 9. Reasoning quality bonus ────────────────────────────────────────────
    # Shaping bonus: reward agents that produce structured reasoning chains
    shaping_bonus = 0.0
    if len(reasoning_steps) >= 3:
        shaping_bonus += 0.05
    if len(reasoning_steps) >= 5:
        shaping_bonus += 0.05

    # ── Aggregate ─────────────────────────────────────────────────────────────
    raw_positive = sum(breakdown.values())
    raw_penalty = sum(penalties.values())
    total_before_multiplier = raw_positive + raw_penalty + shaping_bonus

    # Apply severity multiplier only to the positive component
    total_reward = round(raw_positive * multiplier + raw_penalty + shaping_bonus, 4)
    total_reward = max(total_reward, -1.0)  # floor at -1.0

    return {
        "total_reward": total_reward,
        "breakdown": breakdown,
        "penalties": penalties,
        "shaping_bonus": shaping_bonus,
        "severity_multiplier": multiplier,
        "is_correct": correct,
        "confidence": confidence,
        "steps_taken": steps_taken,
        "tests_ordered": tests_ordered,
        "unnecessary_tests": unnecessary,
    }


def compute_step_reward(stage: str, action_valid: bool, progress_made: bool) -> float:
    """
    Dense per-step reward for shaping the training signal.
    Called at every environment step during an episode.

    Args:
        stage:          Current pipeline stage name
        action_valid:   Whether the action taken was valid for this stage
        progress_made:  Whether this action moved the case forward

    Returns:
        Small float reward in [-0.05, +0.05]
    """
    reward = 0.0
    if action_valid:
        reward += 0.02
    if progress_made:
        reward += 0.03
    if not action_valid:
        reward -= 0.05
    return round(reward, 4)
