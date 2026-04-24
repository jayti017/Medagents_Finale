"""
task.py — MedAgents-X Multi-Step Task Pipeline
================================================
Controls the structured workflow that agents must follow for each patient case.

Pipeline Stages (in order):
  Stage 0: INITIAL_ASSESSMENT     — GP reviews symptoms, forms differential
  Stage 1: TEST_RECOMMENDATION    — GP + Specialist decide which tests to order
  Stage 2: TEST_ANALYSIS          — Radiologist + Pathologist interpret results
  Stage 3: DIAGNOSIS_DECISION     — Specialist proposes final diagnosis
  Stage 4: SUPERVISOR_REVIEW      — Supervisor validates + gives final verdict
  Stage 5: OVERSIGHT_FEEDBACK     — Oversight AI logs patterns + cross-case analysis

Each stage defines:
  - Which agent(s) are active
  - What actions are valid
  - What information is available (partial observability)
  - Transition conditions (when to move to next stage)
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ─── Stage Definitions ────────────────────────────────────────────────────────

class Stage(Enum):
    INITIAL_ASSESSMENT   = 0
    TEST_RECOMMENDATION  = 1
    TEST_ANALYSIS        = 2
    DIAGNOSIS_DECISION   = 3
    SUPERVISOR_REVIEW    = 4
    OVERSIGHT_FEEDBACK   = 5
    DONE                 = 6


STAGE_NAMES = {
    Stage.INITIAL_ASSESSMENT:   "Initial Assessment",
    Stage.TEST_RECOMMENDATION:  "Test Recommendation",
    Stage.TEST_ANALYSIS:        "Test Analysis",
    Stage.DIAGNOSIS_DECISION:   "Diagnosis Decision",
    Stage.SUPERVISOR_REVIEW:    "Supervisor Review",
    Stage.OVERSIGHT_FEEDBACK:   "Oversight Feedback",
    Stage.DONE:                 "Done",
}

# Which agent(s) act at each stage
STAGE_AGENTS = {
    Stage.INITIAL_ASSESSMENT:   ["gp"],
    Stage.TEST_RECOMMENDATION:  ["gp", "specialist"],
    Stage.TEST_ANALYSIS:        ["radiologist", "pathologist"],
    Stage.DIAGNOSIS_DECISION:   ["specialist"],
    Stage.SUPERVISOR_REVIEW:    ["supervisor"],
    Stage.OVERSIGHT_FEEDBACK:   ["oversight"],
    Stage.DONE:                 [],
}

# Valid action types per stage
STAGE_VALID_ACTIONS = {
    Stage.INITIAL_ASSESSMENT: [
        "form_differential",       # List candidate diagnoses from symptoms
        "note_critical_flags",     # Flag potentially critical conditions
        "request_more_history",    # Ask for additional patient history
    ],
    Stage.TEST_RECOMMENDATION: [
        "order_test",              # Request a specific diagnostic test
        "justify_test",            # Provide clinical rationale for test
        "skip_unnecessary_test",   # Explicitly decide not to order a test
    ],
    Stage.TEST_ANALYSIS: [
        "interpret_imaging",       # Radiologist reads imaging result
        "interpret_lab",           # Pathologist reads lab result
        "flag_abnormal",           # Flag a result as clinically significant
        "request_repeat_test",     # Ask for a test to be repeated
    ],
    Stage.DIAGNOSIS_DECISION: [
        "propose_diagnosis",       # Submit final diagnosis with confidence
        "refine_differential",     # Narrow down candidates before deciding
        "flag_uncertainty",        # Note diagnostic uncertainty
    ],
    Stage.SUPERVISOR_REVIEW: [
        "approve_diagnosis",       # Accept the specialist's diagnosis
        "reject_diagnosis",        # Reject and send back for re-evaluation
        "provide_feedback",        # Give structured feedback to agents
        "escalate_case",           # Mark case as requiring further review
    ],
    Stage.OVERSIGHT_FEEDBACK: [
        "log_pattern",             # Record a cross-case pattern
        "update_agent_rules",      # Push a learned rule to agent memory
        "generate_report",         # Produce oversight summary
    ],
}

# Minimum actions required before transitioning to the next stage
MIN_ACTIONS_PER_STAGE = {
    Stage.INITIAL_ASSESSMENT:   1,
    Stage.TEST_RECOMMENDATION:  1,
    Stage.TEST_ANALYSIS:        1,
    Stage.DIAGNOSIS_DECISION:   1,
    Stage.SUPERVISOR_REVIEW:    1,
    Stage.OVERSIGHT_FEEDBACK:   1,
}


# ─── Action Schema ────────────────────────────────────────────────────────────

@dataclass
class Action:
    """Represents a single agent action at a specific pipeline stage."""
    agent_role: str
    action_type: str
    content: Dict[str, Any] = field(default_factory=dict)

    def is_valid_for_stage(self, stage: Stage) -> bool:
        """Check whether this action is allowed in the current stage."""
        allowed_agents = STAGE_AGENTS.get(stage, [])
        allowed_actions = STAGE_VALID_ACTIONS.get(stage, [])
        return self.agent_role in allowed_agents and self.action_type in allowed_actions


# ─── Task Pipeline ────────────────────────────────────────────────────────────

class TaskPipeline:
    """
    Controls multi-step workflow for a single patient case.

    Responsibilities:
      - Track current stage
      - Validate agent actions
      - Control stage transitions
      - Log all actions for reward computation
    """

    def __init__(self, case_id: int):
        self.case_id = case_id
        self.current_stage = Stage.INITIAL_ASSESSMENT
        self.stage_action_counts: Dict[Stage, int] = {s: 0 for s in Stage}
        self.action_log: List[Dict[str, Any]] = []
        self.stage_outputs: Dict[Stage, List[Dict]] = {s: [] for s in Stage}
        self.is_complete = False
        self.final_diagnosis: Optional[str] = None
        self.final_confidence: Optional[float] = None
        self.approved: bool = False

    # ── Action Execution ──────────────────────────────────────────────────────

    def execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Attempt to execute an action at the current stage.

        Returns a dict with:
          - success: bool
          - message: explanation
          - stage: current stage name
          - transition: whether stage advanced
        """
        if self.is_complete:
            return {"success": False, "message": "Task pipeline is already complete.", "transition": False}

        # Validate action for this stage
        if not action.is_valid_for_stage(self.current_stage):
            return {
                "success": False,
                "message": (
                    f"Action '{action.action_type}' by '{action.agent_role}' "
                    f"is invalid at stage '{STAGE_NAMES[self.current_stage]}'. "
                    f"Valid agents: {STAGE_AGENTS[self.current_stage]}, "
                    f"Valid actions: {STAGE_VALID_ACTIONS[self.current_stage]}"
                ),
                "transition": False,
                "stage": STAGE_NAMES[self.current_stage],
            }

        # Record the action
        log_entry = {
            "stage": self.current_stage.name,
            "agent": action.agent_role,
            "action_type": action.action_type,
            "content": action.content,
        }
        self.action_log.append(log_entry)
        self.stage_outputs[self.current_stage].append(log_entry)
        self.stage_action_counts[self.current_stage] += 1

        # Handle specific action side effects
        if action.action_type == "propose_diagnosis":
            self.final_diagnosis = action.content.get("diagnosis")
            self.final_confidence = action.content.get("confidence", 0.5)

        if action.action_type == "approve_diagnosis":
            self.approved = True

        if action.action_type == "reject_diagnosis":
            self.approved = False
            # Regress to test analysis for re-evaluation
            self.current_stage = Stage.TEST_ANALYSIS
            return {
                "success": True,
                "message": "Diagnosis rejected. Returning to Test Analysis stage.",
                "transition": True,
                "stage": STAGE_NAMES[self.current_stage],
            }

        # Check if stage is complete and should advance
        transitioned = False
        min_required = MIN_ACTIONS_PER_STAGE.get(self.current_stage, 1)
        if self.stage_action_counts[self.current_stage] >= min_required:
            transitioned = self._advance_stage()

        return {
            "success": True,
            "message": f"Action '{action.action_type}' executed successfully.",
            "transition": transitioned,
            "stage": STAGE_NAMES[self.current_stage],
        }

    # ── Stage Transition ──────────────────────────────────────────────────────

    def _advance_stage(self) -> bool:
        """Move to the next stage in the pipeline. Returns True if advanced."""
        stage_order = [
            Stage.INITIAL_ASSESSMENT,
            Stage.TEST_RECOMMENDATION,
            Stage.TEST_ANALYSIS,
            Stage.DIAGNOSIS_DECISION,
            Stage.SUPERVISOR_REVIEW,
            Stage.OVERSIGHT_FEEDBACK,
            Stage.DONE,
        ]
        idx = stage_order.index(self.current_stage)
        if idx < len(stage_order) - 1:
            self.current_stage = stage_order[idx + 1]
            if self.current_stage == Stage.DONE:
                self.is_complete = True
            return True
        return False

    # ── Observation ───────────────────────────────────────────────────────────

    def get_current_observation(self, case_data: Dict) -> Dict[str, Any]:
        """
        Return partial observation for the current stage.
        Implements partial observability: agents only see what's available so far.
        """
        obs = {
            "case_id": self.case_id,
            "stage": self.current_stage.name,
            "stage_label": STAGE_NAMES[self.current_stage],
            "active_agents": STAGE_AGENTS[self.current_stage],
            "valid_actions": STAGE_VALID_ACTIONS.get(self.current_stage, []),
            "is_complete": self.is_complete,
        }

        stage = self.current_stage

        # Stage 0: Only symptoms visible
        if stage == Stage.INITIAL_ASSESSMENT:
            obs["visible_info"] = {
                "symptoms": case_data["symptoms"],
            }

        # Stage 1: Symptoms + possible diseases visible
        elif stage == Stage.TEST_RECOMMENDATION:
            obs["visible_info"] = {
                "symptoms": case_data["symptoms"],
                "possible_diseases": case_data["possible_diseases"],
                "gp_assessment": self._get_stage_output(Stage.INITIAL_ASSESSMENT),
            }

        # Stage 2: Test results become available (delayed reveal)
        elif stage == Stage.TEST_ANALYSIS:
            obs["visible_info"] = {
                "symptoms": case_data["symptoms"],
                "possible_diseases": case_data["possible_diseases"],
                "tests_ordered": self._get_ordered_tests(),
                "test_results": case_data["tests"],  # revealed now
                "gp_assessment": self._get_stage_output(Stage.INITIAL_ASSESSMENT),
            }

        # Stage 3: Full picture for diagnosis decision
        elif stage == Stage.DIAGNOSIS_DECISION:
            obs["visible_info"] = {
                "symptoms": case_data["symptoms"],
                "possible_diseases": case_data["possible_diseases"],
                "test_results": case_data["tests"],
                "radiologist_report": self._get_stage_output(Stage.TEST_ANALYSIS),
                "pathologist_report": self._get_stage_output(Stage.TEST_ANALYSIS),
                "gp_assessment": self._get_stage_output(Stage.INITIAL_ASSESSMENT),
            }

        # Stage 4: Supervisor sees everything
        elif stage == Stage.SUPERVISOR_REVIEW:
            obs["visible_info"] = {
                "full_case": case_data,
                "proposed_diagnosis": self.final_diagnosis,
                "proposed_confidence": self.final_confidence,
                "all_stage_outputs": {
                    s.name: self.stage_outputs[s] for s in Stage if s != Stage.DONE
                },
            }

        # Stage 5: Oversight sees aggregate view
        elif stage == Stage.OVERSIGHT_FEEDBACK:
            obs["visible_info"] = {
                "case_id": self.case_id,
                "final_diagnosis": self.final_diagnosis,
                "approved": self.approved,
                "action_log": self.action_log,
            }

        return obs

    def _get_stage_output(self, stage: Stage) -> List[Dict]:
        return self.stage_outputs.get(stage, [])

    def _get_ordered_tests(self) -> List[str]:
        """Extract test names from TEST_RECOMMENDATION stage outputs."""
        tests = []
        for entry in self.stage_outputs.get(Stage.TEST_RECOMMENDATION, []):
            if entry["action_type"] == "order_test":
                test_name = entry["content"].get("test_name", "")
                if test_name:
                    tests.append(test_name)
        return tests

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the completed pipeline for reward computation."""
        return {
            "case_id": self.case_id,
            "total_steps": len(self.action_log),
            "stages_completed": [
                s.name for s in Stage
                if self.stage_action_counts[s] > 0
            ],
            "final_diagnosis": self.final_diagnosis,
            "final_confidence": self.final_confidence,
            "approved": self.approved,
            "tests_ordered": self._get_ordered_tests(),
            "reasoning_steps": [
                e["content"].get("reasoning", "")
                for e in self.action_log
                if e["content"].get("reasoning")
            ],
            "action_log": self.action_log,
        }
