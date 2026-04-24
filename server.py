"""
server.py — MedAgents-X FastAPI Server
========================================
Exposes the MedAgentsEnv as an HTTP server following the OpenEnv interface.
Required by OpenEnv validator and HuggingFace Spaces deployment.

Endpoints:
    POST /reset         → start new episode, returns initial state
    POST /step          → execute one action, returns next_state + reward + done + info
    GET  /state         → get current state without acting
    GET  /health        → health check
    GET  /info          → environment metadata

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 7860 --reload

OpenEnv validator connects to:
    http://localhost:7860/reset
    http://localhost:7860/step
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import traceback

from environment import MedAgentsEnv, MedState
from task import Action
from memory import MemorySystem
from agents.gp import GPAgent
from agents.radiologist import RadiologistAgent
from agents.pathologist import PathologistAgent
from agents.specialist import SpecialistAgent
from agents.supervisor import SupervisorAgent
from agents.oversight import OversightAgent


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedAgents-X",
    description="Multi-Agent Clinical Decision RL Environment — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global Environment (single instance for server) ─────────────────────────

_memory = MemorySystem()
_env: Optional[MedAgentsEnv] = None
_agents: Optional[Dict] = None
_current_state: Optional[MedState] = None


def _get_env() -> MedAgentsEnv:
    global _env, _agents, _memory
    if _env is None:
        _memory = MemorySystem()
        _env = MedAgentsEnv(
            shuffle=True,
            max_steps_per_episode=25,
            memory_system=_memory,
            seed=42,
        )
        _agents = {
            "gp":          GPAgent(memory=_memory.get_agent_memory("gp"), noise_level=0.35),
            "radiologist": RadiologistAgent(memory=_memory.get_agent_memory("radiologist"), noise_level=0.35),
            "pathologist": PathologistAgent(memory=_memory.get_agent_memory("pathologist"), noise_level=0.35),
            "specialist":  SpecialistAgent(memory=_memory.get_agent_memory("specialist"), noise_level=0.35),
            "supervisor":  SupervisorAgent(memory=_memory.get_agent_memory("supervisor"), noise_level=0.35),
            "oversight":   OversightAgent(memory=_memory),
        }
    return _env


# ─── Request / Response Schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Optional seed for reproducibility")


class StepRequest(BaseModel):
    agent_role: str = Field(..., description="Agent role: gp, radiologist, pathologist, specialist, supervisor, oversight")
    action_type: str = Field(..., description="Action type valid for the current stage")
    content: Dict[str, Any] = Field(default_factory=dict, description="Action payload")


class AutoStepRequest(BaseModel):
    """Run one full stage automatically using the built-in agents."""
    pass


class StateResponse(BaseModel):
    case_id: int
    stage: str
    step_count: int
    episode_done: bool
    cumulative_reward: float
    observation: Dict[str, Any]
    active_agents: List[str]
    valid_actions: List[str]


class StepResponse(BaseModel):
    state: StateResponse
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    state: StateResponse
    message: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _state_to_response(state: MedState, env: MedAgentsEnv) -> StateResponse:
    return StateResponse(
        case_id=state.case_id,
        stage=state.stage,
        step_count=state.step_count,
        episode_done=state.episode_done,
        cumulative_reward=round(state.cumulative_reward, 4),
        observation=state.observation,
        active_agents=env.get_active_agents(),
        valid_actions=env.get_valid_actions(),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check — OpenEnv validator calls this first."""
    return {"status": "ok", "environment": "MedAgents-X", "version": "1.0.0"}


@app.get("/info")
def env_info():
    """Return environment metadata."""
    env = _get_env()
    return {
        "name": "MedAgents-X",
        "description": "Multi-Agent Clinical Decision System",
        "n_cases": env.n_cases,
        "agents": ["gp", "radiologist", "pathologist", "specialist", "supervisor", "oversight"],
        "stages": [
            "INITIAL_ASSESSMENT",
            "TEST_RECOMMENDATION",
            "TEST_ANALYSIS",
            "DIAGNOSIS_DECISION",
            "SUPERVISOR_REVIEW",
            "OVERSIGHT_FEEDBACK",
        ],
        "action_space": "discrete per stage",
        "observation_space": "partial — revealed progressively per stage",
        "reward_range": [-1.0, 2.5],
        "max_steps": 25,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    """
    Start a new episode. Returns the initial state with only symptoms visible.
    Called by OpenEnv validator at the start of each evaluation episode.
    """
    global _current_state
    try:
        env = _get_env()
        if request and request.seed is not None:
            import random
            random.seed(request.seed)

        state = env.reset()
        _current_state = state

        return ResetResponse(
            state=_state_to_response(state, env),
            message=f"New episode started. Case #{state.case_id}. Stage: {state.stage}.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Execute one agent action. Returns next state, reward, done flag, and info.
    Called by OpenEnv validator for each action in an episode.
    """
    global _current_state
    env = _get_env()

    if _current_state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    if _current_state.episode_done:
        raise HTTPException(status_code=400, detail="Episode is already done. Call /reset to start a new one.")

    try:
        action = Action(
            agent_role=request.agent_role,
            action_type=request.action_type,
            content=request.content,
        )
        next_state, reward, done, info = env.step(action)
        _current_state = next_state

        return StepResponse(
            state=_state_to_response(next_state, env),
            reward=round(reward, 4),
            done=done,
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/auto_step")
def auto_step():
    """
    Automatically run one full stage using the built-in agents.
    Useful for demos and quick testing without manually crafting actions.
    """
    global _current_state
    env = _get_env()

    if _current_state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    if _current_state.episode_done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset.")

    try:
        active = env.get_active_agents()
        results = []
        done = False

        for role in active:
            agent = _agents.get(role)
            if not agent:
                continue
            actions = agent.act(_current_state.observation)
            for action in actions:
                next_state, reward, done, info = env.step(action)
                _current_state = next_state
                results.append({
                    "agent": role,
                    "action_type": action.action_type,
                    "reward": round(reward, 4),
                    "content_summary": {
                        k: v for k, v in action.content.items()
                        if k in ("diagnosis", "confidence", "reasoning", "test_name", "finding")
                    },
                })
                if done:
                    break
            if done:
                break

        return {
            "actions_taken": results,
            "state": _state_to_response(_current_state, env),
            "done": done,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-step failed: {str(e)}\n{traceback.format_exc()}")


@app.get("/state", response_model=StateResponse)
def get_state():
    """Get current state without taking any action."""
    env = _get_env()
    if _current_state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _state_to_response(_current_state, env)


@app.post("/run_episode")
def run_full_episode():
    """
    Run one complete episode automatically from reset to done.
    Returns full trajectory with all actions, rewards, and final diagnosis.
    Used for demos and evaluation.
    """
    global _current_state
    env = _get_env()

    try:
        state = env.reset()
        _current_state = state
        trajectory = []
        done = False
        step_count = 0

        while not done and step_count < 30:
            active = env.get_active_agents()
            for role in active:
                agent = _agents.get(role)
                if not agent:
                    continue
                actions = agent.act(_current_state.observation)
                for action in actions:
                    next_state, reward, done, info = env.step(action)
                    _current_state = next_state
                    trajectory.append({
                        "step": step_count + 1,
                        "stage": state.stage,
                        "agent": role,
                        "action_type": action.action_type,
                        "reward": round(reward, 4),
                    })
                    step_count += 1
                    if done:
                        break
                if done:
                    break

        pipeline = env._pipeline
        summary = pipeline.get_summary() if pipeline else {}

        return {
            "case_id": _current_state.case_id,
            "final_diagnosis": summary.get("final_diagnosis", ""),
            "final_confidence": summary.get("final_confidence", 0.0),
            "total_steps": step_count,
            "total_reward": round(_current_state.cumulative_reward, 4),
            "trajectory": trajectory,
            "approved": summary.get("approved", False),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Episode run failed: {str(e)}\n{traceback.format_exc()}")
