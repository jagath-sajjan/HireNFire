"""
Pydantic models for the HireNFire OpenEnv environment.

Defines typed Observation, Action, and Reward models per the OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Available actions the agent can take."""
    RANK = "RANK"
    INTERVIEW = "INTERVIEW"
    HIRE = "HIRE"
    REJECT = "REJECT"
    FINALIZE = "FINALIZE"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── Core Data Models ─────────────────────────────────────────────────────

class Candidate(BaseModel):
    """A single candidate in the hiring pool."""
    id: int
    name: str
    years_experience: int = Field(ge=0, le=30)
    skills: list[str]
    education: str  # "high_school", "bachelors", "masters", "phd"
    demographic_group: str  # abstract labels: "A", "B", "C", "D"
    interview_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Revealed only after an INTERVIEW action"
    )
    # Hidden from agent — used by grader only
    ground_truth_score: float = Field(
        ge=0.0, le=1.0,
        description="True quality score, hidden from the agent"
    )


class TaskConfig(BaseModel):
    """Configuration for a single task / episode."""
    task_id: str
    task_name: str
    role: str
    required_skills: list[str]
    difficulty: Difficulty
    num_candidates: int = Field(ge=5, le=100)
    num_to_hire: int = Field(ge=1, le=20)
    alpha: float = Field(ge=0.0, le=1.0, description="Quality–fairness weight")
    max_steps: int = Field(default=50, ge=5)
    description: str = ""


# ── Observation ──────────────────────────────────────────────────────────

class CandidateView(BaseModel):
    """Agent-visible view of a candidate (no ground_truth_score)."""
    id: int
    name: str
    years_experience: int
    skills: list[str]
    education: str
    demographic_group: str
    interview_score: Optional[float] = None


class Observation(BaseModel):
    """What the agent sees at each step."""
    candidates: list[CandidateView]
    current_step: int
    task_config: TaskConfig
    decisions: list[dict]  # log of prior actions
    remaining_slots: int
    interview_results: dict[int, float] = Field(default_factory=dict)
    hired_ids: list[int] = Field(default_factory=list)
    rejected_ids: list[int] = Field(default_factory=list)


# ── Action ───────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Action taken by the agent."""
    action_type: ActionType
    candidate_id: Optional[int] = None
    reasoning: str = ""


# ── Reward ───────────────────────────────────────────────────────────────

class RewardInfo(BaseModel):
    """Detailed reward breakdown."""
    quality_score: float = Field(ge=0.0, le=1.0)
    fairness_score: float = Field(ge=0.0, le=1.0)
    combined_reward: float = Field(ge=0.0, le=1.0)
    alpha: float
    breakdown: dict = Field(default_factory=dict)
