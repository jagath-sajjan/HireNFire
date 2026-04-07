"""
Typed models for the HireNFire OpenEnv environment.

The environment simulates a structured hiring funnel where an agent reviews
resumes, interviews candidates, and builds a balanced hiring cohort.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Actions available to the hiring agent."""

    RANK = "RANK"
    INTERVIEW = "INTERVIEW"
    HIRE = "HIRE"
    REJECT = "REJECT"
    FINALIZE = "FINALIZE"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Candidate(BaseModel):
    """Full internal candidate record used for grading and state snapshots."""

    id: int
    name: str
    years_experience: int = Field(ge=0, le=25)
    skills: list[str] = Field(default_factory=list)
    education: str = Field(default="bachelors")
    demographic_group: str = Field(default="A")
    certifications: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    resume_summary: str = ""
    expected_salary_band: str = "mid"
    interview_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    resume_score: float = Field(default=0.0, ge=0.0, le=1.0)
    potential_score: float = Field(default=0.0, ge=0.0, le=1.0)
    calibration_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    ground_truth_score: float = Field(ge=0.0, le=1.0)
    archetype: str = ""


class TaskConfig(BaseModel):
    """Configuration for a single hiring task / episode."""

    task_id: str
    task_name: str
    role: str
    required_skills: list[str]
    difficulty: Difficulty
    num_candidates: int = Field(ge=5, le=100)
    num_to_hire: int = Field(ge=1, le=20)
    alpha: float = Field(ge=0.0, le=1.0, description="Quality weight")
    max_steps: int = Field(default=20, ge=5)
    description: str = ""
    group_distribution: dict[str, float] = Field(default_factory=dict)
    interview_budget_hint: int = Field(default=0, ge=0)
    screening_focus: str = ""
    target_mix_strategy: str = ""
    benchmark: str = "hirenfire"


class CandidateView(BaseModel):
    """Agent-visible candidate card."""

    id: int
    name: str
    years_experience: int
    skills: list[str]
    education: str
    demographic_group: str
    certifications: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    resume_summary: str = ""
    expected_salary_band: str = "mid"
    interview_score: Optional[float] = None


class DecisionRecord(BaseModel):
    """Structured log entry for prior agent actions."""

    step: int = Field(ge=1)
    action: str
    candidate_id: Optional[int] = None
    reward: float = Field(ge=0.0, le=1.0)
    done: bool = False
    info: dict = Field(default_factory=dict)


class Observation(BaseModel):
    """Agent-visible observation returned from reset() and step()."""

    candidates: list[CandidateView]
    current_step: int
    task_config: TaskConfig
    decisions: list[DecisionRecord] = Field(default_factory=list)
    remaining_slots: int
    remaining_candidates: int
    interview_results: dict[int, float] = Field(default_factory=dict)
    hired_ids: list[int] = Field(default_factory=list)
    rejected_ids: list[int] = Field(default_factory=list)
    fairness_targets: dict[str, int] = Field(default_factory=dict)
    recent_event: str = ""


class Action(BaseModel):
    """Action payload accepted by the environment."""

    action_type: ActionType
    candidate_id: Optional[int] = None
    reasoning: str = ""


class RewardInfo(BaseModel):
    """Reward payload returned by step()."""

    quality_score: float = Field(ge=0.0, le=1.0)
    fairness_score: float = Field(ge=0.0, le=1.0)
    combined_reward: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(ge=0.0, le=1.0)
    breakdown: dict = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Full state snapshot returned by state()."""

    task_config: TaskConfig
    candidates: list[Candidate]
    hired_ids: list[int] = Field(default_factory=list)
    rejected_ids: list[int] = Field(default_factory=list)
    interview_results: dict[int, float] = Field(default_factory=dict)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    current_step: int = Field(default=0, ge=0)
    done: bool = False
    invalid_actions: int = Field(default=0, ge=0)
    final_reward: Optional[RewardInfo] = None
