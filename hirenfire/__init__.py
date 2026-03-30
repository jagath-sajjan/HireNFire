"""
HireNFire — Bias-Aware Hiring Screener OpenEnv Environment

An AI agent screens resumes, conducts structured interviews, and makes
hiring decisions — graded on both quality of hire AND demographic fairness.

reward = α · quality + (1 − α) · fairness
"""

from hirenfire.models import Candidate, TaskConfig, Observation, Action, ActionType, RewardInfo
from hirenfire.env import HiringEnv
from hirenfire.tasks import EASY_TASK, MEDIUM_TASK, HARD_TASK, ALL_TASKS

__version__ = "1.0.0"
__all__ = [
    "Candidate", "TaskConfig", "Observation", "Action", "ActionType", "RewardInfo",
    "HiringEnv",
    "EASY_TASK", "MEDIUM_TASK", "HARD_TASK", "ALL_TASKS",
]
