"""
HireNFire — structured hiring benchmark for OpenEnv.
"""

from hirenfire.models import (
    Action,
    ActionType,
    Candidate,
    EnvironmentState,
    Observation,
    RewardInfo,
    TaskConfig,
)
from hirenfire.env import HiringEnv
from hirenfire.tasks import ALL_TASKS, EASY_TASK, HARD_TASK, MEDIUM_TASK, TASK_BY_ID

__version__ = "1.1.0"
__all__ = [
    "Candidate",
    "TaskConfig",
    "Observation",
    "EnvironmentState",
    "Action",
    "ActionType",
    "RewardInfo",
    "HiringEnv",
    "EASY_TASK", "MEDIUM_TASK", "HARD_TASK", "ALL_TASKS",
    "TASK_BY_ID",
]
