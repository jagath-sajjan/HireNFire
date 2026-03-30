"""
Task definitions for the HireNFire environment.

Three tasks of increasing difficulty:
  - Easy:   Simple ranking of 10 candidates (α=0.7)
  - Medium: High-potential screening of 25 candidates (α=0.5)
  - Hard:   Fair hiring at scale with 50 candidates (α=0.3)
"""

from hirenfire.models import TaskConfig, Difficulty


EASY_TASK = TaskConfig(
    task_id="easy",
    task_name="Simple Ranking",
    role="Software Engineer",
    required_skills=["Python", "JavaScript", "Docker", "SQL", "Git", "REST APIs"],
    difficulty=Difficulty.EASY,
    num_candidates=10,
    num_to_hire=3,
    alpha=0.7,
    max_steps=25,
    description=(
        "Rank 10 candidates by explicit criteria for a Software Engineer role. "
        "Clear skill gaps make the correct ranking obvious. "
        "Quality-weighted reward (α=0.7)."
    ),
)

MEDIUM_TASK = TaskConfig(
    task_id="medium",
    task_name="High-Potential Screening",
    role="Data Scientist",
    required_skills=[
        "Python", "SQL", "Statistics", "Scikit-learn", "Pandas",
        "Feature Engineering", "A/B Testing", "Deep Learning",
    ],
    difficulty=Difficulty.MEDIUM,
    num_candidates=25,
    num_to_hire=5,
    alpha=0.5,
    max_steps=40,
    description=(
        "Screen 25 candidates for a Data Scientist role. Some candidates are "
        "underqualified on paper but show high potential. Agent must balance "
        "strict criteria with potential. Balanced reward (α=0.5)."
    ),
)

HARD_TASK = TaskConfig(
    task_id="hard",
    task_name="Fair Hiring at Scale",
    role="Senior ML Engineer",
    required_skills=[
        "Python", "PyTorch", "MLOps", "Kubernetes", "Docker",
        "Model Serving", "Distributed Training", "System Design",
        "Data Pipelines", "Model Monitoring",
    ],
    difficulty=Difficulty.HARD,
    num_candidates=50,
    num_to_hire=8,
    alpha=0.3,
    max_steps=60,
    description=(
        "Hire from a pool of 50 candidates for a Senior ML Engineer role. "
        "The candidate pool has demographic skew and tight qualification margins. "
        "Maintaining demographic parity while hiring quality candidates is "
        "genuinely difficult. Fairness-weighted reward (α=0.3)."
    ),
)

ALL_TASKS = [EASY_TASK, MEDIUM_TASK, HARD_TASK]
