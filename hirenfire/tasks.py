"""
Task definitions for the HireNFire environment.

The tasks progress from straightforward resume screening to harder cases where
interviews and cohort balance matter materially.
"""

from hirenfire.models import Difficulty, TaskConfig


EASY_TASK = TaskConfig(
    task_id="easy",
    task_name="simple-screen",
    role="Software Engineer",
    required_skills=["Python", "JavaScript", "Docker", "SQL", "Git", "REST APIs"],
    difficulty=Difficulty.EASY,
    num_candidates=10,
    num_to_hire=3,
    alpha=0.72,
    max_steps=14,
    group_distribution={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    interview_budget_hint=4,
    screening_focus="Clear skills-based screening with a few borderline resumes.",
    target_mix_strategy="Match pool representation while prioritizing role fit.",
    description=(
        "Fill three software engineering seats from a balanced pool. The strongest "
        "resumes are mostly obvious, but interviewing a few borderline candidates "
        "still helps avoid premature rejection."
    ),
)

MEDIUM_TASK = TaskConfig(
    task_id="medium",
    task_name="potential-calibration",
    role="Data Scientist",
    required_skills=[
        "Python",
        "SQL",
        "Statistics",
        "Scikit-learn",
        "Pandas",
        "Feature Engineering",
        "A/B Testing",
        "Deep Learning",
    ],
    difficulty=Difficulty.MEDIUM,
    num_candidates=25,
    num_to_hire=5,
    alpha=0.55,
    max_steps=20,
    group_distribution={"A": 0.36, "B": 0.28, "C": 0.22, "D": 0.14},
    interview_budget_hint=7,
    screening_focus="Separate polished resumes from high-upside candidates.",
    target_mix_strategy="Hit role quality while keeping cohort counts close to pool mix.",
    description=(
        "Hire a balanced data science cohort where some of the best candidates are "
        "career switchers or under-credentialed applicants who outperform in "
        "structured interviews."
    ),
)

HARD_TASK = TaskConfig(
    task_id="hard",
    task_name="fair-panel-hiring",
    role="Senior ML Engineer",
    required_skills=[
        "Python",
        "PyTorch",
        "MLOps",
        "Kubernetes",
        "Docker",
        "Model Serving",
        "Distributed Training",
        "System Design",
        "Data Pipelines",
        "Model Monitoring",
    ],
    difficulty=Difficulty.HARD,
    num_candidates=50,
    num_to_hire=8,
    alpha=0.38,
    max_steps=28,
    group_distribution={"A": 0.44, "B": 0.28, "C": 0.18, "D": 0.10},
    interview_budget_hint=10,
    screening_focus="Use interviews to correct resume bias and build a fair senior cohort.",
    target_mix_strategy="Balance final cohort against pipeline mix without collapsing quality.",
    description=(
        "Screen a large, skewed senior ML pipeline. Resume quality is noisy, several "
        "underrepresented candidates interview much better than their paper profile, "
        "and over-indexing on polished resumes creates an immediate fairness penalty."
    ),
)


ALL_TASKS = [EASY_TASK, MEDIUM_TASK, HARD_TASK]
TASK_BY_ID = {task.task_id: task for task in ALL_TASKS}
