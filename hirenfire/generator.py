"""
Synthetic candidate data generation for HireNFire.

Generates realistic candidate pools with controlled demographic distributions
and ground-truth quality scores for grading.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from hirenfire.models import Candidate, TaskConfig, Difficulty


# ── Name pools ───────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Parker", "Sage", "Reese", "Hayden", "Drew", "Blake", "Emerson", "Dakota",
    "Finley", "Phoenix", "Rowan", "Skyler", "Cameron", "Charlie", "Oakley",
    "Remy", "Ellis", "Lennox", "Arden", "Kai", "Shiloh", "Marlowe",
    "River", "Wren", "Indigo", "Harley", "Eden", "Gray", "Noel", "Lane",
    "Sterling", "Harbor", "Bellamy", "Briar", "Ever", "Sutton", "Tatum",
    "Winter", "Frankie", "Landry", "Reign", "Aiden",
]

LAST_NAMES = [
    "Smith", "Chen", "Patel", "Kim", "Garcia", "Nguyen", "Williams", "Brown",
    "Jones", "Miller", "Davis", "Rodriguez", "Martinez", "Lopez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Hill", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner",
]

# ── Skill pools per role ─────────────────────────────────────────────────

SKILL_POOLS = {
    "Software Engineer": [
        "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++",
        "React", "Node.js", "Docker", "Kubernetes", "AWS", "SQL", "NoSQL",
        "Git", "CI/CD", "REST APIs", "GraphQL", "System Design", "TDD",
    ],
    "Data Scientist": [
        "Python", "R", "SQL", "TensorFlow", "PyTorch", "Scikit-learn",
        "Pandas", "NumPy", "Statistics", "A/B Testing", "NLP", "Computer Vision",
        "Feature Engineering", "Data Visualization", "Spark", "Hadoop",
        "Bayesian Methods", "Time Series", "Deep Learning", "MLOps",
    ],
    "Senior ML Engineer": [
        "Python", "PyTorch", "TensorFlow", "MLOps", "Kubernetes", "Docker",
        "AWS SageMaker", "Model Serving", "Feature Stores", "Data Pipelines",
        "Distributed Training", "Model Monitoring", "A/B Testing", "CI/CD",
        "Spark", "Ray", "ONNX", "Triton", "System Design", "Mentoring",
    ],
}

EDUCATION_LEVELS = ["high_school", "bachelors", "masters", "phd"]
EDUCATION_WEIGHTS = {
    "high_school": 0.1,
    "bachelors": 0.4,
    "masters": 0.7,
    "phd": 0.9,
}

DEMOGRAPHIC_GROUPS = ["A", "B", "C", "D"]


def _compute_ground_truth(
    candidate_skills: list[str],
    required_skills: list[str],
    years_exp: int,
    education: str,
    noise: float = 0.05,
) -> float:
    """Compute ground-truth quality score for a candidate."""
    # Skill match ratio
    matched = len(set(candidate_skills) & set(required_skills))
    skill_score = matched / max(len(required_skills), 1)

    # Experience score (0-1, saturates around 10 years)
    exp_score = min(years_exp / 10.0, 1.0)

    # Education score
    edu_score = EDUCATION_WEIGHTS.get(education, 0.3)

    # Weighted combination
    raw = 0.50 * skill_score + 0.30 * exp_score + 0.20 * edu_score

    # Add small noise
    noisy = raw + np.random.normal(0, noise)
    return float(np.clip(noisy, 0.0, 1.0))


def generate_candidates(
    task_config: TaskConfig,
    seed: Optional[int] = None,
) -> list[Candidate]:
    """
    Generate a pool of synthetic candidates for the given task.

    Difficulty controls:
    - EASY: Clear skill gaps, balanced demographics
    - MEDIUM: Some ambiguous candidates, slight demographic imbalance
    - HARD: Tight margins, significant demographic skew in the pool
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    role = task_config.role
    required = task_config.required_skills
    n = task_config.num_candidates
    difficulty = task_config.difficulty
    all_skills = SKILL_POOLS.get(role, SKILL_POOLS["Software Engineer"])

    # Demographic distribution depends on difficulty
    if difficulty == Difficulty.EASY:
        # Balanced groups
        demo_weights = [0.25, 0.25, 0.25, 0.25]
    elif difficulty == Difficulty.MEDIUM:
        # Slight imbalance
        demo_weights = [0.35, 0.30, 0.20, 0.15]
    else:
        # Significant skew — group A overrepresented
        demo_weights = [0.45, 0.30, 0.15, 0.10]

    candidates = []
    for i in range(n):
        # Name
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"

        # Demographics
        group = np.random.choice(DEMOGRAPHIC_GROUPS, p=demo_weights)

        # Education
        education = random.choice(EDUCATION_LEVELS)

        # Experience (skewed by difficulty)
        if difficulty == Difficulty.EASY:
            years = random.randint(0, 15)
        elif difficulty == Difficulty.MEDIUM:
            years = random.randint(0, 12)
        else:
            # Hard: tighter range → less obvious
            years = random.randint(2, 10)

        # Skills — number of skills varies by difficulty
        if difficulty == Difficulty.EASY:
            num_skills = random.randint(2, 8)
        elif difficulty == Difficulty.MEDIUM:
            num_skills = random.randint(3, 7)
        else:
            num_skills = random.randint(4, 7)  # tighter range

        skills = random.sample(all_skills, min(num_skills, len(all_skills)))

        # Ground truth
        noise = {Difficulty.EASY: 0.03, Difficulty.MEDIUM: 0.06, Difficulty.HARD: 0.10}
        gt_score = _compute_ground_truth(
            skills, required, years, education,
            noise=noise[difficulty],
        )

        # For hard difficulty, inject bias: slightly boost group A scores
        # This makes fairness genuinely harder
        if difficulty == Difficulty.HARD and group == "A":
            gt_score = min(gt_score + 0.08, 1.0)

        candidates.append(Candidate(
            id=i,
            name=name,
            years_experience=years,
            skills=skills,
            education=education,
            demographic_group=group,
            interview_score=None,
            ground_truth_score=round(gt_score, 4),
        ))

    return candidates
