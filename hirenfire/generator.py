"""
Deterministic candidate generation for HireNFire.

The generator intentionally mixes strong resumes, hidden-upside candidates, and
polished-but-overrated applicants so that interviews provide real signal.
"""

from __future__ import annotations

import random
from typing import Optional

from hirenfire.models import Candidate, Difficulty, TaskConfig


FIRST_NAMES = [
    "Alex",
    "Jordan",
    "Taylor",
    "Morgan",
    "Casey",
    "Riley",
    "Avery",
    "Quinn",
    "Parker",
    "Sage",
    "Reese",
    "Hayden",
    "Drew",
    "Blake",
    "Emerson",
    "Dakota",
    "Finley",
    "Phoenix",
    "Rowan",
    "Skyler",
    "Cameron",
    "Charlie",
    "Ellis",
    "Kai",
    "River",
    "Noel",
]

LAST_NAMES = [
    "Smith",
    "Chen",
    "Patel",
    "Kim",
    "Garcia",
    "Nguyen",
    "Williams",
    "Brown",
    "Jones",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Lopez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Lee",
    "Thompson",
    "White",
    "Harris",
    "Clark",
    "Lewis",
]

SKILL_POOLS = {
    "Software Engineer": [
        "Python",
        "JavaScript",
        "TypeScript",
        "Go",
        "Rust",
        "Java",
        "C++",
        "React",
        "Node.js",
        "Docker",
        "Kubernetes",
        "AWS",
        "SQL",
        "NoSQL",
        "Git",
        "CI/CD",
        "REST APIs",
        "GraphQL",
        "System Design",
        "TDD",
    ],
    "Data Scientist": [
        "Python",
        "R",
        "SQL",
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
        "Pandas",
        "NumPy",
        "Statistics",
        "A/B Testing",
        "NLP",
        "Computer Vision",
        "Feature Engineering",
        "Data Visualization",
        "Spark",
        "Hadoop",
        "Bayesian Methods",
        "Time Series",
        "Deep Learning",
        "MLOps",
    ],
    "Senior ML Engineer": [
        "Python",
        "PyTorch",
        "TensorFlow",
        "MLOps",
        "Kubernetes",
        "Docker",
        "AWS SageMaker",
        "Model Serving",
        "Feature Stores",
        "Data Pipelines",
        "Distributed Training",
        "Model Monitoring",
        "A/B Testing",
        "CI/CD",
        "Spark",
        "Ray",
        "ONNX",
        "Triton",
        "System Design",
        "Mentoring",
    ],
}

CERTIFICATION_POOLS = {
    "Software Engineer": ["AWS Certified Developer", "CKA", "Azure Developer Associate"],
    "Data Scientist": ["TensorFlow Developer", "Databricks Associate", "AWS ML Specialty"],
    "Senior ML Engineer": ["AWS ML Specialty", "CKA", "Databricks Professional"],
}

EDUCATION_LEVELS = ["high_school", "bachelors", "masters", "phd"]
EDUCATION_WEIGHTS = {
    "high_school": 0.18,
    "bachelors": 0.52,
    "masters": 0.76,
    "phd": 0.92,
}

DEMOGRAPHIC_GROUPS = ["A", "B", "C", "D"]

ARCHETYPE_PROFILES = {
    "clear_match": {
        "resume_shift": 0.12,
        "potential_shift": 0.06,
        "interview_shift": 0.04,
        "experience": (5, 11),
        "strengths": ["Direct role fit", "Strong production evidence"],
        "concerns": ["Limited leadership scope"],
    },
    "steady_operator": {
        "resume_shift": 0.04,
        "potential_shift": 0.02,
        "interview_shift": 0.00,
        "experience": (4, 9),
        "strengths": ["Consistent execution", "Reliable delivery history"],
        "concerns": ["Not obviously differentiated"],
    },
    "latent_gem": {
        "resume_shift": -0.06,
        "potential_shift": 0.18,
        "interview_shift": 0.16,
        "experience": (2, 8),
        "strengths": ["High-upside portfolio", "Strong structured interview upside"],
        "concerns": ["Resume undersells depth"],
    },
    "polished_resume": {
        "resume_shift": 0.10,
        "potential_shift": -0.08,
        "interview_shift": -0.16,
        "experience": (5, 10),
        "strengths": ["Polished storytelling", "Strong keyword match"],
        "concerns": ["Interview calibration risk"],
    },
    "specialist": {
        "resume_shift": 0.08,
        "potential_shift": 0.04,
        "interview_shift": 0.02,
        "experience": (6, 12),
        "strengths": ["Deep expertise in a narrow area", "Strong domain ownership"],
        "concerns": ["May be narrow outside specialty"],
    },
    "stretch_candidate": {
        "resume_shift": -0.14,
        "potential_shift": -0.02,
        "interview_shift": -0.10,
        "experience": (1, 5),
        "strengths": ["Motivated learner"],
        "concerns": ["Needs more role depth"],
    },
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _education_for_archetype(archetype: str, rng: random.Random) -> str:
    weights = {
        "clear_match": [0.06, 0.48, 0.34, 0.12],
        "steady_operator": [0.08, 0.52, 0.30, 0.10],
        "latent_gem": [0.10, 0.50, 0.28, 0.12],
        "polished_resume": [0.02, 0.40, 0.38, 0.20],
        "specialist": [0.00, 0.36, 0.42, 0.22],
        "stretch_candidate": [0.18, 0.54, 0.22, 0.06],
    }[archetype]
    return rng.choices(EDUCATION_LEVELS, weights=weights, k=1)[0]


def _choose_group(task_config: TaskConfig, rng: random.Random) -> str:
    distribution = task_config.group_distribution or {group: 0.25 for group in DEMOGRAPHIC_GROUPS}
    groups = list(distribution.keys())
    weights = [distribution[group] for group in groups]
    return rng.choices(groups, weights=weights, k=1)[0]


def _choose_archetype(
    task_config: TaskConfig,
    group: str,
    rng: random.Random,
) -> str:
    if task_config.difficulty == Difficulty.EASY:
        names = ["clear_match", "steady_operator", "specialist", "stretch_candidate", "polished_resume"]
        weights = [0.34, 0.26, 0.18, 0.14, 0.08]
    elif task_config.difficulty == Difficulty.MEDIUM:
        names = ["clear_match", "steady_operator", "latent_gem", "specialist", "polished_resume", "stretch_candidate"]
        weights = [0.22, 0.18, 0.24, 0.16, 0.12, 0.08]
    elif group in {"C", "D"}:
        names = ["latent_gem", "specialist", "steady_operator", "clear_match", "stretch_candidate", "polished_resume"]
        weights = [0.34, 0.20, 0.14, 0.14, 0.10, 0.08]
    else:
        names = ["polished_resume", "clear_match", "steady_operator", "specialist", "latent_gem", "stretch_candidate"]
        weights = [0.26, 0.22, 0.18, 0.16, 0.10, 0.08]
    return rng.choices(names, weights=weights, k=1)[0]


def _base_skill_ratio(archetype: str) -> float:
    return {
        "clear_match": 0.82,
        "steady_operator": 0.68,
        "latent_gem": 0.56,
        "polished_resume": 0.76,
        "specialist": 0.78,
        "stretch_candidate": 0.42,
    }[archetype]


def _sample_skills(
    role: str,
    required_skills: list[str],
    archetype: str,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    all_skills = SKILL_POOLS.get(role, SKILL_POOLS["Software Engineer"])
    optional_skills = [skill for skill in all_skills if skill not in required_skills]
    base_ratio = _base_skill_ratio(archetype)
    matched_count = max(1, min(len(required_skills), round(base_ratio * len(required_skills) + rng.randint(-1, 1))))
    extra_count = max(1, min(4, rng.randint(1, 3) + (1 if archetype in {"clear_match", "specialist"} else 0)))

    matched = rng.sample(required_skills, k=matched_count)
    extras = rng.sample(optional_skills, k=min(len(optional_skills), extra_count))
    full_skill_set = list(dict.fromkeys(matched + extras))

    missing = [skill for skill in required_skills if skill not in full_skill_set]
    return full_skill_set, missing


def _build_strengths(role: str, skills: list[str], archetype: str, missing: list[str]) -> list[str]:
    profile = ARCHETYPE_PROFILES[archetype]
    strengths = list(profile["strengths"])
    if skills:
        strengths.append(f"Hands-on with {', '.join(skills[:2])}")
    if role == "Senior ML Engineer" and "System Design" in skills:
        strengths.append("Shows architecture ownership")
    if role == "Data Scientist" and any(skill in skills for skill in ["Statistics", "A/B Testing", "Feature Engineering"]):
        strengths.append("Strong experimentation mindset")
    return strengths[:3]


def _build_concerns(archetype: str, missing: list[str], years_experience: int) -> list[str]:
    profile = ARCHETYPE_PROFILES[archetype]
    concerns = list(profile["concerns"])
    if years_experience <= 3:
        concerns.append("Limited years in comparable scope")
    if missing:
        concerns.append(f"Skill gaps: {', '.join(missing[:2])}")
    return concerns[:3]


def _resume_summary(
    role: str,
    archetype: str,
    skills: list[str],
    strengths: list[str],
    concerns: list[str],
) -> str:
    lead = {
        "clear_match": "Directly aligned resume with shipped work in the target stack.",
        "steady_operator": "Solid generalist profile with consistent delivery signals.",
        "latent_gem": "Non-obvious profile with better upside than the paper resume suggests.",
        "polished_resume": "Well-packaged resume that checks many surface-level boxes.",
        "specialist": "Deep specialist background with stronger domain depth than breadth.",
        "stretch_candidate": "Early-career profile that may need ramp time in-role.",
    }[archetype]
    tail = f"Key skills: {', '.join(skills[:3])}."
    if strengths:
        tail += f" Strongest visible signal: {strengths[0]}."
    if concerns:
        tail += f" Main concern: {concerns[0]}."
    return f"{role}: {lead} {tail}"


def _certifications(role: str, archetype: str, rng: random.Random) -> list[str]:
    if archetype in {"stretch_candidate", "steady_operator"}:
        max_count = 1
    else:
        max_count = 2
    pool = CERTIFICATION_POOLS.get(role, [])
    count = rng.randint(0, min(max_count, len(pool)))
    return rng.sample(pool, k=count)


def _salary_band(years_experience: int) -> str:
    if years_experience >= 9:
        return "senior"
    if years_experience >= 4:
        return "mid"
    return "junior"


def _compute_resume_score(
    skills: list[str],
    required_skills: list[str],
    years_experience: int,
    education: str,
    certifications: list[str],
    archetype: str,
    difficulty: Difficulty,
    group: str,
    rng: random.Random,
) -> float:
    skill_match = len(set(skills) & set(required_skills)) / max(len(required_skills), 1)
    breadth = min(len(skills) / max(len(required_skills), 1), 1.0)
    experience_score = min(years_experience / 10.0, 1.0)
    education_score = EDUCATION_WEIGHTS.get(education, 0.4)
    cert_score = min(len(certifications) / 2.0, 1.0)

    raw = (
        0.42 * skill_match
        + 0.16 * breadth
        + 0.22 * experience_score
        + 0.14 * education_score
        + 0.06 * cert_score
    )
    raw += ARCHETYPE_PROFILES[archetype]["resume_shift"]
    raw += rng.uniform(-0.03, 0.03)

    if difficulty == Difficulty.HARD:
        if group in {"C", "D"} and archetype == "latent_gem":
            raw -= 0.05
        if group == "A" and archetype == "polished_resume":
            raw += 0.03
    elif difficulty == Difficulty.MEDIUM and archetype == "latent_gem":
        raw -= 0.02

    return round(_clamp(raw), 4)


def _compute_potential_scores(
    resume_score: float,
    archetype: str,
    difficulty: Difficulty,
    group: str,
    rng: random.Random,
) -> tuple[float, float, float]:
    profile = ARCHETYPE_PROFILES[archetype]
    potential = resume_score + profile["potential_shift"] + rng.uniform(-0.05, 0.05)
    interview = (
        0.52 * resume_score
        + 0.48 * potential
        + profile["interview_shift"]
        + rng.uniform(-0.04, 0.04)
    )
    if difficulty == Difficulty.EASY:
        interview += 0.02 if archetype == "clear_match" else 0.0
    elif difficulty == Difficulty.HARD:
        if group in {"C", "D"} and archetype == "latent_gem":
            potential += 0.08
            interview += 0.10
        if group == "A" and archetype == "polished_resume":
            potential -= 0.06
            interview -= 0.12
    calibration_risk = abs(interview - resume_score)
    return round(_clamp(potential), 4), round(_clamp(interview), 4), round(_clamp(calibration_risk), 4)


def _compute_ground_truth(
    resume_score: float,
    potential_score: float,
    interview_score: float,
    difficulty: Difficulty,
) -> float:
    if difficulty == Difficulty.EASY:
        weights = (0.58, 0.14, 0.28)
    elif difficulty == Difficulty.MEDIUM:
        weights = (0.40, 0.22, 0.38)
    else:
        weights = (0.28, 0.24, 0.48)
    return round(
        _clamp(
            weights[0] * resume_score
            + weights[1] * potential_score
            + weights[2] * interview_score
        ),
        4,
    )


def generate_candidates(task_config: TaskConfig, seed: Optional[int] = None) -> list[Candidate]:
    """
    Generate a deterministic candidate pool for the provided task configuration.

    The generator is seeded so benchmark runs are reproducible.
    """

    rng = random.Random(seed)
    candidates: list[Candidate] = []

    for cid in range(task_config.num_candidates):
        group = _choose_group(task_config, rng)
        archetype = _choose_archetype(task_config, group, rng)
        profile = ARCHETYPE_PROFILES[archetype]
        years_experience = rng.randint(*profile["experience"])
        education = _education_for_archetype(archetype, rng)
        certifications = _certifications(task_config.role, archetype, rng)
        skills, missing = _sample_skills(task_config.role, task_config.required_skills, archetype, rng)
        strengths = _build_strengths(task_config.role, skills, archetype, missing)
        concerns = _build_concerns(archetype, missing, years_experience)
        resume_summary = _resume_summary(task_config.role, archetype, skills, strengths, concerns)
        resume_score = _compute_resume_score(
            skills,
            task_config.required_skills,
            years_experience,
            education,
            certifications,
            archetype,
            task_config.difficulty,
            group,
            rng,
        )
        potential_score, interview_score, calibration_risk = _compute_potential_scores(
            resume_score,
            archetype,
            task_config.difficulty,
            group,
            rng,
        )
        ground_truth_score = _compute_ground_truth(
            resume_score,
            potential_score,
            interview_score,
            task_config.difficulty,
        )
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

        candidates.append(
            Candidate(
                id=cid,
                name=name,
                years_experience=years_experience,
                skills=skills,
                education=education,
                demographic_group=group,
                certifications=certifications,
                strengths=strengths,
                concerns=concerns,
                resume_summary=resume_summary,
                expected_salary_band=_salary_band(years_experience),
                interview_score=None,
                resume_score=resume_score,
                potential_score=potential_score,
                calibration_risk=calibration_risk,
                ground_truth_score=ground_truth_score,
                archetype=archetype,
            )
        )

    return candidates
