#!/usr/bin/env python3
"""
Deterministic baseline / sanity-check runner for HireNFire.
"""

from __future__ import annotations

from collections import Counter

from hirenfire import Action, ActionType, HiringEnv
from hirenfire.models import CandidateView
from hirenfire.tasks import ALL_TASKS


EDU_SCORE = {
    "high_school": 0.18,
    "bachelors": 0.52,
    "masters": 0.76,
    "phd": 0.92,
}


def visible_candidate_score(candidate: CandidateView, required_skills: list[str]) -> float:
    """Estimate candidate quality using only agent-visible fields."""

    skill_match = len(set(candidate.skills) & set(required_skills)) / max(len(required_skills), 1)
    experience_score = min(candidate.years_experience / 10.0, 1.0)
    education_score = EDU_SCORE.get(candidate.education, 0.45)
    certification_score = min(len(candidate.certifications) / 2.0, 1.0)
    strengths_bonus = min(len(candidate.strengths), 3) * 0.04
    concerns_penalty = min(len(candidate.concerns), 3) * 0.03

    base = (
        0.42 * skill_match
        + 0.24 * experience_score
        + 0.18 * education_score
        + 0.08 * certification_score
        + strengths_bonus
        - concerns_penalty
    )
    if candidate.interview_score is not None:
        base = 0.58 * base + 0.42 * candidate.interview_score
    return round(max(0.0, min(base, 1.0)), 4)


def interview_priority(candidate: CandidateView, required_skills: list[str], group_need: float) -> float:
    """Estimate how valuable it is to spend an interview on this candidate."""

    base = visible_candidate_score(candidate, required_skills)
    concern_bonus = 0.05 * sum(
        1
        for concern in candidate.concerns
        if "undersells" in concern.lower()
        or "gap" in concern.lower()
        or "limited" in concern.lower()
    )
    upside_bonus = 0.06 * sum(
        1
        for strength in candidate.strengths
        if "upside" in strength.lower()
        or "portfolio" in strength.lower()
        or "ownership" in strength.lower()
    )
    return round(base + concern_bonus + upside_bonus + group_need, 4)


def heuristic_agent(env: HiringEnv) -> dict:
    """Run a deterministic, fairness-aware heuristic agent."""

    obs = env.reset()
    task = obs.task_config
    fairness_targets = obs.fairness_targets

    current_group_hires: Counter[str] = Counter()
    interview_budget = max(2, task.num_to_hire)

    interview_order = sorted(
        obs.candidates,
        key=lambda candidate: interview_priority(
            candidate,
            task.required_skills,
            0.08 if fairness_targets.get(candidate.demographic_group, 0) > 0 else 0.0,
        ),
        reverse=True,
    )

    for candidate in interview_order[:interview_budget]:
        obs, _, done, _ = env.step(
            Action(
                action_type=ActionType.INTERVIEW,
                candidate_id=candidate.id,
                reasoning="Interview to calibrate ambiguous resume signal.",
            )
        )
        if done:
            return env.state()

    scored_candidates = sorted(
        obs.candidates,
        key=lambda candidate: visible_candidate_score(candidate, task.required_skills),
        reverse=True,
    )

    while len(obs.hired_ids) < task.num_to_hire:
        best_candidate = None
        best_score = -1.0
        for candidate in scored_candidates:
            if candidate.id in obs.hired_ids or candidate.id in obs.rejected_ids:
                continue

            quality = visible_candidate_score(candidate, task.required_skills)
            target_for_group = fairness_targets.get(candidate.demographic_group, 0)
            group_gap = max(target_for_group - current_group_hires.get(candidate.demographic_group, 0), 0)
            fairness_bonus = 0.04 * group_gap
            score = quality + fairness_bonus
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break

        obs, _, done, _ = env.step(
            Action(
                action_type=ActionType.HIRE,
                candidate_id=best_candidate.id,
                reasoning=f"Visible score {best_score:.3f} with fairness-aware tie-break.",
            )
        )
        current_group_hires[best_candidate.demographic_group] += 1
        if done:
            return env.state()

    if not env.state()["done"]:
        env.step(Action(action_type=ActionType.FINALIZE, reasoning="Heuristic hiring plan complete."))
    return env.state()


def run_heuristic_demo() -> list[tuple[str, dict]]:
    """Run the heuristic baseline across all tasks and print a compact table."""

    print("HireNFire heuristic baseline")
    print("=" * 72)
    results: list[tuple[str, dict]] = []

    for task in ALL_TASKS:
        env = HiringEnv(task, seed=42)
        state = heuristic_agent(env)
        reward = state["final_reward"]
        results.append((task.task_id, reward))
        print(
            f"{task.task_id:<8} role={task.role:<20} "
            f"quality={reward['quality_score']:.4f} "
            f"fairness={reward['fairness_score']:.4f} "
            f"combined={reward['combined_reward']:.4f}"
        )

    return results


if __name__ == "__main__":
    run_heuristic_demo()
