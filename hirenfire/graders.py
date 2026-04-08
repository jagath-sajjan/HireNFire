"""
Graders and reward shaping for HireNFire.

Final score:
    reward = alpha * quality + (1 - alpha) * fairness

Intermediate rewards add process signal for interviewing the right candidates,
avoiding invalid actions, and not rejecting high-value applicants too early.
"""

from __future__ import annotations

from collections import Counter

from hirenfire.models import Candidate, RewardInfo

STRICT_SCORE_EPSILON = 0.001


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _strict_score(value: float) -> float:
    """
    Clamp benchmark scores to the open interval (0, 1).

    The evaluator rejects exact 0.0 and 1.0 task scores, so final scorer values
    need a small interior margin even when the underlying cohort is perfect.
    """

    return _clamp(value, STRICT_SCORE_EPSILON, 1.0 - STRICT_SCORE_EPSILON)


def _candidate_lookup(candidates: list[Candidate]) -> dict[int, Candidate]:
    return {candidate.id: candidate for candidate in candidates}


def _unique_existing_ids(candidates: list[Candidate], candidate_ids: list[int]) -> list[int]:
    lookup = _candidate_lookup(candidates)
    seen: set[int] = set()
    ordered: list[int] = []
    for candidate_id in candidate_ids:
        if candidate_id in lookup and candidate_id not in seen:
            seen.add(candidate_id)
            ordered.append(candidate_id)
    return ordered


def target_group_hires(candidates: list[Candidate], num_to_hire: int) -> dict[str, int]:
    """Allocate target cohort counts by applicant-pool proportions."""

    groups = sorted({candidate.demographic_group for candidate in candidates})
    if num_to_hire <= 0 or not groups:
        return {group: 0 for group in groups}

    pool_counts = Counter(candidate.demographic_group for candidate in candidates)
    total_pool = sum(pool_counts.values()) or 1

    raw_targets = {
        group: (pool_counts[group] / total_pool) * num_to_hire
        for group in groups
    }
    targets = {group: int(raw_targets[group]) for group in groups}
    remaining = num_to_hire - sum(targets.values())

    remainders = sorted(
        groups,
        key=lambda group: (raw_targets[group] - targets[group], pool_counts[group]),
        reverse=True,
    )
    for group in remainders[:remaining]:
        targets[group] += 1

    return targets


def quality_score(
    candidates: list[Candidate],
    hired_ids: list[int],
    num_to_hire: int,
) -> float:
    """
    Score hire quality as achieved cohort value vs the optimal cohort value.

    This is smoother than an exact top-K match and better reflects real hiring:
    a near-optimal cohort still gets credit, while under-hiring is naturally
    penalized because the selected cohort value is compared against the best
    possible fully staffed cohort.
    """

    if num_to_hire <= 0 or not candidates or not hired_ids:
        return 0.0

    unique_hires = _unique_existing_ids(candidates, hired_ids)
    lookup = _candidate_lookup(candidates)
    selected_total = sum(lookup[candidate_id].ground_truth_score for candidate_id in unique_hires)
    optimal_total = sum(
        candidate.ground_truth_score
        for candidate in sorted(candidates, key=lambda item: item.ground_truth_score, reverse=True)[:num_to_hire]
    )
    if optimal_total <= 0:
        return 0.0
    return round(_strict_score(selected_total / optimal_total), 4)


def fairness_score(
    candidates: list[Candidate],
    hired_ids: list[int],
    num_to_hire: int | None = None,
) -> float:
    """
    Score fairness by comparing selected group counts to the target cohort mix.

    The target mix is derived from the applicant pool via largest-remainder
    rounding. The metric is strict enough that hiring everyone from the same
    group scores poorly, while small deviations are penalized proportionally.
    """

    if not candidates or not hired_ids:
        return 0.0

    effective_hires = len(_unique_existing_ids(candidates, hired_ids))
    target_size = num_to_hire if num_to_hire is not None else effective_hires
    if target_size <= 0:
        return 0.0

    targets = target_group_hires(candidates, target_size)
    selected_counts = Counter(
        _candidate_lookup(candidates)[candidate_id].demographic_group
        for candidate_id in _unique_existing_ids(candidates, hired_ids)
    )
    deviation = sum(abs(selected_counts.get(group, 0) - targets.get(group, 0)) for group in targets)
    fill_ratio = min(effective_hires / target_size, 1.0)
    mix_score = 1.0 - min(deviation / target_size, 1.0)
    return round(_strict_score(mix_score * fill_ratio), 4)


def compute_reward(
    candidates: list[Candidate],
    hired_ids: list[int],
    num_to_hire: int,
    alpha: float,
    extra_info: dict | None = None,
) -> RewardInfo:
    """Compute the final task reward."""

    q = quality_score(candidates, hired_ids, num_to_hire)
    f = fairness_score(candidates, hired_ids, num_to_hire=num_to_hire)
    combined = _strict_score(alpha * q + (1.0 - alpha) * f)

    targets = target_group_hires(candidates, num_to_hire)
    selected_counts = Counter(
        _candidate_lookup(candidates)[candidate_id].demographic_group
        for candidate_id in _unique_existing_ids(candidates, hired_ids)
    )
    optimal_total = sum(
        candidate.ground_truth_score
        for candidate in sorted(candidates, key=lambda item: item.ground_truth_score, reverse=True)[:num_to_hire]
    )
    selected_total = sum(
        _candidate_lookup(candidates)[candidate_id].ground_truth_score
        for candidate_id in _unique_existing_ids(candidates, hired_ids)
    )

    breakdown = {
        "quality_score": q,
        "fairness_score": f,
        "alpha": alpha,
        "num_hired": len(_unique_existing_ids(candidates, hired_ids)),
        "num_to_hire": num_to_hire,
        "selected_group_counts": dict(selected_counts),
        "target_group_counts": targets,
        "selected_cohort_value": round(selected_total, 4),
        "optimal_cohort_value": round(optimal_total, 4),
        "formula": f"{alpha:.2f} * {q:.4f} + {1 - alpha:.2f} * {f:.4f} = {combined:.4f}",
    }
    if extra_info:
        breakdown.update(extra_info)

    return RewardInfo(
        quality_score=round(_strict_score(q), 4),
        fairness_score=round(_strict_score(f), 4),
        combined_reward=round(combined, 4),
        alpha=alpha,
        breakdown=breakdown,
    )


def compute_partial_reward(
    candidates: list[Candidate],
    hired_ids: list[int],
    rejected_ids: list[int],
    interview_results: dict[int, float],
    num_to_hire: int,
    alpha: float,
    step: int,
    max_steps: int,
    invalid_actions: int = 0,
) -> RewardInfo:
    """Compute dense trajectory reward while the episode is still running."""

    unique_hires = _unique_existing_ids(candidates, hired_ids)
    unique_rejections = set(_unique_existing_ids(candidates, rejected_ids))
    lookup = _candidate_lookup(candidates)

    if unique_hires:
        partial_optimal = sum(
            candidate.ground_truth_score
            for candidate in sorted(candidates, key=lambda item: item.ground_truth_score, reverse=True)[: len(unique_hires)]
        )
        partial_selected = sum(lookup[candidate_id].ground_truth_score for candidate_id in unique_hires)
        partial_quality = _strict_score(partial_selected / partial_optimal) if partial_optimal > 0 else STRICT_SCORE_EPSILON
        partial_fairness = fairness_score(candidates, unique_hires, num_to_hire=len(unique_hires))
    else:
        partial_quality = STRICT_SCORE_EPSILON
        partial_fairness = STRICT_SCORE_EPSILON

    priority_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.ground_truth_score + 0.6 * candidate.calibration_risk + 0.2 * candidate.potential_score
        ),
        reverse=True,
    )[: max(num_to_hire * 2, 4)]
    priority_ids = {candidate.id for candidate in priority_candidates}
    interview_coverage = (
        len(priority_ids & set(interview_results.keys())) / max(len(priority_ids), 1)
        if priority_ids
        else 0.0
    )
    rejected_priority = len(priority_ids & unique_rejections) / max(len(priority_ids), 1)
    efficiency_penalty = min(step / max(max_steps, 1), 1.0) * 0.08
    invalid_penalty = min(invalid_actions * 0.10, 0.30)

    combined = (
        alpha * partial_quality
        + (1.0 - alpha) * partial_fairness
        + 0.12 * interview_coverage
        - 0.18 * rejected_priority
        - efficiency_penalty
        - invalid_penalty
    )

    return RewardInfo(
        quality_score=round(_strict_score(partial_quality), 4),
        fairness_score=round(_strict_score(partial_fairness), 4),
        combined_reward=round(_strict_score(combined), 4),
        alpha=alpha,
        breakdown={
            "type": "partial",
            "step": step,
            "interview_coverage": round(interview_coverage, 4),
            "rejected_priority_fraction": round(rejected_priority, 4),
            "invalid_actions": invalid_actions,
            "efficiency_penalty": round(efficiency_penalty, 4),
        },
    )
