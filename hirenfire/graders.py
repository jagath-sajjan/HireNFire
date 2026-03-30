"""
Graders for the HireNFire environment.

Quality grader: Precision/recall of hire decisions vs ground-truth top-K.
Fairness grader: Demographic parity of hire decisions across groups.
Combined: reward = α * quality + (1 - α) * fairness
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from hirenfire.models import Candidate, RewardInfo


def quality_score(
    candidates: list[Candidate],
    hired_ids: list[int],
    num_to_hire: int,
) -> float:
    """
    Compute quality of hire as precision/recall against ground truth top-K.

    Ground truth top-K = the `num_to_hire` candidates with highest ground_truth_score.
    Quality = F1 score of the agent's hires vs the ideal hires.
    """
    if not hired_ids:
        return 0.0

    # Ground truth: top-K by ground_truth_score
    sorted_candidates = sorted(candidates, key=lambda c: c.ground_truth_score, reverse=True)
    ideal_ids = set(c.id for c in sorted_candidates[:num_to_hire])
    hired_set = set(hired_ids)

    # Precision: what fraction of hires are in the ideal set
    true_positives = len(hired_set & ideal_ids)
    precision = true_positives / len(hired_set) if hired_set else 0.0

    # Recall: what fraction of ideal candidates were hired
    recall = true_positives / len(ideal_ids) if ideal_ids else 0.0

    # F1
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(float(f1), 4)


def fairness_score(
    candidates: list[Candidate],
    hired_ids: list[int],
) -> float:
    """
    Compute demographic parity of hiring decisions.

    Measures how evenly hires are distributed across demographic groups
    relative to the applicant pool composition.

    Perfect parity (1.0) = hire rate is identical across all groups.
    Worst case (0.0) = all hires from a single group when pool is balanced.
    """
    if not hired_ids:
        return 0.0

    hired_set = set(hired_ids)

    # Count applicants and hires per group
    pool_counts: Counter = Counter()
    hire_counts: Counter = Counter()

    for c in candidates:
        pool_counts[c.demographic_group] += 1
        if c.id in hired_set:
            hire_counts[c.demographic_group] += 1

    # Compute hire rate per group
    groups_present = list(pool_counts.keys())
    if len(groups_present) <= 1:
        return 1.0  # Only one group, parity is trivially satisfied

    hire_rates = []
    for g in groups_present:
        pool_n = pool_counts[g]
        hired_n = hire_counts.get(g, 0)
        rate = hired_n / pool_n if pool_n > 0 else 0.0
        hire_rates.append(rate)

    # Parity metric: 1 - (max_rate - min_rate)
    # This is the "demographic parity distance"
    max_rate = max(hire_rates)
    min_rate = min(hire_rates)

    if max_rate == 0:
        return 0.0

    # Normalize: 1.0 when all rates equal, 0.0 when max disparity
    parity = 1.0 - (max_rate - min_rate)
    return round(float(np.clip(parity, 0.0, 1.0)), 4)


def compute_reward(
    candidates: list[Candidate],
    hired_ids: list[int],
    num_to_hire: int,
    alpha: float,
    extra_info: Optional[dict] = None,
) -> RewardInfo:
    """
    Compute the combined reward: α * quality + (1 - α) * fairness.

    Args:
        candidates: Full candidate pool with ground truth scores
        hired_ids: List of hired candidate IDs
        num_to_hire: Target number of hires
        alpha: Weight for quality (1-alpha for fairness)
        extra_info: Optional additional info for the breakdown

    Returns:
        RewardInfo with quality, fairness, and combined scores
    """
    q = quality_score(candidates, hired_ids, num_to_hire)
    f = fairness_score(candidates, hired_ids)
    combined = alpha * q + (1.0 - alpha) * f

    breakdown = {
        "quality_score": q,
        "fairness_score": f,
        "alpha": alpha,
        "num_hired": len(hired_ids),
        "num_to_hire": num_to_hire,
        "formula": f"{alpha:.2f} * {q:.4f} + {1 - alpha:.2f} * {f:.4f} = {combined:.4f}",
    }
    if extra_info:
        breakdown.update(extra_info)

    return RewardInfo(
        quality_score=round(q, 4),
        fairness_score=round(f, 4),
        combined_reward=round(float(np.clip(combined, 0.0, 1.0)), 4),
        alpha=alpha,
        breakdown=breakdown,
    )


def intermediate_reward(
    candidates: list[Candidate],
    hired_ids: list[int],
    rejected_ids: list[int],
    num_to_hire: int,
    alpha: float,
    step: int,
) -> float:
    """
    Compute a partial progress signal during the episode.
    Gives the agent incremental feedback, not just end-of-episode.
    """
    if not hired_ids and not rejected_ids:
        return 0.0

    # Partial quality signal from hires so far
    if hired_ids:
        sorted_cands = sorted(candidates, key=lambda c: c.ground_truth_score, reverse=True)
        ideal_ids = set(c.id for c in sorted_cands[:num_to_hire])
        good_hires = len(set(hired_ids) & ideal_ids)
        partial_quality = good_hires / max(len(hired_ids), 1)
    else:
        partial_quality = 0.0

    # Partial fairness signal
    partial_fairness = fairness_score(candidates, hired_ids) if hired_ids else 0.5

    # Small penalty for excessive steps (discourages loops)
    step_penalty = max(0, step - 30) * 0.01

    partial = alpha * partial_quality + (1.0 - alpha) * partial_fairness - step_penalty
    return round(float(np.clip(partial, 0.0, 1.0)), 4)
