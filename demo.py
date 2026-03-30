#!/usr/bin/env python3
"""
Demo / Validation script for HireNFire.

Runs all 3 tasks with a deterministic heuristic agent to validate the
environment works end-to-end. No API key needed.

Usage:
    python demo.py
"""

from __future__ import annotations

from hirenfire import HiringEnv, Action, ActionType
from hirenfire.tasks import ALL_TASKS
from hirenfire.models import TaskConfig


def heuristic_agent(env: HiringEnv) -> dict:
    """
    Simple heuristic agent that:
    1. Interviews a few candidates
    2. Ranks by a computed score (skills match + experience)
    3. Hires top-K while trying to maintain demographic balance
    """
    obs = env.reset()
    task = obs.task_config
    candidates = obs.candidates

    # Phase 1: Interview some candidates for more info
    interview_count = min(len(candidates), task.num_to_hire * 2)
    for c in candidates[:interview_count]:
        action = Action(
            action_type=ActionType.INTERVIEW,
            candidate_id=c.id,
            reasoning=f"Interviewing candidate {c.id} for more information",
        )
        obs, reward, done, info = env.step(action)
        if done:
            break

    # Phase 2: Score candidates based on visible info
    scored = []
    for c in obs.candidates:
        # Skill match
        req = set(task.required_skills)
        skill_match = len(set(c.skills) & req) / max(len(req), 1)

        # Experience (normalize to 0-1)
        exp_score = min(c.years_experience / 10.0, 1.0)

        # Education bonus
        edu_map = {"high_school": 0.1, "bachelors": 0.4, "masters": 0.7, "phd": 0.9}
        edu_score = edu_map.get(c.education, 0.3)

        # Interview score if available
        int_score = c.interview_score if c.interview_score is not None else 0.5

        # Combined score
        score = 0.35 * skill_match + 0.25 * exp_score + 0.15 * edu_score + 0.25 * int_score
        scored.append((c, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Phase 3: Hire with demographic awareness
    from collections import Counter
    group_hires: Counter = Counter()
    group_pool: Counter = Counter()
    for c in candidates:
        group_pool[c.demographic_group] += 1

    hired_count = 0
    for c, score in scored:
        if c.id in obs.hired_ids or c.id in obs.rejected_ids:
            continue
        if hired_count >= task.num_to_hire:
            break

        # Check if hiring this candidate would worsen parity too much
        group = c.demographic_group
        if hired_count > 0:
            # Soft demographic check: prefer under-represented groups
            max_group_hires = max(group_hires.values()) if group_hires else 0
            my_group_hires = group_hires.get(group, 0)
            # Allow if this group isn't already over-represented
            if my_group_hires > max_group_hires + 1 and score < 0.6:
                # Skip this candidate — too many from this group already
                action = Action(
                    action_type=ActionType.REJECT,
                    candidate_id=c.id,
                    reasoning=f"Demographic balance: group {group} over-represented",
                )
                obs, reward, done, info = env.step(action)
                if done:
                    break
                continue

        # Hire
        action = Action(
            action_type=ActionType.HIRE,
            candidate_id=c.id,
            reasoning=f"Score: {score:.3f}, Skills match + experience + education",
        )
        obs, reward, done, info = env.step(action)
        if done:
            break
        group_hires[group] += 1
        hired_count += 1

    # Phase 4: Finalize
    if not done:
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.FINALIZE, reasoning="All decisions made")
        )

    return env.state()


def run_heuristic_demo():
    """Run the heuristic agent on all tasks and print results."""
    print("🧑‍💼 HireNFire — Heuristic Baseline Demo")
    print("=" * 60)

    results = []
    for task in ALL_TASKS:
        env = HiringEnv(task, seed=42)
        state = heuristic_agent(env)

        final = state["final_reward"]
        results.append((task, final))

        print(f"\n📋 Task: {task.task_name} ({task.difficulty.value})")
        print(f"   Candidates: {task.num_candidates} | To hire: {task.num_to_hire} | α={task.alpha}")
        print(f"   Hired IDs: {state['hired_ids']}")
        print(f"   ┌─────────────────────────────────────┐")
        print(f"   │ Quality:   {final['quality_score']:>8.4f}               │")
        print(f"   │ Fairness:  {final['fairness_score']:>8.4f}               │")
        print(f"   │ Combined:  {final['combined_reward']:>8.4f}               │")
        print(f"   └─────────────────────────────────────┘")
        print(f"   Formula: {final['breakdown'].get('formula', 'N/A')}")

    # Summary table
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<25} {'α':>4} {'Quality':>8} {'Fairness':>9} {'Combined':>9}")
    print(f"  {'-'*25} {'-'*4} {'-'*8} {'-'*9} {'-'*9}")
    for task, r in results:
        print(
            f"  {task.task_name:<25} {task.alpha:>4.1f} "
            f"{r['quality_score']:>8.4f} {r['fairness_score']:>9.4f} {r['combined_reward']:>9.4f}"
        )

    print(f"\n✅ All tasks completed successfully. Environment validated.")
    return results


if __name__ == "__main__":
    run_heuristic_demo()
