#!/usr/bin/env python3
"""
Baseline inference runner for HireNFire.

This script intentionally emits only the structured stdout lines required by
the evaluator:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from demo import visible_candidate_score
from hirenfire import Action, ActionType, HiringEnv
from hirenfire.models import CandidateView, Observation
from hirenfire.tasks import ALL_TASKS


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
BENCHMARK = "hirenfire"
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 220
SUCCESS_SCORE_THRESHOLD = 0.50

SYSTEM_PROMPT = """You are deciding hiring actions in a structured hiring environment.

Return exactly one JSON object with one of these forms:
{"action_type":"INTERVIEW","candidate_id":12,"reasoning":"..."}
{"action_type":"HIRE","candidate_id":12,"reasoning":"..."}
{"action_type":"REJECT","candidate_id":12,"reasoning":"..."}
{"action_type":"FINALIZE","reasoning":"..."}

Rules:
- Prefer interviewing ambiguous high-value candidates before hiring.
- Use fairness_targets and current group counts when choosing the final cohort.
- Never output markdown or explanations outside JSON.
"""


def _debug(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = (error or "null").replace("\n", " ")
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_string(action: Action) -> str:
    if action.candidate_id is None:
        return f"{action.action_type.value.lower()}()"
    return f"{action.action_type.value.lower()}({action.candidate_id})"


def _candidate_priority(
    candidate: CandidateView,
    observation: Observation,
) -> float:
    base = visible_candidate_score(candidate, observation.task_config.required_skills)
    current_counts = Counter(
        next(item.demographic_group for item in observation.candidates if item.id == candidate_id)
        for candidate_id in observation.hired_ids
    )
    target_gap = max(
        observation.fairness_targets.get(candidate.demographic_group, 0)
        - current_counts.get(candidate.demographic_group, 0),
        0,
    )
    interview_bonus = 0.07 if candidate.interview_score is None else 0.0
    concern_bonus = 0.04 * sum(
        1
        for concern in candidate.concerns
        if "gap" in concern.lower() or "limited" in concern.lower() or "undersells" in concern.lower()
    )
    return round(base + 0.06 * target_gap + interview_bonus + concern_bonus, 4)


def _available_candidates(observation: Observation) -> list[CandidateView]:
    excluded = set(observation.hired_ids) | set(observation.rejected_ids)
    return [candidate for candidate in observation.candidates if candidate.id not in excluded]


def _shortlist(observation: Observation, limit: int = 8) -> list[CandidateView]:
    candidates = _available_candidates(observation)
    return sorted(candidates, key=lambda candidate: _candidate_priority(candidate, observation), reverse=True)[:limit]


def _heuristic_action(observation: Observation) -> Action:
    shortlist = _shortlist(observation, limit=10)
    if not shortlist or observation.remaining_slots <= 0:
        return Action(action_type=ActionType.FINALIZE, reasoning="No remaining useful actions.")

    interview_budget = observation.task_config.interview_budget_hint
    interviewed = len(observation.interview_results)
    interview_candidates = [candidate for candidate in shortlist if candidate.interview_score is None]

    if interviewed < interview_budget and interview_candidates:
        target = max(interview_candidates, key=lambda candidate: _candidate_priority(candidate, observation))
        return Action(
            action_type=ActionType.INTERVIEW,
            candidate_id=target.id,
            reasoning="Calibrate a high-priority candidate before final selection.",
        )

    current_counts = Counter(
        next(item.demographic_group for item in observation.candidates if item.id == candidate_id)
        for candidate_id in observation.hired_ids
    )
    best_candidate = None
    best_score = -1.0
    for candidate in shortlist:
        quality = visible_candidate_score(candidate, observation.task_config.required_skills)
        group_gap = max(
            observation.fairness_targets.get(candidate.demographic_group, 0)
            - current_counts.get(candidate.demographic_group, 0),
            0,
        )
        score = quality + 0.08 * group_gap
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        return Action(action_type=ActionType.FINALIZE, reasoning="No viable candidates remain.")

    return Action(
        action_type=ActionType.HIRE,
        candidate_id=best_candidate.id,
        reasoning=f"Best visible score after fairness-aware tie-break ({best_score:.3f}).",
    )


def _parse_model_action(text: str) -> Optional[Action]:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
        return Action(**json.loads(cleaned))
    except Exception:
        return None


def _build_user_prompt(observation: Observation) -> str:
    current_counts = Counter(
        next(item.demographic_group for item in observation.candidates if item.id == candidate_id)
        for candidate_id in observation.hired_ids
    )
    shortlist = _shortlist(observation)
    lines = [
        f"task={observation.task_config.task_id}",
        f"role={observation.task_config.role}",
        f"remaining_slots={observation.remaining_slots}",
        f"current_step={observation.current_step}",
        f"fairness_targets={json.dumps(observation.fairness_targets, sort_keys=True)}",
        f"current_group_hires={json.dumps(dict(current_counts), sort_keys=True)}",
        f"recent_event={observation.recent_event}",
        "shortlist:",
    ]
    for candidate in shortlist:
        lines.append(
            json.dumps(
                {
                    "id": candidate.id,
                    "group": candidate.demographic_group,
                    "experience": candidate.years_experience,
                    "education": candidate.education,
                    "skills": candidate.skills[:5],
                    "strengths": candidate.strengths[:2],
                    "concerns": candidate.concerns[:2],
                    "interview_score": candidate.interview_score,
                    "resume_summary": candidate.resume_summary,
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def _llm_action(client: OpenAI, observation: Observation) -> Action:
    heuristic = _heuristic_action(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation)},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_model_action(content)
        if parsed is None:
            return heuristic
        available_ids = {candidate.id for candidate in _available_candidates(observation)}
        if parsed.action_type in {ActionType.INTERVIEW, ActionType.HIRE, ActionType.REJECT} and parsed.candidate_id not in available_ids:
            return heuristic
        return parsed
    except Exception as exc:
        _debug(f"[DEBUG] Model request failed: {exc}")
        return heuristic


def run_task(task, client: Optional[OpenAI]) -> float:
    env = HiringEnv(task, seed=SEED)
    observation = env.reset()
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task.task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, task.max_steps + 1):
            if client is not None:
                action = _llm_action(client, observation)
            else:
                action = _heuristic_action(observation)

            observation, reward, done, info = env.step(action)
            rewards.append(reward.combined_reward)
            steps_taken = step
            log_step(
                step=step,
                action=_action_string(action),
                reward=reward.combined_reward,
                done=done,
                error=info.get("error"),
            )
            if done:
                break

        if not env.state()["done"]:
            final_action = Action(action_type=ActionType.FINALIZE, reasoning="Reached planner stop condition.")
            observation, reward, done, info = env.step(final_action)
            rewards.append(reward.combined_reward)
            steps_taken += 1
            log_step(
                step=steps_taken,
                action=_action_string(final_action),
                reward=reward.combined_reward,
                done=done,
                error=info.get("error"),
            )

        final_state = env.state()
        final_reward = final_state.get("final_reward") or {}
        score = float(final_reward.get("combined_reward", rewards[-1] if rewards else 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception as exc:
        _debug(f"[DEBUG] Task {task.task_id} failed: {exc}")
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client: Optional[OpenAI] = None
    if API_KEY and OpenAI is not None:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    elif API_KEY and OpenAI is None:
        _debug("[DEBUG] openai package unavailable, falling back to heuristic policy.")

    for task in ALL_TASKS:
        run_task(task, client)


if __name__ == "__main__":
    main()
