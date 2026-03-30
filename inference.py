#!/usr/bin/env python3
"""
inference.py — HireNFire LLM Baseline Inference Script

Runs an LLM agent via the OpenAI client against all 3 tasks and produces
reproducible baseline scores. Required environment variables:

    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use
    HF_TOKEN       Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-api-key"
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from hirenfire import HiringEnv, Action, ActionType
from hirenfire.tasks import ALL_TASKS
from hirenfire.models import TaskConfig

# ── Read credentials from environment ────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

if not HF_TOKEN:
    print("⚠ WARNING: HF_TOKEN / OPENAI_API_KEY not set.")
    print("  Running heuristic fallback instead of LLM inference.\n")

SEED = 42

# ── Prompts ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a fair and skilled hiring assistant AI agent.

You screen candidates for job roles. Your goal is to BOTH:
1. QUALITY: hire the most qualified candidates (skills, experience, education match)
2. FAIRNESS: ensure demographic parity — hire proportionally from groups A, B, C, D

You receive observations and must output a single JSON action:
  {"action_type": "INTERVIEW", "candidate_id": <int>, "reasoning": "<str>"}
  {"action_type": "HIRE",      "candidate_id": <int>, "reasoning": "<str>"}
  {"action_type": "REJECT",    "candidate_id": <int>, "reasoning": "<str>"}
  {"action_type": "FINALIZE",  "reasoning": "<str>"}

Rules:
- INTERVIEW a candidate first to see their interview score before deciding
- Track which demographic groups you hire from — aim for proportional representation
- FINALIZE when all slots are filled or you have screened enough candidates
- Output ONLY valid JSON. No markdown, no extra text.

Strategy: Interview top candidates from each group → hire best per slot → FINALIZE."""


def _format_obs(obs) -> str:
    cfg = obs.task_config
    lines = [
        f"TASK: {cfg.task_name} | Role: {cfg.role}",
        f"Required Skills: {', '.join(cfg.required_skills)}",
        f"Slots remaining: {obs.remaining_slots} | Step: {obs.current_step}/{cfg.max_steps}",
        f"Hired IDs: {obs.hired_ids} | Rejected IDs: {obs.rejected_ids}",
        "",
        "CANDIDATES (ID | Name | Group | Education | Exp | Skills | InterviewScore | Status):",
    ]
    for c in obs.candidates:
        if c.id in obs.hired_ids:
            status = "HIRED"
        elif c.id in obs.rejected_ids:
            status = "REJECTED"
        else:
            status = "pending"
        int_sc = f"{c.interview_score:.2f}" if c.interview_score is not None else "unknown"
        skills_str = ", ".join(c.skills[:5])
        lines.append(
            f"  [{c.id}] {c.name} | Grp:{c.demographic_group} | {c.education} | "
            f"{c.years_experience}yr | {skills_str} | IntScore:{int_sc} | {status}"
        )
    return "\n".join(lines)


def _parse_action(text: str) -> Action:
    """Parse LLM output into an Action. Falls back to FINALIZE on error."""
    try:
        clean = text.strip()
        # Strip markdown code fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(clean)
        return Action(**data)
    except Exception:
        return Action(action_type=ActionType.FINALIZE, reasoning="Parse fallback")


def run_llm_task(client: OpenAI, task_config: TaskConfig) -> dict:
    """Run one task episode with the LLM agent. Returns final reward dict."""
    env = HiringEnv(task_config, seed=SEED)
    obs = env.reset()

    print(f"\n{'─'*58}")
    print(f"  📋 {task_config.task_name} ({task_config.difficulty.value.upper()})")
    print(f"  Candidates: {task_config.num_candidates} | Hire: {task_config.num_to_hire} | α={task_config.alpha}")
    print(f"{'─'*58}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False

    for step_num in range(task_config.max_steps):
        user_content = _format_obs(obs)
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ API error step {step_num + 1}: {e}")
            reply = '{"action_type": "FINALIZE", "reasoning": "API error"}'

        messages.append({"role": "assistant", "content": reply})
        action = _parse_action(reply)

        obs, reward, done, info = env.step(action)
        cid_str = f" → candidate {action.candidate_id}" if action.candidate_id is not None else ""
        print(f"  Step {step_num + 1:2d}: {action.action_type.value}{cid_str}")

        if done:
            break
        # Small delay to respect rate limits
        time.sleep(0.1)

    if not done:
        obs, reward, done, _ = env.step(Action(action_type=ActionType.FINALIZE))

    state = env.state()
    r = state["final_reward"]
    print(f"\n  Quality:   {r['quality_score']:.4f}")
    print(f"  Fairness:  {r['fairness_score']:.4f}")
    print(f"  Combined:  {r['combined_reward']:.4f}  ({r['breakdown'].get('formula','')})")
    print(f"  Hired:     {state['hired_ids']}")
    return r


def run_heuristic_fallback() -> list[dict]:
    """Run heuristic agent when no API key is available."""
    from demo import heuristic_agent
    results = []
    for task in ALL_TASKS:
        env = HiringEnv(task, seed=SEED)
        state = heuristic_agent(env)
        r = state["final_reward"]
        print(f"  {task.task_name:<28} Q={r['quality_score']:.4f}  F={r['fairness_score']:.4f}  Combined={r['combined_reward']:.4f}")
        results.append(r)
    return results


def main():
    print("=" * 58)
    print("  🧑‍💼 HireNFire — Inference Script")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  API URL:  {API_BASE_URL}")
    print(f"  Seed:     {SEED}")
    print("=" * 58)

    if not HF_TOKEN:
        print("\n⚠ No API key — running heuristic baseline:\n")
        results = run_heuristic_fallback()
    else:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        results = []
        for task in ALL_TASKS:
            r = run_llm_task(client, task)
            results.append(r)

    # Final summary table
    print(f"\n{'='*58}")
    print("  📊 BASELINE SUMMARY")
    print(f"{'='*58}")
    print(f"  {'Task':<28} {'α':>4} {'Q':>7} {'F':>7} {'Combined':>9}")
    print(f"  {'-'*28} {'-'*4} {'-'*7} {'-'*7} {'-'*9}")
    for task, r in zip(ALL_TASKS, results):
        print(
            f"  {task.task_name:<28} {task.alpha:>4.1f} "
            f"{r['quality_score']:>7.4f} {r['fairness_score']:>7.4f} {r['combined_reward']:>9.4f}"
        )

    # Machine-readable output for validators
    output = {
        "model": MODEL_NAME,
        "seed": SEED,
        "results": [
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "difficulty": task.difficulty.value,
                "alpha": task.alpha,
                "quality_score": r["quality_score"],
                "fairness_score": r["fairness_score"],
                "combined_reward": r["combined_reward"],
            }
            for task, r in zip(ALL_TASKS, results)
        ]
    }
    print(f"\n📄 JSON Results:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
