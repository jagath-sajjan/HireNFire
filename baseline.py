#!/usr/bin/env python3
"""
Baseline inference script for HireNFire.

Uses the OpenAI API client to run a model against all 3 tasks and
produce reproducible baseline scores.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python baseline.py
"""

from __future__ import annotations

import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from hirenfire import HiringEnv, Action, ActionType
from hirenfire.tasks import ALL_TASKS


def format_candidates_for_llm(obs) -> str:
    """Format the observation into a prompt for the LLM."""
    lines = [
        f"## Task: {obs.task_config.task_name}",
        f"**Role**: {obs.task_config.role}",
        f"**Required Skills**: {', '.join(obs.task_config.required_skills)}",
        f"**Remaining hiring slots**: {obs.remaining_slots}",
        f"**Step**: {obs.current_step}",
        "",
        "## Candidates:",
    ]

    for c in obs.candidates:
        if c.id in obs.hired_ids:
            status = " ✅ HIRED"
        elif c.id in obs.rejected_ids:
            status = " ❌ REJECTED"
        else:
            status = ""

        interview = f" | Interview Score: {c.interview_score:.2f}" if c.interview_score is not None else ""
        lines.append(
            f"- **ID {c.id}**: {c.name} | {c.years_experience}yr exp | "
            f"Education: {c.education} | Group: {c.demographic_group} | "
            f"Skills: {', '.join(c.skills)}{interview}{status}"
        )

    if obs.hired_ids:
        lines.append(f"\n**Already hired**: {obs.hired_ids}")
    if obs.rejected_ids:
        lines.append(f"**Already rejected**: {obs.rejected_ids}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a hiring assistant AI. You must screen candidates and make hiring decisions.

You must balance TWO objectives:
1. QUALITY: Hire the most qualified candidates based on skills, experience, and education
2. FAIRNESS: Ensure demographic parity — hire proportionally from all demographic groups

Respond with a JSON action. Valid actions:
- {"action_type": "INTERVIEW", "candidate_id": <id>, "reasoning": "..."}
- {"action_type": "HIRE", "candidate_id": <id>, "reasoning": "..."}
- {"action_type": "REJECT", "candidate_id": <id>, "reasoning": "..."}
- {"action_type": "FINALIZE", "reasoning": "..."}

Strategy tips:
- Interview a few candidates first for more information
- Keep track of demographic groups when hiring to maintain fairness
- When done hiring, use FINALIZE

Respond ONLY with valid JSON. No markdown, no explanation outside JSON."""


def run_baseline_task(client: OpenAI, task_config, model: str = "gpt-4o-mini", seed: int = 42):
    """Run a single task with the LLM agent."""
    env = HiringEnv(task_config, seed=seed)
    obs = env.reset()

    print(f"\n{'='*60}")
    print(f"  Task: {task_config.task_name} ({task_config.difficulty.value})")
    print(f"  Candidates: {task_config.num_candidates} | Hire: {task_config.num_to_hire} | α={task_config.alpha}")
    print(f"{'='*60}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_num in range(task_config.max_steps):
        # Format observation for the LLM
        user_msg = format_candidates_for_llm(obs)
        messages.append({"role": "user", "content": user_msg})

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ API error at step {step_num}: {e}")
            # Fallback: finalize
            reply = '{"action_type": "FINALIZE", "reasoning": "API error"}'

        messages.append({"role": "assistant", "content": reply})

        # Parse action from LLM response
        try:
            # Clean up response (strip markdown code fences if present)
            clean = reply.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                clean = clean.rsplit("```", 1)[0]
            action_data = json.loads(clean)
            action = Action(**action_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  ⚠ Parse error at step {step_num}: {e}")
            action = Action(action_type=ActionType.FINALIZE, reasoning="Parse error fallback")

        # Step
        obs, reward, done, info = env.step(action)
        print(f"  Step {step_num + 1}: {action.action_type.value}"
              f"{f' (candidate {action.candidate_id})' if action.candidate_id is not None else ''}")

        if done:
            break

    # If not done yet, finalize
    if not done:
        obs, reward, done, info = env.step(Action(action_type=ActionType.FINALIZE))

    # Print results
    state = env.state()
    final = state["final_reward"]
    print(f"\n  📊 Results:")
    print(f"     Quality:  {final['quality_score']:.4f}")
    print(f"     Fairness: {final['fairness_score']:.4f}")
    print(f"     Combined: {final['combined_reward']:.4f}")
    print(f"     Hired: {state['hired_ids']}")

    return final


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set. Set it to run LLM baseline.")
        print("  Running heuristic baseline instead...\n")
        from demo import run_heuristic_demo
        run_heuristic_demo()
        return

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    print("🧑‍💼 HireNFire — LLM Baseline Inference")
    print(f"   Model: {model}")
    print(f"   Seed: 42")

    results = []
    for task in ALL_TASKS:
        result = run_baseline_task(client, task, model=model, seed=42)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  📋 BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<25} {'Quality':>8} {'Fairness':>9} {'Combined':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*9}")
    for task, r in zip(ALL_TASKS, results):
        print(f"  {task.task_name:<25} {r['quality_score']:>8.4f} {r['fairness_score']:>9.4f} {r['combined_reward']:>9.4f}")


if __name__ == "__main__":
    main()
