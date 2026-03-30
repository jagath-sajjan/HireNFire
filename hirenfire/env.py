"""
HiringEnv — the core OpenEnv environment class.

Implements step() / reset() / state() per the OpenEnv specification.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hirenfire.models import (
    Action, ActionType, Candidate, CandidateView, Observation, RewardInfo, TaskConfig,
)
from hirenfire.generator import generate_candidates
from hirenfire.graders import compute_reward, intermediate_reward


class HiringEnv:
    """
    Bias-Aware Hiring Screener Environment.

    The agent screens candidates, conducts interviews, and makes hiring
    decisions. Graded on both quality and fairness.

    API:
        reset() → Observation
        step(Action) → (Observation, RewardInfo, bool, dict)
        state() → dict
    """

    def __init__(self, task_config: TaskConfig, seed: int | None = None):
        self.task_config = task_config
        self.seed = seed

        # Internal state (set on reset)
        self._candidates: list[Candidate] = []
        self._hired_ids: list[int] = []
        self._rejected_ids: list[int] = []
        self._interview_results: dict[int, float] = {}
        self._decisions: list[dict] = []
        self._current_step: int = 0
        self._done: bool = False
        self._final_reward: RewardInfo | None = None

    # ── OpenEnv API ──────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._candidates = generate_candidates(self.task_config, seed=self.seed)
        self._hired_ids = []
        self._rejected_ids = []
        self._interview_results = {}
        self._decisions = []
        self._current_step = 0
        self._done = False
        self._final_reward = None
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, RewardInfo, bool, dict]:
        """
        Take an action and return (observation, reward, done, info).

        Actions:
            RANK      — No-op; the agent's ranking is implicit in its reasoning.
            INTERVIEW — Reveals the interview_score for the specified candidate.
            HIRE      — Hire a candidate (fills a slot).
            REJECT    — Reject a candidate (removed from consideration).
            FINALIZE  — End the episode and compute final scores.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._current_step += 1
        info: dict[str, Any] = {"step": self._current_step, "action": action.action_type.value}

        # ── Process action ───────────────────────────────────────────────
        if action.action_type == ActionType.RANK:
            info["message"] = "Ranking noted. Use HIRE/REJECT to act on your ranking."

        elif action.action_type == ActionType.INTERVIEW:
            cid = action.candidate_id
            if cid is None:
                info["error"] = "INTERVIEW requires a candidate_id"
            elif cid in self._interview_results:
                info["message"] = f"Candidate {cid} already interviewed."
            else:
                candidate = self._find_candidate(cid)
                if candidate is None:
                    info["error"] = f"Candidate {cid} not found."
                else:
                    # Reveal interview score (ground truth + small noise)
                    noise = np.random.normal(0, 0.05)
                    iscore = float(np.clip(candidate.ground_truth_score + noise, 0.0, 1.0))
                    iscore = round(iscore, 4)
                    candidate.interview_score = iscore
                    self._interview_results[cid] = iscore
                    info["interview_score"] = iscore
                    info["message"] = f"Interviewed candidate {cid}: score = {iscore}"

        elif action.action_type == ActionType.HIRE:
            cid = action.candidate_id
            if cid is None:
                info["error"] = "HIRE requires a candidate_id"
            elif cid in self._hired_ids:
                info["error"] = f"Candidate {cid} already hired."
            elif cid in self._rejected_ids:
                info["error"] = f"Candidate {cid} already rejected."
            elif len(self._hired_ids) >= self.task_config.num_to_hire:
                info["error"] = "All hiring slots filled. FINALIZE to end."
            else:
                self._hired_ids.append(cid)
                info["message"] = f"Hired candidate {cid}. Reason: {action.reasoning}"
                info["remaining_slots"] = self.task_config.num_to_hire - len(self._hired_ids)

        elif action.action_type == ActionType.REJECT:
            cid = action.candidate_id
            if cid is None:
                info["error"] = "REJECT requires a candidate_id"
            elif cid in self._hired_ids:
                info["error"] = f"Candidate {cid} already hired."
            elif cid in self._rejected_ids:
                info["error"] = f"Candidate {cid} already rejected."
            else:
                self._rejected_ids.append(cid)
                info["message"] = f"Rejected candidate {cid}. Reason: {action.reasoning}"

        elif action.action_type == ActionType.FINALIZE:
            self._done = True
            self._final_reward = compute_reward(
                self._candidates,
                self._hired_ids,
                self.task_config.num_to_hire,
                self.task_config.alpha,
            )
            info["message"] = "Episode finalized."
            info["final_reward"] = self._final_reward.model_dump()

        # ── Record decision ──────────────────────────────────────────────
        self._decisions.append({
            "step": self._current_step,
            "action": action.model_dump(),
            "info": info,
        })

        # ── Auto-finalize if max steps reached or all slots filled ───────
        if not self._done and self._current_step >= self.task_config.max_steps:
            self._done = True
            self._final_reward = compute_reward(
                self._candidates,
                self._hired_ids,
                self.task_config.num_to_hire,
                self.task_config.alpha,
                extra_info={"auto_finalized": True, "reason": "max_steps_reached"},
            )
            info["auto_finalized"] = True

        # ── Compute reward ───────────────────────────────────────────────
        if self._done and self._final_reward:
            reward = self._final_reward
        else:
            # Intermediate partial reward
            ir = intermediate_reward(
                self._candidates, self._hired_ids, self._rejected_ids,
                self.task_config.num_to_hire, self.task_config.alpha,
                self._current_step,
            )
            reward = RewardInfo(
                quality_score=ir,
                fairness_score=ir,
                combined_reward=ir,
                alpha=self.task_config.alpha,
                breakdown={"type": "intermediate", "step": self._current_step},
            )

        return self._make_observation(), reward, self._done, info

    def state(self) -> dict:
        """Return the full current state of the environment."""
        return {
            "task_config": self.task_config.model_dump(),
            "candidates": [c.model_dump() for c in self._candidates],
            "hired_ids": self._hired_ids.copy(),
            "rejected_ids": self._rejected_ids.copy(),
            "interview_results": self._interview_results.copy(),
            "decisions": self._decisions.copy(),
            "current_step": self._current_step,
            "done": self._done,
            "final_reward": self._final_reward.model_dump() if self._final_reward else None,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_candidate(self, cid: int) -> Candidate | None:
        for c in self._candidates:
            if c.id == cid:
                return c
        return None

    def _make_observation(self) -> Observation:
        """Create agent-visible observation (hides ground_truth_score)."""
        candidate_views = []
        for c in self._candidates:
            candidate_views.append(CandidateView(
                id=c.id,
                name=c.name,
                years_experience=c.years_experience,
                skills=c.skills,
                education=c.education,
                demographic_group=c.demographic_group,
                interview_score=self._interview_results.get(c.id),
            ))

        return Observation(
            candidates=candidate_views,
            current_step=self._current_step,
            task_config=self.task_config,
            decisions=[
                {"step": d["step"], "action": d["action"]["action_type"], "info_keys": list(d["info"].keys())}
                for d in self._decisions
            ],
            remaining_slots=self.task_config.num_to_hire - len(self._hired_ids),
            interview_results=self._interview_results.copy(),
            hired_ids=self._hired_ids.copy(),
            rejected_ids=self._rejected_ids.copy(),
        )
