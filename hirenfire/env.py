"""
HiringEnv — core OpenEnv environment for the HireNFire benchmark.
"""

from __future__ import annotations

from typing import Any

from hirenfire.generator import generate_candidates
from hirenfire.graders import compute_partial_reward, compute_reward, target_group_hires
from hirenfire.models import (
    Action,
    ActionType,
    Candidate,
    CandidateView,
    DecisionRecord,
    EnvironmentState,
    Observation,
    RewardInfo,
    TaskConfig,
)


class HiringEnv:
    """
    Bias-aware hiring simulation.

    The agent reviews candidate cards, interviews candidates for more signal,
    and hires a final cohort that should balance candidate quality with fair
    representation across the observed pipeline.
    """

    def __init__(self, task_config: TaskConfig, seed: int | None = None):
        self.task_config = task_config
        self.seed = seed
        self._candidates: list[Candidate] = []
        self._hired_ids: list[int] = []
        self._rejected_ids: list[int] = []
        self._interview_results: dict[int, float] = {}
        self._decisions: list[DecisionRecord] = []
        self._current_step: int = 0
        self._done: bool = False
        self._invalid_actions: int = 0
        self._recent_event: str = ""
        self._final_reward: RewardInfo | None = None

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""

        self._candidates = generate_candidates(self.task_config, seed=self.seed)
        self._hired_ids = []
        self._rejected_ids = []
        self._interview_results = {}
        self._decisions = []
        self._current_step = 0
        self._done = False
        self._invalid_actions = 0
        self._recent_event = f"New slate loaded for {self.task_config.role}."
        self._final_reward = None
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, RewardInfo, bool, dict]:
        """Execute an agent action and return (observation, reward, done, info)."""

        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._current_step += 1
        info: dict[str, Any] = {"step": self._current_step, "action": action.action_type.value}
        candidate = self._find_candidate(action.candidate_id) if action.candidate_id is not None else None

        if action.action_type == ActionType.RANK:
            ranked_ids = [
                item.id
                for item in sorted(
                    self._candidates,
                    key=lambda candidate_item: candidate_item.resume_score,
                    reverse=True,
                )[: min(5, len(self._candidates))]
            ]
            info["suggested_resume_order"] = ranked_ids
            info["message"] = "Resume-only ranking requested."
            self._recent_event = "Generated a resume-only ranking view."

        elif action.action_type == ActionType.INTERVIEW:
            if action.candidate_id is None:
                info["error"] = "INTERVIEW requires a candidate_id."
            elif candidate is None:
                info["error"] = f"Candidate {action.candidate_id} not found."
            elif action.candidate_id in self._rejected_ids:
                info["error"] = f"Candidate {action.candidate_id} was already rejected."
            elif action.candidate_id in self._hired_ids:
                info["error"] = f"Candidate {action.candidate_id} was already hired."
            elif action.candidate_id in self._interview_results:
                info["error"] = f"Candidate {action.candidate_id} was already interviewed."
            else:
                interview_score = round(
                    max(
                        0.0,
                        min(
                            1.0,
                            candidate.potential_score * 0.35
                            + candidate.ground_truth_score * 0.65,
                        ),
                    ),
                    4,
                )
                candidate.interview_score = interview_score
                self._interview_results[action.candidate_id] = interview_score
                info["interview_score"] = interview_score
                info["message"] = f"Interviewed candidate {action.candidate_id}."
                self._recent_event = f"Interviewed {candidate.name}; signal now available."

        elif action.action_type == ActionType.HIRE:
            if action.candidate_id is None:
                info["error"] = "HIRE requires a candidate_id."
            elif candidate is None:
                info["error"] = f"Candidate {action.candidate_id} not found."
            elif action.candidate_id in self._hired_ids:
                info["error"] = f"Candidate {action.candidate_id} was already hired."
            elif action.candidate_id in self._rejected_ids:
                info["error"] = f"Candidate {action.candidate_id} was already rejected."
            elif len(self._hired_ids) >= self.task_config.num_to_hire:
                info["error"] = "All hiring slots are already filled."
            else:
                self._hired_ids.append(action.candidate_id)
                info["remaining_slots"] = self.task_config.num_to_hire - len(self._hired_ids)
                info["message"] = f"Hired candidate {action.candidate_id}."
                self._recent_event = f"Added {candidate.name} to the hire slate."

        elif action.action_type == ActionType.REJECT:
            if action.candidate_id is None:
                info["error"] = "REJECT requires a candidate_id."
            elif candidate is None:
                info["error"] = f"Candidate {action.candidate_id} not found."
            elif action.candidate_id in self._hired_ids:
                info["error"] = f"Candidate {action.candidate_id} was already hired."
            elif action.candidate_id in self._rejected_ids:
                info["error"] = f"Candidate {action.candidate_id} was already rejected."
            else:
                self._rejected_ids.append(action.candidate_id)
                info["message"] = f"Rejected candidate {action.candidate_id}."
                self._recent_event = f"Rejected {candidate.name} from consideration."

        elif action.action_type == ActionType.FINALIZE:
            self._done = True
            self._final_reward = compute_reward(
                self._candidates,
                self._hired_ids,
                self.task_config.num_to_hire,
                self.task_config.alpha,
                extra_info={"finalized_by_agent": True},
            )
            info["final_reward"] = self._final_reward.model_dump(mode="json")
            info["message"] = "Episode finalized."
            self._recent_event = "Hiring plan finalized."

        if "error" in info:
            self._invalid_actions += 1
            self._recent_event = info["error"]

        if not self._done and len(self._hired_ids) >= self.task_config.num_to_hire:
            self._done = True
            self._final_reward = compute_reward(
                self._candidates,
                self._hired_ids,
                self.task_config.num_to_hire,
                self.task_config.alpha,
                extra_info={"auto_finalized": True, "reason": "target_hires_reached"},
            )
            info["auto_finalized"] = True
            info["final_reward"] = self._final_reward.model_dump(mode="json")
            self._recent_event = "Hiring slate filled; episode auto-finalized."

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
            info["final_reward"] = self._final_reward.model_dump(mode="json")
            self._recent_event = "Reached max steps; episode auto-finalized."

        if self._done and self._final_reward is not None:
            reward = self._final_reward
        else:
            reward = compute_partial_reward(
                self._candidates,
                self._hired_ids,
                self._rejected_ids,
                self._interview_results,
                self.task_config.num_to_hire,
                self.task_config.alpha,
                self._current_step,
                self.task_config.max_steps,
                invalid_actions=self._invalid_actions,
            )

        self._decisions.append(
            DecisionRecord(
                step=self._current_step,
                action=action.action_type.value,
                candidate_id=action.candidate_id,
                reward=reward.combined_reward,
                done=self._done,
                info=info,
            )
        )

        return self._make_observation(), reward, self._done, info

    def state(self) -> dict:
        """Return the full current state as JSON-serializable data."""

        snapshot = EnvironmentState(
            task_config=self.task_config,
            candidates=self._candidates,
            hired_ids=self._hired_ids,
            rejected_ids=self._rejected_ids,
            interview_results=self._interview_results,
            decisions=self._decisions,
            current_step=self._current_step,
            done=self._done,
            invalid_actions=self._invalid_actions,
            final_reward=self._final_reward,
        )
        return snapshot.model_dump(mode="json")

    def _find_candidate(self, candidate_id: int | None) -> Candidate | None:
        if candidate_id is None:
            return None
        for candidate in self._candidates:
            if candidate.id == candidate_id:
                return candidate
        return None

    def _make_observation(self) -> Observation:
        candidates = [
            CandidateView(
                id=candidate.id,
                name=candidate.name,
                years_experience=candidate.years_experience,
                skills=candidate.skills,
                education=candidate.education,
                demographic_group=candidate.demographic_group,
                certifications=candidate.certifications,
                strengths=candidate.strengths,
                concerns=candidate.concerns,
                resume_summary=candidate.resume_summary,
                expected_salary_band=candidate.expected_salary_band,
                interview_score=self._interview_results.get(candidate.id),
            )
            for candidate in self._candidates
        ]

        observation = Observation(
            candidates=candidates,
            current_step=self._current_step,
            task_config=self.task_config,
            decisions=self._decisions,
            remaining_slots=max(self.task_config.num_to_hire - len(self._hired_ids), 0),
            remaining_candidates=max(
                len(self._candidates) - len(self._hired_ids) - len(self._rejected_ids),
                0,
            ),
            interview_results=self._interview_results,
            hired_ids=self._hired_ids,
            rejected_ids=self._rejected_ids,
            fairness_targets=target_group_hires(self._candidates, self.task_config.num_to_hire),
            recent_event=self._recent_event,
        )
        return observation
