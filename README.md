---
title: HireNFire
emoji: 🧑‍💼
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - hiring
  - fairness
  - llm-evaluation
  - simulation
license: mit
short_description: Structured hiring benchmark for quality and fairness aware agents
pinned: false
---

# HireNFire

HireNFire is an OpenEnv environment for structured hiring decisions. An agent reviews candidate resumes, decides who to interview, and finalizes a hiring cohort. The benchmark is designed around a real operational tradeoff: picking strong candidates while keeping the final slate aligned with the applicant-pool mix.

The environment is intentionally not a toy ranking task. Resumes are noisy, some candidates outperform their paper profile in interviews, some polished resumes overstate fit, and the hard task forces the agent to use interviews to recover hidden upside rather than greedily hiring the most keyword-matched applicants.

## Why This Environment Exists

Real recruiting teams do more than sort resumes by keyword overlap. They:

- screen a pipeline with incomplete information
- decide where to spend scarce interview bandwidth
- balance candidate quality against cohort-level fairness goals
- avoid over-indexing on polished but misleading resumes

HireNFire packages that workflow into a deterministic, typed OpenEnv benchmark that is usable for RL, planning agents, and LLM evaluation.

## Core API

The environment implements the standard OpenEnv simulation contract:

- `reset() -> Observation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> dict`

Typed models live in [`hirenfire/models.py`](/Users/jagath-sajjan/HireNFire/hirenfire/models.py). Runtime endpoints are exposed from [`app.py`](/Users/jagath-sajjan/HireNFire/app.py):

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`

## Observation Space

Each observation contains:

- visible candidate cards with experience, skills, education, demographic group, certifications, strengths, concerns, and a resume summary
- any interview scores revealed so far
- remaining slots and remaining candidates
- prior decision records
- fairness targets derived from the current applicant pool
- a recent event string summarizing the last state change

Candidate cards intentionally expose enough signal for a strong policy without leaking the hidden grader fields.

## Action Space

The agent can take five actions:

- `RANK`: request a resume-only ranking snapshot
- `INTERVIEW(candidate_id)`: reveal structured interview signal for one candidate
- `HIRE(candidate_id)`: add a candidate to the final slate
- `REJECT(candidate_id)`: remove a candidate from consideration
- `FINALIZE`: end the episode and score the slate

## Reward Design

Final reward follows:

`reward = alpha * quality_score + (1 - alpha) * fairness_score`

`quality_score`
- cohort value of the selected hires divided by the optimal fully-staffed cohort value

`fairness_score`
- similarity between selected group counts and pool-proportional target counts

Partial rewards are emitted on every step. They add signal for:

- interviewing high-priority ambiguous candidates
- building a strong partial slate
- keeping partial group mix on track

They also penalize:

- invalid actions
- rejecting too many priority candidates
- wasting steps

Implementation lives in [`hirenfire/graders.py`](/Users/jagath-sajjan/HireNFire/hirenfire/graders.py).

## Tasks

### Easy: `simple-screen`

- Role: Software Engineer
- Pool: 10 candidates
- Hires: 3
- Alpha: `0.72`
- Shape: mostly clear resume signals with a few borderline candidates

### Medium: `potential-calibration`

- Role: Data Scientist
- Pool: 25 candidates
- Hires: 5
- Alpha: `0.55`
- Shape: career switchers and under-credentialed applicants can outperform in interviews

### Hard: `fair-panel-hiring`

- Role: Senior ML Engineer
- Pool: 50 candidates
- Hires: 8
- Alpha: `0.38`
- Shape: skewed pipeline, more polished false positives, and hidden-upside candidates that are hard to find without interviews

Task definitions live in [`hirenfire/tasks.py`](/Users/jagath-sajjan/HireNFire/hirenfire/tasks.py). Candidate generation logic lives in [`hirenfire/generator.py`](/Users/jagath-sajjan/HireNFire/hirenfire/generator.py).

## Baselines

Two baseline runners are included:

- [`demo.py`](/Users/jagath-sajjan/HireNFire/demo.py): deterministic heuristic sanity check
- [`inference.py`](/Users/jagath-sajjan/HireNFire/inference.py): evaluator-facing baseline runner using the OpenAI client and strict structured stdout logs

Current deterministic heuristic sanity-check scores with seed `42`:

| Task | Quality | Fairness | Combined |
|------|---------|----------|----------|
| `simple-screen` | `0.9763` | `0.3333` | `0.7963` |
| `potential-calibration` | `0.9777` | `0.6000` | `0.8077` |
| `fair-panel-hiring` | `0.9959` | `0.7500` | `0.8434` |

These are sanity-check numbers, not a ceiling for stronger LLM or planning agents.

## Inference Script Requirements

`inference.py`:

- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` or `OPENAI_API_KEY`
- uses the OpenAI client for model calls when credentials are present
- emits only the required `[START]`, `[STEP]`, and `[END]` lines to stdout
- falls back to a deterministic heuristic policy when no API key is available

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

## Local Development

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app locally:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

Run validation:

```bash
openenv validate
openenv validate --url http://127.0.0.1:7860
```

Run the deterministic baseline:

```bash
python demo.py
```

## Docker / Hugging Face Spaces

The repository includes:

- [`Dockerfile`](/Users/jagath-sajjan/HireNFire/Dockerfile)
- [`openenv.yaml`](/Users/jagath-sajjan/HireNFire/openenv.yaml)
- [`server/app.py`](/Users/jagath-sajjan/HireNFire/server/app.py)
- [`pyproject.toml`](/Users/jagath-sajjan/HireNFire/pyproject.toml)

The container installs the package directly from the repository and launches the ASGI app on port `7860`, which matches Hugging Face Spaces expectations.
