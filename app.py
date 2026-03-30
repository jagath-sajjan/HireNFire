"""
app.py — HireNFire Dashboard
Bias-Aware Hiring Screener · OpenEnv · Meta x HF Hackathon
"""

from __future__ import annotations
import json, os, random
from collections import Counter
from typing import Optional

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hirenfire import HiringEnv, Action, ActionType
from hirenfire.models import Candidate
from hirenfire.tasks import EASY_TASK, MEDIUM_TASK, HARD_TASK, ALL_TASKS
from hirenfire.graders import quality_score, fairness_score
from hirenfire.generator import SKILL_POOLS

# ─── Global state ──────────────────────────────────────────────────────────
_env: HiringEnv | None = None
_current_task_id = "easy"
_reward_history: list[float] = []
_step_history: list[int] = []
_task_map = {"easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK}
_custom_candidates: list[Candidate] = []
_custom_id_counter = 1000

TASK_LABEL_TO_ID = {
    "Easy — Simple Ranking": "easy",
    "Medium — High Potential Screening": "medium",
    "Hard — Fair Hiring at Scale": "hard",
}

def _get_env() -> HiringEnv:
    global _env
    if _env is None:
        _env = HiringEnv(_task_map[_current_task_id], seed=42)
        _env.reset()
    return _env

def _reset_env(task_id: str) -> HiringEnv:
    global _env, _current_task_id, _reward_history, _step_history
    _current_task_id = task_id
    _reward_history = []
    _step_history = []
    _env = HiringEnv(_task_map[task_id], seed=random.randint(0, 9999))
    _env.reset()
    for cc in _custom_candidates:
        _env._candidates.append(cc)
    return _env

# ─── Data helpers ──────────────────────────────────────────────────────────

def _live_scores(env: HiringEnv):
    state = env.state()
    hired = state["hired_ids"]
    num_to_hire = state["task_config"]["num_to_hire"]
    alpha = state["task_config"]["alpha"]
    if state["done"] and state["final_reward"]:
        r = state["final_reward"]
        return r["quality_score"], r["fairness_score"], r["combined_reward"], True
    candidates = [Candidate(**c) for c in state["candidates"]]
    q = quality_score(candidates, hired, num_to_hire) if hired else 0.0
    f = fairness_score(candidates, hired) if hired else 0.0
    return q, f, alpha * q + (1 - alpha) * f, False

def _table_rows(env: HiringEnv) -> list[list]:
    state = env.state()
    hired = set(state["hired_ids"])
    rejected = set(state["rejected_ids"])
    interviewed = state["interview_results"]
    rows = []
    for c in state["candidates"]:
        cid = c["id"]
        status = "Hired" if cid in hired else ("Rejected" if cid in rejected else "Pending")
        int_score = f"{interviewed[cid]:.2f}" if cid in interviewed else "—"
        tag = " ★" if cid >= 1000 else ""
        rows.append([
            cid,
            c["name"] + tag,
            c["demographic_group"],
            c["education"].replace("_", " ").title(),
            c["years_experience"],
            "  ".join(c["skills"][:4]) + (" +" if len(c["skills"]) > 4 else ""),
            int_score,
            status,
        ])
    return rows

# ─── Plotly charts ─────────────────────────────────────────────────────────

DARK   = "#0a0a0a"
CARD   = "#111"
BORDER = "#222"
TEXT   = "#e5e5e5"
MUTED  = "#555"
INDIGO = "#6366f1"
TEAL   = "#14b8a6"
AMBER  = "#f59e0b"
RED    = "#ef4444"
GREEN  = "#10b981"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK,
    plot_bgcolor=DARK,
    font=dict(family="'Inter', sans-serif", color=TEXT, size=11),
    margin=dict(l=12, r=16, t=36, b=12),
)


def _gauge_chart(q: float, f: float, combined: float, is_final: bool) -> go.Figure:
    fig = go.Figure()

    vals  = [q,       f,       combined]
    names = ["Quality","Fairness","Combined"]
    cols  = [INDIGO,   TEAL,     AMBER]

    for i, (val, name, col) in enumerate(zip(vals, names, cols)):
        y = 2 - i
        # Track
        fig.add_shape(type="rect", x0=0, x1=1, y0=y-0.18, y1=y+0.18,
                      fillcolor=BORDER, line_width=0, layer="below")
        # Fill
        if val > 0:
            fig.add_shape(type="rect", x0=0, x1=val, y0=y-0.18, y1=y+0.18,
                          fillcolor=col, opacity=0.9, line_width=0)
        # Label
        fig.add_annotation(x=-0.02, y=y, text=name, showarrow=False,
                           xanchor="right", font=dict(color=MUTED, size=11))
        fig.add_annotation(x=1.02, y=y, text=f"{val:.3f}", showarrow=False,
                           xanchor="left", font=dict(color=col, size=13, family="Inter"))

    badge = "FINAL" if is_final else "LIVE"
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"<b>Score Panel</b>   <span style='font-size:10px;color:{MUTED}'>{badge}</span>",
                   font=dict(size=13, color=TEXT), x=0),
        xaxis=dict(visible=False, range=[-0.35, 1.35]),
        yaxis=dict(visible=False, range=[-0.5, 2.7]),
        height=200,
    )
    return fig


def _demo_parity_chart(env: HiringEnv) -> go.Figure:
    state = env.state()
    hired_set = set(state["hired_ids"])
    pool_counts: Counter = Counter()
    hire_counts: Counter = Counter()
    for c in state["candidates"]:
        pool_counts[c["demographic_group"]] += 1
        if c["id"] in hired_set:
            hire_counts[c["demographic_group"]] += 1

    groups = sorted(pool_counts.keys())
    pool_ns  = [pool_counts[g] for g in groups]
    hire_ns  = [hire_counts.get(g, 0) for g in groups]
    rates    = [h / p if p else 0 for h, p in zip(hire_ns, pool_ns)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups, y=pool_ns, name="Pool", marker_color=BORDER,
        text=pool_ns, textposition="outside",
        textfont=dict(color=MUTED, size=11),
    ))
    fig.add_trace(go.Bar(
        x=groups, y=hire_ns, name="Hired", marker_color=INDIGO,
        text=hire_ns, textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))

    # Parity line (ideal rate)
    total_pool  = sum(pool_ns) or 1
    total_hired = sum(hire_ns)
    ideal_rate  = total_hired / total_pool
    ideal_vals  = [ideal_rate * p for p in pool_ns]
    fig.add_trace(go.Scatter(
        x=groups, y=ideal_vals, mode="lines+markers",
        line=dict(color=AMBER, dash="dot", width=2),
        marker=dict(size=6, color=AMBER),
        name="Ideal",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Demographic Parity</b>", font=dict(size=13, color=TEXT), x=0),
        barmode="group",
        bargap=0.3,
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Group",
                   title_font=dict(color=MUTED, size=11)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False),
        showlegend=True,
        legend=dict(bgcolor=DARK, bordercolor=BORDER, font=dict(color=MUTED, size=10),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280,
    )
    return fig


def _score_progression_chart() -> go.Figure:
    fig = go.Figure()
    if not _step_history:
        fig.add_annotation(text="Take actions to see score progression",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=MUTED, size=12), xref="paper", yref="paper")
    else:
        fig.add_trace(go.Scatter(
            x=_step_history, y=_reward_history,
            mode="lines+markers",
            line=dict(color=AMBER, width=2.5),
            marker=dict(size=6, color=AMBER, line=dict(color=DARK, width=2)),
            fill="tozeroy", fillcolor=f"rgba(245,158,11,0.08)",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Reward Trajectory</b>", font=dict(size=13, color=TEXT), x=0),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Step",
                   title_font=dict(color=MUTED, size=11)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, range=[0, 1],
                   title="Reward", title_font=dict(color=MUTED, size=11)),
        height=220,
    )
    return fig


def _quality_dist_chart(env: HiringEnv) -> go.Figure:
    state = env.state()
    hired_set    = set(state["hired_ids"])
    rejected_set = set(state["rejected_ids"])

    hired_scores    = []
    rejected_scores = []
    pending_scores  = []

    for c in state["candidates"]:
        score = c["ground_truth_score"]  # only used internally for the chart
        if c["id"] in hired_set:
            hired_scores.append(score)
        elif c["id"] in rejected_set:
            rejected_scores.append(score)
        else:
            pending_scores.append(score)

    fig = go.Figure()
    for scores, name, col in [
        (pending_scores,  "Pending",  MUTED),
        (rejected_scores, "Rejected", RED),
        (hired_scores,    "Hired",    GREEN),
    ]:
        if scores:
            fig.add_trace(go.Histogram(
                x=scores, name=name, nbinsx=10,
                marker_color=col, opacity=0.75,
                histnorm="",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>Quality Distribution</b> <span style='font-size:10px;color:#4b5563'>(ground truth — evaluator view)</span>",
                   font=dict(size=13, color=TEXT), x=0),
        barmode="overlay",
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="True Score",
                   title_font=dict(color=MUTED, size=11), range=[0, 1]),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Count",
                   title_font=dict(color=MUTED, size=11)),
        showlegend=True,
        legend=dict(bgcolor=DARK, bordercolor=BORDER, font=dict(color=MUTED, size=10),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=250,
    )
    return fig


def _stat_card(val: str, label: str, color: str, sub: str = "") -> str:
    return f"""<div style="background:#111;border:1px solid #222;border-radius:6px;
        padding:10px 14px;flex:1;min-width:80px">
      <div style="font-size:18px;font-weight:700;color:#e5e5e5;font-variant-numeric:tabular-nums;
           white-space:nowrap;letter-spacing:-.02em;line-height:1.2">{val}</div>
      <div style="font-size:9.5px;color:#444;text-transform:uppercase;letter-spacing:.1em;
           margin-top:4px;font-weight:700">{label}</div>
    </div>"""


def _stat_strip_html(env: HiringEnv) -> str:
    s = env.state()
    cfg = s["task_config"]
    hired    = len(s["hired_ids"])
    rejected = len(s["rejected_ids"])
    total    = cfg["num_candidates"]
    pending  = total - hired - rejected
    pct_done = round((hired + rejected) / max(total, 1) * 100)
    cards = "".join([
        _stat_card(str(total),                "Pool",       ""),
        _stat_card(f"{hired}/{cfg['num_to_hire']}", "Hired",  ""),
        _stat_card(str(rejected),             "Rejected",   ""),
        _stat_card(str(pending),              "Pending",    ""),
        _stat_card(f"{pct_done}%",            "Screened",   ""),
        _stat_card(cfg["difficulty"].title(), "Difficulty", ""),
        _stat_card(f"α={cfg['alpha']}",       "Alpha",      ""),
    ])
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 10px;font-family:Inter,sans-serif">{cards}</div>'



def _stat_strip_html(env: HiringEnv) -> str:
    s = env.state()
    cfg = s["task_config"]
    hired    = len(s["hired_ids"])
    rejected = len(s["rejected_ids"])
    total    = cfg["num_candidates"]
    pending  = total - hired - rejected
    pct_done = round((hired + rejected) / max(total, 1) * 100)
    diff_color = {"easy": GREEN, "medium": AMBER, "hard": RED}.get(cfg["difficulty"], MUTED)
    cards = "".join([
        _stat_card(str(total),                  "Pool",       "#64748b"),
        _stat_card(str(hired),                  "Hired",      GREEN,   f"of {cfg['num_to_hire']} target"),
        _stat_card(str(rejected),               "Rejected",   RED),
        _stat_card(str(pending),                "Pending",    "#64748b"),
        _stat_card(f"{pct_done}%",              "Screened",   INDIGO),
        _stat_card(cfg["difficulty"].title(),   "Difficulty", diff_color),
        _stat_card(f"α={cfg['alpha']}",         "Alpha",      AMBER,   "quality weight"),
    ])
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 12px;font-family:Inter,system-ui">{cards}</div>'



# ─── Action handlers ────────────────────────────────────────────────────────

def _all_outputs(env: HiringEnv, msg: str):
    q, f, combined, is_final = _live_scores(env)
    _step_history.append(env.state()["current_step"])
    _reward_history.append(combined)
    return (
        _table_rows(env),
        _stat_strip_html(env),
        _gauge_chart(q, f, combined, is_final),
        _demo_parity_chart(env),
        _score_progression_chart(),
        _quality_dist_chart(env),
        msg,
    )

ALL_OUT_COUNT = 7  # keep in sync with above tuple

def do_reset(task_label: str):
    tid = TASK_LABEL_TO_ID.get(task_label, "easy")
    env = _reset_env(tid)
    q, f, combined, is_final = _live_scores(env)
    return (
        _table_rows(env),
        _stat_strip_html(env),
        _gauge_chart(q, f, combined, is_final),
        _demo_parity_chart(env),
        _score_progression_chart(),
        _quality_dist_chart(env),
        "Environment reset",
    )

def do_interview(cid: int):
    env = _get_env()
    if env.state()["done"]: return _all_outputs(env, "Episode complete — reset to continue")
    _, _, _, info = env.step(Action(action_type=ActionType.INTERVIEW,
                                    candidate_id=int(cid), reasoning="Structured interview"))
    return _all_outputs(env, info.get("message", info.get("error", "")))

def do_hire(cid: int, reasoning: str):
    env = _get_env()
    if env.state()["done"]: return _all_outputs(env, "Episode complete — reset to continue")
    _, _, _, info = env.step(Action(action_type=ActionType.HIRE,
                                    candidate_id=int(cid), reasoning=reasoning or "Strong candidate"))
    return _all_outputs(env, info.get("message", info.get("error", "")))

def do_reject(cid: int, reasoning: str):
    env = _get_env()
    if env.state()["done"]: return _all_outputs(env, "Episode complete — reset to continue")
    _, _, _, info = env.step(Action(action_type=ActionType.REJECT,
                                    candidate_id=int(cid), reasoning=reasoning or "Does not meet criteria"))
    return _all_outputs(env, info.get("message", info.get("error", "")))

def do_finalize():
    env = _get_env()
    if env.state()["done"]: return _all_outputs(env, "Already finalized")
    _, reward, _, info = env.step(Action(action_type=ActionType.FINALIZE, reasoning="User finalized"))
    r = env.state()["final_reward"]
    msg = f"Final scores — Quality {r['quality_score']:.3f}  Fairness {r['fairness_score']:.3f}  Combined {r['combined_reward']:.3f}"
    return _all_outputs(env, msg)

def do_heuristic(task_label: str):
    from demo import heuristic_agent
    tid = TASK_LABEL_TO_ID.get(task_label, "easy")
    env = _reset_env(tid)
    heuristic_agent(env)
    r = env.state()["final_reward"]
    # Record the final scores in history
    _step_history.append(env.state()["current_step"])
    _reward_history.append(r["combined_reward"])
    msg = f"Heuristic agent — Quality {r['quality_score']:.3f}  Fairness {r['fairness_score']:.3f}  Combined {r['combined_reward']:.3f}"
    return _all_outputs(env, msg)

# ─── Custom candidate ───────────────────────────────────────────────────────

def add_custom_candidate(name: str, experience: int, education: str, group: str, skills_raw: str):
    global _custom_id_counter, _custom_candidates
    if not name.strip():
        return "Name is required", _custom_table_rows()
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()] or ["Python"]
    edu_w = {"high_school": 0.1, "bachelors": 0.4, "masters": 0.7, "phd": 0.9}
    gt = round(0.35 * min(len(skills)/6,1) + 0.35 * min(int(experience)/10,1) + 0.30 * edu_w.get(education,0.3), 4)
    c = Candidate(id=_custom_id_counter, name=name.strip(),
                  years_experience=int(experience), skills=skills,
                  education=education, demographic_group=group,
                  interview_score=None, ground_truth_score=gt)
    _custom_candidates.append(c)
    _custom_id_counter += 1
    _get_env()._candidates.append(c)
    return f"Added {name.strip()} (ID {c.id})", _custom_table_rows()

def clear_custom():
    global _custom_candidates
    _custom_candidates = []
    return "Cleared all custom candidates", _custom_table_rows()

def _custom_table_rows() -> list[list]:
    return [[c.id, c.name, c.demographic_group, c.education.replace("_"," ").title(),
             c.years_experience, "  ".join(c.skills[:5]), f"{c.ground_truth_score:.3f}"]
            for c in _custom_candidates]

# ─── OpenEnv API ────────────────────────────────────────────────────────────

def api_reset(req: str) -> str:
    try:
        data = json.loads(req) if req.strip() else {}
        env = _reset_env(data.get("task_id", "easy"))
        return json.dumps({"status": "ok", "observation": env._make_observation().model_dump()},
                          indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)

def api_step(req: str) -> str:
    try:
        obs, reward, done, info = _get_env().step(Action(**json.loads(req)))
        return json.dumps({"status": "ok", "observation": obs.model_dump(),
                           "reward": reward.model_dump(), "done": done, "info": info},
                          indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)

def api_state() -> str:
    try:
        return json.dumps({"status": "ok", "state": _get_env().state()}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)

# ─── CSS ────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { background: #0a0a0a !important; font-family: 'Inter', sans-serif !important; color: #e5e5e5 !important; }
.gradio-container { max-width: 1440px !important; }
footer { display: none !important; }

.tab-nav { background: #0a0a0a !important; border-bottom: 1px solid #222 !important; }
.tab-nav button { color: #555 !important; font-size: 11px !important; font-weight: 600 !important;
  letter-spacing: .08em !important; text-transform: uppercase !important; border: none !important;
  padding: 10px 20px !important; background: transparent !important; transition: color .15s !important; }
.tab-nav button.selected { color: #fff !important; border-bottom: 2px solid #fff !important; }
.tab-nav button:hover:not(.selected) { color: #aaa !important; }

label > span { color: #555 !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
input, textarea, select { background: #111 !important; border: 1px solid #222 !important; color: #e5e5e5 !important; border-radius: 6px !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important; }
input:focus, textarea:focus { border-color: #555 !important; outline: none !important; }

button.primary { background: #fff !important; color: #000 !important; border: none !important; border-radius: 6px !important; font-weight: 700 !important; font-size: 13px !important; transition: opacity .15s !important; }
button.primary:hover { opacity: .85 !important; }
button.secondary { background: #111 !important; border: 1px solid #222 !important; color: #aaa !important; border-radius: 6px !important; font-size: 13px !important; font-weight: 500 !important; transition: border-color .15s !important; }
button.secondary:hover { border-color: #555 !important; color: #e5e5e5 !important; }
button.stop { background: transparent !important; border: 1px solid #333 !important; color: #e5e5e5 !important; border-radius: 6px !important; font-size: 13px !important; transition: border-color .15s !important; }
button.stop:hover { border-color: #888 !important; }

.dataframe th { background: #0a0a0a !important; color: #444 !important; font-size: 10px !important; font-weight: 700 !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-bottom: 1px solid #222 !important; padding: 10px 14px !important; }
.dataframe td { background: #0a0a0a !important; color: #ccc !important; font-size: 12.5px !important; border-bottom: 1px solid #111 !important; padding: 8px 14px !important; font-variant-numeric: tabular-nums !important; }
.dataframe tr:hover td { background: #111 !important; }

.prose h1, .prose h2, .prose h3 { color: #fff !important; border-bottom: 1px solid #222 !important; padding-bottom: 6px !important; }
.prose p, .prose li { color: #888 !important; line-height: 1.7 !important; font-size: 13px !important; }
.prose code { background: #111 !important; color: #ccc !important; border: 1px solid #222 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; border-radius: 4px !important; padding: 1px 5px !important; }
.prose pre { background: #111 !important; border: 1px solid #222 !important; border-radius: 8px !important; padding: 14px !important; }
.prose table th { background: #111 !important; color: #555 !important; font-size: 11px !important; font-weight: 700 !important; padding: 8px 12px !important; border-bottom: 1px solid #222 !important; }
.prose table td { border-bottom: 1px solid #111 !important; color: #888 !important; padding: 8px 12px !important; }
.prose strong { color: #e5e5e5 !important; }
.code-wrap code, pre { background: #111 !important; border: 1px solid #222 !important; border-radius: 8px !important; color: #ccc !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }
"""



HEADER = """
<div style="font-family:'Inter',sans-serif;padding:22px 8px 18px;border-bottom:1px solid #222;margin-bottom:4px">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <div>
      <div style="font-size:22px;font-weight:700;color:#fff;letter-spacing:-.03em;line-height:1.1">HireNFire</div>
      <div style="font-size:11.5px;color:#555;margin-top:3px;letter-spacing:.03em">Bias-Aware Hiring Screener &nbsp;·&nbsp; OpenEnv Submission</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
      <span style="border:1px solid #333;color:#aaa;font-size:10px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;padding:3px 10px;border-radius:4px">OpenEnv</span>
      <span style="border:1px solid #333;color:#aaa;font-size:10px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;padding:3px 10px;border-radius:4px">RL Agent</span>
      <span style="border:1px solid #333;color:#aaa;font-size:10px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;padding:3px 10px;border-radius:4px">Meta × HF</span>
    </div>
  </div>
  <div style="margin-top:12px;display:flex;align-items:center;gap:20px;flex-wrap:wrap">
    <div style="font-size:12px;color:#555;line-height:1.65;max-width:520px">
      AI agent screens candidates, conducts interviews, and makes hiring decisions.
      Scored on <span style="color:#ccc">quality of hire</span> and <span style="color:#ccc">demographic fairness</span>.
      Three difficulty tiers with a full OpenEnv-compatible JSON API.
    </div>
    <code style="font-size:12px;background:#111;border:1px solid #222;border-radius:5px;padding:5px 12px;color:#ccc;white-space:nowrap;font-family:'JetBrains Mono',monospace">
      R = α·Q + (1−α)·F
    </code>
  </div>
</div>
"""


# ─── Build UI ────────────────────────────────────────────────────────────────


TASK_LABELS = list(TASK_LABEL_TO_ID.keys())

with gr.Blocks(title="HireNFire", css=CSS) as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Live Environment ──────────────────────────────────────
        with gr.Tab("Environment"):
            with gr.Row(equal_height=False):

                # LEFT sidebar
                with gr.Column(scale=1, min_width=220):
                    task_dd = gr.Dropdown(choices=TASK_LABELS, value=TASK_LABELS[0], label="Task")
                    reset_btn = gr.Button("Reset", variant="primary", size="lg")
                    gr.HTML('<div style="height:1px;background:#222;margin:12px 0"></div>')

                    gr.HTML('<div style="font-size:10px;color:#444;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-bottom:8px">Actions</div>')
                    cid_input = gr.Number(label="Candidate ID", value=0, precision=0)
                    reasoning_input = gr.Textbox(label="Reasoning", lines=2, placeholder="optional")
                    with gr.Row():
                        interview_btn = gr.Button("Interview", variant="secondary", size="sm")
                        hire_btn      = gr.Button("Hire",      variant="primary",   size="sm")
                    with gr.Row():
                        reject_btn  = gr.Button("Reject",    variant="stop",      size="sm")
                        finalize_btn = gr.Button("Finalize", variant="secondary", size="sm")
                    gr.HTML('<div style="height:1px;background:#222;margin:12px 0"></div>')
                    heuristic_btn = gr.Button("Run Heuristic Agent", variant="secondary")
                    action_msg = gr.Textbox(label="Last Result", interactive=False, lines=2)

                # CENTER — table + stat strip
                with gr.Column(scale=3):
                    stat_strip = gr.HTML()
                    candidates_table = gr.Dataframe(
                        headers=["ID", "Name", "Group", "Education", "Exp yr", "Skills", "Interview", "Status"],
                        label="Candidate Pool",
                        interactive=False,
                        wrap=False,
                    )
                    with gr.Row():
                        parity_chart   = gr.Plot(label="Demographic Parity")
                        quality_chart  = gr.Plot(label="Quality Distribution")

                # RIGHT — scores + trajectory
                with gr.Column(scale=2, min_width=280):
                    gauge_chart = gr.Plot(label="Scores")
                    traj_chart  = gr.Plot(label="Reward Trajectory")

            all_outputs = [candidates_table, stat_strip, gauge_chart,
                           parity_chart, traj_chart, quality_chart, action_msg]

            reset_btn.click(do_reset,     inputs=[task_dd],                      outputs=all_outputs)
            interview_btn.click(do_interview, inputs=[cid_input],                outputs=all_outputs)
            hire_btn.click(do_hire,       inputs=[cid_input, reasoning_input],   outputs=all_outputs)
            reject_btn.click(do_reject,   inputs=[cid_input, reasoning_input],   outputs=all_outputs)
            finalize_btn.click(do_finalize,                                      outputs=all_outputs)
            heuristic_btn.click(do_heuristic, inputs=[task_dd],                  outputs=all_outputs)

        # ── Tab 2: Add Candidate ─────────────────────────────────────────
        with gr.Tab("Add Candidate"):
            gr.HTML("""<div style="font-family:Inter,system-ui;padding:16px 0 12px">
              <div style="font-size:15px;font-weight:600;color:#f1f5f9">Custom Candidates</div>
              <div style="font-size:12px;color:#4b5563;margin-top:4px">
                Add your own candidate to the live pool and interact with them through the Environment tab<br>
                Custom candidates are marked with ★ in the pool
              </div></div>""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    with gr.Row():
                        cc_name = gr.Textbox(label="Full Name", placeholder="eg Jordan Chen")
                        cc_exp  = gr.Slider(label="Years of Experience", minimum=0, maximum=25, value=4, step=1)
                    with gr.Row():
                        cc_edu   = gr.Dropdown(label="Education", choices=["bachelors","masters","phd","high_school"], value="bachelors")
                        cc_group = gr.Dropdown(label="Demographic Group", choices=["A","B","C","D"], value="A",
                                               info="Abstract label used only for fairness grading")
                    cc_skills = gr.Textbox(label="Skills (comma separated)",
                                           placeholder="Python  PyTorch  Docker  SQL  React", lines=2)
                    with gr.Row():
                        add_btn   = gr.Button("Add to Pool", variant="primary")
                        clear_btn = gr.Button("Clear All", variant="secondary")
                    cc_msg = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=3):
                    gr.HTML('<div style="font-size:10px;color:#374151;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-bottom:8px">Custom Pool</div>')
                    custom_table = gr.Dataframe(
                        headers=["ID","Name","Group","Education","Exp yr","Skills","Est Score"],
                        label="", interactive=False,
                    )
                    gr.HTML("""<div style="font-size:11px;color:#374151;margin-top:8px;line-height:1.6">
                      Est Score is computed from skills count  experience  and education<br>
                      It is hidden from the agent until after an INTERVIEW action
                    </div>""")
            add_btn.click(add_custom_candidate, inputs=[cc_name,cc_exp,cc_edu,cc_group,cc_skills], outputs=[cc_msg,custom_table])
            clear_btn.click(clear_custom, outputs=[cc_msg, custom_table])

        # ── Tab 3: Tasks & Graders ────────────────────────────────────────
        with gr.Tab("Tasks"):
            gr.HTML("""
<div style="font-family:'Inter',system-ui;padding:20px 4px 8px;max-width:960px">

  <div style="font-size:18px;font-weight:700;color:#f8fafc;letter-spacing:-.02em;margin-bottom:4px">Task Definitions</div>
  <div style="font-size:12.5px;color:#374151;margin-bottom:24px;line-height:1.6">
    Three difficulty tiers test different agent capabilities from simple ranking to fair hiring at scale.
    Each task uses a different role, candidate pool size, and alpha weighting between quality and fairness.
  </div>

  <!-- Task Cards -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:28px">

    <!-- Easy -->
    <div style="background:#080d1a;border:1px solid #1a2235;border-top:3px solid #10b981;border-radius:12px;padding:20px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        <span style="background:rgba(16,185,129,.12);color:#34d399;font-size:9px;font-weight:700;
             letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;
             border:1px solid rgba(16,185,129,.2)">Easy</span>
        <span style="font-size:13px;font-weight:600;color:#f1f5f9">Simple Ranking</span>
      </div>
      <div style="font-size:11.5px;color:#4b5563;line-height:1.65;margin-bottom:14px">
        Rank 10 Software Engineer candidates by explicit criteria. Clear skill gaps make the optimal
        ranking obvious. A good starting point to understand the environment mechanics.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
        <div style="background:#060c1a;border:1px solid #1a2235;border-radius:7px;padding:8px 10px">
          <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:2px">Role</div>
          <div style="font-size:11.5px;color:#94a3b8;font-weight:500">Software Engineer</div>
        </div>
        <div style="background:#060c1a;border:1px solid #1a2235;border-radius:7px;padding:8px 10px">
          <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:2px">Pool → Hire</div>
          <div style="font-size:11.5px;color:#94a3b8;font-weight:500">10 candidates → 3</div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
        <span style="background:rgba(99,102,241,.1);color:#818cf8;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(99,102,241,.2)">α = 0.7</span>
        <span style="background:rgba(16,185,129,.08);color:#34d399;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(16,185,129,.15)">70% Quality</span>
        <span style="background:rgba(20,184,166,.08);color:#2dd4bf;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(20,184,166,.15)">30% Fairness</span>
      </div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5">
        <span style="color:#4b5563;font-weight:600">Grader:</span> F1 vs ground truth top 3 + demographic parity distance
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid #1a2235;display:flex;gap:12px">
        <div><div style="font-size:15px;font-weight:700;color:#10b981">1.000</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Quality</div></div>
        <div><div style="font-size:15px;font-weight:700;color:#14b8a6">0.500</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Fairness</div></div>
        <div><div style="font-size:15px;font-weight:700;color:#f59e0b">0.850</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Combined</div></div>
        <div style="margin-left:auto;font-size:9px;color:#374151;align-self:flex-end;text-align:right">heuristic<br>baseline</div>
      </div>
    </div>

    <!-- Medium -->
    <div style="background:#080d1a;border:1px solid #1a2235;border-top:3px solid #f59e0b;border-radius:12px;padding:20px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        <span style="background:rgba(245,158,11,.12);color:#fbbf24;font-size:9px;font-weight:700;
             letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;
             border:1px solid rgba(245,158,11,.2)">Medium</span>
        <span style="font-size:13px;font-weight:600;color:#f1f5f9">High-Potential Screening</span>
      </div>
      <div style="font-size:11.5px;color:#4b5563;line-height:1.65;margin-bottom:14px">
        Some candidates are underqualified on paper but show high interview potential.
        Agent must weigh hard criteria against latent potential signals revealed by INTERVIEW.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
        <div style="background:#060c1a;border:1px solid #1a2235;border-radius:7px;padding:8px 10px">
          <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:2px">Role</div>
          <div style="font-size:11.5px;color:#94a3b8;font-weight:500">Data Scientist</div>
        </div>
        <div style="background:#060c1a;border:1px solid #1a2235;border-radius:7px;padding:8px 10px">
          <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:2px">Pool → Hire</div>
          <div style="font-size:11.5px;color:#94a3b8;font-weight:500">25 candidates → 5</div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
        <span style="background:rgba(99,102,241,.1);color:#818cf8;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(99,102,241,.2)">α = 0.5</span>
        <span style="background:rgba(16,185,129,.08);color:#34d399;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(16,185,129,.15)">50% Quality</span>
        <span style="background:rgba(20,184,166,.08);color:#2dd4bf;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(20,184,166,.15)">50% Fairness</span>
      </div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5">
        <span style="color:#4b5563;font-weight:600">Grader:</span> F1 vs top 5 + demographic parity. Interview reveals hidden potential.
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid #1a2235;display:flex;gap:12px">
        <div><div style="font-size:15px;font-weight:700;color:#10b981">0.800</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Quality</div></div>
        <div><div style="font-size:15px;font-weight:700;color:#14b8a6">0.500</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Fairness</div></div>
        <div><div style="font-size:15px;font-weight:700;color:#f59e0b">0.650</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Combined</div></div>
        <div style="margin-left:auto;font-size:9px;color:#374151;align-self:flex-end;text-align:right">heuristic<br>baseline</div>
      </div>
    </div>

    <!-- Hard -->
    <div style="background:#080d1a;border:1px solid #1a2235;border-top:3px solid #ef4444;border-radius:12px;padding:20px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        <span style="background:rgba(239,68,68,.12);color:#f87171;font-size:9px;font-weight:700;
             letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:99px;
             border:1px solid rgba(239,68,68,.2)">Hard</span>
        <span style="font-size:13px;font-weight:600;color:#f1f5f9">Fair Hiring at Scale</span>
      </div>
      <div style="font-size:11.5px;color:#4b5563;line-height:1.65;margin-bottom:14px">
        Pool has deliberate demographic skew group A overrepresented by design.
        Qualification margins are tight. Achieving both quality and parity is genuinely difficult.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
        <div style="background:#060c1a;border:1px solid #1a2235;border-radius:7px;padding:8px 10px">
          <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:2px">Role</div>
          <div style="font-size:11.5px;color:#94a3b8;font-weight:500">Senior ML Engineer</div>
        </div>
        <div style="background:#060c1a;border:1px solid #1a2235;border-radius:7px;padding:8px 10px">
          <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:2px">Pool → Hire</div>
          <div style="font-size:11.5px;color:#94a3b8;font-weight:500">50 candidates → 8</div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
        <span style="background:rgba(99,102,241,.1);color:#818cf8;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(99,102,241,.2)">α = 0.3</span>
        <span style="background:rgba(16,185,129,.08);color:#34d399;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(16,185,129,.15)">30% Quality</span>
        <span style="background:rgba(20,184,166,.08);color:#2dd4bf;font-size:10px;padding:2px 8px;border-radius:5px;border:1px solid rgba(20,184,166,.15)">70% Fairness</span>
      </div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5">
        <span style="color:#4b5563;font-weight:600">Grader:</span> F1 vs top 8 + parity. Injected group A score boost makes this hard by design.
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid #1a2235;display:flex;gap:12px">
        <div><div style="font-size:15px;font-weight:700;color:#10b981">0.375</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Quality</div></div>
        <div><div style="font-size:15px;font-weight:700;color:#14b8a6">0.714</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Fairness</div></div>
        <div><div style="font-size:15px;font-weight:700;color:#f59e0b">0.613</div><div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:.08em">Combined</div></div>
        <div style="margin-left:auto;font-size:9px;color:#374151;align-self:flex-end;text-align:right">heuristic<br>baseline</div>
      </div>
    </div>
  </div>

  <!-- Reward Function -->
  <div style="background:#080d1a;border:1px solid #1a2235;border-radius:12px;padding:20px;margin-bottom:20px">
    <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:14px">Reward Function</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div>
        <div style="font-size:10px;color:#374151;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Formula</div>
        <pre style="background:#060c1a;border:1px solid #1a2235;border-radius:8px;padding:14px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#86efac;margin:0;overflow-x:auto">R = α × quality + (1−α) × fairness

quality  = F1(hired, ground_truth_top_K)
         = 2·precision·recall / (P+R)

fairness = 1 − (max_rate − min_rate)</pre>
      </div>
      <div style="font-size:12px;color:#4b5563;line-height:1.7">
        <div style="font-size:10px;color:#374151;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Notes</div>
        <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:7px">
          <li style="display:flex;gap:8px"><span style="color:#818cf8;flex-shrink:0">›</span> Partial rewards emitted at every step not just episode end</li>
          <li style="display:flex;gap:8px"><span style="color:#818cf8;flex-shrink:0">›</span> Step penalty activates after step 30 to discourage infinite loops</li>
          <li style="display:flex;gap:8px"><span style="color:#818cf8;flex-shrink:0">›</span> INTERVIEW reveals a noisy score before HIRE / REJECT decisions</li>
          <li style="display:flex;gap:8px"><span style="color:#818cf8;flex-shrink:0">›</span> Ground truth scores are hidden from the agent evaluator only</li>
        </ul>
      </div>
    </div>
  </div>

</div>
""")

        # ── Tab 4: OpenEnv API ────────────────────────────────────────────
        with gr.Tab("API"):
            gr.HTML("""<div style="font-family:'Inter',system-ui;padding:20px 4px 8px">
              <div style="font-size:18px;font-weight:700;color:#f8fafc;letter-spacing:-.02em;margin-bottom:4px">OpenEnv JSON API</div>
              <div style="font-size:12.5px;color:#374151;line-height:1.6;max-width:640px">
                Full <code style="background:#0d1525;color:#93c5fd;border:1px solid #1a2235;border-radius:4px;padding:1px 6px;font-family:'JetBrains Mono',monospace;font-size:11px">step()</code> /
                <code style="background:#0d1525;color:#93c5fd;border:1px solid #1a2235;border-radius:4px;padding:1px 6px;font-family:'JetBrains Mono',monospace;font-size:11px">reset()</code> /
                <code style="background:#0d1525;color:#93c5fd;border:1px solid #1a2235;border-radius:4px;padding:1px 6px;font-family:'JetBrains Mono',monospace;font-size:11px">state()</code>
                API — used by automated validators and LLM agent runners. Returns JSON-serializable observations.
              </div>
            </div>""")

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div style="font-size:10px;color:#374151;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-bottom:6px">reset(task_id)</div>')
                    api_reset_in  = gr.Textbox(value='{\n  "task_id": "easy"\n}', lines=4, label="Request")
                    api_reset_btn = gr.Button("Call reset()", variant="primary", size="sm")
                    reset_out     = gr.Code(language="json", label="Response")
                    api_reset_btn.click(api_reset, inputs=[api_reset_in], outputs=[reset_out])

                with gr.Column():
                    gr.HTML('<div style="font-size:10px;color:#374151;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-bottom:6px">step(action)</div>')
                    api_step_in  = gr.Textbox(
                        value='{\n  "action_type": "INTERVIEW",\n  "candidate_id": 0,\n  "reasoning": "checking skills fit"\n}',
                        lines=7, label="Action JSON",
                    )
                    api_step_btn = gr.Button("Call step()", variant="primary", size="sm")
                    step_out     = gr.Code(language="json", label="Response")
                    api_step_btn.click(api_step, inputs=[api_step_in], outputs=[step_out])

                with gr.Column():
                    gr.HTML('<div style="font-size:10px;color:#374151;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-bottom:6px">state()</div>')
                    api_state_btn = gr.Button("Call state()", variant="primary", size="sm")
                    state_out     = gr.Code(language="json", label="Response")
                    api_state_btn.click(api_state, outputs=[state_out])

# ─── FastAPI REST API (OpenEnv validator endpoints) ──────────────────────────
# The OpenEnv platform checks POST /reset returns HTTP 200 with JSON.
# We mount these as proper REST routes on the Gradio ASGI app.

app = FastAPI()


@app.post("/reset")
async def rest_reset(request: Request):
    """OpenEnv reset endpoint — POST /reset with optional {task_id: str}."""
    try:
        body = await request.body()
        data = json.loads(body) if body.strip() else {}
        task_id = data.get("task_id", "easy")
        if task_id not in _task_map:
            task_id = "easy"
        env = _reset_env(task_id)
        obs = env._make_observation()
        return JSONResponse({
            "status": "ok",
            "task_id": task_id,
            "observation": obs.model_dump(),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/step")
async def rest_step(request: Request):
    """OpenEnv step endpoint — POST /step with Action JSON."""
    try:
        body = await request.body()
        data = json.loads(body)
        obs, reward, done, info = _get_env().step(Action(**data))
        return JSONResponse({
            "status": "ok",
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/state")
async def rest_state():
    """OpenEnv state endpoint — GET /state."""
    try:
        return JSONResponse({"status": "ok", "state": _get_env().state()})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/health")
async def health():
    """Health check."""
    return JSONResponse({"status": "ok"})


# Mount Gradio onto the FastAPI app at root
app = gr.mount_gradio_app(app, demo, path="/")


# ─── Launch ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    _env = HiringEnv(EASY_TASK, seed=42)
    _env.reset()
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
