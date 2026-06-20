"""Decision Node — gates the rewrite and selects an explainable strategy.

Separating *what to do* (this node) from *how to do it* (the rewrite node) keeps
the agent auditable: every run records why a rewrite was or wasn't triggered and
which scoped strategy was chosen.

Reads:  overall_match_score, missing_required_skills, ats_match_score
Writes: rewrite_required, rewrite_reason, rewrite_strategy
"""

from __future__ import annotations

from v_final.config import settings
from v_final.state.job_application_state import JobApplicationState


def decision_node(state: JobApplicationState) -> JobApplicationState:
    cfg = settings.decision
    overall = float(state.get("overall_match_score", 0.0))
    ats = float(state.get("ats_match_score", 0.0))
    missing = state.get("missing_required_skills", []) or []

    strong_enough = overall >= cfg.overall_threshold and len(missing) <= cfg.max_missing_required

    if strong_enough:
        state["rewrite_required"] = False
        state["rewrite_strategy"] = "none"
        state["rewrite_reason"] = (
            f"Strong fit (overall={overall:.2f} ≥ {cfg.overall_threshold:.2f}, "
            f"{len(missing)} missing required skill(s)); no rewrite needed."
        )
        return state

    # Rewrite warranted — pick the narrowest effective scope.
    if overall < 0.40:
        strategy = "full_rewrite"
    elif len(missing) > cfg.max_missing_required:
        strategy = "summary_and_skills"
    elif ats < 0.50:
        strategy = "summary_only"
    else:
        strategy = "experience_bullets"

    reasons = [f"overall={overall:.2f} < {cfg.overall_threshold:.2f}"]
    if len(missing) > cfg.max_missing_required:
        reasons.append(f"{len(missing)} required skills under-represented: {', '.join(missing[:5])}")
    if ats < 0.50:
        reasons.append(f"low ATS keyword coverage ({ats:.2f})")

    state["rewrite_required"] = True
    state["rewrite_strategy"] = strategy
    state["rewrite_reason"] = f"Rewrite ({strategy}): " + "; ".join(reasons) + "."
    return state
