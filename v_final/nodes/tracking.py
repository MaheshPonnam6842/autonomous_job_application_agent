"""Tracking Node — persists an auditable run artifact.

Writes a compact JSON summary of each run (scores, decision, gaps, outputs) to
the artifacts directory so runs are reproducible and reviewable. This is the
agent's audit trail — useful for debugging, evals, and demos.

Reads:  most scoring/decision/output fields
Writes: run_artifact_path, agent_notes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from v_final.config import settings
from v_final.state.job_application_state import JobApplicationState

# Fields worth persisting (kept small and JSON-safe).
_SUMMARY_KEYS = (
    "jd_title", "jd_domain", "jd_seniority_level",
    "skill_match_score", "semantic_similarity_score", "ats_match_score",
    "overall_match_score", "score_breakdown",
    "matched_skills", "missing_required_skills",
    "skill_gap_hard", "skill_gap_soft",
    "rewrite_required", "rewrite_strategy", "rewrite_reason",
    "resume_version", "warnings",
    "rewrite_attempts", "rewrite_candidates", "rewrite_score_comparison",
)


def tracking_node(state: JobApplicationState) -> JobApplicationState:
    artifacts_dir = settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"timestamp_utc": ts}
    summary.update({k: state.get(k) for k in _SUMMARY_KEYS if k in state})
    # Include outputs but truncated, so artifacts stay small.
    summary["optimized_resume_preview"] = (state.get("optimized_resume_text", "") or "")[:600]
    summary["outreach_dm_text"] = state.get("outreach_dm_text", "")

    path = artifacts_dir / f"run_{ts}.json"
    try:
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        state["run_artifact_path"] = str(path)
        state.setdefault("agent_notes", []).append(f"Run artifact saved: {path}")
    except OSError as exc:
        state.setdefault("warnings", []).append(f"tracking_write_failed: {exc}")

    return state
