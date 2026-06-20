"""Programmatic entrypoint for the v_final agent.

Use :func:`run` to invoke the agent from code, or run this module directly
(``python -m v_final.pipeline``) to execute it on a bundled sample and print a
readable report.
"""

from __future__ import annotations

import logging

from v_final.config import settings
from v_final.graph.job_application_graph import graph
from v_final.state.job_application_state import JobApplicationState


def run(
    resume_text: str,
    job_description_text: str,
    *,
    resume_source: str = "manual",
    resume_pdf_path: str | None = None,
) -> JobApplicationState:
    """Run the full agent and return the final state."""
    initial: JobApplicationState = {
        "resume_raw_text": resume_text,
        "job_description_text": job_description_text,
        "resume_source": resume_source,
    }
    if resume_pdf_path:
        initial["resume_pdf_path"] = resume_pdf_path
    return graph.invoke(initial)


def format_report(state: JobApplicationState) -> str:
    sb = state.get("score_breakdown", {}) or {}
    lines = [
        "=" * 60,
        "JOB APPLICATION AGENT - RUN REPORT",
        "=" * 60,
        f"JD title      : {state.get('jd_title') or '(unknown)'}",
        f"JD domain     : {state.get('jd_domain')}   seniority: {state.get('jd_seniority_level')}",
        "",
        "RESUME STRUCTURE",
        f"  experience : {len(state.get('experience_entries', []) or [])} entries "
        f"({', '.join(e.get('company', '?') for e in state.get('experience_entries', []) or []) or '-'})",
        f"  projects   : {len(state.get('project_entries', []) or [])} entries",
        f"  education  : {len(state.get('education_entries', []) or [])} entries",
        "",
        "SCORES",
        f"  skill match     : {sb.get('skill_match', state.get('skill_match_score'))}",
        f"  semantic sim    : {sb.get('semantic_similarity', state.get('semantic_similarity_score'))}",
        f"  ats match       : {sb.get('ats_match', state.get('ats_match_score'))}",
        f"  OVERALL         : {state.get('overall_match_score')}",
        f"  ATS pass chance : {state.get('ats_pass_score')}  ({state.get('ats_pass_label')})",
        "",
        f"Matched skills  : {', '.join(state.get('matched_skills', []) or []) or '(none)'}",
        f"Missing required: {', '.join(state.get('missing_required_skills', []) or []) or '(none)'}",
        "",
        "DECISION",
        f"  rewrite_required: {state.get('rewrite_required')}",
        f"  strategy        : {state.get('rewrite_strategy')}",
        f"  reason          : {state.get('rewrite_reason')}",
        f"  resume_version  : {state.get('resume_version')}",
    ]

    cmp = state.get("rewrite_score_comparison")
    if cmp:
        lines += ["", f"REWRITE IMPACT  (best of {state.get('rewrite_attempts', 0)} attempt(s))"]
        for label, key in [("ATS", "ats_match"), ("ATS pass", "ats_pass"), ("overall", "overall"),
                           ("skill", "skill_match"), ("semantic", "semantic_similarity")]:
            c = cmp.get(key)
            if c:
                d = c.get("delta", 0.0)
                sign = "+" if d > 0 else ""
                lines.append(f"  {label:9s}: {c.get('before')} -> {c.get('after')}  ({sign}{d})")

    if state.get("warnings"):
        lines += ["", "WARNINGS:"] + [f"  - {w}" for w in state["warnings"]]
    if state.get("run_artifact_path"):
        lines += ["", f"Artifact: {state['run_artifact_path']}"]
    lines.append("=" * 60)
    return "\n".join(lines)


# A small, realistic sample so the agent runs out of the box.
_SAMPLE_RESUME = """
Data Scientist with 3+ years of experience in Python, machine learning, and AWS.
SKILLS
Machine Learning: classification, regression, feature engineering
Cloud & MLOps: AWS (Lambda, SageMaker, S3), Docker, CI/CD (GitHub Actions)
Data Engineering: PySpark, Snowflake, SQL
EXPERIENCE
Acme Corp
Data Scientist
- Built end-to-end ML pipelines and deployed models using Docker and SageMaker.
- Developed SQL data pipelines to support model training and validation.
EDUCATION
M.S. Information Systems
""".strip()

_SAMPLE_JD = """
We are looking for a Data Scientist with strong Python skills, experience in AWS,
SQL, machine learning, and data pipelines. Experience deploying models with Docker
is required. Familiarity with PySpark and feature engineering is a plus.
""".strip()


def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    state = run(_SAMPLE_RESUME, _SAMPLE_JD, resume_source="sample")
    print(format_report(state))


if __name__ == "__main__":
    main()
