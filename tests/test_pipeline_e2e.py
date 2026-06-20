"""End-to-end test: the whole graph runs offline and produces a coherent state."""

from v_final.pipeline import run

RESUME = """
Data Scientist with 3+ years in Python, machine learning, and AWS.
SKILLS
Cloud & MLOps: AWS, Docker, CI/CD
Data Engineering: SQL, PySpark
EXPERIENCE
Acme Corp
Data Scientist
- Built and deployed ML pipelines using Docker and SQL.
""".strip()

JD = "Data Scientist needed with Python, SQL, AWS, machine learning, and Docker."


def test_pipeline_runs_offline_and_is_consistent():
    state = run(RESUME, JD, resume_source="test")

    # Core contract is populated
    assert "overall_match_score" in state
    assert "rewrite_required" in state
    assert state["resume_version"] in {"original", "reformatted", "rewritten"}

    # With the LLM disabled, the rewrite still assembles an ATS-clean resume
    assert state["optimized_resume_text"]

    # Outreach drafts always produced
    assert "{your_name}" in state["outreach_dm_text"]
    assert state["outreach_email_text"].startswith("Subject:")

    # Run artifact recorded
    assert state.get("run_artifact_path")


def test_llm_unavailable_is_recorded_not_raised():
    state = run(RESUME, JD)
    warnings = state.get("warnings", [])
    assert any("llm_unavailable" in w or "llm_unavailable" in w for w in warnings)
