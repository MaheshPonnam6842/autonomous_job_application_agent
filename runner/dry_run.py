"""Dry run of the v_final agent on bundled sample inputs.

Equivalent to ``python -m v_final.pipeline``. Tip: set ``JOBAGENT_LLM_ENABLED=0``
to exercise the deterministic path with no model server running.
"""

from v_final.pipeline import format_report, run

_RESUME = """
Data Scientist with 3+ years of experience in Python, machine learning, and AWS.
Built end-to-end ML pipelines and deployed models using Docker.
SKILLS
Cloud & MLOps: AWS, Docker, CI/CD
Data Engineering: SQL, PySpark
""".strip()

_JD = """
We are looking for a Data Scientist with strong Python skills, experience in AWS,
SQL, machine learning, and data pipelines. Experience deploying models is a plus.
""".strip()


if __name__ == "__main__":
    state = run(_RESUME, _JD, resume_source="manual_upload")
    print(format_report(state))
