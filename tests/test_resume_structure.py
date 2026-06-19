"""Structured-parsing tests against a realistic multi-section resume."""

from utils.resume_parsing import (
    parse_education_entries,
    parse_experience_entries,
    parse_project_entries,
)
from v_final.nodes.resume_ingest import resume_ingest_node
from v_final.nodes.resume_structure import resume_structure_node

SAMPLE = """
MAHESH PONNAM
PROFESSIONAL SUMMARY
Data Scientist with 3+ years of experience in credit risk and fraud analytics.
SKILLS
Machine Learning: Classification, Feature Engineering
Cloud & MLOps: AWS, Docker
PROFESSIONAL EXPERIENCE
Northern Trust
Data Scientist
USA
February 2025 - Present
•  Built ESG and factor data pipelines using Python and AWS Glue to improve data accuracy by 18
percent and reduce manual reconciliation.
•  Designed near real time ingestion using AWS Lambda and S3 to enable scalable analytics.
Mphasis
Data Scientist
India
April 2020 - November 2022
•  Led development of credit risk and fraud models that improved approval accuracy by 18 percent.
•  Built scalable ETL pipelines using PySpark and SQL to support model training.
PROJECTS & OUTSIDE EXPERIENCE
End to End Credit Risk Prediction System-  Link to project
•  Built an end to end credit risk system to predict probability of loan default, covering
ingestion and production inference service.
•  Trained and compared Logistic Regression, Random Forest, and XGBoost.
EDUCATION
Wilmington University
Master's, Information Systems
Vignana Bharathi Institute of Technology, India
Master's, Computer Engineering
July 2022 - December 2024
July 2018 - July 2020
""".strip()


def _structured():
    state = resume_ingest_node({"resume_raw_text": SAMPLE, "resume_source": "test"})
    return resume_structure_node(state)


def test_experience_entries_parsed():
    out = _structured()
    exp = out["experience_entries"]
    companies = {e["company"] for e in exp}
    assert companies == {"Northern Trust", "Mphasis"}
    for e in exp:
        assert "Data Scientist" in e["role"]
        assert e["start_date"] and e["end_date"]
        assert len(e["bullets"]) == 2
    # wrapped bullet line was merged, not split into a phantom entry
    assert any("reconciliation" in b for e in exp for b in e["bullets"])


def test_project_entries_parsed():
    out = _structured()
    projects = out["project_entries"]
    assert len(projects) >= 1
    p = projects[0]
    assert "Credit Risk" in p["name"]
    assert "link" not in p["name"].lower()
    assert len(p["bullets"]) == 2


def test_education_entries_parsed():
    out = _structured()
    edu = out["education_entries"]
    assert len(edu) == 2
    assert all("Master" in e["degree"] for e in edu)
    institutions = {e["institution"] for e in edu}
    assert "Wilmington University" in institutions


def test_parsers_are_empty_safe():
    assert parse_experience_entries("") == []
    assert parse_project_entries("") == []
    assert parse_education_entries("") == []
