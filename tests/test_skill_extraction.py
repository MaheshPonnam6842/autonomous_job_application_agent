from v_final.nodes.skill_extraction import (
    _build_profile,
    _canon,
    _scan_vocab,
    skill_extraction_node,
)


def test_canon_maps_synonyms():
    assert _canon("ML") == "machine learning"
    assert _canon("Amazon Web Services") == "aws"
    assert _canon("scikit learn") == "scikit-learn"


def test_scan_vocab_respects_token_boundaries():
    found = set(_scan_vocab("Experienced in Python and React, strong SQL."))
    assert "python" in found
    assert "sql" in found
    # 'r' (R language) must not be matched inside 'react'
    assert "r" not in found


def test_build_profile_buckets_terms():
    profile = _build_profile(["python", "aws", "docker", "machine learning", "communication"])
    assert "python" in profile["hard_skills"]
    assert "aws" in profile["cloud"]
    assert "docker" in profile["tools"]
    assert "machine learning" in profile["ml_concepts"]
    assert "communication" in profile["soft_skills"]


def test_overlap_and_gaps():
    state = {
        "resume_clean_text": "Python developer with SQL and AWS experience.",
        "jd_clean_text": "Need Python, SQL, Docker and Kubernetes.",
        "jd_skills_required": ["python", "sql", "docker", "kubernetes"],
    }
    out = skill_extraction_node(state)
    matched = {s for vals in out["skill_overlap"].values() for s in vals}
    missing = {s for vals in out["skill_gap_hard"].values() for s in vals}
    assert {"python", "sql"} <= matched
    assert {"docker", "kubernetes"} <= missing
