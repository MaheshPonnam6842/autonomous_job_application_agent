from v_final.nodes.decision import decision_node
from v_final.nodes.matching import matching_node
from v_final.nodes.skill_extraction import skill_extraction_node


def _analyze(resume_text, jd_text, jd_required):
    state = {
        "resume_clean_text": resume_text,
        "jd_clean_text": jd_text,
        "jd_skills_required": jd_required,
    }
    state = skill_extraction_node(state)
    state = matching_node(state)
    return decision_node(state)


def test_scores_are_bounded():
    out = _analyze(
        "Python, SQL, AWS, Docker, machine learning.",
        "Looking for Python, SQL, AWS, Docker, machine learning.",
        ["python", "sql", "aws", "docker", "machine learning"],
    )
    for key in ("skill_match_score", "semantic_similarity_score", "ats_match_score", "overall_match_score"):
        assert 0.0 <= out[key] <= 1.0


def test_strong_match_skips_rewrite():
    out = _analyze(
        "Expert in Python, SQL, AWS, Docker, machine learning, feature engineering, snowflake.",
        "Python, SQL, AWS, Docker, machine learning required.",
        ["python", "sql", "aws", "docker", "machine learning"],
    )
    assert out["ats_match_score"] == 1.0
    assert out["rewrite_required"] is False
    assert out["rewrite_strategy"] == "none"


def test_weak_match_triggers_rewrite_with_strategy():
    out = _analyze(
        "Java developer with Spring and REST APIs.",
        "Need Python, SQL, AWS, Docker, Kubernetes, machine learning.",
        ["python", "sql", "aws", "docker", "kubernetes", "machine learning"],
    )
    assert out["rewrite_required"] is True
    assert out["rewrite_strategy"] in {
        "full_rewrite", "summary_and_skills", "summary_only", "experience_bullets",
    }
    assert "missing_required_skills" in out and len(out["missing_required_skills"]) > 0
