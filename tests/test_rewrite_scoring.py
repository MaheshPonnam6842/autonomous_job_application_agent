"""Tests for the iterative rewrite-and-rescore loop (all offline)."""

from v_final.graph.rewrite_graph import build_rewrite_graph
from v_final.nodes.matching import ats_pass_estimate, ats_pass_label, score_resume_against_jd
from v_final.nodes.rewrite_loop import _is_better, improve_router

_JD_STATE = {
    "jd_clean_text": "python sql aws docker kubernetes",
    "jd_skill_profile": {"hard_skills": ["python", "sql"], "cloud": ["aws"], "tools": ["docker", "kubernetes"]},
    "jd_skills_required": ["python", "sql", "aws", "docker", "kubernetes"],
}


# ---- scorer ----------------------------------------------------------------

def test_score_is_bounded_and_monotonic_in_ats():
    low = score_resume_against_jd("python", _JD_STATE)
    high = score_resume_against_jd("python sql aws docker kubernetes", _JD_STATE)
    for s in (low, high):
        for k in ("skill_match", "semantic_similarity", "ats_match", "overall"):
            assert 0.0 <= s[k] <= 1.0
    assert high["ats_match"] > low["ats_match"]
    assert high["overall"] >= low["overall"]
    assert high["ats_pass"] >= low["ats_pass"]


def test_ats_pass_estimate_penalizes_missing_required():
    assert ats_pass_estimate(0.8, 0) == 0.8
    assert ats_pass_estimate(0.8, 1) == 0.64          # 0.8 * 0.8
    assert ats_pass_estimate(0.8, 2) < ats_pass_estimate(0.8, 1)
    assert ats_pass_estimate(0.0, 0) == 0.0
    assert 0.0 <= ats_pass_estimate(1.0, 5) <= 1.0


def test_ats_pass_label_buckets():
    assert "Strong" in ats_pass_label(0.9)
    assert "Moderate" in ats_pass_label(0.6)
    assert "Low" in ats_pass_label(0.4)
    assert "Very low" in ats_pass_label(0.1)


# ---- router / best-keeping -------------------------------------------------

def test_is_better_keeps_higher_target_then_overall():
    best = {"ats_match": 0.5, "overall": 0.5}
    assert _is_better({"ats_match": 0.6, "overall": 0.4}, best, "ats_match", 0.0)
    assert not _is_better({"ats_match": 0.5, "overall": 0.4}, best, "ats_match", 0.0)
    # tie on ats -> higher overall wins
    assert _is_better({"ats_match": 0.5, "overall": 0.7}, best, "ats_match", 0.0)
    assert _is_better({"ats_match": 0.1, "overall": 0.1}, None, "ats_match", 0.0)


def test_router_stops_on_cap_convergence_and_failure():
    assert improve_router({"resume_version": "rewritten", "rewrite_attempts": 1, "rewrite_improved": True}) == "retry"
    assert improve_router({"resume_version": "rewritten", "rewrite_attempts": 99, "rewrite_improved": True}) == "done"
    assert improve_router({"resume_version": "rewritten", "rewrite_attempts": 1, "rewrite_improved": False}) == "done"
    assert improve_router({"resume_version": "rewrite_failed_fallback", "rewrite_attempts": 0}) == "done"


# ---- full loop with a fake LLM --------------------------------------------

class _FakeResult:
    def __init__(self, text):
        self.ok, self.text, self.error, self.fallback = True, text, None, False


class _FakeClient:
    """Returns a fixed sequence of rewrite candidates with rising keyword coverage."""

    def __init__(self, texts):
        self._texts, self._i = list(texts), 0

    def chat(self, *args, **kwargs):
        text = self._texts[min(self._i, len(self._texts) - 1)]
        self._i += 1
        return _FakeResult(text)


def test_loop_keeps_best_candidate_and_terminates(monkeypatch):
    candidates = ["python sql", "python sql aws docker", "python sql aws docker"]
    fake = _FakeClient(candidates)
    monkeypatch.setattr("v_final.nodes.resume_rewrite.get_client", lambda: fake)

    init = {
        **_JD_STATE,
        "rewrite_required": True,
        "rewrite_strategy": "summary_and_skills",
        "resume_clean_text": "experienced python developer",   # baseline: only 'python' surfaced
        "resume_raw_text": "experienced python developer",
        "resume_skill_profile": _JD_STATE["jd_skill_profile"],  # resume HAS all the skills
        "missing_required_skills": ["sql", "aws", "docker", "kubernetes"],
    }
    final = build_rewrite_graph().invoke(init)

    assert final["rewrite_attempts"] == 3
    assert len(final["rewrite_candidates"]) == 3
    # best candidate (highest ATS) is kept as the optimized output
    assert final["optimized_resume_text"] == "python sql aws docker"

    cmp = final["rewrite_score_comparison"]["ats_match"]
    assert cmp["before"] == 0.2          # 1 of 5 surfaced
    assert cmp["after"] == 0.8           # 4 of 5 surfaced
    assert cmp["after"] > cmp["before"]

    # ATS pass chance is tracked too, and the verdict is exposed
    assert "ats_pass" in final["rewrite_score_comparison"]
    assert final["optimized_ats_pass_label"]


def test_offline_rewrite_does_not_loop():
    """With the LLM disabled (conftest), the rewrite falls back and the loop is a no-op."""
    init = {
        **_JD_STATE,
        "rewrite_required": True,
        "rewrite_strategy": "full_rewrite",
        "resume_clean_text": "python developer",
        "resume_raw_text": "python developer",
        "resume_skill_profile": _JD_STATE["jd_skill_profile"],
    }
    final = build_rewrite_graph().invoke(init)
    assert final["resume_version"] == "rewrite_failed_fallback"
    assert final.get("rewrite_attempts", 0) == 0
    assert "rewrite_score_comparison" not in final
