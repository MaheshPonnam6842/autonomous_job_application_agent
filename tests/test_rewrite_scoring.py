"""Tests for the iterative rewrite-and-rescore loop (all offline)."""

from v_final.graph.rewrite_graph import build_rewrite_graph
from v_final.llm import RewrittenExperience, StructuredRewrite
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
    base = {"best_scores": {"ats_match": 0.5, "overall": 0.5}}
    assert improve_router({**base, "resume_version": "rewritten", "rewrite_attempts": 1, "rewrite_improved": True}) == "retry"
    assert improve_router({**base, "resume_version": "rewritten", "rewrite_attempts": 99, "rewrite_improved": True}) == "done"
    assert improve_router({**base, "resume_version": "rewritten", "rewrite_attempts": 1, "rewrite_improved": False}) == "done"
    assert improve_router({"resume_version": "reformatted", "rewrite_attempts": 0}) == "done"


def test_router_early_exits_when_strong_enough():
    from v_final.config import settings
    thr = settings.rewrite.target_threshold
    strong = {"resume_version": "rewritten", "rewrite_attempts": 1, "rewrite_improved": True,
              "best_scores": {"ats_match": thr, "overall": 0.9}}
    assert improve_router(strong) == "done"           # good enough -> save an LLM call
    weak = {**strong, "best_scores": {"ats_match": thr - 0.3, "overall": 0.5}}
    assert improve_router(weak) == "retry"


def test_embedding_is_memoized(monkeypatch):
    from v_final.llm.client import OllamaClient

    calls = {"n": 0}

    class _FakeOllama:
        def embeddings(self, model, prompt, keep_alive=None):
            calls["n"] += 1
            return {"embedding": [0.1, 0.2, 0.3]}

    c = OllamaClient()
    monkeypatch.setattr(c, "available", lambda force=False: True)
    c._client = _FakeOllama()

    assert c.embed("same text") == [0.1, 0.2, 0.3]
    assert c.embed("same text") == [0.1, 0.2, 0.3]
    assert calls["n"] == 1                # second call served from cache
    c.embed("different text")
    assert calls["n"] == 2                # new content -> one more embed


# ---- full loop with a fake structured-rewrite LLM --------------------------

class _FakeStructuredClient:
    """Returns StructuredRewrite candidates whose bullets surface rising coverage."""

    def __init__(self, rewrites):
        self._r, self._i = list(rewrites), 0

    def chat_structured(self, system, user, schema, model=None):
        r = self._r[min(self._i, len(self._r) - 1)]
        self._i += 1
        return r


def _exp(bullet):
    return StructuredRewrite(experience=[RewrittenExperience(bullets=[bullet])])


_LOOP_INIT = {
    **_JD_STATE,
    "rewrite_required": True,
    "resume_clean_text": "jane doe experienced python developer",  # baseline: only python
    "resume_raw_text": "Jane Doe",
    "resume_sections": {"summary": "Python developer."},
    "resume_skills_structured": {"Skills": ["python"]},            # skills line carries only python
    "resume_skill_profile": _JD_STATE["jd_skill_profile"],         # resume HAS all (for feedback)
    "experience_entries": [{"company": "Acme", "role": "Engineer",
                            "start_date": "2021", "end_date": "Present",
                            "bullets": ["did python work"]}],
}


def test_loop_keeps_best_candidate_and_terminates(monkeypatch):
    rewrites = [
        _exp("Delivered python and sql solutions"),
        _exp("Delivered python sql aws docker solutions"),
        _exp("Delivered python sql aws docker solutions"),
    ]
    fake = _FakeStructuredClient(rewrites)   # one instance so the index persists across attempts
    monkeypatch.setattr("v_final.nodes.structured_rewrite.get_client", lambda: fake)

    final = build_rewrite_graph().invoke(dict(_LOOP_INIT))

    # default cap is 2 attempts; the best (2nd) candidate is reached and kept
    assert final["rewrite_attempts"] == 2
    assert len(final["rewrite_candidates"]) == 2
    # best candidate (more keywords) is kept and assembled into the output + struct
    assert "aws" in final["optimized_resume_text"] and "docker" in final["optimized_resume_text"]
    assert final["optimized_resume_struct"]["experience"]

    cmp = final["rewrite_score_comparison"]["ats_match"]
    assert cmp["before"] == 0.2          # 1 of 5 surfaced originally
    assert cmp["after"] == 0.8           # 4 of 5 surfaced after
    assert cmp["after"] > cmp["before"]
    assert "ats_pass" in final["rewrite_score_comparison"]
    assert final["optimized_ats_pass_label"]


def test_offline_reformats_without_looping():
    """LLM disabled: structured rewrite assembles an ATS-clean resume, no loop."""
    final = build_rewrite_graph().invoke(dict(_LOOP_INIT))
    assert final["resume_version"] == "reformatted"
    assert final.get("rewrite_attempts", 0) == 0
    assert "rewrite_score_comparison" not in final
    # still produces an assembled doc + struct for export
    assert "EXPERIENCE" in final["optimized_resume_text"]
    assert final["optimized_resume_struct"]["experience"]


# ---- ATS .docx export ------------------------------------------------------

def test_docx_export_builds_valid_file(tmp_path):
    from docx import Document

    from v_final.export import build_ats_docx
    struct = {
        "header": ["Jane Doe", "jane@example.com"],
        "summary": "Data scientist with ML experience.",
        "skills": ["python", "sql"],
        "experience": [{"company": "Acme", "role": "Data Scientist", "dates": "2021 - Present",
                        "bullets": [{"text": "Built ML pipelines.", "has_metric": False}]}],
        "education": ["M.S. Data Science  |  University (2020)"],
    }
    out = build_ats_docx(struct, tmp_path / "resume.docx")
    text = "\n".join(p.text for p in Document(out).paragraphs)
    assert "Jane Doe" in text
    assert "EXPERIENCE" in text and "Acme" in text and "Built ML pipelines." in text
