"""Matching Node — composite, explainable fit scoring.

Produces three interpretable sub-scores and a weighted overall score:
- ``skill_match_score``    : coverage of JD skills by the resume
- ``semantic_similarity_score`` : embedding cosine of resume vs JD (lexical fallback)
- ``ats_match_score``      : coverage of required/keyword ATS terms

The scoring core is factored into :func:`score_resume_against_jd` so the same
metrics can be computed for the original resume AND each rewrite candidate (used
by the iterative rewrite loop) with *identical* text extraction — which is what
makes the before/after deltas trustworthy.

Reads:  resume_skill_profile, jd_skill_profile, *clean_text, jd_skills_*
Writes: skill_match_score, semantic_similarity_score, ats_match_score,
        overall_match_score, score_breakdown, matched_skills, missing_required_skills
"""

from __future__ import annotations

import math
import re
from collections import Counter

from v_final.config import settings
from v_final.llm import get_client
from v_final.nodes.skill_extraction import _build_profile, _canon, _flatten, _scan_vocab
from v_final.state.job_application_state import JobApplicationState

_TOKEN_RE = re.compile(r"[a-z0-9+#]+(?:[./-][a-z0-9+#]+)*")
_STOP = {
    "the", "and", "for", "with", "you", "are", "our", "that", "this", "will",
    "have", "has", "your", "from", "into", "their", "they", "but", "not", "all",
    "can", "who", "may", "out", "use", "via", "per", "etc", "a", "an", "of", "to",
    "in", "on", "as", "by", "or", "is", "be", "we", "it",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in _STOP]


def _lexical_cosine(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _vec_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _semantic_similarity(resume_text: str, jd_text: str) -> float:
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    client = get_client()
    er, ej = client.embed(resume_text), client.embed(jd_text)
    if er and ej:
        return max(0.0, _vec_cosine(er, ej))
    return _lexical_cosine(resume_text, jd_text)  # deterministic fallback


def _ats_score(state: JobApplicationState, resume_set: set[str]) -> float:
    terms: set[str] = set()
    for key in ("jd_skills_required", "jd_skills_preferred", "jd_keywords"):
        for t in state.get(key, []) or []:
            c = _canon(t)
            if c:
                terms.add(c)
    if not terms:
        return 0.0
    present = sum(1 for t in terms if t in resume_set)
    return present / len(terms)


def score_resume_against_jd(resume_text: str, state: JobApplicationState) -> dict:
    """Score an arbitrary resume text against the JD already parsed into ``state``.

    Uses pure text -> vocabulary extraction so any resume version (original or a
    rewrite candidate) is scored by the *same* method, making deltas meaningful.
    """
    resume_set = _flatten(_build_profile(_scan_vocab(resume_text)))
    jd_set = _flatten(state.get("jd_skill_profile") or {})
    jd_text = state.get("jd_clean_text") or state.get("job_description_text", "")

    skill_match = len(resume_set & jd_set) / len(jd_set) if jd_set else 0.0
    semantic = _semantic_similarity(resume_text, jd_text)
    ats = _ats_score(state, resume_set)
    w_skill, w_sem, w_ats = settings.scoring.normalized
    overall = w_skill * skill_match + w_sem * semantic + w_ats * ats

    jd_required = {_canon(s) for s in (state.get("jd_skills_required") or []) if _canon(s)}
    return {
        "skill_match": round(skill_match, 4),
        "semantic_similarity": round(semantic, 4),
        "ats_match": round(ats, 4),
        "overall": round(overall, 4),
        "matched_skills": sorted(resume_set & jd_set),
        "missing_required_skills": sorted(jd_required - resume_set),
    }


def matching_node(state: JobApplicationState) -> JobApplicationState:
    resume_set = _flatten(state.get("resume_skill_profile") or {})
    jd_set = _flatten(state.get("jd_skill_profile") or {})

    skill_match = len(resume_set & jd_set) / len(jd_set) if jd_set else 0.0
    resume_text = state.get("resume_clean_text") or state.get("resume_raw_text", "")
    jd_text = state.get("jd_clean_text") or state.get("job_description_text", "")
    semantic = _semantic_similarity(resume_text, jd_text)
    ats = _ats_score(state, resume_set)

    w_skill, w_sem, w_ats = settings.scoring.normalized
    overall = w_skill * skill_match + w_sem * semantic + w_ats * ats

    jd_required = {_canon(s) for s in (state.get("jd_skills_required") or []) if _canon(s)}

    state["skill_match_score"] = round(skill_match, 4)
    state["semantic_similarity_score"] = round(semantic, 4)
    state["ats_match_score"] = round(ats, 4)
    state["overall_match_score"] = round(overall, 4)
    state["score_breakdown"] = {
        "skill_match": round(skill_match, 4),
        "semantic_similarity": round(semantic, 4),
        "ats_match": round(ats, 4),
        "overall": round(overall, 4),
    }
    state["matched_skills"] = sorted(resume_set & jd_set)
    state["missing_required_skills"] = sorted(jd_required - resume_set)
    return state
