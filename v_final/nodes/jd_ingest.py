"""JD Ingest Node — LLM extraction of structured job-description understanding.

Uses the validated LLM client (:meth:`chat_structured`): the model returns JSON,
which is parsed into a :class:`JDExtraction` Pydantic object. If the server is
unavailable or the output is malformed, the node degrades to empty fields plus a
warning rather than crashing the graph.

Reads:  job_description_text
Writes: jd_clean_text, jd_title, jd_domain, jd_seniority_level,
        jd_skills_required, jd_skills_preferred, jd_tools_process,
        jd_responsibilities, jd_keywords, warnings
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from utils.text_normalization import _priority_dedupe, normalize_text
from v_final.config import settings
from v_final.llm import JDExtraction, get_client
from v_final.state.job_application_state import JobApplicationState

_SYSTEM = "Return JSON only. Follow the schema strictly. Canonicalize as instructed."

# Memoize JD understanding by content hash: parsing the same JD twice (e.g. the
# user tweaks the resume and re-analyzes) is the slowest repeated step, and it's
# pure -> cache it. Bounded so it can't grow without limit.
_JD_CACHE: dict[str, dict[str, Any]] = {}
_JD_CACHE_MAX = 64


def _build_jd_extract_prompt(jd_text: str) -> str:
    # Deliberately compact: only the fields scoring + rewrite need, so the model
    # generates a short JSON and the analyze stays responsive on CPU.
    return f"""
Extract job-description keywords. Return ONLY valid JSON (no prose):
{{
  "jd_title": string,
  "domain": string,
  "seniority_level": "intern"|"junior"|"mid"|"senior"|"staff"|"principal"|"lead"|"manager"|"director"|"vp"|"unknown",
  "skills_required": [string],   // must-have skills, tools, technologies
  "skills_preferred": [string],  // nice-to-have / "a plus"
  "keywords": [string]           // other important ATS terms
}}
Rules: lowercase list items; split compounds (e.g. "a/b testing" stays one term,
"networking/comm protocols" -> two); merge obvious synonyms ("amazon web services"->"aws",
"ml"->"machine learning"). Keep each list tight (no filler). No "responsibilities" field.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"
""".strip()


def _empty_jd_fields() -> dict[str, Any]:
    return {
        "jd_title": "",
        "jd_domain": "unknown",
        "jd_seniority_level": "unknown",
        "jd_skills_required": [],
        "jd_skills_preferred": [],
        "jd_tools_process": [],
        "jd_responsibilities": [],
        "jd_keywords": [],
    }


def jd_ingest_node(state: JobApplicationState) -> JobApplicationState:
    jd_clean = normalize_text(state.get("job_description_text", ""))
    state["jd_clean_text"] = jd_clean
    state.update(_empty_jd_fields())

    if not jd_clean.strip():
        state.setdefault("warnings", []).append("jd_ingest_skipped: empty job description")
        return state

    cache_key = hashlib.md5(jd_clean.encode("utf-8")).hexdigest()
    cached = _JD_CACHE.get(cache_key)
    if cached is not None:
        state.update(cached)
        return state

    extraction = get_client().chat_structured(
        _SYSTEM, _build_jd_extract_prompt(jd_clean), JDExtraction,
        num_predict=350, timeout=settings.llm.analyze_timeout,
    )
    if extraction is None:
        state.setdefault("warnings", []).append(
            "jd_ingest_llm_unavailable: JD parsed with empty structured fields"
        )
        return state

    # Priority-based dedupe so a term never appears in two buckets.
    req, pref, tools, kw = _priority_dedupe(
        extraction.skills_required,
        extraction.skills_preferred,
        extraction.tools_process,
        extraction.keywords,
    )
    responsibilities = [
        re.sub(r"\s+", " ", r.strip()) for r in extraction.responsibilities if r.strip()
    ]

    fields: dict[str, Any] = {
        "jd_title": extraction.jd_title,
        "jd_domain": extraction.domain or "unknown",
        "jd_seniority_level": extraction.seniority_level or "unknown",
        "jd_skills_required": req,
        "jd_skills_preferred": pref,
        "jd_tools_process": tools,
        "jd_responsibilities": responsibilities,
        "jd_keywords": kw,
    }
    state.update(fields)
    if len(_JD_CACHE) < _JD_CACHE_MAX:
        _JD_CACHE[cache_key] = fields
    return state
