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
from v_final.llm import JDExtraction, get_client
from v_final.state.job_application_state import JobApplicationState

_SYSTEM = "Return JSON only. Follow the schema strictly. Canonicalize as instructed."

# Memoize JD understanding by content hash: parsing the same JD twice (e.g. the
# user tweaks the resume and re-analyzes) is the slowest repeated step, and it's
# pure -> cache it. Bounded so it can't grow without limit.
_JD_CACHE: dict[str, dict[str, Any]] = {}
_JD_CACHE_MAX = 64


def _build_jd_extract_prompt(jd_text: str) -> str:
    return f"""
You are an information extraction engine for job descriptions.

Return ONLY valid JSON (no markdown, no commentary).

SCHEMA (must follow exactly):
{{
  "jd_title": string,
  "domain": string,
  "seniority_level": "intern"|"junior"|"mid"|"senior"|"staff"|"principal"|"lead"|"manager"|"director"|"vp"|"unknown",
  "skills_required": [string],
  "skills_preferred": [string],
  "tools_process": [string],
  "responsibilities": [string],
  "keywords": [string]
}}

CANONICALIZATION RULES (IMPORTANT):
- All list items must be lowercase.
- Merge synonyms into canonical short forms where obvious:
  examples: "amazon web services"->"aws", "continuous integration"->"cicd",
            "machine learning" stays "machine learning", "ml"->"machine learning".
- Split compound items into atomic skills:
  example: "networking/communication protocols" -> ["networking protocols","communication protocols"]
- Keep "tools_process" limited to frameworks/tools/process terms:
  examples: agile, scrum, waterfall, sdlc, jira, itil, change management, stakeholder management.
- Put technical skills/tools (python, spark, tensorflow, kubernetes, rag, langgraph) into skills lists, NOT tools_process.
- skills_required: must-have/minimum qualifications (or clearly required).
- skills_preferred: preferred/nice-to-have/plus.
- responsibilities: 6-12 verb-led action statements.
- keywords: extra ATS terms not already included above.

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
        _SYSTEM, _build_jd_extract_prompt(jd_clean), JDExtraction, num_predict=900
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
